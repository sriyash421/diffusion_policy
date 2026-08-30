"""Render per-decision search videos for the PushT best-of-N eval.

One video frame per DECISION (every n_action_steps env steps), because a decision is
exactly one complete best-of-N search: the frame shows the state the search was run from,
every sampled candidate's executed action chunk overlaid on the scene, and both the ARGMAX
candidate (AMBER, the one executed) and the FINAL candidate (MAGENTA, slot n-1, the
last generated) highlighted. Sub-decision motion is not shown -- MultiStepWrapper only
surfaces the last n_obs_steps frames per step call, and the interesting object here is the
search, not the interpolation between waypoints.

Slot n-1 is only a meaningful "most-conditioned" draw for a k>1 search transformer. For the
UNet BC baseline (predict_action discards the context) and for ST k=1 (max_actions == 1, so
any n>1 runs a rolling window with an empty history) the n candidates are i.i.d. and slot
n-1 is merely "draw #n"; those frames say so.

BLIND DECISIONS ARE SKIPPED BY DEFAULT. The verifier value is a function of where the
T-block ends up, so when no candidate contacts the block every candidate returns the
identical value and the search carries zero information -- ~21% of control steps on a
measured reference run, and only ~17% of those are the initial approach; the rest are
mid-episode losses of contact. The ROLLOUT IS NOT FILTERED: the argmax still executes at
every step, so trajectories and success rates are untouched and only rendering is scoped.
The caption carries the true decision index and a running skip count, and the sidecar
records every skipped decision. --no-skip-blind renders them stamped instead.

Usage:
  python scripts/render_search_videos.py -c <ckpt> --label stk1 \
      --n 1 --n 2 --n 8 --n 16 --n 64 \
      --out-dir videos/aug19_bcvSTk1_actions_viz --n-seeds 10 --seed 42

Layout is `[ scene | zoom ]` side by side; --no-closeup drops the zoom and renders the scene
alone. The per-candidate value bar chart is OFF by default (--value-strip re-enables it): the
values are recorded exactly in the per-step JSON, and the bars were rescaled per frame so
they showed relative spread within one decision rather than an absolute scale.

Writes <out-dir>/seed<NN>_n<N>_step<STEP>_<label>_<succ|trunc>.mp4 plus a sidecar
episodes_<label>_step<STEP>_n<N>.json with per-seed length / success / reward / blind counts
and the layout switches the video was rendered under.
"""
if __name__ == "__main__":
    import sys, os, pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import json
import pathlib

import click
import cv2
import dill
import imageio
import numpy as np
import torch
import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from eval_search_pusht import (
    build_envs, get_test_states, load_policy, SUCCESS_REWARD, _episode_seed)


def expert_trajectories(cfg, episode_idxs):
    """Recorded demo agent path per episode, in raw 0-512 coords.

    Drawn as a static reference: once the policy diverges from the demo there is no
    ground-truth action for the state it is actually in, so a per-state expert arrow would
    be a claim about a different state. The whole path is well defined regardless.
    """
    rb = ReplayBuffer.copy_from_path(cfg.task.dataset.zarr_path, keys=['agent_pos'])
    ends = np.asarray(rb.episode_ends[:])
    starts = np.concatenate([[0], ends[:-1]])
    ap = np.asarray(rb['agent_pos'])
    return [ap[starts[i]:ends[i]] for i in episode_idxs]

def resolved_verifier_value(policy, cfg):
    """Which VALUE_FNS key this rollout actually scores with.

    Recorded in every artifact because the scoring rule changed on 2026-08-19 (`t_goal` ->
    `armT`, adding an arm-to-T approach term) and the two are NOT comparable: `t_goal` is
    flat across candidates until the arm touches the block, `armT` is not. A checkpoint
    saved before the cutover carries no `verifier_value` key, so it resolves to the
    pre-cutover default and is scored on exactly what it was trained on -- but an artifact
    that does not SAY so cannot be told apart from one produced under armT.
    """
    from diffusion_policy.env.pusht.pusht_verifier import DEFAULT_VALUE_FN, VALUE_FNS
    mode = None
    try:
        mode = cfg.policy.get('verifier_value', None)
    except Exception:
        pass
    mode = str(mode or DEFAULT_VALUE_FN)
    assert mode in VALUE_FNS, f'unknown verifier_value {mode!r}'
    return mode


WINDOW = 512          # PushT action / agent_pos coordinate space
SCENE = 384           # rendered scene panel, px
PAD = 10
HEADER = 34


def _scene(img96, size=SCENE):
    """96x96 obs -> upscaled BGR scene panel."""
    img = cv2.resize(np.asarray(img96), (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _pt(xy, size=SCENE):
    return (int(round(xy[0] / WINDOW * size)), int(round(xy[1] / WINDOW * size)))


def _fan(canvas, chunks, to_px, highlights, subsample=None,
         w_loser=1, w_hi=2, dot=4, alpha=0.6):
    """Draw the candidate fan onto `canvas` using the coordinate map `to_px`.

    `highlights` is an ordered list of (index, bgr, ring) drawn LAST-ON-TOP, so the caller
    controls which of several marked candidates ends up visible where they overlap. Losers
    go down first and translucent, so no highlight is ever buried under them. Colours are
    BGR. Candidate order is generation order, which is also context order.

    `subsample` optionally restricts which LOSERS get a polyline (see draw_candidates). It
    never restricts the endpoint scatter or the highlights: the scatter is the honest
    picture of the candidate distribution and must show all n, while a highlight that
    vanished because it fell outside a subsample would be a lie about the search.
    """
    hi_idx = {h[0] for h in highlights}
    drawn = range(len(chunks)) if subsample is None else subsample

    overlay = canvas.copy()
    for c in drawn:
        if c in hi_idx:
            continue
        pts = np.array([to_px(p) for p in chunks[c]], dtype=np.int32)
        cv2.polylines(overlay, [pts], False, (150, 150, 150), w_loser, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    # Endpoint scatter for ALL n candidates, always. At n=64 the polylines are only
    # illustration -- this is what actually carries the spread.
    for c in range(len(chunks)):
        if c in hi_idx:
            continue
        p = to_px(chunks[c][-1])
        cv2.circle(canvas, p, max(1, dot // 2), (110, 110, 110), -1, cv2.LINE_AA)

    for idx, colour, ring in highlights:
        pts = np.array([to_px(p) for p in chunks[idx]], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, (0, 0, 0), w_hi + 2, cv2.LINE_AA)   # halo
        cv2.polylines(canvas, [pts], False, colour, w_hi, cv2.LINE_AA)
        # direction: eight waypoints of one chunk read as a smudge without it
        if len(pts) >= 2 and not np.array_equal(pts[-1], pts[-2]):
            cv2.arrowedLine(canvas, tuple(pts[-2]), tuple(pts[-1]), colour,
                            w_hi, cv2.LINE_AA, tipLength=0.5)
        cv2.circle(canvas, tuple(pts[0]), dot, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pts[-1]), dot, colour, -1, cv2.LINE_AA)
        if ring:   # the EXECUTED candidate, a distinct property from "is the argmax"
            cv2.circle(canvas, tuple(pts[-1]), dot + 2, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# OpenCV works in BGR and frames are converted to RGB at write time, so the DISPLAYED
# colour is the reverse of the constant: (60,200,255) shows as RGB(255,200,60), which is
# AMBER -- not the "cyan" the original captions claimed for this exact value. Named for what
# a viewer actually sees, because the on-frame legend is the only thing telling them which
# path is which. Amber rather than true cyan on purpose: PushT's scene is grey/blue/red/green
# and cyan competes with the blue block.
COL_ARGMAX = (60, 200, 255)     # displays AMBER   -- argmax, the executed candidate
COL_FINAL = (255, 64, 255)      # displays MAGENTA -- slot n-1, the last generated
COL_BOTH = (200, 130, 255)      # displays PINK    -- the two coincide

# Palette for --highlight-slots, assigned in the order the slots are requested. Chosen to
# stay clear of everything already on the frame: AMBER/MAGENTA/PINK above, the demo path
# (120,255,120 = pale green), the white start dot and the white executed ring, and PushT's
# own grey/blue/red scene. Named for the DISPLAYED RGB, i.e. the reverse of the constant,
# following the convention established above.
COL_SLOTS = [
    ((0, 255, 255), 'YELLOW'),
    ((255, 200, 0), 'AZURE'),
    ((0, 128, 255), 'ORANGE'),
    ((255, 0, 128), 'VIOLET'),
]


def highlight_spec(best, final, execute='argmax', extra_slots=(), n_actions=None):
    """(highlights, coincide, legend) for one decision.

    When the argmax IS the last-generated candidate the two markers would be drawn on top
    of each other and one would silently win. Collapse them to a single blended path
    instead and let the caller say so in the caption -- dropping one marker would misreport
    which candidates the search actually distinguished.

    ``extra_slots`` adds fixed slot indices (``--highlight-slots``) on top of those two.
    THE SAME OVERDRAW RULE APPLIES, and it is the reason this returns a legend rather than
    just paths: at n=16 slot 15 IS `final`, and slot k is `best` whenever the argmax lands
    there, so a naive extra path would be drawn over an existing one and the viewer would
    see a single marker while the caption claimed two. A coinciding slot is therefore NOT
    given its own path -- it is folded into the existing entry and the legend says which
    slot that entry also is.

    Slots outside ``range(n_actions)`` are dropped rather than indexed: at n=8 a request for
    slot 15 is a mistake, and wrapping it to slot 7 would silently relabel a different
    candidate. The caller warns once.

    ``legend`` is a list of (name, colour_word) for the caption, so the on-frame key always
    describes exactly the paths that were drawn.
    """
    executed = best if execute == 'argmax' else final
    coincide = best == final
    if coincide:
        base = [(best, COL_BOTH, True)]
        legend = [(f'argmax=final(#{best})', 'PINK', 0)]
    else:
        base = [(final, COL_FINAL, executed == final),
                (best, COL_ARGMAX, executed == best)]
        # SAME ORDER AS `base`. These two lists are indexed by the same key below, so a
        # legend ordered independently of the paths mislabels which colour is which the
        # moment an extra slot folds into one of them.
        legend = [('final(slot n-1)', 'MAGENTA', 1), ('argmax', 'AMBER', 0)]

    if not extra_slots:
        return base, coincide, legend

    # index -> position in `base`, so a coinciding slot annotates the existing entry
    at = {h[0]: i for i, h in enumerate(base)}
    for j, k in enumerate(extra_slots):
        if n_actions is not None and not (0 <= k < n_actions):
            continue                      # caller warned; never wrap into a wrong candidate
        if k in at:
            name, word, order = legend[at[k]]
            legend[at[k]] = (f'{name}=slot{k}', word, order)
            continue
        colour, word = COL_SLOTS[j % len(COL_SLOTS)]
        at[k] = len(base)
        base.append((k, colour, executed == k))
        legend.append((f'slot{k}', word, 2 + j))
    return base, coincide, legend


ZOOM = SCENE    # zoom panel, px. Sized to the scene: it is a sibling panel, not a
                # picture-in-picture overlay, so there is no reason to shrink it.


def draw_expert(panel, traj):
    """Draw the recorded demo's agent path as a static reference.

    The per-state expert ACTION is undefined once the policy diverges from the demo, so
    what is drawn is the whole recorded trajectory: a fixed reference path for the episode
    rather than a claim about the state the policy is currently in.
    """
    if traj is None or len(traj) < 2:
        return panel
    pts = np.array([_pt(p) for p in traj], dtype=np.int32)
    cv2.polylines(panel, [pts], False, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.polylines(panel, [pts], False, (120, 255, 120), 1, cv2.LINE_AA)
    return panel


def _fan_subsample(n, max_fan):
    """Which losers get a polyline in the scene panel.

    POSITIONAL, never by score: picking the top-k or a random-by-value subset would make
    the drawn fan a biased sample of exactly the spread these videos exist to show.
    """
    if max_fan is None or n <= max_fan:
        return None
    return set(np.linspace(0, n - 1, max_fan).round().astype(int).tolist())


def draw_candidates(panel, chunks, highlights, max_fan=24, zoom_px=ZOOM, zoom_q=0.995,
                    closeup=True):
    """Scene with the candidate fan, plus a zoomed inset of that fan.

    Returns ``(scene, inset)``. The inset is a SEPARATE panel that compose() lays out beside
    the scene -- it used to be pasted into the scene's bottom-left corner, where it covered a
    quarter of the very view it was meant to explain (routinely the T block itself). Nothing
    is occluded now: both are full size and side by side.

    An executed chunk spans only ~8 small waypoints, so at scene scale the whole search
    collapses into a smudge beside the agent. The inset crops the fan's bounding box out of
    the scene and blows it up, which is the only way the spread between candidates -- the
    thing these videos exist to show -- is actually legible. The inset draws ALL n
    candidates; only the scene panel subsamples.

    The crop box uses per-axis QUANTILES rather than min/max: one stray candidate would
    otherwise inflate the box until the inset zoomed less at n=64 than at n=8, i.e. the
    wider the search the less you could see of it.
    """
    n = len(chunks)
    sub = _fan_subsample(n, max_fan)
    alpha = float(np.clip(0.45 * 24.0 / max(len(sub) if sub else n, 1), 0.12, 0.60))
    inset = None
    if not closeup:
        # scene only. The fan is a few px across at this scale, so this is for judging the
        # trajectory in context, not the candidate spread -- use the zoom for the latter.
        _fan(panel, chunks, _pt, highlights, subsample=sub, alpha=alpha)
        return panel, None

    # Crop the CLEAN scene: _fan below draws onto `panel`, so cropping panel afterwards
    # double-draws the fan into its own zoom (every line rendered twice, at two scales).
    clean = panel.copy()
    _fan(panel, chunks, _pt, highlights, subsample=sub, alpha=alpha)

    all_pts = chunks.reshape(-1, 2)
    lo = np.quantile(all_pts, 1.0 - zoom_q, axis=0)
    hi = np.quantile(all_pts, zoom_q, axis=0)
    n_clipped = int(np.sum(np.any((all_pts < lo) | (all_pts > hi), axis=1)))
    ctr, half = (lo + hi) / 2.0, max((hi - lo).max() * 0.75, 24.0)
    x0, y0, x1, y1 = ctr[0] - half, ctr[1] - half, ctr[0] + half, ctr[1] + half
    src = panel.shape[0]
    cx0, cy0 = max(0, int(x0 / WINDOW * src)), max(0, int(y0 / WINDOW * src))
    cx1, cy1 = min(src, int(x1 / WINDOW * src)), min(src, int(y1 / WINDOW * src))
    if cx1 - cx0 >= 4 and cy1 - cy0 >= 4:
        crop = cv2.resize(clean[cy0:cy1, cx0:cx1], (zoom_px, zoom_px),
                          interpolation=cv2.INTER_NEAREST)
        sx, sy = zoom_px / max(1, (cx1 - cx0)), zoom_px / max(1, (cy1 - cy0))

        def to_zoom(p):
            # clamp rather than drop: a candidate outside the quantile box still exists,
            # and n_clipped in the caption says how many are pinned to the border
            return (int(np.clip(round((p[0] / WINDOW * src - cx0) * sx), 0, zoom_px - 1)),
                    int(np.clip(round((p[1] / WINDOW * src - cy0) * sy), 0, zoom_px - 1)))

        _fan(crop, chunks, to_zoom, highlights, subsample=None,
             w_loser=1, w_hi=2, dot=3, alpha=alpha)
        cv2.rectangle(crop, (0, 0), (zoom_px - 1, zoom_px - 1), (200, 200, 200), 1)
        tag = f'zoom {2*half:.0f}px' + (f'   {n_clipped} clipped' if n_clipped else '')
        cv2.putText(crop, tag, (4, zoom_px - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (200, 200, 200), 1, cv2.LINE_AA)
        inset = crop
    return panel, inset


LABEL_H = 13     # per-thumbnail caption strip
STRIP_PAD = 14   # breathing room beneath the whole subgoal block


def value_strip(values, best, final, width=SCENE, height=92):
    """Per-candidate verifier value as a bar chart, in generation order.

    Replaces the subgoal thumbnail strip as the default. An 8-step chunk barely moves the
    scene, so those thumbnails were near-identical by construction (~2/255 mean pairwise
    difference on the overfit checkpoints) while costing an extra render per candidate --
    64 of them per control step at n=64. The value is the quantity that actually separates
    the candidates, it is the quantity argmax ranks on, and it is the same number the
    per-step JSON records, so the videos and the JSON tell one story.

    Y axis is DISTANCE in px (-value), so shorter is better and the argmax bar is the
    shortest one. Bars are drawn from the top down for that reason.
    """
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    strip = np.full((height + LABEL_H, width, 3), 24, dtype=np.uint8)
    dist = -v
    lo, hi = float(dist.min()), float(dist.max())
    span = max(hi - lo, 1e-6)
    bw = max(1, width // max(n, 1))
    plot_h = height - 16
    for i in range(n):
        x0 = i * bw
        frac = (dist[i] - lo) / span                  # 0 = best, 1 = worst
        h = int(round(4 + frac * (plot_h - 4)))
        col = ((COL_BOTH if best == final else COL_ARGMAX) if i == best
               else COL_FINAL if i == final else (105, 105, 105))
        cv2.rectangle(strip, (x0 + 1, 12), (x0 + max(1, bw - 1), 12 + h), col, -1)
    # mean line, so "is the executed one actually better than typical" is readable
    ym = 12 + int(round(4 + ((dist.mean() - lo) / span) * (plot_h - 4)))
    for x in range(0, width, 6):
        cv2.line(strip, (x, ym), (min(x + 3, width - 1), ym), (160, 160, 160), 1)
    cv2.putText(strip, f'dist px  best {lo:.1f}   worst {hi:.1f}   mean {dist.mean():.1f}',
                (4, height + LABEL_H - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                (190, 190, 190), 1, cv2.LINE_AA)
    return strip


def subgoal_strip(images, best, values=None, width=SCENE, rows=None):
    """Tile per-candidate reached observations, generation order, chosen one outlined.

    Kept behind --subgoals. Each thumbnail is the observation that candidate i's executed
    chunk actually REACHES in the verifier sim -- so it is what the NEXT candidate gets to
    condition on.
    """
    n = len(images)
    cols = 16 if n > 16 else 8
    rows = rows or int(np.ceil(n / cols))
    cell = width // cols
    row_h = cell + LABEL_H
    strip = np.full((rows * row_h + STRIP_PAD, width, 3), 24, dtype=np.uint8)
    lo = float(np.min(values)) if values is not None else 0.0
    hi = float(np.max(values)) if values is not None else 1.0
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        y0, x0 = r * row_h, c * cell
        thumb = cv2.resize(np.asarray(im), (cell, cell), interpolation=cv2.INTER_AREA)
        thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
        if i == best:
            cv2.rectangle(thumb, (0, 0), (cell - 1, cell - 1), COL_ARGMAX, 2)
        strip[y0:y0 + cell, x0:x0 + cell] = thumb
        if values is not None:
            val = float(values[i])
            t = 0.0 if hi <= lo else (val - lo) / (hi - lo)
            col = COL_ARGMAX if i == best else (int(90 + 90 * t),) * 3
            cv2.putText(strip, f'{val:.0f}', (x0 + 2, y0 + cell + LABEL_H - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, col, 1, cv2.LINE_AA)
    return strip


CAP_SCALE = 0.36     # nominal caption font scale
CAP_MIN_SCALE = 0.27  # shrink to here before truncating


def _fit_text(text, width, scale=CAP_SCALE, min_scale=CAP_MIN_SCALE):
    """(text, scale) that fits `width`. Shrink first, truncate only as a last resort.

    Caption content changes with n, with the arm and with the blind-skip count, so a fixed
    font size silently ran off the right edge -- 'grn=dem' and a truncated score range were
    being rendered into every frame with nothing to signal it. Measuring is the only way a
    caption cannot lie by omission.
    """
    avail = width - 12
    scale = float(scale)
    while scale > min_scale:
        if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] <= avail:
            return text, scale
        scale -= 0.01
    while len(text) > 4 and \
            cv2.getTextSize(text + '..', cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > avail:
        text = text[:-1]
    return text + '..', scale


GUTTER = 8      # px between the scene and the zoom panel


def compose(panel, inset, strip, lines):
    """header / [scene | zoom] / strip, with auto-fitted caption lines.

    The zoom sits BESIDE the scene rather than on top of it. Overlaying it into the scene's
    bottom-left corner hid a quarter of the scene -- frequently the T block and the goal
    outline, i.e. the context needed to interpret the very fan the zoom was magnifying.
    Both panels are full size here and neither occludes anything.
    """
    if inset is not None:
        h = max(panel.shape[0], inset.shape[0])
        row = np.full((h, panel.shape[1] + GUTTER + inset.shape[1], 3), 24, dtype=np.uint8)
        row[:panel.shape[0], :panel.shape[1]] = panel
        x = panel.shape[1] + GUTTER
        row[:inset.shape[0], x:x + inset.shape[1]] = inset
        panel = row
    w = panel.shape[1]
    head = HEADER + 14 * (len(lines) - 1)
    parts = [np.full((head, w, 3), 24, dtype=np.uint8), panel]
    if strip is not None:
        # the strip spans the full width, so widen it if it was built at scene width
        if strip.shape[1] != w:
            pad = np.full((strip.shape[0], w - strip.shape[1], 3), 24, dtype=np.uint8)
            strip = np.hstack([strip, pad]) if strip.shape[1] < w else strip[:, :w]
        parts += [np.full((PAD, w, 3), 24, dtype=np.uint8), strip]
    frame = np.vstack(parts)
    for i, t in enumerate(lines):
        txt, sc = _fit_text(t, w)
        cv2.putText(frame, txt, (8, 18 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, sc,
                    (235, 235, 235), 1, cv2.LINE_AA)
    # libx264 + yuv420p chroma-subsamples 2x2 and REFUSES an odd width or height: ffmpeg
    # dies with "Error while opening encoder ... maybe incorrect parameters" and the writer
    # surfaces it only as a broken pipe several frames later. The frame height here is
    # HEADER + 14*(len(lines)-1) + SCENE + PAD + strip, so it flips parity with the caption
    # line count and the strip choice -- i.e. an extra caption line is enough to break
    # encoding. Pad to even rather than constraining the layout.
    h, w = frame.shape[:2]
    if h % 2 or w % 2:
        frame = np.pad(frame, ((0, h % 2), (0, w % 2), (0, 0)),
                       mode='constant', constant_values=24)
    return frame


def render_one(policy, cfg, env, states, episode_idxs, experts, n_actions, out, label,
               step, seed, device, max_steps, fps, hold, subgoals, max_fan, zoom_px,
               zoom_q, skip_blind, blind_eps, execute, verifier_value='t_goal',
               value_strip_on=False, closeup=True, hl_slots=()):
    """Roll all episodes once at one search width and write one video per episode."""
    To, Ta = policy.n_obs_steps, policy.n_action_steps
    sl = slice(To - 1, To - 1 + Ta)
    B = len(states)

    # At n=1 there is one candidate, so the spread is identically zero and EVERY decision
    # would look blind -- skipping would emit empty videos. Blindness is a statement about
    # candidates disagreeing, which needs at least two of them.
    skipping = skip_blind and n_actions >= 2

    def make_init_fn(state):
        state = np.asarray(state, dtype=np.float64)
        def _fn(e):
            e.unwrapped.reset_to_state = state
        return _fn

    env.call_each('run_dill_function',
                  args_list=[(dill.dumps(make_init_fn(s)),) for s in states])
    obs = env.reset()
    policy.reset()
    # Per-EPISODE noise streams keyed on the episode's position in the test split -- the
    # same key _eval_split_at_n uses -- so these are eval's episodes 0..B-1 on eval's seeds,
    # and the two arms are paired decision-for-decision rather than merely both random.
    # NOTE this pairs on (episode, draw index), NOT on the noise realization: the UNet and
    # the transformer denoise different-shaped tensors, so the Gaussians themselves differ.
    seeder = getattr(policy, 'set_sample_seeds', None)
    if seeder is not None:
        seeder([_episode_seed(seed, n_actions, i) for i in range(B)])

    # Temp names must be unique per (label, step, n): several invocations share one output
    # folder and may be in flight at once, so a bare `.tmp_seed0.mp4` lets one job rename
    # the other's file out from under it.
    stem = f'.tmp_{label}_{step}_n{n_actions}_seed'
    writers = [imageio.get_writer(str(out / f'{stem}{i}.mp4'), fps=fps, codec='libx264',
                                  quality=8, macro_block_size=1) for i in range(B)]
    frames_written = np.zeros(B, dtype=int)
    blind_count = np.zeros(B, dtype=int)
    blind_decisions = [[] for _ in range(B)]
    ep_done = np.zeros(B, dtype=bool)
    decision = 0
    done = False
    pbar = tqdm.tqdm(total=max_steps // Ta + 1, desc=f'{label} n={n_actions} step{step}')
    while not done:
        obs_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to(device=device))
        with torch.no_grad():
            res = policy.predict_n_actions(
                obs_dict, verifier=policy.verifier, n_actions=n_actions,
                return_scores=True, return_subgoals=subgoals)
        actions, _, scores = res[0], res[1], res[2]
        sub = res[3] if subgoals else None
        best = scores.argmax(dim=1)

        scenes = env.call('render', 'rgb_array')
        chunks = actions[:, :, sl].detach().cpu().numpy()          # (B, n, Ta, 2)
        sc = scores.detach().cpu().numpy()
        bi = best.detach().cpu().numpy()
        final_i = n_actions - 1
        sub_img = None
        if sub is not None and sub.get('image', None) is not None:
            sub_img = (sub['image'].clamp(0, 1) * 255).byte().permute(0, 1, 3, 4, 2) \
                      .cpu().numpy()                                # (B, n, H, W, 3)

        for i in range(B):
            # A finished episode is still stepped to keep the batch square; its frames are
            # states the policy was never really in.
            if ep_done[i]:
                continue
            spread = float(sc[i].max() - sc[i].min())
            blind = n_actions >= 2 and spread <= blind_eps
            if blind:
                blind_count[i] += 1
                blind_decisions[i].append(decision)
                if skipping:
                    continue

            panel = draw_expert(_scene(scenes[i]), experts[i])
            hl, coincide, legend = highlight_spec(
                int(bi[i]), final_i, execute, extra_slots=hl_slots,
                n_actions=n_actions)
            panel, inset = draw_candidates(panel, chunks[i], hl, max_fan=max_fan,
                                           zoom_px=zoom_px, zoom_q=zoom_q,
                                           closeup=closeup)
            # any strip spans the whole composed width, scene + gutter + zoom
            strip_w = panel.shape[1] + (GUTTER + inset.shape[1] if inset is not None else 0)
            if sub_img is not None:
                strip = subgoal_strip(sub_img[i], int(bi[i]), values=sc[i], width=strip_w)
            elif value_strip_on:
                strip = value_strip(sc[i], int(bi[i]), final_i, width=strip_w)
            else:
                # OFF by default. The per-candidate values are recorded exactly in the
                # per-step JSON (see scripts/dump_candidate_scores.py --per-step-json), and
                # the bar heights here were rescaled per frame, so they showed relative
                # spread within one decision and were not comparable between frames. The
                # caption still carries the executed value and the spread as numbers.
                strip = None

            lo, hi = float(sc[i].min()), float(sc[i].max())
            shown = min(n_actions, max_fan) if max_fan else n_actions
            fan_note = f'  fan {shown}/{n_actions}' if shown < n_actions else ''
            ma = getattr(policy, 'max_actions', 1)
            iid = ma <= 1 or ma >= (1 << 20)
            cap = [
                f'ep{episode_idxs[i]}  dec {decision}'
                + (f' (+{blind_count[i]} blind)' if blind_count[i] else '')
                + f'  n={n_actions}  step{step}  {label}',
                f'executes {execute.upper()}'
                + (' = eval rule' if execute == 'argmax' else ' NOT the eval rule')
                + f'  |  verifier {verifier_value}',
                (f'chosen #{int(bi[i])} == final' if coincide
                 else f'chosen #{int(bi[i])}  final #{final_i}')
                + f'  val {sc[i][int(bi[i])]:.1f}  spread {hi - lo:.1f}'
                + ('  BLIND: no candidate moves the T' if blind else ''),
                # DEFAULT PATH IS THE ORIGINAL STRING, VERBATIM. --highlight-slots
                # defaults to empty and must leave frames byte-identical, and `legend` is
                # ordered to match `base` ([final, best]) for the fold-in indexing -- so
                # rendering it directly would silently swap the caption's word order on
                # every existing render. Only the extra-slot path uses the computed legend,
                # sorted back into display order (argmax, final, then extras as requested).
                ('grn demo  AMBER argmax  MAGENTA final(slot n-1)' if not hl_slots else
                 'grn demo  ' + '  '.join(
                     f'{w} {nm}' for nm, w, _ in sorted(legend, key=lambda e: e[2])))
                + fan_note,
            ]
            if hl_slots:
                # the numbers behind the extra markers, so a viewer can read "does it
                # improve with context" off the caption and not only off the bar strip
                cap.append('  '.join(
                    f'#{k}={sc[i][k]:.1f}' for k in hl_slots if 0 <= k < n_actions))
            if iid and n_actions > 1:
                cap.append('slot n-1 = i.i.d. draw, no context')
            frame = compose(panel, inset, strip, cap)
            for _ in range(hold):
                writers[i].append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames_written[i] += 1

        exec_idx = best if execute == 'argmax' else torch.full_like(best, final_i)
        act = actions[torch.arange(B, device=actions.device), exec_idx][:, sl]
        obs, reward, step_done, info = env.step(act.detach().cpu().numpy())
        ep_done |= np.asarray(step_done, dtype=bool)
        done = np.all(step_done)
        decision += 1
        pbar.update(1)
    pbar.close()

    rewards = env.call('get_attr', 'reward')
    # An episode whose every decision was blind would otherwise produce a zero-frame mp4:
    # the writer errors and the job looks crashed. Emit one stamped frame instead.
    for i in range(B):
        if frames_written[i] == 0:
            blank = np.full((SCENE, SCENE, 3), 24, dtype=np.uint8)
            msg = compose(blank, None, None, [
                f'ep{episode_idxs[i]}  n={n_actions}  step{step}  {label}',
                f'ALL {decision} DECISIONS BLIND -- verifier never discriminated'])
            for _ in range(max(hold, 4)):
                writers[i].append_data(cv2.cvtColor(msg, cv2.COLOR_BGR2RGB))
    for w in writers:
        w.close()

    records = []
    for i in range(B):
        r = np.asarray(rewards[i])
        succ = bool(r.max() >= SUCCESS_REWARD)
        length = int(len(r))
        tag = 'succ' if succ else 'trunc'
        name = f'seed{i:02d}_n{n_actions}_step{step}_{label}_{tag}.mp4'
        (out / f'{stem}{i}.mp4').rename(out / name)
        n_dec = int(np.ceil(length / Ta))
        records.append({'seed': i, 'episode_idx': int(episode_idxs[i]), 'n': n_actions,
                        'step': step, 'label': label, 'success': succ,
                        'env_steps': length, 'decisions': n_dec,
                        'n_decisions': n_dec, 'n_blind': int(blind_count[i]),
                        'blind_decisions': [int(x) for x in blind_decisions[i]],
                        'blind_rate': float(blind_count[i] / max(n_dec, 1)),
                        'frames': int(frames_written[i]), 'skip_blind': bool(skipping),
                        'execute': execute, 'seed_base': int(seed),
                        'verifier_value': verifier_value,
                        'closeup': bool(closeup), 'value_strip': bool(value_strip_on),
                        'max_reward': float(r.max()), 'video': name})
        print(f'  {name}  len={length} steps  frames={frames_written[i]} '
              f'blind={blind_count[i]}/{n_dec}  max_reward={r.max():.3f}')

    # One sidecar per (label, step, n): concurrent invocations into the same folder would
    # otherwise read-modify-write a shared file and drop each other's records, and the
    # label is needed because both arms render into one directory.
    idx = out / f'episodes_{label}_step{step}_n{n_actions}.json'
    idx.write_text(json.dumps(records, indent=2))
    return records


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('--label', required=True, help='policy name, used in filenames')
@click.option('--n', 'n_list', multiple=True, type=int, default=(16,),
              help='candidates per decision; repeatable, swept in one process')
@click.option('--out-dir', required=True)
@click.option('--n-seeds', default=10)
@click.option('-d', '--device', default='cuda:0')
@click.option('--max-steps', default=300)
@click.option('--fps', default=4)
@click.option('--hold', default=4, help='video frames per decision')
@click.option('--seed', default=None, type=int,
              help="base sampling seed; defaults to the checkpoint's training seed")
@click.option('--subgoals/--no-subgoals', default=False,
              help='render the per-candidate reached-observation strip instead of the '
                   'value bars (costs one extra sim render per candidate)')
@click.option('--max-fan', default=24,
              help='cap on loser polylines in the scene panel; 0 disables the cap')
@click.option('--closeup/--no-closeup', default=True,
              help='draw the zoomed candidate panel beside the scene. --no-closeup gives '
                   'the scene alone, for judging the trajectory in context (the fan is only '
                   'a few px across at that scale).')
@click.option('--value-strip/--no-value-strip', default=False,
              help='bar chart of the n candidate values under the frame. OFF by default: '
                   'the values are recorded exactly in the per-step JSON, and the bars were '
                   'rescaled per frame so they were not comparable between frames.')
@click.option('--zoom', 'zoom_px', default=ZOOM, help='zoom panel size, px')
@click.option('--zoom-quantile', 'zoom_q', default=0.995,
              help='per-axis quantile for the inset crop box. 1.0 == plain min/max, which '
                   'lets one stray candidate inflate the box (measured: the n=64 inset '
                   'zoomed LESS than n=8). Below ~0.99 it clips real waypoints -- at 0.98, '
                   '~8%% of them -- so the inset stops showing the true spread. The frame '
                   'prints how many were clipped.')
@click.option('--skip-blind/--no-skip-blind', default=True,
              help='omit decisions where every candidate scores identically (the verifier '
                   'cannot discriminate). The ROLLOUT is unaffected either way.')
@click.option('--blind-eps', default=1e-9, help='spread at or below which a decision is blind')
@click.option('--highlight-slots', default='',
              help='comma-separated candidate slots to draw in addition to argmax and '
                   'final, e.g. "0,7,15". Slot 15 IS final at n=16, and any slot can be '
                   'the argmax, so a coinciding slot is folded into that marker and named '
                   'in the legend rather than overdrawn. Out-of-range slots are dropped '
                   'with a warning. Default empty == the previous two-highlight frames, '
                   'byte-identical.')
@click.option('--execute', type=click.Choice(['argmax', 'final']), default='argmax',
              help="which candidate drives the rollout; 'argmax' is the eval protocol")
@click.option('--verifier-value', default=None,
              help='override the scoring rule, so an arm trained before the 2026-08-19 '
                   'cutover can be rendered under the same value as the arms it is being '
                   'compared with (the UNet BC run carries no verifier_tag and otherwise '
                   'resolves to t_goal). CHANGES THE ROLLOUT: the executed candidate is '
                   "the argmax under this value. Default: the checkpoint's own.")
def main(checkpoint, label, n_list, out_dir, n_seeds, device, max_steps, fps, hold, seed,
         subgoals, max_fan, zoom_px, zoom_q, skip_blind, blind_eps, execute,
         closeup, value_strip, highlight_slots, verifier_value):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    step = int(''.join(ch for ch in pathlib.Path(checkpoint).stem if ch.isdigit()) or 0)
    max_fan = None if not max_fan else int(max_fan)
    # Parsed once here rather than per frame. Duplicates are dropped but ORDER is kept:
    # the palette is assigned in request order, so `--highlight-slots 0,7,15` gives slot 0
    # the first colour in every video and the legend stays stable across arms.
    hl_slots = []
    for tok in (highlight_slots or '').split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            k = int(tok)
        except ValueError:
            raise SystemExit(f'--highlight-slots: {tok!r} is not an integer')
        if k not in hl_slots:
            hl_slots.append(k)

    policy, cfg = load_policy(checkpoint, device)
    # SCORING RULE OVERRIDE. Same mechanism as dump_candidate_scores.collect() and
    # eval_search_pusht --verifier-value, deliberately: one implementation of a rule that
    # decides which candidate is executed, not three that drift. The transformer arms build
    # their verifier in __init__ so theirs is swapped in place; the UNet arm builds lazily
    # (to keep training from forking a 32-process sim pool) and will read the kwarg when it
    # does, so its `verifier` property is NOT touched here -- doing so would force the fork.
    if verifier_value is not None:
        policy.search_kwargs['verifier_value'] = verifier_value
        # Read the ALREADY-BUILT verifier out of __dict__ rather than through the
        # attribute: the UNet arm's `verifier` is a lazy property, and touching it here
        # would fork its 32-process sim pool just to check whether it exists.
        built = policy.__dict__.get('_verifier') or policy.__dict__.get('verifier')
        if built is not None:
            built.value_fn = verifier_value
    if seed is None:
        seed = int(cfg.training.get('seed', 42))
    states, episode_idxs = get_test_states(cfg)
    states, episode_idxs = states[:n_seeds], episode_idxs[:n_seeds]
    experts = expert_trajectories(cfg, episode_idxs)
    To, Ta = policy.n_obs_steps, policy.n_action_steps

    # One env pool and one verifier pool for the whole sweep: both are expensive to build
    # (the verifier spawns 32 sim workers) and neither depends on n.
    native = resolved_verifier_value(policy, cfg)
    vv = verifier_value or native
    if verifier_value is not None and verifier_value != native:
        print(f'verifier value OVERRIDDEN: {native} -> {vv}   '
              f'(the rollout differs from a native render)')
    print(f'verifier value = {vv}'
          + ('   (pre-2026-08-19 rule: flat across candidates until the arm touches the T)'
             if vv == 't_goal' else ''))

    env = build_envs(len(states), To, Ta, max_steps)
    try:
        for n_actions in n_list:
            # Warn ONCE per width, not per frame: a dropped slot is a silent relabel
            # otherwise, and at n<16 a request for slot 15 is a mistake worth seeing.
            dropped = [k for k in hl_slots if not (0 <= k < n_actions)]
            if dropped:
                print(f'warning: --highlight-slots {dropped} outside range(0,{n_actions}) '
                      f'at n={n_actions}; dropped (not wrapped)')
            render_one(policy, cfg, env, states, episode_idxs, experts, n_actions, out,
                       label, step, seed, device, max_steps, fps, hold, subgoals,
                       max_fan, zoom_px, zoom_q, skip_blind, blind_eps, execute, vv,
                       value_strip_on=value_strip, closeup=closeup, hl_slots=hl_slots)
    finally:
        env.close()
        close = getattr(policy, 'close', None)
        if close is not None:
            try:
                close()
            except Exception as e:
                print(f'warning: verifier close failed: {e}')


if __name__ == '__main__':
    main()
