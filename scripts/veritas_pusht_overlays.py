"""Ask Gemini for a Veritas waypoint plan on PushT test starts, and draw it.

Gemini sees the policy's native 96x96 image obs; the overlay is drawn on a x5 nearest
upscale because Veritas's marker sizes are tuned for 640x480. Waypoint 0 is the pusher's
own pixel, prepended as VeritasVerifier.initialize_episode does, so the reference
trajectory starts where Veritas's tracked plan starts. The episode's expert demo (the
pusher's actual path, agent_pos) is drawn dashed beneath it.

Start states are the split's TEST episodes, exactly as PushTSearchImageRunner resets them.
Each run writes into <outdir>/<prompt name>/ and stores the prompt template there as
prompt.txt. Needs GEMINI_API_KEY exported in the shell.

    python scripts/veritas_pusht_overlays.py --prompt v2_push_rules --n-episodes 8
        # -> media/veritas_pusht/v2_push_rules/
    python scripts/veritas_pusht_overlays.py --prompt v2_push_rules --redraw
        # re-draw that folder's overlays from the saved plans; no Gemini calls
"""
import argparse
import json
import logging
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.pusht_image_dataset import (
    get_episode_init_states, load_split_manifest, masks_from_manifest)
from diffusion_policy.env.pusht.feedback_util import (
    GOAL_KEYPOINTS, compute_feedback_from_pose, keypoints_at_pose)
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.env.pusht.veritas.draw import draw_waypoints_rgb
from diffusion_policy.env.pusht.veritas.pusht_client import (
    PLAN_TYPES, PROMPTS, WORKSPACE_SIZE, PushTVeritasClient, build_prompt, gemini_hints,
    obs_to_gemini_inputs, plan_kind, pose_keypoints_px)
from diffusion_policy.env.pusht.veritas.schemas import VeritasDualPlan, VeritasPosePlan

UPSCALE = 5
LINE_H = 17
GRID_COLS = 4
DEMO_COLOR = (40, 40, 40)
DEMO_DASH, DEMO_GAP = 10, 6
LEGENDS = {
    'pusher': ['solid: Gemini ref (0 = start) | dashed: expert demo, square = end'],
    'pose': ['solid+n: planned T (0 = now) | dotted: demo T | dashed: demo pusher'],
    'dual': ['pins 0-n + curve: planned pusher path (0 = pusher now)',
             'outline + Tn: planned T | dotted: demo T | dashed: demo pusher'],
}
PLAN_T_COLOR = (50, 150, 210)    # RGB of draw_waypoints_rgb's pin blue
DOT_DASH, DOT_GAP = 3, 4
N_DEMO_T = 5


def load_replay(zarr_path):
    return ReplayBuffer.copy_from_path(zarr_path, keys=['agent_pos', 'block_pos'])


def split_starts(rb, split_file, split='test'):
    """[(episode_idx, (5,) reset state)] for one split's episodes.

    `test` is what the eval-time verifier values need; `train` (and `val`, on splits
    that have one) is what a TRAINING run under a waypoint verifier_tag needs -- its
    TrackerSnapshotStore refuses to start without a plan for every episode it can touch.
    """
    manifest = load_split_manifest(split_file, episode_ends=rb.episode_ends[:])
    masks = dict(zip(('train', 'val', 'test'),
                     masks_from_manifest(manifest, rb.n_episodes)))
    mask = masks[split]
    if not mask.any():
        raise SystemExit(f'{split_file} has no {split!r} episodes')
    states = get_episode_init_states(rb, mask)
    return list(zip(np.nonzero(mask)[0].tolist(), states))


def demo_t_keypoints(rb, episode_idx, image_size=96, n=N_DEMO_T):
    """(n, 8, 2) pixel keypoints of the demo's T at n evenly spaced times after the start,
    the last being the demo's final pose."""
    ends = np.asarray(rb.episode_ends[:])
    start = 0 if episode_idx == 0 else int(ends[episode_idx - 1])
    poses = np.asarray(rb['block_pos'][start:int(ends[episode_idx])], dtype=np.float64)
    idx = np.round(np.linspace(0, len(poses) - 1, n + 1)[1:]).astype(int)
    return keypoints_at_pose(poses[idx]).astype(np.float64) * (image_size / WORKSPACE_SIZE)


def demo_path(rb, episode_idx):
    """(T, 2) the expert demo's pusher positions (agent_pos), workspace coords."""
    ends = np.asarray(rb.episode_ends[:])
    start = 0 if episode_idx == 0 else int(ends[episode_idx - 1])
    return np.asarray(rb['agent_pos'][start:int(ends[episode_idx])], dtype=np.float64)


def start_obs(episode_idx, state, render_size=96):
    """The first image obs, agent_pos and feedback of an episode, as eval sees them."""
    env = PushTImageEnv(legacy=False, render_size=render_size)
    env.reset_to_state = np.asarray(state)
    env.seed(episode_idx)
    obs = env.reset()
    block_pose = np.array(list(env.block.position) + [env.block.angle])
    feedback = compute_feedback_from_pose(block_pose)
    env.close()
    return obs['image'], obs['agent_pos'], feedback


def _dashed_polyline(img, pts, color, thickness, dash=DEMO_DASH, gap=DEMO_GAP):
    """Dashed line along *pts* (Nx2), dashes spaced by arc length."""
    pts = np.asarray(pts, dtype=np.float64)
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.r_[True, seg_len > 1e-9]]              # np.interp needs increasing arc length
    if len(pts) < 2:
        return
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    for a in np.arange(0.0, s[-1], dash + gap):
        t = np.linspace(a, min(a + dash, s[-1]), 6)
        seg = np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=-1)
        cv2.polylines(img, [np.round(seg).astype(np.int32)], False, color, thickness,
                      cv2.LINE_AA)


def _draw_demo(img, demo_uv):
    """Expert pusher path, dashed with a white halo, and a hollow square at its end."""
    for color, thick in (((255, 255, 255), 4), (DEMO_COLOR, 2)):
        _dashed_polyline(img, demo_uv, color, thick)
    u, v = np.round(demo_uv[-1]).astype(int)
    for color, thick in (((255, 255, 255), 4), (DEMO_COLOR, 2)):
        cv2.rectangle(img, (u - 6, v - 6), (u + 6, v + 6), color, thick, cv2.LINE_AA)


def _t_polygons(kps):
    """(8, 2) keypoints in T_VERTS order -> the bar and stem outlines."""
    return [np.asarray(kps[:4]), np.asarray(kps[4:])]


def _draw_t_outline(img, kps, color, dotted=False):
    """T outline (bar + stem) with a white halo; solid, or dotted when ``dotted``."""
    for poly in _t_polygons(kps):
        closed = np.vstack([poly, poly[:1]])
        for c, thick in (((255, 255, 255), 3), (color, 1)):
            if dotted:
                _dashed_polyline(img, closed, c, thick, DOT_DASH, DOT_GAP)
            else:
                cv2.polylines(img, [np.round(closed).astype(np.int32)], False, c, thick,
                              cv2.LINE_AA)


def _pins(start_uv, pts, labels, start_label=0):
    """draw_waypoints_rgb markers (x5 coords) for ``pts``, after ``start_uv`` if given."""
    wps = [] if start_uv is None else [
        {'uv': [float(c) * UPSCALE for c in start_uv], 'idx': start_label}]
    return wps + [{'uv': [c * UPSCALE for c in uv], 'idx': lab} for uv, lab in zip(pts, labels)]


def overlay(frame, kind, plan, caption, g, demo_uv=None, demo_t_kps=(), image_size=96):
    """x5 upscaled frame with the expert demo (pusher path, T outlines for T-pose plans),
    the plan on top, and a caption.
      pusher: Veritas's numbered waypoints + reference curve from the pusher (0).
      pose:   a T outline per pose, numbered pins + curve from the current T centroid (0).
      dual:   T outlines labelled T1..Tn, and the pusher path numbered from the pusher (0).
    ``g`` is obs_to_gemini_inputs output; ``demo_uv`` / ``demo_t_kps`` in ``frame`` pixels."""
    big = cv2.resize(frame, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_NEAREST)
    for kps in demo_t_kps:
        _draw_t_outline(big, np.asarray(kps) * UPSCALE, DEMO_COLOR, dotted=True)
    if demo_uv is not None:
        _draw_demo(big, np.asarray(demo_uv) * UPSCALE)
    if plan is not None:
        t_poses = _t_poses(plan)
        for wp in t_poses:
            _draw_t_outline(big, pose_keypoints_px(wp, image_size) * UPSCALE, PLAN_T_COLOR)
        if kind == 'dual':
            big = draw_waypoints_rgb(big, _pins(None, [wp.centroid_uv for wp in t_poses],
                                                [f'T{i + 1}' for i in range(len(t_poses))]))
            pusher = [wp.uv for wp in plan.pusher_waypoints]
            big = draw_waypoints_rgb(big, _pins(g['pusher_uv'], pusher,
                                                range(1, len(pusher) + 1)), draw_curve=True)
        else:
            start = g['block_uv'] if kind == 'pose' else g['pusher_uv']
            pts = [wp.centroid_uv if kind == 'pose' else wp.uv for wp in plan.waypoints]
            big = draw_waypoints_rgb(big, _pins(start, pts, range(1, len(pts) + 1)),
                                     draw_curve=True)
    cap_lines, legend = caption.split('\n'), LEGENDS[kind]
    strip = np.full((LINE_H * (len(cap_lines) + len(legend)) + 8, big.shape[1], 3), 255,
                    dtype=np.uint8)
    for i, line in enumerate(cap_lines + legend):
        cv2.putText(strip, line, (6, LINE_H * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42 if i < len(cap_lines) else 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([big, strip], axis=0)


# Paid-tier standard $ per 1M tokens (input, output incl. thinking), ai.google.dev pricing
# as of 2026-09-28; 3.8-flash doubles on 2027-01-01. Batch API is half.
PRICE_PER_M = {'gemini-3.8-flash': (0.75, 3.75), 'gemini-3.5-flash': (1.50, 9.00)}


def cost_usd(usage, model):
    """Standard paid-tier cost of one call, or None if the model or usage is unknown."""
    if not usage or model not in PRICE_PER_M:
        return None
    p_in, p_out = PRICE_PER_M[model]
    out = (usage.get('candidates_token_count') or 0) + (usage.get('thoughts_token_count') or 0)
    return ((usage.get('prompt_token_count') or 0) * p_in + out * p_out) / 1e6


def _t_poses(plan):
    """The plan's T-pose waypoints ([] for a pusher-only plan)."""
    if isinstance(plan, VeritasDualPlan):
        return plan.t_poses
    return plan.waypoints if isinstance(plan, VeritasPosePlan) else []


def final_to_goal_px(plan, image_size=96):
    """Mean keypoint distance (px) from a plan's last T pose to the goal T."""
    kps = pose_keypoints_px(_t_poses(plan)[-1], image_size)
    goal = GOAL_KEYPOINTS * (image_size / WORKSPACE_SIZE)
    return float(np.linalg.norm(kps - goal, axis=-1).mean())


def caption_for(episode_idx, plan, usage=None, model=None):
    if plan is None:
        return f'ep {episode_idx} | no plan'
    if isinstance(plan, VeritasDualPlan):
        cap = (f'ep {episode_idx} | {len(plan.pusher_waypoints)} pusher wp'
               f' | {len(plan.t_poses)} T poses | conf {plan.confidence:.2f}')
    else:
        tols = [wp.tol_px for wp in plan.waypoints]
        cap = (f'ep {episode_idx} | {len(plan.waypoints)} wp | conf {plan.confidence:.2f}'
               f' | tol {min(tols):g}-{max(tols):g} px')
    if _t_poses(plan):
        cap += f' | end-goal {final_to_goal_px(plan):.1f} px'
    if usage:
        cap += f"\nthinking tokens {usage.get('thoughts_token_count')}"
    cost = cost_usd(usage, model)
    if cost is not None:
        cap += f' | cost ${cost:.4f}'
    return cap


def usage_str(usage):
    """' | tokens in/out/thought ...' from a response's usage metadata, or ''."""
    if not usage:
        return ''
    return (f" | tokens in {usage.get('prompt_token_count')} out "
            f"{usage.get('candidates_token_count')} thought {usage.get('thoughts_token_count')}")


def grid(tiles, cols=GRID_COLS):
    cols = min(cols, len(tiles))
    h, w = max(t.shape[0] for t in tiles), tiles[0].shape[1]   # caption heights differ
    rows = -(-len(tiles) // cols)
    out = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[r * h:r * h + t.shape[0], c * w:(c + 1) * w] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr', default='data/pusht_cchi_v7_replay.zarr')
    ap.add_argument('--split-file', default='diffusion_policy/config/splits/pusht_seed42_train176.json')
    ap.add_argument('--outdir', default='media/veritas_pusht')
    ap.add_argument('--prompt', choices=sorted(PROMPTS), default='v2_push_rules')
    ap.add_argument('--n-episodes', type=int, default=8)
    ap.add_argument('--split', choices=('train', 'val', 'test'), default='test',
                    help='which split of the manifest to plan; training under a waypoint '
                         'verifier_tag needs the TRAIN (and any val) episodes planned too')
    ap.add_argument('--model', default='gemini-3.8-flash')
    ap.add_argument('--redraw', action='store_true',
                    help='re-draw the episodes already in the folder from their saved plans')
    ap.add_argument('--skip-planned', action='store_true',
                    help='only call Gemini for episodes whose json has no plan yet, so an '
                         'interrupted or partly-failed sweep resumes without re-paying for '
                         '(or replacing) the plans it already has')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    rb = load_replay(args.zarr)
    starts = split_starts(rb, args.split_file, args.split)
    template = PROMPTS[args.prompt]
    kind = plan_kind(args.prompt)
    plan_cls = PLAN_TYPES[kind][0]
    out = pathlib.Path(args.outdir) / args.prompt
    if args.redraw:
        attempted = {int(p.stem[2:]) for p in out.glob('ep*.png')}
        episodes = [(i, s) for i, s in starts if i in attempted]
    else:
        try:
            client = PushTVeritasClient(model_name=args.model)
        except ValueError as e:
            sys.exit(f'{e}\nExport the key in this terminal first: export GEMINI_API_KEY=...')
        out.mkdir(parents=True, exist_ok=True)
        (out / 'prompt.txt').write_text(
            f'# prompt: {args.prompt}\n# model: {args.model}\n# filled per episode by '
            f'pusht_client.build_prompt; the filled text is in each ep*.json\n\n{template}')
        episodes = starts[:args.n_episodes]

    tiles, planned, called = [], [], []
    for episode_idx, state in episodes:
        image, agent_pos, feedback = start_obs(episode_idx, state)
        g = obs_to_gemini_inputs(image, agent_pos, feedback)
        rec_path = out / f'ep{episode_idx}.json'
        if args.redraw or (args.skip_planned and rec_path.exists()
                           and 'plan' in json.loads(rec_path.read_text())):
            plan = (plan_cls(**json.loads(rec_path.read_text())['plan'])
                    if rec_path.exists() and 'plan' in json.loads(rec_path.read_text())
                    else None)
        else:
            plan = client.call_pusht_plan(
                g['frame'], build_prompt(template, g), gemini_hints(g),
                log_dir=str(out), log_name=f'ep{episode_idx}', kind=kind)
            called.append(episode_idx)
            if plan is not None:
                # the clipped plan beside the raw response the client logged
                rec = json.loads(rec_path.read_text())
                rec['plan'] = plan.model_dump()
                rec_path.write_text(json.dumps(rec, indent=2))
        rec = json.loads(rec_path.read_text()) if plan is not None else {}
        usage = rec.get('usage') or {}
        caption = caption_for(episode_idx, plan, usage, rec.get('model'))
        cost = cost_usd(usage, rec.get('model'))
        print(caption.replace('\n', ' | ') + usage_str(usage), flush=True)
        size = g['frame'].shape[0]
        demo_uv = demo_path(rb, episode_idx) * (size / WORKSPACE_SIZE)
        demo_t = demo_t_keypoints(rb, episode_idx, size) if kind != 'pusher' else ()
        tile = overlay(g['frame'], kind, plan, caption, g, demo_uv, demo_t, size)
        cv2.imwrite(str(out / f'ep{episode_idx}.png'), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
        tiles.append(tile)
        if plan is not None:
            planned.append(tile)

    # one grid per split, so planning the train episodes does not overwrite the test grid
    sfx = '' if args.split == 'test' else f'_{args.split}'
    cv2.imwrite(str(out / f'overlays{sfx}.png'), cv2.cvtColor(grid(tiles), cv2.COLOR_RGB2BGR))
    if planned:
        cv2.imwrite(str(out / f'overlays_success{sfx}.png'),
                    cv2.cvtColor(grid(planned), cv2.COLOR_RGB2BGR))
    print(f'wrote {len(tiles)} overlays ({len(planned)} with a plan) to {out}/')
    print(usage_summary(out, called, args.model))


def usage_summary(out, called, model):
    """Gemini usage + cost of THIS run's calls, and of every plan the folder holds."""
    def tally(paths):
        n, tok, usd = 0, {'in': 0, 'out': 0, 'thought': 0}, 0.0
        for p in paths:
            rec = json.loads(p.read_text())
            u = rec.get('usage') or {}
            if not u:
                continue
            n += 1
            tok['in'] += u.get('prompt_token_count') or 0
            tok['out'] += u.get('candidates_token_count') or 0
            tok['thought'] += u.get('thoughts_token_count') or 0
            usd += cost_usd(u, rec.get('model')) or 0.0
        return n, tok, usd

    def line(label, n, tok, usd):
        return (f'{label}: {n} Gemini call(s), tokens in {tok["in"]} / out {tok["out"]} / '
                f'thought {tok["thought"]}, ${usd:.3f}')

    this = tally(out / f'ep{i}.json' for i in called if (out / f'ep{i}.json').exists())
    every = tally(sorted(out.glob('ep*.json')))
    return (f'GEMINI USAGE ({model})\n  ' + line('this run', *this) + '\n  '
            + line(f'{out}/ total', *every))


if __name__ == '__main__':
    main()
