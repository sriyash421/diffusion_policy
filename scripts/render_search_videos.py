"""Render per-decision search videos for the PushT best-of-N eval.

One video frame per DECISION (every n_action_steps env steps), because a decision is
exactly one complete best-of-N search: the frame shows the state the search was run from,
every sampled candidate's executed action chunk overlaid on the scene, and the chosen
(argmax verifier value) one highlighted. Sub-decision motion is not shown -- MultiStepWrapper
only surfaces the last n_obs_steps frames per step call, and the interesting object here is
the search, not the interpolation between waypoints.

For the subgoal search-context modes the strip beneath the scene shows the observation each
candidate's chunk actually reaches in the verifier's sim, in generation order, with the
executed one outlined -- i.e. what the next candidate got to condition on.

Usage:
  python scripts/render_search_videos.py -c <ckpt> --label value --n 16 \
      --out-dir videos/jul31_debug/value --n-seeds 10

Writes <out-dir>/seed<NN>_n<N>_step<STEP>_<label>_<succ|trunc>.mp4 plus a sidecar
episodes.json with per-seed length / success / reward, appended across invocations.
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
from eval_search_pusht import build_envs, get_test_states, load_policy, SUCCESS_REWARD


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


def _fan(canvas, chunks, best, to_px, w_loser=1, w_best=2, dot=4):
    """Draw the candidate fan onto `canvas` using the coordinate map `to_px`.

    Losers first and translucent, so the chosen path is never buried under them.
    Colours are BGR. Candidate order is generation order, which is also context order.
    """
    overlay = canvas.copy()
    for c in range(len(chunks)):
        if c == best:
            continue
        pts = np.array([to_px(p) for p in chunks[c]], dtype=np.int32)
        cv2.polylines(overlay, [pts], False, (150, 150, 150), w_loser, cv2.LINE_AA)
        cv2.circle(overlay, tuple(pts[-1]), max(1, dot // 2), (110, 110, 110), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

    pts = np.array([to_px(p) for p in chunks[best]], dtype=np.int32)
    cv2.polylines(canvas, [pts], False, (0, 0, 0), w_best + 2, cv2.LINE_AA)      # halo
    cv2.polylines(canvas, [pts], False, (60, 200, 255), w_best, cv2.LINE_AA)     # chosen
    cv2.circle(canvas, tuple(pts[0]), dot, (255, 255, 255), -1, cv2.LINE_AA)     # chunk start
    cv2.circle(canvas, tuple(pts[-1]), dot, (60, 200, 255), -1, cv2.LINE_AA)     # chunk end
    return canvas


ZOOM = 150      # picture-in-picture inset, px


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


def draw_candidates(panel, chunks, scores, best):
    """Scene with the candidate fan, plus a zoomed inset of that fan.

    An executed chunk spans only ~8 small waypoints, so at scene scale the whole search
    collapses into a smudge beside the agent. The inset crops the fan's bounding box out of
    the scene and blows it up, which is the only way the spread between candidates -- the
    thing these videos exist to show -- is actually legible.
    """
    _fan(panel, chunks, best, _pt)

    all_pts = chunks.reshape(-1, 2)
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    ctr, half = (lo + hi) / 2.0, max((hi - lo).max() * 0.75, 24.0)
    x0, y0, x1, y1 = ctr[0] - half, ctr[1] - half, ctr[0] + half, ctr[1] + half
    src = panel.shape[0]
    cx0, cy0 = max(0, int(x0 / WINDOW * src)), max(0, int(y0 / WINDOW * src))
    cx1, cy1 = min(src, int(x1 / WINDOW * src)), min(src, int(y1 / WINDOW * src))
    if cx1 - cx0 >= 4 and cy1 - cy0 >= 4:
        # crop the CLEAN scene (before the fan was drawn) so the zoom is not double-drawn
        crop = cv2.resize(panel[cy0:cy1, cx0:cx1], (ZOOM, ZOOM), interpolation=cv2.INTER_NEAREST)
        sx, sy = ZOOM / max(1, (cx1 - cx0)), ZOOM / max(1, (cy1 - cy0))
        def to_zoom(p):
            return (int(round((p[0] / WINDOW * src - cx0) * sx)),
                    int(round((p[1] / WINDOW * src - cy0) * sy)))
        _fan(crop, chunks, best, to_zoom, w_loser=1, w_best=2, dot=3)
        cv2.rectangle(crop, (0, 0), (ZOOM - 1, ZOOM - 1), (200, 200, 200), 1)
        cv2.putText(crop, f'{2*half:.0f}px', (4, ZOOM - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (200, 200, 200), 1, cv2.LINE_AA)
        panel[src - ZOOM:src, 0:ZOOM] = crop
    return panel


LABEL_H = 13     # per-thumbnail caption strip
STRIP_PAD = 14   # breathing room beneath the whole subgoal block


def subgoal_strip(images, best, values=None, width=SCENE, rows=None):
    """Tile per-candidate reached observations, generation order, chosen one outlined.

    Each thumbnail is the observation that candidate i's executed chunk actually REACHES in
    the verifier sim -- so it is what the NEXT candidate gets to condition on. An 8-step
    chunk barely moves the scene, so these are near-identical by construction (measured at
    ~2/255 mean pairwise difference on the overfit checkpoints); the per-thumbnail verifier
    value below is therefore the part that actually distinguishes them, and is printed here
    rather than left to the header, which only ever showed the chosen candidate's score.
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
            cv2.rectangle(thumb, (0, 0), (cell - 1, cell - 1), (60, 200, 255), 2)
        strip[y0:y0 + cell, x0:x0 + cell] = thumb
        if values is not None:
            v = float(values[i])
            # shade the label by rank so the spread is legible even when the images are not
            t = 0.0 if hi <= lo else (v - lo) / (hi - lo)
            col = (60, 200, 255) if i == best else (int(90 + 90 * t),) * 3
            cv2.putText(strip, f'{v:.0f}', (x0 + 2, y0 + cell + LABEL_H - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, col, 1, cv2.LINE_AA)
    return strip


def compose(panel, strip, lines):
    """Stack header / scene / subgoal strip. `lines` is a list of caption lines -- two
    short lines fit the 384px width where one long one was silently clipped."""
    w = panel.shape[1]
    head = HEADER + 14 * (len(lines) - 1)
    parts = [np.full((head, w, 3), 24, dtype=np.uint8), panel]
    if strip is not None:
        parts += [np.full((PAD, w, 3), 24, dtype=np.uint8), strip]
    frame = np.vstack(parts)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (8, 18 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (235, 235, 235), 1, cv2.LINE_AA)
    return frame


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('--label', required=True, help='feedback mechanism name, used in filenames')
@click.option('--n', 'n_actions', default=16, help='candidates per decision')
@click.option('--out-dir', required=True)
@click.option('--n-seeds', default=10)
@click.option('-d', '--device', default='cuda:0')
@click.option('--max-steps', default=300)
@click.option('--fps', default=4)
@click.option('--hold', default=4, help='video frames per decision')
@click.option('--subgoals/--no-subgoals', default=True,
              help='render the per-candidate reached observation strip')
def main(checkpoint, label, n_actions, out_dir, n_seeds, device, max_steps, fps, hold, subgoals):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    step = int(''.join(ch for ch in pathlib.Path(checkpoint).stem if ch.isdigit()) or 0)

    policy, cfg = load_policy(checkpoint, device)
    states, episode_idxs = get_test_states(cfg)
    states, episode_idxs = states[:n_seeds], episode_idxs[:n_seeds]
    experts = expert_trajectories(cfg, episode_idxs)
    To, Ta = policy.n_obs_steps, policy.n_action_steps
    sl = slice(To - 1, To - 1 + Ta)

    env = build_envs(len(states), To, Ta, max_steps)
    try:
        def make_init_fn(state):
            state = np.asarray(state, dtype=np.float64)
            def _fn(e):
                e.unwrapped.reset_to_state = state
            return _fn
        env.call_each('run_dill_function',
                      args_list=[(dill.dumps(make_init_fn(s)),) for s in states])
        obs = env.reset()
        policy.reset()

        # Temp names must be unique per (label, step, n): several invocations share one
        # output folder (e.g. step 10000 and step 30000 of the same run) and may be in
        # flight at once, so a bare `.tmp_seed0.mp4` lets one job rename the other's file
        # out from under it.
        stem = f'.tmp_{label}_{step}_n{n_actions}_seed'
        writers = [imageio.get_writer(str(out / f'{stem}{i}.mp4'), fps=fps,
                                      codec='libx264', quality=8,
                                      macro_block_size=1) for i in range(len(states))]
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
            sub_img = None
            if sub is not None and sub.get('image', None) is not None:
                sub_img = (sub['image'].clamp(0, 1) * 255).byte().permute(0, 1, 3, 4, 2) \
                          .cpu().numpy()                                # (B, n, H, W, 3)

            for i in range(len(states)):
                panel = draw_expert(_scene(scenes[i]), experts[i])
                panel = draw_candidates(panel, chunks[i], sc[i], int(bi[i]))
                strip = (subgoal_strip(sub_img[i], int(bi[i]), values=sc[i])
                         if sub_img is not None else None)
                lo, hi = float(sc[i].min()), float(sc[i].max())
                cap = [f'ep{episode_idxs[i]}  dec {decision}  n={n_actions}  '
                       f'grn=demo path  cyn=chosen  '
                       f'step{step}  {label}',
                       f'chosen #{int(bi[i])}  value {sc[i][int(bi[i])]:.1f}   '
                       f'spread {hi - lo:.1f}  [{lo:.1f} .. {hi:.1f}]']
                frame = compose(panel, strip, cap)
                for _ in range(hold):
                    writers[i].append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            act = actions[torch.arange(len(states), device=actions.device), best][:, sl]
            obs, reward, done, info = env.step(act.detach().cpu().numpy())
            done = np.all(done)
            decision += 1
            pbar.update(1)
        pbar.close()

        rewards = env.call('get_attr', 'reward')
        for w in writers:
            w.close()

        records = []
        for i in range(len(states)):
            r = np.asarray(rewards[i])
            succ = bool(r.max() >= SUCCESS_REWARD)
            length = int(len(r))
            tag = 'succ' if succ else 'trunc'
            name = f'seed{i:02d}_n{n_actions}_step{step}_{label}_{tag}.mp4'
            (out / f'{stem}{i}.mp4').rename(out / name)
            records.append({'seed': i, 'episode_idx': int(episode_idxs[i]), 'n': n_actions,
                            'step': step, 'label': label, 'success': succ,
                            'env_steps': length, 'decisions': int(np.ceil(length / Ta)),
                            'max_reward': float(r.max()), 'video': name})
            print(f'  {name}  len={length} steps  max_reward={r.max():.3f}')

        # One sidecar per (step, n) rather than a shared episodes.json: concurrent
        # invocations into the same folder would read-modify-write the shared file and
        # silently drop each other's records. Merge at read time instead.
        idx = out / f'episodes_step{step}_n{n_actions}.json'
        idx.write_text(json.dumps(records, indent=2))
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
