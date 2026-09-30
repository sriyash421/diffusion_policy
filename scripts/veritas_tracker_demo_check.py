"""Sanity-check the Veritas waypoint tracker against the expert demos.

For every test episode with a plan in ``--plan-dir``, the episode's EXPERT demo
(agent_pos + block_pos from the zarr, one row per control step) is replayed through a
``DualTracker`` exactly as eval's ``VeritasTrackerWrapper`` would feed it. On a good
plan both tracks' progress should rise as the demo unfolds and the T track should
latch its final (goal) pose by the end -- the demos all solve the task, so a track
that never advances is a plan (or tracker) problem, not a demo problem.

Writes ``demo_tracker_check.png`` beside the plans and prints a per-episode table.

Usage:
    python scripts/veritas_tracker_demo_check.py --plan-dir media/veritas_pusht/v5_pusher_and_t
"""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.pusht_image_dataset import (
    load_split_manifest, masks_from_manifest)
from diffusion_policy.env.pusht.veritas.tracker import DualTracker, load_plan

# Colorblind-safe: hue is paired with line STYLE and each line carries its own label.
PUSHER_STYLE = dict(color='#0072B2', linestyle='-', label='pusher track')
T_STYLE = dict(color='#D55E00', linestyle='--', label='T track')
SCORE_STYLE = dict(color='#444444', linestyle=':', label='dual score')


def episode_slice(rb, episode_idx):
    ends = np.asarray(rb.episode_ends[:])
    start = 0 if episode_idx == 0 else int(ends[episode_idx - 1])
    return start, int(ends[episode_idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr', default='data/pusht_cchi_v7_replay.zarr')
    ap.add_argument('--split-file',
                    default='diffusion_policy/config/splits/pusht_seed42_train176.json')
    ap.add_argument('--plan-dir', default='media/veritas_pusht/v5_pusher_and_t')
    args = ap.parse_args()

    plan_dir = pathlib.Path(args.plan_dir)
    rb = ReplayBuffer.copy_from_path(args.zarr, keys=['agent_pos', 'block_pos'])
    manifest = load_split_manifest(args.split_file, episode_ends=rb.episode_ends[:])
    _, _, test_mask = masks_from_manifest(manifest, rb.n_episodes)
    test_eps = np.nonzero(test_mask)[0].tolist()

    episodes = []
    for ep in test_eps:
        path = plan_dir / f'ep{ep}.json'
        if not path.exists():
            continue
        try:
            episodes.append((ep, load_plan(path)))
        except ValueError:
            print(f'ep{ep}: json has no plan (failed Gemini call), skipped')
    if not episodes:
        sys.exit(f'no plans found in {plan_dir}/')

    cols = 4
    rows = (len(episodes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows),
                             sharey=True, squeeze=False)
    print(f'{"ep":>4} {"steps":>5} {"pusher %":>9} {"T %":>7} {"T latched":>9} {"score":>6}')
    for k, (ep, plan) in enumerate(episodes):
        s, e = episode_slice(rb, ep)
        agent = np.asarray(rb['agent_pos'][s:e], dtype=np.float64)
        block = np.asarray(rb['block_pos'][s:e], dtype=np.float64)
        tracker = DualTracker(plan, agent[0], block[0])
        prog = []
        for a, b in zip(agent, block):
            tracker.update(a, b)
            p_now, t_now = tracker.progress
            prog.append((p_now, t_now, tracker.score))
        prog = np.asarray(prog)     # (T, 3): pusher progress, T progress, dual score
        ax = axes[k // cols][k % cols]
        ax.plot(prog[:, 0], **PUSHER_STYLE)
        if not np.isnan(prog[:, 1]).all():
            ax.plot(prog[:, 1], **T_STYLE)
        ax.plot(prog[:, 2], **SCORE_STYLE)
        t_latched = (tracker.t is not None and tracker.t.done)
        ax.set_title(f'ep {ep}' + ('  [T latched]' if t_latched else ''), fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        if k == 0:
            ax.legend(fontsize=7, loc='upper left')
        p_prog, t_prog = tracker.progress
        print(f'{ep:>4} {e - s:>5} {p_prog:>9.2f} '
              f'{(t_prog if not np.isnan(t_prog) else float("nan")):>7.2f} '
              f'{str(t_latched):>9} {tracker.score:>6.3f}')
    for k in range(len(episodes), rows * cols):
        axes[k // cols][k % cols].axis('off')
    fig.supxlabel('demo control step')
    fig.supylabel('progress / score')
    fig.tight_layout()
    out = plan_dir / 'demo_tracker_check.png'
    fig.savefig(out, dpi=120)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
