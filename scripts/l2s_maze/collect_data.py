"""Collect expert demos for one split: data/l2s_maze/<split>/train/expert.zarr.

Goals come from the split's TRAIN pool, starts are uniform over free cells. Only successful
episodes of length >= --min-episode-steps are kept.

    python scripts/l2s_maze/collect_data.py --split as_is
"""
import argparse
import json
import pathlib

import numpy as np
import zarr
from tqdm import tqdm

from expert_utils import GOAL_TOL, STEP_SCALE, CachedExpert
from maze_layout import (DEFAULT_LAYOUT_DIR, REPO, SPLITS, free_cells, goal_pool,
                         load_layout, sample_pair)


def collect(split, layout_dir, n_episodes, seed, max_expert_steps, min_episode_steps, out):
    maze, layout = load_layout(layout_dir)
    free = free_cells(maze)
    goals = goal_pool(maze, split, 'train', layout['r_inner'])
    rng = np.random.default_rng(seed)
    expert = CachedExpert(maze)

    buf = {'image': [], 'agent_pos': [], 'goal_pos': [], 'action': []}
    ends, starts_ij, goals_ij, lengths = [], [], [], []
    n_attempted = n_failed = n_short = 0
    pbar = tqdm(total=n_episodes, desc=f'collect {split}')
    while len(ends) < n_episodes:
        start, goal = sample_pair(rng, free, goals)
        n_attempted += 1
        success, n_steps, ep = expert.rollout(start, goal, max_expert_steps, record=True)
        if not success:
            n_failed += 1
            continue
        if n_steps < min_episode_steps:
            n_short += 1
            continue
        for k in buf:
            buf[k].extend(ep[k])
        ends.append(len(buf['action']))
        starts_ij.append(start)
        goals_ij.append(goal)
        lengths.append(n_steps)
        pbar.update(1)
    pbar.close()

    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(out), mode='w')
    obs = root.create_group('data').create_group('obs')
    obs.create_dataset('image', data=np.asarray(buf['image'], dtype=np.uint8))
    obs.create_dataset('agent_pos', data=np.asarray(buf['agent_pos'], dtype=np.float32))
    obs.create_dataset('goal_pos', data=np.asarray(buf['goal_pos'], dtype=np.float32))
    root['data'].create_dataset('actions', data=np.asarray(buf['action'], dtype=np.float32))
    root['data'].create_dataset('expert_mask', data=np.ones(len(buf['action']), np.float32))
    meta = root.create_group('meta')
    meta.create_dataset('episode_ends', data=np.asarray(ends, dtype=np.int64))
    meta.create_dataset('start_ij', data=np.asarray(starts_ij, dtype=np.int64))
    meta.create_dataset('goal_ij', data=np.asarray(goals_ij, dtype=np.int64))
    info = {
        'split': split, 'seed': seed, 'layout_sha': layout['sha'],
        'n_episodes': len(ends), 'n_frames': int(ends[-1]),
        'n_attempted': n_attempted, 'n_failed': n_failed, 'n_short': n_short,
        'max_expert_steps': max_expert_steps, 'min_episode_steps': min_episode_steps,
        'step_scale': STEP_SCALE, 'goal_tol': GOAL_TOL, 'n_goal_pool': len(goals),
        'episode_len': {'min': int(min(lengths)), 'max': int(max(lengths)),
                        'mean': float(np.mean(lengths))},
    }
    meta.attrs.update(info)
    (out.parent / 'collect_info.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=SPLITS, required=True)
    ap.add_argument('--layout-dir', default=str(DEFAULT_LAYOUT_DIR))
    ap.add_argument('--n-episodes', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-expert-steps', type=int, default=1000)
    ap.add_argument('--min-episode-steps', type=int, default=16)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = args.out or REPO / f'data/l2s_maze/{args.split}/train/expert.zarr'
    collect(args.split, args.layout_dir, args.n_episodes, args.seed,
            args.max_expert_steps, args.min_episode_steps, out)


if __name__ == '__main__':
    main()
