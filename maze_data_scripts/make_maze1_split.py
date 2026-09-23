#!/usr/bin/env python3
"""
Three-way episode split of the single-maze dataset, keyed on how far the GOAL is.

    test  = the 50 innermost goals (smallest BFS distance from the fixed start)
    train = 100 sampled from the remaining 150
    val   = 20 more from that remainder
    unused= the other 30

"Innermost" is distance along the corridors, not Euclidean -- the start is the maze's
most central free cell, so the two orderings nearly coincide, but only the corridor
distance says how far the policy actually has to travel.

Note the generalisation direction this creates: the arms train on FAR goals and are
tested on NEAR ones. That is interpolation toward the start, not extrapolation.

Schema and checksums are the ones `pusht_image_dataset` already defines, so an existing
loader validates this file unchanged.

    python maze_data_scripts/make_maze1_split.py
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import zarr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from diffusion_policy.dataset.pusht_image_dataset import (episode_ends_checksum,
                                                          split_checksum)
from procgen_maze_env import bfs_distances, make_maze_split

SPLITS = pathlib.Path(__file__).resolve().parents[1] / 'diffusion_policy/config/splits'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr', default='data/procgen_maze_expert_maze1/procgen_maze_train.zarr')
    ap.add_argument('--n-test', type=int, default=50)
    ap.add_argument('--n-train', type=int, default=100)
    ap.add_argument('--n-val', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='procgen_maze1_inner50.json')
    args = ap.parse_args()

    root = zarr.open(args.zarr, mode='r')
    ends = np.asarray(root['meta/episode_ends'])
    goals = np.asarray(root['data/obs/goal_pos'])[ends - 1]
    meta = dict(root['meta'].attrs)
    maze = make_maze_split(meta['maze_pool_size'], seed=meta['maze_pool_seed'])[
        meta['single_maze']]
    dist = bfs_distances(maze, meta['start_ij'])
    d = np.array([dist[tuple(np.rint(g).astype(int))] for g in goals])

    # Ties broken by episode index, so the partition is a function of the data alone.
    order = np.lexsort((np.arange(len(d)), d))
    test = sorted(int(i) for i in order[:args.n_test])
    rest = order[args.n_test:]

    rng = np.random.default_rng(args.seed)
    perm = rest[rng.permutation(len(rest))]
    train = sorted(int(i) for i in perm[:args.n_train])
    val = sorted(int(i) for i in perm[args.n_train:args.n_train + args.n_val])
    unused = sorted(int(i) for i in perm[args.n_train + args.n_val:])
    assert not (set(train) & set(val)) and not (set(train + val) & set(test))

    lengths = np.diff(np.concatenate([[0], ends]))
    manifest = {
        'generated_by': 'maze_data_scripts/make_maze1_split.py',
        'zarr_path': args.zarr,
        'n_episodes': int(len(ends)),
        'episode_ends_checksum': episode_ends_checksum(ends),
        'derivation': {
            'method': f"test = the {args.n_test} goals with the smallest BFS distance "
                      f"from the fixed start (ties by episode index); train/val sampled "
                      f"without replacement from the remaining "
                      f"{len(ends) - args.n_test}",
            'key': 'bfs_distance(start_ij, goal_pos[episode_end])',
            'seed': int(args.seed),
            'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
            'n_unused': len(unused),
            'test_d_max': int(d[test].max()), 'train_d_min': int(d[train].min()),
            'start_ij': list(meta['start_ij']), 'single_maze': meta['single_maze'],
        },
        'train': train, 'val': val, 'test': test, 'unused': unused,
    }
    for name in ('train', 'val', 'test'):
        manifest[f'{name}_frames'] = int(lengths[manifest[name]].sum())
    manifest['checksum'] = split_checksum(train, val, test)

    out = SPLITS / args.out
    out.write_text(json.dumps(manifest, indent=2))
    for name in ('test', 'train', 'val', 'unused'):
        idx = manifest[name]
        print(f"{name:>6}: {len(idx):>3} episodes  goal dist "
              f"{d[idx].min():>3}-{d[idx].max():>3} (mean {d[idx].mean():5.1f})  "
              f"{lengths[idx].sum():>6} frames")
    print(f"\nwrote {out}  checksum {manifest['checksum'][:12]}")


if __name__ == '__main__':
    main()
