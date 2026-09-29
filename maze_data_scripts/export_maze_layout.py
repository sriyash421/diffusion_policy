#!/usr/bin/env python3
"""Write a dataset's maze layout to .npy, for MazeVerifier's `maze_path`.

The verifier is pointed at this file by config and nothing cross-checks it against the
zarr, so REGENERATE IT WHENEVER THE DATASET IS REGENERATED. It is written beside the zarr
for that reason -- the two belong together.

    python maze_data_scripts/export_maze_layout.py --zarr data/.../procgen_maze_train.zarr
"""
import argparse
import pathlib

import numpy as np
import zarr

from procgen_maze_env import make_maze_split, maze_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr', default='data/procgen_maze_expert_maze1/procgen_maze_train.zarr')
    ap.add_argument('--out', default=None, help='default: maze.npy beside the zarr')
    args = ap.parse_args()

    root = zarr.open(args.zarr, mode='r')
    meta = dict(root['meta'].attrs)
    maze = make_maze_split(meta['maze_pool_size'], seed=meta['maze_pool_seed'])[
        meta['single_maze']]
    assert maze_id(maze) == meta['maze_ids'][0], \
        'regenerated layout does not match the one recorded in the zarr'

    out = pathlib.Path(args.out or (pathlib.Path(args.zarr).parent / 'maze.npy'))
    np.save(out, maze)
    print(f"wrote {out}  {maze.shape}  layout {maze_id(maze)[:16]}")


if __name__ == '__main__':
    main()
