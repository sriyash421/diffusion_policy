"""Write the fixed L2S maze layout: data/l2s_maze/layout/{maze.npy, layout.json}.

    python scripts/l2s_maze/build_layout.py --seed 0
"""
import argparse
import json
import pathlib

import numpy as np

from maze_layout import (CANVAS_SIZE, CENTER, DEFAULT_LAYOUT_DIR, MAZE_SIZE, PAD, SPLITS,
                         build_layout, free_cells, goal_pool, inner_radius, layout_sha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--inner-frac', type=float, default=0.25)
    ap.add_argument('--out', default=str(DEFAULT_LAYOUT_DIR))
    args = ap.parse_args()

    maze = build_layout(args.seed)
    r = inner_radius(maze, args.inner_frac)
    n_free = len(free_cells(maze))
    pools = {s: {role: len(goal_pool(maze, s, role, r)) for role in ('train', 'test')}
             for s in SPLITS}
    meta = {
        'seed': args.seed, 'maze_size': MAZE_SIZE, 'canvas_size': CANVAS_SIZE, 'pad': PAD,
        'center': list(CENTER), 'n_free': n_free,
        'inner_frac_target': args.inner_frac, 'r_inner': r,
        'inner_frac': pools['inner_outer']['test'] / n_free,
        'goal_pool_sizes': pools,
        'step_scale': 0.25, 'goal_tol': 0.3,
        'sha': layout_sha(maze),
    }
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'maze.npy', maze)
    (out / 'layout.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
