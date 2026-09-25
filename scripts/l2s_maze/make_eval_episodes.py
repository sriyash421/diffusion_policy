"""Eval episodes for one split: data/l2s_maze/<split>/eval_episodes.json.

Goals come from the split's TEST pool, starts are uniform over free cells. Pairs the expert
fails or solves in < --min-solve-len steps are dropped. max_steps = 2 * expert solve length.

    python scripts/l2s_maze/make_eval_episodes.py --split as_is
"""
import argparse
import json
import pathlib

import numpy as np

from expert_utils import CachedExpert
from maze_layout import (DEFAULT_LAYOUT_DIR, REPO, SPLITS, bfs_to, free_cells, goal_pool,
                         load_layout, sample_pair)


def make_episodes(split, layout_dir, n, seed, min_solve_len, max_expert_steps, out):
    maze, layout = load_layout(layout_dir)
    free = free_cells(maze)
    goals = goal_pool(maze, split, 'test', layout['r_inner'])
    rng = np.random.default_rng(seed)
    expert = CachedExpert(maze)

    episodes, n_attempted, n_dropped = [], 0, 0
    while len(episodes) < n:
        start, goal = sample_pair(rng, free, goals)
        n_attempted += 1
        success, solve_len, _ = expert.rollout(start, goal, max_expert_steps)
        if not success or solve_len < min_solve_len:
            n_dropped += 1
            continue
        episodes.append({
            'start': list(start), 'goal': list(goal), 'solve_len': solve_len,
            'max_steps': 2 * solve_len, 'd0': bfs_to(maze, goal)[start],
        })

    manifest = {
        'split': split, 'seed': seed, 'layout_sha': layout['sha'],
        'min_solve_len': min_solve_len, 'n_attempted': n_attempted, 'n_dropped': n_dropped,
        'n_goal_pool': len(goals), 'episodes': episodes,
    }
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1))
    sl = np.array([e['solve_len'] for e in episodes])
    print(f'{split}: {len(episodes)} episodes ({n_dropped}/{n_attempted} dropped), '
          f'solve_len {sl.min()}-{sl.max()} mean {sl.mean():.1f}')
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=SPLITS, required=True)
    ap.add_argument('--layout-dir', default=str(DEFAULT_LAYOUT_DIR))
    ap.add_argument('--n', type=int, default=100)
    # distinct from the train collection seed (0), so as_is eval pairs are a fresh draw
    ap.add_argument('--seed', type=int, default=20000)
    ap.add_argument('--min-solve-len', type=int, default=16)
    ap.add_argument('--max-expert-steps', type=int, default=1000)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = args.out or REPO / f'data/l2s_maze/{args.split}/eval_episodes.json'
    make_episodes(args.split, args.layout_dir, args.n, args.seed, args.min_solve_len,
                  args.max_expert_steps, out)


if __name__ == '__main__':
    main()
