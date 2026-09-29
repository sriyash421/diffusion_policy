#!/usr/bin/env python3
"""
Compare the train and held-out layout splits of the procgen maze.

Answers two DIFFERENT questions that are easy to conflate:

1. **Overlap** -- does any training layout equal a held-out one? This is the question the
   split exists to answer, and it is decided exactly, by layout hash.
2. **Shift** -- are the two sets drawn from different distributions? With the default
   settings the answer is *no, by construction*: both splits come from the same generator
   and the same cell-pair sampler, and train merely rejects the 50 held-out layouts. An
   i.i.d. sample minus a finite exclusion list is still i.i.d., so every statistic below
   should agree within sampling noise. Disagreement here means a bug, not generalisation
   difficulty.

So "seeing the shift" is really "confirming there isn't one, and that nothing leaked".
To create a REAL train/test shift you have to change generation, not just the seed --
e.g. braid the eval mazes (remove dead ends) or bias eval start/goal pairs to long
solutions -- and then these same statistics are what shows it.

Usage:
    python maze_data_scripts/compare_maze_splits.py --plot splits.png
"""
import argparse
import json

import numpy as np

from procgen_maze_env import ProcgenMazeEnv, free_cells, make_maze_split, maze_id
from procgen_maze_expert import solve_length


def dead_ends(maze_map):
    """Free cells with exactly one free neighbour."""
    count = 0
    for i, j in free_cells(maze_map):
        nb = sum(
            maze_map[i + di, j + dj] == 0
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= i + di < maze_map.shape[0] and 0 <= j + dj < maze_map.shape[1]
        )
        count += (nb == 1)
    return count


def split_stats(mazes, n_probe_episodes, seed, max_steps=300):
    """Layout statistics, plus expert solution lengths over sampled start/goal pairs."""
    env = ProcgenMazeEnv(mazes, seed=seed)
    lengths, failures = [], 0
    for _ in range(n_probe_episodes):
        env.reset()
        n = solve_length(env, max_steps=max_steps)
        if n is None:
            failures += 1
        else:
            lengths.append(n)
    return {
        'n_mazes': len(mazes),
        'wall_fraction': np.array([(m == 1).mean() for m in mazes]),
        'free_cells': np.array([len(free_cells(m)) for m in mazes]),
        'dead_ends': np.array([dead_ends(m) for m in mazes]),
        'solution_steps': np.array(lengths),
        'expert_failures': failures,
    }


def _fmt(name, a):
    return f"  {name:<16} mean {a.mean():7.2f}   sd {a.std():6.2f}   min {a.min():6.1f}   max {a.max():6.1f}"


def main():
    p = argparse.ArgumentParser(description="Compare procgen maze train/eval splits")
    p.add_argument("--eval-mazes", type=int, default=50)
    p.add_argument("--train-mazes", type=int, default=500)
    p.add_argument("--eval-seed", type=int, default=12345)
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--probe-episodes", type=int, default=400,
                   help="Start/goal pairs sampled per split for solution-length stats")
    p.add_argument("--plot", type=str, default=None, help="Write an ECDF comparison here")
    p.add_argument("--json", type=str, default=None, help="Write the summary here")
    args = p.parse_args()

    eval_mazes = make_maze_split(args.eval_mazes, seed=args.eval_seed)
    eval_ids = {maze_id(m) for m in eval_mazes}
    train_mazes = make_maze_split(args.train_mazes, seed=args.train_seed,
                                  exclude_ids=eval_ids)
    train_ids = {maze_id(m) for m in train_mazes}

    overlap = train_ids & eval_ids
    print("=" * 64)
    print("1. OVERLAP (exact, by layout hash)")
    print("=" * 64)
    print(f"  train layouts      {len(train_ids)}")
    print(f"  held-out layouts   {len(eval_ids)}")
    print(f"  shared layouts     {len(overlap)}   <- must be 0")
    assert not overlap, f"{len(overlap)} held-out layouts leaked into train"

    stats = {
        'train': split_stats(train_mazes, args.probe_episodes, seed=1),
        'eval': split_stats(eval_mazes, args.probe_episodes, seed=2),
    }

    print("\n" + "=" * 64)
    print("2. SHIFT (expect agreement within noise -- same generator both sides)")
    print("=" * 64)
    for split, s in stats.items():
        print(f"{split} ({s['n_mazes']} layouts, {len(s['solution_steps'])} solved probes, "
              f"{s['expert_failures']} expert failures)")
        for key in ('wall_fraction', 'free_cells', 'dead_ends', 'solution_steps'):
            print(_fmt(key, s[key]))

    print("\n  difference in mean solution length: "
          f"{stats['eval']['solution_steps'].mean() - stats['train']['solution_steps'].mean():+.2f} steps")

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # Colour is paired with line style and a direct label, never carried by hue alone.
        styles = {'train': ('-', '#0072B2'), 'eval': ('--', '#D55E00')}
        for ax, key, title in (
            (axes[0], 'solution_steps', 'Expert solution length (steps)'),
            (axes[1], 'dead_ends', 'Dead-end cells per layout'),
        ):
            for split, s in stats.items():
                x = np.sort(s[key])
                y = np.arange(1, len(x) + 1) / len(x)
                ls, c = styles[split]
                ax.plot(x, y, ls, color=c, label=split, linewidth=2)
                ax.annotate(split, xy=(x[len(x) // 2], y[len(y) // 2]),
                            xytext=(4, -10 if split == 'eval' else 6),
                            textcoords='offset points', color=c, fontsize=9)
            ax.set_xlabel(title)
            ax.set_ylabel('ECDF')
            ax.grid(alpha=0.3)
        fig.suptitle('Train vs held-out layouts: overlapping curves mean no shift')
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\n  plot written to {args.plot}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({
                'overlap': len(overlap),
                **{split: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in s.items()} for split, s in stats.items()},
            }, f, indent=2)


if __name__ == "__main__":
    main()
