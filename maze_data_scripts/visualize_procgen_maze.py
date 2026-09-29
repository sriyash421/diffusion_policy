#!/usr/bin/env python3
"""
Generate procgen mazes and look at them.

Uses the same generator the dataset does (`make_maze_split`), so a picture here is a
picture of a real training layout -- nothing is re-implemented for display. Each panel is
`env.render()` upscaled with nearest-neighbour, optionally with the expert's trajectory
drawn over it.

Colour carries no information on its own: the agent is a circle, the goal a square, and
the path a dashed line, so the figure survives being printed in grey.

Usage:
    python maze_data_scripts/visualize_procgen_maze.py --n 6 --seed 0 --out mazes.png
    python maze_data_scripts/visualize_procgen_maze.py --no-path      # layouts only
    python maze_data_scripts/visualize_procgen_maze.py --dither       # jittered waypoints
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from procgen_maze_env import (COLOR_AGENT, COLOR_GOAL, ProcgenMazeEnv, make_maze_split)
from procgen_maze_expert import ProcgenMazeExpert

AGENT = np.array(COLOR_AGENT) / 255.0
GOAL = np.array(COLOR_GOAL) / 255.0


def rollout(env, max_steps=300, dither=False):
    """Expert trajectory from the current reset state as (T+1, 2) of (row, col)."""
    expert = ProcgenMazeExpert(env, dither=dither)
    obs = env.get_obs()
    path = [env.agent_xy.copy()]
    solved = False
    for _ in range(max_steps):
        obs, _, success = env.step(expert.compute_action(obs))
        path.append(env.agent_xy.copy())
        if success:
            solved = True
            break
    return np.asarray(path), solved


def draw(ax, maze_map, start, goal, path=None, title="", endpoints=True):
    img = np.ones((*maze_map.shape, 3))
    img[maze_map == 1] = 0.0
    # Cell centres sit on integer coords, so extent shifts by half a cell.
    ax.imshow(img, interpolation="nearest",
              extent=(-0.5, maze_map.shape[1] - 0.5, maze_map.shape[0] - 0.5, -0.5))

    if path is not None:
        ax.plot(path[:, 1], path[:, 0], linestyle="--", linewidth=1.8,
                color=AGENT, label="expert path")
    if endpoints:
        ax.plot(goal[1], goal[0], marker="s", markersize=11, color=GOAL,
                markeredgecolor="black", markeredgewidth=0.6, linestyle="none",
                label="goal")
        ax.plot(start[1], start[0], marker="o", markersize=9, color=AGENT,
                markeredgecolor="black", markeredgewidth=0.6, linestyle="none",
                label="start")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_dataset(path, out):
    """What a collected dataset actually contains: goal coverage, and the demos."""
    import zarr
    root = zarr.open(str(path), mode='r')
    ends = np.asarray(root['meta/episode_ends'])
    agent = np.asarray(root['data/obs/agent_pos'])
    goals = np.asarray(root['data/obs/goal_pos'])
    meta = dict(root['meta'].attrs)
    maze = make_maze_split(meta['maze_pool_size'], seed=meta['maze_pool_seed'])[
        meta['single_maze']]
    start = np.asarray(meta['start_ij'], dtype=float)
    starts = np.concatenate([[0], ends[:-1]])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    ep_goals = goals[ends - 1]
    draw(axes[0], maze, start, ep_goals[0], endpoints=False,
         title=f"{len(ends)} goals, {len(np.unique(ep_goals, axis=0))} distinct")
    axes[0].plot(ep_goals[:, 1], ep_goals[:, 0], marker="s", markersize=4,
                 color=GOAL, linestyle="none", markeredgewidth=0)
    axes[0].plot(start[1], start[0], marker="o", markersize=10, color=AGENT,
                 markeredgecolor="black", markeredgewidth=0.8, linestyle="none")

    draw(axes[1], maze, start, ep_goals[0], endpoints=False,
         title="every demonstration")
    for a, b in zip(starts, ends):
        axes[1].plot(agent[a:b, 1], agent[a:b, 0], color=AGENT, linewidth=0.6, alpha=0.25)
    axes[1].plot(start[1], start[0], marker="o", markersize=10, color=AGENT,
                 markeredgecolor="black", markeredgewidth=0.8, linestyle="none")
    fig.suptitle(f"maze {meta['single_maze']}  start {tuple(meta['start_ij'])}  "
                 f"goals >= {meta['min_goal_dist']} steps away", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=6, help="mazes to draw")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="procgen_mazes.png")
    p.add_argument("--no-path", action="store_true", help="layouts only, skip the expert")
    p.add_argument("--dataset", default=None,
                   help="Draw a collected .zarr instead: goal coverage and every demo")
    p.add_argument("--dither", action="store_true",
                   help="restore the expert's 0.2-cell waypoint jitter (off by default)")
    args = p.parse_args()

    if args.dataset is not None:
        draw_dataset(args.dataset, args.out)
        return

    mazes = make_maze_split(args.n, seed=args.seed)
    env = ProcgenMazeEnv(mazes, seed=args.seed)

    cols = min(args.n, 3)
    rows = int(np.ceil(args.n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)

    for k in range(rows * cols):
        ax = axes[k // cols][k % cols]
        if k >= args.n:
            ax.axis("off")
            continue
        env.reset(maze_index=k)
        start, goal = env.agent_xy.copy(), env.cur_goal_xy.copy()
        path, title = None, f"maze {k}"
        if not args.no_path:
            path, solved = rollout(env, dither=args.dither)
            title += f" -- {len(path) - 1} steps" if solved else " -- expert FAILED"
        draw(ax, env.maze_map, start, goal, path, title)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    # imshow keeps a square aspect, so the axes box is taller than the picture
    # and row-2 titles land on row-1 unless the rows are pushed apart.
    fig.subplots_adjust(hspace=0.12)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
