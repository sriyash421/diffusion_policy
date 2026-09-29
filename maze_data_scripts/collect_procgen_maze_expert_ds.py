#!/usr/bin/env python3
"""
Collect expert demonstrations for the procgen grid maze, in the format
`config/task/procgen_maze.yaml` loads.

Layouts are split: `--eval-mazes` held-out mazes are generated first, and the train split
draws fresh mazes that are rejected if their layout matches any held-out one. Both splits
are reproducible from their seeds.

The default output is a zarr laid out the way `Sim2RealImageMultiDataset` reads it:

    <out>/<name>.zarr/
      data/obs/{image,agent_pos,goal_pos}    image is (N, G, G, 3) uint8, HWC (G = GRID_SIZE)
      data/actions                           (N, 2) float32
      data/expert_mask                       (N,) float32
      meta/episode_ends                      (E,)
      meta/{maze_ids,maze_index,split,...}   provenance; ignored by the loader

Usage:
    python maze_data_scripts/collect_procgen_maze_expert_ds.py --split train --num-episodes 1000
    python maze_data_scripts/collect_procgen_maze_expert_ds.py --split eval --num-episodes 200

Or one fixed layout and start, with goals spread across the maze:

    python maze_data_scripts/collect_procgen_maze_expert_ds.py --single-maze 1 \
        --num-episodes 200 --min-goal-dist 8 --max-steps 800
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm

from procgen_maze_env import (ProcgenMazeEnv, bfs_distances, free_cells,
                              make_maze_split, maze_id)
from procgen_maze_expert import ProcgenMazeExpert


def build_splits(n_eval_mazes, n_train_mazes, eval_seed, train_seed):
    """Held-out mazes first, then train mazes that are guaranteed not to be among them."""
    eval_mazes = make_maze_split(n_eval_mazes, seed=eval_seed)
    eval_ids = {maze_id(m) for m in eval_mazes}
    train_mazes = make_maze_split(n_train_mazes, seed=train_seed, exclude_ids=eval_ids)
    assert not (eval_ids & {maze_id(m) for m in train_mazes}), 'split leak'
    return train_mazes, eval_mazes


def central_cell(maze_map):
    """The free cell nearest the centre -- goals then spread in every direction."""
    mid = (maze_map.shape[0] - 1) / 2
    return min(free_cells(maze_map), key=lambda c: abs(c[0] - mid) + abs(c[1] - mid))


def spread_goals(maze_map, start_ij, n_goals, rng, min_goal_dist):
    """`n_goals` DISTINCT goals, each at least `min_goal_dist` steps along the maze.

    Distinct, not i.i.d.: with the dither off a repeated (start, goal) pair replays the
    same trajectory exactly, so duplicates would add episodes without adding data.

    Returns the WHOLE shuffled eligible list, not just `n_goals` of it, so an episode the
    expert fails is replaced by a fresh goal rather than shrinking the dataset. Only when
    eligible cells are fewer than `n_goals` is the list cycled, and that is reported.
    """
    dist = bfs_distances(maze_map, start_ij)
    eligible = [c for c, d in dist.items() if d >= min_goal_dist]
    if not eligible:
        raise ValueError(f"no free cell is >= {min_goal_dist} steps from {start_ij}")
    order = [eligible[i] for i in rng.permutation(len(eligible))]
    if n_goals > len(order):
        print(f"WARNING: {n_goals} episodes but only {len(order)} eligible goals -- "
              f"{n_goals - len(order)} trajectories will be exact repeats")
        order = order * (n_goals // len(order) + 1)
    return order, len(eligible)


def collect_expert_dataset(
    split: str,
    num_episodes: int,
    n_eval_mazes: int = 50,
    n_train_mazes: int = 500,
    eval_seed: int = 12345,
    train_seed: int = 42,
    env_seed: int = 0,
    max_steps_per_episode: int = 300,
    min_episode_steps: int = 16,
    waypoint_threshold: float = 0.3,
    step_scale: float = 0.25,
    success_only: bool = True,
    output_dir: str = None,
    fmt: str = 'zarr',
    single_maze: int = None,
    maze_pool_size: int = 6,
    maze_pool_seed: int = 0,
    start_ij=None,
    min_goal_dist: int = 8,
):
    goal_schedule = None
    if single_maze is not None:
        # One layout, one start, many goals: the maze the visualizer shows as
        # `maze <single_maze>` for the same pool size and seed.
        mazes = [make_maze_split(maze_pool_size, seed=maze_pool_seed)[single_maze]]
        eval_mazes = []
        if start_ij is None:
            start_ij = central_cell(mazes[0])
        start_ij = tuple(int(x) for x in start_ij)
        goal_schedule, n_eligible = spread_goals(
            mazes[0], start_ij, num_episodes, np.random.default_rng(env_seed),
            min_goal_dist)
        print(f"single maze {single_maze} of pool(n={maze_pool_size}, seed={maze_pool_seed})"
              f"  layout {maze_id(mazes[0])[:16]}")
        print(f"start {start_ij}   {n_eligible} eligible goals "
              f"(>= {min_goal_dist} steps away) for {num_episodes} episodes")
    else:
        train_mazes, eval_mazes = build_splits(
            n_eval_mazes, n_train_mazes, eval_seed, train_seed)
        mazes = train_mazes if split == 'train' else eval_mazes
        print(f"split={split}: {len(mazes)} layouts "
              f"(train pool {len(train_mazes)}, held-out {len(eval_mazes)})")

    env = ProcgenMazeEnv(mazes, step_scale=step_scale, seed=env_seed)

    images, agent_pos, goal_pos, actions = [], [], [], []
    episode_ends, episode_maze_index = [], []
    episodes_collected = episodes_attempted = episodes_rejected_short = 0

    pbar = tqdm(total=num_episodes, desc=f"Collecting procgen_maze ({split})")
    while episodes_collected < num_episodes:
        if goal_schedule is None:
            obs = env.reset()
        else:
            # Advance the schedule per ATTEMPT. Indexing by episodes_collected would
            # retry a goal the expert cannot solve forever.
            if episodes_attempted >= len(goal_schedule):
                print(f"\nWARNING: goal schedule exhausted at {episodes_collected} "
                      f"episodes of {num_episodes} requested")
                break
            obs = env.reset(maze_index=0, start_ij=start_ij,
                            goal_ij=goal_schedule[episodes_attempted])
        expert = ProcgenMazeExpert(env, waypoint_threshold=waypoint_threshold)

        ep = {'image': [], 'agent_pos': [], 'goal_pos': [], 'action': []}
        success = False
        for _ in range(max_steps_per_episode):
            action = expert.compute_action(obs)
            ep['image'].append(obs['image'])
            ep['agent_pos'].append(obs['agent_pos'])
            ep['goal_pos'].append(obs['goal_pos'])
            ep['action'].append(action)

            obs, _reward, success = env.step(action)
            if success:
                break

        episodes_attempted += 1
        if success_only and not success:
            continue
        # A start adjacent to the goal yields an episode shorter than one action chunk;
        # it teaches almost nothing and drags the mean episode length down.
        if len(ep['action']) < min_episode_steps:
            episodes_rejected_short += 1
            continue

        images.extend(ep['image'])
        agent_pos.extend(ep['agent_pos'])
        goal_pos.extend(ep['goal_pos'])
        actions.extend(ep['action'])
        episode_ends.append(len(actions))
        episode_maze_index.append(env.maze_index)

        episodes_collected += 1
        pbar.update(1)
        pbar.set_postfix({
            'success': f"{episodes_collected}/{episodes_attempted}",
            'short_rej': episodes_rejected_short,
            'len': len(ep['action']),
        })
    pbar.close()

    data = {
        'image': np.asarray(images, dtype=np.uint8),
        'agent_pos': np.asarray(agent_pos, dtype=np.float32),
        'goal_pos': np.asarray(goal_pos, dtype=np.float32),
        'actions': np.asarray(actions, dtype=np.float32),
        # Every transition here is expert; the field exists because the loader requires it.
        'expert_mask': np.ones((len(actions),), dtype=np.float32),
        'episode_ends': np.asarray(episode_ends, dtype=np.int64),
        'maze_index': np.asarray(episode_maze_index, dtype=np.int64),
    }
    lengths = np.diff(np.concatenate([[0], data['episode_ends']]))

    print(f"\n{'='*60}\nDataset Collection Complete\n{'='*60}")
    print(f"Split: {split}   layouts used: {len(set(episode_maze_index))}/{len(mazes)}")
    print(f"Episodes: {len(episode_ends)}   steps: {len(actions)}")
    print(f"Episode length: mean {lengths.mean():.1f}  min {lengths.min()}  max {lengths.max()}")
    print(f"Attempted: {episodes_attempted}  rejected (too short): {episodes_rejected_short}")
    for key in ('image', 'agent_pos', 'goal_pos', 'actions'):
        print(f"  {key}: {data[key].shape} {data[key].dtype}")

    if output_dir is None:
        output_dir = f"data/procgen_maze_expert{'' if split == 'train' else '_eval'}"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        'split': split,
        'maze_ids': [maze_id(m) for m in mazes],
        'eval_maze_ids': [maze_id(m) for m in eval_mazes],
        'eval_seed': eval_seed, 'train_seed': train_seed, 'env_seed': env_seed,
        'step_scale': step_scale, 'waypoint_threshold': waypoint_threshold,
        'min_episode_steps': min_episode_steps,
        'grid_size': int(mazes[0].shape[0]),
        'single_maze': single_maze,
        'maze_pool_size': maze_pool_size, 'maze_pool_seed': maze_pool_seed,
        'start_ij': list(start_ij) if start_ij is not None else None,
        'min_goal_dist': min_goal_dist if single_maze is not None else None,
    }

    if fmt == 'pickle':
        path = out / f"procgen_maze_{split}.pkl"
        with open(path, 'wb') as f:
            pickle.dump({**data, 'meta': meta}, f)
    else:
        path = out / f"procgen_maze_{split}.zarr"
        root = zarr.open(str(path), mode='w')
        obs_group = root.create_group('data').create_group('obs')
        obs_group.create_dataset('image', data=data['image'])
        obs_group.create_dataset('agent_pos', data=data['agent_pos'])
        obs_group.create_dataset('goal_pos', data=data['goal_pos'])
        root['data'].create_dataset('actions', data=data['actions'])
        root['data'].create_dataset('expert_mask', data=data['expert_mask'])
        meta_group = root.create_group('meta')
        meta_group.create_dataset('episode_ends', data=data['episode_ends'])
        meta_group.create_dataset('maze_index', data=data['maze_index'])
        meta_group.attrs.update(meta)
    (out / f"procgen_maze_{split}_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDataset saved to: {path}")
    return data


def main():
    p = argparse.ArgumentParser(
        description="Collect expert demonstrations for the procgen grid maze")
    p.add_argument("--split", choices=['train', 'eval'], default='train',
                   help="Which layout split to collect from (default: train)")
    p.add_argument("--num-episodes", type=int, default=1000,
                   help="Number of episodes to collect (default: 1000)")
    p.add_argument("--eval-mazes", type=int, default=50,
                   help="Size of the held-out layout set (default: 50)")
    p.add_argument("--train-mazes", type=int, default=500,
                   help="Size of the train layout pool (default: 500)")
    p.add_argument("--eval-seed", type=int, default=12345,
                   help="Seed defining the held-out layouts (default: 12345)")
    p.add_argument("--train-seed", type=int, default=42,
                   help="Seed defining the train layout pool (default: 42)")
    p.add_argument("--env-seed", type=int, default=0,
                   help="Seed for per-episode layout/start/goal sampling (default: 0)")
    p.add_argument("--max-steps", type=int, default=300,
                   help="Maximum steps per episode (default: 300)")
    p.add_argument("--min-episode-steps", type=int, default=16,
                   help="Drop episodes shorter than this, the config horizon (default: 16)")
    p.add_argument("--waypoint-threshold", type=float, default=0.3,
                   help="Waypoint following threshold (default: 0.3)")
    p.add_argument("--step-scale", type=float, default=0.25,
                   help="Cells moved per unit action (default: 0.25)")
    p.add_argument("--output", type=str, default=None,
                   help="Output directory (default: data/procgen_maze_expert[_eval])")
    p.add_argument("--format", choices=['zarr', 'pickle'], default='zarr',
                   help="zarr is what the task config loads (default: zarr)")
    p.add_argument("--single-maze", type=int, default=None, metavar="INDEX",
                   help="Collect from ONE layout: index into the visualizer's pool. "
                        "Start is fixed and goals are spread across the maze.")
    p.add_argument("--maze-pool-size", type=int, default=6,
                   help="Pool --single-maze indexes into, matching the viz (default: 6)")
    p.add_argument("--maze-pool-seed", type=int, default=0,
                   help="Seed of that pool, matching the viz (default: 0)")
    p.add_argument("--start", type=int, nargs=2, default=None, metavar=("I", "J"),
                   help="Fixed start cell for --single-maze (default: most central free cell)")
    p.add_argument("--min-goal-dist", type=int, default=8,
                   help="Goals must be >= this many steps along the maze (default: 8)")
    p.add_argument("--include-failures", action="store_true",
                   help="Include failed episodes (default: only successful)")
    args = p.parse_args()

    collect_expert_dataset(
        split=args.split,
        num_episodes=args.num_episodes,
        n_eval_mazes=args.eval_mazes,
        n_train_mazes=args.train_mazes,
        eval_seed=args.eval_seed,
        train_seed=args.train_seed,
        env_seed=args.env_seed,
        max_steps_per_episode=args.max_steps,
        min_episode_steps=args.min_episode_steps,
        waypoint_threshold=args.waypoint_threshold,
        step_scale=args.step_scale,
        success_only=not args.include_failures,
        output_dir=args.output,
        fmt=args.format,
        single_maze=args.single_maze,
        maze_pool_size=args.maze_pool_size,
        maze_pool_seed=args.maze_pool_seed,
        start_ij=args.start,
        min_goal_dist=args.min_goal_dist,
    )


if __name__ == "__main__":
    main()
