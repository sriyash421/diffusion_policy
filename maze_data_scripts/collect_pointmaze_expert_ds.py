#!/usr/bin/env python3
"""
Collect expert demonstrations for PointMaze environments in OGBench format.

This script uses the WaypointController expert to collect high-quality demonstrations
and saves them in a format compatible with OGBench datasets.

Usage:
    python maze_data_scripts/collect_pointmaze_expert_ds.py --env pointmaze-large-navigate-singletask-v0 --num-episodes 1000
"""

import argparse
import pickle
import random
import time
import numpy as np
import ogbench
from pathlib import Path
from tqdm import tqdm
from pointmaze_expert import WaypointController
import cv2

def collect_expert_dataset(
    env_name: str,
    num_episodes: int,
    max_steps_per_episode: int = 200,
    waypoint_threshold: float = 0.3,
    success_only: bool = True,
    output_path: str = None,
    viz: bool = False
):
    """
    Collect expert demonstrations using WaypointController.

    Args:
        env_name: Name of the PointMaze environment (e.g., 'pointmaze-large-navigate-singletask-v0')
        num_episodes: Number of episodes to collect
        max_steps_per_episode: Maximum steps per episode
        waypoint_threshold: Distance threshold for waypoint following
        success_only: Only save successful episodes
        output_path: Path to save the dataset (default: {env_name}-expert-dataset.pkl)
    """
    # Create environment and expert
    env = ogbench.make_env_and_datasets(env_name, add_info=True, ob_type='states', env_only=True)
    env.reset()

    # Initialize dataset dict with empty arrays
    expert_dataset = {
        "observations": [],
        "actions": [],
        "terminals": [],
        "rewards": [],
    }

    episodes_collected = 0
    episodes_attempted = 0

    pbar = tqdm(total=num_episodes, desc=f"Collecting expert data for {env_name}")

    while episodes_collected < num_episodes:
        # reset_symbols = {0, "r", "c"}
        # maze_map = env.unwrapped.maze_map
        # valid_cells = []
        # for i in range(maze_map.shape[0]):
        #     for j in range(maze_map.shape[1]):
        #         if maze_map[i, j] in reset_symbols:
        #             valid_cells.append((i, j))
        # init_ij = random.choice(valid_cells)
        # env.unwrapped.cur_task_info['init_ij'] = np.array(init_ij)
        # options = {
        #      'init_ij': init_ij,
        # }
        options = None
        obs, info = env.reset(options=options)

        expert = WaypointController(env.unwrapped, waypoint_threshold=waypoint_threshold)
        episode_data = {
            "observations": [],
            "next_observations": [],
            "actions": [],
            "terminals": [],
            "rewards": [],
        }

        success = False
        for step in range(max_steps_per_episode):
            # Compute expert action
            action = expert.compute_action({
                "achieved_goal": obs,
                "desired_goal": np.array(env.unwrapped.cur_goal_xy)
            })

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Store transition
            episode_data["observations"].append(obs)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["next_observations"].append(next_obs)
            # Terminal is 1 only at the last step if successful
            episode_data["terminals"].append(0)

            obs = next_obs
            if viz:
                # display the frame in a window
                cv2.imshow("Expert Demonstration", env.render())
                cv2.waitKey(1)

                time.sleep(0.01)
            # Check if goal reached (reward >= 0 in pointmaze means success)
            if reward >= 0:
                success = True
                episode_data["terminals"][-1] = 1  # Mark last step as terminal
                break

        episodes_attempted += 1


        # Only add episode if successful or if we're collecting all episodes
        if success or not success_only:
            # If not successful, still mark the last step as terminal
            if not success:
                episode_data["terminals"][-1] = 1

            # Append episode to dataset
            for key in expert_dataset.keys():
                expert_dataset[key].extend(episode_data[key])

            episodes_collected += 1
            pbar.update(1)
            pbar.set_postfix({
                "success_rate": f"{episodes_collected}/{episodes_attempted}",
                "avg_len": f"{len(episode_data['observations']):.1f}"
            })

    pbar.close()

    # Convert lists to numpy arrays
    for key in expert_dataset.keys():
        expert_dataset[key] = np.array(expert_dataset[key])

    # Ensure terminals are integers
    expert_dataset["terminals"] = expert_dataset["terminals"].astype(np.int32)

    # Print statistics
    total_steps = len(expert_dataset["observations"])
    num_episodes_in_data = int(expert_dataset["terminals"].sum())
    avg_episode_length = total_steps / num_episodes_in_data if num_episodes_in_data > 0 else 0

    print(f"\n{'='*60}")
    print(f"Dataset Collection Complete")
    print(f"{'='*60}")
    print(f"Environment: {env_name}")
    print(f"Episodes collected: {num_episodes_in_data}")
    print(f"Total steps: {total_steps}")
    print(f"Average episode length: {avg_episode_length:.1f}")
    print(f"Success rate: {episodes_collected}/{episodes_attempted} ({100*episodes_collected/episodes_attempted:.1f}%)")
    print(f"\nDataset shapes:")
    for key, value in expert_dataset.items():
        print(f"  {key}: {value.shape}")

    # Save dataset
    if output_path is None:
        output_path = f"{env_name}-expert-dataset.pkl"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(expert_dataset, f)

    print(f"\nDataset saved to: {output_path}")

    return expert_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Collect expert demonstrations for PointMaze environments"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pointmaze-medium-navigate-singletask-v0",
        help="PointMaze environment name (default: pointmaze-medium-navigate-singletask-v0)"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1000,
        help="Number of episodes to collect (default: 1000)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum steps per episode (default: 300)"
    )
    parser.add_argument(
        "--waypoint-threshold",
        type=float,
        default=0.3,
        help="Waypoint following threshold (default: 0.3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for dataset pickle file (default: {env}-expert-dataset.pkl)"
    )


    parser.add_argument(
        "--include-failures",
        action="store_true",
        help="Include failed episodes in dataset (default: only successful episodes)"
    )

    parser.add_argument(
        "--viz",
        action="store_true",
        help="Visualize the expert demonstrations (default: False)"
    )


    args = parser.parse_args()

    collect_expert_dataset(
        env_name=args.env,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        waypoint_threshold=args.waypoint_threshold,
        success_only=not args.include_failures,
        output_path=args.output,
        viz=args.viz
    )


if __name__ == "__main__":
    main()
