#!/usr/bin/env python3
"""
Create a 2D goal-reaching zarr dataset with mixed-quality demonstrations.

Output format:
- data/actions        : [T, 2]
- data/obs/x          : [T, 1]
- data/obs/y          : [T, 1]
- data/rewards        : [T]
- data/dones          : [T]
- meta/episode_ends   : [num_episodes]

Dataset composition:
- 50% successful noisy goal-directed trajectories
- 50% random failed trajectories
"""

import os
import shutil
from dataclasses import dataclass
from typing import List

import click
import numpy as np
import zarr
from tqdm import tqdm

@dataclass
class Grid2DGoalEnv:
    action_scale: float = 0.01
    goal_radius: float = 0.03
    max_steps: int = 200
    grid_bins: int = 100

    def __post_init__(self):
        self.goal = np.array([1.0, 1.0], dtype=np.float32)
        self.state = np.zeros(2, dtype=np.float32)
        self.step_count = 0

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        # Sample start from a discrete grid in [0, 1]^2.
        self.state = (
            rng.integers(0, self.grid_bins + 1, size=2).astype(np.float32)
            / float(self.grid_bins)
        )
        self.step_count = 0
        return self.state.copy()

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.state = np.clip(self.state + action * self.action_scale, 0.0, 1.0)
        self.step_count += 1

        dist_to_goal = float(np.linalg.norm(self.goal - self.state))
        success = dist_to_goal <= self.goal_radius
        timeout = self.step_count >= self.max_steps
        done = bool(success or timeout)
        reward = 1.0 if success else 0.0
        info = {
            "is_success": success,
            "distance_to_goal": dist_to_goal,
        }
        return self.state.copy(), reward, done, info


def _guided_action(
    env: Grid2DGoalEnv,
    rng: np.random.Generator,
    noise_std: float,
    random_action_prob: float,
) -> np.ndarray:
    goal_delta = env.goal - env.state
    dist = np.linalg.norm(goal_delta)
    if dist < 1e-8:
        base_action = np.zeros(2, dtype=np.float32)
    else:
        base_action = (goal_delta / dist).astype(np.float32)

    if rng.random() < random_action_prob:
        action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
    else:
        action = base_action + rng.normal(0.0, noise_std, size=2).astype(np.float32)

    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _rollout_episode(
    env: Grid2DGoalEnv,
    rng: np.random.Generator,
    mode: str,
    guided_noise_std: float,
    guided_random_action_prob: float,
):
    obs = env.reset(rng)
    obs_hist = []
    action_hist = []
    reward_hist = []
    done_hist = []

    while True:
        if mode == "guided":
            action = _guided_action(
                env=env,
                rng=rng,
                noise_std=guided_noise_std,
                random_action_prob=guided_random_action_prob,
            )
        elif mode == "random":
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        else:
            raise ValueError(f"Unknown rollout mode: {mode}")

        obs_hist.append(obs.copy())
        action_hist.append(action.copy())

        obs, reward, done, info = env.step(action)
        reward_hist.append(reward)
        done_hist.append(done)

        if done:
            break

    is_success = bool(info["is_success"])
    traj_type = 1 if mode == "guided" else 0
    T = len(obs_hist)
    episode = {
        "obs": np.asarray(obs_hist, dtype=np.float32),       # [T, 2]
        "actions": np.asarray(action_hist, dtype=np.float32),  # [T, 2]
        "rewards": np.asarray(reward_hist, dtype=np.float32),  # [T]
        "dones": np.asarray(done_hist, dtype=np.bool_),        # [T]
        "success": np.full((T,), float(is_success), dtype=np.float32),
        "traj_type": np.full((T,), traj_type, dtype=np.int8),
    }
    return episode, is_success


def _write_zarr_dataset(output_path: str, episodes, meta_dict):
    lengths = np.array([ep["actions"].shape[0] for ep in episodes], dtype=np.int64)
    episode_ends = np.cumsum(lengths, dtype=np.int64)

    obs = np.concatenate([ep["obs"] for ep in episodes], axis=0).astype(np.float32)
    actions = np.concatenate([ep["actions"] for ep in episodes], axis=0).astype(np.float32)
    rewards = np.concatenate([ep["rewards"] for ep in episodes], axis=0).astype(np.float32)
    dones = np.concatenate([ep["dones"] for ep in episodes], axis=0).astype(np.bool_)

    total_steps = int(actions.shape[0])
    chunk_t = min(8192, max(1, total_steps))

    root = zarr.open_group(output_path, mode="w")
    data_group = root.require_group("data", overwrite=True)
    obs_group = data_group.require_group("obs", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    # Required layout.
    data_group.array(
        name="actions",
        data=actions,
        shape=actions.shape,
        chunks=(chunk_t, actions.shape[1]),
    )
    obs_group.array(
        name="x",
        data=obs[:, 0:1],
        shape=(total_steps, 1),
        chunks=(chunk_t, 1),
    )
    obs_group.array(
        name="y",
        data=obs[:, 1:2],
        shape=(total_steps, 1),
        chunks=(chunk_t, 1),
    )
    data_group.array(
        name="rewards",
        data=rewards,
        shape=rewards.shape,
        chunks=(chunk_t,),
    )
    data_group.array(
        name="dones",
        data=dones,
        shape=dones.shape,
        chunks=(chunk_t,),
    )
    meta_group.array(
        name="episode_ends",
        data=episode_ends,
        shape=episode_ends.shape,
        chunks=episode_ends.shape,
    )

    # Helpful metadata (optional).
    for key, value in meta_dict.items():
        arr = np.array(value)
        meta_group.array(
            name=key,
            data=arr,
            shape=arr.shape,
            chunks=arr.shape if arr.shape else (),
        )


def _visualize_trajectories(
    success_episodes: List[dict],
    random_episodes: List[dict],
    output_path: str,
    rng: np.random.Generator,
    n_success: int = 50,
    n_random: int = 50,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib not available; skipping trajectory visualization. "
            "Install matplotlib to enable this feature."
        )
        return None

    n_success = min(n_success, len(success_episodes))
    n_random = min(n_random, len(random_episodes))

    if n_success == 0 and n_random == 0:
        print("No episodes available for visualization.")
        return None

    success_idxs = (
        rng.choice(len(success_episodes), size=n_success, replace=False)
        if n_success > 0
        else np.array([], dtype=np.int64)
    )
    random_idxs = (
        rng.choice(len(random_episodes), size=n_random, replace=False)
        if n_random > 0
        else np.array([], dtype=np.int64)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    ax_success, ax_random = axes

    def _draw_panel(ax, selected_indices, source_episodes, title, color):
        for idx in selected_indices:
            traj = source_episodes[int(idx)]["obs"]
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.28, linewidth=1.2)
            ax.scatter(traj[0, 0], traj[0, 1], color="tab:blue", s=8, alpha=0.45)
        ax.scatter([1.0], [1.0], marker="*", color="black", s=120, label="Goal (1,1)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)

    _draw_panel(
        ax_success,
        success_idxs,
        success_episodes,
        f"Successful trajectories (n={n_success})",
        "tab:green",
    )
    _draw_panel(
        ax_random,
        random_idxs,
        random_episodes,
        f"Random trajectories (n={n_random})",
        "tab:red",
    )

    fig.suptitle("Nav2D trajectory preview", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


@click.command()
@click.option("--output", "-o", required=True, help="Output zarr path (directory).")
@click.option("--n_episodes", "-n", default=1000, show_default=True, type=int)
@click.option("--success_ratio", default=0.5, show_default=True, type=float)
@click.option("--max_steps", default=200, show_default=True, type=int)
@click.option("--action_scale", default=0.01, show_default=True, type=float)
@click.option("--goal_radius", default=0.03, show_default=True, type=float)
@click.option("--guided_noise_std", default=0.35, show_default=True, type=float)
@click.option(
    "--guided_random_action_prob", default=0.25, show_default=True, type=float
)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option(
    "--viz_success_count",
    default=50,
    show_default=True,
    type=int,
    help="Number of successful trajectories to visualize.",
)
@click.option(
    "--viz_random_count",
    default=50,
    show_default=True,
    type=int,
    help="Number of random trajectories to visualize.",
)
@click.option(
    "--disable_viz",
    is_flag=True,
    help="Disable end-of-collection trajectory visualization.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite output path if it exists.")
def main(
    output,
    n_episodes,
    success_ratio,
    max_steps,
    action_scale,
    goal_radius,
    guided_noise_std,
    guided_random_action_prob,
    seed,
    viz_success_count,
    viz_random_count,
    disable_viz,
    overwrite,
):
    if n_episodes <= 0:
        raise click.ClickException("n_episodes must be > 0.")
    if not (0.0 <= success_ratio <= 1.0):
        raise click.ClickException("success_ratio must be in [0, 1].")

    output = os.path.expanduser(output)
    if os.path.exists(output):
        if not overwrite:
            raise click.ClickException(
                f"Output path already exists: {output}. Use --overwrite to replace it."
            )
        shutil.rmtree(output)

    rng = np.random.default_rng(seed)
    env = Grid2DGoalEnv(
        action_scale=action_scale,
        goal_radius=goal_radius,
        max_steps=max_steps,
    )

    n_success_target = int(round(n_episodes * success_ratio))
    n_random_target = n_episodes - n_success_target

    episodes = []
    success_episodes = []
    random_episodes = []

    print(
        "Generating dataset with "
        f"{n_success_target} successful and {n_random_target} random trajectories."
    )

    max_attempts_per_episode = 200

    # Successful (guided) trajectories.
    for _ in tqdm(range(n_success_target), desc="Successful trajectories"):
        for attempt in range(max_attempts_per_episode):
            episode, is_success = _rollout_episode(
                env=env,
                rng=rng,
                mode="guided",
                guided_noise_std=guided_noise_std,
                guided_random_action_prob=guided_random_action_prob,
            )
            if is_success:
                episodes.append(episode)
                success_episodes.append(episode)
                break
            if attempt == max_attempts_per_episode - 1:
                raise RuntimeError(
                    "Failed to generate a successful trajectory within attempt limit."
                )

    # Random failed trajectories.
    for _ in tqdm(range(n_random_target), desc="Random trajectories"):
        for attempt in range(max_attempts_per_episode):
            episode, is_success = _rollout_episode(
                env=env,
                rng=rng,
                mode="random",
                guided_noise_std=guided_noise_std,
                guided_random_action_prob=guided_random_action_prob,
            )
            if not is_success:
                episodes.append(episode)
                random_episodes.append(episode)
                break
            if attempt == max_attempts_per_episode - 1:
                raise RuntimeError(
                    "Failed to generate a random failed trajectory within attempt limit."
                )

    meta = {
        "goal": env.goal.astype(np.float32),
        "action_scale": np.array([action_scale], dtype=np.float32),
        "goal_radius": np.array([goal_radius], dtype=np.float32),
        "seed": np.array([seed], dtype=np.int64),
        "n_episodes": np.array([n_episodes], dtype=np.int64),
        "n_success_target": np.array([n_success_target], dtype=np.int64),
        "n_random_target": np.array([n_random_target], dtype=np.int64),
    }
    _write_zarr_dataset(output, episodes, meta)

    returns = []
    lengths = []
    successes = []
    for ep in episodes:
        returns.append(float(np.sum(ep["rewards"])))
        lengths.append(int(len(ep["rewards"])))
        successes.append(bool(np.max(ep["success"]) > 0.5))

    print(f"Saved dataset to: {output}")
    print(f"Episodes: {len(episodes)}, Steps: {sum(lengths)}")
    print("Zarr layout: data/actions, data/obs/{x,y}, data/rewards, data/dones, meta/episode_ends")
    print(f"Success rate: {np.mean(successes):.4f}")
    print(f"Average return: {np.mean(returns):.4f}")
    print(f"Average length: {np.mean(lengths):.2f}")

    if not disable_viz:
        output_dir = os.path.dirname(output.rstrip("/")) or "."
        output_name = os.path.basename(output.rstrip("/"))
        viz_path = os.path.join(output_dir, f"{output_name}_traj_preview.png")
        saved_path = _visualize_trajectories(
            success_episodes=success_episodes,
            random_episodes=random_episodes,
            output_path=viz_path,
            rng=np.random.default_rng(seed + 17),
            n_success=viz_success_count,
            n_random=viz_random_count,
        )
        if saved_path is not None:
            print(f"Saved trajectory visualization to: {saved_path}")


if __name__ == "__main__":
    main()
