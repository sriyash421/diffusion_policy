"""
Usage examples:

# New grid2d mode (default):
python eval_2d.py --mode grid2d -c <checkpoint.ckpt> -o <output_dir>

# Original env_runner mode:
python eval_2d.py --mode legacy -c <checkpoint.ckpt> -o <output_dir>
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import json
import os
import pathlib
import shutil
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import click
import dill
import hydra
import numpy as np
import torch
import wandb

from diffusion_policy.workspace.base_workspace import BaseWorkspace


def _prepare_output_dir(output_dir: str, overwrite: bool):
    if os.path.exists(output_dir):
        if not overwrite:
            raise click.ClickException(
                f"Output path already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)


def _load_workspace_policy(checkpoint: str, output_dir: str, device: torch.device):
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    if "cfg" not in payload:
        raise click.ClickException(
            "Expected a diffusion_policy workspace checkpoint containing 'cfg'."
        )
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if getattr(cfg.training, "use_ema", False):
        policy = workspace.ema_model

    policy.to(device)
    policy.eval()
    return policy, cfg


def run_legacy_mode(checkpoint: str, output_dir: str, device: str):
    device_t = torch.device(device)
    policy, cfg = _load_workspace_policy(
        checkpoint=checkpoint, output_dir=output_dir, device=device_t
    )

    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    runner_log = env_runner.run(policy)

    json_log = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value

    out_path = os.path.join(output_dir, "eval_log.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_log, f, indent=2, sort_keys=True)
    print(f"Saved legacy eval log to: {out_path}")


@dataclass
class Grid2DGoalEnv:
    action_scale: float = 0.01
    goal_radius: float = 0.03
    max_steps: int = 200
    grid_bins: int = 100
    frame_size: int = 256
    frame_padding: int = 18

    def __post_init__(self):
        self.goal = np.array([1.0, 1.0], dtype=np.float32)
        self.state = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.path: List[np.ndarray] = []

    def reset(self, rng: np.random.Generator):
        self.state = (
            rng.integers(0, self.grid_bins + 1, size=2).astype(np.float32)
            / float(self.grid_bins)
        )
        self.step_count = 0
        self.path = [self.state.copy()]
        return self.state.copy()

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.state = np.clip(self.state + action * self.action_scale, 0.0, 1.0)
        self.step_count += 1
        self.path.append(self.state.copy())

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

    def _to_pixel(self, pos: np.ndarray) -> Tuple[int, int]:
        size = self.frame_size
        pad = self.frame_padding
        x = int(pad + np.clip(pos[0], 0.0, 1.0) * (size - 2 * pad))
        y = int(size - (pad + np.clip(pos[1], 0.0, 1.0) * (size - 2 * pad)))
        return x, y

    @staticmethod
    def _draw_disk(img: np.ndarray, center: Tuple[int, int], radius: int, color):
        cx, cy = center
        h, w, _ = img.shape
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        img[y0:y1, x0:x1][mask] = color

    def _draw_line(self, img: np.ndarray, p0: Tuple[int, int], p1: Tuple[int, int], color):
        x0, y0 = p0
        x1, y1 = p1
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        if n <= 1:
            self._draw_disk(img, p0, radius=1, color=color)
            return
        for i in range(n):
            t = i / float(n - 1)
            x = int(round(x0 + (x1 - x0) * t))
            y = int(round(y0 + (y1 - y0) * t))
            self._draw_disk(img, (x, y), radius=1, color=color)

    def render(self) -> np.ndarray:
        size = self.frame_size
        pad = self.frame_padding
        frame = np.full((size, size, 3), 255, dtype=np.uint8)

        # workspace border
        frame[pad : pad + 2, pad : size - pad] = [30, 30, 30]
        frame[size - pad - 2 : size - pad, pad : size - pad] = [30, 30, 30]
        frame[pad : size - pad, pad : pad + 2] = [30, 30, 30]
        frame[pad : size - pad, size - pad - 2 : size - pad] = [30, 30, 30]

        # trajectory
        if len(self.path) >= 2:
            pixels = [self._to_pixel(p) for p in self.path]
            for i in range(1, len(pixels)):
                self._draw_line(frame, pixels[i - 1], pixels[i], color=(90, 110, 235))

        # goal and agent
        goal_px = self._to_pixel(self.goal)
        agent_px = self._to_pixel(self.state)
        self._draw_disk(frame, goal_px, radius=7, color=(40, 190, 70))
        self._draw_disk(frame, agent_px, radius=6, color=(220, 60, 60))
        return frame


def _extract_first_action(action_like) -> np.ndarray:
    if torch.is_tensor(action_like):
        arr = action_like.detach().cpu().numpy()
    elif isinstance(action_like, np.ndarray):
        arr = action_like
    else:
        raise TypeError(f"Unsupported action type: {type(action_like)}")

    if arr.ndim == 3:
        return arr[0, 0].astype(np.float32)
    if arr.ndim == 2:
        return arr[0].astype(np.float32)
    if arr.ndim == 1:
        return arr.astype(np.float32)
    raise ValueError(f"Unexpected action shape: {arr.shape}")


def _infer_obs_keys(cfg) -> Optional[List[str]]:
    """Infer observation key order from checkpoint config shape_meta."""
    try:
        obs_meta = cfg.task.shape_meta.get("obs", None)
        if obs_meta is None:
            return None
        keys = list(obs_meta.keys())
        return keys if len(keys) > 0 else None
    except Exception:
        return None


def _build_obs_dict(
    obs_history: Deque[np.ndarray],
    device: torch.device,
    obs_keys: Optional[List[str]] = None,
):
    obs_array = np.asarray(obs_history, dtype=np.float32)
    obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(device)  # [B=1, To, 2]

    # Flat style used by lowdim policies.
    fallback_obs_dict = {"obs": obs_tensor}

    # Keyed style used by image/hybrid policies with shape_meta.obs entries.
    if not obs_keys:
        return fallback_obs_dict

    if "x" in obs_keys and "y" in obs_keys and obs_tensor.shape[-1] >= 2:
        return {
            "x": obs_tensor[..., 0:1],
            "y": obs_tensor[..., 1:2],
        }

    # Generic mapping if key count matches feature dimension.
    if len(obs_keys) == obs_tensor.shape[-1]:
        obs_dict = {}
        for i, key in enumerate(obs_keys):
            obs_dict[key] = obs_tensor[..., i : i + 1]
        return obs_dict

    # Single-key case uses the full vector.
    if len(obs_keys) == 1:
        return {obs_keys[0]: obs_tensor}

    return fallback_obs_dict


def _predict_policy_action(
    policy,
    obs_history: Deque[np.ndarray],
    device: torch.device,
    obs_keys: Optional[List[str]] = None,
):
    obs_dict = _build_obs_dict(obs_history=obs_history, device=device, obs_keys=obs_keys)
    obs_tensor = obs_dict.get("obs", None)

    with torch.no_grad():
        if hasattr(policy, "predict_action"):
            out = policy.predict_action(obs_dict)
        else:
            if obs_tensor is None:
                raise RuntimeError(
                    "Policy has no predict_action() and keyed obs was inferred. "
                    "Use a policy with predict_action() or disable keyed obs mapping."
                )
            out = policy(obs_tensor)

    if isinstance(out, dict):
        if "action" not in out:
            raise RuntimeError("Policy output dict does not contain 'action' key.")
        action_like = out["action"]
    else:
        action_like = out

    action = _extract_first_action(action_like)
    return np.clip(action, -1.0, 1.0)


def _rollout_grid2d_episode(
    env: Grid2DGoalEnv,
    policy,
    device: torch.device,
    rng: np.random.Generator,
    n_obs_steps: int,
    obs_keys: Optional[List[str]] = None,
):
    if hasattr(policy, "reset"):
        policy.reset()

    obs = env.reset(rng)
    obs_history: Deque[np.ndarray] = deque(maxlen=n_obs_steps)
    for _ in range(n_obs_steps):
        obs_history.append(obs.copy())

    frames = [env.render()]
    total_reward = 0.0
    done = False
    info = {"is_success": False}

    while not done:
        action = _predict_policy_action(
            policy, obs_history=obs_history, device=device, obs_keys=obs_keys
        )
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        obs_history.append(obs.copy())
        frames.append(env.render())

    return total_reward, env.step_count, bool(info["is_success"]), frames


def run_grid2d_mode(
    checkpoint: str,
    output_dir: str,
    device: str,
    n_episodes: int,
    max_steps: int,
    action_scale: float,
    goal_radius: float,
    n_obs_steps: int,
    seed: int,
    max_video_episodes: int,
    video_fps: int,
    wandb_project: str,
    wandb_entity: str,
    wandb_run_name: str,
    wandb_mode: str,
):
    if n_episodes <= 0:
        raise click.ClickException("n_episodes must be > 0.")

    device_t = torch.device(device)
    policy, cfg = _load_workspace_policy(
        checkpoint=checkpoint, output_dir=output_dir, device=device_t
    )
    obs_keys = _infer_obs_keys(cfg)

    inferred_n_obs_steps = n_obs_steps
    if inferred_n_obs_steps <= 0:
        inferred_n_obs_steps = int(getattr(policy, "n_obs_steps", 1))
        inferred_n_obs_steps = max(1, inferred_n_obs_steps)

    env = Grid2DGoalEnv(
        action_scale=action_scale,
        goal_radius=goal_radius,
        max_steps=max_steps,
    )
    rng = np.random.default_rng(seed)

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_run_name,
        mode="online",
        config={
            "mode": "grid2d",
            "checkpoint": checkpoint,
            "output_dir": output_dir,
            "device": str(device_t),
            "n_episodes": n_episodes,
            "max_steps": max_steps,
            "action_scale": action_scale,
            "goal_radius": goal_radius,
            "n_obs_steps": inferred_n_obs_steps,
            "seed": seed,
            "max_video_episodes": max_video_episodes,
            "video_fps": video_fps,
            "workspace_target": str(getattr(cfg, "_target_", "")),
            "obs_keys": obs_keys if obs_keys is not None else [],
        },
    )

    if obs_keys is not None:
        print(f"Using keyed obs style with keys: {obs_keys}")
    else:
        print("Using flat obs style with key: ['obs']")

    returns = []
    steps = []
    successes = []

    for ep_idx in range(n_episodes):
        ep_return, ep_steps, ep_success, frames = _rollout_grid2d_episode(
            env=env,
            policy=policy,
            device=device_t,
            rng=rng,
            n_obs_steps=inferred_n_obs_steps,
            obs_keys=obs_keys,
        )
        returns.append(ep_return)
        steps.append(ep_steps)
        successes.append(float(ep_success))

        log_data = {
            "episode/return": ep_return,
            "episode/steps": ep_steps,
            "episode/success": float(ep_success),
        }
        if ep_idx < max_video_episodes:
            video = np.stack(frames, axis=0)
            # wandb.Video expects T,C,H,W for numpy input.
            video = np.transpose(video, (0, 3, 1, 2))
            log_data[f"episode/video_{ep_idx:03d}"] = wandb.Video(
                video, fps=video_fps, format="mp4"
            )
        run.log(log_data, step=ep_idx)

    avg_return = float(np.mean(returns))
    avg_steps = float(np.mean(steps))
    success_rate = float(np.mean(successes))

    final_log = {
        "eval/avg_return": avg_return,
        "eval/avg_steps": avg_steps,
        "eval/success_rate": success_rate,
    }
    run.log(final_log, step=n_episodes)
    run.summary["eval/avg_return"] = avg_return
    run.summary["eval/avg_steps"] = avg_steps
    run.summary["eval/success_rate"] = success_rate
    run.finish()

    json_out = {
        "avg_return": avg_return,
        "avg_steps": avg_steps,
        "success_rate": success_rate,
        "n_episodes": n_episodes,
        "returns": [float(x) for x in returns],
        "steps": [int(x) for x in steps],
        "successes": [float(x) for x in successes],
    }
    out_path = os.path.join(output_dir, "grid2d_eval_log.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, sort_keys=True)

    print(f"Saved grid2d eval log to: {out_path}")
    print(f"Average return: {avg_return:.4f}")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Success rate: {success_rate:.4f}")


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["grid2d", "legacy"]),
    default="grid2d",
    show_default=True,
)
@click.option("-c", "--checkpoint", required=True, help="Workspace checkpoint path.")
@click.option("-o", "--output_dir", required=True, help="Output directory.")
@click.option("-d", "--device", default="cuda:0", show_default=True)
@click.option("--overwrite", is_flag=True, help="Overwrite output directory.")
# grid2d options
@click.option("--n_episodes", default=50, show_default=True, type=int)
@click.option("--max_steps", default=200, show_default=True, type=int)
@click.option("--action_scale", default=0.01, show_default=True, type=float)
@click.option("--goal_radius", default=0.03, show_default=True, type=float)
@click.option(
    "--n_obs_steps",
    default=0,
    show_default=True,
    type=int,
    help="If <= 0, infer from policy.n_obs_steps or fallback to 1.",
)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--max_video_episodes", default=5, show_default=True, type=int)
@click.option("--video_fps", default=15, show_default=True, type=int)
@click.option("--wandb_project", default="grid2d_eval", show_default=True)
@click.option("--wandb_entity", default=None)
@click.option("--wandb_run_name", default=None)
@click.option(
    "--wandb_mode",
    type=click.Choice(["online", "offline", "disabled"]),
    default="offline",
    show_default=True,
)
def main(
    mode,
    checkpoint,
    output_dir,
    device,
    overwrite,
    n_episodes,
    max_steps,
    action_scale,
    goal_radius,
    n_obs_steps,
    seed,
    max_video_episodes,
    video_fps,
    wandb_project,
    wandb_entity,
    wandb_run_name,
    wandb_mode,
):
    _prepare_output_dir(output_dir=output_dir, overwrite=overwrite)

    if mode == "legacy":
        run_legacy_mode(checkpoint=checkpoint, output_dir=output_dir, device=device)
        return

    run_grid2d_mode(
        checkpoint=checkpoint,
        output_dir=output_dir,
        device=device,
        n_episodes=n_episodes,
        max_steps=max_steps,
        action_scale=action_scale,
        goal_radius=goal_radius,
        n_obs_steps=n_obs_steps,
        seed=seed,
        max_video_episodes=max_video_episodes,
        video_fps=video_fps,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
    )


if __name__ == "__main__":
    main()
