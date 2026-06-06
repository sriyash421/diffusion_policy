"""Verifier clients for the L2S `SearchPolicy`.

Each verifier exposes a ``get_value(obs_dict, action) -> Tensor`` method that
mirrors the in-process ``MazeVerifier`` interface used by
``diffusion_policy.policy.search_policy.SearchPolicy``.

Both backends are routed through
``RoboMonkey/monkey-verifier/src/verifier_client.VerifierClient``:

* ``server_url="in_process"`` (or ``"0"`` / ``""`` / ``None``) loads
  ``RobotRewardModel`` into this Python process and scores all B
  (image, action) pairs in one GPU forward.
* ``server_url="http://host:port"`` POSTs to ``/process`` on the running
  ``infer_server.py``.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image


def _ensure_verifier_client_on_path() -> None:
    """Make `from verifier_client import VerifierClient` work.

    Honors `MONKEY_VERIFIER_SRC` if set, otherwise tries a few likely
    locations relative to common checkout layouts.
    """
    env_path = os.environ.get("MONKEY_VERIFIER_SRC")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        Path.home() / "RoboMonkey" / "monkey-verifier" / "src",
        Path(__file__).resolve().parents[3] / "RoboMonkey" / "monkey-verifier" / "src",
    ])
    for c in candidates:
        if (c / "verifier_client.py").is_file():
            p = str(c)
            if p not in sys.path:
                sys.path.insert(0, p)
            return
    raise RuntimeError(
        "Could not locate `monkey-verifier/src/verifier_client.py`. "
        "Set MONKEY_VERIFIER_SRC=/path/to/RoboMonkey/monkey-verifier/src."
    )


def _to_hwc_uint8(img: torch.Tensor) -> np.ndarray:
    """Convert a single image tensor to HWC uint8 [0, 255]."""
    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(img)
    arr = img.detach().cpu()
    if arr.dim() != 3:
        raise ValueError(
            f"Expected a single image with 3 dims (CHW or HWC), got shape {tuple(arr.shape)}"
        )
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = arr.permute(1, 2, 0).contiguous()
    elif arr.shape[-1] != 3:
        raise ValueError(
            f"Expected an RGB image with 3 channels, got shape {tuple(arr.shape)}"
        )
    if arr.dtype.is_floating_point:
        max_val = float(arr.max().item()) if arr.numel() > 0 else 0.0
        if max_val <= 1.0 + 1e-3:
            arr = arr * 255.0
        arr = arr.clamp(0, 255).to(torch.uint8)
    else:
        arr = arr.to(torch.uint8)
    return arr.numpy()


def _image_key(im: np.ndarray) -> bytes:
    """Content hash of a HWC uint8 image, used to key the verifier's prefix-KV
    cache so the same observation reuses its encoded prefix across rounds."""
    import hashlib
    return hashlib.blake2b(np.ascontiguousarray(im).tobytes(), digest_size=16).digest()


class RoboMonkeyVerifier:
    """RoboMonkey reward-model client. Supports in-process and HTTP modes.

    Args:
        server_url: ``"in_process"`` (or ``"0"`` / ``""``) loads the verifier
            locally. ``"http://host:port"`` POSTs to the server.
        instruction: Static language instruction used for every call.
        image_obs_key: Key in ``obs_dict`` holding the RGB observation.
            Expected shape ``(B, To, C, H, W)`` or ``(B, To, H, W, C)``;
            the latest step is sent.
        action_chunk_index: Index along the horizon dim of the action chunk
            to score. Default ``-1`` = last step.
        request_timeout: Per-request HTTP timeout (HTTP mode only).
        max_workers: Threads used to fan out parallel HTTP requests
            (HTTP mode only).
        image_save_dir: Temp dir for JPEGs (HTTP mode only; in-process passes
            the numpy frame directly).
        keep_jpegs: If True, leave temp JPEGs on disk after each call
            (HTTP mode only).
        health_check: If True, verify the backend is reachable at construction.
    """

    def __init__(
        self,
        server_url: str,
        instruction: str,
        image_obs_key: str,
        image_save_dir: str = "/tmp/robomonkey_obs",
        action_chunk_index: int = -1,
        # Chunk scoring: when score_indices is set (e.g. the executed action
        # window [1..n_action_steps]), get_value scores EACH of those horizon
        # steps against the current frame and aggregates via score_agg
        # (mean | sum | min) into one value per candidate. Set in the policy
        # config so it holds in BOTH training and inference (the checkpoint
        # carries it). None -> legacy single-step (action_chunk_index) scoring.
        score_indices: Optional[list] = None,
        score_agg: str = "mean",
        request_timeout: float = 300.0,
        max_workers: Optional[int] = 8,
        keep_jpegs: bool = False,
        health_check: bool = True,
        # Unused; kept for backwards-compat with existing YAML configs.
        vocab_size: int = 32064,
        n_bins: int = 512,
        action_min: float = -1.0,
        action_max: float = 1.0,
    ) -> None:
        _ensure_verifier_client_on_path()
        from verifier_client import VerifierClient  # noqa: WPS433

        self.instruction = instruction
        self.image_obs_key = image_obs_key
        self.action_chunk_index = action_chunk_index
        self.score_indices = (list(score_indices)
                              if score_indices is not None else None)
        self.score_agg = str(score_agg)
        self.image_save_dir = Path(image_save_dir)
        self.max_workers = max_workers
        self.keep_jpegs = bool(keep_jpegs)

        self._client = VerifierClient(server_url, request_timeout=request_timeout)
        if not self._client.in_process:
            self.image_save_dir.mkdir(parents=True, exist_ok=True)
        if health_check:
            self._client.health_check()

    def _select_last_frame(self, obs_dict: Dict[str, Any]) -> torch.Tensor:
        if self.image_obs_key not in obs_dict:
            raise KeyError(
                f"RoboMonkeyVerifier: obs_dict missing image key "
                f"'{self.image_obs_key}'. Available: {sorted(obs_dict.keys())}"
            )
        x = obs_dict[self.image_obs_key]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.dim() != 5:
            raise ValueError(
                f"Expected obs '{self.image_obs_key}' to have 5 dims "
                f"(B, To, C, H, W) or (B, To, H, W, C); got {tuple(x.shape)}"
            )
        return x[:, -1]

    @torch.no_grad()
    def get_value(
        self,
        obs_dict: Dict[str, Any],
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Score each candidate against the current frame. Returns (B,).

        By default scores a single horizon step (``action_chunk_index``). If
        ``score_indices`` is set (e.g. the executed action window steps), scores
        EACH of those steps against the same current frame and aggregates them
        into one value per candidate via ``score_agg`` (mean / sum / min). This
        makes "value" reflect the quality of the whole executed chunk rather
        than one step. Cost scales with the number of indices (M steps -> M x
        the verifier forwards).
        """
        if action.dim() != 3:
            raise ValueError(
                f"Expected action (B, horizon, action_dim); got {tuple(action.shape)}"
            )
        last_frames = self._select_last_frame(obs_dict)
        B = action.shape[0]
        if last_frames.shape[0] != B:
            raise ValueError(
                f"Batch size mismatch: action B={B} but obs B={last_frames.shape[0]}"
            )

        # Horizon step(s) to score. Default: single configured index.
        idxs = getattr(self, "score_indices", None)
        if idxs is None:
            idxs = [self.action_chunk_index]
        idxs = [int(i) for i in idxs]
        M = len(idxs)
        # (B, M, action_dim) -> (B*M, action_dim), row order b-major:
        # [b0:i0, b0:i1, ..., b1:i0, ...]. Each (image_b, action_{b,m}) is one
        # scored pair; the image is repeated M times within a batch row.
        action_step = action[:, idxs, :].detach().cpu().numpy().reshape(B * M, action.shape[-1])

        if self._client.in_process:
            frames = [_to_hwc_uint8(last_frames[b]) for b in range(B)]
            images = [frames[b] for b in range(B) for _ in range(M)]
            if getattr(self, "_use_kv_prefix", False):
                # Same images recur across the policy's autoregressive rounds:
                # reuse each image's cached prefix KV, forward only the action
                # suffix. Caller chunks the batch + clears between chunks.
                keys = [_image_key(frames[b]) for b in range(B) for _ in range(M)]
                flat = self._client.score_paired_kvprefix(
                    self.instruction, images, action_step, keys,
                )
            else:
                flat = self._client.score_paired(
                    self.instruction, images, action_step,
                    max_workers=(self.max_workers or 1),
                )
        else:
            run_id = uuid.uuid4().hex
            jpeg_paths = [
                self.image_save_dir / f"obs_{run_id}_{b}.jpg" for b in range(B)
            ]
            for b in range(B):
                Image.fromarray(_to_hwc_uint8(last_frames[b])).save(
                    str(jpeg_paths[b]), format="JPEG", quality=95
                )
            try:
                images = [str(jpeg_paths[b]) for b in range(B) for _ in range(M)]
                flat = self._client.score_paired(
                    self.instruction, images, action_step,
                    max_workers=(self.max_workers or 1),
                )
            finally:
                if not self.keep_jpegs:
                    for p in jpeg_paths:
                        try:
                            os.remove(p)
                        except OSError:
                            pass

        rewards = torch.tensor(
            flat, dtype=action.dtype, device=action.device
        ).reshape(B, M)
        if M == 1:
            return rewards[:, 0]
        agg = getattr(self, "score_agg", "mean")
        if agg == "mean":
            return rewards.mean(dim=1)
        if agg == "sum":
            return rewards.sum(dim=1)
        if agg == "min":
            return rewards.min(dim=1).values
        raise ValueError(f"unknown score_agg {agg!r} (use mean/sum/min)")

    # --- per-image prefix-KV controls (in-process training speedup) ---------
    @property
    def supports_kv_prefix(self) -> bool:
        """KV-prefix reuse is only available for the in-process verifier."""
        return bool(self._client.in_process)

    def set_kv_prefix(self, enabled: bool) -> None:
        """Route get_value through the cached-prefix path while True."""
        self._use_kv_prefix = bool(enabled) and self.supports_kv_prefix

    def clear_prefix_cache(self) -> None:
        """Drop cached prefix KV (call between independent image chunks)."""
        if self.supports_kv_prefix:
            self._client.clear_prefix_cache()


class RoboMonkeyStateVerifier:
    """RoboMonkey verifier for state-only obs.

    Training data in ``state0.zarr`` doesn't store an RGB frame, so we render
    one on the fly from the saved state by driving a SIMPLER env to the
    snapshot pose, then hand off to ``RoboMonkeyVerifier``.
    """

    def __init__(
        self,
        server_url: str,
        instruction: str,
        simpler_task: str,
        arm_qpos_key: str = "arm_joint_pos",
        src_pose_key: str = "insertive_asset_pose",
        tgt_pose_key: str = "receptive_asset_pose",
        resize_size: int = 256,
        image_save_dir: str = "/tmp/robomonkey_obs",
        **rm_kwargs: Any,
    ) -> None:
        # Lazy import — these are heavy and only available in the SIMPLER env.
        import simpler_env  # type: ignore
        from sapien.core import Pose  # type: ignore
        from experiments.robot.simpler.simpler_utils import (  # type: ignore
            get_simpler_img,
        )

        self._simpler_env_mod = simpler_env
        self._Pose = Pose
        self._get_simpler_img = get_simpler_img
        self.simpler_task = simpler_task
        self.arm_qpos_key = arm_qpos_key
        self.src_pose_key = src_pose_key
        self.tgt_pose_key = tgt_pose_key
        self.resize_size = int(resize_size)

        self._env = simpler_env.make(simpler_task)
        self._last_obs, _ = self._env.reset(), None

        self._inner = RoboMonkeyVerifier(
            server_url=server_url,
            instruction=instruction,
            image_obs_key="_rendered",
            image_save_dir=image_save_dir,
            **rm_kwargs,
        )

    def _pose_from_vec(self, vec: np.ndarray):
        # Saved as [x, y, z, qw, qx, qy, qz]
        return self._Pose(p=vec[:3], q=vec[3:7])

    def _set_state(self, arm_qpos: np.ndarray, src_pose: np.ndarray,
                   tgt_pose: np.ndarray):
        env = self._env
        qpos = np.asarray(env.agent.robot.get_qpos()).copy()
        n = min(arm_qpos.shape[0], qpos.shape[0])
        qpos[:n] = arm_qpos[:n]
        env.agent.robot.set_qpos(qpos)
        src = getattr(env, "episode_source_obj", None)
        tgt = getattr(env, "episode_target_obj", None)
        if src is not None:
            src.set_pose(self._pose_from_vec(np.asarray(src_pose)))
        if tgt is not None:
            tgt.set_pose(self._pose_from_vec(np.asarray(tgt_pose)))

    def _render_one(self, arm_qpos: np.ndarray, src_pose: np.ndarray,
                    tgt_pose: np.ndarray) -> np.ndarray:
        self._set_state(arm_qpos, src_pose, tgt_pose)
        try:
            zero_action = np.zeros(self._env.action_space.shape, dtype=np.float32)
            obs, _, _, _, _ = self._env.step(zero_action)
        except Exception:
            obs = self._last_obs
        self._last_obs = obs
        img = self._get_simpler_img(self._env, obs, self.resize_size)
        return np.asarray(img, dtype=np.uint8)

    @torch.no_grad()
    def get_value(
        self,
        obs_dict: Dict[str, Any],
        action: torch.Tensor,
    ) -> torch.Tensor:
        if action.dim() != 3:
            raise ValueError(
                f"Expected action (B, horizon, action_dim); got {tuple(action.shape)}"
            )

        def _last(key):
            x = obs_dict[key]
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            else:
                x = np.asarray(x)
            return x[:, -1]

        arm = _last(self.arm_qpos_key)
        src = _last(self.src_pose_key)
        tgt = _last(self.tgt_pose_key)
        B = action.shape[0]
        rendered = np.stack(
            [self._render_one(arm[b], src[b], tgt[b]) for b in range(B)],
            axis=0,
        )  # (B, H, W, 3)
        rendered_t = torch.from_numpy(rendered).unsqueeze(1)  # (B, 1, H, W, 3)
        return self._inner.get_value({"_rendered": rendered_t}, action)


class MSEVerifier:
    """Verifier scoring an action by its closeness to the dataset expert action.

    Returns the negative MSE between the predicted action chunk and the expert
    action chunk read from ``obs_dict`` — negated so that higher = closer to
    the expert, matching the "higher is better" convention of the reward-model
    verifiers. The expert action is only available at training time, so this
    verifier is meant for training the search policy, not eval-time rollouts.

    Optionally, Gaussian noise can be added to the score to simulate an
    imperfect verifier during training. The noise is drawn per call from
    ``Normal(noise_bias, noise)``, so ``noise`` controls the magnitude of the
    random perturbation and ``noise_bias`` shifts the score by a constant
    offset (a systematically optimistic or pessimistic verifier).

    Args:
        expert_action_key: Key in ``obs_dict`` holding the expert action,
            shape ``(B, horizon, action_dim)``.
        noise: Standard deviation of the Gaussian noise added to each score.
            ``0.0`` (default) disables the random perturbation.
        noise_bias: Constant offset (the mean of the Gaussian noise) added to
            each score. ``0.0`` (default) adds no bias.
    """

    def __init__(
        self,
        expert_action_key: str = "action",
        noise: float = 0.0,
        noise_bias: float = 0.0,
    ) -> None:
        self.expert_action_key = expert_action_key
        self.noise = float(noise)
        self.noise_bias = float(noise_bias)

    @torch.no_grad()
    def get_value(
        self,
        obs_dict: Dict[str, Any],
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Score one action per batch element. Returns (B,) on action.device."""
        if action.dim() != 3:
            raise ValueError(
                f"Expected action (B, horizon, action_dim); got {tuple(action.shape)}"
            )
        if self.expert_action_key not in obs_dict:
            raise KeyError(
                f"MSEVerifier: obs_dict missing expert action key "
                f"'{self.expert_action_key}'. Available: {sorted(obs_dict.keys())}"
            )
        expert = obs_dict[self.expert_action_key]
        if not isinstance(expert, torch.Tensor):
            expert = torch.as_tensor(expert)
        expert = expert.to(device=action.device, dtype=action.dtype)
        if expert.shape != action.shape:
            raise ValueError(
                f"Expert action shape {tuple(expert.shape)} does not match "
                f"predicted action shape {tuple(action.shape)}"
            )
        value = -((action - expert) ** 2).mean(dim=(-1, -2))
        if self.noise > 0.0 or self.noise_bias != 0.0:
            perturbation = torch.randn_like(value) * self.noise + self.noise_bias
            value = value + perturbation
        return value
