"""Veritas pixel-waypoint verifier for SimplerEnv candidate action chunks.

The score is upstream Veritas's candidate cost (veritas/src/agent/eval_verifier_utils.py,
``select_best_action_chunk``, Veritas-px branch), negated so that higher is better:

1. integrate the chunk's translation channels from the current world TCP position,
   rotating base-frame deltas into world with ``R_world_from_base`` (optionally scaled
   per axis by ``action_pos_scale``; upstream is 1:1),
2. project the endpoint into the fixed 3rd-person camera,
3. skip the leading upcoming waypoints already within ``tol_px`` of the endpoint, then
   sum the pixel distances to the remaining ones with weights ``alpha ** k``
   (``lookahead_k`` waypoints, ``lookahead_alpha`` decay). All satisfied -> cost 0.

Once the plan is done (``n_valid == 0``), cost is zero in both training and evaluation.
This is waypoint completion, not task success. For reproducing older evaluations,
``plan_done_mode='source_object_legacy'`` enables upstream's distance-to-source
fallback when the live source pixel is appended to the targets vector. Training
snapshots lack that pixel, so the legacy mode is not a matched training/eval setup.

The upcoming waypoints and tolerance are the live tracker's state, which advances only
on EXECUTED env steps (never on candidates). They reach the verifier through the obs
as a ``veritas_targets`` vector (``env/simpler/veritas/tracker.targets_vector``): at
eval the env wrapper writes it, in training a precomputed per-frame array does.

``action_pos_scale``: ``None`` reproduces upstream exactly. The simulator's PD
controller realises only ~70-85% of a commanded delta (fit on the strict eggplant
data at H=8: ~(0.72, 0.74, 0.87) in base frame), so a per-axis scale corrects the mean
endpoint error; ``scripts/simpler/train_search/calibrate_distance_verifier.py --fit
horizon_cumulative --frame base`` measures it.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from diffusion_policy.env.simpler.veritas.camera import CameraModel
from diffusion_policy.env.simpler.veritas.tracker import (LOOKAHEAD_K, TARGETS_DIM,
                                                           TARGETS_DIM_WITH_SOURCE)


def _last_obs_step(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x[:, -1, :]
    if x.dim() == 2:
        return x
    raise ValueError(f'Expected obs entry with 2 or 3 dims, got {tuple(x.shape)}')


class VeritasPixelVerifier:
    """Duck-typed verifier: ``get_value(obs_dict, action (B, H, 7) raw) -> (B,)``."""

    def __init__(
            self,
            camera_file: str,
            action_pos_scale: Optional[Sequence[float]] = None,
            lookahead_k: int = LOOKAHEAD_K,
            lookahead_alpha: float = 0.5,
            default_tol_px: float = 60.0,
            horizon: Optional[int] = None,
            ee_key: str = 'end_effector_pose',
            targets_key: str = 'veritas_targets',
            image_obs_key: str = 'agentview_image',
            normalize: str = 'none',
            value_mean: float = 0.0,
            value_std: float = 1.0,
            unprojectable_cost: float = 1e4,
            health_check: bool = True,
            plan_done_mode: str = 'zero',
        ):
        if plan_done_mode not in ('zero', 'source_object_legacy'):
            raise ValueError('plan_done_mode must be zero|source_object_legacy')
        self.plan_done_mode = plan_done_mode
        if horizon is not None and horizon < 1:
            raise ValueError('horizon must be positive')
        if normalize not in ('fixed', 'none'):
            raise ValueError(f'normalize must be fixed|none, got {normalize!r}')
        if value_std <= 0:
            raise ValueError(f'value_std must be positive, got {value_std}')
        if lookahead_k < 1 or 2 + 2 * lookahead_k > TARGETS_DIM:
            raise ValueError(f'lookahead_k must be in [1, {(TARGETS_DIM - 2) // 2}]')
        self.camera = CameraModel.load(camera_file)
        self.camera_file = str(camera_file)
        if action_pos_scale is not None:
            if isinstance(action_pos_scale, (int, float)):
                action_pos_scale = (float(action_pos_scale),) * 3
            action_pos_scale = tuple(float(v) for v in action_pos_scale)
            if len(action_pos_scale) != 3:
                raise ValueError(f'action_pos_scale must have 3 entries, got {action_pos_scale}')
        self.action_pos_scale = action_pos_scale
        self.lookahead_k = int(lookahead_k)
        self.lookahead_alpha = float(lookahead_alpha)
        self.default_tol_px = float(default_tol_px)
        self.horizon = horizon
        self.ee_key = ee_key
        self.targets_key = targets_key
        self.image_obs_key = image_obs_key
        # obs keys this verifier reads beyond the policy's own normalizer keys; the eval
        # loop keeps these when it filters obs_dict.
        self.obs_keys = (ee_key, targets_key)
        self.normalize = normalize
        self.value_mean = float(value_mean)
        self.value_std = float(value_std)
        self.unprojectable_cost = float(unprojectable_cost)
        self._health_check = health_check
        self._checked = False
        self._const_cache = None

    # ------------------------------------------------------------------ internals

    def _consts(self, ref: torch.Tensor):
        key = (ref.device, ref.dtype)
        if self._const_cache is None or self._const_cache[0] != key:
            kw = dict(device=ref.device, dtype=ref.dtype)
            R = torch.tensor(self.camera.R_world_from_base, **kw)
            scale = (torch.ones(3, **kw) if self.action_pos_scale is None
                     else torch.tensor(self.action_pos_scale, **kw))
            K = torch.tensor(self.camera.K, **kw)
            T = torch.tensor(self.camera.T_wc, **kw)
            self._const_cache = (key, (R, scale, K, T))
        return self._const_cache[1]

    def endpoints(self, obs_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """World xyz of the end effector after the chunk, (B, 3)."""
        R, scale, _, _ = self._consts(action)
        tcp = _last_obs_step(obs_dict[self.ee_key])[:, :3].to(action.dtype)
        H = action.shape[1] if self.horizon is None else min(self.horizon, action.shape[1])
        delta_base = (action[:, :H, :3] * scale).sum(dim=1)                 # (B, 3)
        return tcp + delta_base @ R.T

    def project(self, world_xyz: torch.Tensor) -> torch.Tensor:
        """(B, 3) world -> (B, 2) pixels; NaN behind the camera."""
        _, _, K, T = self._consts(world_xyz)
        cam = world_xyz @ T[:3, :3].T + T[:3, 3]
        z = cam[:, 2]
        safe = z > 1e-6
        zs = torch.where(safe, z, torch.ones_like(z))
        u = (K[0, 0] * cam[:, 0] + K[0, 2] * zs) / zs
        v = (K[1, 1] * cam[:, 1] + K[1, 2] * zs) / zs
        uv = torch.stack([u, v], dim=-1)
        return torch.where(safe[:, None], uv, torch.full_like(uv, float('nan')))

    def cost(self, obs_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """Upstream lookahead cost, (B,). Lower is better."""
        targets = _last_obs_step(obs_dict[self.targets_key]).to(action.dtype)  # (B, D)
        n_valid = targets[:, 0].round().long().clamp(0, self.lookahead_k)       # (B,)
        tol = torch.where(targets[:, 1] > 0, targets[:, 1],
                          torch.full_like(targets[:, 1], self.default_tol_px))
        wps = targets[:, 2:2 + 2 * self.lookahead_k].reshape(-1, self.lookahead_k, 2)

        pix = self.project(self.endpoints(obs_dict, action))                    # (B, 2)
        bad = ~torch.isfinite(pix).all(dim=-1)
        pix = torch.nan_to_num(pix)
        d = torch.linalg.vector_norm(pix[:, None, :] - wps, dim=-1)             # (B, K)
        k = torch.arange(self.lookahead_k, device=action.device)
        valid = k[None, :] < n_valid[:, None]                                   # (B, K)
        # target_idx = number of LEADING valid waypoints within tol (upstream's while
        # loop stops at the first one outside tol).
        within = (d < tol[:, None]) & valid
        leading = torch.cumprod(within.to(action.dtype), dim=1)                 # (B, K)
        target_idx = leading.sum(dim=1).long()
        remaining = valid & (k[None, :] >= target_idx[:, None])
        w = self.lookahead_alpha ** (k[None, :] - target_idx[:, None]).to(action.dtype)
        cost = (d * w * remaining.to(action.dtype)).sum(dim=1)                  # 0 if none remain
        scored = n_valid > 0
        if (self.plan_done_mode == 'source_object_legacy'
                and targets.shape[1] >= TARGETS_DIM_WITH_SOURCE):
            # plan done -> upstream's source-object distance (pixels rounded and clipped
            # to the image, as compute_distance_to_target does)
            src = targets[:, TARGETS_DIM:TARGETS_DIM + 2]
            fallback = (n_valid == 0) & (src >= 0).all(dim=-1)
            h, w_img = self.camera.image_hw
            lim = torch.tensor([w_img - 1, h - 1], dtype=action.dtype, device=action.device)
            pe = torch.minimum(torch.round(pix).clamp(min=0), lim)
            ps = torch.minimum(src.clamp(min=0), lim)
            cost = torch.where(fallback, torch.linalg.vector_norm(pe - ps, dim=-1), cost)
            scored = scored | fallback
        cost = torch.where(bad & scored, torch.full_like(cost, self.unprojectable_cost), cost)
        return cost

    def _run_health_check(self, obs_dict, action, raw):
        self._checked = True
        with torch.no_grad():
            targets = _last_obs_step(obs_dict[self.targets_key])
            n_valid = targets[:, 0]
            pix = self.project(self.endpoints(obs_dict, action))
            H = action.shape[1] if self.horizon is None else min(self.horizon, action.shape[1])
            print('[VeritasPixelVerifier] health check\n'
                  f'   camera             : {self.camera_file} ({self.camera.image_hw[1]}x{self.camera.image_hw[0]})\n'
                  f'   scored horizon     : {H} of {action.shape[1]} predicted actions\n'
                  f'   action_pos_scale   : {self.action_pos_scale or "none (1:1, upstream)"}\n'
                  f'   lookahead          : k={self.lookahead_k} alpha={self.lookahead_alpha}\n'
                  f'   targets n_valid    : mean {n_valid.float().mean():.2f} '
                  f'(fraction done {(n_valid == 0).float().mean():.2f}; '
                  f'plan_done_mode={self.plan_done_mode})\n'
                  f'   endpoint pixel     : finite {torch.isfinite(pix).all(-1).float().mean():.2f}, '
                  f'mean {pix.nanmean(0).tolist()}\n'
                  f'   raw value (-cost)  : mean {raw.mean():.2f} min {raw.min():.2f} max {raw.max():.2f}\n'
                  f'   normalize          : {self.normalize} (mean={self.value_mean}, std={self.value_std})')

    # ------------------------------------------------------------------ interface

    def get_value(self, obs_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            raw = -self.cost(obs_dict, action)
            if self._health_check and not self._checked:
                self._run_health_check(obs_dict, action, raw)
            if self.normalize == 'fixed':
                return (raw - self.value_mean) / self.value_std
            return raw
