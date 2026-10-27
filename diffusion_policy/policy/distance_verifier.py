"""DistanceVerifier — privileged state-based shaped-distance verifier for the L2S
SearchPolicy.

Drop-in replacement for ``DQCVerifier`` / ``RoboMonkeyVerifier`` (same
``get_value(obs_dict, action) -> Tensor`` interface), scoring a candidate action
chunk by the task's shaped pick-and-place distance rather than by a learned Q:

    d_reach = || ee_pos(t+H)  - egg_pos(t+H)    ||
    d_place = || egg_pos(t+H) - basket_pos(t+H) ||
    raw     = -(d_reach + d_place)          # higher is better, matches Q convention

Motivation: the DQC critic is a learned model and is exploitable off-manifold —
random action diversity drives success to 0% because the critic prefers
off-manifold actions over the expert (see the `dqc-q-exploitable-offmanifold`
finding). A closed-form geometric verifier cannot be exploited that way: it has no
parameters to fool, and its ranking is the task objective by construction. What it
gives up is any notion of *reachability* — it will happily reward a chunk that
teleports the gripper through the table — so it is a shaped ORACLE, not a value
function, and should be read as such.

FORWARD MODEL (offline)
-----------------------
Training is OFFLINE: the verifier scores candidates on recorded demo windows, with
no simulator to step. The state after the chunk is therefore *predicted*, not
observed, from the privileged poses already in the obs:

    step_i       = action_pos_scale * a_i[:3]
    ee_pos(t+H)  = ee_pos(t) + sum_{i<H} step_i          # grasp-independent
    held_i       = held_0 * prod_{j<=i} 1[a_j[gripper] > thr]
    egg_pos(t+H) = egg_pos(t) + sum_{i<H} held_i * step_i
    basket       = static

``grasp_mode="chunk"`` (default) is the ``held_i`` form above: the object accrues
travel only over the steps it is actually carried for, and the cumulative product
encodes "released -> stays released". A candidate that opens the gripper half way
through drops the eggplant AT THE POINT OF RELEASE rather than carrying it to the end
of the chunk. ``grasp_mode="obs"`` is the degenerate case ``held_i == held_0``, which
ignores the candidate's own gripper channel entirely and therefore scores a chunk that
releases identically to one that holds — it is kept only as the ablation.

Note this needs no sequential loop: it is a cumprod and a masked sum, both vectorized,
so the gripper-aware form costs the same as the frozen one.

Re-grasping is deliberately NOT modelled. Starting a new grasp mid-chunk would need a
proximity test, and proximity is not grasp success; crediting an unverifiable re-grasp
is precisely the kind of free reward a search policy learns to exploit.

The grasp term is what makes both distances live. With the eggplant frozen, ``d_place``
is constant across every candidate in a window and contributes exactly zero to the
ranking, collapsing the verifier to "reach the eggplant". Note the converse also holds:
while the object is rigidly held, the chunk displacement cancels out of ``d_reach``, so
that term goes constant instead. At any instant essentially one of the two distances
carries the ranking signal — reach before the grasp, place after.

``action_pos_scale`` converts action units into metres of end-effector travel. It is
PER-AXIS, and that is not a refinement — it is required for correctness. Measured on
the sub50 eggplant set the map is roughly ``diag(-0.61, -0.71, +0.73)``: a 180-degree
flip about z between the action frame and the pose frame, on top of the unit scale.
A single pooled scalar averages the two negative axes against the positive one, very
nearly cancels, and reports a near-zero fit (R^2 = 0.03) for a relationship that is
in fact strong (per-axis |r| = 0.79 / 0.88 / 0.88, R^2 ~ 0.70 with the signs right).
Calibrate with ``scripts/simpler/train_search/calibrate_distance_verifier.py``.

The deltas are NOT body-frame: rotating them by the end-effector quaternion before
integrating makes the fit strictly WORSE (R^2 0.13 vs 0.70), so no rotation is applied
despite the ``_b`` suffix on the neighbouring velocity obs suggesting otherwise.

VALUE NORMALIZATION
-------------------
The policy CONDITIONS on these values (they are half of the search context), so
their scale is not cosmetic. Raw values here are negative metres, O(0.1–1.0), and
DQC values were O(0–1) — feeding raw metres into a model tuned for the latter
silently rescales the value-embedding input.

``normalize="fixed"`` (default) applies a dataset-level affine
``(raw - value_mean) / value_std`` with constants from the same calibration script.
Deterministic, resume-safe, and it preserves the cross-window comparisons that a
per-window scheme would destroy. ``normalize="per_window"`` z-scores across the
candidate axis instead: scale-free, but it throws away "this window is far from the
goal" and is only meaningful once there are ≥2 candidates. ``normalize="none"``
returns raw metres and is for debugging only.

Ranking is invariant under every one of these (all are affine with positive scale
within a window), so normalization changes the CONTEXT the policy sees, never which
candidate argmax selects.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch


def _last_obs_step(x: torch.Tensor) -> torch.Tensor:
    """Reduce an obs entry to the CURRENT frame, (B, D).

    Obs arrive as (B, To, D) from the policy and occasionally as (B, D) from ad-hoc
    callers. ``n_obs_steps`` is 1 on this task, but index -1 rather than 0 so the
    verifier stays correct if the policy is ever given a history: the chunk starts at
    the LAST observed frame, not the first.
    """
    if x.dim() == 3:
        return x[:, -1, :]
    if x.dim() == 2:
        return x
    raise ValueError(f"Expected obs entry with 2 or 3 dims, got {tuple(x.shape)}")


class DistanceVerifier:
    """Closed-form shaped-distance verifier over a predicted action chunk.

    Args:
        horizon: how many leading actions of the candidate to integrate. ``None``
            uses the whole chunk. The config predicts 8 and scores all 8, so the
            scored window IS the predicted chunk (unlike the DQC arm, where the
            scored chunk and the executed window differ).
        ee_key / object_key / target_key: obs keys for the gripper, the eggplant and
            the basket. Each is a 7-vector (pos3 + quat4); only pos is read.
        grasp_key: obs key whose sign indicates a closed gripper. Defaults to
            ``last_gripper_action``, which is already in the task shape_meta, so this
            verifier needs NO change to the policy's obs contract and the arm stays
            comparable to the DQC runs. ``binary_contact`` is the better signal (it is
            true contact rather than a commanded gripper) but is not currently in
            shape_meta, so selecting it requires adding the key there first.
        grasp_closed_is_negative: which sign of ``grasp_key`` means "holding". FALSE
            here, and measured rather than assumed: ``last_gripper_action`` on this
            dataset takes values {0.0, 0.996}, so closed is POSITIVE and the natural
            threshold is 0.5. The opposite (Bridge's gripper<0 convention) yields a
            grasped fraction of exactly 0.000, which silently freezes the object and
            makes the placement distance constant across every candidate.
        action_pos_scale: per-axis metres of EE travel per action unit, (3,).
            CALIBRATE IT — see the module docstring.
        w_reach / w_place: weights on the two distance terms.
        normalize / value_mean / value_std: see the module docstring.
    """

    def __init__(
            self,
            horizon: Optional[int] = None,
            ee_key: str = "end_effector_pose",
            object_key: str = "insertive_asset_pose",
            target_key: str = "receptive_asset_pose",
            grasp_key: str = "last_gripper_action",
            grasp_closed_is_negative: bool = False,
            grasp_threshold: float = 0.5,
            grasp_mode: str = "chunk",
            gripper_index: int = 6,
            gripper_closed_threshold: float = 0.5,
            action_pos_scale: Sequence[float] = (1.0, 1.0, 1.0),
            w_reach: float = 1.0,
            w_place: float = 1.0,
            normalize: str = "fixed",
            value_mean: float = 0.0,
            value_std: float = 1.0,
            health_check: bool = True,
        ):
        if grasp_mode not in ("chunk", "obs"):
            raise ValueError(f"grasp_mode must be chunk|obs, got {grasp_mode!r}")
        if normalize not in ("fixed", "per_window", "none"):
            raise ValueError(
                f"normalize must be one of fixed|per_window|none, got {normalize!r}")
        if value_std <= 0:
            raise ValueError(f"value_std must be positive, got {value_std}")
        self.horizon = horizon
        self.ee_key = ee_key
        self.object_key = object_key
        self.target_key = target_key
        self.grasp_key = grasp_key
        self.grasp_closed_is_negative = grasp_closed_is_negative
        self.grasp_threshold = grasp_threshold
        self.grasp_mode = grasp_mode
        self.gripper_index = gripper_index
        self.gripper_closed_threshold = gripper_closed_threshold
        # Accept a scalar for convenience but store (3,): hydra hands this over as a
        # ListConfig, and a bare float would silently re-introduce the pooled-scalar
        # bug the per-axis calibration exists to avoid.
        if isinstance(action_pos_scale, (int, float)):
            action_pos_scale = (float(action_pos_scale),) * 3
        self.action_pos_scale = tuple(float(v) for v in action_pos_scale)
        if len(self.action_pos_scale) != 3:
            raise ValueError(
                f"action_pos_scale must have 3 entries (x, y, z), got "
                f"{self.action_pos_scale}")
        self._scale_cache = None
        self.w_reach = w_reach
        self.w_place = w_place
        self.normalize = normalize
        self.value_mean = value_mean
        self.value_std = value_std
        self._health_check = health_check
        self._checked = False

        if self.action_pos_scale == (1.0, 1.0, 1.0):
            print("[DistanceVerifier] action_pos_scale=(1,1,1) — the UNCALIBRATED "
                  "default. Predicted chunk displacement will be wrong in both scale "
                  "AND sign (the measured map on the eggplant set is negative on x,y). "
                  "Run scripts/simpler/train_search/calibrate_distance_verifier.py.")
        if normalize == "fixed" and value_mean == 0.0 and value_std == 1.0:
            print("[DistanceVerifier] normalize='fixed' with the identity affine "
                  "(mean=0, std=1) — values are raw negative metres. Calibrate "
                  "value_mean/value_std or the policy conditions on an unfamiliar scale.")

    # ------------------------------------------------------------------ internals

    def _scale(self, ref: torch.Tensor) -> torch.Tensor:
        """Per-axis action->metres scale as a (3,) tensor matching ``ref``.

        Cached on (device, dtype) rather than rebuilt per call: get_value runs once per
        candidate per window, so this is on the hot path of every outer step.
        """
        key = (ref.device, ref.dtype)
        if self._scale_cache is None or self._scale_cache[0] != key:
            tensor = torch.tensor(
                self.action_pos_scale, device=ref.device, dtype=ref.dtype)
            self._scale_cache = (key, tensor)
        return self._scale_cache[1]

    def _grasped(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, 1) float mask: is the gripper holding the object at the current frame."""
        g = _last_obs_step(obs_dict[self.grasp_key])
        # take the first channel: last_gripper_action is 1-d, binary_contact may be wider
        g = g[:, :1]
        if self.grasp_closed_is_negative:
            return (g < self.grasp_threshold).to(g.dtype)
        return (g > self.grasp_threshold).to(g.dtype)

    def _raw_value(
            self,
            obs_dict: Dict[str, torch.Tensor],
            action: torch.Tensor,
        ) -> torch.Tensor:
        """Shaped distance after the chunk, (B,). Higher is better."""
        ee_pos = _last_obs_step(obs_dict[self.ee_key])[:, :3]
        obj_pos = _last_obs_step(obs_dict[self.object_key])[:, :3]
        tgt_pos = _last_obs_step(obs_dict[self.target_key])[:, :3]

        H = action.shape[1] if self.horizon is None else min(self.horizon, action.shape[1])
        # Translation channels only: rotation never moves this metric, and the gripper
        # channel moves it only through which steps the object is carried for (below).
        step = action[:, :H, :3] * self._scale(action)                        # (B, H, 3)
        delta = step.sum(dim=1)                                               # (B, 3)
        ee_final = ee_pos + delta            # the gripper's own path is grasp-independent

        held0 = self._grasped(obs_dict)                                       # (B, 1)
        if self.grasp_mode == "obs":
            # Object rigidly attached for the WHOLE chunk if it was held at time t.
            obj_final = obj_pos + held0 * delta
        else:
            # Gripper-gated: the object accrues travel only over the steps it is
            # actually carried for. cumprod encodes "released -> stays released": once
            # the candidate opens the gripper the product is 0 for every later step, so
            # the object stops at the point of release instead of teleporting to the end
            # of the chunk. Re-grasping is deliberately NOT modelled — that needs a
            # proximity test this forward model cannot verify, and crediting an
            # unverifiable re-grasp is exactly the kind of thing a search policy learns
            # to exploit.
            closed = (action[:, :H, self.gripper_index:self.gripper_index + 1]
                      > self.gripper_closed_threshold).to(action.dtype)       # (B, H, 1)
            held = held0.unsqueeze(1) * torch.cumprod(closed, dim=1)          # (B, H, 1)
            obj_final = obj_pos + (step * held).sum(dim=1)

        d_reach = torch.linalg.vector_norm(ee_final - obj_final, dim=-1)
        d_place = torch.linalg.vector_norm(obj_final - tgt_pos, dim=-1)
        return -(self.w_reach * d_reach + self.w_place * d_place)

    def _release_fraction(self, action: torch.Tensor, H: int) -> torch.Tensor:
        """Fraction of candidates whose gripper is closed at step 0 but opens later.

        This is the exact measure of how much ``grasp_mode="chunk"`` can differ from
        ``"obs"``: the two forms agree on every chunk whose gripper channel never
        toggles. If this reads ~0 the modes are equivalent in practice.
        """
        g = action[:, :H, self.gripper_index] > self.gripper_closed_threshold
        return (g[:, 0] & (~g).any(dim=1)).to(action.dtype).mean()

    def _run_health_check(self, obs_dict, action, raw) -> None:
        """One-time sanity print: the scales that silently break this verifier."""
        self._checked = True
        with torch.no_grad():
            ee = _last_obs_step(obs_dict[self.ee_key])[:, :3]
            obj = _last_obs_step(obs_dict[self.object_key])[:, :3]
            tgt = _last_obs_step(obs_dict[self.target_key])[:, :3]
            H = action.shape[1] if self.horizon is None else min(self.horizon, action.shape[1])
            step = action[:, :H, :3] * self._scale(action)
            print(
                "[DistanceVerifier] health check\n"
                f"   scored horizon      : {H} of {action.shape[1]} predicted actions\n"
                f"   |ee - obj|   (t)    : {torch.linalg.vector_norm(ee - obj, dim=-1).mean():.4f} m\n"
                f"   |obj - tgt|  (t)    : {torch.linalg.vector_norm(obj - tgt, dim=-1).mean():.4f} m\n"
                f"   per-step |dpos|     : {torch.linalg.vector_norm(step, dim=-1).mean():.5f} m "
                f"(x{H} = {torch.linalg.vector_norm(step.sum(1), dim=-1).mean():.4f} m of chunk travel)\n"
                f"   grasped fraction    : {self._grasped(obs_dict).mean():.3f} "
                f"(key={self.grasp_key}, mode={self.grasp_mode})\n"
                # In chunk mode this is the ONLY thing distinguishing the two modes: if
                # no candidate ever opens the gripper mid-chunk, chunk and obs modes are
                # numerically identical and the extra machinery is inert.
                f"   release mid-chunk   : {self._release_fraction(action, H):.3f} "
                f"of candidates\n"
                f"   raw value           : mean {raw.mean():.4f}  min {raw.min():.4f}  "
                f"max {raw.max():.4f}\n"
                f"   normalize           : {self.normalize} "
                f"(mean={self.value_mean}, std={self.value_std})")
            travel = torch.linalg.vector_norm(step.sum(1), dim=-1).mean()
            reach = torch.linalg.vector_norm(ee - obj, dim=-1).mean()
            if travel < 0.02 * reach:
                print("[DistanceVerifier] WARNING: chunk travel is <2% of the current "
                      "reach distance. Candidates are nearly indistinguishable under this "
                      "metric, so the search context carries almost no ranking signal — "
                      "action_pos_scale is very likely wrong (uncalibrated).")

    # ------------------------------------------------------------------ interface

    def get_value(
            self,
            obs_dict: Dict[str, torch.Tensor],
            action: torch.Tensor,
        ) -> torch.Tensor:
        """Score one candidate chunk per batch element. ``action`` is (B, H, Da) in RAW
        action units (the search loop carries candidates unnormalized); returns (B,).
        """
        with torch.no_grad():
            raw = self._raw_value(obs_dict, action)
            if self._health_check and not self._checked:
                self._run_health_check(obs_dict, action, raw)
            if self.normalize == "fixed":
                return (raw - self.value_mean) / self.value_std
            if self.normalize == "per_window":
                # single candidate -> std is undefined; centring alone would zero it out
                if raw.numel() < 2:
                    return raw - raw
                return (raw - raw.mean()) / raw.std().clamp_min(1e-6)
            return raw
