"""Versioned visual-phase verifier. No simulator object poses/contact/success inputs.

Gemini observes real frames once per decision; scoring is a pure tensor function of
that cached assessment and candidate actions. The same phase vector can be stored
with demonstration frames for offline ST training. This module never calls Gemini.
"""
import torch
from diffusion_policy.policy.veritas_verifier import VeritasPixelVerifier, _last_obs_step

VERSION = 'veritas_phase_v1'
PHASES = ('approach', 'grasp', 'lift', 'transport', 'align', 'lower', 'release', 'verify', 'unknown')
# [target_u, target_v, tolerance_px, phase_id, confidence, held, support_contact, stable]
PHASE_DIM = 8


class VeritasPhaseVerifier(VeritasPixelVerifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obs_keys = (self.ee_key, self.targets_key, 'veritas_phase')
        self.version = VERSION

    def components(self, obs_dict, action):
        state = _last_obs_step(obs_dict['veritas_phase']).to(action)
        H = min(action.shape[1], self.horizon or action.shape[1])
        act = action[:, :H]
        tcp = _last_obs_step(obs_dict[self.ee_key]).to(action)[:, :3]
        R, scale, _, _ = self._consts(action)
        traj = tcp[:, None] + torch.cumsum((act[..., :3] * scale) @ R.T, dim=1)
        pix = self.project(traj.reshape(-1, 3)).reshape(-1, H, 2)
        finite = torch.isfinite(pix).all(dim=(1, 2))
        tol = state[:, 2].clamp(min=10, max=80)
        d = torch.linalg.vector_norm(torch.nan_to_num(pix) - state[:, None, :2], dim=-1) / tol[:, None]
        phase = state[:, 3].round().long()
        held = state[:, 5] > 0.5
        supported = state[:, 6] > 0.5
        # Block unsupported phase advancement even if the VLM's phase label contradicts
        # its own predicates. Uncertain grasp is not treated as verified possession.
        requires_held = (phase >= 2) & (phase <= 5)
        unknown = (state[:, 4] < 0.5) | (phase == 8) | (requires_held & ~held)
        unknown |= ((phase == 6) & ~supported)
        # Bounded geometric reward: use the end of the EXECUTED window, not H16 tail.
        geometry = torch.exp(-d[:, -1].clamp(max=20))
        # WidowX dataset: 1=open, 0=closed. Execution thresholds at 0.5.
        opened = (act[..., 6] >= 0.5).to(action.dtype)
        approach_grasp = (phase == 0) | (phase == 1)
        # Closing before arriving at the visual grasp region is penalized; closing
        # near the object is preferred. No claim that the hypothetical close succeeds.
        desired_open = torch.where(approach_grasp[:, None], (d > 1).to(action.dtype),
                                   ((phase == 6) | (phase == 7))[:, None].to(action.dtype))
        grip_cost = (opened - desired_open).abs().mean(dim=1)
        # Vertical progress is observable in TCP coordinates; this does not infer
        # object motion for unexecuted actions.
        dz = traj[:, -1, 2] - tcp[:, 2]
        vertical_cost = torch.where((phase == 2) | (phase == 7), (-dz / .03).clamp(0, 1),
                                    torch.where(phase == 5, (dz / .03).clamp(0, 1), torch.zeros_like(dz)))
        score = geometry - 0.75 * grip_cost - 0.25 * vertical_cost
        # Low-confidence assessments use the versioned original waypoint heuristic,
        # in a bounded scale. A failed API call is an error, never a silent fallback.
        legacy = super().cost(obs_dict, action)
        fallback = torch.exp(-legacy / 60.)
        score = torch.where(unknown, fallback, score)
        score = torch.where(finite, score, torch.full_like(score, -2.))
        return {'score': score, 'geometry': geometry, 'gripper_cost': grip_cost,
                'vertical_cost': vertical_cost, 'fallback': unknown.to(action.dtype),
                'phase': phase, 'old_waypoint_cost': legacy}

    def get_value(self, obs_dict, action):
        with torch.no_grad():
            value = self.components(obs_dict, action)['score']
            if self.normalize == 'fixed':
                value = (value - self.value_mean) / self.value_std
            return value
