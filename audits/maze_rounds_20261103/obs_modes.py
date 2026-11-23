"""Observation modes for the ST policy's encoder input (rounds.OBS_MODE; asteroid.py --obs). Applied in RoundsPolicy._obs_features to
every observation the policy gets (context tokens, the labelled / demo attempt, rollouts, eval). The env, the success test, the expert
and the feedback still use the true state.

  none    unchanged
  window  ASTEROID's partial view (procgen_env.py, visibility 7): agent_pos kept; of the (..., 3, H, W) wall / goal / agent image only the
          7 x 7 window around the agent's rounded cell (radius 3, clipped at the border) is shown, every plane is -1 outside it, so the
          goal plane shows the goal only when it is inside the window; goal_pos = 0
  ddpm    the full image (scaled to [-1, 1]) replaced by sqrt(abar_t) x + sqrt(1 - abar_t) eps (DDPM, T = 1000, beta 1e-4 -> 0.02 linear,
          the schedule of asteroid.noisy_obs), fresh eps at every call (train and test); agent_pos kept; goal_pos = 0
"""
import torch

RADIUS = 3
_ABAR = None


def abar(t):
    global _ABAR
    if _ABAR is None:
        betas = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64)
        _ABAR = torch.cumprod(1 - betas, 0)
    return float(_ABAR[int(t)])


def window_mask(pos, H, W, radius=RADIUS):
    """(..., H, W) bool: cells within `radius` (Chebyshev) of round(pos) (..., 2) = (x, y)."""
    p = torch.round(pos).long()
    ys = torch.arange(H, device=pos.device); xs = torch.arange(W, device=pos.device)
    my = (ys - p[..., 1:2]).abs() <= radius; mx = (xs - p[..., 0:1]).abs() <= radius
    return my[..., :, None] & mx[..., None, :]


def apply_obs(obs, mode, t=400, generator=None):
    if mode in (None, 'none'):
        return obs
    img = obs['image']; H, W = img.shape[-2:]
    if mode == 'window':
        m = window_mask(obs['agent_pos'], H, W)[..., None, :, :]
        img = torch.where(m, img, torch.full_like(img, -1.))
    elif mode == 'ddpm':
        ab = abar(t); x = img * 2 - 1
        eps = torch.randn(img.shape, device=img.device, dtype=img.dtype, generator=generator)
        img = ab ** .5 * x + (1 - ab) ** .5 * eps
    else:
        raise ValueError(mode)
    return {**obs, 'image': img, 'goal_pos': torch.zeros_like(obs['goal_pos'])}
