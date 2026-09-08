"""The fixed goal-region mask, and the image-space corruption that uses it.

WHY IMAGE SPACE. `slot_obs_noise` corrupts `obs_features` -- the encoded vector AFTER the
backbone -- which has no spatial structure, so "noise only where the goal is" cannot be
expressed there. This corruption is applied to the observation IMAGE, before the encoder.
It is therefore a different mechanism from the slot ladder and the two are not comparable.

WHY THE MASK IS A CONSTANT. PushTEnv._setup pins goal_pose = [256, 256, pi/4] for every
episode (feedback_util.GOAL_POSE holds the same value), so the goal T occupies the SAME
pixels in every frame of every episode. The mask is built once; only the noise is redrawn.

THE MARGIN IS IN IMAGE PIXELS, not arena pixels. The arena is 512 px but the observation is
96 px (PushTImageEnv render_size) and the goal T spans only ~26 px of it, so the two
readings of "+10px" differ by 5.3x -- one is a hairline, the other swallows the block.
"""
import numpy as np

ARENA = 512
IMG = 96                       # PushTImageEnv render_size
SCALE = IMG / ARENA
# The T is two convex quads (PushTEnv.add_tee builds shape1 + shape2), not one 8-gon:
# a single loop over all eight vertices traces a self-crossing shape.
T_QUADS = ([0, 1, 2, 3], [4, 5, 6, 7])
# The obs corruption schedule, matching train_pusht_diffusion_search.yaml's
# obs_noise_scheduler (TMRL's VLA schedule). Re-derive `t` if this ever changes.
N_TRAIN_TIMESTEPS, BETA_START, BETA_END = 1000, 1e-4, 0.02


def _alphas_cumprod():
    return np.cumprod(1.0 - np.linspace(BETA_START, BETA_END, N_TRAIN_TIMESTEPS))


def sqrt_alpha_bar(t: int) -> float:
    """sqrt(alpha_bar_t) on the obs schedule -- the fraction of signal retained."""
    return float(np.sqrt(_alphas_cumprod()[int(t)]))


def t_quads_img(pose):
    """A T's two quads in 96px IMAGE coordinates, at an arbitrary pose."""
    from diffusion_policy.env.pusht.feedback_util import keypoints_at_pose
    kp = keypoints_at_pose(np.asarray(pose, dtype=np.float32)) * SCALE
    return [kp[q] for q in T_QUADS]


def goal_quads_img():
    """The goal T's two quads in 96px IMAGE coordinates."""
    from diffusion_policy.env.pusht.feedback_util import GOAL_POSE
    return t_quads_img(GOAL_POSE)


def _seg_dist(px, py, a, b):
    d = b - a
    L2 = float(d @ d)
    t = np.zeros_like(px) if L2 == 0 else np.clip(
        ((px - a[0]) * d[0] + (py - a[1]) * d[1]) / L2, 0.0, 1.0)
    return np.hypot(px - (a[0] + t * d[0]), py - (a[1] + t * d[1]))


def goal_mask(margin_img_px: float = 0.0, size: int = IMG) -> np.ndarray:
    """Boolean (size, size): inside the goal T, or within `margin_img_px` of its boundary.

    Distance-to-polygon rather than a binary dilation, so a sub-pixel margin is exact and
    isotropic -- a 1.9px dilation is not expressible by growing a pixel grid.
    """
    from matplotlib.path import Path
    ys, xs = np.mgrid[0:size, 0:size]
    px, py = xs.ravel() + 0.5, ys.ravel() + 0.5
    pts = np.stack([px, py], axis=1)
    inside = np.zeros(px.shape, dtype=bool)
    dist = np.full(px.shape, np.inf)
    for quad in goal_quads_img():
        q = quad * (size / IMG)
        inside |= Path(q).contains_points(pts)
        for i in range(len(q)):
            dist = np.minimum(dist, _seg_dist(px, py, q[i], q[(i + 1) % len(q)]))
    return (inside | (dist <= margin_img_px)).reshape(size, size)


def goal_mask_excluding_block(base_mask, block_pose, size: int = IMG) -> np.ndarray:
    """`base_mask` with the BLOCK T's own pixels removed, for the block at `block_pose`.

    WHY THIS EXISTS. The goal mask is fixed at goal_pose, and the task is to push the block
    ONTO the goal -- so the closer the episode gets to success the more of the block lies
    inside the mask. At t=800 (sqrt(alpha_bar)=0.039) that means the block is destroyed
    exactly during the endgame, and the policy never sees the configuration it has to solve.
    Subtracting the block leaves the corruption on the goal region proper, which is what
    "hide the goal" was meant to be.

    Only the pixels already inside `base_mask` are tested -- the mask covers ~6% of the
    frame, so this is ~20x less polygon work per sample than rasterising the block over the
    whole image, and this runs in the dataloader for every sample of every epoch.
    """
    from matplotlib.path import Path
    ys, xs = np.nonzero(base_mask)
    if len(xs) == 0:
        return np.asarray(base_mask, dtype=bool)
    pts = np.stack([xs + 0.5, ys + 0.5], axis=1)
    covered = np.zeros(len(pts), dtype=bool)
    for quad in t_quads_img(block_pose):
        covered |= Path(quad * (size / IMG)).contains_points(pts)
    out = np.array(base_mask, dtype=bool, copy=True)
    out[ys[covered], xs[covered]] = False
    return out


def apply_goal_mask_noise(image, mask, t, noise):
    """DDPM forward marginal at `t`, on pixels in [-1,1], inside `mask` only.

    Args:
        image: (..., 3, H, W) in [0, 1] -- what PushTImageDataset emits.
        mask:  (H, W) boolean.
        t:     obs-schedule timestep.
        noise: standard normal, same shape as `image`. Passed in rather than drawn here so
               the caller controls the RNG -- in the dataloader that must be torch's, which
               is re-seeded per worker; numpy's is NOT, and workers would share a stream.
    Returns:
        same shape and range as `image`, clipped to [0, 1].
    """
    a = sqrt_alpha_bar(t) ** 2
    x = image * 2.0 - 1.0
    # float mask, so this broadcasts (H, W) against (..., 3, H, W) identically for numpy
    # and torch. A boolean mask would need different casting rules in each.
    m = mask.to(x.dtype) if hasattr(mask, 'to') else mask.astype(x.dtype)
    noisy = (a ** 0.5) * x + ((1.0 - a) ** 0.5) * noise
    out = x * (1.0 - m) + noisy * m
    out = (out + 1.0) / 2.0
    return out.clamp(0.0, 1.0) if hasattr(out, 'clamp') else out.clip(0.0, 1.0)
