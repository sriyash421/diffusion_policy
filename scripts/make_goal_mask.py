"""Build the fixed goal-region mask for the part-3 goal-only obs corruption, and render it.

WHY A FIXED MASK. PushTEnv._setup pins goal_pose = [256, 256, pi/4] for every episode
(feedback_util.GOAL_POSE hardcodes the same value), so the goal T occupies the SAME pixels
in every frame of every episode. The mask is therefore computed once here and saved; only
the noise is drawn fresh at train time.

THE MARGIN IS AMBIGUOUS BY A FACTOR OF 5.3. The arena is 512 px but the observation image
is 96 px (PushTImageEnv render_size), and the goal T spans only ~26 px of it. So "+10 px"
is either a hairline (10 arena px = 1.9 image px) or a near-doubling of the masked area
(10 image px = 53 arena px). This renders both so the choice is made on the picture.

    python scripts/make_goal_mask.py                 # -> media/goal_mask/
"""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import zarr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from diffusion_policy.env.pusht.feedback_util import GOAL_POSE, keypoints_at_pose

ARENA, IMG = 512, 96
SCALE = IMG / ARENA
# The T is two convex quads (PushTEnv.add_tee builds shape1 + shape2), not one 8-gon:
# a single loop over all eight vertices draws a self-crossing shape.
T_QUADS = ([0, 1, 2, 3], [4, 5, 6, 7])
INK, MUTED = '#1a1a19', '#8a8880'


def goal_quads_img():
    """The goal T's two quads, in 96px IMAGE pixel coordinates."""
    kp = keypoints_at_pose(np.asarray(GOAL_POSE, dtype=np.float32)) * SCALE
    return [kp[q] for q in T_QUADS]


def _seg_dist(px, py, a, b):
    """Distance from points (px,py) to segment a->b."""
    d = b - a
    L2 = float(d @ d)
    t = np.zeros_like(px) if L2 == 0 else np.clip(((px - a[0]) * d[0] + (py - a[1]) * d[1]) / L2, 0, 1)
    return np.hypot(px - (a[0] + t * d[0]), py - (a[1] + t * d[1]))


def goal_mask(margin_img_px):
    """Boolean (96,96): inside the goal T, or within `margin_img_px` of its boundary.

    Distance to the polygon rather than a binary dilation, so the margin is exact and
    isotropic at sub-pixel widths -- a 1.9px dilation is not expressible on a pixel grid.
    """
    from matplotlib.path import Path
    ys, xs = np.mgrid[0:IMG, 0:IMG]
    px, py = xs.ravel() + 0.5, ys.ravel() + 0.5
    inside = np.zeros(px.shape, dtype=bool)
    dist = np.full(px.shape, np.inf)
    for quad in goal_quads_img():
        inside |= Path(quad).contains_points(np.stack([px, py], 1))
        for i in range(len(quad)):
            dist = np.minimum(dist, _seg_dist(px, py, quad[i], quad[(i + 1) % len(quad)]))
    return (inside | (dist <= margin_img_px)).reshape(IMG, IMG)


def ddpm_abar(t, T=1000, b0=1e-4, b1=0.02):
    """sqrt(alpha_bar) at timestep t on the obs schedule (train_pusht_diffusion_search)."""
    betas = np.linspace(b0, b1, T)
    return float(np.cumprod(1.0 - betas)[t])


def apply(img01, mask, t, rng):
    """DDPM forward marginal at t, on pixels in [-1,1], inside `mask` only."""
    x = img01 * 2.0 - 1.0
    a = ddpm_abar(t)
    noisy = np.sqrt(a) * x + np.sqrt(1.0 - a) * rng.standard_normal(x.shape)
    out = np.where(mask[..., None], noisy, x)
    return np.clip((out + 1.0) / 2.0, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr', default='data/pusht_cchi_v7_replay.zarr')
    ap.add_argument('--outdir', default='media/goal_mask')
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--margins', type=float, nargs='+', default=[0, 5, 10],
                    help='mask margins in IMAGE px (96px obs). 10 arena px = 1.875 image px.')
    args = ap.parse_args()
    out = pathlib.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    img = np.asarray(zarr.open(args.zarr, 'r')['data/img'][args.frame]).astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    rng = np.random.default_rng(0)

    # `--margins` in IMAGE px (the 96px observation), since that is the space the mask
    # lives in. The bare T and the two readings of "+10px" are the default context.
    MARGINS = [(f'T + {m:g} image px' if m else 'T only (+0)', float(m))
               for m in args.margins]
    LEVELS = [800, 400]

    fig, axes = plt.subplots(len(MARGINS), 2 + len(LEVELS),
                             figsize=(3.0 * (2 + len(LEVELS)), 3.05 * len(MARGINS)))
    for r, (label, m) in enumerate(MARGINS):
        mask = goal_mask(m)
        np.save(out / f'mask_{label.replace(" ", "").replace("+","p")}.npy', mask)
        axes[r, 0].imshow(img); axes[r, 0].set_ylabel(label, fontsize=10, color=INK)
        if r == 0: axes[r, 0].set_title('observation', fontsize=10, color=INK)
        axes[r, 1].imshow(img); axes[r, 1].imshow(mask, cmap='Reds', alpha=0.45)
        if r == 0: axes[r, 1].set_title('mask overlay', fontsize=10, color=INK)
        axes[r, 1].text(0.03, 0.97, f'{mask.mean()*100:.1f}% of pixels',
                        transform=axes[r, 1].transAxes, va='top', fontsize=8, color=INK,
                        bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none'))
        for c, t in enumerate(LEVELS):
            axes[r, 2 + c].imshow(apply(img, mask, t, rng))
            if r == 0:
                axes[r, 2 + c].set_title(f't = {t}   ' + r'$\sqrt{\bar\alpha}$' +
                                         f' = {np.sqrt(ddpm_abar(t)):.3f}',
                                         fontsize=10, color=INK)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color(MUTED)
    fig.suptitle('Part 3: goal-only observation corruption — mask extent x noise level\n'
                 'noise is the DDPM forward marginal on pixels in [-1,1], inside the mask only',
                 fontsize=12, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = out / 'goal_mask_variants.png'
    fig.savefig(png, dpi=150, facecolor='white'); plt.close(fig)
    print(f'wrote {png}')
    for label, m in MARGINS:
        print(f'  {label:<18} margin {m:5.2f} image px  -> {goal_mask(m).mean()*100:5.1f}% of the frame')


if __name__ == '__main__':
    main()
