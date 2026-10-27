"""Where the T starts for each held-out-quadrant split: one panel per manifest, every
episode's T start pose in the 512x512 arena (y down, as rendered), marker + colour by role.

    python scripts/plot_split_maps.py   # docs/reports/figures/far50_2026-10-06/split_maps.png
"""
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from make_geometric_splits import block_starts, QUADRANT_SIDES, WS

SPLITS = [('pusht_blockquad_bottomleft_train50_far', 'bottomleft', 'blq50far'),
          ('pusht_blockquad_topleft_train50_far_valpool19', 'topleft', 'tlq50far'),
          ('pusht_blockquad_bottomright_train50_far_valpool19', 'bottomright', 'brq50far')]
GOAL = (256, 256)   # PushTEnv goal_pose xy
ROLES = {'train': ('#0072B2', 'o', 'train'), 'val': ('#E69F00', '^', 'val'),
         'test': ('#D55E00', 'x', 'test (held-out quadrant)')}


def main():
    _, xy = block_starts(ROOT / 'data/pusht_cchi_v7_replay.zarr')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, (name, quad, tag) in zip(axes, SPLITS):
        m = json.load(open(ROOT / 'diffusion_policy/config/splits' / f'{name}.json'))
        hx, hy = QUADRANT_SIDES[quad]
        ax.add_patch(Rectangle((WS / 2 if hx else 0, WS / 2 if hy else 0), WS / 2, WS / 2,
                               facecolor='#D55E00', alpha=0.08, edgecolor='none'))
        used = set()
        for role, (c, mk, lab) in ROLES.items():
            idx = np.array(m[role], dtype=int); used |= set(idx.tolist())
            ax.scatter(xy[idx, 0], xy[idx, 1], c=c, marker=mk, s=38, lw=1.3,
                       label=f'{lab} ({len(idx)})', zorder=3)
        rest = np.array(sorted(set(range(len(xy))) - used), dtype=int)
        if len(rest):
            ax.scatter(xy[rest, 0], xy[rest, 1], c='#BBBBBB', marker='.', s=18,
                       label=f'unused ({len(rest)})', zorder=2)
        ax.scatter(*GOAL, marker='*', s=220, c='#009E73', edgecolor='black', lw=0.6,
                   label='goal pose', zorder=4)
        ax.axhline(WS / 2, color='k', lw=0.6, ls=':'); ax.axvline(WS / 2, color='k', lw=0.6, ls=':')
        ax.set_xlim(0, WS); ax.set_ylim(WS, 0); ax.set_aspect('equal')
        ax.set_title(f'{tag}: {quad} quadrant held out\n'
                     f'train = 50 farthest from it (min {m["derivation"]["train_quad_dist_min_px"]:.0f} px)', fontsize=10)
        ax.set_xlabel('T start x (px)'); ax.legend(fontsize=7, loc='upper right' if quad != 'topright' else 'lower left')
    axes[0].set_ylabel('T start y (px, down)')
    fig.suptitle('PushT geometric splits: T start position per episode (206 demos)', fontsize=11)
    fig.tight_layout()
    out = ROOT / 'docs/reports/figures/far50_2026-10-06/split_maps.png'
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=140); print('wrote', out)


if __name__ == '__main__':
    main()
