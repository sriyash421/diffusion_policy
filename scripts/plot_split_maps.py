"""Where the T starts for each split: one panel per manifest, every episode's T start pose in the
512x512 arena (y down, as rendered), marker + colour by role.

    python scripts/plot_split_maps.py [OUT]            # held-out quadrants; default docs/reports/figures/far50_2026-10-06/split_maps.png
    python scripts/plot_split_maps.py --interp OUT     # random 50 / 75 train and their BC-failure test sets
"""
import argparse, json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from make_geometric_splits import block_starts, QUADRANT_SIDES, WS

SPLITS = [('pusht_blockquad_bottomleft_train50_far_valpool50', 'bottomleft', 'blq50far'),
          ('pusht_blockquad_topleft_train50_far_valpool50', 'topleft', 'tlq50far'),
          ('pusht_blockquad_bottomright_train50_far_valpool50', 'bottomright', 'brq50far')]
# (manifest, title): the ST test sets of the random-train splits (make_interp_split.py)
INTERP = [('pusht_rand50_train50_bccov30_25x25', 'rand50: 50 random train starts; test = 25 BC failures + 25 others'),
          ('pusht_rand75_train75_bccov30_25x25', 'rand75: 75 random train starts; test = 25 BC failures + 25 others')]
GOAL = (256, 256)   # PushTEnv goal_pose xy
ROLES = {'train': ('#0072B2', 'o', 'train'), 'val': ('#E69F00', '^', 'val = held-out train'),
         'test': ('#D55E00', 'x', 'test (held-out quadrant)')}


def draw(ax, xy, name, roles, fails=()):
    """Every episode's T start by its role in manifest `name`; `fails` (test starts) drawn apart."""
    m = json.load(open(ROOT / 'diffusion_policy/config/splits' / f'{name}.json'))
    used = set()
    for role, (c, mk, lab) in roles.items():
        idx = np.array([i for i in m[role] if role != 'test' or i not in fails], dtype=int)
        used |= set(m[role])
        if len(idx):
            ax.scatter(xy[idx, 0], xy[idx, 1], c=c, marker=mk, s=38, lw=1.3, label=f'{lab} ({len(idx)})', zorder=3)
    if fails:
        idx = np.array(sorted(fails), dtype=int)
        ax.scatter(xy[idx, 0], xy[idx, 1], c='#CC79A7', marker='D', s=40, edgecolor='black', lw=0.6,
                   label=f'test, BC failure ({len(idx)})', zorder=3)
    rest = np.array(sorted(set(range(len(xy))) - used), dtype=int)
    if len(rest):
        ax.scatter(xy[rest, 0], xy[rest, 1], c='#BBBBBB', marker='.', s=18, label=f'unused ({len(rest)})', zorder=2)
    ax.scatter(*GOAL, marker='*', s=220, c='#009E73', edgecolor='black', lw=0.6, label='goal pose', zorder=4)
    ax.axhline(WS / 2, color='k', lw=0.6, ls=':'); ax.axvline(WS / 2, color='k', lw=0.6, ls=':')
    ax.set_xlim(0, WS); ax.set_ylim(WS, 0); ax.set_aspect('equal'); ax.set_xlabel('T start x (px)')
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?', default=None)
    ap.add_argument('--interp', action='store_true')
    args = ap.parse_args()
    _, xy = block_starts(ROOT / 'data/pusht_cchi_v7_replay.zarr')
    if args.interp:
        sp = ROOT / 'diffusion_policy/config/splits'
        panels = [(n, t) for n, t in INTERP if (sp / f'{n}.json').is_file()]
        fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.6), squeeze=False)
        roles = {**ROLES, 'test': ('#D55E00', 'x', 'test, other')}
        for ax, (name, title) in zip(axes[0], panels):
            side = sp / f'{name}.bcfail.json'
            fails = set(json.loads(side.read_text())['bc_fail_test']) if side.is_file() else set()
            draw(ax, xy, name, roles, fails)
            ax.set_title(title, fontsize=9); ax.legend(fontsize=7, loc='upper right')
        axes[0, 0].set_ylabel('T start y (px, down)')
        fig.suptitle('PushT interpolation split: T start position per episode (206 demos)', fontsize=11)
        default = ROOT / 'docs/reports/figures/interp_rand50/split_maps.png'
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
        for ax, (name, quad, tag) in zip(axes, SPLITS):
            hx, hy = QUADRANT_SIDES[quad]
            ax.add_patch(Rectangle((WS / 2 if hx else 0, WS / 2 if hy else 0), WS / 2, WS / 2,
                                   facecolor='#D55E00', alpha=0.08, edgecolor='none'))
            m = draw(ax, xy, name, ROLES)
            ax.set_title(f'{tag}: {quad} quadrant held out\n'
                         f'train = 50 farthest from it (min {m["derivation"]["train_quad_dist_min_px"]:.0f} px)', fontsize=10)
            ax.legend(fontsize=7, loc='upper right' if quad != 'topright' else 'lower left')
        axes[0].set_ylabel('T start y (px, down)')
        fig.suptitle('PushT geometric splits: T start position per episode (206 demos)', fontsize=11)
        default = ROOT / 'docs/reports/figures/far50_2026-10-06/split_maps.png'
    fig.tight_layout()
    out = pathlib.Path(args.out) if args.out else default
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=140); print('wrote', out)


if __name__ == '__main__':
    main()
