"""Train vs test distribution for the t_goal-moving split (pusht_seed42_train176).

    python scripts/plot_mv_split_shift.py --out analysis/mv_split_shift.png

WHY THIS SPLIT NEEDS THE CHECK MORE THAN THE OTHERS. Every other manifest in config/splits/
draws its test episodes from a seeded permutation, so the two sides are exchangeable by
construction and a shift can only be sampling noise. This one SELECTS its 30 test episodes --
`make_moving_ratio_split.py` picks the subset whose moving-window counts sum nearest to
total*20/120 -- and a rule that selects on a per-episode statistic can bias whatever that
statistic correlates with. This figure is what says whether it did.

Reads block_pos / agent_pos / episode_ends only, never the 2.8 GB image array, so it is safe
on a login node.

Colour is paired with line style AND a direct end-label on every curve, so train/test stay
distinguishable in greyscale.
"""
import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffusion_policy.env.pusht.feedback_util import (  # noqa: E402
    compute_feedback_from_pose, t_goal_distance)
from scripts.moving_transitions_util import moving_frames  # noqa: E402

C_TRAIN, C_TEST = '#2a78d6', '#eb6834'
INK, MUTED = '#1a1a19', '#8a8880'


def ecdf(ax, a, b, title, xlabel, note=''):
    for v, c, ls, lab in ((a, C_TRAIN, '-', 'train (176)'), (b, C_TEST, (0, (5, 2)), 'test (30)')):
        x = np.sort(v)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, color=c, ls=ls, lw=1.8, where='post')
        ax.annotate(lab, xy=(x[-1], 1.0), xytext=(-2, -12), textcoords='offset points',
                    fontsize=7.5, color=c, ha='right', weight='bold')
    from scipy import stats
    p = stats.ks_2samp(a, b).pvalue
    flag = '  ← SHIFTED' if p < 0.05 else ''
    ax.set_title(f'{title}\ntrain {a.mean():.1f}±{a.std():.1f}   test {b.mean():.1f}±{b.std():.1f}'
                 f'   KS p={p:.3f}{flag}', fontsize=9, color=INK, loc='left')
    ax.set_xlabel(xlabel, fontsize=8, color=INK)
    ax.set_ylabel('cumulative fraction', fontsize=8, color=INK)
    ax.tick_params(labelsize=7, colors=MUTED)
    if note:
        ax.text(0.02, 0.06, note, transform=ax.transAxes, fontsize=7, color=MUTED)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr', default='data/pusht_cchi_v7_replay.zarr')
    ap.add_argument('--split', default='diffusion_policy/config/splits/pusht_seed42_train176.json')
    ap.add_argument('--out', default='analysis/mv_split_shift.png')
    args = ap.parse_args()

    g = zarr.open(str(ROOT / args.zarr), 'r')
    ends = np.asarray(g['meta/episode_ends'])
    bp, agp = np.asarray(g['data/block_pos']), np.asarray(g['data/agent_pos'])
    starts = np.concatenate([[0], ends[:-1]])
    man = json.loads((ROOT / args.split).read_text())
    tr, te = man['train'], man['test']

    L = {e: int(ends[e]) - int(starts[e]) for e in range(len(ends))}
    MV, TOT = {}, {}
    for e in range(len(ends)):
        k, t = moving_frames(bp, ends, e)
        MV[e], TOT[e] = len(k), t
    d0 = lambda e: float(t_goal_distance(compute_feedback_from_pose(   # noqa: E731
        bp[int(starts[e])][None].astype(np.float32)))[0])

    get = lambda idxs, f: np.array([f(e) for e in idxs])               # noqa: E731
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))

    ecdf(axes[0, 0], get(tr, lambda e: L[e]), get(te, lambda e: L[e]),
         'Episode length', 'frames',
         'the selection rule picked on moving-window COUNT,\nwhich scales with length')
    ecdf(axes[0, 1], get(tr, lambda e: MV[e]), get(te, lambda e: MV[e]),
         'Moving windows per episode', 'windows',
         'selected on directly — a shift here is by construction')
    ecdf(axes[0, 2], get(tr, lambda e: 100 * MV[e] / TOT[e]),
         get(te, lambda e: 100 * MV[e] / TOT[e]),
         'Moving fraction', '% of the episode\'s windows',
         'length-normalised: this is the one that should NOT shift')
    ecdf(axes[1, 0], get(tr, d0), get(te, d0), 'Initial T-to-goal distance', 'px')

    ax = axes[1, 1]
    for idxs, c, m, lab in ((tr, C_TRAIN, 'o', 'train'), (te, C_TEST, '^', 'test')):
        p = np.array([bp[int(starts[e])][:2] for e in idxs])
        ax.scatter(p[:, 0], p[:, 1], s=26, facecolor='none', edgecolor=c, marker=m,
                   linewidths=1.1, label=lab)
    ax.scatter(*[256], *[256], marker='*', s=200, color=INK, zorder=5, label='goal')
    ax.set_title('Initial block position', fontsize=9, color=INK, loc='left')
    ax.set_xlabel('x (arena px)', fontsize=8); ax.set_ylabel('y (arena px)', fontsize=8)
    ax.set_aspect('equal'); ax.legend(frameon=False, fontsize=7.5)
    ax.tick_params(labelsize=7, colors=MUTED)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    ax = axes[1, 2]
    for idxs, c, m, lab in ((tr, C_TRAIN, 'o', 'train'), (te, C_TEST, '^', 'test')):
        p = np.array([agp[int(starts[e])] for e in idxs])
        ax.scatter(p[:, 0], p[:, 1], s=26, facecolor='none', edgecolor=c, marker=m,
                   linewidths=1.1, label=lab)
    ax.set_title('Initial agent position', fontsize=9, color=INK, loc='left')
    ax.set_xlabel('x (arena px)', fontsize=8); ax.set_ylabel('y (arena px)', fontsize=8)
    ax.set_aspect('equal'); ax.legend(frameon=False, fontsize=7.5)
    ax.tick_params(labelsize=7, colors=MUTED)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    fig.suptitle('pusht_seed42_train176 — train (176) vs held-out test (30)\n'
                 'the 30 were SELECTED to hit a 100:20 moving-transition ratio, not drawn at '
                 'random; KS p<0.05 marks where that biased them',
                 fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
