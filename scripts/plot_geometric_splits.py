"""Draw the geometric split manifests: which start poses each split actually holds.

Two figures, one per family produced by scripts/make_geometric_splits.py:
  1. quadrant -- eval is one 256-px quadrant, train is the rest of the arena.
  2. border   -- train is a band hugging the walls, test is the interior core, val is
                 the ring the wider bands train on and this budget does not.

Each episode is one marker at its T's starting centroid. The T's own outline (as
plot_pusht_split_shift draws it) spans ~120 px, so 150 overlaid outlines smear the very
band edges these figures exist to show -- the goal T is still drawn as an outline, since
there is only one of it. Split identity is carried by colour AND marker shape together.
"""
import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from plot_pusht_split_shift import (          # the renderer, not a second copy of it
    ARENA, INK, MUTED, C_TEST, C_TRAIN30, C_TRAIN126, draw_t, episode_start_poses, frame)
from diffusion_policy.env.pusht.feedback_util import GOAL_POSE

WS = 512
SPLITS = HERE.parents[0] / 'diffusion_policy/config/splits'
# colour AND marker shape, so the three splits stay separable without hue.
# (color, marker, filled, dash pattern for the matching band edge / legend key)
STYLE = {'train': (C_TRAIN126, 'o', True,  (0, (1, 0))),
         'val':   (C_TRAIN30,  's', False, (0, (5, 2))),
         'test':  (C_TEST,     '^', True,  (0, (1.5, 1.5)))}


def start_poses(zarr_path, idxs):
    """episode_start_poses, but an empty split is (0, 3) rather than an index error."""
    if not idxs:
        return np.empty((0, 3))
    return episode_start_poses(zarr_path, np.asarray(idxs, dtype=int))


def inset_square(ax, w, **kw):
    """The locus of points exactly w px from the nearest wall."""
    lo, hi = w, WS - w
    ax.plot([lo, hi, hi, lo, lo], [lo, lo, hi, hi, lo], **kw)


def panel(ax, poses, title, subtitle):
    # test on top: it is the smallest set in the quadrant figure and the one that has to
    # stay countable where it overlaps the train cloud.
    for z, split in enumerate(('train', 'val', 'test')):
        color, marker, filled, _ = STYLE[split]
        p_ = poses.get(split)
        if p_ is None or not len(p_):
            continue
        ax.scatter(p_[:, 0], p_[:, 1], s=26, marker=marker,
                   facecolor=color if filled else 'none', edgecolor=color,
                   lw=1.1, alpha=0.85, zorder=3 + z)
    draw_t(ax, GOAL_POSE, MUTED, INK, 0.5, 1.6, zorder=7)
    frame(ax, compact=True)
    # frame() pads to +-70 px for T outlines that reach past the wall; markers do not.
    ax.set_xlim(-22, 534)
    ax.set_ylim(534, -22)                     # y stays inverted
    ax.text(0.5, 1.035, title, transform=ax.transAxes, ha='center', va='bottom',
            color=INK, fontsize=11)
    ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha='center', va='bottom',
            color=MUTED, fontsize=9)


def legend(fig, manifests, y):
    handles = [plt.Line2D([], [], color=STYLE[s][0], lw=0, marker=STYLE[s][1], ms=7,
                          markerfacecolor=STYLE[s][0] if STYLE[s][2] else 'none',
                          markeredgecolor=STYLE[s][0],
                          label=f'{s}  (n = {" / ".join(str(len(m[s])) for m in manifests)})')
               for s in ('train', 'val', 'test')]
    handles.append(plt.Line2D([], [], color=INK, lw=1.6, ls='-', label='goal T'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, y), ncol=4,
               frameon=False, fontsize=10, labelcolor=INK, columnspacing=2.0)


def fig_quadrant(zarr_path, names, out):
    ms = [json.loads((SPLITS / f'{n}.json').read_text()) for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.9))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.775, bottom=0.03, wspace=0.04)
    for ax, m in zip(axes, ms):
        q = m['derivation']['quadrant']
        poses = {s: start_poses(zarr_path, m[s]) for s in ('train', 'val', 'test')}
        # the quadrant cut itself, so "eval is that corner" is checkable by eye
        ax.axvline(WS / 2, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.axhline(WS / 2, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        x0 = ARENA[0] if 'left' in q else WS / 2
        y0 = WS / 2 if 'bottom' in q else ARENA[0]
        ax.fill([x0, x0 + (WS / 2 - ARENA[0]), x0 + (WS / 2 - ARENA[0]), x0],
                [y0, y0, y0 + (WS / 2 - ARENA[0]), y0 + (WS / 2 - ARENA[0])],
                color=MUTED, alpha=0.07, zorder=0)
        panel(ax, poses,
              f'eval quadrant: {q}',
              f'{m["derivation"]["n_eval_region"]} episodes held out  |  '
              f'train {len(m["train"])}')
    legend(fig, ms, 0.905)
    fig.text(0.02, 0.992, 'PushT geometric splits: eval = one quadrant of T start poses',
             color=INK, fontsize=13.5, va='top')
    fig.text(0.02, 0.955,
             'Shaded corner is the held-out quadrant; train is every episode outside it. '
             'The two are not size-matched -- the demos\nare not uniform over the arena '
             '(69 episodes start bottom-left, 33 top-right). y increases downward.',
             color=MUTED, fontsize=9.5, va='top')
    fig.savefig(out, dpi=160, facecolor='white')
    plt.close(fig)


def fig_border(zarr_path, names, out):
    ms = [json.loads((SPLITS / f'{n}.json').read_text()) for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.4))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.775, bottom=0.03, wspace=0.03)
    for ax, m in zip(axes, ms):
        dv = m['derivation']
        poses = {s: start_poses(zarr_path, m[s]) for s in ('train', 'val', 'test')}
        inset_square(ax, dv['train_d_max_px'], color=C_TRAIN126, lw=1.4,
                     ls=STYLE['train'][3], alpha=0.9, zorder=2)
        inset_square(ax, dv['test_d_min_px'], color=C_TEST, lw=1.4,
                     ls=STYLE['test'][3], alpha=0.9, zorder=2)
        panel(ax, poses,
              f'train {dv["n_train"]}  (band w = {dv["train_d_max_px"]:.0f} px)',
              f'val {len(m["val"])}  |  test {len(m["test"])} '
              f'(core, w > {dv["test_d_min_px"]:.0f} px)')
    legend(fig, ms, 0.895)
    fig.text(0.015, 0.992,
             'PushT geometric splits: train = T starts near a wall, test = the interior core',
             color=INK, fontsize=13.5, va='top')
    fig.text(0.015, 0.952,
             'Thin squares mark the band edges: solid blue = outer edge of the training '
             'band, dotted orange = inner edge of the shared test core.\n'
             'The three panels share one test set verbatim. Val is the ring the wider '
             'bands train on and this budget does not, so val nests the other way '
             '(70 / 40 / 0).',
             color=MUTED, fontsize=9.5, va='top')
    fig.savefig(out, dpi=160, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr', default='data/pusht_cchi_v7_replay.zarr')
    ap.add_argument('--outdir', default='analysis')
    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    q = outdir / 'pusht_geometric_split_quadrant.png'
    b = outdir / 'pusht_geometric_split_border.png'
    fig_quadrant(args.zarr, ['pusht_blockquad_bottomleft_train137',
                             'pusht_blockquad_topright_train173'], q)
    fig_border(args.zarr, [f'pusht_blockborder_train{k}_core50' for k in (30, 60, 100)], b)
    print('wrote', q)
    print('wrote', b)


if __name__ == '__main__':
    main()
