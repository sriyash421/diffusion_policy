"""Where the T starts, and where it has to end up, for the train and test splits.

Two figures:
  1. Initial block pose (x, y) of every episode, train-30 / train-126 / test-50.
  2. The goal pose the T must reach -- a single constant shared by both splits.

The zarr stores no per-episode goal: PushTEnv._setup pins goal_pose = [256, 256, pi/4]
and feedback_util.GOAL_POSE hardcodes the same value, so figure 2 is one pose, not a
distribution. It is drawn anyway to show the target the start poses are measured against.
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import zarr

from diffusion_policy.env.pusht.feedback_util import GOAL_POSE, keypoints_at_pose

# dataviz categorical slots 1-3; the only three that clear the all-pairs CVD floors,
# which is the pairlist a scatter needs. Each is paired with a marker and a fill state
# so identity never rests on hue alone.
C_TRAIN126, C_TEST, C_TRAIN30 = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED = '#1a1a19', '#8a8880'

ARENA = (5, 506)                 # PushTEnv._setup wall segments
# The T is two convex quads (PushTEnv.add_tee builds shape1 + shape2), not one 8-gon:
# T_VERTS[0:4] is the top bar, T_VERTS[4:8] the stem. Closing a single loop over all
# eight vertices draws a self-crossing shape.
T_QUADS = ([0, 1, 2, 3], [4, 5, 6, 7])


def t_quads(pose):
    kp = keypoints_at_pose(np.asarray(pose, dtype=np.float32))
    return [np.vstack([kp[q], kp[q[:1]]]) for q in T_QUADS]


def draw_t(ax, pose, face, edge, fill_alpha, lw, ls='-', alpha=1.0, zorder=0):
    for quad in t_quads(pose):
        if fill_alpha > 0:
            ax.fill(quad[:, 0], quad[:, 1], color=face, alpha=fill_alpha, zorder=zorder)
        ax.plot(quad[:, 0], quad[:, 1], color=edge, lw=lw, ls=ls, alpha=alpha,
                zorder=zorder + 1)


def episode_start_poses(zarr_path, idxs):
    root = zarr.open(str(zarr_path), 'r')
    ends = root['meta']['episode_ends'][:]
    starts = np.concatenate([[0], ends[:-1]])
    return root['data']['block_pos'][:][starts[np.asarray(idxs)]]


def frame(ax, compact=False):
    ax.plot([ARENA[0], ARENA[1], ARENA[1], ARENA[0], ARENA[0]],
            [ARENA[0], ARENA[0], ARENA[1], ARENA[1], ARENA[0]],
            color=MUTED, lw=1, zorder=0)
    ax.set_xlim(-70, 581)        # a T at the wall reaches ~60px past its centre
    ax.set_ylim(-70, 581)
    ax.invert_yaxis()            # pymunk_override.positive_y_is_up is False: y is down
    ax.set_aspect('equal')
    for s in ax.spines.values():
        s.set_visible(False)
    if compact:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        return
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    for name in ('left', 'bottom'):
        ax.spines[name].set_visible(True)
        ax.spines[name].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def fig_start(poses, out):
    """Every episode's starting T drawn at its actual pose, not a centroid marker.

    206 T outlines in one 512px arena overlap several times over, so the overlay drops
    train-30 -- it is a subset of train-126 and adds no coverage -- and each split also
    gets its own facet, which is what stays readable per split.
    """
    facets = [
        ('train-126', poses['train126'], C_TRAIN126, (0, (1, 0))),
        ('train-30', poses['train30'], C_TRAIN30, (0, (5, 2))),
        ('test-50', poses['test'], C_TEST, (0, (1.5, 1.5))),
    ]
    overlay = [f for f in facets if f[0] != 'train-30']

    fig = plt.figure(figsize=(8.6, 12.2))
    # Ratios chosen so each cell is roughly square: an aspect='equal' axes in a
    # non-square cell letterboxes, and the dead band collides with the row below.
    gs = fig.add_gridspec(2, 3, height_ratios=(7.7, 2.45), hspace=0.09, wspace=0.06,
                          left=0.09, right=0.98, top=0.90, bottom=0.03)
    ax = fig.add_subplot(gs[0, :])

    for label, p_, color, ls in overlay:
        for pose in p_:
            draw_t(ax, pose, color, color, 0.0, 1.0, ls=ls, alpha=0.55, zorder=3)
    draw_t(ax, GOAL_POSE, MUTED, INK, 0.55, 2.2, zorder=5)
    ax.annotate('goal T', (GOAL_POSE[0] + 34, GOAL_POSE[1] + 46), xytext=(22, 34),
                textcoords='offset points', color=INK, fontsize=11, zorder=6,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=MUTED, lw=0.8),
                arrowprops=dict(arrowstyle='-', color=INK, lw=1))

    frame(ax)
    # draw_t emits one artist per quad per episode, so the real marks cannot carry a
    # label; these empty lines are the legend proxies.
    handles = [plt.Line2D([], [], color=c, lw=2.2, ls=ls, label=f'{lab}  (n={len(p_)})')
               for lab, p_, c, ls in facets]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.535, 0.945),
               ncol=3, frameon=False, fontsize=10, labelcolor=INK, columnspacing=1.8)
    fig.text(0.09, 0.980, 'PushT initial T pose: train vs test', color=INK, fontsize=15,
             va='top')
    fig.text(0.09, 0.962,
             'top: train-126 vs test-50 overlaid (train-30 omitted -- it is a subset).  '
             'bottom: one facet per split.', color=MUTED, fontsize=10, va='top')

    for col, (label, p_, color, ls) in enumerate(facets):
        axf = fig.add_subplot(gs[1, col])
        for pose in p_:
            draw_t(axf, pose, color, color, 0.0, 0.8, ls=ls, alpha=0.6, zorder=3)
        draw_t(axf, GOAL_POSE, MUTED, INK, 0.55, 1.3, zorder=5)
        frame(axf, compact=True)
        # Labelled inside the panel: an axes title lands in the letterbox band and
        # collides with the overlay's x-axis label above it.
        # (0.13, 0.87) in axes fraction is just inside the arena wall, which sits at
        # 0.115..0.885 because the limits are padded past it by a T half-length.
        axf.text(0.13, 0.87, f'{label}  (n={len(p_)})', transform=axf.transAxes,
                 ha='left', va='top', color=INK, fontsize=10,
                 bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='none'))

    fig.savefig(out, dpi=160, facecolor='white')
    plt.close(fig)


def fig_goal(poses, out):
    fig, ax = plt.subplots(figsize=(7.4, 8.0))

    # Both splits target the identical pose, so drawing each split's goal T traces the
    # same outline twice. That coincidence is the finding, so it is drawn literally:
    # the train T underneath in solid, the test T dotted on top of it.
    draw_t(ax, GOAL_POSE, MUTED, MUTED, 0.22, 1.0, zorder=1)
    draw_t(ax, GOAL_POSE, MUTED, C_TRAIN126, 0.0, 4.5, zorder=3)
    draw_t(ax, GOAL_POSE, MUTED, C_TEST, 0.0, 4.5, ls=(0, (1.2, 1.6)), zorder=4)

    ax.annotate(f'x = {GOAL_POSE[0]:.0f},  y = {GOAL_POSE[1]:.0f},  '
                r'$\theta = \pi/4$',
                (GOAL_POSE[0] + 40, GOAL_POSE[1] + 55), xytext=(70, -105),
                textcoords='offset points', color=INK, fontsize=12,
                ha='center', va='center',
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=1))

    frame(ax)
    handles = [
        plt.Line2D([], [], color=C_TRAIN126, lw=3, ls='-',
                   label='train goal T  (train-30 and train-126)'),
        plt.Line2D([], [], color=C_TEST, lw=3, ls=(0, (1.2, 1.6)),
                   label='test goal T  (test-50)'),
    ]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, -0.08), ncol=1,
              frameon=False, fontsize=11, labelcolor=INK, handletextpad=0.8)
    ax.set_title('PushT goal T pose: train vs test\n'
                 'the two outlines coincide -- one fixed pose, shared by every episode',
                 color=INK, fontsize=13, loc='left', pad=12)
    fig.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr', default='data/pusht_cchi_v7_replay.zarr')
    ap.add_argument('--split30', default='diffusion_policy/config/splits/pusht_seed42_train30.json')
    ap.add_argument('--split126', default='diffusion_policy/config/splits/pusht_seed42_train126.json')
    ap.add_argument('--outdir', default='analysis')
    args = ap.parse_args()

    m30 = json.loads(pathlib.Path(args.split30).read_text())
    m126 = json.loads(pathlib.Path(args.split126).read_text())
    assert m30['test'] == m126['test'], 'manifests disagree on the test split'
    assert set(m30['train']) <= set(m126['train']), 'train-30 is not a subset of train-126'

    poses = {
        'train30': episode_start_poses(args.zarr, m30['train']),
        'train126': episode_start_poses(args.zarr, m126['train']),
        'test': episode_start_poses(args.zarr, m126['test']),
    }
    for k, v in poses.items():
        print(f'{k:9s} n={len(v):3d}  x {v[:,0].mean():6.1f}+-{v[:,0].std():5.1f}  '
              f'y {v[:,1].mean():6.1f}+-{v[:,1].std():5.1f}  '
              f'theta {v[:,2].mean():5.2f}+-{v[:,2].std():4.2f}')

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig_start(poses, outdir / 'pusht_start_T_train_vs_test.png')
    fig_goal(poses, outdir / 'pusht_goal_T_train_vs_test.png')
    print('wrote', outdir / 'pusht_start_T_train_vs_test.png')
    print('wrote', outdir / 'pusht_goal_T_train_vs_test.png')


if __name__ == '__main__':
    main()
