"""Where the agent and the T START, train vs held-out test, for the t_goal-moving split.

    python scripts/plot_start_positions.py --out analysis/start_positions.png

WHAT THIS IS FOR. Any claim that an evaluation is "out of distribution" has to be measured
against where the demonstrations actually begin. Measured here: the demo AGENT start covers
essentially the whole arena -- only one 10px grid cell in the extreme corner (470, 470), which
is outside PushTEnv's own randint(50, 450) spawn range, sits more than 60px from every demo
start, and at 80px there are none. So there is no unseen agent-start REGION; what thin tails
exist are in the JOINT geometry (right-hand panels), not the marginals.

Reads block_pos / agent_pos / episode_ends only -- never the 2.8 GB image array.
Colour is paired with marker and fill so train/test survive greyscale.
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
from diffusion_policy.env.pusht.feedback_util import GOAL_POSE  # noqa: E402

C_TRAIN, C_TEST = '#2a78d6', '#eb6834'
INK, MUTED = '#1a1a19', '#8a8880'
SPAWN = (50, 450)          # PushTEnv.reset: rs.randint(50, 450) for the agent


def ecdf(ax, a, b, title, xlabel):
    for v, c, ls, lab in ((a, C_TRAIN, '-', 'train'), (b, C_TEST, (0, (5, 2)), 'test')):
        x = np.sort(v)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), color=c, ls=ls, lw=1.8, where='post')
        ax.annotate(lab, xy=(x[-1], 1.0), xytext=(-2, -12), textcoords='offset points',
                    fontsize=7.5, color=c, ha='right', weight='bold')
    ax.set_title(f'{title}\ntrain {a.mean():.0f}±{a.std():.0f}   test {b.mean():.0f}±{b.std():.0f}',
                 fontsize=9, color=INK, loc='left')
    ax.set_xlabel(xlabel, fontsize=8, color=INK)
    ax.set_ylabel('cumulative fraction', fontsize=8, color=INK)
    ax.tick_params(labelsize=7, colors=MUTED)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def scatter(ax, tr, te, title, goal=False):
    for p, c, m, lab in ((tr, C_TRAIN, 'o', f'train ({len(tr)})'),
                         (te, C_TEST, '^', f'test ({len(te)})')):
        ax.scatter(p[:, 0], p[:, 1], s=30, facecolor='none', edgecolor=c, marker=m,
                   linewidths=1.2, label=lab)
    if goal:
        ax.scatter(*GOAL_POSE[:2], marker='*', s=300, color=INK, zorder=5, label='goal')
    ax.add_patch(plt.Rectangle((SPAWN[0], SPAWN[0]), SPAWN[1] - SPAWN[0], SPAWN[1] - SPAWN[0],
                               fill=False, edgecolor=MUTED, ls=(0, (3, 3)), lw=1.0))
    ax.text(SPAWN[0] + 4, SPAWN[1] - 14, "env spawn range", fontsize=7, color=MUTED)
    ax.set_xlim(0, 512); ax.set_ylim(0, 512); ax.set_aspect('equal')
    ax.set_title(title, fontsize=9, color=INK, loc='left')
    ax.set_xlabel('x (arena px)', fontsize=8); ax.set_ylabel('y (arena px)', fontsize=8)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK)
    ax.tick_params(labelsize=7, colors=MUTED)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr', default='data/pusht_cchi_v7_replay.zarr')
    ap.add_argument('--split', default='diffusion_policy/config/splits/pusht_seed42_train176.json')
    ap.add_argument('--out', default='analysis/start_positions.png')
    args = ap.parse_args()

    g = zarr.open(str(ROOT / args.zarr), 'r')
    ends = np.asarray(g['meta/episode_ends'])
    ap_, bp = np.asarray(g['data/agent_pos']), np.asarray(g['data/block_pos'])
    st = np.concatenate([[0], ends[:-1]])
    man = json.loads((ROOT / args.split).read_text())

    def grab(idxs, arr, k=2):
        return np.array([arr[int(st[e])][:k] for e in idxs])

    trA, teA = grab(man['train'], ap_), grab(man['test'], ap_)
    trB, teB = grab(man['train'], bp), grab(man['test'], bp)

    def joint(A, B):
        d = np.linalg.norm(A - B, axis=1)
        v1 = (A - B) / np.linalg.norm(A - B, axis=1, keepdims=True)
        v2 = (GOAL_POSE[:2] - B) / np.linalg.norm(GOAL_POSE[:2] - B, axis=1, keepdims=True)
        return d, np.degrees(np.arccos(np.clip((v1 * v2).sum(1), -1, 1)))

    d_tr, a_tr = joint(trA, trB)
    d_te, a_te = joint(teA, teB)

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 10))
    scatter(ax[0, 0], trA, teA, 'AGENT start position')
    scatter(ax[0, 1], trB, teB, 'T (block) start position', goal=True)
    ecdf(ax[1, 0], d_tr, d_te, 'Agent-to-T distance at t=0', 'px')
    ecdf(ax[1, 1], a_tr, a_te, 'Angle agent–T–goal at t=0\n(180° = agent behind the T)', 'degrees')

    fig.suptitle('pusht_seed42_train176 — start positions, train (176) vs held-out test (30)\n'
                 'the agent-start marginal is covered edge to edge in BOTH splits; the thin '
                 'tails are in the joint geometry below',
                 fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
