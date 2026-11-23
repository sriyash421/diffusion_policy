"""Where are the train / test positions, and where does each arm succeed or fail? (2026-11-23, for the DAgger obs report)
  pos_<setting>.png      maze + start cells + train goals + held-out goals (one maze) / example train and held-out mazes (new mazes)
  succmap_<setting>.png  per goal cell: success of the LAST attempt (7 earlier attempts in context; no-context runs: 8th fresh attempt),
                         final-round policy, mean over starts x seeds; rows = observation, cols = arm. Train goals = circles, held-out = squares.
                         New mazes: goal cell in grid coordinates, pooled over levels; rows = observation x (train mazes | held-out mazes).
  python plots_positions.py        (CPU; needs report/pairs_<setting>.npz from dump_pairs.py)
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from plots_simple import SETTINGS, ARMS, REF

I = Path(__file__).resolve().parent; A = I.parent; REP = I / 'report'; PG = A / 'maze_procgen_ast_20261121'
OBS = {'window': '7×7 window', 'ddpm400': 'DDPM t=400'}
COLS = ARMS + [REF]
CM = plt.cm.RdYlGn; WALL = ListedColormap(['white', '#444444'])
TR, TE = '#0072B2', '#E69F00'


def maze(ax, free):
    ax.imshow(~free, cmap=WALL, origin='upper', interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])


def by_goal(goals, succ):
    """succ (pairs,) -> {goal cell: mean success}."""
    cells = np.rint(goals).astype(int); out = {}
    for c in {tuple(c) for c in cells}:
        out[c] = succ[(cells == c).all(1)].mean()
    return out


def last_attempt(z, split):
    return z[f'{split}_success'][-1, -1].mean(-1)      # final round, last attempt: (pairs,)


def run_path(setting, obs, arm):
    pre, root, *_ = SETTINGS[setting]
    return root / (f'{pre}_clean_noctx' if arm == 'clean' else f'{pre}_{obs}_{arm}') / 'eval_ctx.npz'


def one_maze(setting):
    P = np.load(REP / f'pairs_{setting}.npz'); free = P['free']; _, _, sname, test_name = SETTINGS[setting]
    trg = {tuple(c) for c in np.rint(P['train_goal']).astype(int)}; teg = {tuple(c) for c in np.rint(P['test_goal']).astype(int)}
    sts = {tuple(c) for c in np.rint(np.r_[P['train_start'], P['test_start']]).astype(int)}
    fig, ax = plt.subplots(figsize=(4.6, 4.9)); maze(ax, free)
    ax.scatter(*np.array(sorted(sts)).T, s=8, c='#999999', label='start cells', zorder=2)
    ax.scatter(*np.array(sorted(trg)).T, s=70, marker='o', facecolor='none', edgecolor=TR, lw=2, label=f'train goals ({len(trg)})', zorder=3)
    ax.scatter(*np.array(sorted(teg)).T, s=70, marker='s', facecolor='none', edgecolor=TE, lw=2, label=f'{test_name.split(" (")[0]} ({len(teg)})', zorder=3)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.02), ncol=2, fontsize=8, frameon=False); ax.set_title(sname, fontsize=10)
    fig.tight_layout(); fig.savefig(REP / f'pos_{setting}.png', dpi=130); plt.close(fig)
    rows = [o for o in OBS if run_path(setting, o, 'fbR').exists() or run_path(setting, o, 'noctx').exists()]
    fig, axs = plt.subplots(len(rows), len(COLS), figsize=(2.6 * len(COLS), 2.8 * len(rows) + .7), squeeze=False)
    for r, obs in enumerate(rows):
        for c, (arm, label, *_) in enumerate(COLS):
            ax = axs[r, c]; maze(ax, free); p = run_path(setting, obs, arm)
            if r == 0:
                ax.set_title(label.replace(' (', '\n('), fontsize=8)
            if c == 0:
                ax.set_ylabel(OBS[obs], fontsize=9)
            if not p.exists():
                ax.text(.5, .5, 'running', ha='center', va='center', transform=ax.transAxes, color='.5'); continue
            z = np.load(p)
            for split, mk, ec in (('train', 'o', TR), ('test', 's', TE)):
                m = by_goal(P[f'{split}_goal'], last_attempt(z, split)); xy = np.array(list(m)); v = np.array(list(m.values()))
                ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap=CM, vmin=0, vmax=1, s=55, marker=mk, edgecolor=ec, lw=1.2, zorder=3)
    sm = plt.cm.ScalarMappable(cmap=CM, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axs, orientation='horizontal', fraction=.04, pad=.04, label='success of the last attempt (per goal cell)')
    fig.suptitle(f'{sname}: where it succeeds (green) and fails (red) · circles = train goals, squares = held-out goals', fontsize=10)
    fig.savefig(REP / f'succmap_{setting}.png', dpi=130, bbox_inches='tight'); plt.close(fig)
    print('wrote', setting)


def new_mazes():
    L = np.load(PG / 'levels.npz'); nav, S, G, lay, spl = L['nav'].astype(bool), L['start'], L['goal'], L['layout'], L['split']
    H, W = nav.shape[1:]
    fig, axs = plt.subplots(2, 4, figsize=(10, 5.4))
    for r, (sp, col, name) in enumerate(((0, TR, 'train mazes'), (1, TE, 'held-out mazes'))):
        lays = list(dict.fromkeys(lay[spl == sp]))[:4]
        for c, li in enumerate(lays):
            ax = axs[r, c]; idx = np.where((lay == li) & (spl == sp))[0]; maze(ax, nav[idx[0]])
            ax.scatter(*S[idx[0]], s=60, marker='^', c='k', zorder=3)
            ax.scatter(G[idx, 0], G[idx, 1], s=60, marker='o' if sp == 0 else 's', facecolor='none', edgecolor=col, lw=2, zorder=3)
            if c == 0:
                ax.set_ylabel(f'{name}\n({(spl == sp).sum() // len(idx)} layouts × {len(idx)} goals)', fontsize=9)
    fig.suptitle('new mazes: examples · triangle = start, circles / squares = the layout\'s goals', fontsize=10)
    fig.tight_layout(); fig.savefig(REP / 'pos_pg.png', dpi=130); plt.close(fig)
    rows = [(o, s) for o in OBS for s in ('train', 'test') if run_path('pg', o, 'fbR').exists() or run_path('pg', o, 'noctx').exists()]
    fig, axs = plt.subplots(len(rows), len(COLS), figsize=(2.6 * len(COLS), 2.7 * len(rows) + .7), squeeze=False)
    for r, (obs, split) in enumerate(rows):
        for c, (arm, label, *_) in enumerate(COLS):
            ax = axs[r, c]; ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-.5, W - .5); ax.set_ylim(H - .5, -.5); ax.set_facecolor('#eeeeee')
            if r == 0:
                ax.set_title(label.replace(' (', '\n('), fontsize=8)
            if c == 0:
                ax.set_ylabel(f'{OBS[obs]}\n{"train mazes" if split == "train" else "held-out mazes"}', fontsize=9)
            p = run_path('pg', obs, arm)
            if not p.exists():
                ax.text(.5, .5, 'running', ha='center', va='center', transform=ax.transAxes, color='.5'); continue
            z = np.load(p); m = by_goal(G[z[f'{split}_levels']], last_attempt(z, split)); xy = np.array(list(m)); v = np.array(list(m.values()))
            ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap=CM, vmin=0, vmax=1, s=22, marker='o' if split == 'train' else 's', edgecolor='none')
            st = np.unique(np.rint(S[z[f'{split}_levels']]).astype(int), axis=0); ax.scatter(st[:, 0], st[:, 1], s=40, marker='^', c='k')
    sm = plt.cm.ScalarMappable(cmap=CM, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axs, orientation='horizontal', fraction=.03, pad=.03, label='success of the last attempt (per goal cell, pooled over levels)')
    fig.suptitle('new mazes: where it succeeds (green) and fails (red) by goal position · triangle = start', fontsize=10)
    fig.savefig(REP / 'succmap_pg.png', dpi=130, bbox_inches='tight'); plt.close(fig)
    print('wrote pg')


if __name__ == '__main__':
    one_maze('interp30c'); one_maze('11brc'); new_mazes()
