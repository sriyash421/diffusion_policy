"""Visualize and check one split's data: data/l2s_maze/<split>/viz/{split,trajectories,stats}.png.

Asserts the split invariants (goal pools, train/test disjointness, eval max_steps) and that
every zarr episode starts at its recorded start and ends within goal_tol of its goal.

    python scripts/l2s_maze/visualize_data.py --split as_is
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from expert_utils import GOAL_TOL  # noqa: E402
from maze_layout import (CENTER, DEFAULT_LAYOUT_DIR, REPO, SPLITS, bfs_to,  # noqa: E402
                         free_cells, goal_pool, load_layout)

TRAIN_C, TEST_C = '#2a78d6', '#eb6834'      # categorical slots 1, 2
UNUSED_C, WALL_C, FREE_C = '#b5b3ad', '#3b3a37', '#fcfcfb'
INK, INK_2 = '#0b0b0b', '#52514e'


def load_split(split, layout_dir):
    maze, layout = load_layout(layout_dir)
    root = zarr.open(str(REPO / f'data/l2s_maze/{split}/train/expert.zarr'), mode='r')
    ends = np.asarray(root['meta/episode_ends'])
    train = {
        'ends': ends, 'starts_idx': np.concatenate([[0], ends[:-1]]),
        'start': [tuple(x) for x in np.asarray(root['meta/start_ij'])],
        'goal': [tuple(x) for x in np.asarray(root['meta/goal_ij'])],
        'agent_pos': np.asarray(root['data/obs/agent_pos']),
        'goal_pos': np.asarray(root['data/obs/goal_pos']),
        'actions': np.asarray(root['data/actions']),
        'attrs': dict(root['meta'].attrs),
    }
    manifest = json.loads((REPO / f'data/l2s_maze/{split}/eval_episodes.json').read_text())
    return maze, layout, train, manifest


def check(split, maze, layout, train, manifest):
    """Assert invariants; return printable stats."""
    r = layout['r_inner']
    train_pool = set(goal_pool(maze, split, 'train', r))
    test_pool = set(goal_pool(maze, split, 'test', r))
    eval_eps = manifest['episodes']
    train_goals, eval_goals = set(train['goal']), {tuple(e['goal']) for e in eval_eps}

    assert layout['sha'] == train['attrs']['layout_sha'] == manifest['layout_sha']
    assert train_goals <= train_pool, 'train goal outside train pool'
    assert eval_goals <= test_pool, 'eval goal outside test pool'
    if split != 'as_is':
        assert not (train_pool & test_pool), 'train/test goal pools overlap'
        assert not (train_goals & eval_goals), 'train/eval goals overlap'
    assert abs(layout['inner_frac'] - layout['inner_frac_target']) < 0.05
    assert len(eval_eps) == 100
    for e in eval_eps:
        assert e['max_steps'] == 2 * e['solve_len'] and e['solve_len'] >= 16
        assert tuple(e['start']) != tuple(e['goal'])

    for k, (s, t) in enumerate(zip(train['starts_idx'], train['ends'])):
        assert np.allclose(train['agent_pos'][s], train['start'][k]), f'ep {k} start'
        assert np.allclose(train['goal_pos'][s:t], train['goal'][k]), f'ep {k} goal_pos'
        # last recorded obs precedes the final (successful) step: within one step of goal
        assert np.linalg.norm(train['agent_pos'][t - 1] - train['goal'][k]) <= GOAL_TOL + 0.36
    assert np.abs(train['actions']).max() <= 1.0 + 1e-6

    lengths = np.diff(np.concatenate([[0], train['ends']]))
    return {
        'split': split, 'n_train_episodes': len(train['ends']),
        'n_train_frames': int(train['ends'][-1]),
        'expert_success_rate': 1 - train['attrs']['n_failed'] / train['attrs']['n_attempted'],
        'n_short_rejected': train['attrs']['n_short'],
        'train_len': [int(lengths.min()), float(lengths.mean()), int(lengths.max())],
        'n_train_goal_pool': len(train_pool), 'n_test_goal_pool': len(test_pool),
        'n_distinct_train_goals': len(train_goals), 'n_distinct_eval_goals': len(eval_goals),
        'n_eval_episodes': len(eval_eps),
        'eval_solve_len': [min(e['solve_len'] for e in eval_eps),
                           float(np.mean([e['solve_len'] for e in eval_eps])),
                           max(e['solve_len'] for e in eval_eps)],
        'eval_dropped': f"{manifest['n_dropped']}/{manifest['n_attempted']}",
        'train_eval_pair_overlap': len(set(zip(train['start'], train['goal']))
                                       & {(tuple(e['start']), tuple(e['goal']))
                                          for e in eval_eps}),
    }


def draw_maze(ax, maze, split=None, r=None):
    """Walls/free; with `split`, free cells are tinted by goal role (train/test/unused)."""
    ax.imshow(maze, cmap=ListedColormap([FREE_C, WALL_C]), vmin=0, vmax=1,
              interpolation='nearest')
    if split is not None:
        role = np.full(maze.shape, np.nan)
        role[tuple(np.array(free_cells(maze)).T)] = 2  # unused
        for c in goal_pool(maze, split, 'train', r):
            role[c] = 0
        for c in goal_pool(maze, split, 'test', r):
            role[c] = 1 if split != 'as_is' else role[c]
        ax.imshow(np.ma.masked_invalid(role), cmap=ListedColormap([TRAIN_C, TEST_C, UNUSED_C]),
                  vmin=0, vmax=2, alpha=0.28, interpolation='nearest')
    ax.plot(CENTER[1], CENTER[0], marker='+', color=INK, ms=9, mew=1.5)
    ax.set_xticks([])
    ax.set_yticks([])


def region_legend(split):
    if split == 'as_is':
        return [Patch(color=TRAIN_C, alpha=0.4, label='train = test goal region (all free)')]
    return [Patch(color=TRAIN_C, alpha=0.4, label='train goal region'),
            Patch(color=TEST_C, alpha=0.4, label='test goal region'),
            Patch(color=UNUSED_C, alpha=0.6, label='unused as goal')]


def jitter(rng, pts, s=0.18):
    pts = np.asarray(pts, dtype=float)
    return pts + rng.uniform(-s, s, pts.shape)


def plot_split(split, maze, layout, train, manifest, out):
    rng = np.random.default_rng(0)
    eval_eps = manifest['episodes']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    titles = ['Goal regions', f'Train pairs (n={len(train["goal"])})',
              f'Eval pairs (n={len(eval_eps)})']
    for ax, title in zip(axes, titles):
        draw_maze(ax, maze, split, layout['r_inner'])
        ax.set_title(title, color=INK, fontsize=12)
    for ax, starts, goals in (
            (axes[1], train['start'], train['goal']),
            (axes[2], [e['start'] for e in eval_eps], [e['goal'] for e in eval_eps])):
        s, g = jitter(rng, starts), jitter(rng, goals)
        ax.scatter(s[:, 1], s[:, 0], s=10, facecolor='none', edgecolor=INK_2, lw=0.8,
                   label='start')
        ax.scatter(g[:, 1], g[:, 0], s=12, color=INK, marker='x', lw=1.0, label='goal')
        ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    axes[0].legend(handles=region_legend(split), loc='lower left', fontsize=8, framealpha=0.9)
    extra = f"   r_inner={layout['r_inner']}" if split == 'inner_outer' else ''
    fig.suptitle(f'{split}: 17x17 maze in 25x25 canvas, + = centre {tuple(CENTER)}{extra}',
                 color=INK)
    fig.tight_layout()
    fig.savefig(out / 'split.png', dpi=130)
    plt.close(fig)


def plot_trajectories(split, maze, layout, train, manifest, out, n_traj=20):
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    draw_maze(axes[0], maze, split, layout['r_inner'])
    idx = rng.choice(len(train['ends']), size=min(n_traj, len(train['ends'])), replace=False)
    for k in idx:
        p = train['agent_pos'][train['starts_idx'][k]:train['ends'][k]]
        axes[0].plot(p[:, 1], p[:, 0], color=TRAIN_C, lw=1.2, alpha=0.8)
        axes[0].plot(p[0, 1], p[0, 0], 'o', color=TRAIN_C, ms=4, mfc='none')
        axes[0].plot(*train['goal'][k][::-1], 'x', color=INK, ms=5)
    axes[0].set_title(f'{len(idx)} random expert trajectories (o start, x goal)', color=INK)

    draw_maze(axes[1], maze, split, layout['r_inner'])
    for e in manifest['episodes']:
        (si, sj), (gi, gj) = e['start'], e['goal']
        axes[1].annotate('', xy=(gj, gi), xytext=(sj, si),
                         arrowprops=dict(arrowstyle='->', color=TEST_C, lw=0.8, alpha=0.7))
    axes[1].set_title('Eval episodes: start -> goal', color=INK)
    fig.suptitle(split, color=INK)
    fig.tight_layout()
    fig.savefig(out / 'trajectories.png', dpi=130)
    plt.close(fig)


def plot_stats(split, maze, train, manifest, out):
    eval_eps = manifest['episodes']
    lengths = np.diff(np.concatenate([[0], train['ends']]))
    train_d0 = [bfs_to(maze, g)[s] for s, g in zip(train['start'], train['goal'])]
    eval_d0 = [e['d0'] for e in eval_eps]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax = axes[0, 0]
    bins = np.linspace(0, max(lengths.max(), max(e['solve_len'] for e in eval_eps)), 30)
    ax.hist(lengths, bins=bins, color=TRAIN_C, alpha=0.7, density=True,
            label=f'train episode length (n={len(lengths)})')
    ax.hist([e['solve_len'] for e in eval_eps], bins=bins, color=TEST_C, alpha=0.7,
            density=True, label=f'eval expert solve length (n={len(eval_eps)})')
    ax.set_xlabel('env steps')
    ax.set_ylabel('density')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    bins = np.arange(0, max(max(train_d0), max(eval_d0)) + 3, 2)
    ax.hist(train_d0, bins=bins, color=TRAIN_C, alpha=0.7, density=True, label='train')
    ax.hist(eval_d0, bins=bins, color=TEST_C, alpha=0.7, density=True, label='eval')
    ax.set_xlabel('BFS distance start -> goal (cells)')
    ax.set_ylabel('density')
    ax.legend(fontsize=8)

    for ax, goals, title in ((axes[1, 0], train['goal'], 'Train goal counts'),
                             (axes[1, 1], [tuple(e['goal']) for e in eval_eps],
                              'Eval goal counts')):
        counts = np.zeros(maze.shape)
        for g in goals:
            counts[g] += 1
        draw_maze(ax, maze)
        im = ax.imshow(np.ma.masked_where(counts == 0, counts), cmap='Blues',
                       interpolation='nearest')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, color=INK)
    for ax in axes[0]:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#e4e3df', lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(split, color=INK)
    fig.tight_layout()
    fig.savefig(out / 'stats.png', dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=SPLITS, required=True)
    ap.add_argument('--layout-dir', default=str(DEFAULT_LAYOUT_DIR))
    args = ap.parse_args()

    maze, layout, train, manifest = load_split(args.split, args.layout_dir)
    stats = check(args.split, maze, layout, train, manifest)
    out = REPO / f'data/l2s_maze/{args.split}/viz'
    out.mkdir(parents=True, exist_ok=True)
    plot_split(args.split, maze, layout, train, manifest, out)
    plot_trajectories(args.split, maze, layout, train, manifest, out)
    plot_stats(args.split, maze, train, manifest, out)
    (out / 'stats.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f'checks passed; wrote {out}')


if __name__ == '__main__':
    main()
