"""Attempt grids on chosen test episodes, all attempts of a chain overlaid in one panel (colour = attempt number).

A. unsolved: test episodes that ast_main never solves with STk2 or STk7 (0 of 8 seeds each), one per goal cell.
   Columns: ast_main STk2 (3 attempts in a row), ast_main STk7 (8 attempts in a row).
B. 10k solves, 30 steps does not: STk7 success of ast_main minus ast_s30 >= 0.5 (of 8 seeds), largest first, one per goal cell.
   Columns: ast_s30 STk7, ast_main STk7 (8 attempts in a row each).
Success fractions in the titles are over the 8 eval seeds; the drawn chain is one rollout (seed 0).
  python fail_grid.py
"""
import json
import numpy as np
import torch
import plot_clear as pc
from rounds import HERE, make_policy, load_data, train_starts
from asteroid import chain, test_set
from bon_eval import bfs_of
from fail_viz import shortest_path

TAGS = {'ast_main': '10k updates per cycle', 'ast_s30': '30 steps per batch'}


def st_success(tag):
    """(R, A, episodes) test success, mean over seeds, attempt k+1 of STk at [k, k]."""
    f = HERE / 'runs' / tag / 'eval_seeds8.npz'
    s = np.load(f)['test_success'] if f.exists() else np.load(HERE / 'runs' / tag / 'bon_seeds8.npz')['test_st_success']
    return s.mean(-1)


def one_per_goal(eps, g, order_key, n):
    out, seen = [], set()
    for i in sorted(eps, key=order_key):
        if tuple(g[i]) not in seen:
            seen.add(tuple(g[i])); out.append(int(i))
    return out[:n]


def rollouts(tag, k, eps, S, G, free1, device, seed=0):
    cfg = json.loads((HERE / 'runs' / tag / 'config.json').read_text())
    model, pcfg = make_policy(cfg['rounds'], device); ds = rollouts.ds or load_data(pcfg['shape_meta']); rollouts.ds = ds
    model.set_normalizer(ds.get_normalizer())
    model.load_state_dict(torch.load(HERE / 'runs' / tag / f'round_{k}.pt', map_location='cpu', weights_only=False)['model'])
    model.to(device).eval(); torch.manual_seed(seed)
    with torch.no_grad():
        mem, succ, _ = chain(model, free1, S[eps], G[eps], k + 1, cfg['rounds'] - 1, 512, device)
    return mem, succ          # mem[row][attempt], succ (attempts, rows)
rollouts.ds = None


def panel(ax, free, start, goal, trajs, succ, bfs, title):
    import matplotlib.pyplot as plt
    H, W = free.shape; A = len(trajs); cols = plt.cm.viridis(np.linspace(0, .92, 8))
    ax.imshow(np.where(free, 1., .15), cmap='gray', vmin=0, vmax=1, origin='upper', extent=(-.5, W - .5, H - .5, -.5))
    sp = shortest_path(free, start, goal); ax.plot(sp[:, 0], sp[:, 1], '--', color='#2166AC', lw=1.5, alpha=.8)
    for j, t in enumerate(trajs):
        t = np.asarray(t, float) + (j - (A - 1) / 2) * .07         # small offset so identical attempts stay visible
        ax.plot(t[:, 0], t[:, 1], '-', color=cols[j], lw=1.6, alpha=.9)
        ax.plot(t[-1, 0], t[-1, 1], 'o' if succ[j] else 'X', ms=7, mfc=cols[j], mec='k', zorder=5)
    ax.plot(*start, 'o', ms=10, mfc='#2166AC', mec='k', zorder=4); ax.plot(*goal, '*', ms=16, mfc='#FFD700', mec='k', zorder=4)
    ax.set_title(title + f'\n{int(np.sum(succ))} of {A} attempts succeed; final BFS {list(map(int, bfs))}', fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([])


def grid(path, free, S, G, rows, cols, suptitle):
    """rows: episode ids; cols: list of (label, tag, k, rollouts dict ep->(trajs, succ), success table)."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(3.6 * len(cols), 3.75 * len(rows) + .9), squeeze=False)
    cache = {}
    for r, i in enumerate(rows):
        for c, (lab, tag, k, ro, s) in enumerate(cols):
            trajs, succ = ro[i]; bfs = bfs_of(free, [G[i]] * len(trajs), [t[-1] for t in trajs], cache)
            panel(axs[r, c], free, S[i], G[i], trajs, succ, bfs,
                  f'ep {i}, goal {tuple(int(v) for v in G[i])}: {lab}, {s[k, k][i]:.0%} of 8 seeds')
    cl = plt.cm.viridis(np.linspace(0, .92, 8))
    h = [Line2D([], [], color=cl[j], lw=2, label=f'attempt {j + 1}') for j in range(8)]
    h += [Line2D([], [], ls='--', color='#2166AC', label='shortest path'), Line2D([], [], ls='', marker='X', mfc='w', mec='k', label='final cell, fail'),
          Line2D([], [], ls='', marker='o', mfc='w', mec='k', label='final cell, success')]
    fig.legend(handles=h, loc='lower center', ncol=6, fontsize=7.5, frameon=False)
    fig.suptitle(suptitle + '\nblue circle = start, star = goal; attempts overlaid (slightly offset), colour = attempt number', fontsize=9.5, y=1 - .08 / fig.get_size_inches()[1])
    hin = fig.get_size_inches()[1]; fig.tight_layout(rect=(0, .55 / hin, 1, 1 - .75 / hin))
    fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close(fig); print('written', path)


def main():
    device = 'cuda'; st = pc.starts(); g = np.round(st['test_goal']).astype(int)
    m, s30 = st_success('ast_main'), st_success('ast_s30')
    never = np.where((m[2, 2] == 0) & (m[7, 7] == 0))[0]
    A_eps = one_per_goal(never, g, lambda i: (i != 0, i), 5)
    win = np.where(m[7, 7] - s30[7, 7] >= .5)[0]
    B_eps = one_per_goal(win, g, lambda i: -(m[7, 7][i] - s30[7, 7][i]), 4)
    print('unsolved by 10k:', [(i, tuple(g[i])) for i in A_eps]); print('10k solves, 30 steps not:', [(i, tuple(g[i]), m[7, 7][i], s30[7, 7][i]) for i in B_eps])
    ds = load_data(make_policy(8, device)[1]['shape_meta']); rollouts.ds = ds
    _, _, free = train_starts(ds); free1 = torch.as_tensor(free, device=device)[None]; _, S, G = test_set(free); fb = free.astype(bool)

    def ro(tag, k, eps):
        mem, succ = rollouts(tag, k, eps, S, G, free1, device)
        return {i: (mem[r], succ[:, r]) for r, i in enumerate(eps)}
    grid(HERE / 'report' / 'failgrid_unsolved_ast_main.png', fb, S, G, A_eps,
         [('STk2 (3 attempts)', 'ast_main', 2, ro('ast_main', 2, A_eps), m), ('STk7 (8 attempts)', 'ast_main', 7, ro('ast_main', 7, A_eps), m)],
         '10k updates per cycle: test goals never solved by STk2 or STk7 (0 of 8 seeds)')
    grid(HERE / 'report' / 'failgrid_main_vs_s30.png', fb, S, G, B_eps,
         [('30 steps STk7', 'ast_s30', 7, ro('ast_s30', 7, B_eps), s30), ('10k updates STk7', 'ast_main', 7, ro('ast_main', 7, B_eps), m)],
         'Test episodes STk7 solves with 10k updates per cycle but not with 30 steps per batch')


if __name__ == '__main__':
    main()
