"""Readable versions of success_by_start.png: starts ordered by how far they are from the goal (shortest-path length).

  success_vs_distance.png  per split: mean success vs shortest-path length (binned), one line per round + BC
  success_heatmap.png      per split: rows = BC, round 0..R-1; columns = starts sorted by path length; colour = success over seeds

  python plot_clear.py <tag> [<tag> ...]      (CPU only; caches starts/goals/path lengths in starts_cache.npz)
"""
import json, sys
from collections import deque
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; CACHE = HERE / 'starts_cache.npz'


def bfs_len(free, start, goal):
    """Shortest 4-connected path length from start to goal; cells are (x, y), free[y, x]."""
    H, W = free.shape; d = -np.ones((H, W), int); gx, gy = int(goal[0]), int(goal[1]); d[gy, gx] = 0; q = deque([(gx, gy)])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            u, v = x + dx, y + dy
            if 0 <= u < W and 0 <= v < H and free[v, u] and d[v, u] < 0:
                d[v, u] = d[y, x] + 1; q.append((u, v))
    return d[int(start[1]), int(start[0])]


def starts():
    if CACHE.exists():
        z = np.load(CACHE); return {k: z[k] for k in z}
    from rounds import load_data, train_starts, test_episodes, make_policy
    import json as _j
    pc = _j.loads((HERE.parent / 'maze_quadrants_delta_20261019/runs/11br/dt_st_step__clean/seed_0/config.json').read_text())['policy']
    ds = load_data(pc['shape_meta']); s, g, free = train_starts(ds)
    demo = np.array([int(np.asarray(ds[i]['attention_mask']).sum()) if 'attention_mask' in ds[i] else -1 for i in range(len(ds))])
    te = test_episodes(); ts = np.stack([e['start'] for e in te]); tg = np.stack([e['goal'] for e in te])
    out = dict(train_start=s, train_goal=g, test_start=ts, test_goal=tg, free=free, train_demo_len=demo,
               train_len=np.array([bfs_len(free, a, b) for a, b in zip(s, g)]), test_len=np.array([bfs_len(free, a, b) for a, b in zip(ts, tg)]))
    np.savez(CACHE, **out); return out


def per_round(z, split):
    s = z[f'{split}_success']                    # rounds.py: (R, starts, seeds); asteroid.py: (R, attempts, starts, seeds)
    if s.ndim == 4:
        s = np.stack([s[m, min(m, s.shape[1] - 1)] for m in range(s.shape[0])])
    return s.mean(-1), z[f'{split}_bc_success'].mean(-1)


def main(tags):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    st = starts()
    if (st['train_demo_len'] >= 0).all():
        print('path length vs demo length (train): exact match', float((st['train_len'] == st['train_demo_len'] - 1).mean()),
              'or', float((st['train_len'] == st['train_demo_len']).mean()))
    for tag in tags:
        out = HERE / 'runs' / tag; z = np.load(out / 'eval_seeds8.npz'); R = per_round(z, 'train')[0].shape[0]
        cols = plt.cm.viridis(np.linspace(0, .95, R)); names = {'train': 'training starts (637)', 'test': 'test starts (100, held-out quadrant)'}
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.6)); fig2, axs2 = plt.subplots(2, 1, figsize=(15, 6.5), gridspec_kw=dict(height_ratios=[3, 1.4]))
        for i, split in enumerate(['train', 'test']):
            s, bc = per_round(z, split); L = st[f'{split}_len']
            edges = np.array([0, 5, 10, 15, 20, 25, 30, 40, 100]); b = np.digitize(L, edges) - 1; ok = [k for k in range(len(edges) - 1) if (b == k).sum() >= 3]
            xs = [(edges[k] + min(edges[k + 1], L.max() + 1)) / 2 for k in ok]; ax = axs[i]
            for m in range(R):
                ax.plot(xs, [s[m, b == k].mean() for k in ok], 'o-', ms=3, color=cols[m], label=f'round {m}: {s[m].mean():.2f}')
            ax.plot(xs, [bc[b == k].mean() for k in ok], 's--', ms=3, color='#D55E00', label=f'BC (UNet): {bc.mean():.2f}')
            for x, k in zip(xs, ok):
                ax.annotate(f'n={int((b == k).sum())}', (x, 1.02), ha='center', fontsize=6, color='gray')
            ax.set_xlabel('shortest path from start to goal (steps)'); ax.set_ylabel('mean success'); ax.set_ylim(-.03, 1.08)
            ax.set_title(names[split], fontsize=9); ax.grid(alpha=.3); ax.legend(fontsize=6.5, ncol=2)
            o = np.lexsort((-s.mean(0), L)); M = np.vstack([bc[None], s])[:, o]
            ax = axs2[i]; im = ax.imshow(M, aspect='auto', cmap='magma', vmin=0, vmax=1, interpolation='nearest')
            ax.set_yticks(range(R + 1)); ax.set_yticklabels(['BC (UNet)'] + [f'round {m}' for m in range(R)], fontsize=7)
            ticks = [np.searchsorted(L[o], v) for v in (5, 10, 15, 20, 25, 30)]; ticks = [t for t in ticks if t < len(o)]
            ax.set_xticks(ticks); ax.set_xticklabels([f'≥{v}' for v, t in zip((5, 10, 15, 20, 25, 30), ticks) if t < len(o)][:len(ticks)], fontsize=7)
            ax.set_xlabel(f'{names[split]}, sorted by path length to goal (ticks = path length)', fontsize=8)
        fig.colorbar(im, ax=axs2, fraction=.02, pad=.01, label='success over 8 seeds')
        fig.suptitle(f'{tag}: success vs distance to goal', fontsize=10); fig.tight_layout(); fig.savefig(out / 'success_vs_distance.png', dpi=130); plt.close(fig)
        fig2.suptitle(f'{tag}: per-start success (each column = one start)', fontsize=10); fig2.savefig(out / 'success_heatmap.png', dpi=130, bbox_inches='tight'); plt.close(fig2)
        print(tag, 'written')


def bc_ref(tag, split='test'):
    for f in [HERE / 'runs' / tag / 'ctx_eval_seeds8.npz', HERE / 'runs' / tag / 'eval_seeds8.npz', HERE / 'runs/spi20/eval_seeds8.npz']:
        if f.exists() and f'{split}_bc_success' in np.load(f):
            return float(np.load(f)[f'{split}_bc_success'].mean())


def context_plot(tag):
    """Per round policy: success of its n-th consecutive attempt (n-1 own attempts in context), n = 1, 2, 4, 8."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    out = HERE / 'runs' / tag; f = out / 'ctx_eval_seeds8.npz'
    f = f if f.exists() else out / 'eval_seeds8.npz'; z = np.load(f)
    splits = [sp for sp in ('test', 'train') if f'{sp}_success' in z and z[f'{sp}_success'].ndim == 4]
    if not splits:
        return
    fig, axs = plt.subplots(len(splits), 2, figsize=(13, 4.4 * len(splits)), squeeze=False); res = {}
    for i, sp in enumerate(splits):
        S = z[f'{sp}_success'].mean((2, 3)); R, A = S.shape; bc = float(z[f'{sp}_bc_success'].mean()); ns = [n for n in (1, 2, 4, 8) if n <= A]
        nc = plt.cm.plasma(np.linspace(0, .85, len(ns))); rc = plt.cm.viridis(np.linspace(0, .95, R))
        ax = axs[i, 0]
        for c, n in zip(nc, ns):
            ax.plot(range(R), S[:, n - 1], 'o-', color=c, label=f'attempt {n} ({n - 1} own earlier in context)')
        ax.axhline(bc, color='#D55E00', ls='--', label=f'BC (UNet): {bc:.2f}')
        ax.set_xlabel('round policy (trained with ≤ n contexts at round n)'); ax.set_ylabel('success'); ax.set_ylim(-.03, 1.03); ax.grid(alpha=.3)
        ax.legend(fontsize=7); ax.set_title(f'{sp} starts: does more context help each policy?', fontsize=9)
        ax = axs[i, 1]
        for m in range(R):
            ax.plot(range(1, A + 1), S[m], 'o-', ms=3, color=rc[m], label=f'round {m}')
        ax.axhline(bc, color='#D55E00', ls='--', label='BC (UNet)')
        ax.set_xlabel('attempt n (n-1 own earlier attempts in context)'); ax.set_ylabel('success'); ax.set_ylim(-.03, 1.03); ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=2); ax.set_title(f'{sp} starts: success of the n-th attempt, per round policy', fontsize=9)
        res[sp] = dict(bc=round(bc, 4), by_round={m: [round(float(v), 4) for v in S[m]] for m in range(R)})
    win = int(z['window']) if 'window' in z else None
    fig.suptitle(f'{tag}: success vs context length (each policy makes {A} attempts in a row, own earlier attempts in context'
                 + (f', at most {win})' if win is not None else ')'), fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'success_vs_context.png', dpi=130); plt.close(fig)
    (out / 'context_summary.json').write_text(json.dumps(res, indent=1)); print(tag, 'context', json.dumps(res))


def curve_plot(tag):
    """Test-start success logged during training (asteroid.py --test-every): one line per round."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    out = HERE / 'runs' / tag; rows = [json.loads(l) for l in (out / 'log.jsonl').read_text().splitlines() if 'test_success' in l]
    if not rows:
        return
    R = max(r['round'] for r in rows) + 1; cols = plt.cm.viridis(np.linspace(0, .95, max(R, 2))); bc = bc_ref(tag)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    for m in range(R):
        rr = sorted((r for r in rows if r['round'] == m), key=lambda r: r['update'])
        axs[0].plot([r['update'] for r in rr], [r['test_success'] for r in rr], '-', marker='o', ms=2.5, color=cols[m], label=f'round {m} ({m} own traj. in context)')
        off = m * max(r['update'] for r in rr) if rr else 0
        axs[1].plot([off + r['update'] for r in rr], [r['test_success'] for r in rr], '-', color=cols[m])
    for ax in axs:
        if bc is not None:
            ax.axhline(bc, color='#D55E00', ls='--', label=f'BC (UNet): {bc:.2f}')
        ax.set_ylabel('test-start success'); ax.set_ylim(-.03, 1.03); ax.grid(alpha=.3)
    axs[0].set_xlabel('update within the round'); axs[0].legend(fontsize=7, ncol=2); axs[0].set_title('per round (update 0 = previous round\'s final weights)', fontsize=9)
    axs[1].set_xlabel('update, all rounds in sequence'); axs[1].set_title('whole run', fontsize=9)
    fig.suptitle(f'{tag}: test-start success during training (100 starts × 8 seeds, round n scored on attempt n+1)', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'test_learning_curve.png', dpi=130); plt.close(fig); print(tag, 'curve written')


def run_info(tag):
    """(description, rollouts per training start collected before STk is trained, as a function of k)."""
    c = json.loads((HERE / 'runs' / tag / 'config.json').read_text())
    if 'updates' in c:                                   # asteroid.py: round j collects a chain of j trajectories per start
        cps = c.get('ctx_per_start', 1)
        return (f"current-policy context chains, {c['updates']} updates/round", lambda k: cps * k * (k + 1) // 2)
    upr = c['iters'] * c['steps_per_iter']               # rounds.py: one trajectory per start per round
    return (f"{c['iters']} iterations × {c['steps_per_iter']} gradient step(s) = {upr} updates/round", lambda k: k)


def iter_plot(tag):
    """Two panels (training starts | test starts): x = DAgger iterations × rollouts (training-start rollouts collected before
    STk is trained); y = mean success. STkn = STk (policy after DAgger iteration k) on its n-th attempt in a row, n-1 own
    earlier attempts in context; black = STk at the context it was trained with (n = k+1); BC = fixed reference."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    out = HERE / 'runs' / tag; zs = [np.load(f) for f in (out / 'ctx_eval_seeds8.npz', out / 'eval_seeds8.npz') if f.exists()]
    if not zs:
        return
    desc, rps = run_info(tag); NS = 637
    names = {'train': 'TRAIN: the 637 training episodes (goals outside the bottom-right quadrant)',
             'test': 'TEST: 100 held-out episodes (goals in the bottom-right quadrant)'}
    fig, axs = plt.subplots(1, 2, figsize=(14, 5)); res = {}
    for ax, sp in zip(axs, ('train', 'test')):
        z4 = next((z for z in zs if f'{sp}_success' in z and z[f'{sp}_success'].ndim == 4), None)
        z3 = next((z for z in zs if f'{sp}_success' in z), None)
        if z3 is None:
            ax.set_title(f'{names[sp]}: not evaluated', fontsize=8); continue
        bc = float(z3[f'{sp}_bc_success'].mean()); r = res[sp] = dict(bc=round(bc, 4))
        if z4 is not None:
            S = z4[f'{sp}_success'].mean((2, 3)); R, A = S.shape
            own = [S[k, min(k, A - 1)] for k in range(R)]
        else:
            own = list(z3[f'{sp}_success'].mean((1, 2))); R = len(own)
        x = [rps(k) * NS for k in range(R)]; r['rollouts'] = x
        if z4 is not None:
            ns = list(range(1, A + 1))
            for c, n in zip([tuple(.62 * v for v in plt.cm.tab10(i)[:3]) for i in range(len(ns))], ns):  # tab10, darkened
                ax.plot(x, S[:, n - 1], 'o-', ms=4, lw=1.8, color=c, label=f'STk{n}: final {S[-1, n - 1]:.2f}')
                r[f'STk{n}'] = [round(float(v), 4) for v in S[:, n - 1]]
        ax.plot(x, own, ':', marker='x', ms=7, lw=2, color='k', label=f'STk(k+1) (its training context): final {own[-1]:.2f}')
        r['STk(k+1)'] = [round(float(v), 4) for v in own]
        ax.axhline(bc, color='#7F0000', ls='--', lw=2.2, label=f'BC (UNet, demos only, no DAgger): {bc:.2f}')
        ax.set_xticks(x); ax.set_xticklabels([f'{v}\nk={k}' for k, v in enumerate(x)], fontsize=7)
        ax.set_xlabel('DAgger iterations × rollouts  (training-start rollouts collected before STk is trained)', fontsize=8)
        ax.set_ylabel('mean success (starts × 8 seeds)'); ax.set_ylim(-.03, 1.03); ax.grid(alpha=.3); ax.legend(fontsize=6.5, ncol=2)
        ax.set_title(names[sp], fontsize=8)
    fig.suptitle(f'{tag} ({desc})   STkn = policy after DAgger iteration k, n-th attempt in a row (n-1 own earlier attempts in context)',
                 fontsize=9)
    fig.tight_layout(); fig.savefig(out / 'success_vs_iteration.png', dpi=130); plt.close(fig)
    (out / 'iteration_summary.json').write_text(json.dumps(res, indent=1)); print(tag, 'iteration', json.dumps(res))


DARK = lambda i: tuple(.62 * v for v in __import__('matplotlib').cm.tab10(i % 10)[:3])   # tab10, darkened
BC_C = '#7F0000'
SPLIT_T = {'train': 'TRAIN starts: the 637 training episodes (goals outside the bottom-right quadrant)',
           'test': 'TEST starts: 100 held-out episodes (goals in the bottom-right quadrant)'}


def maze_split(path=HERE / 'report' / 'maze_split.png'):
    """One panel: walls opaque black; train / test start and goal cells; held-out quadrant boxed."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    st = starts(); free = st['free']; H, W = free.shape; cx, cy = W // 2, H // 2
    fig, ax = plt.subplots(figsize=(7.2, 8))
    ax.imshow(~free, cmap=ListedColormap(['white', 'black']), vmin=0, vmax=1, origin='upper', extent=(-.5, W - .5, H - .5, -.5))
    ax.add_patch(Rectangle((cx + .5, cy + .5), W - cx - 1, H - cy - 1, fill=False, ec='#B35800', lw=3, ls='--', zorder=4))
    for v in range(W + 1):
        ax.axvline(v - .5, color='#BBBBBB', lw=.5, zorder=1); ax.axhline(v - .5, color='#BBBBBB', lw=.5, zorder=1)
    cells = lambda pts: np.unique(np.round(pts).astype(int), axis=0)
    spec = [('train_goal', '#0B3D91', '*', 260, (0, .08), 'train goals'), ('test_goal', '#B35800', '*', 260, (0, .08), 'test goals'),
            ('train_start', '#0B6E2F', 'o', 45, (-.3, -.3), 'train starts'), ('test_start', '#5B1A8B', 's', 45, (.3, -.3), 'test starts')]
    for k, c, m, sz, (dx, dy), lab in spec:
        u = cells(st[k]); ax.scatter(u[:, 0] + dx, u[:, 1] + dy, s=sz, c=c, marker=m, edgecolors='white' if m == '*' else c, linewidths=.6,
                                     label=f'{lab}: {len(st[k])} episodes on {len(u)} cells', zorder=5)
    ax.set_xlim(-.5, W - .5); ax.set_ylim(H - .5, -.5); ax.set_xticks(range(W)); ax.set_yticks(range(H)); ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.04), ncol=2, frameon=False)
    Ltr, Lte = st['train_len'], st['test_len']
    ax.set_title('11×11 maze (black = wall). Dashed box = held-out bottom-right quadrant:\n'
                 'test goals only inside it, train goals only outside it (centre row/column never a goal); starts anywhere.\n'
                 f'Shortest path start→goal: train median {int(np.median(Ltr))} (max {int(Ltr.max())}), '
                 f'test median {int(np.median(Lte))} (max {int(Lte.max())}) steps', fontsize=8.5)
    fig.tight_layout(); path.parent.mkdir(exist_ok=True); fig.savefig(path, dpi=140, bbox_inches='tight'); plt.close(fig); print('maze split written')


def rows(tag):
    return [json.loads(l) for l in (HERE / 'runs' / tag / 'log.jsonl').read_text().splitlines() if 'train_success' in l]


def round_curves(tag, label, path):
    """x = iteration within the round (0 = previous round's final weights); one line per STn (n earlier trajectories in
    context, made by ST0..ST(n-1)); two panels: train / test starts. BC dashed."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rr = rows(tag); R = max(r['round'] for r in rr) + 1
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, sp in zip(axs, ('train', 'test')):
        for n in range(R):
            q = list({r['update']: r for r in rr if r['round'] == n}.values()); q.sort(key=lambda r: r['update'])   # last record per update (requeue reruns a cycle)
            ax.plot([r['update'] for r in q], [r[f'{sp}_success'] for r in q], '-', lw=1.8, color=DARK(n), label=f'STk{n} (context = {n})')
        ax.axhline(bc_ref('spi20', sp), color=BC_C, ls='--', lw=2, label='BC (UNet, demos only)')
        c = json.loads((HERE / 'runs' / tag / 'config.json').read_text()); spi = c['steps_per_iter'] or c['test_every']
        ax.set_title(SPLIT_T[sp], fontsize=9); ax.set_xlabel(f'gradient steps within collect-and-train cycle k (eval every {spi})', fontsize=8.5)
        ax.set_ylim(0, 1); ax.grid(alpha=.25)
    axs[0].set_ylabel('success (mean over starts × seeds)'); axs[1].legend(fontsize=7.5, ncol=2, loc='upper left')
    fig.suptitle(label, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); print(tag, 'round curves written')


def compare(runs, path):
    """x = rollouts collected; y = success of STk at the end of round k (from the in-training eval); BC."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True); out = {}
    for ax, sp in zip(axs, ('train', 'test')):
        for i, (tag, lab) in enumerate(runs.items()):
            rr = rows(tag); last = max(r['iter'] for r in rr)
            rr = sorted((r for r in rr if r['iter'] == last), key=lambda r: r['round'])
            cps = json.loads((HERE / 'runs' / tag / 'config.json').read_text()).get('ctx_per_start', 1)
            x = [637 * cps * r['round'] * (r['round'] + 1) // 2 for r in rr]; y = [r[f'{sp}_success'] for r in rr]
            ax.plot(x, y, 'o-', ms=5, lw=2, color=DARK(i), label=f'{lab}: last {y[-1]:.2f}'); out.setdefault(tag, {})[sp] = y
        bc = bc_ref('spi20', sp); ax.axhline(bc, color=BC_C, ls='--', lw=2, label=f'BC (UNet, demos only): {bc:.2f}')
        ax.set_xlabel('own rollouts collected from training starts before STk is trained (summed over cycles)', fontsize=8.5)
        ax.set_title(SPLIT_T[sp], fontsize=9); ax.set_ylim(0, 1); ax.grid(alpha=.25); ax.legend(fontsize=8)
    axs[0].set_ylabel('success of STk (end of collect-and-train cycle k)')
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); print('compare written', json.dumps(out))

if __name__ == '__main__':
    tags = sys.argv[1:]
    for t in tags:
        if (HERE / 'runs' / t / 'eval_seeds8.npz').exists():
            main([t])
        context_plot(t); curve_plot(t); iter_plot(t)


def goal_map(tag, path, src='eval_seeds8.npz'):
    """3×3 mazes (BC, STk0..STk7): every goal cell filled with the success rate of the episodes that have that goal
    (mean over those episodes × 8 seeds; STk = round-k weights on its attempt k+1, k own attempts in context). Walls black,
    cells that are never a goal white, held-out quadrant (test goals) in the dashed box. From runs/<tag>/eval_seeds8.npz."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    z = np.load(HERE / 'runs' / tag / src)
    if 'test_st_success' in z:   # bon_eval.py output: STk attempts (R, A, starts, seeds); BC = its first candidate per seed
        z = {k: v for sp in ('train', 'test') for k, v in ((f'{sp}_success', z[f'{sp}_st_success']), (f'{sp}_bc_success', z[f'{sp}_bc_success'][..., 0]))}
    st = starts(); free = st['free']; H, W = free.shape; cx, cy = W // 2, H // 2
    tr, bc_tr = per_round(z, 'train'); te, bc_te = per_round(z, 'test')
    pols = [('BC (UNet, demos only)', bc_tr, bc_te)] + [(f'STk{k} (context = {k})', tr[k], te[k]) for k in range(len(tr))]
    cmap = plt.get_cmap('RdYlGn'); nr = int(np.ceil(len(pols) / 3))
    fig, axs = plt.subplots(nr, 3, figsize=(13.5, 4.6 * nr), layout='constrained'); axs = axs.ravel()
    for ax, (name, s_tr, s_te) in zip(axs, pols):
        img = np.ones((H, W, 3)); img[~free] = 0
        for G, s in ((st['train_goal'], s_tr), (st['test_goal'], s_te)):
            g = np.round(G).astype(int)
            for c in np.unique(g, axis=0):
                img[c[1], c[0]] = cmap(float(s[(g == c).all(1)].mean()))[:3]
        ax.imshow(img, origin='upper', extent=(-.5, W - .5, H - .5, -.5), interpolation='nearest')
        ax.add_patch(Rectangle((cx + .5, cy + .5), W - cx - 1, H - cy - 1, fill=False, ec='#3A1F00', lw=2.5, ls='--'))
        for v in range(W + 1):
            ax.axvline(v - .5, color='#999999', lw=.4); ax.axhline(v - .5, color='#999999', lw=.4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{name}\ntrain goals {s_tr.mean():.2f}   ·   test goals (box) {s_te.mean():.2f}', fontsize=10)
    for ax in axs[len(pols):]:
        ax.axis('off')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axs.tolist(), shrink=.6, pad=.01, label='success rate of episodes with this goal (red = fail, green = success)')
    fig.suptitle(f'{tag}: success per goal cell (black = wall, white = never a goal, dashed box = held-out test goals)', fontsize=11)
    fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close(fig); print(tag, 'goal map written')


CTRL = {'ast_s20': ('main run: 20 steps per batch, context = k own attempts with feedback', '#000000', '-'),
        'ast_s30': ('30 steps per batch (context with feedback)', '#009E73', '-'),
        'ast_s20_nofb': ('context without feedback (20 steps per batch)', '#0072B2', '-'),
        'ast_k0long': ('no context ever: one long round of demos only (20 steps per batch)', '#D55E00', '-'),
        'ast_s20_on400': ('+ training obs noise, flat t = 400', '#CC79A7', '-'),
        'ast_s20_onramp': ('+ training obs noise, ramp t = 600 (no context) → 200 (7 attempts)', '#E69F00', '-')}
CTRL_ROWS = (('Does the context help? (same total gradient steps)', ('ast_s20', 'ast_s30', 'ast_s20_nofb', 'ast_k0long')),
             ('Training observation noise (otherwise = main run)', ('ast_s20', 'ast_s20_on400', 'ast_s20_onramp')))


def steps_compare(tags, path):
    """x = total gradient steps since the start of training; y = mean of the last 5 evals of the current weights with the current
    round's context (k own attempts in round k; always none for ast_k0long). Dots = end of round k (the STk numbers in the tables).
    Row 1 = context controls, row 2 = observation noise; the main run (black) is in both rows."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.4), sharey=True)
    for row, (rtitle, rtags) in zip(axs, CTRL_ROWS):
        for t in rtags:
            if t not in tags:
                continue
            rr = rows(t); cfg = json.loads((HERE / 'runs' / t / 'config.json').read_text()); upr = cfg['iters'] * cfg['steps_per_iter']
            lab, c, ls = CTRL[t]
            for ax, sp in zip(row, ('train', 'test')):
                x = np.array([r['round'] * upr + r['update'] for r in rr]); y = np.array([r[f'{sp}_success'] for r in rr])
                sm = np.array([y[max(0, i - 4):i + 1].mean() for i in range(len(y))])
                ax.plot(x, sm, color=c, lw=2.4 if t == 'ast_s20' else 2, ls=ls, label=lab, zorder=3 if t == 'ast_s20' else 2)
                ends = [i for i, r in enumerate(rr) if r['update'] == upr]
                ax.plot(x[ends], sm[ends], 'o', color=c, ms=5, zorder=4)
        for ax, sp in zip(row, ('train', 'test')):
            ax.axhline(bc_ref('spi20', sp), color='#777777', ls='--', lw=1.6, label='BC (UNet, demos only)')
            ax.set_title(f'{rtitle}\n' + SPLIT_T[sp], fontsize=9); ax.set_ylim(0, 1); ax.grid(alpha=.25)
            ax.set_xlabel('total gradient steps since the start of training', fontsize=9)
        row[0].set_ylabel('success, mean of the last 5 evals'); row[1].legend(fontsize=7.5, loc='upper left')
    fig.suptitle('Same total training. Each line = one run (mean of its last 5 evals); dot = end of round k = STk', fontsize=10.5)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); print('steps compare written')


def feedback_compare(path, runs=(('ast_s20', 'with feedback', '#000000'), ('ast_s20_nofb', 'without feedback', '#0072B2'))):
    """Context with vs without feedback (both 20 steps per batch). x = round k; y = success of the round-k weights, mean of the
    last 5 in-training evals of round k (single seed each). Solid = attempt k+1 (k own attempts in context, = STk);
    dotted = attempt 1 of the same weights (no context). BC dashed."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True); out = {}
    for ax, sp in zip(axs, ('train', 'test')):
        for tag, lab, c in runs:
            by = {}
            for r in rows(tag):
                if f'{sp}_by_attempt' in r: by.setdefault(r['round'], []).append(r)
            ks = sorted(by); last = [by[k][-5:] for k in ks]
            yk = [float(np.mean([r[f'{sp}_by_attempt'][-1] for r in rs])) for rs in last]
            y1 = [float(np.mean([r[f'{sp}_by_attempt'][0] for r in rs])) for rs in last]
            ax.plot(ks, yk, 'o-', color=c, lw=2, ms=5, label=f'{lab}: attempt k+1 (k own attempts in context) = STk')
            ax.plot(ks, y1, 'o:', color=c, lw=1.6, ms=4, alpha=.8, label=f'{lab}: attempt 1 of the same weights (no context)')
            out.setdefault(tag, {})[sp] = {'attempt_k1': [round(v, 3) for v in yk], 'attempt_1': [round(v, 3) for v in y1]}
        bc = bc_ref('spi20', sp); ax.axhline(bc, color='#777777', ls='--', lw=1.6, label='BC (UNet, demos only)')
        ax.set_xlabel('round k (STk = round-k weights; mean of the last 5 evals of the round)', fontsize=8.5)
        ax.set_title(SPLIT_T[sp], fontsize=9); ax.set_ylim(0, 1); ax.grid(alpha=.25)
    axs[0].set_ylabel('success')
    fig.suptitle('Context with vs without feedback (20 gradient steps per batch)', fontsize=11)
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .1, 1, 1)); fig.savefig(path, dpi=140); plt.close(fig); print('feedback compare written', json.dumps(out))


def bon(success, bfs, axis):
    """Best-of-n with the verifier (-BFS of the final cell; first candidate on ties) over the first n candidates along `axis`,
    for n = 1..A. Returns (A,) + the remaining shape, as success of the executed candidate."""
    s, b = np.moveaxis(success, axis, 0), np.moveaxis(bfs, axis, 0); out = []
    for n in range(1, s.shape[0] + 1):
        i = np.argmin(b[:n], 0); out.append(np.take_along_axis(s[:n], i[None], 0)[0])
    return np.stack(out)


def bon_plots(tag, path_curves, path_line, title):
    """From runs/<tag>/bon_seeds8.npz. Curves: x = n candidates (1..8), BoN success of BC (n iid samples) and of STk0..7 (n
    attempts in a row, attempt j with j-1 own attempts in context). Line: STk at n = k+1 candidates (its last candidate has
    k attempts in context, the length it was trained for), BC at the same n."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    z = np.load(HERE / 'runs' / tag / 'bon_seeds8.npz'); splits = [sp for sp in ('train', 'test') if f'{sp}_st_success' in z]
    cols = plt.cm.viridis(np.linspace(0, .92, 8)); out = {}
    f1, a1 = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True); f2, a2 = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    for i, sp in enumerate(('train', 'test')):
        if sp not in splits:
            a1[i].set_title(SPLIT_T[sp] + '\n(eval running)', fontsize=9); a2[i].set_title(SPLIT_T[sp] + '\n(eval running)', fontsize=9); continue
        st = bon(z[f'{sp}_st_success'], z[f'{sp}_st_bfs'], 1).mean((-1, -2))          # (n, R)
        bc = bon(z[f'{sp}_bc_success'], z[f'{sp}_bc_bfs'], 2).mean((-1, -2))          # (n,)
        R, A = st.shape[1], st.shape[0]; n = np.arange(1, A + 1)
        for m in range(R):
            a1[i].plot(n, st[:, m], 'o-', ms=3.5, lw=1.6, color=cols[m], label=f'STk{m} (trained with {m} attempts in context)')
        a1[i].plot(n, bc, 's--', ms=4, lw=2, color='#555555', label='BC (UNet): n iid samples')
        diag = np.array([st[min(m, A - 1), m] for m in range(R)]); x = np.arange(1, R + 1)
        a2[i].plot(x, diag, 'o-', lw=2.2, ms=6, color='#000000', label='STk at n = k+1 candidates (k attempts in context, as trained)')
        for m in range(R):
            a2[i].annotate(f'STk{m}', (x[m], diag[m]), textcoords='offset points', xytext=(0, -14 if diag[m] > .9 else 7), ha='center', fontsize=7.5)
        a2[i].plot(n, bc, 's--', lw=2, ms=5, color='#777777', label='BC (UNet): best of n iid samples')
        for ax in (a1[i], a2[i]):
            ax.set_title(SPLIT_T[sp], fontsize=9); ax.set_ylim(0, 1.02); ax.grid(alpha=.25); ax.set_xticks(n)
        a1[i].set_xlabel('n candidates (best of n with the verifier, the chosen candidate is executed)', fontsize=8.5)
        a2[i].set_xlabel('n candidates = k+1 for STk (the last candidate has k attempts in context)', fontsize=8.5)
        out[sp] = {'bc': bc.round(3).tolist(), 'stk_at_k1': diag.round(3).tolist(), 'st': st.T.round(3).tolist()}
    a1[0].set_ylabel('success of the executed candidate'); a2[0].set_ylabel('success of the executed candidate')
    f1.legend(*a1[0].get_legend_handles_labels() if a1[0].lines else a1[1].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=8, frameon=False)
    f2.legend(*a2[0].get_legend_handles_labels() if a2[0].lines else a2[1].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=8, frameon=False)
    f1.suptitle(f'{title}: best of n candidates with the verifier (−BFS of the final cell), STk0 … STk7 and BC', fontsize=10.5)
    f2.suptitle(f'{title}: each STk at its trained context length (n = k+1), BC at the same n', fontsize=10.5)
    f1.tight_layout(rect=(0, .13, 1, 1)); f1.savefig(path_curves, dpi=140); plt.close(f1)
    f2.tight_layout(rect=(0, .08, 1, 1)); f2.savefig(path_line, dpi=140); plt.close(f2); print('bon plots written', json.dumps(out))


def perf_at_n(tag, path, title):
    """From runs/<tag>/bon_seeds8.npz: x = attempt n (1..8); y = success of attempt n alone, no selection (STk: n-1 own earlier
    attempts in context; BC: its n-th iid sample). Same layout as the best-of-n curves."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    z = np.load(HERE / 'runs' / tag / 'bon_seeds8.npz'); cols = plt.cm.viridis(np.linspace(0, .92, 8)); out = {}
    f, a = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for i, sp in enumerate(('train', 'test')):
        if f'{sp}_st_success' not in z:
            a[i].set_title(SPLIT_T[sp] + '\n(eval running)', fontsize=9); continue
        st = z[f'{sp}_st_success'].mean((-1, -2)); bc = z[f'{sp}_bc_success'].mean((0, 1)); R, A = st.shape; n = np.arange(1, A + 1)
        for m in range(R):
            a[i].plot(n, st[m], 'o-', ms=3.5, lw=1.6, color=cols[m], label=f'STk{m} (trained with {m} attempts in context)')
        a[i].plot(n, bc, 's--', ms=4, lw=2, color='#555555', label='BC (UNet): n-th iid sample')
        a[i].set_title(SPLIT_T[sp], fontsize=9); a[i].set_ylim(0, 1.02); a[i].grid(alpha=.25); a[i].set_xticks(n)
        a[i].set_xlabel('attempt n (n − 1 own earlier attempts in context; no selection)', fontsize=8.5)
        out[sp] = {'bc': bc.round(3).tolist(), 'st': st.round(3).tolist()}
    a[0].set_ylabel('success of attempt n')
    h = a[0] if a[0].lines else a[1]; f.legend(*h.get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=8, frameon=False)
    f.suptitle(f'{title}: perf@n, success of attempt n alone, STk0 … STk7 and BC', fontsize=10.5)
    f.tight_layout(rect=(0, .13, 1, 1)); f.savefig(path, dpi=140); plt.close(f); print('perf@n written', json.dumps(out))


def perf_bon(tag, path, title, src='bon_seeds8.npz'):
    """From runs/<tag>/bon_seeds8.npz. Main (left, wide): perf@n, success of attempt n alone (STk: n-1 own attempts in context;
    BC: its n-th iid sample). Side (right, narrow): best of the first n with the verifier (-BFS of the final cell), executed.
    Rows: train, test starts."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    z = np.load(HERE / 'runs' / tag / src); cols = plt.cm.viridis(np.linspace(0, .92, 8))
    f, a = plt.subplots(2, 2, figsize=(13, 9.6), gridspec_kw={'width_ratios': [2.3, 1]}, sharey=True)
    for r, sp in enumerate(('train', 'test')):
        if f'{sp}_st_success' not in z:
            a[r, 0].set_title(SPLIT_T[sp] + '\n(eval running)', fontsize=9); continue
        S, Bf = z[f'{sp}_st_success'], z[f'{sp}_st_bfs']; C, Cb = z[f'{sp}_bc_success'], z[f'{sp}_bc_bfs']
        perf, perf_bc = S.mean((-1, -2)), C.mean((0, 1))                                  # (R, A), (A,)
        best, best_bc = bon(S, Bf, 1).mean((-1, -2)), bon(C, Cb, 2).mean((-1, -2))         # (n, R), (n,)
        R, A = perf.shape; n = np.arange(1, A + 1)
        for m in range(R):
            a[r, 0].plot(n, perf[m], 'o-', ms=4, lw=1.9, color=cols[m], label=f'STk{m} (trained with {m} attempts in context)')
            a[r, 1].plot(n, best[:, m], 'o-', ms=2.5, lw=1.2, color=cols[m])
        a[r, 0].plot(n, perf_bc, 's--', ms=4.5, lw=2.2, color='#555555', label='BC (UNet): n-th iid sample / best of n iid samples')
        a[r, 1].plot(n, best_bc, 's--', ms=3, lw=1.6, color='#555555')
        a[r, 0].set_title('perf@n — ' + SPLIT_T[sp], fontsize=9); a[r, 1].set_title('best of n with the verifier', fontsize=9)
        a[r, 0].set_xlabel('attempt n (n − 1 own earlier attempts in context; no selection)', fontsize=8.5)
        a[r, 1].set_xlabel('n candidates (verifier picks one, it is executed)', fontsize=8)
        for ax in a[r]:
            ax.set_ylim(0, 1.02); ax.grid(alpha=.25); ax.set_xticks(n)
        a[r, 0].set_ylabel('success')
    h = a[0, 0] if a[0, 0].lines else a[1, 0]
    f.legend(*h.get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=8, frameon=False)
    import textwrap
    f.suptitle('\n'.join(textwrap.wrap(f'{title}: perf@n (left) and best of n (right), STk0 … STk{R - 1} and BC', 120)), fontsize=10.5)
    f.tight_layout(rect=(0, .07, 1, 1)); f.savefig(path, dpi=140); plt.close(f); print(tag, 'perf + bon written')


def bfs_per_attempt(path, runs=(('ast_s30', '30 gradient steps per batch'), ('ast_main', '10k updates per collect-and-train cycle'))):
    """Test starts, from runs/<tag>/bon_seeds8.npz. Top: mean final-cell BFS of attempt n (STk: n-1 own attempts in context;
    BC: n-th iid sample). Bottom: STk7 only, split by goal cell: cells STk7 solves at attempt 8 in >= 50% of episodes x seeds,
    in (0, 50%), and never."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    st = starts(); g = np.round(st['test_goal']).astype(int); cols = plt.cm.viridis(np.linspace(0, .92, 8))
    f, a = plt.subplots(2, len(runs), figsize=(13, 9.4), sharey='row')
    for c, (tag, lab) in enumerate(runs):
        z = np.load(HERE / 'runs' / tag / 'bon_seeds8.npz'); B, S = z['test_st_bfs'], z['test_st_success']; R, A = B.shape[:2]; n = np.arange(1, A + 1)
        for m in range(R):
            a[0, c].plot(n, B[m].mean((1, 2)), 'o-', ms=3.5, lw=1.7, color=cols[m], label=f'STk{m} (trained with {m} attempts in context)')
        a[0, c].plot(n, z['test_bc_bfs'].mean((0, 1)), 's--', ms=4, lw=2, color='#555555', label='BC (UNet): n-th iid sample')
        a[0, c].set_title(f'{lab}: mean final-cell BFS, all 100 test episodes', fontsize=9)
        cell = {tuple(x): S[7, 7][(g == x).all(1)].mean() for x in np.unique(g, axis=0)}; r = np.array([cell[tuple(x)] for x in g])
        for name, m_, col in ((f'goal cells STk7 solves in ≥ 50%', r >= .5, '#1F7A3A'), ('goal cells solved in (0, 50%)', (r > 0) & (r < .5), '#E69F00'),
                              ('goal cells never solved', r == 0, '#B2182B')):
            ncell = len({tuple(x) for x in g[m_]})
            a[1, c].plot(n, B[7][:, m_].mean((1, 2)), 'o-', lw=2, color=col, label=f'{name} ({ncell} cells, {int(m_.sum())} episodes)')
        a[1, c].set_title(f'{lab}: STk7, 8 attempts in a row, by goal cell', fontsize=9); a[1, c].legend(fontsize=7.5, loc='upper right')
        for ax in a[:, c]:
            ax.set_xticks(n); ax.grid(alpha=.25); ax.set_xlabel('attempt n (n − 1 own earlier attempts in context)', fontsize=8.5)
    for row in a:
        row[0].set_ylim(0, 1.08 * max(ax.dataLim.y1 for ax in row))
    a[0, 0].set_ylabel('mean BFS of the final cell'); a[1, 0].set_ylabel('mean BFS of the final cell (STk7)')
    f.legend(*a[0, 0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=8, frameon=False)
    f.suptitle('Test starts: maze distance (BFS) from the final cell to the goal, per attempt; 8 seeds; success = BFS ≤ 1', fontsize=10.5)
    f.tight_layout(rect=(0, .07, 1, 1)); f.savefig(path, dpi=140); plt.close(f); print('bfs per attempt written')
