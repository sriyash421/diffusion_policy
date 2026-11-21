"""Why does the average test success rise while most test goals stay unsolved? (ast_s20, STk2 vs STk7)

A. Per test episode / goal cell: success of STk2 (attempt 3) and STk7 (attempt 8) over 8 seeds, from runs/<tag>/eval_seeds8.npz;
   how much of the mean change comes from which goals.
B. One test position that both never solve: replay STk2 (3 attempts) and STk7 (8 attempts) in a row, attempt j with its j-1
   earlier attempts in context, and draw every attempt on the maze (one panel per attempt). Two figures.
  python fail_viz.py --tag ast_s20 [--episode i]
"""
import argparse, json
import numpy as np
import torch
import plot_clear as pc
from rounds import HERE, make_policy, load_data, train_starts
from asteroid import chain, test_set, drop_context_feedback, set_fb


def analysis(tag):
    st = pc.starts(); f = HERE / 'runs' / tag / 'eval_seeds8.npz'
    if f.exists():
        z = np.load(f); s = z['test_success']; bc = z['test_bc_success'].mean(-1)               # (R, A, episodes, seeds)
    else:   # bon_eval output: same ST layout, BC (episodes, seeds, candidates) -> first candidate
        z = np.load(HERE / 'runs' / tag / 'bon_seeds8.npz'); s = z['test_st_success']; bc = z['test_bc_success'][..., 0].mean(-1)
    s2, s7 = s[2, 2].mean(-1), s[7, 7].mean(-1)
    g = np.round(st['test_goal']).astype(int); cells = [tuple(c) for c in np.unique(g, axis=0)]
    print(f'test mean: BC {bc.mean():.3f}  STk2 {s2.mean():.3f}  STk7 {s7.mean():.3f}  change {s7.mean() - s2.mean():+.3f}')
    rows = []
    for c in cells:
        m = (g == np.array(c)).all(1)
        rows.append((c, int(m.sum()), s2[m].mean(), s7[m].mean(), (s7[m] - s2[m]).sum() / len(s2)))
    rows.sort(key=lambda r: -r[4])
    print('goal (x,y)  episodes  STk2  STk7  share of the mean change')
    for c, n, a, b, d in rows:
        print(f'{str(c):10s} {n:4d}     {a:.2f}  {b:.2f}  {d:+.3f}')
    never = np.where((s2 == 0) & (s7 == 0))[0]
    print(f'episodes never solved by STk2 or STk7 (0 of 8 seeds each): {len(never)} of {len(s2)}; also BC = 0: {int((bc[never] == 0).sum())}')
    ep_up, ep_dn = int((s7 > s2).sum()), int((s7 < s2).sum())
    print(f'episodes STk7 > STk2: {ep_up}, STk7 < STk2: {ep_dn}, equal: {len(s2) - ep_up - ep_dn}')
    return st, never, s2, s7


def shortest_path(free, start, goal):
    from bon_eval import bfs_table
    d = bfs_table(free, goal); x, y = int(round(start[0])), int(round(start[1])); out = [(x, y)]; H, W = free.shape
    while d[y, x] > 0:
        x, y = min(((x + a, y + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= x + a < W and 0 <= y + b < H and free[y + b, x + a]),
                   key=lambda c: d[c[1], c[0]]); out.append((x, y))
    return np.array(out, float)


def draw(path, free, start, goal, trajs, succ, bfs, title):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    n = len(trajs); nc = min(4, n); nr = int(np.ceil(n / nc)); H, W = free.shape
    fig, axs = plt.subplots(nr, nc, figsize=(3.3 * nc, 3.5 * nr + .5), squeeze=False, layout='constrained')
    for k, ax in enumerate(axs.ravel()):
        if k >= n:
            ax.axis('off'); continue
        img = np.where(free, 1., .15); ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='upper', extent=(-.5, W - .5, H - .5, -.5))
        sp = shortest_path(free, start, goal); ax.plot(sp[:, 0], sp[:, 1], '--', color='#2166AC', lw=1.6, alpha=.8)
        t = np.asarray(trajs[k], float); col = '#1F7A3A' if succ[k] else '#B2182B'
        ax.plot(t[:, 0], t[:, 1], '-', color=col, lw=2.2, alpha=.85)
        ax.scatter(t[1:-1, 0], t[1:-1, 1], s=10, color=col, zorder=3)
        ax.plot(*start, 'o', ms=10, mfc='#2166AC', mec='k', zorder=4); ax.plot(*goal, '*', ms=17, mfc='#FFD700', mec='k', zorder=4)
        ax.plot(t[-1, 0], t[-1, 1], 'X', ms=10, mfc=col, mec='k', zorder=5)
        ax.set_title(f'attempt {k + 1} ({k} in context): {"success" if succ[k] else "fail"}, final BFS {bfs[k]}, {len(t) - 1} moves', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title + '\nblue circle = start, star = goal, X = final cell, dashed blue = shortest path, dark = wall', fontsize=10)
    fig.savefig(path, dpi=130); plt.close(fig); print('written', path)


def main(args):
    st, never, s2, s7 = analysis(args.tag)
    free_np = st['free'].astype(bool); g = np.round(st['test_goal']).astype(int)
    if args.episode < 0:   # a never-solved episode whose goal cell has the most never-solved episodes
        cnt = {tuple(c): int(((g[never] == c).all(1)).sum()) for c in g[never]}
        best = max(cnt, key=cnt.get); args.episode = int(next(i for i in never if tuple(g[i]) == best))
    i = args.episode; print('episode', i, 'start', st['test_start'][i], 'goal', st['test_goal'][i], f'STk2 {s2[i]:.2f} STk7 {s7[i]:.2f}')
    device = 'cuda'; out = HERE / 'runs' / args.tag; cfg = json.loads((out / 'config.json').read_text())
    set_fb(cfg.get('fb'))
    if cfg.get('no_ctx_feedback'):
        drop_context_feedback()
    model, pcfg = make_policy(cfg['rounds'], device); ds = load_data(pcfg['shape_meta']); model.set_normalizer(ds.get_normalizer())
    _, _, free = train_starts(ds); free1 = torch.as_tensor(free, device=device)[None]; te, te_s, te_g = test_set(free)
    S, G = te_s[i:i + 1], te_g[i:i + 1]; cache = {}
    from bon_eval import bfs_of
    for k in (2, 7):
        model.load_state_dict(torch.load(out / f'round_{k}.pt', map_location='cpu', weights_only=False)['model']); model.to(device).eval()
        torch.manual_seed(args.seed)
        with torch.no_grad():
            mem, succ, _ = chain(model, free1, S, G, k + 1, cfg['rounds'] - 1, 512, device)
        trajs = mem[0]; bfs = bfs_of(free.astype(bool), [G[0]] * len(trajs), [t[-1] for t in trajs], cache)
        draw(HERE / 'report' / f'fail_{args.tag}_ep{i}_STk{k}.png', free.astype(bool), S[0], G[0], trajs, succ[:, 0], bfs,
             f'{args.tag} STk{k} on test episode {i} (STk2 {s2[i]:.0%}, STk7 {s7[i]:.0%} success over 8 seeds): '
             f'{k + 1} attempts in a row, attempt {k + 1} is the scored one')
        print(f'STk{k} attempts: success {succ[:, 0].astype(int).tolist()} final BFS {bfs.tolist()} moves {[len(t) - 1 for t in trajs]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='ast_s20'); ap.add_argument('--episode', type=int, default=-1)
    ap.add_argument('--seed', type=int, default=0); main(ap.parse_args())
