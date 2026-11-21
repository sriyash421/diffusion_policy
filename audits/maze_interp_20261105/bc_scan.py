"""BC scan on every (start, held-out goal) pair with 3 rollout seeds, then the 50-pair test set (BC failures first).

BC = the interp30 UNet at its train_heldout-selected step (bc_selection.json), evaluated with rounds.bc_eval
(open-loop H-step chunk, cardinal snap, success = radius 1), as every BC line in the rounds reports.
Test set: pairs sorted by BC successes out of 3 (0 first), random tie-break (seed 0), first 50.
  MAZE_SETTING=interp30 python bc_scan.py          (GPU job; run from anywhere)
Writes bc_scan.npz, test_episodes.json, report/bc_scan_goals.png.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MAZE_SETTING', 'interp30'); assert os.environ['MAZE_SETTING'] == 'interp30'
sys.path[:0] = [str(HERE.parent / 'maze_rounds_20261103')]
import rounds  # noqa: E402

SEEDS, N_TEST = 3, 50


def main():
    g = json.loads((HERE / 'goals.json').read_text()); nav = np.load(rounds.SPEC['layouts'])['nav'][0]
    free = np.argwhere(nav)[:, [1, 0]].astype(int); step = rounds.BC_CKPT['test']
    pairs = np.array([(s[0], s[1], gx, gy) for gx, gy in g['test'] for s in free if (s[0], s[1]) != (gx, gy)], np.int32)
    f = HERE / 'bc_scan.npz'
    if f.exists():
        succ = np.load(f)['success']
    else:
        torch.manual_seed(1105)
        succ = rounds.bc_eval(step, pairs[:, :2].astype(np.float32), pairs[:, 2:].astype(np.float32), None, SEEDS, 'cuda')
        np.savez_compressed(f, pairs=pairs, success=succ, step=step, seeds=SEEDS)
    k = succ.sum(1)                                                # BC successes out of 3, per pair
    print(f'BC step {step}: {len(pairs)} pairs, mean success {succ.mean():.3f}; pairs by successes/3:',
          {int(c): int((k == c).sum()) for c in range(SEEDS + 1)})
    order = np.lexsort((np.random.default_rng(0).permutation(len(pairs)), k))[:N_TEST]
    eps = [dict(start=pairs[i, :2].tolist(), goal=pairs[i, 2:].tolist(), bc_scan_successes=int(k[i]), bc_fail=bool(k[i] <= 1))
           for i in order]
    (HERE / 'test_episodes.json').write_text(json.dumps(dict(
        rule=f'all (start, held-out goal) pairs sorted by BC successes out of {SEEDS} (ascending), random tie-break seed 0, first {N_TEST}; '
             f'bc_fail = at most 1 success of {SEEDS}', bc_step=int(step), episodes=eps), indent=1))
    sel_k = k[order]
    print('test set by BC successes/3:', {int(c): int((sel_k == c).sum()) for c in range(SEEDS + 1)},
          '| bc_fail', int((sel_k <= 1).sum()), '| distinct goals', len({tuple(e['goal']) for e in eps}),
          '| distinct goals among bc_fail', len({tuple(e['goal']) for e in eps if e['bc_fail']}))
    goal_fig(nav, g, pairs, succ, eps)


def goal_fig(nav, g, pairs, succ, eps, path=HERE / 'report' / 'bc_scan_goals.png'):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    H, W = nav.shape; rate = np.full((H, W), np.nan); n_test = np.zeros((H, W), int)
    for (gx, gy) in g['test']:
        m = (pairs[:, 2] == gx) & (pairs[:, 3] == gy); rate[gy, gx] = succ[m].mean()
    for e in eps:
        n_test[e['goal'][1], e['goal'][0]] += 1
    fig, ax = plt.subplots(figsize=(6.4, 5.6)); ax.imshow(np.where(nav[..., None] > 0, 1., 0.) * np.ones(3), interpolation='nearest')
    im = ax.imshow(np.ma.masked_invalid(rate), cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    for (x, y) in g['train']:
        ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, color='#0072B2')); ax.text(x, y, 'T', ha='center', va='center', color='w', fontsize=8)
    for (x, y) in g['test']:
        ax.text(x, y, f'{rate[y, x]:.2f}' + (f'\n×{n_test[y, x]}' if n_test[y, x] else ''), ha='center', va='center', fontsize=6)
    ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=.046, label='BC success over all starts × 3 seeds')
    ax.set_title('BC on held-out goals (colour = success per goal cell; ×k = test pairs with this goal)\n'
                 'blue T = train goal, black = wall', fontsize=8)
    path.parent.mkdir(exist_ok=True); fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches='tight'); plt.close(fig); print('figure', path)




def control():
    """Control set: 50 pairs BC solved on all 3 scan seeds, random (seed 0) -> control_episodes.json (eval only)."""
    z = np.load(HERE / 'bc_scan.npz'); pairs, k = z['pairs'], z['success'].sum(1)
    ok = np.flatnonzero(k == SEEDS); pick = np.sort(np.random.default_rng(0).choice(ok, N_TEST, replace=False))
    eps = [dict(start=pairs[i, :2].tolist(), goal=pairs[i, 2:].tolist(), bc_scan_successes=int(k[i]), bc_fail=False) for i in pick]
    (HERE / 'control_episodes.json').write_text(json.dumps(dict(
        rule=f'{N_TEST} pairs drawn at random (seed 0) from the {len(ok)} (start, held-out goal) pairs BC solved on all {SEEDS} scan seeds',
        bc_step=int(z['step']), episodes=eps), indent=1))
    print('control:', len(ok), 'pairs at 3/3;', N_TEST, 'picked over', len({tuple(e['goal']) for e in eps}), 'goals')


if __name__ == '__main__':
    control() if sys.argv[1:] == ['control'] else main()
