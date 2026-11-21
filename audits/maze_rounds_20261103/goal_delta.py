"""30 steps per batch (ast_s30) vs 10k updates per cycle (ast_main), test split, from runs/<tag>/bon_seeds8.npz.
A. per goal cell / episode: STk success at attempt k+1, improve / same / worse.
B. chains (episode, seed) of STk7 (8 attempts in a row): first successful attempt; success at attempt 1 given success at 8.
"""
import numpy as np
import plot_clear as pc
from rounds import HERE
st = pc.starts(); g = np.round(st['test_goal']).astype(int); cells = [tuple(c) for c in np.unique(g, axis=0)]
Z = {t: np.load(HERE / 'runs' / t / 'bon_seeds8.npz') for t in ('ast_s30', 'ast_main')}
for k in (0, 2, 7):
    a, b = (Z[t]['test_st_success'][k, k].mean(-1) for t in ('ast_s30', 'ast_main'))     # (episodes,)
    ca = np.array([a[(g == c).all(1)].mean() for c in cells]); cb = np.array([b[(g == c).all(1)].mean() for c in cells])
    for name, x, y, tol in (('goal cells', ca, cb, 0), ('goal cells |d|<=.05 same', ca, cb, .05), ('episodes', a, b, 0), ('episodes |d|<=1/8 same', a, b, .125)):
        d = y - x; print(f'STk{k} {name:26s} n={len(d):3d} improve {(d > tol).sum():3d} same {(abs(d) <= tol).sum():3d} worse {(d < -tol).sum():3d}'
                        + (f'   both 0: {((x == 0) & (y == 0)).sum()}, both 1: {((x == 1) & (y == 1)).sum()}' if tol == 0 else ''))
    if k == 7:
        for c, x, y in sorted(zip(cells, ca, cb), key=lambda r: r[2] - r[1]):
            print('   ', c, f'{x:.2f} -> {y:.2f}')
for t in Z:
    s = Z[t]['test_st_success'][7]                                     # (A, episodes, seeds)
    ok8 = s[7]; first = np.where(s.any(0), s.argmax(0) + 1, 0)
    print(t, 'STk7 chains:', s.shape[1] * s.shape[2], '| attempt 8 succeeds:', ok8.sum(), '| of those attempt 1 also succeeds:', (s[0] & ok8).sum(),
          f'({(s[0] & ok8).sum() / max(ok8.sum(), 1):.0%})', '| first success at attempt n (n=1..8):', np.bincount(first[ok8], minlength=9)[1:].tolist())
    print('   attempt-1 success vs attempt-8 success (all chains):', s[0].mean().round(3), s[7].mean().round(3),
          '| chains succeeding at any attempt:', s.any(0).mean().round(3), '| a8 ok & a1 fail:', (ok8 & ~s[0]).sum(), ' a1 ok & a8 fail:', (s[0] & ~ok8).sum())
    b = Z[t]['test_st_bfs'][7].mean((1, 2)); print('   STk7 mean final BFS by attempt:', b.round(2).tolist())
