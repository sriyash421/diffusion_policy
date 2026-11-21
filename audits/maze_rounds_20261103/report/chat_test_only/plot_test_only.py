"""Test-split-only versions of the best-of-n plots (30-step run) and the feedback comparison, for the chat."""
import sys, json; from pathlib import Path
import numpy as np
HERE = Path('/gscratch/robotics/harine/codex_audits/maze_rounds_20261103'); sys.path.insert(0, str(HERE)); import plot_clear as p
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
O = HERE / 'report' / 'chat_test_only'; z = np.load(HERE / 'runs/ast_s30/bon_seeds8.npz')
st = p.bon(z['test_st_success'], z['test_st_bfs'], 1).mean((-1, -2)); bc = p.bon(z['test_bc_success'], z['test_bc_bfs'], 2).mean((-1, -2))
A, R = st.shape; n = np.arange(1, A + 1); cols = plt.cm.viridis(np.linspace(0, .92, R)); T = '100 held-out test episodes × 8 seeds'
f, ax = plt.subplots(figsize=(8.5, 5.6))
for m in range(R):
    ax.plot(n, st[:, m], 'o-', ms=4, lw=1.8, color=cols[m], label=f'STk{m}')
ax.plot(n, bc, 's--', ms=4.5, lw=2, color='#555555', label='BC (UNet), n iid samples')
ax.set(xlabel='n candidates (verifier = −BFS of the final cell picks one; it is executed)', ylabel='test success', ylim=(0, 1), xticks=n)
ax.grid(alpha=.25); ax.legend(fontsize=8.5, ncol=3, loc='upper left'); ax.set_title(f'30 gradient steps per batch: best of n, test\n{T}', fontsize=10)
f.tight_layout(); f.savefig(O / 'bon_curves_test.png', dpi=140); plt.close(f)
diag = np.array([st[min(m, A - 1), m] for m in range(R)]); x = np.arange(1, R + 1)
f, ax = plt.subplots(figsize=(8.5, 5.0))
ax.plot(x, diag, 'o-', lw=2.2, ms=6, color='k', label='STk at n = k+1 (k attempts in context, as trained)')
for m in range(R):
    ax.annotate(f'STk{m}', (x[m], diag[m]), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8)
ax.plot(n, bc, 's--', lw=2, ms=5, color='#777777', label='BC (UNet), best of n iid samples')
ax.set(xlabel='n candidates (= k+1 for STk)', ylabel='test success', ylim=(0, 1), xticks=n); ax.grid(alpha=.25); ax.legend(fontsize=9, loc='upper left')
ax.set_title(f'30 gradient steps per batch: STk at its trained context length, test\n{T}', fontsize=10)
f.tight_layout(); f.savefig(O / 'bon_line_test.png', dpi=140); plt.close(f)
f, ax = plt.subplots(figsize=(8.5, 5.2))
for tag, lab, c in (('ast_s20', 'with feedback', '#000000'), ('ast_s20_nofb', 'without feedback', '#0072B2')):
    by = {}
    for r in p.rows(tag):
        if 'test_by_attempt' in r: by.setdefault(r['round'], []).append(r)
    ks = sorted(by); yk = [np.mean([r['test_by_attempt'][-1] for r in by[k][-5:]]) for k in ks]; y1 = [np.mean([r['test_by_attempt'][0] for r in by[k][-5:]]) for k in ks]
    ax.plot(ks, yk, 'o-', color=c, lw=2, ms=5, label=f'{lab}: attempt k+1 (k own attempts in context) = STk')
    ax.plot(ks, y1, 'o:', color=c, lw=1.6, ms=4, alpha=.8, label=f'{lab}: attempt 1 of the same weights (no context)')
ax.axhline(p.bc_ref('spi20', 'test'), color='#777777', ls='--', lw=1.6, label='BC (UNet, demos only)')
ax.set(xlabel='round k (mean of the last 5 evals of the round)', ylabel='test success', ylim=(0, 1)); ax.grid(alpha=.25); ax.legend(fontsize=8, loc='upper left')
ax.set_title('Context with vs without feedback (20 gradient steps per batch), test\n100 held-out test episodes × 8 seeds', fontsize=10)
f.tight_layout(); f.savefig(O / 'feedback_compare_test.png', dpi=140); plt.close(f); print('written', list(O.glob('*.png')))
