"""STk (20 steps per batch, ast_s20): is the gain from context or from training longer?
  black solid  : STk, attempt k+1 = k own earlier attempts in context (how STk is scored)
  black dotted : the same STk weights, attempt 1 = no context
  orange       : a policy that never sees context, trained on demos only for the same total steps as STk ((k+1) x 600)
STk lines from runs/ast_s20/eval_seeds8.npz (637 train / 100 test episodes x 8 seeds); orange from runs/ast_k0long/log.jsonl
(mean of the 5 evals ending at (k+1) x 600 steps; train 1 seed, test 8 seeds)."""
import json, sys
import numpy as np
D = '/gscratch/robotics/harine/codex_audits/maze_rounds_20261103'
sys.path.insert(0, D); import plot_clear as p
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

z = np.load(f'{D}/runs/ast_s20/eval_seeds8.npz')
rows = [json.loads(l) for l in open(f'{D}/runs/ast_k0long/log.jsonl')]; rows = [r for r in rows if 'train_success' in r]
upd = np.array([r['update'] for r in rows]); R = 8; steps = 600 * np.arange(1, R + 1); out = {}
fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
for ax, sp in zip(axs, ('train', 'test')):
    s = z[f'{sp}_success'].mean((-1, -2))                    # (R, A): round-k weights, attempt a
    ctx = np.array([s[k, k] for k in range(R)]); noctx = s[:, 0]
    y = np.array([r[f'{sp}_success'] for r in rows])
    long = np.array([y[max(0, i - 4):i + 1].mean() for i in [int(np.where(upd == t)[0][0]) for t in steps]])
    bc = z[f'{sp}_bc_success'].mean()
    ax.plot(range(R), ctx, 'o-', color='#000000', lw=2.2, ms=6, label='STk, attempt k+1: k own earlier attempts in context')
    ax.plot(range(R), noctx, 'o:', color='#000000', lw=1.8, ms=5, label='same STk weights, attempt 1: no context')
    ax.plot(range(R), long, 's-', color='#D55E00', lw=2, ms=5, label='never any context, demos only, same total steps as STk')
    ax.axhline(bc, color='#777777', ls='--', lw=1.6, label='BC (UNet, demos only)')
    ax.set_xticks(range(R)); ax.set_xticklabels([f'STk{k}\n{t:,} steps' for k, t in zip(range(R), steps)], fontsize=8)
    ax.set_title(p.SPLIT_T[sp], fontsize=9); ax.set_ylim(0, 1.02); ax.grid(alpha=.25)
    ax.set_xlabel('round k  (total gradient steps so far: each round = 600 steps, 20 per batch)', fontsize=9)
    out[sp] = dict(ctx=ctx.round(3).tolist(), noctx=noctx.round(3).tolist(), long=long.round(3).tolist(), bc=round(float(bc), 3))
axs[0].set_ylabel('success')
fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=8.5, frameon=False)
fig.suptitle('Is STk better because of the context or because it was trained longer?  (20 gradient steps per batch)', fontsize=11)
fig.tight_layout(rect=(0, .1, 1, 1)); fig.savefig(f'{D}/report/ctx_vs_steps_ast_s20.png', dpi=140); print(json.dumps(out))
