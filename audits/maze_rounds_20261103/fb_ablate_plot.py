"""Plot runs/<tag>/fb_ablate_r<round>.npz: the same STk weights with the context feedback (black) vs with it zeroed (blue).
Top row: success of attempt n (n-1 own attempts in context). Bottom row: best of the first n attempts with the verifier
(-BFS of the final cell; the chosen attempt is executed). BC (UNet, 1 sample) from runs/<tag>/eval_seeds8.npz.
  python fb_ablate_plot.py --tag ast_s20 --round 7
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent


def bon(s, b):
    """(A, ...) -> (A, ...): success of the min-BFS candidate among the first n, first on ties."""
    return np.stack([np.take_along_axis(s[:n], np.argmin(b[:n], 0)[None], 0)[0] for n in range(1, len(s) + 1)])


def main(args):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    z = np.load(HERE / 'runs' / args.tag / f'fb_ablate_r{args.round}.npz'); ev = np.load(HERE / 'runs' / args.tag / 'eval_seeds8.npz')
    T = {'train': 'TRAIN starts: 637 training episodes × 8 seeds', 'test': 'TEST starts: 100 held-out episodes × 8 seeds'}
    lines = (('fb', f'STk{args.round} with the feedback in context (as trained)', '#000000'),
             ('zero', f'STk{args.round}, same weights, context feedback set to 0', '#0072B2'))
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.6), sharey=True); out = {}
    for j, sp in enumerate(('train', 'test')):
        bc = float(ev[f'{sp}_bc_success'].mean())
        for mode, lab, c in lines:
            if f'{mode}_{sp}_success' not in z:
                continue
            s, b = z[f'{mode}_{sp}_success'], z[f'{mode}_{sp}_bfs']; n = np.arange(1, len(s) + 1)
            att, best = s.mean((1, 2)), bon(s, b).mean((1, 2))
            axs[0, j].plot(n, att, 'o-', color=c, lw=2.2, ms=5, label=lab)
            axs[1, j].plot(n, best, 'o-', color=c, lw=2.2, ms=5, label=lab)
            out.setdefault(sp, {})[mode] = {'attempt': att.round(3).tolist(), 'best_of_n': best.round(3).tolist()}
        for i in range(2):
            ax = axs[i, j]; ax.axhline(bc, color='#777777', ls='--', lw=1.6, label='BC (UNet, demos only), 1 sample')
            ax.set_ylim(0, 1.02); ax.grid(alpha=.25); ax.set_xticks(range(1, 9)); ax.set_title(T[sp], fontsize=9)
        axs[0, j].set_xlabel('attempt n (n − 1 own earlier attempts in context)', fontsize=9)
        axs[1, j].set_xlabel('n attempts (best of the first n with the verifier, −BFS of the final cell)', fontsize=9)
    axs[0, 0].set_ylabel('success of attempt n'); axs[1, 0].set_ylabel('success of the executed (best) attempt')
    fig.legend(*axs[0, 1].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle(f'{args.tag}: STk{args.round} weights with vs without the feedback in its context '
                 '(zeroed: the earlier attempts\' cells and moves stay; the current attempt keeps its own per-step −BFS)', fontsize=9.5)
    fig.tight_layout(rect=(0, .05, 1, 1)); p = HERE / 'report' / f'fb_ablate_{args.tag}_r{args.round}.png'; fig.savefig(p, dpi=140)
    print('written', p, json.dumps(out))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='ast_s20'); ap.add_argument('--round', type=int, default=7)
    main(ap.parse_args())
