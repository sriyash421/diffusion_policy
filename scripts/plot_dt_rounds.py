"""Per-start max coverage of a context-rounds DT run: one line per round plus the U-Net BC.

Columns train / test starts, rows replay / own context (eval_dt_rounds_pusht.py). The BC line is
the U-Net BC full300's first attempt (argmax@1) at its test-best step, from eval_dt_pusht's
per-episode npz, matched to the rounds eval by episode index.

    python scripts/plot_dt_rounds.py <rounds run_dir> --bc-run <unet run_dir> --bc-step 50000
"""
import pathlib

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

MARKERS = ('o', 's', '^', 'v', 'D', 'P', 'X', '*')
STYLES = ('-', '--', ':', '-.')
BC_DIRS = {'test': 'dt_eval', 'train': 'dt_eval_train'}


def bc_coverage(bc_run, step, split, idxs):
    z = np.load(pathlib.Path(bc_run) / 'bon_search' / BC_DIRS[split] / f'step_{step:07d}'
                / f'{split}_candidates.npz', allow_pickle=True)
    by_ep = dict(zip(z['episode_idxs'].tolist(), z['max_coverage'][:, 0]))
    return np.array([by_ep[i] for i in idxs])


@click.command()
@click.argument('run_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--bc-run', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--bc-step', required=True, type=int)
@click.option('--out', default=None)
def main(run_dir, bc_run, bc_step, out):
    run_dir = pathlib.Path(run_dir)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True, squeeze=False)
    for c, split in enumerate(('train', 'test')):
        for r, mode in enumerate(('replay', 'own')):
            ax = axes[r, c]
            path = run_dir / 'rounds_eval' / f'{split}_{mode}.npz'
            ax.set_title(f'{split} starts, {mode} context')
            if not path.is_file():
                ax.text(0.5, 0.5, 'not evaluated', ha='center', transform=ax.transAxes)
                continue
            z = np.load(path)
            cov, idxs = z['max_coverage'], z['episode_idxs'].tolist()
            x = np.arange(len(idxs))
            colors = plt.cm.viridis(np.linspace(0, 0.9, len(cov)))
            for n, y in enumerate(cov):
                ax.plot(x, y, color=colors[n], marker=MARKERS[n % len(MARKERS)], ms=4,
                        ls=STYLES[n % len(STYLES)], lw=1, label=f'round {n} (mean {y.mean():.2f})')
                ax.annotate(f'r{n}', (x[-1], y[-1]), xytext=(4, 0), textcoords='offset points',
                            color=colors[n], fontsize=8, va='center')
            bc = bc_coverage(bc_run, bc_step, split, idxs)
            ax.plot(x, bc, color='black', ls='--', lw=2, marker='x', ms=5,
                    label=f'U-Net BC (mean {bc.mean():.2f})')
            ax.annotate('BC', (x[-1], bc[-1]), xytext=(4, 0), textcoords='offset points', fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_xlabel('start index')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylabel('max coverage')
            ax.legend(fontsize=7, ncol=3, loc='upper right')
    fig.suptitle('DT context rounds: per-start max coverage')
    fig.tight_layout()
    out = pathlib.Path(out) if out else run_dir / 'rounds_eval' / 'per_start_coverage.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
