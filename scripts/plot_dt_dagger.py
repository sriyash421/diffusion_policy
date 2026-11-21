"""Figures for a DT DAgger run (train_pusht_dt_dagger), laid out as the maze in-context DAgger report.

  dagger_curves.png   x = gradient steps within stage n; y = mean max coverage, over the 150 training-start rollouts, of
                      the collection's last attempt (attempt n, n-1 own attempts in context) by the current weights, logged
                      before each iteration's updates. pi_k0 collects nothing. Train starts only: nothing is
                      evaluated on test starts during training.
  dagger_compare.png  x = own rollouts collected from training starts through stage n; y = end-of-stage pi_kn, own context
                      (attempt n+1 after n own attempts; eval_dt_rounds_pusht.py --mode own); train / test panels.
Both carry the U-Net BC full300 (argmax@1 at its test-best step) as a dashed line.

    python scripts/plot_dt_dagger.py <run_dir> --bc-run <unet run_dir> --bc-step 50000
"""
import json
import pathlib

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from plot_dt_rounds import MARKERS, STYLES, bc_coverage

SPLIT_T = {'train': 'TRAIN starts: the 50 training demos', 'test': 'TEST starts: 50 held-out (blq50far)'}


def stage_colors(n):
    return plt.cm.viridis(np.linspace(0, 0.9, n))


def bc_mean(bc_run, bc_step, split, run_dir):
    z = np.load(run_dir / 'rounds_eval' / f'{split}_own.npz')
    return float(bc_coverage(bc_run, bc_step, split, z['episode_idxs'].tolist()).mean())


def curves(run_dir, cfg, bc_train, out):
    D = cfg.dagger
    per_iter = D.epochs * -(-cfg.dagger.repeats * cfg.n_demos // cfg.dataloader.batch_size)
    rows = [json.loads(l) for l in open(run_dir / 'logs.json.txt') if '"collect/seconds"' in l]
    stages = sorted({r['stage'] for r in rows})
    colors = stage_colors(D.n_stages)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for n in stages:
        q = sorted((r for r in rows if r['stage'] == n), key=lambda r: r['iteration'])
        x = [r['iteration'] * per_iter for r in q]
        y = [r[f'collect/attempt{n - 1}_mean_max_coverage'] for r in q]
        ax.plot(x, y, color=colors[n], ls=STYLES[n % len(STYLES)], marker=MARKERS[n % len(MARKERS)], ms=3, lw=1.5,
                label=f'pi_k{n} (context = {n - 1})')
        ax.annotate(f'k{n}', (x[-1], y[-1]), xytext=(4, 0), textcoords='offset points', color=colors[n],
                    fontsize=8, va='center')
    if bc_train is not None:
        ax.axhline(bc_train, color='black', ls='--', lw=2, label=f'BC (UNet, demos only): {bc_train:.2f}')
    ax.set_xlabel(f'gradient steps in stage n (current weights collect before every {per_iter} steps)', fontsize=9)
    ax.set_ylabel('max coverage of attempt n (mean over 150 rollouts)')
    ax.set_title(SPLIT_T['train'] + ' -- collection rollouts', fontsize=9)
    ax.set_ylim(0, 1); ax.grid(alpha=.25); ax.legend(fontsize=7.5, ncol=3, loc='upper left')
    fig.suptitle(f'{run_dir.name[:60]}...: per-iteration collection, stages {stages[0]}..{stages[-1]}', fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print('wrote', out)


def compare(run_dir, cfg, bc, out):
    D = cfg.dagger
    per_stage = D.iters_per_stage * D.repeats * cfg.n_demos        # rollouts per collected attempt index, per stage
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    res = {}
    for ax, split in zip(axs, ('train', 'test')):
        path = run_dir / 'rounds_eval' / f'{split}_own.npz'
        ax.set_title(SPLIT_T[split], fontsize=9)
        if not path.is_file():
            ax.text(.5, .5, 'not evaluated', ha='center', transform=ax.transAxes); continue
        y = np.load(path)['max_coverage'].mean(1)
        n = np.arange(len(y)); x = per_stage * n * (n + 1) // 2
        ax.plot(x, y, 'o-', ms=5, lw=2, color='#0072B2', label=f'pi_kn, own context: last {y[-1]:.2f}')
        for i in n:
            ax.annotate(f'k{i}', (x[i], y[i]), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8)
        ax.axhline(bc[split], color='black', ls='--', lw=2, label=f'BC (UNet, demos only): {bc[split]:.2f}')
        ax.set_xlabel('own rollouts collected from training starts through stage n (summed over stages)', fontsize=8.5)
        ax.set_ylim(0, 1); ax.grid(alpha=.25); ax.legend(fontsize=8)
        res[split] = [round(float(v), 3) for v in y]
    axs[0].set_ylabel('max coverage of pi_kn, attempt n+1 (end of stage n)')
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print('wrote', out, json.dumps(res))


@click.command()
@click.argument('run_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--bc-run', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--bc-step', required=True, type=int)
def main(run_dir, bc_run, bc_step):
    run_dir = pathlib.Path(run_dir)
    cfg = OmegaConf.load(run_dir / '.hydra' / 'config.yaml')
    out = run_dir / 'rounds_eval'; out.mkdir(exist_ok=True)
    bc = {s: bc_mean(bc_run, bc_step, s, run_dir) for s in ('train', 'test')
          if (out / f'{s}_own.npz').is_file()}
    curves(run_dir, cfg, bc.get('train'), out / 'dagger_curves.png')
    if bc:
        compare(run_dir, cfg, bc, out / 'dagger_compare.png')


if __name__ == '__main__':
    main()
