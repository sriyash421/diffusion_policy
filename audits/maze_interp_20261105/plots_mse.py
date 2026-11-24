"""Plots from mse_eval.py (2026-11-23).
  dag_mse_overview_heldout.png  MSE of the policy's mean action to the expert's move (held-out split) vs candidates in context;
                                rows = observation, cols = setting, lines = arm (+ clean-image reference) — the layout of dag_overview_*.png
  dag_mse_overview_train.png    the same on train starts
  dag_end.png                   how attempts end (success / stopped elsewhere / ran out of steps), first attempt, per run
  python plots_mse.py           (CPU)
"""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from plots_simple import SETTINGS, OBS, LINES, REP

OBS_ROWS = {k: v for k, v in OBS.items() if k in ('window', 'ddpm400')}
SHORT_OBS = {'window': 'window', 'ddpm400': 'DDPM'}
SHORT_ARM = {'fbR': 'reward', 'fbB': 'BFS', 'nofb': 'no fb', 'noctx': 'no ctx'}


def load(setting, obs):
    pre, root, *_ = SETTINGS[setting]; out = {}
    for a, *_ in LINES:
        p = root / (f'{pre}_clean_noctx' if a == 'clean' else f'{pre}_{obs}_{a}') / 'mse_eval.npz'
        if p.exists():
            out[a] = np.load(p)
    return out if any(a != 'clean' for a in out) else {}


def overview(split):
    fig, axs = plt.subplots(len(OBS_ROWS), 3, figsize=(12, 3.1 * len(OBS_ROWS) + .6), sharex=True, sharey=True, squeeze=False); top = 0
    for r, (obs, oname) in enumerate(OBS_ROWS.items()):
        for c, (setting, (_, _, sname, test_name)) in enumerate(SETTINGS.items()):
            ax = axs[r, c]; runs = load(setting, obs)
            if not runs:
                ax.text(.5, .5, 'running', ha='center', va='center', transform=ax.transAxes, color='.5')
            for a, label, col, mk, ls in LINES:
                if a in runs:
                    y = np.nanmean(runs[a][f'{split}_mse'], 1); top = max(top, np.nanmax(y))
                    ax.plot(np.arange(len(y)), y, ls=ls, marker=mk, ms=4, color=col, lw=1.8)
            ax.set_xticks(range(8)); ax.grid(alpha=.3)
            if r == 0:
                ax.set_title(sname + (f'\n{test_name}' if split == 'test' else ''), fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{oname}\nMSE to expert action', fontsize=9)
            if r == len(OBS_ROWS) - 1:
                ax.set_xlabel('candidates in context')
    axs[0, 0].set_ylim(0, top * 1.1)
    h = [plt.Line2D([], [], ls=ls, marker=mk, ms=4, color=col, lw=1.8) for _, _, col, mk, ls in LINES]
    fig.legend(h, [a[1] for a in LINES], loc='lower center', ncol=3, fontsize=9, frameon=False)
    fig.suptitle(('Held out' if split == 'test' else 'Train starts') + ': action error to the expert (lower is better) vs candidates in context', fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, 1)); out = REP / f'dag_mse_overview_{"heldout" if split == "test" else "train"}.png'
    fig.savefig(out, dpi=130); plt.close(fig); print('wrote', out)


def ends():
    """Stacked bars: the first attempt ends in success / stopped elsewhere / ran out of steps, per run; train starts and held out."""
    cols = ['#009E73', '#E69F00', '#999999']; names = ['success', 'stopped elsewhere', 'ran out of steps']
    fig, axs = plt.subplots(2, 3, figsize=(14, 5.6), sharey=True)
    for c, setting in enumerate(SETTINGS):
        bars = []                                            # (label, npz)
        for obs, oname in OBS_ROWS.items():
            runs = load(setting, obs)
            bars += [(f'{SHORT_OBS[obs]}\n{SHORT_ARM[a]}', runs[a]) for a, *_ in LINES if a in runs and a != 'clean']
        ref = load(setting, 'window').get('clean') or load(setting, 'ddpm400').get('clean')
        if ref is not None:
            bars.append(('clean\nno ctx', ref))
        for r, split in enumerate(('train', 'test')):
            ax = axs[r, c]
            for x, (_, z) in enumerate(bars):
                e = z[f'{split}_end'][0]; b = 0
                for k in range(3):
                    f = (e == k).mean(); ax.bar(x, f, bottom=b, color=cols[k], width=.8); b += f
            ax.set_xticks(range(len(bars))); ax.set_xticklabels([l for l, _ in bars], fontsize=7)
            ax.set_title(f'{SETTINGS[setting][2]}\n{"train starts" if split == "train" else SETTINGS[setting][3]}', fontsize=9)
    axs[0, 0].set_ylabel('first attempt'); axs[1, 0].set_ylabel('first attempt')
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=k) for k in cols], names, loc='lower center', ncol=3, fontsize=9, frameon=False)
    fig.suptitle('How the first attempt ends', fontsize=11)
    fig.tight_layout(rect=(0, .05, 1, 1)); out = REP / 'dag_end.png'; fig.savefig(out, dpi=130); plt.close(fig); print('wrote', out)


if __name__ == '__main__':
    overview('test'); overview('train'); ends()
