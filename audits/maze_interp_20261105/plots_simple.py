"""Simple plots for the DAgger runs with partial / noised observations (2026-11-22): success vs number of candidates (own earlier attempts)
in context. x = candidates in context n = 0..7, y = success of attempt n+1 of the final-round policy; one line per arm.
  dag_overview_train.png    the main question: on TRAIN data, does success rise with more candidates? rows = observation, cols = setting
  dag_overview_heldout.png  same on held-out goals / quadrant / mazes
  dag_<setting>_<obs>.png   one setting x observation, one panel per split
  dag_summary.json          success at n=0 and n=7 (+ s.e. over pairs) per arm / split
  python plots_simple.py            (CPU job)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

I = Path(__file__).resolve().parent; A = I.parent; REP = I / 'report'
SETTINGS = {'interp30c': ('c_dag', I / 'runs', 'one maze, random split', 'held-out goals (41)'),
            '11brc': ('br_dag', A / 'maze_rounds_20261103' / 'runs', 'one maze, bottom-right quadrant held out', 'held-out quadrant goals'),
            'pg': ('pg_dag', A / 'maze_procgen_ast_20261121' / 'runs', 'new mazes (800 train layouts)', 'held-out mazes (200)')}
OBS = {'window': 'partial view (7×7 window)', 'ddpm200': 'noised image, DDPM t=200',
       'ddpm400': 'noised image, DDPM t=400', 'ddpm600': 'noised image, DDPM t=600'}
ARMS = [('fbR', 'context + reward feedback', '#0072B2', 'o', '-'),
        ('fbB', 'context + BFS + goal offset (leak control)', '#D55E00', 's', '-'),
        ('nofb', 'context, no feedback', '#009E73', '^', '-'),
        ('noctx', 'trained without context', '#000000', None, '--')]
REF = ('clean', 'trained without context, clean image (goal visible)', '#999999', None, ':')   # run <pre>_clean_noctx (--obs none), same in every row
LINES = ARMS + [REF]
SPLIT_NAME = {'train': 'train starts', 'train_heldout': 'unseen pairs of train goals'}


def curve(z, split):
    s = z[f'{split}_success'][-1]                        # final round: (attempts, pairs, seeds)
    per_pair = s.mean(2)                                  # (attempts, pairs)
    return per_pair.mean(1), per_pair.std(1) / np.sqrt(per_pair.shape[1])


def load(setting, obs):
    pre, root, *_ = SETTINGS[setting]
    runs = {a: root / f'{pre}_{obs}_{a}' / 'eval_ctx.npz' for a, *_ in ARMS}
    runs = {a: np.load(p) for a, p in runs.items() if p.exists()}
    ref = root / f'{pre}_clean_noctx' / 'eval_ctx.npz'
    if runs and ref.exists():
        runs['clean'] = np.load(ref)
    return runs


def draw(ax, runs, split):
    top = 0
    for a, label, col, mk, ls in LINES:
        if a in runs and f'{split}_success' in runs[a].files:
            y, _ = curve(runs[a], split); x = np.arange(len(y)); top = max(top, y.max())
            ax.plot(x, y, ls=ls, marker=mk, ms=4, color=col, lw=1.8, label=label)
    ax.set_xticks(range(8)); ax.grid(alpha=.3)
    return top


def finish(fig, axs, top, title):
    for ax in np.ravel(axs):
        ax.set_ylim(0, min(1, np.ceil((top + .05) * 10) / 10))
    h = [plt.Line2D([], [], ls=ls, marker=mk, ms=4, color=col, lw=1.8) for _, _, col, mk, ls in LINES]
    two = fig.get_figwidth() < 11
    fig.legend(h, [a[1] for a in LINES], loc='lower center', ncol=2 if two else 3, fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=11)


def overview(split_kind):
    obs_rows = [o for o in OBS if any(load(s, o) for s in SETTINGS)]
    if not obs_rows:
        return
    fig, axs = plt.subplots(len(obs_rows), 3, figsize=(12, 3.1 * len(obs_rows) + .6), sharex=True, squeeze=False)
    top = 0
    for r, obs in enumerate(obs_rows):
        for c, (setting, (_, _, sname, test_name)) in enumerate(SETTINGS.items()):
            ax = axs[r, c]; runs = load(setting, obs)
            if not runs:
                ax.text(.5, .5, 'running', ha='center', va='center', transform=ax.transAxes, color='.5')
            top = max(top, draw(ax, runs, 'train' if split_kind == 'train' else 'test'))
            if r == 0:
                ax.set_title(sname + ('' if split_kind == 'train' else f'\n{test_name}'), fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{OBS[obs]}\nsuccess', fontsize=9)
            if r == len(obs_rows) - 1:
                ax.set_xlabel('candidates in context')
    finish(fig, axs, top, 'Train starts: does success rise with more candidates in context?' if split_kind == 'train'
           else 'Held out: does success rise with more candidates in context?')
    fig.tight_layout(rect=(0, .11 if len(obs_rows) < 2 else .07, 1, 1)); out = REP / f'dag_overview_{split_kind}.png'; fig.savefig(out, dpi=130); plt.close(fig)
    print('wrote', out)


SCHED = [('ddpm400', '8 rounds · 7.2k updates'), ('ddpm400r3', '3 rounds · 7.2k updates'), ('ddpm400r3x3', '3 rounds · 21.6k updates (3× longer)')]


def schedules():
    """DDPM t=400, one maze: the 8-round runs vs 3 rounds (same total updates) vs 3 rounds trained 3x longer -> dag_sched_<setting>.png.
    rows = train starts | held-out goals, cols = schedule, lines = arm (3-round runs: 0..2 candidates in context)."""
    for setting in ('interp30c', '11brc'):
        _, _, sname, test_name = SETTINGS[setting]; cols = [(k, n, {a: z for a, z in load(setting, k).items() if k == 'ddpm400' or a != 'clean'}) for k, n in SCHED]
        if not any(r for k, _, r in cols if k != 'ddpm400'):
            continue
        fig, axs = plt.subplots(2, 3, figsize=(12, 6.8), sharey=True, sharex=True, squeeze=False); top = 0
        for c, (k, name, runs) in enumerate(cols):
            for r, split in enumerate(('train', 'test')):
                ax = axs[r, c]
                if not runs:
                    ax.text(.5, .5, 'running', ha='center', va='center', transform=ax.transAxes, color='.5')
                top = max(top, draw(ax, runs, split))
                if r == 0:
                    ax.set_title(name, fontsize=10)
                if c == 0:
                    ax.set_ylabel(f'{"train starts" if split == "train" else test_name}\nsuccess', fontsize=9)
                if r == 1:
                    ax.set_xlabel('candidates in context')
        finish(fig, axs, top, f'{sname}, DDPM t=400: fewer rounds, same updates · and 3× longer')
        fig.tight_layout(rect=(0, .09, 1, 1)); out = REP / f'dag_sched_{setting}.png'; fig.savefig(out, dpi=130); plt.close(fig)
        print('wrote', out)


def main():
    summary = {}
    for setting, (pre, root, sname, test_name) in SETTINGS.items():
        for obs, obs_name in OBS.items():
            runs = load(setting, obs)
            if not runs:
                continue
            splits = [s for s in ('train', 'train_heldout', 'test') if any(f'{s}_success' in z.files for z in runs.values())]
            fig, axs = plt.subplots(1, len(splits), figsize=(4.3 * len(splits), 3.9), sharey=True, squeeze=False)
            top = 0
            for ax, split in zip(axs[0], splits):
                top = max(top, draw(ax, runs, split))
                ax.set_title(test_name if split == 'test' else SPLIT_NAME[split], fontsize=10); ax.set_xlabel('candidates in context')
                for a, z in runs.items():
                    if f'{split}_success' in z.files:
                        y, se = curve(z, split)
                        summary.setdefault(f'{setting}/{obs}', {}).setdefault(a, {})[split] = dict(
                            n0=round(float(y[0]), 3), n7=round(float(y[-1]), 3), se=round(float(se.mean()), 3), pairs=int(z[f'{split}_success'].shape[2]))
            axs[0, 0].set_ylabel('success')
            finish(fig, axs, top, f'{sname}: {obs_name}')
            fig.tight_layout(rect=(0, .17 if len(splits) < 3 else .13, 1, 1)); out = REP / f'dag_{setting}_{obs}.png'; fig.savefig(out, dpi=130); plt.close(fig)
            print('wrote', out)
    overview('train'); overview('heldout'); schedules()
    (REP / 'dag_summary.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
