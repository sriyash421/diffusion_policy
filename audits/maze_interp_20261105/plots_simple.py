"""Simple plots for the DAgger runs with partial / noised observations (2026-11-22): success vs number of candidates (own earlier attempts)
in context. x = candidates in context n = 0..7, y = success of attempt n+1 of the final-round policy; one line per arm.
  dag_overview_train.png    the main question: on TRAIN data, does success rise with more candidates? rows = observation, cols = setting
  dag_overview_heldout.png  same on held-out goals / quadrant / mazes
  dag_<setting>_<obs>.png   one setting x observation, one panel per split
  dag_summary.json          success at n=0 and n=7 (+ s.e. over pairs) per arm / split
  python plots_simple.py            (CPU job)
"""
import json, os
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
VAR = os.environ.get('DAG_VARIANT', '')   # 'ns': the no-stop retrain (tags <pre>ns_*, launch_nostop.sh); figures *_ns.png
if VAR:
    SETTINGS = {k: (v[0] + VAR,) + v[1:] for k, v in SETTINGS.items()}
SUF = f'_{VAR}' if VAR else ''
TNOTE = ' · stop head off' if VAR == 'ns' else ''
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
        ax.set_ylim(0, min(1.02, np.ceil((top + .05) * 10) / 10))   # 1.02: a line at success 1 stays visible
    h = [plt.Line2D([], [], ls=ls, marker=mk, ms=4, color=col, lw=1.8) for _, _, col, mk, ls in LINES]
    two = fig.get_figwidth() < 11
    fig.legend(h, [a[1] for a in LINES], loc='lower center', ncol=2 if two else 3, fontsize=9, frameon=False)
    fig.suptitle(title + TNOTE, fontsize=11)


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
    fig.tight_layout(rect=(0, .11 if len(obs_rows) < 2 else .07, 1, 1)); out = REP / f'dag_overview_{split_kind}{SUF}.png'; fig.savefig(out, dpi=130); plt.close(fig)
    print('wrote', out)


class Trunc(dict):
    """An eval_ctx npz cut to the first `a` attempts (8-round no-context / clean runs drawn next to the capped-context runs)."""
    def __init__(self, z, a):
        super().__init__({k: (z[k][:, :a] if k.endswith('_success') else z[k]) for k in z.files}); self.files = list(self)


# Columns of the <=2-in-context figures: (setting, column title, r8c2 tag part, no-context tag, clean tag)
CAP_COLS = [('interp30c', 'one maze, random split', 'ddpm400r8c2', 'ddpm400_noctx', 'clean_noctx'),
            ('interp30c', 'one maze, random split, 160 train pairs', 'ddpm400d160r8c2', 'ddpm400d160_noctx', 'cleand160_noctx'),
            ('11brc', 'one maze, bottom-right quadrant held out', 'ddpm400r8c2', 'ddpm400_noctx', 'clean_noctx'),
            ('pg', 'new mazes (800 train layouts)', 'ddpm400r8c2', 'ddpm400_noctx', 'clean_noctx')]
CAP_TEST = {'interp30c': 'held-out goals', '11brc': 'held-out quadrant goals', 'pg': 'held-out mazes'}


def cap_runs(col, fname, cut=None):
    """eval npz per arm for one CAP_COLS column; cut = attempts kept (the 3-attempt figure cuts the 8-attempt reference evals)."""
    setting, _, part, nc, cl = col; pre, root, *_ = SETTINGS[setting]
    tags = {a: f'{pre}_{part}_{a}' for a in ('fbR', 'fbB', 'nofb')} | {'noctx': f'{pre}_{nc}', 'clean': f'{pre}_{cl}'}
    runs = {a: np.load(root / t / fname) for a, t in tags.items() if (root / t / fname).exists()}
    if not any(a in runs for a in ('fbR', 'fbB', 'nofb')):
        return {}
    return {a: (Trunc(z, cut) if cut and z['train_success'].shape[1] > cut else z) for a, z in runs.items()}


def cap_figure(fname, kind, out_name, title, cut=None):
    """rows = train starts | held out, cols = CAP_COLS; kind 'perf' = success of attempt n, 'best' = solved within n attempts."""
    cols = [(c, cap_runs(c, fname, cut)) for c in CAP_COLS]
    if not any(r for _, r in cols):
        return
    fig, axs = plt.subplots(2, len(cols), figsize=(4.2 * len(cols), 6.6), sharey=True, squeeze=False)
    for c, (col, runs) in enumerate(cols):
        for r, split in enumerate(('train', 'test')):
            ax = axs[r, c]
            if not runs:
                ax.text(.5, .5, 'running', ha='center', va='center', transform=ax.transAxes, color='.5')
            A = 0
            for a, label, colr, mk, ls in LINES:
                if a in runs and f'{split}_success' in runs[a].files:
                    v = runs[a][f'{split}_success'][-1].astype(float)          # (attempts, pairs, seeds)
                    if kind == 'best':
                        v = np.maximum.accumulate(v, axis=0)
                    y = v.mean((1, 2)); A = max(A, len(y)); ax.plot(np.arange(1, len(y) + 1), y, ls=ls, marker=mk, ms=3, color=colr, lw=1.8)
            A = A or cut or 16; ax.set_xticks([1, 2, 3] if A <= 3 else [1, 4, 8, 12, 16]); ax.set_xlim(.7, A + .3); ax.grid(alpha=.3)
            if r == 0:
                ax.set_title(col[1], fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{"train starts" if split == "train" else "held out"}\n{"success" if kind == "perf" else "solved within n"}', fontsize=9)
            else:
                ax.set_ylabel('' if split == 'train' else CAP_TEST[col[0]], fontsize=9)
            if r == 1:
                ax.set_xlabel('attempt n')
    finish(fig, axs, .95, title)
    fig.tight_layout(rect=(0, .09, 1, 1)); out = REP / out_name; fig.savefig(out, dpi=130); plt.close(fig)
    print('wrote', out)


def capped():
    """8 DAgger rounds, at most 2 earlier attempts in context, DDPM t=400 (tags <pre>_ddpm400r8c2_<arm>; 160 train pairs: ddpm400d160r8c2):
    dag_r8c2.png = 3 attempts (eval_ctx.npz; references cut to 3); dag_r8c2_n16.png = 16 attempts, each seeing the last 2 (eval_ctx_n16.npz);
    dag_r8c2_n16_within.png (appendix) = solved within n of those 16 attempts."""
    T = '8 DAgger rounds, ≤ 2 earlier attempts in context · DDPM t=400'
    cap_figure('eval_ctx.npz', 'perf', 'dag_r8c2.png', T, cut=3)
    cap_figure('eval_ctx_n16.npz', 'perf', 'dag_r8c2_n16.png', T + ' · 16 attempts, each sees the last 2')
    cap_figure('eval_ctx_n16.npz', 'best', 'dag_r8c2_n16_within.png', T + ' · 16 attempts: solved within n')


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
            fig.tight_layout(rect=(0, .17 if len(splits) < 3 else .13, 1, 1)); out = REP / f'dag_{setting}_{obs}{SUF}.png'; fig.savefig(out, dpi=130); plt.close(fig)
            print('wrote', out)
    overview('train'); overview('heldout')
    if not VAR:
        capped()
    (REP / f'dag_summary{SUF}.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
