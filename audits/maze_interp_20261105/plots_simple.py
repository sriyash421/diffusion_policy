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


def capped():
    """8 DAgger rounds with at most 2 earlier attempts in context (tags <pre>_ddpm400r8c2_<arm>), DDPM t=400, one maze -> dag_r8c2.png.
    rows = train starts | held-out goals, cols = setting; no-context / clean lines = the 8-round runs cut to 3 attempts."""
    cols = [s for s in ('interp30c', '11brc') if load(s, 'ddpm400r8c2')]
    if not cols:
        return
    fig, axs = plt.subplots(2, len(cols), figsize=(4.6 * len(cols), 6.6), sharex=True, squeeze=False); top = 0
    for c, setting in enumerate(cols):
        _, _, sname, test_name = SETTINGS[setting]; base = load(setting, 'ddpm400')
        runs = {a: z for a, z in load(setting, 'ddpm400r8c2').items() if a not in ('noctx', 'clean')}
        runs.update({a: Trunc(base[a], 3) for a in ('noctx', 'clean') if a in base})
        for r, split in enumerate(('train', 'test')):
            ax = axs[r, c]; top = max(top, draw(ax, runs, split)); ax.set_xticks(range(3)); ax.set_xlim(-.25, 2.25)
            if r == 0:
                ax.set_title(sname, fontsize=10)
            if c == 0 or r == 1:
                ax.set_ylabel(f'{"train starts" if split == "train" else test_name}\nsuccess', fontsize=9)
            if r == 1:
                ax.set_xlabel('candidates in context')
    finish(fig, axs, top, '8 DAgger rounds, ≤ 2 earlier attempts in context · DDPM t=400')
    fig.tight_layout(rect=(0, .1, 1, 1)); out = REP / 'dag_r8c2.png'; fig.savefig(out, dpi=130); plt.close(fig)
    print('wrote', out)


def capped16():
    """The <=2-in-context runs (r8c2) making 16 attempts in a row, each seeing the last 2 (sliding window); no-context and clean runs make 16
    attempts from an empty stream (eval_n16.sbatch -> eval_ctx_n16.npz). -> dag_r8c2_n16.png: success of attempt n (left two columns) and
    solved within n attempts (right two columns); rows = train starts | held-out goals."""
    cols = [s for s in ('interp30c', '11brc')]
    def get(setting):
        pre, root, *_ = SETTINGS[setting]
        tags = {a: f'{pre}_ddpm400r8c2_{a}' for a in ('fbR', 'fbB', 'nofb')} | {'noctx': f'{pre}_ddpm400_noctx', 'clean': f'{pre}_clean_noctx'}
        return {a: np.load(root / t / 'eval_ctx_n16.npz') for a, t in tags.items() if (root / t / 'eval_ctx_n16.npz').exists()}
    data = {s: get(s) for s in cols}
    if not any(data.values()):
        return
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.8), sharex=True, sharey=True, squeeze=False)
    for k, (kind, kname) in enumerate((('perf', 'success of attempt n'), ('best', 'solved within n attempts'))):
        for c, setting in enumerate(cols):
            _, _, sname, test_name = SETTINGS[setting]; runs = data[setting]
            for r, split in enumerate(('train', 'test')):
                ax = axs[r, 2 * k + c]
                if not runs:
                    ax.text(.5, .5, 'running', ha='center', va='center', transform=ax.transAxes, color='.5')
                for a, label, col, mk, ls in LINES:
                    if a in runs and f'{split}_success' in runs[a].files:
                        s = runs[a][f'{split}_success'][-1].astype(float)          # (attempts, pairs, seeds)
                        if kind == 'best':
                            s = np.maximum.accumulate(s, axis=0)
                        y = s.mean((1, 2)); ax.plot(np.arange(1, len(y) + 1), y, ls=ls, marker=mk, ms=3, color=col, lw=1.8)
                ax.set_xticks([1, 4, 8, 12, 16]); ax.grid(alpha=.3)
                if r == 0:
                    ax.set_title(f'{kname}\n{sname}', fontsize=10)
                if 2 * k + c == 0:
                    ax.set_ylabel(f'{"train starts" if split == "train" else "held-out goals"}\nsuccess', fontsize=9)
                if r == 1:
                    ax.set_xlabel('attempt n')
    finish(fig, axs, 0.95, '8 DAgger rounds, ≤ 2 earlier attempts in context · 16 attempts, each sees the last 2 · DDPM t=400')
    fig.tight_layout(rect=(0, .08, 1, 1)); out = REP / 'dag_r8c2_n16.png'; fig.savefig(out, dpi=130); plt.close(fig)
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
            fig.tight_layout(rect=(0, .17 if len(splits) < 3 else .13, 1, 1)); out = REP / f'dag_{setting}_{obs}{SUF}.png'; fig.savefig(out, dpi=130); plt.close(fig)
            print('wrote', out)
    overview('train'); overview('heldout')
    if not VAR:
        capped(); capped16()
    (REP / f'dag_summary{SUF}.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
