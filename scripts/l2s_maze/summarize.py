"""Results table + plots from data/l2s_maze_eval/<split>/<model>_<arm>/<select>/N<n>.json.

Writes <root>/results.md (success and progress, mean ± SE over episodes) and one figure
per selection rule, <root>/results_<select>.png (rows: metric, cols: split; one line per
model x arm vs log2 N).

    python scripts/l2s_maze/summarize.py
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from maze_layout import REPO, SPLITS  # noqa: E402

# colour = arm (categorical slots 1-3), line style + marker = model
# labels: see arm_label()
ARM_STYLE = {'bon': ('#2a78d6', 'BoN'), 'fuzzy_bon': ('#eb6834', 'FuzzyBoN'),
             'ours': ('#1baf7a', 'Ours')}
MODEL_STYLE = {'st_gaussian': ('-', 'o', 'ST gaussian'),
               'bc_unet': ('--', 's', 'BC UNet'),
               'st_diffusion': (':', '^', 'ST diffusion')}
SELECTS = ('argmax', 'last')
METRICS = ('success', 'progress')


def arm_label(model, arm):
    """ST BoN is STk1 (token 0, no search context); 'fuzzy' marks noised obs."""
    base = 'STk1' if model.startswith('st_') and arm in ('bon', 'fuzzy_bon') \
        else ARM_STYLE[arm][1].replace('FuzzyBoN', 'BoN')
    return base + (' fuzzy' if arm == 'fuzzy_bon' else '')
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def load_results(root):
    """{(split, model, arm, select): {n: record}} plus {split: expert record}."""
    root = pathlib.Path(root)
    res, expert = {}, {}
    for split in SPLITS:
        f = root / split / 'expert' / 'expert.json'
        if f.exists():
            expert[split] = json.loads(f.read_text())
        for f in (root / split).glob('*/*/N*.json'):
            r = json.loads(f.read_text())
            res.setdefault((split, r['model'], r['arm'], r['select']), {})[r['n']] = r
    return res, expert


def make_table(res, expert):
    ns = sorted({n for runs in res.values() for n in runs})
    lines = []
    for metric in METRICS:
        lines += [f'## {metric} (mean ± SE over episodes)', '',
                  '| split | model | arm | select | ' + ' | '.join(f'N={n}' for n in ns) + ' |',
                  '|---|---|---|---|' + '---|' * len(ns)]
        for split in SPLITS:
            for model, (_, _, model_label) in MODEL_STYLE.items():
                for arm in ARM_STYLE:
                    for select in SELECTS:
                        runs = res.get((split, model, arm, select))
                        if not runs:
                            continue
                        cells = [f"{runs[n][metric]:.2f} ± {runs[n][metric + '_se']:.2f}"
                                 if n in runs else '' for n in ns]
                        lines.append(f'| {split} | {model_label} | {arm_label(model, arm)} | {select} | '
                                     + ' | '.join(cells) + ' |')
            if split in expert:
                lines.append(f'| {split} | expert | — | — | {expert[split][metric]:.2f} |'
                             + ' |' * (len(ns) - 1))
        lines.append('')
    return '\n'.join(lines)


def plot(res, select, out_png):
    splits = [s for s in SPLITS if any(k[0] == s and k[3] == select for k in res)]
    if not splits:
        return False
    fig, axes = plt.subplots(len(METRICS), len(splits), figsize=(4.6 * len(splits) + 1, 7.8),
                             squeeze=False, sharex=True, sharey='row')
    for j, split in enumerate(splits):
        for i, metric in enumerate(METRICS):
            ax = axes[i, j]
            for model, (ls, marker, model_label) in MODEL_STYLE.items():
                for arm, (color, _) in ARM_STYLE.items():
                    runs = res.get((split, model, arm, select))
                    if not runs:
                        continue
                    ns = sorted(runs)
                    m = np.array([runs[n][metric] for n in ns])
                    se = np.array([runs[n][metric + '_se'] for n in ns])
                    ax.plot(ns, m, ls=ls, marker=marker, ms=5, lw=2, color=color,
                            label=f'{model_label} {arm_label(model, arm)}')
                    ax.fill_between(ns, m - se, m + se, color=color, alpha=0.12, lw=0)
            ax.set_xscale('log', base=2)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(color=GRID, lw=0.6)
            ax.spines[['top', 'right']].set_visible(False)
            if i == 0:
                ax.set_title(split, color=INK)
            if i == len(METRICS) - 1:
                ax.set_xlabel('N candidates per replan', color=INK_2)
            if j == 0:
                ax.set_ylabel(metric, color=INK_2)
    handles, labels = [], []
    for ax in axes.flat:   # union of series across panels, in first-seen order
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f'executed candidate: {select}', color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(REPO / 'data/l2s_maze_eval'))
    args = ap.parse_args()
    res, expert = load_results(args.root)
    if not res:
        raise SystemExit(f'no results under {args.root}')
    root = pathlib.Path(args.root)
    table = make_table(res, expert)
    (root / 'results.md').write_text(table)
    written = [f'results_{s}.png' for s in SELECTS if plot(res, s, root / f'results_{s}.png')]
    print(table)
    print(f'wrote {root}/results.md, ' + ', '.join(f'{root}/{w}' for w in written))


if __name__ == '__main__':
    main()
