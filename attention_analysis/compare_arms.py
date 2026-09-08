"""Compare context use ACROSS arms, from the per-run attention.json files.

    python attention_analysis/compare_arms.py [--dir /gscratch/.../attention_analysis]

Each run's visualize_attention.py writes `context_mass_per_slot`: the share of the decoder's
cross-attention that lands on the search-context block, per candidate slot. This collects
them so the arms can be read against each other, which is the actual question -- does an
obs-noise ladder, or a goal mask, change HOW the policy uses its context, not just how well
it scores.

TWO THINGS THAT ARE NOT COMPARABLE, and are separated rather than averaged away:
  * ST k=1 has max_context_actions == 0, so it has no context block at all. It appears in
    the table with '--', never as a zero, because zero would read as "has context, ignores
    it" when the truth is "has none".
  * Arms differ in how far they have trained. The step is printed on every row; a 40k row is
    not a 100k row and the two are never placed on one line.

THE PLOTTED STEP MUST BE MATCHED. Context share DECLINES over training (baseline blq at
denoise 7, slot 15: 0.50 at 10k, 0.40 at 50k, 0.395 at 100k), so an arm drawn at its own
latest 40k against another at 100k shows a gap that is partly training time and not the
arm. The default is therefore the largest step present in EVERY arm with a context block;
--step pins it explicitly, and --latest restores the per-arm-latest view with the step on
each label. Arms lacking the chosen step are listed as omitted, never silently dropped.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from analysis_run_name import canonical

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED = '#1a1a19', '#8a8880'
# colour + linestyle + marker together: the families must stay distinguishable in greyscale
FAMILIES = [
    # `son-none` is the explicit no-schedule token in the dataset-first folder name; the
    # second clause keeps the older policy-first names classifying correctly too.
    ('baseline (no noise)', lambda r: '_son-none' in r or ('_son-' not in r and '_gm-' not in r),
     '#2a78d6', (0, (1, 0)), 'o'),
    ('flat ladder',         lambda r: '_son-flat' in r, '#eb6834', (0, (5, 2)), 's'),
    ('ramp ladder',         lambda r: '_son-ramp' in r, '#1baf7a', (0, (1.5, 1.5)), '^'),
    ('goal mask',           lambda r: '_gm-' in r,      '#8e5fd0', (0, (6, 2, 1, 2)), 'D'),
]


def family(run):
    for name, pred, c, ls, m in FAMILIES:
        if pred(run):
            return name, c, ls, m
    return 'other', MUTED, (0, (1, 1)), 'x'


def collect(root):
    """Rows from every attention.json, one run per directory, labelled by FOLDER name.

    A run may appear under more than one directory when it was dumped before the current
    naming convention (`k16_blq/` holds the same run as the canonical dataset-first folder).
    Folders are keyed by the run's canonical name (scripts/analysis_run_name.py) and the
    directory actually named that wins, so the older copy neither double-prints in the table
    nor silently wins in the plot.

    Rows carry the FOLDER name rather than the json's `run` field: the folder is filed
    dataset-first, which is the order these are read in as a group, while `run` remains the
    training run's own name.
    """
    seen, out = {}, []
    for j in sorted(pathlib.Path(root).glob('*/attention.json')):
        run = json.loads(j.read_text())['run']
        key = canonical(run) or run
        prev = seen.get(key)
        if prev is None or (j.parent.name == key and prev.parent.name != key):
            seen[key] = j
    for key, j in sorted(seen.items()):
        d = json.loads(j.read_text())
        if j.parent.name != key:
            print(f'note: {key}\n      read from legacy dir {j.parent.name}/')
        for step, rec in d['steps'].items():
            for di, r in rec['denoise'].items():
                out.append({
                    'run': j.parent.name, 'step': int(step), 'denoise': int(di),
                    'K': rec.get('K'), 'mca': rec.get('max_context_actions', 0),
                    'per_slot': r.get('context_mass_per_slot'),
                    'mean': r.get('context_mass_mean'),
                })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/gscratch/robotics/harine/attention_analysis')
    ap.add_argument('--denoise', type=int, default=None,
                    help='which denoising step to plot; default = the largest present')
    ap.add_argument('--step', type=int, default=None,
                    help='checkpoint to compare at; default = largest step present in every arm')
    ap.add_argument('--latest', action='store_true',
                    help='plot each arm at its OWN latest step (confounds arm with training time)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    root = pathlib.Path(args.dir)
    rows = collect(root)
    if not rows:
        print(f'no attention.json under {root}'); return

    dsel = args.denoise if args.denoise is not None else max(r['denoise'] for r in rows)
    print(f'context share of cross-attention, denoising step {dsel}\n')
    print(f'{"arm":<52} {"step":>6} {"slot1":>6} {"slot8":>6} {"slotK-1":>8} {"mean":>6}')
    plot = []
    for r in sorted(rows, key=lambda r: (r['run'], r['step'])):
        if r['denoise'] != dsel:
            continue
        name = (r['run'].replace('_value_k16_ver-t_goal', '_k16')
                        .replace('_value_k1_ver-t_goal', '_k1')
                        .replace('value_k16_ver-t_goal', 'k16')
                        .replace('value_k1_ver-t_goal', 'k1')
                        .replace('_enc-resnet18', '').replace('_seed-42', ''))
        ps = r['per_slot']
        if not ps:
            # ST k=1: no context block. '--' not 0.0 -- see the module docstring.
            print(f'{name:<52} {r["step"]//1000:5d}k {"--":>6} {"--":>6} {"--":>8} {"--":>6}'
                  f'   (no context tokens)')
            continue
        print(f'{name:<52} {r["step"]//1000:5d}k {ps[1]:6.2f} {ps[8]:6.2f} {ps[-1]:8.2f} '
              f'{np.mean(ps[1:]):6.2f}')
        plot.append((name, r['step'], np.asarray(ps)))

    if not plot:
        print('\nno arm has a context block yet'); return

    per_run = {}
    for name, step, ps in plot:
        per_run.setdefault(name, {})[step] = ps

    if args.latest:
        chosen = {n: max(s) for n, s in per_run.items()}
        label, tag = 'each arm at its own latest checkpoint', 'latest'
    else:
        if args.step is not None:
            step = args.step
        else:
            common = set.intersection(*(set(s) for s in per_run.values()))
            if not common:
                print('\nno step is present in every arm; pass --step or --latest')
                for n, s in sorted(per_run.items()):
                    print(f'  {n:<52} steps {sorted(k//1000 for k in s)}')
                return
            step = max(common)
        chosen = {n: step for n, s in per_run.items() if step in s}
        dropped = sorted(set(per_run) - set(chosen))
        label, tag = f'all arms at {step//1000}k', f'{step//1000}k'
        if dropped:
            print(f'\nomitted, no {step//1000}k checkpoint ({len(dropped)}):')
            for n in dropped:
                print(f'  {n:<52} has {sorted(k//1000 for k in per_run[n])}')

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for name in sorted(chosen):
        step, ps = chosen[name], per_run[name][chosen[name]]
        fam, c, ls, m = family(name)
        ax.plot(np.arange(len(ps)), ps, color=c, ls=ls, marker=m, ms=3.5, lw=1.5,
                label=f'{name}  ({step//1000}k)' if args.latest else name)
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(0.2, 0.51, 'parity with observation', fontsize=8, color=MUTED)
    ax.set_xlabel('candidate slot  (0 has no context by construction)', color=INK)
    ax.set_ylabel('context share of cross-attention', color=INK)
    ax.set_ylim(0, 1)
    ax.set_title(f'Context use by arm — {label}, denoising step {dsel}',
                 color=INK, fontsize=12, loc='left')
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, ncol=2,
              loc='upper left', bbox_to_anchor=(0, -0.12))
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    out = pathlib.Path(args.out or (root / f'compare_arms_d{dsel}_{tag}.png'))
    fig.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
