"""Compare candidate dispersion ACROSS arms, from the per-run modes.json files.

    python mode_analysis/compare_arms.py [--step 100000]

Each run's candidate_modes.py writes `endpoint_dispersion_px`: the mean pairwise distance
between the R chunk endpoints, per candidate slot. This collects them so the arms can be read
against each other -- does an obs-noise ladder, or a goal mask, actually widen the candidate
distribution as the slot index rises, or is every arm equally collapsed.

THE STEP MUST BE MATCHED. Dispersion falls by roughly an order of magnitude over training
(baseline blq slot 0: 35.9px at 10k, 10.9px at 50k, 5.6px at 100k), so an arm plotted at 50k
against one at 100k would show a difference that is entirely training time. The default is
therefore the largest step present in EVERY arm, not each arm's own latest; --step pins it
explicitly and --latest restores the per-arm-latest view with the step on each label.

DISPERSION IS A SPREAD STATISTIC, NOT A MODE COUNT. It rises both when a unimodal cloud
widens and when it splits in two. The per-state scatter panels
(modes_step<N>_state<i>.png) are what distinguish those; this plot is the one number that is
comparable across arms without fitting anything.
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED = '#1a1a19', '#8a8880'
# colour + linestyle + marker together: the families must stay distinguishable in greyscale.
# Mirrors attention_analysis/compare_arms.py so an arm keeps one identity across both plots.
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


def short(run):
    return (run.replace('_value_k16_ver-t_goal', '_k16')
               .replace('_value_k1_ver-t_goal', '_k1')
               .replace('value_k16_ver-t_goal', 'k16')
               .replace('value_k1_ver-t_goal', 'k1')
               .replace('_enc-resnet18', '').replace('_seed-42', ''))


def collect(root):
    """{folder: {step: [dispersion per slot]}}

    Keyed by the FOLDER name, not the `run` field inside the json. The folder is filed
    dataset-first (scripts/analysis_run_name.py) while `run` stays the training run's own
    name, so keying on the folder is what puts the two experiment axes at the front of every
    label and sorts the arms into dataset blocks.
    """
    out = {}
    for j in sorted(pathlib.Path(root).glob('*/modes.json')):
        d = json.loads(j.read_text())
        per_step = {}
        for step, rec in d.get('steps', {}).items():
            v = rec.get('endpoint_dispersion_px')
            if v:
                per_step[int(step)] = np.asarray(v, dtype=float)
        if per_step:
            out[j.parent.name] = per_step
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/gscratch/robotics/harine/mode_analysis')
    ap.add_argument('--step', type=int, default=None,
                    help='checkpoint to compare at; default = largest step present in every arm')
    ap.add_argument('--latest', action='store_true',
                    help='plot each arm at its OWN latest step (confounds arm with training time)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    root = pathlib.Path(args.dir)
    runs = collect(root)
    if not runs:
        print(f'no modes.json under {root}')
        return

    if args.latest:
        chosen = {r: max(s) for r, s in runs.items()}
        label, tag = 'each arm at its own latest checkpoint', 'latest'
    else:
        if args.step is not None:
            step = args.step
        else:
            common = set.intersection(*(set(s) for s in runs.values()))
            if not common:
                print('no step is present in every arm; pass --step or --latest')
                for r, s in sorted(runs.items()):
                    print(f'  {short(r):<62} steps {sorted(k//1000 for k in s)}')
                return
            step = max(common)
        chosen = {r: step for r, s in runs.items() if step in s}
        dropped = sorted(set(runs) - set(chosen))
        label, tag = f'all arms at {step//1000}k', f'{step//1000}k'
        if dropped:
            print(f'omitted, no {step//1000}k checkpoint ({len(dropped)}):')
            for r in dropped:
                print(f'  {short(r):<62} has {sorted(k//1000 for k in runs[r])}')
            print()

    print(f'endpoint dispersion (px), {label}\n')
    print(f'{"arm":<62} {"step":>6} {"slot0":>7} {"slot15":>7} {"mean":>7} {"s15/s0":>7}')
    plot = []
    for run in sorted(chosen):
        d = runs[run][chosen[run]]
        ratio = d[-1] / d[0] if d[0] else float('nan')
        print(f'{short(run):<62} {chosen[run]//1000:5d}k {d[0]:7.2f} {d[-1]:7.2f} '
              f'{d.mean():7.2f} {ratio:7.2f}')
        plot.append((short(run), chosen[run], d))

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for name, step, d in plot:
        fam, c, ls, m = family(name)
        lab = f'{name}  ({step//1000}k)' if args.latest else name
        ax.plot(np.arange(len(d)), d, color=c, ls=ls, marker=m, ms=3.5, lw=1.5, label=lab)
    ax.set_xlabel('candidate slot  (= n index)', color=INK)
    ax.set_ylabel('mean pairwise endpoint distance (px)', color=INK)
    ax.set_ylim(bottom=0)
    ax.set_title(f'Candidate dispersion by arm — {label}\n'
                 f'arena is 512px; a spread statistic, not a mode count',
                 color=INK, fontsize=12, loc='left')
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, ncol=2,
              loc='upper left', bbox_to_anchor=(0, -0.12))
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    out = pathlib.Path(args.out or (root / f'compare_arms_{tag}.png'))
    fig.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
