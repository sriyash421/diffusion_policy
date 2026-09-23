"""Compare context use ACROSS arms, from the per-run attention.json files.

    python attention_analysis/compare_arms.py [--dir /gscratch/.../attention_analysis]

Each run's visualize_attention.py writes `context_mass_per_slot`: the share of the decoder's
cross-attention that lands on the search-context block, per candidate slot. This collects
them so the arms can be read against each other, which is the actual question -- does an
obs-noise ladder, or a goal mask, change HOW the policy uses its context, not just how well
it scores.

THREE THINGS THAT ARE NOT COMPARABLE, and are separated rather than averaged away:
  * Arms probed at different STATES. `states: episode-init` reads attention at each test
    episode's t=0, where the T has not been touched; `states: test-windows` reads it on the
    held-out transitions the arm was actually trained on. Those are different measurements of
    different states, so each probe set gets its OWN figure and its own table block -- never
    one axes. (The same reason the mode comparator splits by width.)
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
import fnmatch
import json
import pathlib
import re
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


def short(run):
    """Collapse the policy tokens to `k<N>`, for ANY width.

    Spelled as a regex rather than one replace per width: the k=4 arms were added in
    2026-09 and a hardcoded k16/k1 list silently left them with the long token, which
    breaks the column alignment the analysis-folder naming convention exists for.
    """
    run = re.sub(r'_?value_k(\d+)_ver-t_goal', lambda m: f'_k{m.group(1)}', run)
    return run.replace('_enc-resnet18', '').replace('_seed-42', '').lstrip('_')


def collect(root, pats=None):
    """Rows from every attention.json, one run per directory, labelled by FOLDER name.

    `pats` is a list of fnmatch globs on the folder name. The tree holds every generation
    at once, so without a filter one axes mixes arms trained on different splits.

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
        if pats and not any(fnmatch.fnmatch(j.parent.name, p) for p in pats):
            continue
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
                    # Which observations the attention was measured on. Absent means
                    # episode-init: every attention.json written before 2026-09-18 probed
                    # each test episode's t=0 state, and that is not the same measurement as
                    # the held-out-transition probe the filtered arms use.
                    'states': d.get('states', 'episode-init'),
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
    ap.add_argument('--runs', default=None,
                    help='comma-separated globs on the analysis folder name, e.g. "*split-mv*"; '
                         'default is every arm in --dir, which spans generations')
    ap.add_argument('--states', default='all',
                    help="probe set to report: 'all' (default, one figure each), "
                         "'episode-init' or 'test-windows'")
    args = ap.parse_args()
    root = pathlib.Path(args.dir)
    pats = [x for x in (args.runs or '').split(',') if x]
    rows = collect(root, pats)
    if not rows:
        print(f'no attention.json under {root}'); return

    # ONE REPORT PER PROBE SET. episode-init and test-windows measure attention at different
    # states, so putting them on one axes would compare two different experiments. Every set
    # present is reported -- nothing is dropped for being in the minority.
    sets = sorted({r['states'] for r in rows})
    if args.states != 'all':
        sets = [x for x in sets if x == args.states]
        if not sets:
            print(f'no arm was probed with states={args.states}'); return
    if len(sets) > 1:
        print(f'{len(sets)} probe sets present: {", ".join(sets)} -- reported separately, '
              f'because they are not the same measurement\n')
    for st in sets:
        report([r for r in rows if r['states'] == st], st, args, root)


def report(rows, states, args, root):
    dsel = args.denoise if args.denoise is not None else max(r['denoise'] for r in rows)
    print(f'=== probe set: {states} ===')
    print(f'context share of cross-attention, denoising step {dsel}\n')
    print(f'{"arm":<52} {"step":>6} {"K":>3} {"slot1":>6} {"slotK/2":>7} {"slotK-1":>8} '
          f'{"mean":>6}')
    plot = []
    for r in sorted(rows, key=lambda r: (r['run'], r['step'])):
        if r['denoise'] != dsel:
            continue
        name = short(r['run'])
        ps = r['per_slot']
        if not ps:
            # ST k=1: no context block. '--' not 0.0 -- see the module docstring.
            print(f'{name:<52} {r["step"]//1000:5d}k {"--":>3} {"--":>6} {"--":>7} '
                  f'{"--":>8} {"--":>6}   (no context tokens)')
            continue
        # Columns are K-RELATIVE. A fixed slot 8 is an IndexError at K=4, and comparing
        # absolute slot 8 across widths would compare different fractions of the ladder.
        K = len(ps)
        print(f'{name:<52} {r["step"]//1000:5d}k {K:3d} {ps[1]:6.2f} {ps[K//2]:7.2f} '
              f'{ps[-1]:8.2f} {np.mean(ps[1:]):6.2f}')
        plot.append((name, r['step'], np.asarray(ps)))

    if not plot:
        print('  no arm has a context block yet\n'); return

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
    ax.set_title(f'Context use by arm — {label}, denoising step {dsel}\n'
                 f'probed on {states}',
                 color=INK, fontsize=12, loc='left')
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, ncol=2,
              loc='upper left', bbox_to_anchor=(0, -0.12))
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    stem = pathlib.Path(args.out).with_suffix('') if args.out else (
        root / f'compare_arms_d{dsel}_{tag}')
    out = pathlib.Path(f'{stem}_{states}.png')
    fig.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
