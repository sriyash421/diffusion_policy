"""Best-of-n sweeps for the 2026-10-06 far-50 reruns: success and coverage vs n, argmax and
final-candidate selection, one figure per held-out quadrant.

Series are the 10k checkpoint of every arm (the only checkpoint all k=16 arms have reached)
plus the BC checkpoint a deployment would pick: the one with the highest n=1 success in its
sweep (ties -> latest step). Lines are the mean over seeds, bands the min-max across seeds;
the legend says how many seeds each series has. Colour + marker + line style distinguish
series (dashed = corrupted-observation eval), so no hue-only distinction.

    python scripts/plot_far50_sweeps.py                 # ST at 10k  -> <split>_sweep.png
    python scripts/plot_far50_sweeps.py --st-ckpt best  # ST at its best checkpoint anywhere in
                                                        # the sweep -> <split>_sweep_best.png
"best" = per arm and seed, the checkpoint holding the highest success cell over n, selection
and observation mode (ties: smaller n, earlier step, argmax -- select_pusht_sweep's order).
That is selection on the test set and is labelled as such; BC keeps its deployment rule.
"""
import argparse, csv, glob, json, pathlib, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search_imgonly'
OUT = pathlib.Path(__file__).resolve().parents[1] / 'docs/reports/figures/far50_2026-10-06'
CSV = pathlib.Path(__file__).resolve().parents[1] / 'docs/reports/summaries/far50_sweeps_2026-10-06.csv'
NS = [1, 2, 4, 8, 16, 32, 64]
SPLITS = {'blq50far': 'bottom-left held out (blq50far)',
          'tlq50far': 'top-left held out (tlq50far, val from train pool)',
          'brq50far': 'bottom-right held out (brq50far, val from train pool)'}
# Okabe-Ito palette; one marker per arm.
STYLE = {'BC': ('#000000', 'o'), 'ST k1': ('#E69F00', '^'), 'ST k16 uniform': ('#56B4E9', 'v'),
         'ST k16 flat600': ('#009E73', 'D'), 'ST k16 ramp200to600': ('#D55E00', 'P')}


def arm_of(run):
    if run.startswith('unetbc'): return 'BC'
    if '_k1_' in run: return 'ST k1'
    if 'flat600' in run: return 'ST k16 flat600'
    if 'ramp200to600' in run: return 'ST k16 ramp200to600'
    return 'ST k16 uniform'


def load():
    rows = []
    for f in glob.glob(f'{ROOT}/*/*_split-*50far*/bon_search*/step_*/success_curve.json'):
        d = json.load(open(f))
        if d.get('candidate_rng') != 'paired_decision_v3' or not d.get('terminal_scoring'):
            continue
        run = f.split('/')[-4]
        rows.append(dict(
            split=re.search(r'_split-(\w+?50far)', run).group(1),
            seed=int(re.search(r'_seed-(\d+)', run).group(1)), arm=arm_of(run),
            sel=d.get('selection', 'argmax'), obs='corrupt' if d.get('corrupt_obs_eval') else 'clean',
            step=int(f.split('/')[-2].split('_')[1]), n=d['n'], sr=d['success_rate'],
            ci=d['success_ci'], cov=d['mean_coverage'], n_episodes=d['n_episodes']))
    return rows


def bc_deploy_step(rows, split, seed):
    """Highest n=1 argmax success over the BC sweep; ties -> latest step."""
    cands = [(dict(zip(r['n'], r['sr'])).get(1, -1), r['step']) for r in rows
             if r['split'] == split and r['seed'] == seed and r['arm'] == 'BC' and r['sel'] == 'argmax']
    return max(cands)[1] if cands else None


def best_step(rows, split, arm, seed):
    """Checkpoint with the best cell anywhere in this arm/seed's sweep; None if no rows."""
    cells = []
    for r in rows:
        if (r['split'], r['arm'], r['seed']) != (split, arm, seed):
            continue
        for n, sr in zip(r['n'], r['sr']):
            cells.append((sr, -n, -r['step'], r['sel'] == 'argmax', r['step'], n, r['sel'], r['obs']))
    if not cells:
        return None
    best = max(cells)
    return dict(step=best[4], n=best[5], sel=best[6], obs=best[7], success=best[0])


def series(rows, split, arm, sel, obs, step_of):
    """Per-seed (n -> value) dicts for success and coverage."""
    out = []
    for seed in (42, 43, 44):
        step = step_of(seed)
        rs = [r for r in rows if r['split'] == split and r['seed'] == seed and r['arm'] == arm
              and r['sel'] == sel and r['obs'] == obs and r['step'] == step]
        if rs:
            r = rs[0]
            out.append((seed, step, dict(zip(r['n'], r['sr'])), dict(zip(r['n'], r['cov']))))
    return out


def draw(ax, per_seed, key, label, color, marker, ls, hollow):
    if not per_seed:
        return
    idx = 2 if key == 'sr' else 3
    ns = [n for n in NS if all(n in s[idx] for s in per_seed)]
    if not ns:
        return
    vals = np.array([[s[idx][n] for n in ns] for s in per_seed])
    mean, lo, hi = vals.mean(0), vals.min(0), vals.max(0)
    mfc = 'white' if hollow else color
    ax.plot(ns, mean, color=color, marker=marker, ls=ls, mfc=mfc, mec=color, lw=1.6, ms=6,
            label=f'{label} [{len(per_seed)} seed{"s" if len(per_seed) > 1 else ""}]')
    if len(per_seed) > 1:
        ax.fill_between(ns, lo, hi, color=color, alpha=0.12, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--st-ckpt', choices=['10k', 'best'], default='10k')
    a = ap.parse_args()
    rows = load()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(CSV, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['split', 'arm', 'seed', 'selection', 'obs', 'step', 'n', 'success', 'ci_lo', 'ci_hi',
                    'coverage', 'n_episodes'])
        for r in sorted(rows, key=lambda r: (r['split'], r['arm'], r['seed'], r['sel'], r['obs'], r['step'])):
            for n, s, c, cv in zip(r['n'], r['sr'], r['ci'], r['cov']):
                w.writerow([r['split'], r['arm'], r['seed'], r['sel'], r['obs'], r['step'], n,
                            f'{s:.4f}', f'{c[0]:.4f}', f'{c[1]:.4f}', f'{cv:.4f}', r['n_episodes']])
    print('wrote', CSV)

    for split, title in SPLITS.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
        deploy = {s: bc_deploy_step(rows, split, s) for s in (42, 43, 44)}
        best = {(arm, s): best_step(rows, split, arm, s) for arm in STYLE for s in (42, 43, 44)}
        if a.st_ckpt == 'best':
            with open(OUT / f'{split}_best_ckpts.json', 'w') as fh:
                json.dump({f'{arm}|{s}': v for (arm, s), v in best.items() if v}, fh, indent=1)
        for col, sel in enumerate(['argmax', 'final_pass']):
            for row, key in enumerate(['sr', 'cov']):
                ax = axes[row, col]
                c, m = STYLE['BC']
                # BC only has argmax; it is the reference in both columns. The deployment
                # checkpoint is the baseline (dark); the 10k checkpoint is shown grey.
                ck = '/'.join(f'{deploy[s] // 1000}' for s in (42, 43, 44) if deploy.get(s))
                draw(ax, series(rows, split, 'BC', 'argmax', 'clean', lambda s: deploy[s]), key,
                     f'BC, ckpt {ck}k (argmax)', c, m, '-', False)
                draw(ax, series(rows, split, 'BC', 'argmax', 'clean', lambda s: 10000), key,
                     'BC undertrained, 10k (argmax)', '#7F7F7F', 's', ':', False)
                for arm in ['ST k1', 'ST k16 uniform', 'ST k16 flat600', 'ST k16 ramp200to600']:
                    c, m = STYLE[arm]
                    if a.st_ckpt == '10k':
                        step_of, tag = (lambda s: 10000), '10k'
                    else:
                        step_of = lambda s, arm=arm: best[(arm, s)]['step'] if best[(arm, s)] else None
                        tag = 'best ckpt ' + '/'.join(f'{best[(arm, s)]["step"] // 1000}'
                                                      for s in (42, 43, 44) if best[(arm, s)]) + 'k'
                    for obs, ls, hollow in (('clean', '-', False), ('corrupt', '--', True)):
                        draw(ax, series(rows, split, arm, sel, obs, step_of), key,
                             f'{arm} {tag}' + (' (corrupt obs)' if obs == 'corrupt' else ''), c, m, ls, hollow)
                ax.set_xscale('log', base=2); ax.set_xticks(NS); ax.set_xticklabels(NS)
                ax.set_ylim(0, 1 if key == 'sr' else 1); ax.grid(alpha=0.3)
                ax.set_title(('success rate' if key == 'sr' else 'mean coverage')
                             + f' -- {"argmax" if sel == "argmax" else "final candidate"} selection')
                if row == 1: ax.set_xlabel('n candidates')
                if col == 0: ax.set_ylabel('success rate' if key == 'sr' else 'mean coverage')
        fig.suptitle(f'PushT 50 demos, {title}\n'
                     f'BC = deployment checkpoint (highest n=1 success in its sweep, ties -> latest). '
                     + ('ST arms at 10k.' if a.st_ckpt == '10k' else
                        'ST arms at their best checkpoint anywhere in the sweep (selected on test).')
                     + '\nCheckpoints listed per seed 42/43/44. Lines = mean over seeds, bands = min-max.',
                     fontsize=10)
        handles, labels = axes[0, 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.15, 1, 0.94))
        p = OUT / f'{split}_sweep{"" if a.st_ckpt == "10k" else "_best"}.png'; fig.savefig(p, dpi=140); plt.close(fig); print('wrote', p)


if __name__ == '__main__':
    main()
