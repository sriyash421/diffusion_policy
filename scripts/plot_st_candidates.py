"""Best-of-N figures for the stage policies STk0..STk7 of the in-context arms (train_pusht_dt_asteroid),
from <run>/bon_search/stage_eval/pi_k<k>/{train,test}_candidates.npz (scripts/slurm/eval_stage_candidates.sbatch)
and the U-Net BC full300's candidates at its test-best step.

  bon_<i>.png           per arm; x = candidates n = 1..8; per STk: coverage of the verifier-argmax among the first n
                        candidates (best of n); BC dashed black.
  perf_<i>.png          the same for candidate n alone (perf@n).
  curves_<tag>.png      per arm, during training; x = gradient steps within stage k (eval every iteration), one line
                        per STk: attempt k+1 after its own k attempts; BC = its first candidate.
  full_context.png      all arms; STk only at n = k+1 (k own attempts in context): solid BoN@k+1, dashed candidate k+1;
                        BC: BoN@n and candidate n.
  bcfail_context.png    (--bcfail) all test starts, then split into BC-failure / other; against k: coverage gain of STk's
                        candidate k+1 over BC's candidate 1 (per start), then coverage and success; numbers also in
                        bcfail_context.json. An arm named in --first-candidate (the no-context control) is scored on
                        candidate 1 instead, here only.
Metric: max coverage of the T in the executed candidate, mean over the starts.

    python scripts/plot_st_candidates.py --arm 'clean, 1 chain=<run>' [--arm ...] --bc-run <unet run> --bc-step 50000 --out-dir <dir>
"""
import json
import pathlib
import re

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_dt_rounds import BC_DIRS, MARKERS

SPLITS = (('train', 'TRAIN starts: the training demos'), ('test', 'TEST starts: 50 held-out (blq50far)'))
ARM_COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9')
ARM_MARKERS = ('o', 's', '^', 'D', 'v', 'P')
N = 8
FIRST = set()   # arm labels scored on candidate 1 in bcfail_context (and left out of the other figures)


def bon_final(z):
    """(BoN@n, candidate n) mean max coverage for n = 1..N."""
    cov, val = z['max_coverage'][:, :N], z['value'][:, :N]
    rows = np.arange(len(cov))
    bon = [cov[rows, val[:, :n].argmax(1)].mean() for n in range(1, N + 1)]
    return np.array(bon), cov.mean(0)


def stage_curves(run, split):
    """{k: (bon, final)} for every stage with candidates on this split."""
    out = {}
    for d in (pathlib.Path(run) / 'bon_search' / 'stage_eval').glob('pi_k*'):
        f = d / f'{split}_candidates.npz'
        if f.is_file():
            out[int(re.search(r'(\d+)$', d.name).group(1))] = bon_final(np.load(f, allow_pickle=True))
    return dict(sorted(out.items()))


def bc_candidates(bc_run, step, split, keep=None):
    """BC's {split}_candidates.npz as a dict; `keep` restricts its rows to those episodes (a BC scored on a
    parent manifest's larger test set, read on a child manifest's test)."""
    z = dict(np.load(pathlib.Path(bc_run) / 'bon_search' / BC_DIRS[split] / f'step_{step:07d}'
                     / f'{split}_candidates.npz', allow_pickle=True))
    if keep is None:
        return z
    rows = np.isin(z['episode_idxs'], sorted(keep))
    assert rows.sum() == len(keep), f'BC {split} eval lacks {len(keep) - rows.sum()} of the requested episodes'
    return {k: v[rows] if getattr(v, 'shape', ())[:1] == rows.shape else v for k, v in z.items()}


def manifest_test(path):
    return set(json.loads(pathlib.Path(path).read_text())['test']) if path else None


def bc_curves(bc_run, step, split, keep=None):
    return bon_final(bc_candidates(bc_run, step, split, keep if split == 'test' else None))


KINDS = {'bon': (0, 'best of n: the verifier value picks one of the first n, it is executed',
                 'candidates generated n (verifier picks one, it is executed)', 'BoN@8'),
         'perf': (1, 'perf@n: candidate n alone, no selection (STk: n-1 own earlier candidates in context; '
                     'BC: its n-th i.i.d. sample)', 'candidate n (n-1 own earlier candidates in context)', 'perf@8')}


def plot_arm(label, run, bc, path, kind):
    j, what, xlabel, tag = KINDS[kind]
    x = np.arange(1, N + 1)
    colors = plt.cm.viridis(np.linspace(0, .9, 8))
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (split, title) in zip(axs, SPLITS):
        handles = []
        for k, curves in stage_curves(run, split).items():
            y, c, m = curves[j], colors[k], MARKERS[k % len(MARKERS)]
            ax.plot(x, y, color=c, marker=m, ms=4, lw=1.6)
            ax.annotate(f'k{k}', (x[-1], y[-1]), xytext=(4, 0), textcoords='offset points', color=c, fontsize=8,
                        va='center')
            handles.append(Line2D([], [], color=c, marker=m, label=f'STk{k} (context = {k}): {tag} {y[-1]:.2f}'))
        y = bc[split][j]
        ax.plot(x, y, color='black', marker='x', lw=2, ls='--')
        handles.append(Line2D([], [], color='black', marker='x', ls='--', label=f'BC (UNet): {tag} {y[-1]:.2f}'))
        ax.legend(handles=handles, fontsize=7, ncol=2, loc='upper left')
        ax.set_title(title, fontsize=9); ax.set_xlabel(xlabel)
        ax.set_xticks(x); ax.set_ylim(0, 1); ax.grid(alpha=.25)
    axs[0].set_ylabel('max coverage (mean over starts, 1 seed)')
    fig.suptitle(f'{label} -- {what}', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close(fig)
    print('wrote', path)


def style_legend(ax, extra):
    h = [Line2D([], [], color='gray', ls='-', label='solid: BoN (verifier argmax over the first n)'),
         Line2D([], [], color='gray', ls='--', label='dashed: candidate n (final)')]
    # below the axes: the BC lines run along the top of the panel
    ax.legend(handles=extra + h, fontsize=7, ncol=2, loc='upper center', bbox_to_anchor=(.5, -.14))


def stage_logs(run):
    """{k: (updates, {split: coverage})} from the eval lines of <run>/logs.json.txt; a resumed job
    re-logs the evals after its last mid-stage save, so the last line per update wins."""
    by = {}
    f = pathlib.Path(run) / 'logs.json.txt'
    for line in (f.read_text().splitlines() if f.is_file() else []):
        if '"eval/train_coverage"' not in line:
            continue
        d = json.loads(line)
        by.setdefault(d['stage'], {})[d['update']] = {s: d[f'eval/{s}_coverage'] for s, _ in SPLITS}
    return {k: (sorted(v), {s: [v[u][s] for u in sorted(v)] for s, _ in SPLITS}) for k, v in by.items()}


def plot_curves(label, run, bc, path):
    logs = stage_logs(run)
    if not logs:
        return
    colors = plt.cm.viridis(np.linspace(0, .9, 8))
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (split, title) in zip(axs, SPLITS):
        handles = []
        for k, (u, cv) in sorted(logs.items()):
            c, m = colors[k], MARKERS[k % len(MARKERS)]
            ax.plot(u, cv[split], color=c, marker=m, ms=3, lw=1.3, markevery=5)
            ax.annotate(f'k{k}', (u[-1], cv[split][-1]), xytext=(4, 0), textcoords='offset points', color=c,
                        fontsize=8, va='center')
            handles.append(Line2D([], [], color=c, marker=m, label=f'STk{k}: end {cv[split][-1]:.2f}'))
        f1 = bc[split][1][0]
        ax.axhline(f1, color='black', ls='--', lw=1.5)
        handles.append(Line2D([], [], color='black', ls='--', label=f'BC (UNet), one candidate: {f1:.2f}'))
        ax.legend(handles=handles, fontsize=7, ncol=2, loc='upper left')
        ax.set_title(title, fontsize=9); ax.set_xlabel('gradient steps within stage k')
        ax.set_ylim(0, 1); ax.grid(alpha=.25)
    axs[0].set_ylabel('max coverage of attempt k+1 (mean over starts, 1 seed)')
    u = logs[min(logs)][0]
    every = u[1] - u[0] if len(u) > 1 else 0
    fig.suptitle(f'{label}: during training -- STk after its own k attempts, eval every {every} gradient steps',
                 fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close(fig)
    print('wrote', path)


def plot_full_context(arms, bc, path):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (split, title) in zip(axs, SPLITS):
        handles = []
        for i, (label, run) in enumerate(arms.items()):
            st = stage_curves(run, split)
            if not st or label in FIRST:
                continue
            ks = np.array(list(st))
            b = np.array([st[k][0][k] for k in ks]); f = np.array([st[k][1][k] for k in ks])
            c, m = ARM_COLORS[i], ARM_MARKERS[i]
            ax.plot(ks + 1, b, color=c, marker=m, lw=2, ls='-')
            ax.plot(ks + 1, f, color=c, marker=m, lw=1.4, ls='--', alpha=.85)
            handles.append(Line2D([], [], color=c, marker=m, label=f'{label}: STk{ks[-1]} BoN {b[-1]:.2f}, final {f[-1]:.2f}'))
        x = np.arange(1, N + 1); b, f = bc[split]
        ax.plot(x, b, color='black', marker='x', lw=2, ls='-')
        ax.plot(x, f, color='black', marker='x', lw=1.5, ls='--')
        handles.append(Line2D([], [], color='black', marker='x', label=f'BC (UNet): BoN@8 {b[-1]:.2f}, final@8 {f[-1]:.2f}'))
        style_legend(ax, handles)
        ax.set_title(title, fontsize=9); ax.set_xticks(x)
        ax.set_xlabel('candidates n  (STk at n = k+1: k own attempts in context)')
        ax.set_ylim(0, 1); ax.grid(alpha=.25)
    axs[0].set_ylabel('max coverage (mean over starts)')
    fig.suptitle('STk at its full context (n = k+1) vs BC', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close(fig)
    print('wrote', path)


def candidate_k1(run, k, keep, c=None):
    """(coverage, success) of STk's candidate k+1 (or candidate c+1) per test start, ordered by episode, rows in
    `keep`."""
    f = pathlib.Path(run) / 'bon_search' / 'stage_eval' / f'pi_k{k}' / 'test_candidates.npz'
    if not f.is_file():
        return None
    z = np.load(f, allow_pickle=True)
    rows = {int(e): i for i, e in enumerate(z['episode_idxs'])}
    idx = [rows[e] for e in keep]
    c = k if c is None else c
    return z['max_coverage'][idx, c], z['success'][idx, c].astype(bool), z['success'][idx].any(1)


def plot_bcfail(arms, sidecar, path):
    """Test starts split by whether BC failed there. Headline: coverage gain of STk's candidate k+1 over BC's
    candidate 1 (per start; BC = mean over its eval seeds), then raw coverage and success, against k."""
    side = json.loads(pathlib.Path(sidecar).read_text())
    per = {int(e): v for e, v in side['per_start'].items()}
    groups = [('all test starts (BC-failure + other)', sorted(per)),
              (f'BC-failure test starts (BC candidate 1 {side.get("fail_rule", "not a success")} under '
               f'>= {side["min_fail"]} of {len(side["eval_seeds"])} eval seeds)', sorted(e for e, v in per.items() if v['bc_fail'])),
              ('other test starts', sorted(e for e, v in per.items() if not v['bc_fail']))]
    groups = [g for g in groups if g[1]]
    if len(groups) == 2:                                   # one group is empty: "all" would repeat the other
        groups = groups[1:]
    rows = (('gain', 'coverage gain over BC: STk candidate k+1 - BC candidate 1\n(per start, mean over starts)'),
            ('cov', 'max coverage of candidate k+1 (mean over starts)'),
            ('succ', 'success rate of candidate k+1 (coverage >= 0.95)'))
    fig, axs = plt.subplots(len(rows), len(groups), figsize=(7 * len(groups), 13), sharex=True, squeeze=False)
    summary = {}
    for col, (title, eps) in enumerate(groups):
        bc_cov = np.array([per[e]['cov'] for e in eps]); bc_succ = np.array([per[e]['success'] for e in eps])
        bc_start = bc_cov.mean(1)                          # per-start BC coverage, mean over eval seeds
        st = {label: {k: v for k in range(N)
                      if (v := candidate_k1(run, k, eps, 0 if label in FIRST else None)) is not None}
              for label, run in arms.items()}
        for row, (key, ylab) in enumerate(rows):
            ax = axs[row, col]
            if key == 'gain':
                ax.axhline(0, color='black', ls='--', lw=1.8)
                handles = [Line2D([], [], color='black', ls='--', label='no gain over BC')]
            else:
                bc_m = (bc_cov if key == 'cov' else bc_succ).mean(0)
                ax.axhspan(bc_m.min(), bc_m.max(), color='black', alpha=.12, lw=0)
                ax.axhline(bc_m.mean(), color='black', ls='--', lw=1.8)
                handles = [Line2D([], [], color='black', ls='--',
                                  label=f'BC (UNet) candidate 1: {bc_m.mean():.2f} (band: range over eval seeds)')]
            for i, (label, pts) in enumerate(st.items()):
                if not pts:
                    continue
                ks = np.array(list(pts))
                if key == 'gain':
                    y = np.array([(pts[k][0] - bc_start).mean() for k in ks])
                    up = (pts[ks[-1]][0] > bc_start).mean()
                    lab = f'{label}: STk{ks[-1]} {y[-1]:+.2f}, better than BC on {up:.0%} of starts'
                else:
                    y = np.array([pts[k][0 if key == 'cov' else 1].mean() for k in ks])
                    lab = f'{label}: STk{ks[-1]} {y[-1]:.2f}'
                c, m = ARM_COLORS[i], ARM_MARKERS[i]
                ax.plot(ks, y, color=c, marker=m, lw=2)
                ax.annotate(label, (ks[-1], y[-1]), xytext=(4, 0), textcoords='offset points', color=c, fontsize=8,
                            va='center')
                handles.append(Line2D([], [], color=c, marker=m, label=lab))
                if row == 0:
                    kl = ks[-1]
                    summary.setdefault(label, {})[title] = dict(
                        n_starts=len(eps), last_stage=int(kl),
                        bc_coverage_c1=round(float(bc_start.mean()), 3),
                        st_coverage_by_k={int(k): round(float(pts[k][0].mean()), 3) for k in ks},
                        coverage_gain_by_k={int(k): round(float((pts[k][0] - bc_start).mean()), 3) for k in ks},
                        frac_starts_improved_by_k={int(k): round(float((pts[k][0] > bc_start).mean()), 3) for k in ks},
                        bc_success_c1_by_seed=bc_succ.mean(0).round(3).tolist(),
                        st_success_by_k={int(k): round(float(pts[k][1].mean()), 3) for k in ks},
                        n_solved_by_last_candidate=int(pts[kl][1].sum()),
                        n_solved_in_any_of_8_attempts_of_last_stage=int(pts[kl][2].sum()))
            ax.legend(handles=handles, fontsize=7, loc='upper left')
            ax.set_ylim(*((-1, 1) if key == 'gain' else (0, 1))); ax.grid(alpha=.25); ax.set_xticks(range(N))
            if col == 0:
                ax.set_ylabel(ylab, fontsize=9)
            if row == 0:
                ax.set_title(f'{title}\n{len(eps)} starts', fontsize=9)
            if row == len(rows) - 1:
                ax.set_xlabel('stage k (STk, k own attempts in context; candidate k+1 scored)')
    fig.suptitle('Does in-context feedback (veritas + T-to-goal heuristic) improve coverage where BC fails?',
                 fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close(fig)
    pathlib.Path(path).with_suffix('.json').write_text(json.dumps(summary, indent=1) + '\n')
    print('wrote', path)
    print(json.dumps(summary, indent=1))


@click.command()
@click.option('--arm', 'arms', multiple=True, required=True, help="'label=run_dir'")
@click.option('--bc-run', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--bc-step', required=True, type=int)
@click.option('--out-dir', required=True)
@click.option('--prefix', default='', help='output file-name prefix, for a second group of arms')
@click.option('--test-manifest', default=None, help="the arms' manifest: BC's test rows are cut to its test starts")
@click.option('--split-label', default=None, help="test panel title, replacing the blq50far one")
@click.option('--bcfail', default=None, help='<manifest>.bcfail.json (make_interp_split.py): adds bcfail_context.png')
@click.option('--first-candidate', 'first', multiple=True, help='arm label scored on candidate 1 in bcfail_context')
def main(arms, bc_run, bc_step, out_dir, prefix, test_manifest, split_label, bcfail, first):
    global SPLITS
    FIRST.update(first)
    if split_label:
        SPLITS = (SPLITS[0], ('test', split_label))
    arms = dict(a.split('=', 1) for a in arms)
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    keep = manifest_test(test_manifest)
    bc = {s: bc_curves(bc_run, bc_step, s, keep) for s, _ in SPLITS}
    for i, (label, run) in enumerate(arms.items()):
        if label in FIRST:
            continue
        plot_curves(label, run, bc, out / f'{prefix}curves_{i}.png')
        if any(stage_curves(run, s) for s, _ in SPLITS):
            for kind in KINDS:
                plot_arm(label, run, bc, out / f'{prefix}{kind}_{i}.png', kind)
    plot_full_context(arms, bc, out / f'{prefix}full_context.png')
    if bcfail:
        plot_bcfail(arms, bcfail, out / f'{prefix}bcfail_context.png')


if __name__ == '__main__':
    main()
