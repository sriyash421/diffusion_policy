"""Build / refresh the PushT W&B report (the maze report's structure) in l2sml/pusht_reports.

  1 · Chunked actions, quadrant 50-demo training sets (blq/tlq/brq50far, seeds 42-44)
  2 · Full trajectories (blq50far, seed 42)
  3 · Metrics on generated candidates (blq50far)
  4 · Data viz and diagnostics (split map, verifier example)

Two held-out sets: test (the held-out quadrant) and held-out train (val of the run's
`*_train50_far_valpool50` manifest: 50 training-region episodes it never trained on).
Checkpoints are chosen on each set separately (scripts/launch_valpool50_evals.best_step: BC by
N = 1 success, the rest by best success at any N) and each choice is scored on test, held-out
train and the 50 training demos themselves (`--phase train`); success and coverage figures. Results not yet on disk are left out of a figure and named in the text.

Figures are matplotlib images logged to one run in l2sml/pusht_reports and embedded as media
panels; the report keeps its URL across reruns (report_url.txt next to the figures).

    python scripts/wandb_fulltraj_report.py [--out DIR] [--dry-run]
"""
import argparse
import csv
import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from launch_valpool50_evals import (CHUNKED, FULL, IMG, QUADS, SEEDS, best_step, cand_dir, curves,  # noqa: E402
                                    readout, split_file, val_curves)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENTITY, PROJECT, TITLE = 'l2sml', 'pusht_reports', 'L2S PushT Experiments'
QNAME = {'blq50far': 'Bottom-left held out', 'tlq50far': 'Top-left held out', 'brq50far': 'Bottom-right held out'}
VERIFIER_FIG = pathlib.Path('/gscratch/robotics/harine/codex_reports/pusht_bc_st_20261004/outputs/'
                            'verifier_examples/verifier_feedback_examples.png')
# chunked arms as plotted: (label, CHUNKED label, corrupted obs); colour + marker, never hue alone
ARMS = [('BC U-Net · best N=1', 'BC', False), ('ST k=1', 'ST k1', False),
        ('ST k=16', 'ST k16 uniform', False), ('ST k=16 · flat600', 'ST k16 flat600', False),
        ('ST k=16 · ramp', 'ST k16 ramp200to600', False),
        ('ST k=16 · flat600 · corrupted obs', 'ST k16 flat600', True),
        ('ST k=16 · ramp · corrupted obs', 'ST k16 ramp200to600', True)]
STYLE = [('#009E73', 'D'), ('#000000', 's'), ('#D55E00', '^'), ('#CC79A7', 'h'), ('#56B4E9', 'v'),
         ('#882255', 'P'), ('#E69F00', 'X')]
NS = [1, 2, 4, 8, 16, 32, 64]
CAND_ARMS = [('ST k=1', 'ST k1'), ('ST k=16', 'ST k16 uniform'), ('ST k=16 · flat600', 'ST k16 flat600'),
             ('ST k=16 · ramp', 'ST k16 ramp200to600')]
CAND_NS = [4, 8, 16, 64]
CAND_FULL = ['Online search (GPT-2)', 'DT k=8 · verifier end-feedback', 'DT k=8 · coverage end-feedback']
SET = {'test': 'Held-out test (quadrant)', 'val': 'Held-out train (unused training-region episodes)',
       'train': 'Training demos (the 50 trained on)'}


def kstep(step):
    return 'latest' if step < 0 else f'{step // 1000}k' if step >= 1000 else str(step)


def ci(rows):
    """Mean and 95% CI half-width over rows (seeds / episodes), ignoring NaN."""
    rows = np.asarray(rows, float); k = np.sum(~np.isnan(rows), 0)
    mu = np.nanmean(rows, 0); sd = np.nanstd(rows, 0, ddof=1) if len(rows) > 1 else np.zeros_like(mu)
    return mu, np.where(k > 1, 1.96 * np.nan_to_num(sd) / np.sqrt(np.maximum(k, 1)), 0)


def succ(curve):
    sr = dict(zip(curve['n'], curve['success_rate']))
    return np.array([sr.get(n, np.nan) for n in NS])


def cov(curve):
    """Mean over episodes of the episode's max reward, min(coverage / 0.95, 1)."""
    r = curve.get('per_n_rewards', {})
    return np.array([np.mean(r[str(n)]) if str(n) in r else np.nan for n in NS])


METRIC = {'success': (succ, 'Success rate'), 'coverage': (cov, 'Mean max reward = min(coverage / 0.95, 1)')}


# ------------------------------------------------------------------ 1 · chunked quadrants

def chunked_runs(q, arm):
    return [(s, IMG / tmpl.format(q=q, s=s)) for s in SEEDS for a, tmpl, _ in CHUNKED if a == arm]


# {panel: title}; key = '<set scored>@<set the checkpoint was chosen on>', laid out 2 x 3
# (rows: chosen on; columns: scored)
PANELS = {'test@test': 'Held-out test · checkpoint best on test',
          'val@test': 'Held-out train · checkpoint best on test',
          'train@test': 'Training demos · checkpoint best on test',
          'test@val': 'Held-out test · checkpoint best on held-out train',
          'val@val': 'Held-out train · checkpoint best on held-out train',
          'train@val': 'Training demos · checkpoint best on held-out train'}


def chunked_panels(q):
    """{panel: [(arm idx, {metric: (argmax per seed, final per seed)}, steps per seed)]}; missing cells skipped."""
    out = {p: [] for p in PANELS}
    for i, (_, arm, noisy) in enumerate(ARMS):
        per = {p: ({m: ([], []) for m in METRIC}, []) for p in out}
        for s, run in chunked_runs(q, arm):
            got = {split: (curves(run, readout(arm, 'argmax', noisy, split, q)),
                           curves(run, readout(arm, 'final_pass', noisy, split, q))) for split in ('val', 'train')}
            got['test'] = (curves(run, readout(arm, 'argmax', noisy)), curves(run, readout(arm, 'final_pass', noisy)))
            chosen = {'test': best_step(got['test'][0], arm), 'val': best_step(val_curves(run, arm, noisy, q) or {}, arm)}
            for panel in out:
                (src_a, src_f), step = got[panel.split('@')[0]], chosen[panel.split('@')[1]]
                if step is None or step not in src_a or len(src_a[step]['n']) < len(NS):
                    continue
                for m, (fn, _) in METRIC.items():
                    per[panel][0][m][0].append(fn(src_a[step]))
                    per[panel][0][m][1].append(fn(src_f[step]) if step in src_f else np.full(len(NS), np.nan))
                per[panel][1].append(step)
        for p, (by_m, st) in per.items():
            if st:
                out[p].append((i, by_m, st))
    return out


def chunked_figure(q, panels, metric):
    fig, axes = plt.subplots(2, 3, figsize=(27, 15))
    for ax, (p, lines) in zip(axes.flat, panels.items()):
        for i, by_m, st in lines:
            c, mk = STYLE[i]
            a, f = by_m[metric]
            mu, h = ci(a)
            ax.plot(NS, mu, color=c, marker=mk, lw=2.2, ms=7,
                    label=f'{ARMS[i][0]} · ckpt {", ".join(kstep(x) for x in st)}')
            ax.fill_between(NS, mu - h, mu + h, color=c, alpha=0.15, lw=0)
            if not np.all(np.isnan(f)):
                ax.plot(NS, np.nanmean(np.asarray(f, float), 0), color=c, marker=mk, ls='--', lw=1.5, ms=6,
                        markerfacecolor='white')
        ax.set_xscale('log', base=2); ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS])
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.set_title(PANELS[p], fontsize=13)
        ax.set_xlabel('N (candidates per decision)'); ax.set_ylabel(METRIC[metric][1])
        if lines:
            ax.legend(fontsize=9, loc='upper left', frameon=True, framealpha=0.85)
        else:
            ax.text(0.5, 0.5, 'evaluations pending', ha='center', transform=ax.transAxes, color='#555555')
    fig.suptitle(f'Chunked actions · {metric} · {QNAME[q]} ({q}) · 50 demos, seeds 42–44', fontsize=16, x=0.02, ha='left')
    fig.text(0.02, 0.945, 'solid: argmax (shaded: 95% CI across seeds)   ·   dashed, hollow: final candidate   ·   '
             'legend: checkpoint per seed (42, 43, 44)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    return fig


# ------------------------------------------------------------------ 2 · full trajectories

FULL_NS = [1, 2, 4, 8]


def dt_rows(run, sub, split):
    """{step: {(n, rule): row}} for one split of a run's eval_dt_pusht outputs."""
    out = {}
    for f in sorted((run / 'bon_search' / sub).glob('*/metrics.csv')):
        name = f.parent.name
        step = int(name[5:]) if name.startswith('step_') else -1
        rows = [r for r in csv.DictReader(open(f)) if r['split'] == split]
        if rows:
            out[step] = {(int(r['n']), r['rule']): r for r in rows}
    return out


def full_best(rows, bc):
    def key(step):
        cov = [float(rows[step][(n, 'argmax')]['mean_max_coverage']) for n in ([1] if bc else FULL_NS)
               if (n, 'argmax') in rows[step]]
        return (max(cov) if cov else -1, step)
    return max(rows, key=key) if rows else None


VAL_SUB = 'dt_eval_' + pathlib.Path(split_file('blq50far')).stem


def full_selection():
    """{label: {'test' / 'val' / 'train': rows, 'step_test' / 'step_val': step chosen on that set}}."""
    out = {}
    for i, (label, run) in enumerate(FULL):
        test, val = dt_rows(run, 'dt_eval', 'test'), dt_rows(run, VAL_SUB, 'val')
        complete = set(test) <= set(val)   # select only once every checkpoint has a held-out-train eval
        out[label] = dict(test=test, val=val, train=dt_rows(run, 'dt_eval_train', 'train'), step_test=full_best(test, i == 0),
                          step_val=full_best(val, i == 0) if complete else None)
    return out


def full_figure(sel):
    fstyle = [('#009E73', 'D'), ('#000000', 's'), ('#0072B2', 'o'), ('#56B4E9', 'v'), ('#D55E00', '^'), ('#CC79A7', 'P')]
    fig, axes = plt.subplots(2, 3, figsize=(27, 15))
    for ax, (p, title) in zip(axes.flat, PANELS.items()):
        src, which = p.split('@')[0], 'step_' + p.split('@')[1]
        drawn = False
        for (label, d), (c, mk) in zip(sel.items(), fstyle):
            step, rows = d[which], d[src]
            if step is None or step not in rows:
                continue
            for rule, ls, fill in (('argmax', '-', True), ('last', '--', False)):
                ys = [float(rows[step][(n, rule)]['mean_max_coverage']) if (n, rule) in rows[step] else np.nan for n in FULL_NS]
                ax.plot(FULL_NS, ys, color=c, marker=mk, ls=ls, lw=2.2 if fill else 1.5, ms=7,
                        markerfacecolor=c if fill else 'white',
                        label=f'{label} · ckpt {kstep(step)}' if fill else None)
            drawn = True
        ax.set_xscale('log', base=2); ax.set_xticks(FULL_NS); ax.set_xticklabels([str(n) for n in FULL_NS])
        ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.set_title(title, fontsize=13)
        ax.set_xlabel('N (full-trajectory attempts)'); ax.set_ylabel('Mean max coverage')
        if drawn:
            ax.legend(fontsize=9, loc='upper left', frameon=True, framealpha=0.85)
        else:
            ax.text(0.5, 0.5, 'evaluations pending', ha='center', transform=ax.transAxes, color='#555555')
    fig.suptitle('Full trajectories · Bottom-left held out (blq50far) · 50 demos, seed 42', fontsize=16, x=0.02, ha='left')
    fig.text(0.02, 0.945, 'solid: argmax (verifier value)   ·   dashed, hollow: final attempt   ·   '
             'coverage, not success: attempts almost never reach 0.95', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    return fig


# ------------------------------------------------------------------ 3 · candidate metrics

def row_panels(axs, series_by_n, ylabel, ns, raw):
    """series_by_n[n] = (lines (L, n), group ids (L,)). raw: thin lines; else per-group mean, then 95% CI."""
    for ax, n in zip(axs, ns):
        lines, grp = series_by_n[n]; x = np.arange(1, n + 1)
        if raw:
            for l in lines:
                ax.plot(x, l, color='#0072B2', alpha=max(0.03, min(0.4, 8 / len(lines))), lw=0.8)
        else:
            mu, h = ci([np.nanmean(lines[grp == g], 0) for g in np.unique(grp)])
            ax.plot(x, mu, color='#D55E00', lw=2); ax.fill_between(x, mu - h, mu + h, color='#D55E00', alpha=0.25)
        ax.set_title(f'N = {n} · ' + ('raw' if raw else 'mean over episodes'), fontsize=10)
        ax.set_xlabel('candidate #'); ax.grid(alpha=0.3)
        if n <= 16:
            ax.set_xticks(x)
    axs[0].set_ylabel(ylabel)


def metric_figure(series, rows, ns, title):
    fig, axs = plt.subplots(2 * len(rows), len(ns), figsize=(4.2 * len(ns), 3.0 * 2 * len(rows)), squeeze=False)
    for i, (key, lab) in enumerate(rows):
        row_panels(axs[2 * i], {n: series[n][key] for n in ns}, lab + '\n(every line)', ns, True)
        row_panels(axs[2 * i + 1], {n: series[n][key] for n in ns}, lab + '\n(mean, 95% CI)', ns, False)
    fig.suptitle(title, fontsize=12); fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    return fig


def chosen_figure(value, ns, title):
    """value: (L, max n), N = its first n columns; or {n: (L_n, n)}, each N its own rollout."""
    fig, axs = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 3.0), squeeze=False)
    for ax, n in zip(axs[0], ns):
        k = np.argmax(value[n] if isinstance(value, dict) else value[:, :n], 1); share = np.bincount(k, minlength=n)[:n] / max(len(k), 1)
        ax.bar(np.arange(1, n + 1), share, color='#009E73'); ax.set_title(f'N = {n} · {len(k)} choices', fontsize=10)
        ax.set_xlabel('chosen candidate # (argmax)'); ax.grid(alpha=0.3, axis='y')
        if n <= 16:
            ax.set_xticks(np.arange(1, n + 1))
    axs[0][0].set_ylabel('share'); fig.suptitle(title, fontsize=12); fig.tight_layout(rect=(0, 0, 1, 0.85))
    return fig


def chunked_cand():
    """{(label, split): (value (W, 64), rmse (W, 64), group ids, steps)} pooled over seeds."""
    out = {}
    for label, arm in CAND_ARMS:
        for split in ('test', 'val'):
            v, r, g, st = [], [], [], []
            for s in SEEDS:
                f = cand_dir(arm, s, split) / 'candidates.npz'
                if f.exists():
                    z = np.load(f)
                    v.append(z['value']); r.append(z['rmse']); g.append(z['episode'] + 1000 * s)
                    st.append(int(pathlib.Path(str(z['checkpoint'])).stem[5:]))
            if v:
                out[(label, split)] = (np.concatenate(v), np.concatenate(r), np.concatenate(g), st)
    return out


def chunked_rollout():
    """{(label, split): ({n: (value per decision (D, n), episode ids (D,))}, steps)} from the argmax
    rollouts at N = 4/8/16/64 (each N its own rollout), pooled over seeds."""
    out = {}
    for label, arm in CAND_ARMS:
        for split in ('test', 'val'):
            by_n, steps = {n: ([], []) for n in CAND_NS}, []
            for s in SEEDS:
                for f in cand_dir(arm, s, split, kind='rollout').glob('step_*/candidate_scores.jsonl'):
                    rows = [json.loads(line) for line in open(f)]
                    if {r['n'] for r in rows} >= set(CAND_NS):
                        steps.append(int(f.parent.name[5:]))
                        for r in rows:
                            by_n[r['n']][0].append(r['scores']); by_n[r['n']][1].append(r['episode'] + 1000 * s)
            if steps:
                out[(label, split)] = ({n: (np.array(v, float), np.array(g)) for n, (v, g) in by_n.items()}, steps)
    return out


def videos(label, split):
    """Tiled N = 4 · 8 · 16 · 64 videos of episodes 0-2, seed 42."""
    arm = dict(CAND_ARMS)[label]
    return sorted(cand_dir(arm, 42, split, kind='video').glob('ep0[0-2]_tiled_*.mp4'))


def demo_actions():
    import zarr
    z = zarr.open(str(ROOT / 'data/pusht_cchi_v7_replay.zarr'), 'r')
    ends = z['meta/episode_ends'][:]; act = z['data/action'][:]
    starts = np.r_[0, ends[:-1]]
    return {e: act[s:t] for e, (s, t) in enumerate(zip(starts, ends))}


def attempt_rmse(z, demos):
    """(E, K) RMSE in px of each attempt's actions vs the demo's, from the episode start; the shorter
    sequence holds its last action (PushT actions are target positions) to the longer's length."""
    E, K = z['value'].shape; out = np.full((E, K), np.nan)
    for e in range(E):
        d = demos[int(z['episode_idxs'][e])]
        for k in range(K):
            n = int(z['n_steps'][e, k])
            a = np.asarray(list(z['actions'][e, k][:n]), float).reshape(-1, 2)
            if not len(a):
                continue
            T = max(len(a), len(d))
            pa = np.concatenate([a, np.repeat(a[-1:], T - len(a), 0)]); pd = np.concatenate([d, np.repeat(d[-1:], T - len(d), 0)])
            out[e, k] = np.sqrt(((pa - pd) ** 2).mean())
    return out


def full_cand(sel, demos):
    """{(label, split): (value, max coverage, rmse) (E, 8), step}."""
    out = {}
    for label, run in FULL:
        if label not in CAND_FULL:
            continue
        d = sel[label]
        for split, step, sub in (('test', d['step_test'], 'dt_eval'), ('val', d['step_val'], VAL_SUB)):
            if step is None:
                continue
            f = run / 'bon_search' / sub / ('latest' if step < 0 else f'step_{step:07d}') / f'{split}_candidates.npz'
            if f.exists():
                z = np.load(f, allow_pickle=True)
                out[(label, split)] = (z['value'].astype(float), z['max_coverage'].astype(float), attempt_rmse(z, demos), step)
    return out


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
                     + ['| ' + ' | '.join(r) + ' |' for r in rows])


def slug(label):
    return ''.join(c if c.isalnum() else '_' for c in label).strip('_').replace('__', '_')


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/_fulltraj_report')
    ap.add_argument('--dry-run', action='store_true', help='figures only, no W&B')
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    figures, notes, video_paths = {}, {}, {}

    for q in QUADS:
        panels = chunked_panels(q)
        for m in METRIC:
            figures[f'chunked/{q}_{m}'] = chunked_figure(q, panels, m)
        notes[q] = ' '.join(f'{what} still running for: ' + ', '.join(miss) + '.' for what, miss in
                            (('Held-out-train evaluations', [a for i, (a, _, _) in enumerate(ARMS) if i not in {l[0] for l in panels['val@val']}]),
                             ('Training-demo evaluations', [a for i, (a, _, _) in enumerate(ARMS) if i not in {l[0] for l in panels['train@val']}]))
                            if miss)
    sel = full_selection()
    figures['full/blq50far'] = full_figure(sel)
    notes['full'] = ', '.join(l for l, d in sel.items() if d['step_val'] is None)
    notes['full_train'] = ', '.join(l for l, d in sel.items() if d['step_test'] not in d['train'])

    cand_chunk, rollout = chunked_cand(), chunked_rollout()
    chunk_rows, chunk_sections = [], []
    for label, _ in CAND_ARMS:
        for split in ('test', 'val'):
            ro, wi = rollout.get((label, split)), cand_chunk.get((label, split))
            if ro is None and wi is None:
                continue
            series, rows = {n: {} for n in CAND_NS}, []
            if ro is not None:
                rows.append(('v', 'verifier value\n(rollout decisions)'))
                for n in CAND_NS:
                    series[n]['v'] = ro[0][n]
            if wi is not None:
                rows.append(('r', 'RMSE to expert chunk\n(demo windows, normalized)'))
                for n in CAND_NS:
                    series[n]['r'] = (wi[1][:, :n], wi[2])
            steps = ro[1] if ro is not None else wi[3]
            head = f'{label} · {SET[split]} · checkpoint {", ".join(kstep(x) for x in sorted(set(steps)))}'
            title = f'Chunked · {head} · blq50far, seeds 42–44'
            key = f'cand/chunked_{slug(label)}_{split}'
            figures[key + '_metrics'] = metric_figure(series, rows, CAND_NS, title)
            if ro is not None:
                figures[key + '_chosen'] = chosen_figure({n: ro[0][n][0] for n in CAND_NS}, CAND_NS, title)
            vids = videos(label, split)
            for i, v in enumerate(vids):
                video_paths[f'{key}_video_e{i}'] = v
            chunk_sections.append((head, key, ro is not None, len(vids)))
            for n in (16, 64):
                val = (f'{ro[0][n][0][:, 0].mean():.2f} → {ro[0][n][0][:, n - 1].mean():.2f}' if ro is not None else '–')
                rm = (f'{wi[1][:, 0].mean():.2f} → {wi[1][:, n - 1].mean():.2f}' if wi is not None else '–')
                ch = (f'{np.argmax(ro[0][n][0], 1).mean() + 1:.1f} of {n}' if ro is not None else '–')
                chunk_rows.append((label, SET[split].split(' (')[0], str(n), val, rm, ch))
    demos = demo_actions()
    cand_full = full_cand(sel, demos)
    full_rows, full_sections = [], []
    for (label, split), (v, cov, r, step) in cand_full.items():
        eps = np.arange(len(v))
        series = {n: dict(v=(v[:, :n], eps), c=(cov[:, :n], eps), r=(r[:, :n], eps)) for n in (2, 4, 8)}
        title = f'Full trajectory · {label} · {SET[split]} · checkpoint {kstep(step)} · one line per episode (N = first N attempts)'
        key = f'cand/full_{slug(label)}_{split}'
        full_sections.append((f'{label} · {SET[split]} · checkpoint {kstep(step)}', key))
        figures[key + '_metrics'] = metric_figure(series, [('v', 'verifier value\n(final state)'), ('c', 'max coverage'),
                                                           ('r', 'RMSE whole attempt\n(vs demo, px)')], [2, 4, 8], title)
        figures[key + '_chosen'] = chosen_figure(v, [2, 4, 8], title)
        full_rows.append((label, SET[split].split(' (')[0], '8', f'{v[:, 0].mean():.2f} → {v[:, 7].mean():.2f}',
                          f'{cov[:, 0].mean():.2f} → {cov[:, 7].mean():.2f}', f'{np.nanmean(r[:, 0]):.0f} → {np.nanmean(r[:, 7]):.0f}',
                          f'{np.argmax(v, 1).mean() + 1:.1f} of 8'))

    paths = {}
    for key, fig in figures.items():
        p = out / (key.replace('/', '__') + '.png')
        fig.savefig(p, dpi=110); plt.close(fig); paths[key] = p
    paths['data/split_maps'] = out / 'split_maps.png'
    paths['data/verifier_example'] = VERIFIER_FIG
    print('figures:', len(paths), '->', out, '· videos:', len(video_paths))
    for k, v in notes.items():
        if v:
            print('pending', k, v)
    if args.dry_run:
        return

    import wandb
    import wandb_workspaces.reports.v2 as wr
    rid_file = out / 'run_id.txt'
    rid = rid_file.read_text().strip() if rid_file.exists() else wandb.util.generate_id()
    rid_file.write_text(rid)
    run = wandb.init(entity=ENTITY, project=PROJECT, id=rid, resume='allow', dir=str(out),
                     name='pusht_report_figures', settings=wandb.Settings(start_method='fork'))
    run.log({**{k: wandb.Image(str(p)) for k, p in paths.items()},
             **{k: wandb.Video(str(p), format='mp4') for k, p in video_paths.items()}})
    run.finish()

    def grid(keys, h=13):
        keys = [keys] if isinstance(keys, str) else keys
        rs = wr.Runset(entity=ENTITY, project=PROJECT, name='figures')
        w = 24 // len(keys)
        panels = [wr.MediaBrowser(media_keys=[k], num_columns=1, layout=wr.Layout(x=i * w, y=0, w=w, h=h))
                  for i, k in enumerate(keys)]
        return wr.PanelGrid(runsets=[rs], panels=panels)

    md = wr.MarkdownBlock
    blocks = [md('Solid: argmax over N candidates, shaded with a 95% CI across seeds · Dashed: final candidate.  \n'
                 'Two held-out sets: **test** = the held-out quadrant; **held-out train** = 50 training-region episodes '
                 'the run never trained on (val of the `*_train50_far_valpool50` manifest). Checkpoints are chosen '
                 'separately on each set (BC: best N = 1 success; the rest: best success at any N), and each is scored '
                 'on both sets and on the **training demos** themselves (the 50 trained on: a fit reference, not a '
                 'generalisation result): rows = the set the checkpoint was chosen on, columns = the set scored. '
                 'Legends list the checkpoint per seed. **Coverage** = mean over episodes of the episode\'s max '
                 'reward, min(coverage / 0.95, 1); success = that reward reaching 1.'),
              md('# 1 · Chunked actions'),
              md('## Quadrant 50-demo training sets\n'
                 '**Training set:** the 50 demos whose T starts farthest from the held-out quadrant.  \n'
                 '**Input:** image (ResNet-18 encoder).  \n'
                 '**Context:** up to 16 earlier candidate 16-step action chunks and their verifier values at the same '
                 'decision (execute 8, then replan); ST k=1 has none. flat600 / ramp: slot observation noise in training.  \n'
                 '**Feedback:** waypoint verifier (`wp_v3_la_tgn`, Gemini waypoint plan) on the simulated chunk.')]
    for q in QUADS:
        blocks += [md(f'### {QNAME[q]} ({q})' + (f'\n_{notes[q]}_' if notes[q] else '')),
                   md('**Success rate**'), grid(f'chunked/{q}_success', h=18),
                   md('**Coverage**'), grid(f'chunked/{q}_coverage', h=18)]
    blocks += [md('# 2 · Full trajectories'),
               md('## Bottom-left held out (blq50far)\n'
                  '**Input:** image (ResNet-18 encoder) at every step; the policy decides when to stop.  \n'
                  '**Context:** up to 7 earlier full attempts (their actions and feedback); k=1 arms and the open-loop '
                  'BC U-Net (whole 300-step trajectory from the first frame, i.i.d. attempts) have none.  \n'
                  '**Feedback:** −(T-to-goal + Veritas pin cost) at the realized state; *coverage end-feedback* uses the '
                  'attempt\'s coverage instead.'
                  + (f'\n_Held-out-train evaluations still running for: {notes["full"]}._' if notes['full'] else '')
                  + (f'\n_Training-demo evaluations still running for: {notes["full_train"]}._' if notes['full_train'] else '')),
               grid('full/blq50far', h=18)]

    blocks += [md('# 3 · Metrics on generated candidates\n'
                  '**Do ST policies with context generate better candidates as candidate # grows?** blq50far; test at '
                  'the checkpoint chosen on test, held-out train at the one chosen on held-out train.  \n'
                  '**Verifier value** = what argmax ranks (`wp_v3_la_tgn`, terminal scoring).  \n'
                  '**RMSE to expert** = vs the training target, the demo\'s next action chunk. PushT has no planner '
                  'expert (Maze\'s A*): the target exists only at states the demo visited, so it is measured on demo '
                  'windows, not rollouts · *chunked:* 256 windows per seed and set, each candidate vs the demo\'s next '
                  '16-step chunk over the 8 executed steps, normalized action space; one 64-candidate stream per '
                  'window, N = its first N candidates · *full trajectory:* each attempt\'s actions vs the demo\'s from '
                  'the episode start, in pixels, the shorter one holding its last action.  \n'
                  '**Chunked:** value and chosen # come from closed-loop argmax rollouts, each N its own rollout '
                  '(the ones behind section 1\'s points), one line per replanning decision; seeds 42–44 pooled. '
                  '**Full trajectory:** N = the first N attempts of one 8-attempt stream, one line per episode. Mean = '
                  'per-episode mean, then mean over episodes with a 95% CI. Histogram = how often each candidate # is '
                  'chosen (argmax). Videos: episodes 0–2, seed 42, N = 4 · 8 · 16 · 64 side by side.'),
               md('## Chunked actions\nMean over decisions (value, chosen #) and demo windows (RMSE) of candidate #1 '
                  'and candidate #N:\n\n'
                  + (table(['Policy', 'Set', 'N', 'value #1 → #N', 'RMSE to expert #1 → #N', 'mean chosen #'], chunk_rows)
                     if chunk_rows else '_Candidate rollouts still running._'))]
    for head, key, has_chosen, n_vid in chunk_sections:
        blocks += [md(f'### {head}'), grid(key + '_metrics', h=17)]
        if has_chosen:
            blocks.append(grid(key + '_chosen', h=6))
        if n_vid:
            blocks.append(grid([f'{key}_video_e{i}' for i in range(n_vid)], h=7))
    blocks += [md('## Full trajectories\nMean over episodes of attempt #1 and attempt #8:\n\n'
                  + (table(['Policy', 'Set', 'N', 'value #1 → #N', 'max coverage #1 → #N', 'RMSE to demo (px) #1 → #N',
                            'mean chosen #'], full_rows) if full_rows else '_Pending._'))]
    for head, k in full_sections:
        blocks += [md(f'### {head}'), grid(k + '_metrics', h=30), grid(k + '_chosen', h=6)]
    blocks += [md('# 4 · Data viz and diagnostics'),
               md('## Datasets\nT start position of every episode, by role, per held-out quadrant.'),
               grid('data/split_maps', h=9),
               md('## Verifier example\nThe same 64 ST candidates scored by each verifier; the runs above use Veritas v3 '
                  'lookahead (`wp_v3_la_tgn`).'),
               grid('data/verifier_example', h=14)]

    rep_file = out / 'report_url.txt'
    report = None
    if rep_file.exists():
        try:
            report = wr.Report.from_url(rep_file.read_text().strip())
        except Exception as e:   # noqa: BLE001 -- a deleted report just means "make a new one"
            print('could not load existing report:', e)
    if report is None:
        report = wr.Report(entity=ENTITY, project=PROJECT, title=TITLE, blocks=blocks)
    else:
        report.title = TITLE
        report.blocks = blocks
    report.description = 'Chunked quadrant 50-demo results, full trajectories, candidate metrics, data and diagnostics.'
    report.save()
    rep_file.write_text(report.url)
    print('report:', report.url.replace('https://wandb.ai/', 'https://forge.coreweave.com/wandb/'))


if __name__ == '__main__':
    main()
