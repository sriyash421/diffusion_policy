"""Build / refresh the W&B report "L2S pushT full trajectory" from eval_dt_pusht outputs.

Figures (matplotlib, logged as images to l2sml/pusht_reports, embedded as media panels -- the
maze report's recipe) and a one-line legend; no tables. Re-running updates the same report
(its id is kept next to the figures).

    python scripts/wandb_fulltraj_report.py [--out DIR] [--dry-run]

Checkpoint per arm: chosen on the VAL split -- BC by argmax@1 success then coverage (maze's
rule for BC), everything else by argmax@8 success then argmax@8 coverage. The test split is
then read at that checkpoint. Arms whose evals are missing are skipped.
"""
import argparse
import csv
import glob
import json
import os
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = '/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/'
ENTITY, PROJECT, TITLE = 'l2sml', 'pusht_reports', 'L2S pushT full trajectory'
NS = [1, 2, 4, 8]
# label, run-dir glob, selection rule ('bc' -> argmax@1, else argmax@8)
ARMS = [
    ('BC U-Net open-loop 300 (i.i.d.)', 'pusht_image_search_imgonly/unet_bc/unetbc-full300_ver-t_goal_enc-resnet18_demos-50_split-blq50far_seed-42', 'bc'),
    ('Online search (GPT-2, obs feedback)', 'pusht_image_online/online_search/online_ntraj-8_enc-resnet18_demos-50_split-blq50far_seed-42', 'st'),
    ('DT k=1 · image', 'pusht_image_search_imgonly/dt_search/dt-gpt2_k1_fb-step_ver-wp_v3_la_tgn_enc-resnet18_demos-50_split-blq50far_seed-42', 'st'),
    ('DT k=1 · image+pos+fb', 'pusht_image_search_imgonly/dt_search/dt-gpt2-posfb_k1_fb-step_ver-wp_v3_la_tgn_enc-resnet18_demos-50_split-blq50far_seed-42', 'st'),
    ('DT k=8 · verifier end-feedback', 'pusht_image_search_imgonly/dt_search/dt-gpt2_k8_fb-step_ver-wp_v3_la_tgn_enc-resnet18_demos-50_split-blq50far_seed-42', 'st'),
    ('DT k=8 · coverage end-feedback', 'pusht_image_search_imgonly/dt_search/dt-gpt2_k8_fb-step_tf-cov_ver-wp_v3_la_tgn_enc-resnet18_demos-50_split-blq50far_seed-42', 'st'),
]
# colour + marker per arm (Okabe-Ito), never hue alone: linestyle encodes the rule
STYLE = [('#009E73', 'D'), ('#000000', 's'), ('#0072B2', 'o'), ('#56B4E9', 'v'), ('#D55E00', '^'), ('#CC79A7', 'P')]


def load_evals(run_dir):
    """{step: {(split, n, rule): row}} for every evaluated checkpoint of a run."""
    out = {}
    for d in sorted(glob.glob(os.path.join(ROOT, run_dir, 'bon_search', 'dt_eval', '*'))):
        f = os.path.join(d, 'metrics.csv')
        if not os.path.exists(f):
            continue
        name = os.path.basename(d)
        step = int(name[5:]) if name.startswith('step_') else -1
        rows = list(csv.DictReader(open(f)))
        out[step] = {(r['split'], int(r['n']), r['rule']): r for r in rows}
    return out


def pick(evals, rule):
    """Checkpoint step chosen on val; falls back to the last one if val is missing."""
    def key(step):
        m = evals[step]
        k = ('val', 1, 'argmax') if rule == 'bc' else ('val', 8, 'argmax')
        if k not in m:
            return (-1, -1, step)
        return (float(m[k]['success_rate']), float(m[k]['mean_max_coverage']), step)
    return max(evals, key=key)


def sweep_figure(arms, styles, split, metric, rules, ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, evals, step), (c, mk) in zip(arms, styles):
        m = evals[step]
        for rule, ls, fill in rules:
            ys = [float(m[(split, n, rule)][metric]) if (split, n, rule) in m else np.nan for n in NS]
            ax.plot(NS, ys, color=c, marker=mk, linestyle=ls, markerfacecolor=c if fill else 'white',
                    markeredgecolor=c, label=f'{label} · {rule} · ckpt {step if step >= 0 else "latest"}' if rule == rules[0][0] else None)
    ax.set_xscale('log', base=2); ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlabel('full-trajectory candidates (sequential attempts)'); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(alpha=0.3)
    if metric == 'success_rate':
        ax.set_ylim(-0.02, 1.0)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    fig.text(0.01, 0.01, 'solid/filled: ' + rules[0][0] + ' · dashed/hollow: ' + rules[1][0]
             + ' · test = 50 held-out blq50far episodes, val = 19', fontsize=8)
    fig.tight_layout()
    return fig


def checkpoint_figure(label, evals, color):
    steps = sorted(s for s in evals if s >= 0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, ylabel in zip(axes, ('success_rate', 'mean_max_coverage'), ('success rate', 'mean max coverage')):
        for n, mk, ls in ((1, 'o', ':'), (8, 's', '-')):
            for rule, alpha in (('argmax', 1.0), ('argmax_cov', 0.5)):
                ys = [float(evals[s][('test', n, rule)][metric]) if ('test', n, rule) in evals[s] else np.nan for s in steps]
                ax.plot(steps, ys, color=color, marker=mk, linestyle=ls, alpha=alpha,
                        label=f'{rule}@{n}')
        ax.set_xlabel('training step'); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    axes[0].set_ylim(-0.02, 1.0); axes[0].legend(fontsize=8)
    fig.suptitle(f'{label} · test split, per checkpoint'); fig.tight_layout()
    return fig


def attempt_order_figure(arms, styles):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (label, run_dir, step), (c, mk) in zip(arms, styles):
        d = os.path.join(ROOT, run_dir, 'bon_search', 'dt_eval', f'step_{step:07d}' if step >= 0 else 'latest')
        f = os.path.join(d, 'test_candidates.npz')
        if not os.path.exists(f):
            continue
        cnd = np.load(f, allow_pickle=True)
        ks = np.arange(1, cnd['value'].shape[1] + 1)
        axes[0].plot(ks, cnd['value'].mean(0), color=c, marker=mk, label=label)
        axes[1].plot(ks, cnd['max_coverage'].mean(0), color=c, marker=mk, label=label)
    axes[0].set_ylabel('verifier value at the final state'); axes[1].set_ylabel('max coverage')
    for ax in axes:
        ax.set_xlabel('attempt index (generation order)'); ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle('Does a later attempt do better? mean over the 50 test episodes'); fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/_fulltraj_report')
    ap.add_argument('--dry-run', action='store_true', help='figures only, no W&B')
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    arms, styles = [], []
    for (label, run_dir, rule), style in zip(ARMS, STYLE):
        evals = load_evals(run_dir)
        if not evals:
            print(f'skip {label}: no evals under {run_dir}')
            continue
        step = pick(evals, rule)
        arms.append((label, run_dir, evals, step)); styles.append(style)
        print(f'{label}: {len(evals)} checkpoints, chosen {step}')
    sweep_arms = [(l, e, s) for l, _, e, s in arms]
    figures = {}
    for split in ('test', 'val'):
        figures[f'sweeps/{split}_success'] = sweep_figure(
            sweep_arms, styles, split, 'success_rate', [('argmax', '-', True), ('last', '--', False)],
            'success rate', f'{split}: argmax (verifier value) vs final candidate')
        figures[f'sweeps/{split}_coverage'] = sweep_figure(
            sweep_arms, styles, split, 'mean_max_coverage', [('argmax', '-', True), ('last', '--', False)],
            'mean max coverage', f'{split}: argmax (verifier value) vs final candidate')
        figures[f'sweeps/{split}_success_cov'] = sweep_figure(
            sweep_arms, styles, split, 'success_rate', [('argmax_cov', '-', True), ('any', '--', False)],
            'success rate', f'{split}: argmax by observed coverage vs any (perf@n)')
    for (label, run_dir, evals, step), (c, _) in zip(arms, styles):
        if len([s for s in evals if s >= 0]) > 1:
            figures['checkpoints/' + label.split(' ·')[0].replace(' ', '_') + ('_' + label.split('· ')[1].replace(' ', '_') if '· ' in label else '')] = checkpoint_figure(label, evals, c)
    figures['diagnostics/attempt_order'] = attempt_order_figure([(l, r, s) for l, r, _, s in arms], styles)
    paths = {}
    for key, fig in figures.items():
        p = out / (key.replace('/', '__') + '.png')
        fig.savefig(p, dpi=130); plt.close(fig); paths[key] = p
    print('figures:', len(paths), '->', out)
    if args.dry_run:
        return

    import wandb
    import wandb_workspaces.reports.v2 as wr
    rid_file = out / 'run_id.txt'
    rid = rid_file.read_text().strip() if rid_file.exists() else wandb.util.generate_id()
    rid_file.write_text(rid)
    run = wandb.init(entity=ENTITY, project=PROJECT, id=rid, resume='allow', dir=str(out),
                     name='fulltraj_figures', settings=wandb.Settings(start_method='fork'))
    run.log({k: wandb.Image(str(p)) for k, p in paths.items()})
    run.finish()

    def grid(keys, h=14):
        keys = [keys] if isinstance(keys, str) else keys
        rs = wr.Runset(entity=ENTITY, project=PROJECT, name='figures')
        w = 24 // len(keys)
        panels = []
        for i, key in enumerate(keys):
            p = wr.MediaBrowser(media_keys=[key], num_columns=1)
            p.layout = wr.Layout(x=i * w, y=0, w=w, h=h)
            panels.append(p)
        return wr.PanelGrid(runsets=[rs], panels=panels)

    md = wr.MarkdownBlock
    blocks = [
        md('blq50far split (50 train demos; 50 test / 19 val held-out episodes). Every arm: 8 sequential '
           'full-trajectory attempts per episode in the live env, attempts end on coverage > 0.95 or 300 steps; '
           'per-step feedback = −(T-to-goal + veritas pin cost) at the realized state. '
           'Solid: argmax over the first N attempts · dashed: the N-th attempt (final). '
           'BC U-Net predicts the whole 300-step trajectory from the first frame and executes it open-loop, attempts i.i.d.; online search and DT k=8 see the earlier attempts in context.'),
        wr.H1(text='Sweeps · test'), grid(['sweeps/test_success', 'sweeps/test_coverage']), grid('sweeps/test_success_cov', h=12),
        wr.H1(text='Sweeps · val'), grid(['sweeps/val_success', 'sweeps/val_coverage']), grid('sweeps/val_success_cov', h=12),
        wr.H1(text='Per checkpoint (test, N=1 and N=8)'),
    ]
    ck = [k for k in paths if k.startswith('checkpoints/')]
    for i in range(0, len(ck), 2):
        blocks.append(grid(ck[i:i + 2], h=10))
    blocks += [wr.H1(text='Attempt order'), grid('diagnostics/attempt_order', h=10)]

    api = wandb.Api()
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
    report.save()
    rep_file.write_text(report.url)
    print('report:', report.url)


if __name__ == '__main__':
    main()
