"""Refresh the 2026-10-06 far-50 figures, log them to a wandb asset run, and (re)write the
"Seeds and held-out quadrants" section of the BC-vs-ST report.

Needs the `wandb_reports` env (wandb >= 0.18 + wandb-workspaces); the training env's wandb
0.13 report API is rejected by the server. Figures come from scripts/plot_far50_sweeps.py and
scripts/plot_split_maps.py (run them first in the training env). Images are logged to ONE
fixed run (report_assets_far50_20261006) so a rerun replaces them in place and the report's
media panels show the latest figures without editing the report again.

    /gscratch/robotics/harine/miniconda3/envs/wandb_reports/bin/python scripts/update_far50_report.py
    ... --dry-run     # print the blocks that would be written, save nothing
"""
import argparse, csv, collections, json, pathlib, statistics
import wandb
import wandb_workspaces.reports.v2 as wr

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / 'docs/reports/figures/far50_2026-10-06'
CSV = ROOT / 'docs/reports/summaries/far50_sweeps_2026-10-06.csv'
ENTITY, PROJECT = 'l2sml', 'pushT_diffusion_search'
ASSET_RUN = 'report_assets_far50_20261006'
REPORT_URL = ('https://wandb.ai/l2sml/pushT_diffusion_search/reports/'
              'Push-T-BC-vs-ST-ablations-and-minimum-gap--VmlldzoxODA1NTkyNg')
SECTION = 'Seeds and held-out quadrants (2026-10-06)'
SPLITS = [('blq50far', 'Bottom-left held out (blq50far)'),
          ('tlq50far', 'Top-left held out (tlq50far)'),
          ('brq50far', 'Bottom-right held out (brq50far)')]
ARMS = ['BC', 'ST k1', 'ST k16 uniform', 'ST k16 flat600', 'ST k16 ramp200to600']


def load_rows():
    rows = collections.defaultdict(dict)   # (split, arm, seed, sel, obs, step) -> {n: (succ, cov)}
    for r in csv.DictReader(open(CSV)):
        k = (r['split'], r['arm'], int(r['seed']), r['selection'], r['obs'], int(r['step']))
        rows[k][int(r['n'])] = (float(r['success']), float(r['coverage']))
    return rows


def table(rows, split):
    """Markdown table: mean over seeds of success at n=1/16/64 and coverage at n=64, argmax."""
    deploy = {}
    for seed in (42, 43, 44):
        c = [(v.get(1, (-1,))[0], k[5]) for k, v in rows.items()
             if k[:5] == (split, 'BC', seed, 'argmax', 'clean')]
        if c: deploy[seed] = max(c)[1]
    lines = ['| arm (argmax) | seeds | n=1 | n=16 | n=64 | coverage @64 |', '|---|---|---|---|---|---|']
    specs = [('BC (deployment ckpt: ' + ', '.join(f's{s} {st // 1000}k' for s, st in deploy.items()) + ')',
              'BC', 'clean', lambda s: deploy.get(s)),
             ('BC undertrained, 10k', 'BC', 'clean', lambda s: 10000)]
    for arm in ARMS[1:]:
        specs.append((f'{arm} 10k', arm, 'clean', lambda s: 10000))
        specs.append((f'{arm} 10k, corrupt obs', arm, 'corrupt', lambda s: 10000))
    for label, arm, obs, step_of in specs:
        per = [rows[(split, arm, s, 'argmax', obs, step_of(s))] for s in (42, 43, 44)
               if step_of(s) and (split, arm, s, 'argmax', obs, step_of(s)) in rows]
        if not per: continue
        def m(n, i):
            v = [p[n][i] for p in per if n in p]
            return f'{statistics.mean(v):.2f}' + (f' ({len(v)})' if len(v) != len(per) else '') if v else '--'
        lines.append(f'| {label} | {len(per)} | {m(1, 0)} | {m(16, 0)} | {m(64, 0)} | {m(64, 1)} |')
    return '\n'.join(lines)


def best_note(split):
    """Markdown: per ST arm and seed, the checkpoint holding the best cell in its sweep."""
    f = FIG / f'{split}_best_ckpts.json'
    if not f.exists():
        return '_best-checkpoint figure not generated yet_'
    best = json.load(open(f))
    lines = ['Best checkpoint anywhere in each sweep (selected on the test set; the figure above uses these '
             'for the ST arms, BC keeps its deployment checkpoint):', '',
             '| arm | seed | best ckpt | n | selection | obs | success |', '|---|---|---|---|---|---|---|']
    for arm in ARMS:
        for seed in (42, 43, 44):
            v = best.get(f'{arm}|{seed}')
            if v:
                lines.append(f'| {arm} | {seed} | {v["step"] // 1000}k | {v["n"]} | {v["sel"]} | {v["obs"]} | {v["success"]:.2f} |')
    k16 = [(k.split('|')[0], k.split('|')[1], v) for k, v in best.items() if 'k16' in k]
    if k16:
        top = max(k16, key=lambda t: t[2]['success'])
        lines += ['', f'**Best k=16 cell on this split:** {top[0]} seed {top[1]} at {top[2]["step"] // 1000}k, '
                  f'n={top[2]["n"]}, {top[2]["sel"]}, {top[2]["obs"]} obs: {top[2]["success"]:.2f}.']
    return '\n'.join(lines)


def media_grid(key):
    return wr.PanelGrid(
        runsets=[wr.Runset(entity=ENTITY, project=PROJECT, name='report assets',
                           filters=f'Name == "{ASSET_RUN}"')],
        panels=[wr.MediaBrowser(media_keys=[key], num_columns=1, layout=wr.Layout(x=0, y=0, w=24, h=14))])


def build_blocks(rows):
    b = [wr.H2(text=SECTION),
         wr.P(text=('Corrected protocol (fresh_context, terminal scoring, paired_decision_v3), 50 demos, '
                    '50 test episodes per cell (46 for top-left). Seeds 42/43/44 on every arm. The two new '
                    'quadrants use the same far-50 training rule as blq50far; their val set is 19 episodes '
                    'drawn from the leftover train pool because the held-out region is too small to spare one. '
                    'Plots: BC is the checkpoint a deployment would pick (highest n=1 success in its sweep, '
                    'ties -> latest step, dark line); its under-trained 10k checkpoint is shown in grey. '
                    'ST arms are at 10k. Lines are seed means, bands '
                    'min-max. Dashed = corrupted-observation eval. BC has no final-candidate rule, so its '
                    'argmax curve is repeated in the right-hand column as the reference. ST k=16 arms are '
                    'still training; their later checkpoints appear here as they are evaluated.')),
         wr.H3(text='Dataset splits'), media_grid('far50/split_maps')]
    for split, title in SPLITS:
        b += [wr.H3(text=title), wr.P(text='ST arms at 10k:'), media_grid(f'far50/{split}_sweep'),
              wr.MarkdownBlock(text=table(rows, split))]
        if (FIG / f'{split}_sweep_best.png').exists():
            b += [wr.P(text='ST arms at their best checkpoint in the sweep:'),
                  media_grid(f'far50/{split}_sweep_best'), wr.MarkdownBlock(text=best_note(split))]
    return b


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
    rows = load_rows()
    blocks = build_blocks(rows)
    if a.dry_run:
        for blk in blocks: print(type(blk).__name__, getattr(blk, 'text', '')[:2000] if hasattr(blk, 'text') else '')
        return
    run = wandb.init(entity=ENTITY, project=PROJECT, id=ASSET_RUN, name=ASSET_RUN, resume='allow',
                     job_type='report_assets', dir='/gscratch/robotics/harine/wandb')
    run.log({'far50/split_maps': wandb.Image(str(FIG / 'split_maps.png')),
             **{f'far50/{s}_sweep': wandb.Image(str(FIG / f'{s}_sweep.png')) for s, _ in SPLITS},
             **{f'far50/{s}_sweep_best': wandb.Image(str(FIG / f'{s}_sweep_best.png')) for s, _ in SPLITS
                if (FIG / f'{s}_sweep_best.png').exists()}})
    run.finish()
    report = wr.Report.from_url(REPORT_URL)
    old = report.blocks
    # Replace a previous copy of this section (from its H2 to the next H2), else append.
    start = next((i for i, blk in enumerate(old) if isinstance(blk, wr.H2) and blk.text == SECTION), None)
    if start is None:
        new = old + blocks
    else:
        end = next((i for i in range(start + 1, len(old)) if isinstance(old[i], wr.H2)), len(old))
        new = old[:start] + blocks + old[end:]
    report.blocks = new
    report.save()
    print('saved', report.url, '| blocks', len(old), '->', len(new))


if __name__ == '__main__':
    main()
