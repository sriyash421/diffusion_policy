"""Regenerate SUCCESS_RATES_30_100.md from the on-disk eval output of the 30/100-demo 3x2.

The sweep is three policy families x two demo budgets, every arm trained to 100k gradient
steps with a checkpoint every 10k, and every checkpoint swept over n = 1..64 on the SAME 50
test episodes. This reads bon_search/success_curves.jsonl and nothing else, so it is safe to
re-run while the sweep is still going: arms that have not reached a checkpoint simply have
fewer rows.

    python scripts/build_30_100_success_doc.py [-o SUCCESS_RATES_30_100.md]

Deliberately does NOT nominate a best checkpoint or a best n. Every evaluated cell is
printed and the reader picks; a doc that highlighted a winner would be doing selection on
test, which is the thing this experiment must not do.
"""
import argparse
import json
import os
import pathlib
import re

ROOT = pathlib.Path(os.environ.get(
    'DP_OUTPUT_ROOT', '/gscratch/robotics/harine/diffusion_policy_outputs'))
BASE = ROOT / 'pusht_search' / 'pusht_image_search'
NS = [1, 2, 4, 8, 16, 32, 64]
STEPS = [10000 * k for k in range(1, 11)]

# (label, family, demos, run dir). The trainer is part of the path: the diffusion arm
# amortizes its search context over an outer/inner loop, the other two do not.
ARMS = [
    ('ST-diffusion', 'search transformer, diffusion head', 30,
     BASE / 'outer_inner' / 'value_k16_corrupt-False_demos-30_seed-42'),
    ('ST-diffusion', 'search transformer, diffusion head', 100,
     BASE / 'outer_inner' / 'value_k16_corrupt-False_demos-100_seed-42'),
    ('ST-gaussian', 'search transformer, Gaussian head', 30,
     BASE / 'offline' / 'gaussian_k16_corrupt-False_demos-30_seed-42'),
    ('ST-gaussian', 'search transformer, Gaussian head', 100,
     BASE / 'offline' / 'gaussian_k16_corrupt-False_demos-100_seed-42'),
    ('BC', 'same policy class as ST-diffusion, max_actions=1', 30,
     BASE / 'offline' / 'bc_demos-30_seed-42'),
    ('BC', 'same policy class as ST-diffusion, max_actions=1', 100,
     BASE / 'offline' / 'bc_demos-100_seed-42'),
    # 30-demo additions. A different ARCHITECTURE (UNet) and a ~24x wider transformer
    # trunk at both search widths, all on the same manifest/seed/protocol as the six
    # above, so they drop straight into the same tables.
    ('UNet BC', 'diffusion UNet, i.i.d. best-of-n (no search context)', 30,
     BASE / 'unet_bc' / 'unetbc_demos-30_seed-42'),
    ('ST-big k=1', 'search transformer 6/8/1024 (~75M trunk), width 1', 30,
     BASE / 'offline' / 'value_k1_arch-6x8x1024_corrupt-False_demos-30_seed-42'),
    ('ST-big k=16', 'search transformer 6/8/1024 (~75M trunk), width 16', 30,
     BASE / 'outer_inner' / 'value_k16_arch-6x8x1024_corrupt-False_demos-30_seed-42'),
]


def read_rows(run):
    p = run / 'bon_search' / 'success_curves.jsonl'
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass          # a row still being written; the next run picks it up
    return out


def by_step(rows, field):
    """{step: {n: value}} for one field, merging the per-n rows of a checkpoint."""
    agg = {}
    for r in rows:
        m = re.search(r'step_(\d+)', r.get('checkpoint', ''))
        step = int(m.group(1)) if m else r.get('step')
        if step is None:
            continue
        vals = r.get(field) or []
        agg.setdefault(int(step), {}).update(dict(zip(r.get('n') or [], vals)))
    return agg


def table(agg, fmt='{:.2f}'):
    head = '| step | ' + ' | '.join(f'n={n}' for n in NS) + ' |'
    rule = '|---:|' + '---:|' * len(NS)
    lines = [head, rule]
    for step in STEPS:
        if step not in agg:
            continue
        cells = [fmt.format(agg[step][n]) if n in agg[step] else '–' for n in NS]
        lines.append(f'| {step:,} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) if len(lines) > 2 else '_no checkpoints evaluated yet_'


def provenance(rows, run):
    ck = sorted(int(re.search(r'step_(\d+)', p.name).group(1))
                for p in (run / 'checkpoints').glob('step_*.ckpt')) \
        if (run / 'checkpoints').is_dir() else []
    agg = by_step(rows, 'success_rate')
    done = sorted(s for s, v in agg.items() if set(NS) <= set(v))
    partial = sorted(s for s, v in agg.items() if not set(NS) <= set(v))
    facts = {k: {r.get(k) for r in rows} for k in
             ('selection', 'selection_temperature', 'n_episodes', 'seed')}
    return ck, done, partial, facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='SUCCESS_RATES_30_100.md')
    args = ap.parse_args()

    L = []
    L.append('# PushT 30/100-demo sweep — success rates\n')
    L.append('Three policy families x two demo budgets, plus three 30-demo additions (a diffusion UNet baseline, and a ~24x wider search transformer at widths 1 and 16). Every arm trains to 100k gradient '
             'steps with a checkpoint every 10k; every checkpoint is swept over '
             '`n = 1, 2, 4, 8, 16, 32, 64` on the same 50 held-out test episodes.\n')
    L.append('Regenerate with `python scripts/build_30_100_success_doc.py`. '
             'Source of truth is each run\'s `bon_search/success_curves.jsonl`.\n')

    L.append('## What the numbers mean\n')
    L.append('`n` is the **search width**: n action candidates are generated for the '
             'current observation, each is rolled out in a PushT simulator (the '
             '"verifier") to a scalar value, and one is executed. So n is a *test-time '
             'compute* axis — the same weights are read out n different ways.\n')
    L.append('Success rate is the fraction of the 50 test episodes reaching the goal '
             'coverage threshold. At 50 episodes a single cell carries a 95% CI of '
             'roughly +/-0.13 near 0.5, so **cell-to-cell differences under ~0.15 are not '
             'separable**; read down a column or across several checkpoints, not one cell.\n')

    L.append('## Choice mechanism\n')
    L.append('How the executed action is picked out of the n scored candidates, and how '
             'each candidate is produced. Identical across all six arms, so the arms '
             'differ only in policy family and demo budget.\n')
    L.append('| property | value | where it comes from |')
    L.append('|---|---|---|')
    L.append('| selection rule | **`argmax`** over the verifier value | recorded in every '
             'curve row as `selection` |')
    L.append('| selection temperature | n/a (argmax is not sampled) | `selection_temperature` '
             'null in every row |')
    L.append('| ranking signal | scalar verifier value = simulated rollout reward | '
             '`search_context: value` |')
    L.append('| candidates per decision | n (the sweep axis), i.i.d. given the obs | '
             '`n_generations` equals n in every row |')
    L.append('| eval episodes | 50 test episodes, `--skip-val` | `n_episodes` = 50 |')
    L.append('| eval seed | 42 (= `training.seed`) | `seed` in every row |')
    L.append('')
    L.append('**Sampler.** The diffusion arms use `DDIMScheduler`, 100 train timesteps, '
             '**8 inference steps**, `prediction_type: epsilon`, and no '
             '`scheduler_step_kwargs` — so `DDIMScheduler.step` runs at its default '
             '`eta = 0.0`, the deterministic DDIM ODE. **No noise is injected during '
             'denoising.** The initial latent *is* a fresh `randn` per candidate, which is '
             'exactly what makes the n candidates differ and what best-of-n exploits. '
             'The ST-gaussian arm instead draws one `rsample` from a Normal head per '
             'candidate. Evals passed no `--noise-scheduler` / `--num-inference-steps` '
             'override, so every number below used the trained configuration.\n')
    L.append('**Arms.** BC is *not* a UNet: it is the same `PushTDiffusionSearchPolicy` as '
             'ST-diffusion with `max_actions: 1`, so at n>1 it is best-of-n over i.i.d. '
             'samples with an empty search context. That is the point — it gives BC the '
             'same test-time budget, so any gap is attributable to the learned search '
             'context rather than to drawing more samples. (A separate UNet BC config, '
             '`train_pusht_unet_bc`, exists but is not part of this sweep.)\n')

    L.append('## Data\n')
    L.append('The 30-demo train set is the first 30 episodes of the 100-demo train list in '
             'its own order; val (30) and test (50) are copied verbatim between them, so '
             '30-vs-100 isolates training-set size alone. Manifests: '
             '`config/splits/pusht_seed42_train{30,100}.json`.\n')

    for label, family, demos, run in ARMS:
        rows = read_rows(run)
        ck, done, partial, facts = provenance(rows, run)
        L.append(f'## {label} — {demos} demos\n')
        L.append(f'`{run.parent.name}/{run.name}` — {family}\n')
        state = (f'{len(ck)}/10 checkpoints written, {len(done)} fully swept'
                 + (f', {len(partial)} partial ({partial})' if partial else ''))
        L.append(f'_{state}._\n')
        L.append('### Test success rate\n')
        L.append(table(by_step(rows, 'success_rate')))
        L.append('')
        mr = by_step(rows, 'mean_reward')
        if any(mr.values()):
            L.append('### Test mean reward (max coverage reached)\n')
            L.append(table(mr, '{:.3f}'))
            L.append('')
        odd = {k: v for k, v in facts.items()
               if k in ('selection', 'n_episodes', 'seed')
               and v - {'argmax', 50, 42, None}}
        if odd:
            L.append(f'> **Note:** this arm deviates from the common protocol: {odd}\n')

    L.append('## Coverage\n')
    L.append('| arm | demos | checkpoints | fully swept | pending |')
    L.append('|---|---:|---:|---:|---|')
    for label, _f, demos, run in ARMS:
        ck, done, partial, _ = provenance(read_rows(run), run)
        pend = sorted(set(ck) - set(done))
        L.append(f'| {label} | {demos} | {len(ck)}/10 | {len(done)} | '
                 f'{pend if pend else "—"} |')
    L.append('')
    L.append('_No checkpoint or n is nominated as best: every evaluated cell is printed '
             'and selection is never done on the test split._\n')

    pathlib.Path(args.out).write_text('\n'.join(L))
    print(f'wrote {args.out} ({len(L)} blocks)')


if __name__ == '__main__':
    main()
