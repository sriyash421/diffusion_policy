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
import datetime
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
    ('ST-diffusion k=16 (4/4/256, 17.1M)', 'search transformer, diffusion head', 30,
     BASE / 'outer_inner' / 'value_k16_corrupt-False_demos-30_seed-42'),
    ('ST-diffusion k=16 (4/4/256, 17.1M)', 'search transformer, diffusion head', 100,
     BASE / 'outer_inner' / 'value_k16_corrupt-False_demos-100_seed-42'),
    ('ST-gaussian k=16 (4/4/256, 14.6M)', 'search transformer, Gaussian head', 30,
     BASE / 'offline' / 'gaussian_k16_corrupt-False_demos-30_seed-42'),
    ('ST-gaussian k=16 (4/4/256, 14.6M)', 'search transformer, Gaussian head', 100,
     BASE / 'offline' / 'gaussian_k16_corrupt-False_demos-100_seed-42'),
    ('ST-diffusion k=1 (4/4/256, 17.1M)', 'same class as k=16, width 1 (empty search context)', 30,
     BASE / 'offline' / 'value_k1_demos-30_seed-42'),
    ('ST-diffusion k=1 (4/4/256, 17.1M)', 'same class as k=16, width 1 (empty search context)', 100,
     BASE / 'offline' / 'value_k1_demos-100_seed-42'),
    # 30-demo additions. A different ARCHITECTURE (UNet) and a ~24x wider transformer
    # trunk at both search widths, all on the same manifest/seed/protocol as the six
    # above, so they drop straight into the same tables.
    ('UNet BC (293.4M)', 'diffusion UNet, i.i.d. best-of-n (no search context)', 30,
     BASE / 'unet_bc' / 'unetbc_demos-30_seed-42'),
    ('ST-diffusion k=1 (6/8/1024, 137.8M)', 'the wide trunk at width 1', 30,
     BASE / 'offline' / 'value_k1_arch-6x8x1024_corrupt-False_demos-30_seed-42'),
    ('ST-diffusion k=16 (6/8/1024, 137.8M)', 'the wide trunk at width 16', 30,
     BASE / 'outer_inner' / 'value_k16_arch-6x8x1024_corrupt-False_demos-30_seed-42'),
]

# Arms measured under a verifier that ALSO scores the arm's distance to the T. Separate
# list, and rendered in its own section, because these numbers are NOT comparable to the
# t_goal tables above -- same episodes and same protocol, but a different ranking rule.
#
# (label, family, demos, run dir, verifier, eval subdir)
ARM_DISTANCE_ARMS = [
    ('ST-diffusion k=1 (4/4/256) — armTn', 'trained AND selected under armTn', 30,
     BASE / 'offline' / 'value_k1_ver-armTn_corrupt-False_demos-30_seed-42',
     'armTn', 'bon_search'),
    ('ST-diffusion k=16 (4/4/256) — armTn', 'trained AND selected under armTn', 30,
     BASE / 'outer_inner' / 'value_k16_ver-armTn_corrupt-False_demos-30_seed-42',
     'armTn', 'bon_search'),
    # SAME WEIGHTS as the t_goal UNet BC arm above -- only the ranking rule differs, since
    # this policy has no search context and the verifier touches nothing but selection.
    # That makes it the cleanest read on what the raw arm term does to best-of-n.
    #
    # These two and the t_goal `UNet BC (293.4M)` arm are ONE set of frozen weights scored
    # three ways, so they isolate the ranking rule with nothing else moving -- unlike the ST
    # armTn arms above, which were also TRAINED under their verifier and therefore conflate
    # the two. Their n=1 columns must be identical to each other (no selection happens at
    # n=1); if they are not, something other than the ranking rule changed.
    ('UNet BC (293.4M) — armT, re-ranked', 'the t_goal UNet BC weights, re-scored under raw armT', 30,
     BASE / 'unet_bc' / 'unetbc_demos-30_seed-42',
     'armT', 'bon_search_ver-armT'),
    ('UNet BC (293.4M) — armTn, re-ranked', 'the t_goal UNet BC weights, re-scored under armTn', 30,
     BASE / 'unet_bc' / 'unetbc_demos-30_seed-42',
     'armTn', 'bon_search_ver-armTn'),
    ('UNet BC (293.4M) — armTd, re-ranked',
     'the same weights again, re-scored under armTd (spreads measured per control step)', 30,
     BASE / 'unet_bc' / 'unetbc_demos-30_seed-42',
     'armTd', 'bon_search_ver-armTd'),
]


# This doc reports ONE verifier: t_goal, `-(mean per-keypoint distance of the T from the
# goal T)`. Every arm below was trained and evaluated under it, so every number here is
# comparable to every other.
#
# The filter is not a formality. Since 2026-08-19 the verifier is selectable
# (pusht_verifier.VALUE_FNS) and the alternatives rank candidates differently enough to move
# success rates by tens of points -- raw `armT` took the UNet BC arm from 0.70 to 0.00 at
# n=64. Runs from the new era write to the SAME `bon_search/` filename, so without this
# filter, adding one of their directories to ARMS would silently average two verifiers into
# one row. Curves written before the cutover carry no `verifier_value` key at all; that
# absence means t_goal.
VERIFIER = 't_goal'


# The arms named by success_rates_stk1_skk16_bc_atmTn_Tgoal.md: the three 30-demo policy
# families (ST k=1, ST k=16, UNet BC) under the two LIVE verifiers (t_goal, armTn).
# Selected by (run-dir basename, eval subdir) because that pair is what actually identifies
# a measurement -- one run dir holds several verifiers' sweeps side by side.
#
# Everything else the builder knows about (the 100-demo halves, ST-gaussian, the 6/8/1024
# trunk, and the retired armT / eval-only armTd re-ranks) is still in ARMS above and comes
# back with --all. This is a VIEW, not a deletion.
FOCUS = {
    ('value_k1_demos-30_seed-42', 'bon_search'),
    ('value_k16_corrupt-False_demos-30_seed-42', 'bon_search'),
    ('unetbc_demos-30_seed-42', 'bon_search'),
    ('value_k1_ver-armTn_corrupt-False_demos-30_seed-42', 'bon_search'),
    ('value_k16_ver-armTn_corrupt-False_demos-30_seed-42', 'bon_search'),
    ('unetbc_demos-30_seed-42', 'bon_search_ver-armTn'),
}


def read_rows(run, sub='bon_search', verifier=VERIFIER):
    """Curve rows for one arm under one verifier.

    The DIRECTORY is the authority on which verifier produced a row, not the row's own
    `verifier_value` field: that field was only added on 2026-08-19 and, because
    `_row_from_curve` is an explicit whitelist, did not actually reach the jsonl until it
    was fixed -- so every row written before then lacks it regardless of which verifier ran.
    Rows that DO carry it are cross-checked against the declaration and dropped on a
    mismatch, which catches a mislabeled ARMS entry.
    """
    p = run / sub / 'success_curves.jsonl'
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue      # a row still being written; the next run picks it up
            if r.get('verifier_value') in (None, verifier):
                out.append(r)
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
    # ONE canonical file, rewritten in place. The date-stamped variant this briefly used
    # spawned a new near-identical 30KB doc per regeneration (three in four days, two of
    # them byte-identical) and broke every inbound reference each time. The generation date
    # is recorded in the doc body instead, which is the part anyone actually needs.
    today = datetime.date.today().isoformat()
    ap.add_argument('-o', '--out', default='success_rates_stk1_skk16_bc_atmTn_Tgoal.md')
    ap.add_argument('--all', action='store_true',
                    help='render every known arm instead of the FOCUS set '
                         '(adds the 100-demo halves, ST-gaussian, the 6/8/1024 trunk, and '
                         'the retired armT / eval-only armTd re-ranks)')
    args = ap.parse_args()

    L = []
    L.append('# PushT 30/100-demo sweep — success rates\n')
    L.append('Three policy families x two demo budgets, plus three 30-demo additions (a diffusion UNet baseline, and a ~24x wider search transformer at widths 1 and 16). Every arm trains to 100k gradient '
             'steps with a checkpoint every 10k; every checkpoint is swept over '
             '`n = 1, 2, 4, 8, 16, 32, 64` on the same 50 held-out test episodes.\n')
    L.append(f'_Generated {today} from the on-disk eval output._\n')
    L.append('Regenerate with `python scripts/build_30_100_success_doc.py` — it rewrites '
             'this file in place. Source of truth is each run\'s '
             '`bon_search*/success_curves.jsonl`.\n')
    L.append('**Verifier: `t_goal`** — the candidate value is `-(mean per-keypoint distance '
             'of the T from the goal T)`. Every arm here was trained and evaluated under it, '
             'so all numbers are mutually comparable. Rows measured under any other verifier '
             'are filtered out, not averaged in: the verifier became selectable on '
             '2026-08-19 (`pusht_verifier.VALUE_FNS`) and the choice moves success rates by '
             'tens of points, so the two eras must never share a table.\n')

    L.append('Arm labels carry `(n_layer/n_head/n_emb, total params)` — the transformer '
             'shape and the whole policy\'s parameter count including the shared 11.2M '
             'ResNet-18 encoder, counted from each run\'s checkpoint. The trunk alone is '
             '5.9M at 4/4/256, 126.6M at 6/8/1024, and 282.2M for the UNet.\n')
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
    L.append('**The two width-1 baselines are different things, and neither is "BC" '
             'alone.**\n')
    L.append('`ST k=1` is the *same transformer* as `ST-diffusion k16` '
             '(`PushTDiffusionSearchPolicy`) trained at `max_actions: 1`. Its search '
             'context is always empty, so it isolates the *learned search context*: it '
             'shares architecture, encoder, scheduler, optimizer and data with k16, and '
             'differs only in whether candidates condition on each other during training. '
             '`ST-big k=1` is the same thing at a ~24x wider trunk.\n')
    L.append('`UNet BC` is a *different architecture* — `PushTUNetSearchPolicy`, a '
             'convolutional diffusion UNet with no transformer and no search context at '
             'all. It isolates the *backbone*. It is matched to the ST arms on everything '
             'outside the backbone: same 30-demo manifest, seed 42, 100k steps, DDIM at 8 '
             'inference steps, ResNet-18/ImageNet encoder with the same [76,76] random '
             'crop, batch 32, lr 1e-4, EMA 0.995.\n')
    L.append('At n>1 **both** are plain best-of-n over i.i.d. samples scored by the same '
             'verifier, so all arms are compared at a matched test-time budget and any gap '
             'is attributable to the trained policy rather than to drawing more samples.\n')

    L.append('## Data\n')
    L.append('The 30-demo train set is the first 30 episodes of the 100-demo train list in '
             'its own order; val (30) and test (50) are copied verbatim between them, so '
             '30-vs-100 isolates training-set size alone. Manifests: '
             '`config/splits/pusht_seed42_train{30,100}.json`.\n')

    def emit_arm(label, family, demos, run, sub='bon_search', verifier=VERIFIER):
        rows = read_rows(run, sub, verifier)
        ck, done, partial, facts = provenance(rows, run)
        L.append(f'## {label} — {demos} demos\n')
        L.append(f'`{run.parent.name}/{run.name}/{sub}` — {family}\n')
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

    for label, family, demos, run in ARMS:
        if args.all or (run.name, 'bon_search') in FOCUS:
            emit_arm(label, family, demos, run)

    # ---- second verifier -----------------------------------------------------
    L.append('# Arm-distance verifiers (`armT`, `armTn`, `armTd`)\n')
    L.append('Everything above ranks candidates by `t_goal`. The arms below add the '
             'arm-to-T-centre distance to that value. **Do not read these tables against '
             'the ones above cell-by-cell as if they were the same experiment** — the '
             'episodes, protocol and seed match, but the ranking rule does not, and that '
             'is the whole variable.\n')
    L.append('| verifier | value | why |')
    L.append('|---|---|---|')
    L.append('| `t_goal` | `-(T-to-goal)` | the original. Before the arm touches the block '
             'no candidate can move the T, so this value is *identical* across candidates '
             'and argmax is a coin flip on every approach step. |')
    L.append('| `armT` | `-(T-to-goal + arm-to-T)` | raw sum. Superseded: the two terms '
             'have comparable marginal scale (80 vs 91 px) but the spread argmax actually '
             'sees — across candidates *within* one control step — is 13.6 vs 52.1 px, so '
             'the arm term outvotes task progress ~4:1. |')
    L.append('| `armTn` | `-(T-to-goal/13.6 + arm-to-T/52.1)` | each term divided by its '
             'own within-step spread, so a 1-sigma gain in either counts the same. |\n')
    L.append('| `armTd` | `-(z(T-to-goal) + z(arm-to-T))` | the two constants above replaced by the spread measured across the n candidates *of that control step*, so the weighting tracks what the choice can actually change now. Eval-only: it is defined over a candidate SET, so it has no per-candidate scalar to train against, and its score is standardized rather than pixels — not `<= 0`, and comparable only WITHIN a step. |')
    for label, family, demos, run, verifier, sub in ARM_DISTANCE_ARMS:
        if args.all or (run.name, sub) in FOCUS:
            emit_arm(label, family, demos, run, sub, verifier)

    # ---- head-to-head --------------------------------------------------------
    L.append('## Head-to-head: `armTn` vs `t_goal` at k=16\n')
    L.append('The controlled comparison — same 4/4/256 trunk, same 30 demos, same seed 42, '
             'same 50 test episodes, same argmax rule. Only the verifier differs.\n')
    a = by_step(read_rows(BASE / 'outer_inner' /
                          'value_k16_ver-armTn_corrupt-False_demos-30_seed-42',
                          'bon_search', 'armTn'), 'success_rate')
    b = by_step(read_rows(BASE / 'outer_inner' /
                          'value_k16_corrupt-False_demos-30_seed-42'), 'success_rate')
    L.append('| step | ' + ' | '.join(f'n={n}' for n in NS) + ' | mean |')
    L.append('|---:|' + '---:|' * (len(NS) + 1))
    alld = []
    for step in STEPS:
        if step not in a or step not in b:
            continue
        cells, diffs = [], []
        for n in NS:
            if n in a[step] and n in b[step]:
                d = a[step][n] - b[step][n]
                diffs.append(d); cells.append(f'{d:+.2f}')
            else:
                cells.append('–')
        alld += diffs
        m = f'**{sum(diffs)/len(diffs):+.3f}**' if diffs else '–'
        L.append(f'| {step:,} | ' + ' | '.join(cells) + f' | {m} |')
    if alld:
        L.append('')
        L.append(f'_Cells are `armTn − t_goal` success rate. Across all {len(alld)} shared '
                 f'points: mean **{sum(alld)/len(alld):+.3f}**, armTn ahead at '
                 f'{sum(1 for x in alld if x > 0)}, behind at '
                 f'{sum(1 for x in alld if x < 0)}. At 50 episodes a single cell carries a '
                 f'95% CI of about +/-0.13, so read the mean column, not one cell._\n')

    L.append('## Coverage\n')
    L.append('| arm | demos | checkpoints | fully swept | pending |')
    L.append('|---|---:|---:|---:|---|')
    for label, _f, demos, run, sub, ver in (
            [(a, b, c, d, 'bon_search', VERIFIER) for a, b, c, d in ARMS]
            + [(a, b, c, d, sub, v) for a, b, c, d, v, sub in ARM_DISTANCE_ARMS]):
        if not (args.all or (run.name, sub) in FOCUS):
            continue
        ck, done, partial, _ = provenance(read_rows(run, sub, ver), run)
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
