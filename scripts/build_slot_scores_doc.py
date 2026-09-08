"""Regenerate slot_verifier_scores_noised_obs.md from candidate_scores.jsonl.

THE QUESTION. The per-slot observation ladder assumes slot k improves with k: slot 0 sees
the most-corrupted observation and no search context, slot K-1 sees the cleanest and the
full context. Success rates cannot test that -- they report what the argmax executed, not
what each slot produced. The per-candidate verifier scores can, because candidate i
conditions on exactly i scored context entries, so **i IS the slot index** the training loss
decodes (search_procedure.py, `for i in range(n_actions)`).

Values are `t_goal` = -(mean per-keypoint distance of the T from the goal), in pixels, so
HIGHER IS BETTER and the numbers are negative.

THE CONTROL IS THE POINT. A slot gradient in a ladder arm means nothing on its own: the
later slots also carry more search context, and that alone could explain a trend. The
uniform arm has no ladder but the identical context structure, so it isolates the ladder.
Read every ladder row against it.

TIES ARE NOT INCIDENTAL. A candidate that never touches the block returns the identical sim
value, so a third of decisions have all K scores exactly equal. Two consequences, both of
which produced wrong readings on the first pass and are corrected here:

  * `argmax` breaks ties toward the lowest index, which manufactures a slot-0 spike. Even
    with RANDOM tie-breaking the control still shows slot 0 far above the 1/K baseline while
    its means are flat -- an ordering bias this doc has not isolated -- so the argmax
    histogram is reported as descriptive colour and nothing is concluded from it.
  * the median paired difference is 0 by construction. The paired win-rate is therefore
    computed over decisions where the two slots actually DIFFER.

The mean profile and the tie-excluded win-rate are the two numbers to read.

    python scripts/build_slot_scores_doc.py [-o slot_verifier_scores_noised_obs.md]

Produce the inputs with a single-checkpoint eval, one per arm:

    sbatch scripts/slurm/eval_ckpt_pusht_search.sbatch <ckpt> \
        --n-list 16 --store-scores --skip-val [--corrupt-obs-eval]

`--corrupt-obs-eval` is REQUIRED on a ladder arm: without it corrupt_obs_eval stays False,
every slot is evaluated on the same clean observation, and the ladder under test is simply
not applied. It is meaningless on the uniform arm, which has no ladder to gate.
"""
import argparse
import datetime
import json
import pathlib
import random
import statistics as st

ROOT = pathlib.Path('/gscratch/robotics/harine/diffusion_policy_outputs')
BASE = ROOT / 'pusht_search' / 'pusht_image_search_imgonly' / 'outer_inner'
K = 16

# label | run dir | bon subdir | step | note
ARMS = [
    ('uniform control (no ladder)',
     'value_k16_ver-t_goal_enc-resnet18_demos-30_seed-42',
     'bon_search', 100000,
     'No ladder: `slot_obs_noise` uniform, so all 16 slots see the same clean observation. '
     'Slots still differ in SEARCH CONTEXT, so this measures what conditioning alone buys — '
     'and it is the row every other row must be read against. Rollouts are clean because '
     'there is no ladder to switch on.'),
    ('1. linear in t, cap 999',
     'value_k16_ver-t_goal_son-lint-cap999_enc-resnet18_demos-30_seed-42',
     'bon_search_obs-corrupt', 100000,
     '`t_k = (15-k)/15 * 999`. Slots 0-3 sit at sqrt(alpha_bar) 0.01/0.01/0.02/0.04 — four '
     'near-identical near-blind slots, so most of the ladder\'s usable range is squeezed '
     'into the clean end.'),
    ('1. linear in t, cap 400',
     'value_k16_ver-t_goal_son-lint-cap400_enc-resnet18_demos-30_seed-42',
     'bon_search_obs-corrupt', 100000,
     'The same shape compressed into [0, 400]: slot 0 at sqrt(alpha_bar) 0.44 rather than '
     '0.01, i.e. degraded rather than near-blind.'),
    ('2. linear in a_bar, cap 999',
     'value_k16_ver-t_goal_son-linsig-cap999_enc-resnet18_demos-30_seed-42',
     'bon_search_obs-corrupt', 100000,
     'sqrt(alpha_bar) falls in equal ~0.066 steps from 0.01 to 1.00. The only shape whose 16 '
     'levels are all distinct, and the one the config recommends as default.'),
    ('2. linear in a_bar, cap 400',
     'value_k16_ver-t_goal_son-linsig-cap400_enc-resnet18_demos-30_seed-42',
     'bon_search_obs-corrupt', 100000,
     'The same even grading compressed into [0, 400]. The compression is even in TIMESTEP, '
     'so the retained-signal steps are no longer equal.'),
    ('3. geometric in t, cap 999',
     'value_k16_ver-t_goal_son-geo85-cap999_enc-resnet18_demos-30_seed-42',
     'bon_search_obs-corrupt', 100000,
     '`t_k = 999 * 0.85^k`. Decay 0.85 rather than slot_weights\' 0.7, which at K=16 leaves '
     '6 of 15 adjacent slots indistinguishable. The trade: slot 15 lands at t=87, so this '
     'arm\'s cleanest slot is not fully clean.'),
    ('3. geometric in t, cap 400',
     'value_k16_ver-t_goal_son-geo85-cap400_enc-resnet18_demos-30_seed-42',
     'bon_search_obs-corrupt', 100000,
     'The same decay compressed into [0, 400]; 2 of 15 adjacent pairs fall within 0.005, so '
     'it is mildly collapsed at the clean end where the 999 arm is not.'),
    ('4. random base, noised rollouts',
     'value_k16_ver-t_goal_son-rndlinsig-cap999_enc-resnet18_demos-30_seed-42',
     'bon_search_obs-corrupt', 100000,
     'No fixed ladder: slot 0\'s timestep is drawn PER SAMPLE from [0, 999] and the '
     'linear_signal curve rescaled into [0, that draw], so what varies between samples is the '
     'ladder\'s EXTENT. **This row is a MECHANISM PROBE, outside the success-rate protocol**, '
     'which evaluates this arm clean only. It is included because a clean rollout does not '
     'apply the ladder at all, so it cannot answer whether the ladder built a slot gradient.'),
    ('4. random base, clean rollouts',
     'value_k16_ver-t_goal_son-rndlinsig-cap999_enc-resnet18_demos-30_seed-42',
     'bon_search_obs-clean', 100000,
     'The same weights read the way the success-rate tables read them — clean. With '
     '`corrupt_obs_eval` False the corruption is the identity, so all 16 slots see the same '
     'observation and this row should look like the uniform control. It is the check that '
     'the row above is measuring the ladder and not something else.'),
]



def stats(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r['scores'] for r in rows
            if r.get('n') == K and r.get('split') == 'test' and len(r.get('scores', [])) == K]
    if not rows:
        return None
    rnd = random.Random(0)
    means = [sum(s[j] for s in rows) / len(rows) for j in range(K)]
    sds = [st.pstdev([s[j] for s in rows]) for j in range(K)]
    hist = [0] * K
    for s in rows:
        m = max(s)
        hist[rnd.choice([j for j in range(K) if s[j] >= m - 1e-9])] += 1
    diff = [s[K - 1] - s[0] for s in rows]
    nz = [d for d in diff if abs(d) > 1e-9]
    return dict(
        n=len(rows), means=means, sds=sds, hist=hist,
        alltied=sum(1 for s in rows if max(s) - min(s) < 1e-9) / len(rows),
        endtied=1 - len(nz) / len(rows),
        win=(sum(1 for d in nz if d > 0) / len(nz)) if nz else float('nan'),
        nz=len(nz), meandiff=st.mean(diff),
        lo=sum(sum(s[0:8]) for s in rows) / (8 * len(rows)),
        hi=sum(sum(s[8:16]) for s in rows) / (8 * len(rows)))


def row(vals, fmt='{:.1f}'):
    return '| ' + ' | '.join(fmt.format(v) for v in vals) + ' |'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='slot_verifier_scores_noised_obs.md')
    args = ap.parse_args()

    L = ['# Per-slot verifier scores under the observation ladder', '',
         f'_Generated {datetime.date.today().isoformat()} by '
         '`scripts/build_slot_scores_doc.py`. Re-run to refresh._', '',
         'Does slot 15 actually produce better candidates than slot 0? The ladder assumes '
         'so — slot 0 sees the most-corrupted observation and no context, slot 15 the '
         'cleanest and the full context. Success rates cannot answer it: they report what '
         'the argmax executed, not what each slot produced.', '',
         'Candidate *i* conditions on exactly *i* scored context entries, so **i is the slot '
         'index** the training loss decodes. Scoring every candidate of a closed-loop '
         'rollout therefore reads the ladder directly.', '',
         'Values are `t_goal` = −(mean per-keypoint distance of the T from the goal), in '
         'pixels: **negative, and higher is better**. 50 test episodes, n = 16, seed 42, '
         'ResNet18 end-to-end, 30 demos.', '',
         '## Headline', '',
         '| arm | step | slot 0 | slot 15 | Δ mean | slot 15 > slot 0 |',
         '|---|---:|---:|---:|---:|---:|']
    got = []
    for label, run, sub, step, note in ARMS:
        p = BASE / run / sub / f'step_{step:07d}' / 'candidate_scores.jsonl'
        s = stats(p) if p.exists() else None
        got.append((label, run, sub, step, note, s))
        if s is None:
            L.append(f'| {label} | {step:,} | – | – | – | _not dumped yet_ |')
            continue
        L.append(f'| {label} | {step:,} | {s["means"][0]:.1f} | {s["means"][15]:.1f} | '
                 f'{s["meandiff"]:+.2f} | **{100*s["win"]:.1f}%** |')
    L += ['',
          'The win-rate excludes tied decisions (see Method). **50% is chance.** The control '
          'row is what every other row must be read against: it has the identical context '
          'structure and no ladder, so whatever it shows is the contribution of conditioning '
          'alone, and only the excess over it belongs to the ladder.', '']

    for label, run, sub, step, note, s in got:
        L += [f'## {label}', '', note, '',
              f'<sub>`{run}/{sub}/step_{step:07d}`</sub>', '']
        if s is None:
            L += ['_no candidate scores dumped for this arm yet_', '']
            continue
        L += [f'{s["n"]} decisions.', '',
              '| slot | ' + ' | '.join(str(j) for j in range(K)) + ' |',
              '|---|' + '---:|' * K,
              '| mean `t_goal` ' + row(s['means']),
              '| std ' + row(s['sds']),
              '| argmax share ' + row([100 * h / s['n'] for h in s['hist']], '{:.1f}%'),
              '',
              f'- slots 0–7 mean **{s["lo"]:.1f}**, slots 8–15 mean **{s["hi"]:.1f}** '
              f'(Δ {s["hi"]-s["lo"]:+.2f})',
              f'- slot 15 − slot 0: mean **{s["meandiff"]:+.2f}**, '
              f'slot 15 better in **{100*s["win"]:.1f}%** of the {s["nz"]} decisions where '
              f'the two differ',
              f'- ties: {100*s["alltied"]:.1f}% of decisions have all 16 scores equal; '
              f'{100*s["endtied"]:.1f}% have slot 0 = slot 15', '']

    L += ['## Method', '',
          '### Producing the data', '',
          '```',
          'sbatch scripts/slurm/eval_ckpt_pusht_search.sbatch <ckpt> \\',
          '    --n-list 16 --store-scores --skip-val [--corrupt-obs-eval]',
          '```', '',
          '`--corrupt-obs-eval` is **required** on a ladder arm. Without it `corrupt_obs_eval` '
          'stays False, every slot is evaluated on the same clean observation, and the ladder '
          'under test is not applied at all. It is meaningless on the uniform arm.', '',
          'Rollouts are closed-loop and execute the argmax, so the recorded scores are the '
          'ones a real deployment would have seen — not scores from a distribution the policy '
          'never visits.', '',
          '### Why ties are handled explicitly', '',
          'A candidate that never touches the block makes the simulator return the identical '
          'value, so roughly a third of decisions have all 16 scores exactly equal. That '
          'breaks two obvious statistics:', '',
          '- **The median paired difference is 0 by construction.** The win-rate here is '
          'therefore computed only over decisions where slot 0 and slot 15 actually differ.',
          '- **`argmax` breaks ties toward the lowest index**, manufacturing a slot-0 spike. '
          'This doc breaks them at random instead — but even then the control shows slot 0 '
          'far above the 1/16 = 6.25% baseline while its means are flat, so an ordering bias '
          'remains that has not been isolated. **Draw no conclusion from the argmax row**; it '
          'is descriptive colour. The mean profile and the tie-excluded win-rate are the '
          'numbers to read.', '',
          '## Caveats', '',
          '**Steps are not matched.** Each arm is read at whatever checkpoint existed when '
          'the dump ran, so cross-arm ABSOLUTE levels are indicative only. The within-arm '
          'slot profile — the thing under test — is unaffected, since all 16 slots come from '
          'one checkpoint and one set of decisions.', '',
          '**The ladder arms are scored under noised rollouts and the control under clean '
          'ones.** That is not an oversight: the control has no ladder, so there is nothing '
          'to switch on. It does mean the absolute gap between control and ladder arms '
          'confounds "trained with a ladder" against "deployed on corrupted observations".',
          '',
          '**One checkpoint per arm, 1900 decisions.** Enough to separate a 60% win-rate from '
          'chance, not enough to rank two arms a few points apart.', '']

    pathlib.Path(args.out).write_text('\n'.join(L) + '\n')
    print(f'wrote {args.out} ({len(L)} lines)')


if __name__ == '__main__':
    main()
