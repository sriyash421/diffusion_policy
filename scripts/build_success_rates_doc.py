"""Regenerate SUCCESS_RATES.md from the on-disk eval output.

Organised by DEMO BUDGET, because that is the axis the experiment varies:

    1. 100 demos  -- BC + the six search arms (Round 7, committed manifest)
    2.  29 demos  -- BC + the six legacy search arms (legacy manifest, val 10)
    A. archive    -- BC@25, and the pre-encoder-fix outer/inner runs

Each section carries three tables -- test success, val success, test mean reward -- with
one row per (feedback mechanism, obs, checkpoint) and n as COLUMNS. Every evaluated
checkpoint appears. Selection is never done on test.

    python scripts/build_success_rates_doc.py
"""
import json
import os
import pathlib
import re

import numpy as np

# $DP_OUTPUT_ROOT so this runs off-cluster against a copied results tree; the Hyak path is
# only the default. Everything it reads is the small bon_search/ JSON, not the checkpoints.
ROOT = pathlib.Path(os.environ.get(
    'DP_OUTPUT_ROOT', '/gscratch/robotics/harine/diffusion_policy_outputs'))
OFF = ROOT / 'pusht_search' / 'pusht_image_search' / 'offline'
NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# (arm label, corrupt|clean, bon_search dir).
#
# Directories are keyed by ARM, not by search_context: `selection` is a second axis, so
# `subgoal` means `subgoal-chosen4value` under argmax and `subgoal-only` under final_pass.
# Renamed 2026-08-05; see AUDIT.md 9.9. The old ctx-* paths survive as back-symlinks.
ARMS = (('value', 'value'),
        ('subgoal-chosen4value', 'subgoal'),
        ('subgoal-value', 'subgoal_value'))

D100 = [('none (BC)', '—', OFF / 'bc_demos-100_seed-42' / 'bon_search')]
for arm, _ctx in ARMS:
    for corrupt in ('False', 'True'):
        D100.append((arm, 'corrupt' if corrupt == 'True' else 'clean',
                     OFF / f'{arm}_corrupt-{corrupt}_demos-100_seed-42' / 'bon_search'))
# subgoal-only: SELECTION differs, not just the context. The arms above execute the
# verifier-argmax candidate; these draw one MORE sample conditioned on the n scored
# candidates and execute it unsimulated, so the verifier takes no part in selection at all.
# They also train to 100k steps rather than 20k.
#
# DISCOVERED BY GLOB, not listed. Variants get added faster than this file does -- three
# `k*_cd0.9` arms trained with no doc row because they postdated the hardcoded list, and a
# missing arm reads as "not run yet" rather than as an error. Anything matching the pattern
# shows up here the moment its directory exists.
#
# The label carries whatever the run name encodes beyond the arm: `k<N>` is the search
# width and `cd<x>` the context_decay (recency discount on the search context).
for bon in sorted((OFF).glob('subgoal-only*_demos-100_seed-42')):
    name = bon.name
    corrupt = 'corrupt' if '_corrupt-True' in name else 'clean'
    variant = (name.replace('subgoal-only', '')
                   .replace('_corrupt-True', '').replace('_corrupt-False', '')
                   .replace('_demos-100_seed-42', '').strip('_'))
    label = 'subgoal-only (final_pass)' + (f' {variant}' if variant else '')
    D100.append((label, corrupt, bon / 'bon_search'))

# Two generations share this budget, trained on the IDENTICAL 29 episodes, so the label
# has to say which is which in the table itself -- read side by side without it, the legacy
# rows look like a weaker version of the same policy when they are a different one.
#
#   legacy  the original runs. Two defects, both marked inline:
#             [no-EMA]      use_ema was False, so every number is from the LIVE weights,
#                           while all 100-demo numbers are from the EMA average (0.995).
#             [split-crop]  crop_shape/random_crop unset on the policy, so cropping only
#                           happened inside the encoder's CropRandomizer and an observation
#                           and the subgoal images derived from it got INDEPENDENT crop
#                           offsets -- translated relative to each other at train time.
#                           That is a correctness defect for the subgoal_* contexts, which
#                           is exactly where their numbers are read.
#           also val=10 (SE ~9.5pp) and 20k steps.
#   r8      the same 29 episodes with EMA on, one crop offset shared across the whole
#           search, val=30, and a 100k budget. Directory suffix `-r8`.
LEGACY29 = ' `[no-EMA]` `[split-crop]`'
D29 = [('none (BC)' + LEGACY29, 'legacy', OFF / 'bc_demos-29_seed-42' / 'bon_search')]
for arm, _ctx in ARMS:
    for corrupt in ('False', 'True'):
        D29.append((arm + LEGACY29, 'legacy ' + ('corrupt' if corrupt == 'True' else 'clean'),
                    OFF / f'{arm}_corrupt-{corrupt}_demos-29_seed-42' / 'bon_search'))
# Round 8 rows appear the moment their directories do; globbed so a new arm cannot go
# missing the way the k*_cd0.9 variants did.
for arm, _ctx in ARMS:
    for corrupt in ('False', 'True'):
        bon = OFF / f'{arm}_corrupt-{corrupt}_demos-29-r8_seed-42' / 'bon_search'
        if bon.parent.exists():
            D29.append((arm + ' **(r8)**',
                        'r8 ' + ('corrupt' if corrupt == 'True' else 'clean'), bon))

# ---------------------------------------------------------------------------
# OUTER/INNER trainer runs. `trainer` is a path component of hydra.run.dir
#   ${output_root}/${exp_name}/${task_name}/${trainer}/${run_name}
# so these live under outer_inner/ and NOT under offline/ -- which is why every table here
# silently omitted them until this block existed. They are a different trainer, not a
# variant of the offline arms: the search context is generated once per pool of 256 windows
# and reused for 4 inner epochs, rather than regenerated from the current weights on every
# gradient step.
#
# Globbed rather than listed, for the same reason the subgoal-only rows are: an arm that
# postdates this file should appear the moment its directory does, instead of reading as
# "not run yet". Budget is parsed off the run name so a run lands in its own section.
OI = ROOT / 'pusht_search' / 'pusht_image_search' / 'outer_inner'


def oi_sources(n_demos):
    """(label, obs, bon_search) for every outer/inner run at this demo budget."""
    out = []
    for run in sorted(OI.glob(f'*_demos-{n_demos}_seed-42')):
        arm = run.name.split('_corrupt-')[0]
        corrupt = 'corrupt' if '_corrupt-True' in run.name else 'clean'
        # Carry slot_weight_decay into the LABEL, not just the directory name. It changes
        # the training objective, and run_name gained a `swd-` component precisely so the
        # two decay variants of an arm are different experiments -- but this label was
        # derived by splitting on '_corrupt-', which drops everything after it. Both
        # variants of an arm therefore rendered as the same row label: two different
        # experiments, indistinguishable in the table, which is what _ARM_LABELS exists
        # to prevent on the training side.
        #
        # Absent for runs predating the rename, which were all trained at swd 1.0; those
        # keep their bare label so previously-published rows do not shift.
        for part in run.name.split('_'):
            if part.startswith('swd-'):
                arm = f'{arm} [{part}]'
                break
        out.append((arm + ' **(oi)**', 'oi ' + corrupt, run / 'bon_search'))
    return out


D100 += oi_sources(100)
D29 += oi_sources(29)

ARCHIVE = [('none (BC)', '25 demos', OFF / 'bc_demos-25_seed-42' / 'bon_search')]
# DIRECTORY NAMES, not config names -- the 2026-07-30 outer/inner runs live under
# ROOT/runs/<name>/ and are named after the configs that produced them. Those three config
# files were deleted on 2026-08-13 when outer/inner became the default in
# train_pusht_diffusion_search.yaml and they became exact duplicates of
# `base + search_context=<ctx> arm=<label>`. The directories are unaffected, and CURRENT
# outer/inner runs are discovered by oi_sources() above rather than listed here.
for name, lbl in (('train_pusht_search_outer_inner', 'value'),
                  ('train_pusht_search_outer_inner_subgoal', 'subgoal'),
                  ('train_pusht_search_outer_inner_subgoal_verifier', 'subgoal_value')):
    ARCHIVE.append((lbl, '29, pre-fix', ROOT / 'runs' / name / 'bon_search'))


# reward series a row can carry, in the order rows_for packs them after the success dicts
REWARD_SERIES = (
    ('reward',          'mean_reward'),              # test, episode MAX  (defines success)
    ('reward_final',    'mean_reward_final'),        # test, LAST step
    ('reward_disc',     'mean_reward_discounted'),   # test, discounted
    ('val_reward',      'val_mean_reward'),          # val, episode MAX   (the tie-breaker)
)


def rows_for(bon):
    """[(step, {n: test succ}, {n: val succ}, *{n: reward} for each REWARD_SERIES)]."""
    jl = bon / 'success_curves.jsonl'
    if not jl.is_file():
        return []
    out = []
    for line in jl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sr = dict(zip(r['n'], r['success_rate']))
        vs = dict(zip(r['n'], r.get('val_success_rate') or []))
        # `mean_reward` is now written straight into the jsonl. Before it existed this
        # opened each checkpoint's success_curve.json and averaged `per_n_rewards` -- ~370
        # multi-MB reads per regen, which is what made the build take minutes. The fallback
        # stays for any curve written by an older eval and not yet backfilled
        # (scripts/backfill_mean_reward.py); it produces the identical number.
        series = []
        for _, key in REWARD_SERIES:
            series.append({int(n): v for n, v in zip(r['n'], r.get(key) or [])
                           if v is not None})
        # Only the episode-MAX series has a fallback: it is the one every curve ever written
        # can reconstruct, because per_n_rewards has always stored it. `final` and
        # `discounted` are reductions of the per-STEP reward sequence, which was thrown away
        # at rollout time before 2026-08-05 -- there is nothing on disk to derive them from,
        # so a checkpoint evaluated before that shows blank until it is re-evaluated.
        if len(series[0]) < len(r['n']):
            det = bon / f'step_{r["step"]:07d}' / 'success_curve.json'
            if det.is_file():
                try:
                    d = json.loads(det.read_text())
                except json.JSONDecodeError:
                    d = {}
                for n, v in (d.get('per_n_rewards') or {}).items():
                    series[0].setdefault(int(n), float(np.mean(v)))
        out.append((r['step'], sr, vs, *series))
    return sorted(out)


def demo_table(sources, which='test', fmt=None):
    """EVERY checkpoint, with n as COLUMNS. One row per (mechanism, obs, checkpoint).

    Exhaustive rather than best-per-n: the per-checkpoint trajectory is what shows whether
    an arm peaks early and decays, which a summary row hides. `which` selects the quantity
    -- 'test' / 'val' success rate, or 'reward' (test mean reward).

    No checkpoint is marked. These tables used to star the val-selected best, which made one
    selection rule (val success at that n) read as the answer; picking a checkpoint is an
    analysis step, done from the curves, not something this generator should decide.
    """
    idx = {'test': 1, 'val': 2}.get(which)
    if idx is None:
        idx = 3 + [k for k, _ in REWARD_SERIES].index(which)
    if fmt is None:
        fmt = ((lambda v: f'{100*v:.0f}%') if which in ('test', 'val')
               else (lambda v: f'{v:.3f}'))
    L = ['| feedback mechanism | obs | checkpoint | ' + ' | '.join(f'n={n}' for n in NS) + ' |',
         '|---|---|---|' + '---|' * len(NS)]
    for mech, obs, bon in sources:
        rs = rows_for(bon)
        if not rs:
            # Distinguish "not evaluated" from "evaluated under the OTHER protocol". A run
            # with a criteria sweep but no n-sweep has real numbers in section 6, and a bare
            # `_pending_` here reads as though nothing has been measured for it at all.
            has_criteria = (bon.parent / 'criteria_search' / 'criteria_curves.jsonl').is_file()
            note = '_n-sweep pending; see §6_' if has_criteria else '_pending_'
            L.append(f'| {mech} | {obs} | {note} | ' + ' | '.join(['—'] * len(NS)) + ' |')
            continue
        first = True
        for row in rs:
            step, d = row[0], row[idx]
            cells = [fmt(d[n]) if n in d else '—' for n in NS]
            L.append(f'| {mech if first else ""} | {obs if first else ""} | {step} | '
                     + ' | '.join(cells) + ' |')
            first = False
    return L


REWARD_BLURB = ('\\nThe three reward series answer different questions, and PushT can separate them:\\n\\n| series | definition | what it catches that the others miss |\\n|---|---|---|\\n| **max** | best `clip(coverage/0.95,0,1)` over the episode | nothing extra — this is the one `success` thresholds at 1.0, so `mean reward (max) >= success rate` always |\\n| **final** | the same quantity at the LAST step | the policy that reaches the goal and then leaves. `max` scores that a perfect 1.0 by construction; `final` scores what the block is actually on at the end |\\n| **discounted** | `(1-gamma) * sum_t gamma^t r_t`, gamma=0.99 | how FAST. Two episodes that both end solved score identically on max and final, and this separates them. NOTE the scale: because the env terminates the instant coverage clears 0.95, the sum is truncated at the solve and the observed range across all 394 evaluated (checkpoint, n) points is only **[0.029, 0.372]** -- it never approaches 1.0, so read it as an ordering, not as a coverage fraction |\\n\\nVerified against success across all 394 points with all three series: rank-correlation with success rate is **max +0.947, final +0.939, discounted +0.880** -- all three order arms the same way as success, so none of them inverts. (A horizon SUM without the `(1-gamma)` factor does invert on this task, because a fast solve contributes fewer terms than a slow one; that is why the normalization is there.)\n\n**final and discounted only exist for checkpoints evaluated on or after 2026-08-05.** They are reductions of the per-STEP reward sequence, and every earlier eval reduced each episode to its max at rollout time and discarded the rest — so unlike `max`, they cannot be backfilled from anything on disk. Blank cells mean not-yet-re-evaluated, not zero.\\n')

L = ['# PushT best-of-N success rate\n',
     'Every evaluated checkpoint, with **n as columns**. One row per '
     '(feedback mechanism, obs, checkpoint).\n',
     '\n**No checkpoint is selected here.** These tables report measurements; they mark no '
     'winner. Earlier revisions starred the best checkpoint under one rule (val success at '
     'the largest n common to every row) and recorded the same pick in `bon_search/best.json`, '
     'which made that one rule read as the result. Choose a checkpoint yourself, from '
     '`bon_search/success_curves.jsonl`, and say which rule you used.\n',
     '\nWhen you do pick: taking the max over many noisy **test** estimates and reporting that '
     'same maximum inflates it by one to two standard errors, so select on val and read test '
     'at the selected step rather than maximising it.\n',
     '\n`success = coverage >= 95%`; `mean reward = clip(coverage/0.95, 0, 1)` averaged over '
     'episodes. At 50 test episodes SE is ~7pp near 50%, so gaps under ~14pp are not '
     'resolvable.\n',
     '\nThe 50 **test** episodes are identical in every section (`test = perm[:50]` does not '
     'move when the val split changes), so test columns are comparable across sections even '
     'though the training sets and the selectors are not.\n',
     '\nRegenerate with `python scripts/build_success_rates_doc.py`. It reads '
     '`$DP_OUTPUT_ROOT` (default: the Hyak path), so it rebuilds off-cluster against a copied '
     'results tree.\n',
     '\n**To analyse these results off the cluster, see '
     '[aug10_results2copy.md](aug10_results2copy.md)** — which directories to copy, what is '
     'inside each, and an rsync recipe. Everything except the model weights is ~2.3 GB.\n']

L += ['\n## What each arm is\n',
      'All arms share the same policy class, encoder, backbone and trainer. They differ only '
      'in what an already-generated candidate feeds back into the context the next candidate '
      'is conditioned on.\n',
      '',
      '| feedback mechanism | what a candidate contributes |',
      '|---|---|',
      '| **none (BC)** | nothing — `max_actions: 1`, so the context is always empty. '
      'Training is plain denoising BC and the verifier is never used. n=1 is an ordinary BC '
      'rollout; n>1 is best-of-n over **i.i.d.** samples, i.e. BC given the same test-time '
      'budget as search. |',
      '| **value** | the verifier scalar (−mean keypoint distance to the goal T); '
      'argmax selection |',
      '| **subgoal-chosen4value** | the observation its action chunk actually reaches in the '
      "sim, through the policy's own obs encoder. No scalar in the CONTEXT -- but the "
      'verifier scalar still *chooses* the executed candidate, which is what the name '
      'records. |',
      '| **subgoal-value** | both — encoded subgoal observation plus the scalar; argmax '
      'selection |',
      '| **subgoal-only (final_pass)** | the same subgoal context, but a different '
      '**selection rule**: once the n candidates are scored, one further sample is drawn '
      'conditioned on all of them and executed *unsimulated*, so the verifier ranks '
      'nothing. Trains to 100k steps rather than 20k. |',
      '| **clean vs corrupt** | `corrupt` adds forward-diffusion noise to the encoded obs '
      'features, at train and inference. PushT is fully observed, so on a clean obs the '
      'search context is a deterministic function of what the policy already sees; '
      'corruption is what gives the context something to add. **Read the corrupt rows '
      'against the caveats below before comparing them to anything.** |',
      '',
      'Candidate **ranking** uses the verifier scalar in every arm EXCEPT subgoal-only, so '
      'those are directly comparable at a given n. subgoal-only changes what n even means, '
      'since no candidate is selected by score.\n']

# ---------------------------------------------------------------------------
# The corrupt rows need their own health warning. Every `corrupt` row in every table
# below is affected; without this the columns read as a clean ablation against the
# `clean` rows sitting directly above them, which is the one thing they are not.
L += ['\n### What `corrupt` currently means — read before comparing corrupt rows\n',
      '\nCorruption is **not** applied to the images, the actions, the context or the '
      'subgoal embedding. It is forward-diffusion noise added to the **encoded observation '
      'feature vector** (`corrupt_obs_features`, '
      '`diffusion_transformer_search_policy.py`), DDPM linear beta 0.001 -> 0.02 over 100 '
      'steps, with the timestep drawn uniformly and independently on **every call**.\n',
      '\n| # | issue | effect on these tables |',
      '|---|---|---|',
      '| **P0-1** | the flag has no `self.training` gate, so corruption is active at '
      '**evaluation and rollout too**, not just training | a corrupt arm is solving a '
      'strictly harder task than the clean arm beside it. **corrupt vs clean is not a '
      'controlled comparison, and corrupt vs BC is not either** (BC has no corrupt run at '
      'any budget). |',
      '| **P0-2** | the magnitude is uncalibrated. `t ~ U{0..99}` is redrawn per call, so '
      'the noise coefficient ranges over `[0.032, 0.808]` (expected **0.476**); at t=99 the '
      'signal is scaled to 0.59 with noise of std 0.81 on top. The noise is absolute unit '
      'variance against a feature vector that concatenates unnormalized ResNet activations '
      'with two low-dim keys normalized to ~[-1,1] | **there is no single corruption level '
      'to quote.** Every corrupt cell averages over that whole range, at an SNR that '
      'differs across the feature vector by an unknown factor. A corrupt row is not "the '
      'clean row at noise level x". |',
      '| **P0-3** | each candidate in one search draws its **own** corruption, and '
      "`compute_loss` draws one more for the conditional it trains. So within a single "
      'decision candidate 0 may see a near-clean observation and candidate 5 a heavily '
      'corrupted one, yet their feedback is concatenated into one context as if all of them '
      'had been evaluated from the same observation | the search context in a corrupt arm '
      'mixes candidates scored from different observations. The agent has one observation, '
      'not n. |',
      '',
      '**Status:** P0-3 is the one with an agreed fix — share a single corruption draw '
      'across the whole decision, as the crop offset already is. It is deliberately **not '
      'applied yet**: two of the six r8 arms are corrupt and still training, and changing '
      'the semantics mid-run would leave their curve mixing both. It lands once those runs '
      'finish. P0-1 and P0-2 remain open. See `AUDIT.md` section 8.\n',
      '\nWhy corruption exists at all: with a clean observation on a fully observed task '
      '`p(a* | obs, context) = p(a* | obs)` exactly, so the Bayes-optimal model ignores the '
      'context and the clean arms cannot show a context effect even in principle '
      '(`AUDIT.md` 9.1). Corruption is what is supposed to break that — P0-1/P0-2 mean it '
      'does not do so cleanly.\n']

L += ['\n## 1. 100 demos\n',
      'Round 7. Committed manifest: **100 train / 30 val / 50 test**. BC trains to 300k '
      'steps (its own optimum); the six argmax search arms to 20k, since they peak at step '
      '1k–8k and decline after; subgoal-only to 100k.\n']

# ---- what the mean-reward quantity is, before the per-checkpoint detail
L += ['\n### 1·0 How to read mean reward (episode max)\n',
      '\n**What the number is.** For each episode, PushT scores every env step\n'
      '`reward = clip(coverage / 0.95, 0, 1)`, where `coverage` is the fraction of the goal\n'
      'area the block currently covers. Take the **maximum over the episode** — the best the\n'
      'block was ever placed — then the **mean over the 50 test episodes**. So a cell is the\n'
      "*average best coverage ratio achieved*, capped at 1.0.\n",
      '\nIt reads back as coverage directly: 0.66 means the average episode\'s best moment\n'
      'covered `0.66 × 0.95 ≈ 63%` of the goal. And 1.0 means essentially every episode\n'
      'reached the 95% threshold.\n',
      '\n**Why read it next to the binary rate.** `success` is exactly this quantity\n'
      'thresholded at 1.0, so it throws away everything below 95% coverage. Two policies that\n'
      'fail every episode are identical under `success` while one parks the block at 90%\n'
      'coverage and the other never touches it. That gap is not hypothetical here — it is the\n'
      'whole story of the subgoal-only arm, which sits near zero on success and 0.66–0.92 on\n'
      'this table.\n',
      '\nThe per-checkpoint tables below carry every measurement; read them as curves.\n']

L += ['\n### 1a. Binary success rate — TEST (50 episodes)\n']
L += demo_table(D100, 'test')
L += ['\n### 1b. Binary success rate — VAL (30 episodes)\n']
L += demo_table(D100, 'val')
L += ['\n### 1c. Mean reward, episode max — TEST\n',
      'Continuous, so it separates "nearly solved" from "never moved", which the binary rate '
      'does not. It is also what shows subgoal-only is learning normally (0.215 → 0.778 over '
      'its run) while its binary success stays near zero — the policy is fine, the selection '
      'rule is what fails.\n', REWARD_BLURB]
L += demo_table(D100, 'reward')
L += ['\n### 1d. Mean reward, FINAL step — TEST\n']
L += demo_table(D100, 'reward_final')
L += ['\n### 1e. Mean reward, DISCOUNTED (gamma=0.99) — TEST\n']
L += demo_table(D100, 'reward_disc')
L += ['\n### 1f. Mean reward, episode max — VAL (the selector\'s tie-break)\n']
L += demo_table(D100, 'val_reward')

L += ['\n## 2. 29 demos\n',
      'Three generations, trained on the **identical 29 episodes**, marked inline in every '
      'table below. Read side by side without the markers the legacy rows look like a '
      'weaker version of the same policy; they are a different one.\n',
      '\n**Naming.** Everything trained before Round 8 is the **legacy 29** generation — '
      'the rows carrying `[no-EMA]` `[split-crop]` and an `obs` of `legacy clean` / '
      '`legacy corrupt`, in run directories `<arm>_corrupt-<c>_demos-29_seed-42`. The '
      'Round-8 re-runs are marked **(r8)** with `obs` `r8 clean` / `r8 corrupt`, in '
      '`<arm>_corrupt-<c>_demos-29-r8_seed-42`. Two defects and two budget differences '
      'separate them:\n',
      '\n| marker | what is wrong with the legacy runs | why it matters here |\n'
      '|---|---|---|\n'
      '| **`[no-EMA]`** | `use_ema: False`. Every legacy number comes from the **live** '
      'weights. Every 100-demo number in section 1 comes from the **EMA average** (decay '
      '0.995, a ~200-step window). | Different weights, not just different data. A '
      'diffusion policy sees one random timestep x one noise draw per update, so the live '
      'iterate rattles; the legacy rows carry that jitter and the section-1 rows do not. |\n'
      '| **`[split-crop]`** | `crop_shape` / `random_crop` unset on the **policy**, so '
      "cropping happened only inside the encoder's `CropRandomizer`. An observation and the "
      'subgoal images derived from it therefore received **independent** crop offsets and '
      'were translated relative to each other at train time. The 100-demo arms put one '
      'offset in scope for the whole search. | A correctness defect, not a tuning choice — '
      'and it lands hardest on `subgoal-chosen4value` and `subgoal-value`, whose entire '
      'context is a subgoal image that no longer registers with the observation it came '
      'from. |\n',
      '\nLegacy runs also use **val = 10** (SE ~9.5pp at p=0.9 — three of them tied at '
      '9/10 while their test numbers were 84 / 70 / 32%) and stop at **20k** steps. The '
      '`(r8)` rows fix all four: EMA on, one shared crop offset, val = 30, 100k steps. '
      'Each legacy run directory carries the same list in `RUN_CAVEATS.json`.\n',
      '\nThe 29 training episodes themselves are **exactly reproducible** — '
      '`downsample_mask` over `get_split_masks_3way` at `train_ratio: 0.2`, seed 42, '
      'recorded verbatim in `pusht_seed42_legacy_val10_train29.json` and verified '
      'index-for-index against the runs\' own saved config. (They cannot be reproduced '
      'with `n_train_episodes: 29`, which takes a prefix of the permuted pool and overlaps '
      'in only **4 of 29**.) What the legacy directories lack is provenance: they predate '
      '`splits.json`, so nothing *on disk* ties those checkpoints to those episodes. The '
      'r8 runs train on the same 29 via a committed manifest, so old-vs-new isolates the '
      'four changes above.\n',
      '\n**The `(oi)` rows are a THIRD generation, and the one thing that differs is the '
      'training LOOP.** Same policy class, same 29 episodes (the r8 manifest, val = 30), '
      'same EMA and shared crop offset as `(r8)`, same 20k budget — but trained by '
      '`TrainSearchOuterInnerWorkspace` instead of `TrainMLPImageWorkspace`, so they live '
      'under `outer_inner/<arm>_corrupt-<c>_demos-29_seed-42` rather than `offline/`. The '
      'offline loop regenerates the whole search context from the *current* weights on '
      'every gradient step; the outer/inner loop generates it once for a pool of 256 '
      'windows and reuses it for 4 inner epochs — about 4x cheaper per update (480 -> 120 '
      'verifier sims), paid for with a context drawn from weights up to 32 updates stale. '
      'That staleness is measured, not assumed: `train_drift_mse_eps` in each run\'s '
      '`logs.json.txt` is the epsilon-space MSE against a frozen snapshot of the policy '
      'that filled the buffer, which is proportional to the per-denoising-step KL.\n',
      '\nSo `(oi)` vs `(r8)` at the same arm isolates the loop, and nothing else. Note the '
      '`(r8)` rows run to 100k steps while `(oi)` stops at 20k, so compare them at a '
      'matched step rather than at each row\'s end.\n',
      '\n**The r8 runs have no val curve at all.** Their eval watchers were launched '
      '`--skip-val` (`scripts/slurm/launch_round8_29demo.sh`) so the whole budget went to the '
      '50 test episodes. Any checkpoint picked from these rows is therefore picked on TEST, '
      'and a test number read at a step chosen on test is not a held-out estimate. Read the '
      'r8 rows as a curve; if you need a held-out number from them, re-evaluate on val '
      'first.\n',
      '\nThe r8 generation is also still **in progress** and covers only the three argmax '
      'arms — there is no r8 `subgoal-only` and no r8 BC, so the crop/EMA fix has not been '
      'applied to the `final_pass` selection rule at any budget.\n']
L += ['\n### 2a. Binary success rate \u2014 TEST (50 episodes)\n']
L += demo_table(D29, 'test')
L += ['\n### 2b. Binary success rate \u2014 VAL (10 episodes; this is the selector)\n',
      'Only 10 episodes, so each cell moves in 10pp steps and SE is ~9.5pp at p=0.9. Three '
      'of these arms tied at 9/10 while their test numbers were 84 / 70 / 32%.\n']
L += demo_table(D29, 'val')
L += ['\n### 2c. Mean reward \u2014 TEST\n']
L += demo_table(D29, 'reward')
L += ['\n### 2d. Mean reward, FINAL step — TEST\n']
L += demo_table(D29, 'reward_final')
L += ['\n### 2e. Mean reward, DISCOUNTED (gamma=0.99) — TEST\n']
L += demo_table(D29, 'reward_disc')
L += ['\n### 2f. Mean reward, episode max — VAL (the selector\'s tie-break)\n']
L += demo_table(D29, 'val_reward')

L += ['\n### What differs between the 29- and 100-demo generations, besides the demo count\n',
      'From a direct diff of the two saved `.hydra/config.yaml` files, so this is what the '
      'runs actually used rather than what the configs say today. **The 29-demo search arms '
      'are not simply a smaller-data version of the 100-demo ones.**\n',
      '',
      '| | 29-demo (legacy) | 100-demo (Round 7) | what it affects |',
      '|---|---|---|---|',
      '| `use_ema` | **False** | **True** (decay 0.995) | **the evaluated weights.** Legacy '
      'numbers come from the live weights; Round 7 numbers from the EMA average. |',
      '| `policy.crop_shape` / `random_crop` | **unset** — cropping happens only inside the '
      "encoder's `CropRandomizer` | **[76,76] / True** — crop owned by the policy, one "
      'offset shared per sample | **the trained model.** The policy-level crop is what lets '
      'an observation and the subgoal images derived from it share a crop offset; with the '
      'encoder-only crop they are translated relative to each other at train time. |',
      '| val split | 10 episodes | 30 episodes | **checkpoint selection.** At 10 episodes SE '
      'is ~9.5pp at p=0.9 — three legacy arms tied at 9/10 while their test numbers were '
      '84 / 70 / 32%. |',
      '| split source | derived at runtime from seed + `train_ratio` | committed manifest | '
      'reproducibility only |',
      '| eval cadences | epoch-based | gradient-step based (`*_every_steps`) | which '
      'checkpoints exist, and when metrics were logged |',
      '',
      'So a 29-vs-100 comparison confounds demo count with EMA and with the crop change. '
      'BC@29 sits on the legacy *data* split but uses the *current* config (EMA on, policy '
      'crop on), so **BC@29 vs BC@100 isolates demo count cleanly**, while BC@29 vs the '
      '29-demo search arms carries the caveats above.\n']


# ---------------------------------------------------------------- selection sweep -------
# Selection readouts swept post-hoc, each in its own bon_search_sel-<mode>/ directory.
#
# `final_pass` is here as a READOUT on arms that were trained under argmax -- the model
# synthesises one extra action conditioned on all n scored candidates and executes it
# unsimulated, instead of picking the oracle's best. Swept at n=16 only on the K=16 arms,
# which is the width those models were actually trained at (compute_loss conditions on
# max_actions-1 = 15 context entries) and, per AUDIT.md P2-7, the one width nothing else
# evaluates. Listing it here rather than hardcoding two modes is what stops a swept mode
# from landing on disk and appearing in no table -- which is exactly what happened to the
# k*_cd0.9 arms in section 1 before that list was globbed.
SEL_MODES = ('argmax', 'softmax', 'final_pass')


def selection_rows(bon_parent):
    """{(step, mode): {n: test success}} from bon_search_sel-<mode>/ next to bon_search/."""
    out = {}
    for mode in SEL_MODES:
        jl = bon_parent / f'bon_search_sel-{mode}' / 'success_curves.jsonl'
        if not jl.is_file():
            continue
        for line in jl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r['step'], mode)] = dict(zip(r['n'], r['success_rate']))
    return out


def selection_table():
    """argmax vs softmax on the SAME weights, test split only.

    Selection is a pure readout rule, so both run on an already-trained checkpoint and the
    only thing that differs is how the verifier's ranking is consumed. That is what makes
    this a controlled comparison -- and what `final_pass` cannot isolate, since it also
    changes who produces the action.
    """
    L = ['| run | native selection | ckpt | rule | '
         + ' | '.join(f'n={n}' for n in NS) + ' |',
         '|---|---|---|---|' + '---|' * len(NS)]
    for d in sorted(OFF.glob('*_seed-42')):
        if d.is_symlink() or not d.is_dir():
            continue
        rows = selection_rows(d)
        if not rows:
            continue
        native = 'final_pass' if 'subgoal-only' in d.name else (
            'none (BC)' if d.name.startswith('bc_') else 'argmax')
        name = d.name.replace('_demos-100_seed-42', '').replace('_seed-42', '')
        first = True
        for step in sorted({s for s, _ in rows}):
            # the step prints once per block, on whichever mode happens to come first --
            # keying it to 'argmax' left the step column blank for any step where only
            # softmax or only final_pass had landed yet
            first_of_step = True
            for mode in SEL_MODES:
                sr = rows.get((step, mode))
                if sr is None:
                    continue
                cells = [f'{100*sr[n]:.0f}%' if n in sr else '—' for n in NS]
                L.append(f'| {name if first else ""} | {native if first else ""} | '
                         f'{step if first_of_step else ""} | `{mode}` | '
                         + ' | '.join(cells) + ' |')
                first = False
                first_of_step = False
    return L


L += ["\n## 3. Selection rule: argmax vs softmax vs final_pass (TEST only)\n",
      'Both rules run on the **same trained weights** — selection is a pure readout over '
      'the n scored candidates, so it can be swapped after training. Only the 50 test '
      'episodes are evaluated (`--skip-val`), since the checkpoints here are named up '
      'front and nothing is being selected.\n',
      '\n`softmax` samples the executed candidate from `softmax(z / T)`, `T = 1`, where '
      '`z` is the verifier score **standardized across the n candidates**. Standardizing is '
      'what lets one temperature mean the same thing everywhere: the raw score is −mean '
      'keypoint distance in pixels and its spread varies by arm, by checkpoint and with n. '
      'The limits are exact — `T→0` reproduces `argmax` candidate for candidate, `T→∞` is a '
      'uniform pick. **n=1 is identical between the two rules by construction** (a single '
      'candidate leaves nothing to choose), which is why that column agrees.\n',
      '\n`native selection` is the rule the checkpoint was TRAINED under. For the '
      '`subgoal-only` rows that is `final_pass`, so the `argmax` line here is a readout the '
      'policy never saw during training.\n',
      '\n**`final_pass` rows on an argmax-native arm are the reverse of that**, and they '
      'answer a different question from the argmax/softmax pair. argmax and softmax both '
      'PICK one of the n simulated candidates, so they isolate how the verifier ranking is '
      'consumed. `final_pass` instead draws one MORE sample conditioned on all n scored '
      'candidates and executes it **unsimulated**, so the verifier never selects anything — '
      "`action_value_final − action_value_best` is that arm's whole question: does the "
      "model's own synthesis beat the oracle argmax it replaces? It is swept at **n=16 "
      'only, on the K=16 arms**, because that is the width those models were trained at '
      '(`compute_loss` conditions on `max_actions - 1` = 15 context entries) and, per '
      '`AUDIT.md` P2-7, the one width nothing else evaluates. Note it costs n+1 samples to '
      "argmax's n, so compare at equal samples, not equal n.\n"]
L += selection_table()

# ---------------------------------------------------------------------------
# SECTION 6 -- selection criteria at fixed width. A DIFFERENT EXPERIMENT from sections 1-3,
# and it gets its own section rather than extra columns for that reason: there n is the
# axis and selection is held at argmax, here n is pinned at 16 and the READ-OUT RULE is the
# axis. The two do not belong on one x axis any more than eval_bon and eval_search do.
#
# Source is criteria_search/criteria_curves.jsonl, written by
# `eval_search_pusht.py --criteria-sweep`, which is keyed on (step, criterion).
# ---------------------------------------------------------------------------
CRITERIA_ORDER = ['cand-last', 'cand-8th-from-last', 'argmax-all', 'argmax-last8',
                  'softmax-all', 'softmax-last8']


def criteria_rows(run_dir):
    """{step: {criterion: row}} for one run, or {} if it has no criteria sweep."""
    jl = run_dir / 'criteria_search' / 'criteria_curves.jsonl'
    if not jl.is_file():
        return {}
    out = {}
    for line in jl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue      # torn final line from a preempted write
        out.setdefault(r['step'], {})[r['criterion']] = r
    return out


def criteria_table(run_dir, key, fmt):
    """One run's table: criterion across columns, checkpoint down rows."""
    data = criteria_rows(run_dir)
    if not data:
        return []
    L = ['| checkpoint | ' + ' | '.join(f'`{c}`' for c in CRITERIA_ORDER) + ' |',
         '|---|' + '---|' * len(CRITERIA_ORDER)]
    for step in sorted(data):
        cells = []
        for c in CRITERIA_ORDER:
            r = data[step].get(c)
            cells.append(fmt(r[key]) if r and r.get(key) is not None else '—')
        L.append(f'| {step} | ' + ' | '.join(cells) + ' |')
    return L


CRIT_RUNS = sorted(OI.glob('*_swd-*_demos-100_seed-42'))
if CRIT_RUNS:
    L += ['\n## 6. Selection rule at fixed search width (n=16)\n',
          'A different experiment from sections 1-3, which is why it is a separate section '
          'rather than extra columns: there `n` is the axis and selection is held at '
          '`argmax`; here `n` is pinned at **16** and the READ-OUT RULE is the axis. Every '
          'criterion scores the SAME 16 generated-and-scored candidates, so generation is '
          'identical across a row and only the choice of which candidate to execute '
          'differs. No retraining is involved.\n',
          '\nCandidate *k* is conditioned on candidates 0..*k*-1, so the trailing candidates '
          'are the deeply-conditioned ones. That is what the pairs separate:\n',
          '\n| column | what it executes |',
          '|---|---|',
          '| `cand-last` | candidate 15 (most conditioned). Verifier IGNORED. |',
          '| `cand-8th-from-last` | candidate 8. Verifier IGNORED. |',
          '| `argmax-all` | highest verifier value of all 16 |',
          '| `argmax-last8` | highest verifier value among candidates 8-15 |',
          '| `softmax-all` | sampled from softmax(z/T) over all 16 |',
          '| `softmax-last8` | sampled from softmax(z/T) over candidates 8-15 |',
          '',
          '\n`argmax-all` minus `cand-last` is the verifier ranking\'s contribution on top '
          'of conditioning depth; `cand-last` minus `cand-8th-from-last` is what the extra '
          'conditioning alone buys. `*-last8` minus `*-all` says whether restricting the '
          'oracle to well-conditioned candidates helps or merely removes options.\n',
          '\n**TEST split, 50 episodes, `--skip-val`.** With no val curve, a checkpoint '
          'picked off these tables is picked on test and is NOT a held-out estimate -- read '
          'whole curves, and say which rule you used if you quote a single step. At 50 '
          'episodes SE is ~7pp near 50%, so gaps under ~14pp are not resolvable.\n',
          '\nPer-step traces (every candidate\'s verifier value and the executed index) sit '
          'in `criteria_search/step_*/traces/<criterion>.npz`; validate one with '
          '`python scripts/check_criteria_traces.py <step dir>`.\n']
    n_shown = 0
    for run in CRIT_RUNS:
        name = run.name.replace('_corrupt-False', '').replace('_demos-100_seed-42', '')
        blocks = []
        for letter, label, key, fmt in (
                ('a', 'binary success rate', 'success_rate', lambda v: f'{100*v:.0f}%'),
                ('b', 'mean reward, episode max', 'mean_reward', lambda v: f'{v:.3f}'),
        ):
            tbl = criteria_table(run, key, fmt)
            if tbl:
                blocks += [f'\n#### 6.{n_shown + 1}{letter} `{name}` — {label}\n'] + tbl
        if blocks:
            n_shown += 1
            L += [f'\n### 6.{n_shown} {name}\n'] + blocks
    if n_shown == 0:
        # every run globbed but none has a criteria sweep yet: say so rather than emit a
        # section header with nothing under it
        L += ['\n_No criteria sweeps have been evaluated yet._\n']

L += ['\n## 4. Where the raw results are\n',
      'Everything in this file is DERIVED. The raw per-checkpoint eval output lives under '
      '`$DP_OUTPUT_ROOT` = `/gscratch/robotics/harine/diffusion_policy_outputs`; nothing is '
      "on home, where one run's checkpoints alone exceed the 10G quota.\n",
      '\n**Copying this off the cluster: [aug10_results2copy.md](aug10_results2copy.md).** It '
      'lists all 26 run directories plus the outer/inner runs and `candidate_scores/`, says '
      'what each file below is worth keeping for, and carries an rsync recipe. Everything '
      'except the model weights is ~2.3 GB against 213 GB for a full mirror; set '
      '`DP_OUTPUT_ROOT` to the copy and this document rebuilds from it. Two traps it '
      'documents: the `ctx-*` entries are back-symlinks, so `rsync -L` copies every run '
      'twice, and the `checkpoint` paths recorded inside the jsonl are absolute cluster '
      'paths.\n',
      '',
      '| file | what it holds |',
      '|---|---|',
      '| `bon_search/success_curves.jsonl` | **the raw source for every table here** — one '
      'JSON row per checkpoint: `n`, `success_rate` (test), `val_success_rate`, '
      '`val_mean_reward`, `success_ci`, `episode_idxs`, `seed`. Rewritten as a merge under a '
      'file lock, so concurrent per-n jobs cannot lose each other\'s levels. |',
      '| `bon_search/step_XXXXXXX/success_curve.json` | per-checkpoint detail: '
      '`per_n_rewards`, the full 50-episode reward vector at each n, from which the mean '
      'reward tables are computed. |',
      '| `logs.json.txt` | every training metric per step (losses, nRMSE, rollout scores). |',
      '| `splits.json` | the exact val/test episode indices the run trained under; the eval '
      'cross-checks against it and refuses to score a mismatched partition. |',
      '',
      '```',
      '$DP_OUTPUT_ROOT/pusht_search/pusht_image_search/offline/',
      '    bc_demos-{25,29,100}_seed-42/                  BC baselines (max_actions 1)',
      '    ctx-<mech>_corrupt-<bool>_demos-100_seed-42/   100-demo search arms  [section 1]',
      '    subgoal-only_corrupt-<bool>_demos-100_seed-42/ 100-demo, final_pass  [section 1]',
      '    ctx-<mech>_corrupt-<bool>_seed-42/             29-demo search arms   [section 2]',
      '        checkpoints/step_XXXXXXX.ckpt',
      '        bon_search/success_curves.jsonl           one row per checkpoint',
      '        bon_search/step_XXXXXXX/success_curve.json  per_n_rewards, episode_idxs',
      '```',
      '',
      'Split manifests (`diffusion_policy/config/splits/`):',
      '```',
      'pusht_seed42_train25.json                25 train / 30 val / 50 test',
      'pusht_seed42_train100.json              100 train / 30 val / 50 test',
      'pusht_seed42_legacy_val10_train29.json   29 train / 10 val / 50 test  (legacy)',
      '```',
      '',
      '```',
      'scripts/build_success_rates_doc.py       regenerates this file',
      'scripts/slurm/autoupdate_success_doc.sh  re-runs the above on a timer while evals land',
      'scripts/slurm/run_status.sh              every run on disk: step, losses, ckpt/eval counts',
      'scripts/slurm/launch_round8_29demo.sh    the r8 29-demo trainings (one config + overrides)',
      'scripts/slurm/submit_selection_sweep.sh  section 3: argmax vs softmax, every 10k step',
      'scripts/slurm/submit_large_n_evals.sh    the n>=128 tail, ONE JOB PER n',
      'scripts/slurm/fill_eval_gaps.sh          finds checkpoints/levels with no 50-episode eval',
      'scripts/dump_pusht_splits.py             generates/verifies the split manifests',
      'archive/launch_round7*.sh            the Round 7 trainings (superseded, see archive/)',
      '```',
      '',
      'The large-n tail runs one job per n because cost is linear in n and the levels sum to '
      '~2*max_n: a single n<=1024 sweep is ~10h per checkpoint. `eval_search_pusht.py` '
      'persists after every n and merges under a file lock, so levels run independently and '
      'preemption is survivable.\n']

L += ['\n---\n', '\n# A. Archive\n',
      '**Superseded — do not compare to sections 1 or 2.**\n',
      '',
      '- **BC @ 25 demos** — a budget no search arm matches, so it has no counterpart to be '
      'a baseline for. Kept because it is the run that established BC needed ~300k steps '
      'rather than 20k: at 20k it solved 1 of 1000 test episodes at n=1. It ended at 99k '
      'when `num_epochs: 1000` bound before `max_gradient_steps` — harmless, since its '
      'val_loss bottoms at 0.049 @ 14k and rises to 0.132 by 88k.',
      '- **outer/inner runs (2026-07-30)** — ImageNet weights together with GroupNorm, an '
      'incompatible pairing (`use_group_norm` replaces every BatchNorm with a freshly '
      'initialised GroupNorm, discarding exactly the statistics the pretrained weights '
      'depend on), and no augmentation. Overfit by ~3k steps then trained to 100k, so every '
      'evaluated checkpoint sits deep in the overfit regime.',
      '']
arch = demo_table([(m, t, bon) for m, t, bon in ARCHIVE], 'test')
L += ['\n### Archive \u2014 binary success rate (TEST)\n'] + arch

# ---------------------------------------------------------------------------
# HAND-WRITTEN SECTIONS. This builder does not own the whole document.
#
# It discovers offline/ and outer_inner/ only -- there is no unet_bc/ root here -- so the
# architecture 2x2 (section 5) is maintained by hand. It sits BETWEEN generated sections,
# and this script used to end with a bare write_text() that replaced the file wholesale, so
# running the regenerate command this very document advertises silently deleted it.
#
# Fix: hand-written regions are fenced with sentinels in SUCCESS_RATES.md,
#
#   <!-- HAND-WRITTEN after: ## 3. Selection rule ... -->
#   ...prose and tables this script knows nothing about...
#   <!-- END HAND-WRITTEN -->
#
# and are re-inserted immediately before the heading named in `after:`. Sentinels rather
# than diffing the heading set, because the generated headings themselves change between
# revisions and a diff would then treat a renamed section as hand-written.
#
# A block whose anchor no longer exists is APPENDED with a warning rather than dropped --
# losing content silently is the exact failure this is here to prevent.
# ---------------------------------------------------------------------------
HW_OPEN = re.compile(r'^<!--\s*HAND-WRITTEN\s+after:\s*(.+?)\s*-->\s*$')
HW_CLOSE = '<!-- END HAND-WRITTEN -->'


def extract_hand_written(text):
    """[(anchor_heading, block_lines)] for every sentinel-fenced region in `text`."""
    blocks, lines, i = [], text.split('\n'), 0
    while i < len(lines):
        m = HW_OPEN.match(lines[i])
        if not m:
            i += 1
            continue
        anchor, start = m.group(1), i
        j = i + 1
        while j < len(lines) and lines[j].strip() != HW_CLOSE:
            j += 1
        if j >= len(lines):
            print(f'WARNING: unclosed HAND-WRITTEN block at line {start + 1} '
                  f'(anchor {anchor!r}); it will NOT be preserved. Add {HW_CLOSE}.')
            break
        blocks.append((anchor, lines[start:j + 1]))
        i = j + 1
    return blocks


def splice_hand_written(generated, blocks):
    """Re-insert each block immediately before its anchor heading."""
    lines = '\n'.join(generated).split('\n')
    for anchor, block in blocks:
        try:
            at = next(k for k, ln in enumerate(lines) if ln.strip() == anchor.strip())
        except StopIteration:
            print(f'WARNING: anchor {anchor!r} is no longer generated; appending the '
                  f'hand-written block at the end so it is not lost. Re-point its '
                  f'`after:` sentinel at a heading that still exists.')
            lines += [''] + block
            continue
        lines[at:at] = block + ['']
    return '\n'.join(lines)


out = pathlib.Path(__file__).resolve().parent.parent / 'SUCCESS_RATES.md'
preserved = extract_hand_written(out.read_text()) if out.is_file() else []
out.write_text(splice_hand_written(L, preserved).rstrip('\n') + '\n')
n_ck = sum(len(rows_for(b)) for _, _, b in D100 + D29 + ARCHIVE)
kept = ', '.join(a for a, _ in preserved) or 'none'
print(f'wrote {out} ({out.stat().st_size} bytes, {n_ck} checkpoints across all sections)')
print(f'hand-written blocks preserved: {kept}')
