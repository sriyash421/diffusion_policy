"""Regenerate success_rates_slot_weights_armTn.md from on-disk eval output.

Two per-candidate-slot knobs, each varied ONE AT A TIME off the arms already on disk:

  slot_loss_norm.mode=l2tol1                     which NORM each slot's loss term uses
  slot_weights.mode=linear ratio=4.857           how much each slot's term is SCALED

Everything else is matched to the controls -- 30 demos, seed 42, armTn verifier, 100k
gradient steps, a checkpoint every 10k, and the same 50 held-out test episodes swept over
n = 1..64.

    python scripts/build_slot_norm_doc.py [-o success_rates_slot_weights_armTn.md]

Safe to re-run mid-sweep: it reads each run's bon_search/success_curves.jsonl and nothing
else, so arms that have not reached a checkpoint simply render as pending. Like
build_30_100_success_doc.py it deliberately nominates no best checkpoint and no best n --
every evaluated cell is printed and the reader picks.
"""
import argparse
import datetime
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# The row reader, per-step aggregation, table rendering and provenance are identical to the
# 30/100 doc's; importing them keeps the two docs literally the same measurement.
from build_30_100_success_doc import (            # noqa: E402
    BASE, NS, STEPS, by_step, provenance, read_rows, table)

VERIFIER = 'armTn'
SUB = 'bon_search'            # trained AND evaluated under armTn, so the default subdir
K = 16
RATIO = 4.857                 # = 0.9^-(K-1): the endpoint spread of a geometric decay 0.9

CTRL_K1 = 'ST k=1 — uniform, L2'
CTRL_K16 = 'ST k=16 — uniform, L2'

# Every arm is labelled by its PROFILE, its RATIO and its NORM. With eleven arms "linear
# slot weights" no longer identifies one, and two arms differing only in ratio would read as
# duplicates. Labels only -- the run DIRECTORIES are not renamed: their paths are referenced
# by the watcher list, the read-out submitter and every curve already on disk.
#
# (label, run dir, what differs from the matched control)
_K16 = BASE / 'outer_inner'
_R1, _R2 = 4.857, 100          # round-1 and round-2 linear endpoint spreads
ARMS = [
    (CTRL_K1,
     BASE / 'offline' / 'value_k1_ver-armTn_corrupt-False_demos-30_seed-42',
     'uniform slot weights, plain L2'),
    ('ST k=1 — uniform, l2tol1',
     BASE / 'offline' / 'value_k1_ver-armTn_l2tol1_corrupt-False_demos-30_seed-42',
     '`slot_loss_norm.mode: l2tol1`'),
    (CTRL_K16,
     _K16 / 'value_k16_ver-armTn_corrupt-False_demos-30_seed-42',
     'uniform slot weights, plain L2'),
    ('ST k=16 — uniform, l2tol1',
     _K16 / 'value_k16_ver-armTn_l2tol1_corrupt-False_demos-30_seed-42',
     '`slot_loss_norm.mode: l2tol1`'),
    (f'ST k=16 — linear r={_R1}, L2',
     _K16 / 'value_k16_ver-armTn_sw-lin4857_corrupt-False_demos-30_seed-42',
     f'`slot_weights: linear, ratio {_R1}`'),
    # ---- round 2: the same two knobs at a 100:1 endpoint spread ----
    (f'ST k=16 — linear r={_R2}, L2',
     _K16 / 'value_k16_ver-armTn_sw-lin100-l2_corrupt-False_demos-30_seed-42',
     f'`slot_weights: linear, ratio {_R2}`'),
    (f'ST k=16 — linear r={_R2}, l2tol1',
     _K16 / 'value_k16_ver-armTn_sw-lin100-l2tol1_corrupt-False_demos-30_seed-42',
     f'`slot_weights: linear, ratio {_R2}` + `slot_loss_norm: l2tol1`'),
    ('ST k=16 — geometric d=0.735, L2',
     _K16 / 'value_k16_ver-armTn_sw-geo735-l2_corrupt-False_demos-30_seed-42',
     '`slot_weights: geometric, decay 0.735`'),
    ('ST k=16 — geometric d=0.735, l2tol1',
     _K16 / 'value_k16_ver-armTn_sw-geo735-l2tol1_corrupt-False_demos-30_seed-42',
     '`slot_weights: geometric, decay 0.735` + `slot_loss_norm: l2tol1`'),
    (f'ST k=16 — curriculum→linear r={_R2}, L2',
     _K16 / 'value_k16_ver-armTn_sw-curr-lin100-l2_corrupt-False_demos-30_seed-42',
     f'30k steps uniform, then `linear, ratio {_R2}`'),
    (f'ST k=16 — curriculum→linear r={_R2}, l2tol1',
     _K16 / 'value_k16_ver-armTn_sw-curr-lin100-l2tol1_corrupt-False_demos-30_seed-42',
     f'30k steps uniform, then `linear, ratio {_R2}` + `slot_loss_norm: l2tol1`'),
]

# every k=16 arm except the control, i.e. everything differenced against it
_K16_VARIANTS = [a[0] for a in ARMS
                 if a[0].startswith('ST k=16') and a[0] != CTRL_K16]

# Extra READ-OUT rules, run on the k=16 variant arms' trained weights. Each writes to its
# own bon_search_sel-<label>/ so no two rules can merge into one curve.
#
# `argmax` is the PAIRED CONTROL and is not redundant with the native bon_search/ column:
# the search is stochastic, and a --selection argmax re-run of the same weights disagrees
# with the native curve at 20 of 24 checkpoints, by up to 22pp (see the header of
# submit_selection_sweep.sh). final_pass and index8 are read against THIS, never against
# the native column.
READOUT_NS = [1, 8, 16]
READOUTS = [
    ('argmax', 'bon_search_sel-argmax', 'best-of-n over the verifier — the paired control'),
    ('final_pass', 'bon_search_sel-final_pass',
     "n-1 candidates scored, the n'th generated conditioned on them and executed unsimulated"),
    ('cand 8', 'bon_search_sel-index8',
     'the 8th candidate in generation order (1-based), scores ignored entirely'),
]
READOUT_ARMS = [CTRL_K16] + _K16_VARIANTS

# The UNet BC reference, read out the same way. Not a slot-weighting arm -- it has no slots
# -- so it is NOT in ARMS and gets its own subsection. Its curves live under the armTn
# verifier subdirs so they sit on the same footing as everything else here.
BC_ARM = ('UNet BC (reference)', BASE / 'unet_bc' / 'unetbc_demos-30_seed-42')
BC_READOUTS = [
    ('argmax', 'bon_search_sel-argmax_ver-armTn', 'best-of-n over the verifier'),
    ('final_pass', 'bon_search_sel-final_pass_ver-armTn',
     'the last of n i.i.d. draws -- see the caveat below'),
]

# (variant, its control). A variant is only ever differenced against the arm it was matched
# to -- k=1 against k=1, k=16 against k=16.
PAIRS = ([('ST k=1 — uniform, l2tol1', CTRL_K1)]
         + [(v, CTRL_K16) for v in _K16_VARIANTS])


def alphas(k=K):
    """The l2tol1 L1 fraction per slot. Degenerate (single pure-L2 slot) at k=1."""
    return [i / (k - 1) for i in range(k)] if k > 1 else [0.0]


def linear_weights(ratio=RATIO, k=K):
    """slot_weights.mode=linear, renormalized to mean 1 exactly as _slot_weights does."""
    base = [1.0 + (ratio - 1.0) * i / (k - 1) for i in range(k)]
    m = sum(base) / k
    return [b / m for b in base]


def geometric_weights(decay, k=K):
    """slot_weights.mode=geometric, renormalized to mean 1 exactly as _slot_weights does."""
    base = [decay ** (k - 1 - i) for i in range(k)]
    m = sum(base) / k
    return [b / m for b in base]


def row(name, vals, fmt='{:.3f}'):
    return f'| {name} | ' + ' | '.join(fmt.format(v) for v in vals) + ' |'


def delta_table(agg_ctrl, agg_var):
    """variant - control per (step, n). Blank where either side is missing; never imputed."""
    lines = ['| step | ' + ' | '.join(f'n={n}' for n in NS) + ' |',
             '|---:|' + '---:|' * len(NS)]
    any_row = False
    for step in STEPS:
        if step not in agg_ctrl or step not in agg_var:
            continue
        cells = [f'{agg_var[step][n] - agg_ctrl[step][n]:+.2f}'
                 if n in agg_ctrl[step] and n in agg_var[step] else '–' for n in NS]
        lines.append(f'| {step:,} | ' + ' | '.join(cells) + ' |')
        any_row = True
    return '\n'.join(lines) if any_row else \
        '_not yet comparable — one side has no swept checkpoint_'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='success_rates_slot_weights_armTn.md')
    args = ap.parse_args()

    L = []
    A = L.append
    A('# PushT per-slot weighting experiments (armTn) — success rates\n')
    A('Two per-candidate-slot knobs, each varied one at a time off an arm already on disk. '
      'Every arm: 30 demos, seed 42, `armTn` verifier, 100k gradient steps with a checkpoint '
      'every 10k, every checkpoint swept over `n = 1, 2, 4, 8, 16, 32, 64` on the same 50 '
      'held-out test episodes.\n')
    A(f'_Generated {datetime.date.today().isoformat()} from the on-disk eval output._\n')
    A('⚠️ **The six round-2 arms were stopped before 100k steps** (2026-08-27), to re-run the '
      'same profiles under the `d_t_goal` verifier instead. Their curves are complete up to '
      'the last checkpoint each reached — read the Status table for how many that is, and do '
      'not read a 7/10 arm as a finished run. Everything already written was fully '
      'evaluated; nothing was orphaned mid-sweep.\n')
    A('Regenerate with `python scripts/build_slot_norm_doc.py` — it rewrites this file in '
      'place. Source of truth is each run\'s `bon_search/success_curves.jsonl`.\n')

    A('## The slots, and the two knobs\n')
    A('One forward decodes all K candidate slots against the same expert action; slot `k` '
      'may attend to exactly the first `k` scored context candidates. So the model fits a '
      'family of conditionals at once, from no context (slot 0) to K-1 scored candidates '
      '(slot K-1). The two knobs act on different parts of that:\n')
    A('\n| knob | acts on | shape used here |')
    A('|---|---|---|')
    A('| `slot_loss_norm` | the **norm** of each slot\'s loss term | `l2tol1` |')
    A(f'| `slot_weights` | the **scale** of each slot\'s loss term | `linear`, ratio {RATIO} |')
    A('\nThey are orthogonal and compose; each arm below turns on exactly one.\n')

    A('### `slot_loss_norm: l2tol1`\n')
    A('Slot 0 (no context) pure L2, slot K-1 (full context) pure L1, linear between. Per '
      'element, with `α_k = k/(K-1)` the L1 fraction:\n')
    A('```\nloss_k = (1 - α_k)·(pred - target)² + α_k·|pred - target|\n```\n')
    a = alphas()
    A('\n| slot | ' + ' | '.join(str(i) for i in range(K)) + ' |')
    A('|---|' + '---|' * K)
    A(row('α (L1)', a))
    A(row('1-α (L2)', [1 - v for v in a]))
    A('\nNot renormalized to mean 1 — it picks a norm per slot rather than splitting a fixed '
      'loss budget.\n')
    A('\n⚠️ **At k=1 the profile is degenerate.** With one slot there is nothing to '
      'interpolate and slot 0 IS the pure-L2 end, so `_slot_norm_alphas` returns `None` and '
      'the k=1 `l2tol1` arm trains under plain MSE — its training log prints no α vector, '
      'while the k=16 one prints the row above. Its numbers are expected to match the k=1 '
      'control up to run-to-run nondeterminism, and that agreement is a consistency check '
      'on this table rather than a result. **The k=16 pairs are the real comparisons.**\n')

    A(f'### `slot_weights: linear, ratio {RATIO}`\n')
    A('`w_k ∝ 1 + (ratio-1)·k/(K-1)`, renormalized to mean 1 — so switching profiles cannot '
      'move the loss SCALE that `gradient_clip_norm`, the effective step size and `val_loss` '
      f'all read. Ratio {RATIO} = `0.9^-(K-1)` is the endpoint spread of a geometric decay '
      'of 0.9, the documented pairing; there is no 30-demo armTn geometric arm on disk, so '
      'what this measures is **linear-vs-uniform**, not linear-vs-geometric curvature.\n')
    w = linear_weights()
    A('\n| slot | ' + ' | '.join(str(i) for i in range(K)) + ' |')
    A('|---|' + '---|' * K)
    A(row('weight', w))
    A(f'\nLast/first = {w[-1] / w[0]:.3f}, mean = {sum(w) / K:.3f}. `slot_weights.val` stays '
      '`uniform`, so `val_loss` remains a fixed cross-arm yardstick.\n')

    # ---- round 2: the same knobs at a 100:1 spread ------------------------------------
    A('### Round 2 — 100:1 profiles\n')
    A(f'Round 1 (linear r={RATIO}, and l2tol1) moved nothing: both variants sat within ~0.03 '
      'of the uniform control in every cell of a 70-cell n-sweep, against a ±0.13 per-cell '
      'CI. Round 2 asks whether a much steeper tilt does anything, and whether SHAPE '
      'matters at that tilt — a straight ramp against a sharp late spike with the same '
      'endpoint ratio.\n')
    lin100 = linear_weights(_R2)
    geo735 = geometric_weights(0.735)
    A('\n| slot | ' + ' | '.join(str(i) for i in range(K)) + ' |')
    A('|---|' + '---|' * K)
    A(row(f'linear r={_R2}', lin100))
    A(row('geometric d=0.735', geo735))
    A(f'\n`linear r={_R2}`: {lin100[0]:.4f} → {lin100[-1]:.4f}, a straight ramp. '
      f'`geometric d=0.735`: {geo735[0]:.4f} → {geo735[-1]:.4f}, endpoint ratio '
      f'{geo735[-1] / geo735[0]:.1f}:1 — it stays under 1.0 until slot 10 and then spikes, '
      'so it concentrates the gradient on the last few conditionals rather than tilting all '
      'sixteen. Both are mean 1.\n')

    A('### Round 2 — the curriculum\n')
    A('Two arms warm up on uniform weights before the profile is switched on:\n')
    A('\n| steps | duration | profile |')
    A('|---|---|---|')
    A('| 0 → 30k | 30k | uniform (all 1.000) |')
    A(f'| 30k → 100k | 70k | linear r={_R2} ({lin100[0]:.4f} → {lin100[-1]:.4f}) |')
    A('\nThe profile is **held** for each stretch rather than interpolated through '
      '(`slot_weights.interp: step`), so each objective is actually trained on for its whole '
      'span. Both profiles are mean 1, so the switch at 30k does not move the loss scale.\n')

    A('## How to read the numbers\n')
    A('Success rate is the fraction of the 50 test episodes reaching the goal coverage '
      'threshold. At 50 episodes a single cell carries a 95% CI of roughly ±0.13 near 0.5, '
      'so **cell-to-cell differences under ~0.15 are not separable** — read down a column '
      'or across several checkpoints, never one cell. The delta tables are printed in full '
      'with no cell singled out; picking the largest one is selection on test.\n')

    seen = {}
    for label, run, knob in ARMS:
        A(f'## {label}\n')
        A(f'`{run.relative_to(BASE)}/{SUB}` — {knob}\n')
        rows = read_rows(run, SUB, VERIFIER)
        ck, done, partial, _facts = provenance(rows, run)
        if not ck:
            A('_no checkpoints written yet — training has not reached step 10,000._\n')
            seen[label] = {}
            continue
        A(f'_{len(ck)}/10 checkpoints written, {len(done)} fully swept'
          + (f', {len(partial)} partial' if partial else '') + '._\n')
        agg = by_step(rows, 'success_rate')
        seen[label] = agg
        A('### Test success rate\n')
        A(table(agg))
        rw = by_step(rows, 'mean_reward')
        # by_step keys every step it sees, so an absent FIELD yields a dict of empty dicts
        # -- truthy, and it rendered a table of dashes. Gate on a cell actually existing.
        if any(v for v in rw.values()):
            A('\n### Test mean reward (max coverage reached)\n')
            A(table(rw))
        A('')

    A('## Variant − control, paired by step and n\n')
    A('Positive = the variant scored higher at that checkpoint and that search width. Blank '
      'where either side has no measurement; nothing is imputed.\n')
    for var, ctrl in PAIRS:
        A(f'\n### {var}  vs  {ctrl}\n')
        A(delta_table(seen.get(ctrl, {}), seen.get(var, {})))
    A('')

    A('## Choice mechanism (k=16 arms, n = 1, 8 and 16)\n')
    A('Selection is a pure read-out — which of the n scored candidates is executed — so all '
      'three rules run on the SAME trained weights. `n = 1, 8, 16`: 1 is the no-search '
      'anchor (at n=1 `final_pass` takes the empty-context branch, so it returns the same '
      'unconditioned action argmax would), 8 is the smallest width at which an 8th '
      'candidate exists, and 16 is the width the model is trained at (`compute_loss` '
      'conditions on `max_actions-1` = 15 context entries).\n')
    for label, _sub, what in READOUTS:
        A(f'* **{label}** — {what}')
    A('\n⚠️ **Read these against the `argmax` row, not against the tables above.** The search '
      'is stochastic (every candidate is a fresh DDIM draw), and a `--selection argmax` '
      're-run of the same weights disagrees with the native `bon_search` curve at 20 of 24 '
      'checkpoints, by up to 22pp. The paired argmax re-run is the only valid control here.\n')
    A('\n**At n=8, `final_pass` and `cand 8` are the same read-out.** `final_pass` at n '
      'generates n-1 candidates, scores them, and executes the n\'th conditioned on all of '
      'them; `cand 8` at n=8 executes the 8th of 8, which is likewise conditioned on the 7 '
      'scored candidates before it. Same conditional, so those two columns should agree to '
      'within rollout noise — a free correctness check on two independent code paths, not '
      'two results. They diverge only at n=16, where `final_pass` executes the 16th '
      'generation conditioned on 15 and `cand 8` still executes the 8th conditioned on 7.\n')
    A('\nThe `cand 8` row measures the SAME conditional at both widths — candidate 8 is '
      'generated identically whether the search runs 8 or 16 candidates — so the gap '
      'between its two columns is a direct read on rollout noise at 50 episodes.\n')
    for arm in READOUT_ARMS:
        run = dict((lbl, r) for lbl, r, _k in ARMS)[arm]
        A(f'\n### {arm}\n')
        cols = [(lbl, n) for lbl, _s, _w in READOUTS for n in READOUT_NS]
        aggs = {lbl: by_step(read_rows(run, sub, VERIFIER), 'success_rate')
                for lbl, sub, _w in READOUTS}
        rows = []
        for step in STEPS:
            cells = [('n/a' if lbl == 'cand 8' and n < 8 else
                      f'{aggs[lbl][step][n]:.2f}'
                      if step in aggs[lbl] and n in aggs[lbl][step] else '–')
                     for lbl, n in cols]
            if any(c != '–' for c in cells):
                rows.append(f'| {step:,} | ' + ' | '.join(cells) + ' |')
        if not rows:
            A('_no read-out evals on disk yet._\n')
            continue
        A('| step | ' + ' | '.join(f'{lbl} n={n}' for lbl, n in cols) + ' |')
        A('|---:|' + '---:|' * len(cols))
        A('\n'.join(rows))
        A('')

    # ---- the BC reference, rendered apart because final_pass does not mean the same thing
    bc_label, bc_run = BC_ARM
    A(f'\n### {bc_label}\n')
    A('`unet_bc/unetbc_demos-30_seed-42` — a diffusion UNet with no search context, under '
      'the same armTn verifier and the same 50 test episodes.\n')
    bc_cols = [(lbl, n) for lbl, _s, _w in BC_READOUTS for n in READOUT_NS]
    bc_aggs = {lbl: by_step(read_rows(bc_run, sub, VERIFIER), 'success_rate')
               for lbl, sub, _w in BC_READOUTS}
    bc_rows = []
    for step in STEPS:
        cells = [(f'{bc_aggs[lbl][step][n]:.2f}'
                  if step in bc_aggs[lbl] and n in bc_aggs[lbl][step] else '–')
                 for lbl, n in bc_cols]
        if any(c != '–' for c in cells):
            bc_rows.append(f'| {step:,} | ' + ' | '.join(cells) + ' |')
    if bc_rows:
        A('| step | ' + ' | '.join(f'{lbl} n={n}' for lbl, n in bc_cols) + ' |')
        A('|---:|' + '---:|' * len(bc_cols))
        A('\n'.join(bc_rows))
    else:
        A('_no read-out evals on disk yet._')
    A('')

    A('## Status\n')
    A('| arm | differs by | checkpoints | fully swept | selection | episodes | seed |')
    A('|---|---|---:|---:|---|---:|---:|')
    for label, run, knob in ARMS:
        rows = read_rows(run, SUB, VERIFIER)
        ck, done, _p, facts = provenance(rows, run)

        def one(k):
            v = {x for x in facts.get(k, set()) if x is not None}
            return sorted(map(str, v))[0] if len(v) == 1 else \
                ('—' if not v else '/'.join(sorted(map(str, v))))
        A(f'| {label} | {knob} | {len(ck)}/10 | {len(done)} | {one("selection")} | '
          f'{one("n_episodes")} | {one("seed")} |')
    A('')

    out = pathlib.Path(args.out)
    out.write_text('\n'.join(L) + '\n')
    print(f'wrote {out} ({len(ARMS)} arms, '
          f'{sum(len(v) for v in seen.values())} evaluated checkpoints total)')


if __name__ == '__main__':
    main()
