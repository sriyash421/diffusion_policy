"""Regenerate success_rates_slot_weights_d_t_goal.md from on-disk eval output.

The round-2 slot-weight profiles re-run under `d_t_goal` -- `-d_T->goal / 13.6`, the task
term on armTn's normalized footing -- so they sit alongside the t_goal-family reference
arms (ST k=1, ST k=16 uniform, UNet BC) instead of the armTn ones.

    python scripts/build_dtgoal_doc.py [-o success_rates_slot_weights_d_t_goal.md]

Safe to re-run mid-sweep: it reads success_curves.jsonl and nothing else. Like the armTn
doc it nominates no best checkpoint and no best n -- every evaluated cell is printed.
"""
import argparse
import datetime
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from build_30_100_success_doc import (            # noqa: E402
    BASE, NS, STEPS, by_step, provenance, read_rows, table)

VER = 'd_t_goal'
K, R2 = 16, 100
_K16 = BASE / 'outer_inner'
CTRL = 'ST k=16 — uniform, L2'

# The three REFERENCE arms carry `t_goal` in their native n-sweep -- they were trained and
# swept under it, and d_t_goal ranks identically (a positive divisor is monotone), so those
# curves are valid here. Verified end to end: UNet BC step-100k argmax n=8 returns 0.480
# under both values on identical flags, rewards agreeing to every digit.
#
# (label, run dir, native-sweep verifier, what differs from the control)
ARMS = [
    ('ST k=1 — uniform, L2',
     BASE / 'offline' / 'bc_demos-30_seed-42', 't_goal', 'width 1: no search context'),
    (CTRL, _K16 / 'value_k16_corrupt-False_demos-30_seed-42', 't_goal',
     'uniform slot weights, plain L2'),
    (f'ST k=16 — linear r={R2}, L2',
     _K16 / f'value_k16_ver-{VER}_sw-lin100-l2_corrupt-False_demos-30_seed-42', VER,
     f'`slot_weights: linear, ratio {R2}`'),
    (f'ST k=16 — linear r={R2}, l2tol1',
     _K16 / f'value_k16_ver-{VER}_sw-lin100-l2tol1_corrupt-False_demos-30_seed-42', VER,
     f'`slot_weights: linear, ratio {R2}` + `slot_loss_norm: l2tol1`'),
    ('ST k=16 — geometric d=0.735, L2',
     _K16 / f'value_k16_ver-{VER}_sw-geo735-l2_corrupt-False_demos-30_seed-42', VER,
     '`slot_weights: geometric, decay 0.735`'),
    ('ST k=16 — geometric d=0.735, l2tol1',
     _K16 / f'value_k16_ver-{VER}_sw-geo735-l2tol1_corrupt-False_demos-30_seed-42', VER,
     '`slot_weights: geometric, decay 0.735` + `slot_loss_norm: l2tol1`'),
    (f'ST k=16 — curriculum→linear r={R2}, L2',
     _K16 / f'value_k16_ver-{VER}_sw-curr-lin100-l2_corrupt-False_demos-30_seed-42', VER,
     f'30k steps uniform, then `linear, ratio {R2}`'),
    (f'ST k=16 — curriculum→linear r={R2}, l2tol1',
     _K16 / f'value_k16_ver-{VER}_sw-curr-lin100-l2tol1_corrupt-False_demos-30_seed-42', VER,
     f'30k steps uniform, then `linear, ratio {R2}` + `slot_loss_norm: l2tol1`'),
]
BC_ARM = ('UNet BC (reference)', BASE / 'unet_bc' / 'unetbc_demos-30_seed-42', 't_goal')

# argmax and final_pass only -- the two rules this round asks for. Every row is measured
# UNDER d_t_goal, so they share one verifier suffix.
READOUT_NS = [1, 8, 16]
READOUTS = [('argmax', f'bon_search_sel-argmax_ver-{VER}'),
            ('final_pass', f'bon_search_sel-final_pass_ver-{VER}')]


def linear_weights(ratio, k=K):
    base = [1.0 + (ratio - 1.0) * i / (k - 1) for i in range(k)]
    m = sum(base) / k
    return [b / m for b in base]


def geometric_weights(decay, k=K):
    base = [decay ** (k - 1 - i) for i in range(k)]
    m = sum(base) / k
    return [b / m for b in base]


def row(name, vals, fmt='{:.3f}'):
    return f'| {name} | ' + ' | '.join(fmt.format(v) for v in vals) + ' |'


def delta_table(a_ctrl, a_var):
    lines = ['| step | ' + ' | '.join(f'n={n}' for n in NS) + ' |',
             '|---:|' + '---:|' * len(NS)]
    any_row = False
    for step in STEPS:
        if step not in a_ctrl or step not in a_var:
            continue
        lines.append(f'| {step:,} | ' + ' | '.join(
            f'{a_var[step][n] - a_ctrl[step][n]:+.2f}'
            if n in a_ctrl[step] and n in a_var[step] else '–' for n in NS) + ' |')
        any_row = True
    return '\n'.join(lines) if any_row else \
        '_not yet comparable — one side has no swept checkpoint_'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='success_rates_slot_weights_d_t_goal.md')
    args = ap.parse_args()
    L = []
    A = L.append

    A('# PushT per-slot weighting experiments (d_t_goal) — success rates\n')
    A('The round-2 slot-weight profiles at a 100:1 endpoint spread, trained and evaluated '
      'under **`d_t_goal`**. Every arm: k=16, 30 demos, seed 42, 100k gradient steps with a '
      'checkpoint every 10k, on the same 50 held-out test episodes.\n')
    A(f'_Generated {datetime.date.today().isoformat()} from the on-disk eval output._\n')
    A('Regenerate with `python scripts/build_dtgoal_doc.py`. Source of truth is each run\'s '
      '`success_curves.jsonl`.\n')

    A('## The verifier\n')
    A('`d_t_goal` = `-d_T→goal / 13.6` — the T-to-goal distance over its within-step '
      'candidate spread, i.e. `t_goal` on the same normalized footing as each armTn term. '
      'It **ranks identically to `t_goal`**: a positive divisor is monotone, so argmax picks '
      'the same candidate. What changes is the magnitude of the recorded scalar and of the '
      'context the model trains against.\n')
    A('\nThat is why the three **reference arms** below are read from their `t_goal` sweeps '
      'and are legitimate rows here. Verified end to end rather than asserted: UNet BC at '
      'step 100k, argmax, n=8, on byte-identical flags returns **0.480 under both values**, '
      'with mean rewards agreeing to every digit.\n')

    A('## The profiles\n')
    lin, geo = linear_weights(R2), geometric_weights(0.735)
    A('\n| slot | ' + ' | '.join(str(i) for i in range(K)) + ' |')
    A('|---|' + '---|' * K)
    A(row(f'linear r={R2}', lin))
    A(row('geometric d=0.735', geo))
    A(f'\nBoth mean 1. `linear r={R2}`: {lin[0]:.4f} → {lin[-1]:.4f}, a straight ramp. '
      f'`geometric d=0.735`: {geo[0]:.4f} → {geo[-1]:.4f} (ratio {geo[-1] / geo[0]:.1f}:1) — '
      'under 1.0 until slot 10, then a spike, so it concentrates the gradient on the last '
      'few conditionals. The **curriculum** arms hold uniform weights for 30k steps, then '
      f'switch to `linear r={R2}` for the remaining 70k (`interp: step`, so each objective '
      'is trained on for its whole stretch).\n')

    A('## How to read the numbers\n')
    A('Success rate is the fraction of the 50 test episodes reaching the goal coverage '
      'threshold. At 50 episodes a single cell carries a 95% CI of roughly ±0.13 near 0.5, '
      'so **cell-to-cell differences under ~0.15 are not separable** — read down a column, '
      'not one cell. No cell is singled out; picking the largest is selection on test.\n')

    seen = {}
    for label, run, ver, knob in ARMS:
        A(f'## {label}\n')
        A(f'`{run.relative_to(BASE)}` — {knob}\n')
        rows = read_rows(run, 'bon_search', ver)
        ck, done, partial, _f = provenance(rows, run)
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
        if any(v for v in rw.values()):
            A('\n### Test mean reward (max coverage reached)\n')
            A(table(rw))
        A('')

    A('## Variant − control, paired by step and n\n')
    A(f'Positive = the variant scored higher than **{CTRL}** at that checkpoint and width. '
      'Blank where either side has no measurement; nothing is imputed.\n')
    for label, _r, _v, _k in ARMS:
        if label in (CTRL, 'ST k=1 — uniform, L2'):
            continue
        A(f'\n### {label}  vs  {CTRL}\n')
        A(delta_table(seen.get(CTRL, {}), seen.get(label, {})))
    A('')

    A('## Choice mechanism (n = 1, 8, 16)\n')
    A('How the executed action is picked, on the SAME trained weights. `argmax` takes the '
      'best of n by verifier value; `final_pass` generates n-1 candidates, scores them, and '
      "executes the n'th conditioned on all of them, unsimulated.\n")
    A('\n⚠️ At **n=1** the two coincide — with one candidate there is nothing to select, so '
      '`final_pass` returns the same unconditioned action `argmax` does. Those two columns '
      'agreeing is a consistency check, not two results.\n')
    cols = [(lbl, n) for lbl, _s in READOUTS for n in READOUT_NS]
    for label, run, _ver, _k in ARMS + [(BC_ARM[0], BC_ARM[1], None, None)]:
        aggs = {lbl: by_step(read_rows(run, sub, VER), 'success_rate')
                for lbl, sub in READOUTS}
        rows_out = []
        for step in STEPS:
            cells = [(f'{aggs[l][step][n]:.2f}'
                      if step in aggs[l] and n in aggs[l][step] else '–') for l, n in cols]
            if any(c != '–' for c in cells):
                rows_out.append(f'| {step:,} | ' + ' | '.join(cells) + ' |')
        A(f'\n### {label}\n')
        if not rows_out:
            A('_no read-out evals on disk yet._')
            continue
        A('| step | ' + ' | '.join(f'{l} n={n}' for l, n in cols) + ' |')
        A('|---:|' + '---:|' * len(cols))
        A('\n'.join(rows_out))
    A('')

    A('## Status\n')
    A('| arm | differs by | checkpoints | fully swept |')
    A('|---|---|---:|---:|')
    for label, run, ver, knob in ARMS:
        ck, done, _p, _f = provenance(read_rows(run, 'bon_search', ver), run)
        A(f'| {label} | {knob} | {len(ck)}/10 | {len(done)} |')
    ck, done, _p, _f = provenance(read_rows(BC_ARM[1], 'bon_search', BC_ARM[2]), BC_ARM[1])
    A(f'| {BC_ARM[0]} | eval only — no slots, a reference row | {len(ck)}/10 | {len(done)} |')
    A('')

    out = pathlib.Path(args.out)
    out.write_text('\n'.join(L) + '\n')
    print(f'wrote {out} ({len(ARMS) + 1} arms, '
          f'{sum(len(v) for v in seen.values())} evaluated checkpoints)')


if __name__ == '__main__':
    main()
