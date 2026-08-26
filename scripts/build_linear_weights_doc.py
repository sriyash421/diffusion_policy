"""Regenerate SUCCESS_RATES_LINEAR_WEIGHTS_DRUMKIT.md from the sweep logs.

The arm is ST-diffusion k=16 (4/4/256) trained on 30 demos for 100k gradient steps with
LINEAR per-slot loss weights at ratio 4.857, launched by
scripts/run_st_k16_linear_drumkit.sh on drumkit. Every 10k checkpoint is swept over
{argmax, softmax, final_pass} x n = 1..64 on the same 50 test / 30 val episodes.

    python scripts/build_linear_weights_doc.py [-o SUCCESS_RATES_LINEAR_WEIGHTS_DRUMKIT.md]

SOURCE IS THE DRIVER LOG, NOT success_curves.jsonl -- and that is not a preference.
eval_search_pusht.py writes its curve keyed on the CHECKPOINT, not on (checkpoint,
selection), so the three selection rules of one checkpoint overwrite each other in both
`<arm>/success_curves.jsonl` and `<arm>/step_*/success_curve.json`: whichever ran last
(final_pass, under this driver's loop order) is the only one that survives, wearing all
seven n values. The pre-existing `search` arm shows the identical collapse -- 10 rows, all
final_pass -- so two thirds of every grid ever run is absent from the jsonl. The per-n
numbers are printed to stdout for all three rules, so the log is the only complete record,
which is why scripts/bon_grid_table.py parses it too. Reading the jsonl here would silently
have labelled final_pass results as argmax.

Safe to re-run mid-sweep: cells that have not been evaluated simply have no line to parse.
The driver calls it after every cell, so the doc tracks the sweep live.

Like the other success-rate builders this deliberately does NOT nominate a best checkpoint
or a best n. Every evaluated cell is printed; picking a winner here would be selection on
test, which is the thing these grids exist to avoid.
"""
import argparse
import json
import os
import pathlib
import re

ROOT = pathlib.Path(os.environ.get(
    'DP_OUTPUT_ROOT', '/home/harine/diffusion_policy_outputs'))
BASE = ROOT / 'pusht_search' / 'pusht_image_search'
NS = [1, 2, 4, 8, 16, 32, 64]
STEPS = [10000 * k for k in range(1, 11)]
SELECTIONS = ('argmax', 'softmax', 'final_pass')

# Every log the driver writes. Repeatable on the command line; later logs win on collision,
# so a re-run of a timed-out cell supersedes the cell it replaces.
LOGS = ['logs/run_st_k16_linear_drumkit.log',
        'logs/run_st_k16_uniform_drumkit.log']

# (log arm key, run dir, heading, one-line gloss). The uniform armTn control is listed but
# renders "not trained" until someone runs --uniform-control -- it is the arm this doc's
# numbers actually need, so it is named here rather than left implicit.
ARMS = [
    ('st-k16-lin4857',
     BASE / 'outer_inner' / 'value_k16_ver-armTn_sw-lin4857_corrupt-False_demos-30_seed-42',
     'Linear slot weights, ratio 4.857',
     'w_k affine in the slot index, mean 1, w_last/w_first = 4.857'),
    ('st-k16-unif-armTn',
     BASE / 'outer_inner' / 'value_k16_ver-armTn_corrupt-False_demos-30_seed-42',
     'Uniform slot weights (matched control)',
     'the same arm with slot_weights untouched -- every slot at weight 1'),
]

# The linear profile at K=16, ratio 4.857, printed rather than recomputed so the doc states
# what the run actually trained under even if the resolver later changes.
LINEAR_W = [0.341, 0.429, 0.517, 0.605, 0.693, 0.780, 0.868, 0.956,
            1.044, 1.132, 1.220, 1.307, 1.395, 1.483, 1.571, 1.659]

# `=== [<iso>] <arm> <step_XXXXXXX> <selection> ===`, echoed once per n-slice.
_HEADER = re.compile(r'=== \S+ ([\w-]+) step_(\d+) (\w+) ===')
_CELL = re.compile(r'(val|test) n=(\d+): success_rate=([\d.]+)')


def read_cells(paths):
    """{(arm, step, selection, split): {n: success_rate}} parsed from the driver logs."""
    out = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        txt = open(path, errors='ignore').read()
        parts = _HEADER.split(txt)
        # split() yields [preamble, arm, step, sel, body, arm, step, sel, body, ...]
        for i in range(1, len(parts), 4):
            arm, step, sel, body = parts[i], int(parts[i + 1]), parts[i + 2], parts[i + 3]
            for split, n, sr in _CELL.findall(body):
                out.setdefault((arm, step, sel, split), {})[int(n)] = float(sr)
    return out


def table(cells, arm, selection, split):
    head = '| step | ' + ' | '.join(f'n={n}' for n in NS) + ' |'
    lines = [head, '|---:|' + '---:|' * len(NS)]
    for step in STEPS:
        row = cells.get((arm, step, selection, split))
        if not row:
            continue
        lines.append(f'| {step:,} | '
                     + ' | '.join(f'{row[n]:.2f}' if n in row else '–' for n in NS) + ' |')
    return '\n'.join(lines) if len(lines) > 2 else '_no checkpoints evaluated yet_'


def val_loss_curve(run):
    """(min_val, min_step, final_val, final_step) from the trainer's logs.json.txt.

    val_loss is computed under the CANONICAL objective -- uniform weights, plain L2, via
    compute_loss(slot_weighting=False) -- precisely so it does not move with the weighting
    being ablated. That makes it the one number comparable across these arms even when the
    verifier differs, and it is what says whether a weighting broke the fit.
    """
    p = run / 'logs.json.txt'
    if not p.exists():
        return None
    v = []
    for line in p.read_text().splitlines():
        if '"val_loss"' not in line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get('global_step') is not None:
            v.append((int(d['global_step']), float(d['val_loss'])))
    if not v:
        return None
    ms, mv = min(v, key=lambda x: x[1])
    return mv, ms, v[-1][1], v[-1][0]


def written_steps(run):
    d = run / 'checkpoints'
    if not d.is_dir():
        return []
    return sorted(int(p.name[len('step_'):-len('.ckpt')]) for p in d.glob('step_*.ckpt'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='SUCCESS_RATES_LINEAR_WEIGHTS_DRUMKIT.md')
    ap.add_argument('--log', action='append', default=None,
                    help='driver log to parse; repeatable, later logs win')
    args = ap.parse_args()
    cells = read_cells(args.log or LOGS)

    L = []
    L.append('# ST-diffusion k=16 with linear slot weights — success rates (drumkit)\n')
    L.append('One arm and its control: the 30-demo search transformer at 4/4/256, trained '
             'to 100k gradient steps with a checkpoint every 10k, where the K=16 candidate '
             'slots carry **linear** rather than uniform loss weights. Every checkpoint is '
             'swept over `n = 1, 2, 4, 8, 16, 32, 64` under all three selection rules, on '
             'the same 50 held-out test episodes.\n')
    L.append('Launched by `scripts/run_st_k16_linear_drumkit.sh`. Regenerate this doc with '
             '`python scripts/build_linear_weights_doc.py`.\n')
    L.append('**Source is the driver log, not `success_curves.jsonl`.** '
             '`eval_search_pusht.py` keys its curve on the checkpoint rather than on '
             '(checkpoint, selection), so one checkpoint\'s three selection rules overwrite '
             'each other on disk and only the last to run — `final_pass` — survives, '
             'wearing all seven n values. The pre-existing `search` arm shows the same '
             'collapse (10 rows, all `final_pass`). The per-n numbers are printed to stdout '
             'for all three rules, so the log is the only complete record; '
             '`scripts/bon_grid_table.py` parses it for the same reason.\n')

    L.append('## What is being varied\n')
    L.append('One forward decodes all K=16 candidate slots against the same expert action, '
             'and the staircase memory mask lets slot k attend to exactly the first k '
             'scored candidates — so the model fits a family of conditionals from *no '
             'context* (slot 0) to *15 scored candidates* (slot 15). `slot_weights` is the '
             'SCALE of each of those slots\' loss terms, always renormalized to mean 1 so '
             'switching profiles cannot move the loss scale that `gradient_clip_norm` and '
             'the effective step size both read.\n')
    L.append('`mode: linear` makes the weight affine in the slot index, '
             '`w_k ∝ 1 + (ratio-1)·k/(K-1)`:\n')
    L.append('| slot k | ' + ' | '.join(str(i) for i in range(16)) + ' |')
    L.append('|---|' + '---|' * 16)
    L.append('| w_k | ' + ' | '.join(f'{w:.3f}' for w in LINEAR_W) + ' |\n')
    L.append('**Why ratio 4.857.** `linear` has no default — the resolver raises without a '
             'ratio — so the number is a choice. 4.857 = `0.9^-(K-1)` at K=16, which is '
             'exactly the endpoint spread of the legacy `slot_weight_decay: 0.9` geometric '
             'profile (`w_last/w_first` = 4.857 for both). Holding the spread fixed is what '
             'isolates **curvature** — geometric front-loads its down-weighting into the '
             'low-context slots, linear spreads it evenly — from the much larger effect of '
             'simply tilting harder. Any other ratio confounds the two.\n')
    L.append('`val` stays `uniform`, so `val_loss` is computed under the canonical objective '
             '(uniform weights, plain L2) and remains a fixed cross-arm yardstick instead of '
             'moving with the weighting. `slot_loss_norm` stays `l2`. The two knobs are '
             'orthogonal and only the first is exercised here.\n')

    L.append('### The caveat this arm runs into\n')
    L.append('Under `argmax` **every slot is deployed** — all n candidates come from slots '
             '0..K-1 and the executed action is the best of them. The objective is a good '
             '*max over the pool*, not a good final conditional, so up-weighting the '
             'high-context slots may be the wrong direction there; the last-slot-heavy '
             'argument is a `final_pass` argument, where slot K-1 *is* the deployment '
             'condition. That is why the grid sweeps all three rules rather than argmax '
             'alone — the columns should not be expected to move together.\n')

    L.append('## Comparability — read this before using the numbers\n')
    L.append('`verifier_tag` is **armTn** here, and the slot-weight code path requires it. '
             'The 4/4/256 uniform k=16 run these tables would naturally be read against — '
             '`outer_inner/value_k16_corrupt-False_demos-30_seed-42`, the `search` arm of '
             '`SUCCESS_RATES_30_100.md` — carries no `verifier_value` in its '
             '`.hydra/config.yaml` at all: it predates the cutover and trained under '
             '`t_goal`. That is a different scoring rule, and it does not only rescore at '
             'eval — it feeds the search context the model conditions on during training. '
             '`pusht_base.yaml` states outright that runs across the two are not comparable, '
             'which is why `ver-` is in `run_name`.\n')
    L.append('**So do not diff these tables against `SUCCESS_RATES_30_100.md`.** The valid '
             'control is the uniform armTn arm below; until it is trained '
             '(`bash scripts/run_st_k16_linear_drumkit.sh --uniform-control`) the linear '
             'numbers describe one arm in isolation and carry no claim about the weighting.\n')
    L.append('**The overfitting below is the 30-demo regime, not something the weighting '
             'did.** The uniform `t_goal` k=16 arm on the same 30 demos runs the same curve '
             '— `val_loss` min 0.1029 at step 4,096 rising to 0.3174 by 99,328, 3.08x — '
             'against the linear arm\'s 0.1006 → 0.2850, 2.83x. Near-identical shape, '
             'marginally *less* drift under linear weights. `val_loss` is computed under the '
             'canonical objective in both, so it is comparable across them even though '
             'their verifiers are not.\n')
    L.append('At 50 test episodes a single cell carries a 95% CI of roughly ±0.13 near 0.5, '
             'so **cell-to-cell differences under ~0.15 are not separable**; read down a '
             'column or across several checkpoints, never one cell.\n')

    for arm_key, run, heading, gloss in ARMS:
        have = {k for k in cells if k[0] == arm_key}
        ck = written_steps(run)
        L.append(f'## {heading}\n')
        L.append(f'`outer_inner/{run.name}` → log arm `{arm_key}` — {gloss}\n')
        if not ck and not have:
            L.append('_Not trained yet — no checkpoints on disk and no evaluated cells._\n')
            continue
        full = sum(1 for s in STEPS
                   if set(NS) <= set(cells.get((arm_key, s, 'final_pass', 'test'), {})))
        L.append(f'_{len(ck)}/10 checkpoints written; {full}/10 fully swept '
                 f'(all three rules, n=1..64)._\n')
        vl = val_loss_curve(run)
        if vl:
            mv, ms, fv, fs = vl
            L.append(f'**Fit.** `val_loss` bottoms at **{mv:.4f}** by step {ms:,} and rises '
                     f'to **{fv:.4f}** by {fs:,} — {fv/mv:.2f}x off its minimum. Read the '
                     f'step rows below with that in mind: the late checkpoints are more '
                     f'overfit than the early ones, and a success rate that falls down the '
                     f'column is the expected shape here, not a surprise.\n')
        for sel in SELECTIONS:
            L.append(f'### Test success rate — `{sel}`\n')
            L.append(table(cells, arm_key, sel, 'test') + '\n')
        L.append('### Val success rate — `argmax` (30 episodes)\n')
        L.append(table(cells, arm_key, 'argmax', 'val') + '\n')
        L.append('Val is the split a checkpoint and an n may honestly be chosen on; the '
                 'test tables above are the read-off, not the search space.\n')

    pathlib.Path(args.out).write_text('\n'.join(L) + '\n')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
