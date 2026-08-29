"""Which checkpoint each slot-weight arm is analysed at, under one rule fixed in advance.

THE RULE: highest `final_pass` success rate at n=16; on a tie, the LATER checkpoint wins.

Why final_pass at n=16. Under `selection: final_pass` the deployed sample is the one
conditioned on every searched candidate, and the deployed slot is `min(n, K-1)` -- which at
n=16 with K=16 is slot 15. So this rule selects each arm at the checkpoint where slot 15
deploys best, i.e. on the read-out that most directly depends on the quantity
scripts/slot_context_analysis.py goes on to measure.

That is deliberate but it IS mildly circular, and the circularity is the reason the analysis
carries a second tier: step 10000, chosen by no criterion at all, identical across arms.
Tier A carries any cross-arm claim; this tier is a within-arm check only, because the steps
this script returns differ per arm and are therefore confounded with training duration.

The numbers are noisy enough that the rule matters more than its output: final_pass at n=16
spans ~0.06-0.34 over 50 episodes, so the Wilson intervals printed beside each pick overlap
almost everything. Fixing the rule in advance is what keeps this from being a search over
the test set; it does not make the winner reproducible.

    python scripts/pick_final_pass_step.py                 # the table
    python scripts/pick_final_pass_step.py --sbatch-line   # paste into STEP_B=(...)

Re-run it after training finishes and paste the new STEP_B into
scripts/slurm/dump_slot_wts_grid.sbatch; that is the whole of the post-training refresh.
"""
import json
import os
import pathlib

import click

# stdlib-only module on purpose: importing eval_search_pusht (which also re-exports
# this) costs ~88 s and ~440 MB of torch+hydra, for a script that reads JSON.
from diffusion_policy.common.stats_util import wilson_interval

ROOT = pathlib.Path(os.environ.get(
    'DP_OUTPUT_ROOT', '/gscratch/robotics/harine/diffusion_policy_outputs'))
BASE = ROOT / 'pusht_search' / 'pusht_image_search'
_K16 = 'outer_inner/value_k16_ver-armTn%s_corrupt-False_demos-30_seed-42'

# (label, run dir, final_pass subdir). unetbc's armTn read-outs live under a _ver-armTn
# suffix because it is scored with an override -- it was trained before the verifier
# cutover and is natively t_goal, so its plain bon_search_sel-final_pass/ is a DIFFERENT
# scoring rule and must not be read here.
ARMS = [
    ('lin100-l2',          _K16 % '_sw-lin100-l2',          'bon_search_sel-final_pass'),
    ('lin100-l2tol1',      _K16 % '_sw-lin100-l2tol1',      'bon_search_sel-final_pass'),
    ('geo735-l2',          _K16 % '_sw-geo735-l2',          'bon_search_sel-final_pass'),
    ('geo735-l2tol1',      _K16 % '_sw-geo735-l2tol1',      'bon_search_sel-final_pass'),
    ('curr-lin100-l2',     _K16 % '_sw-curr-lin100-l2',     'bon_search_sel-final_pass'),
    ('curr-lin100-l2tol1', _K16 % '_sw-curr-lin100-l2tol1', 'bon_search_sel-final_pass'),
    ('stk1',   'offline/value_k1_ver-armTn_corrupt-False_demos-30_seed-42',
     'bon_search_sel-final_pass'),
    ('unetbc', 'unet_bc/unetbc_demos-30_seed-42',
     'bon_search_sel-final_pass_ver-armTn'),
]
# stk1 has no final_pass read-outs at all, so the rule is undefined for it. It takes the
# modal pick of the other arms rather than a hardcoded constant, so a post-training refresh
# moves it along with everything else instead of silently freezing at today's answer.
FALLBACK_ARMS = {'stk1'}
TARGET_N = 16


def read_final_pass(run_dir, sub, target_n=TARGET_N):
    """{step: (rate, n_episodes)} from a run's final_pass sweep, at one search width.

    Reads step_XXXXXXX/success_curve.json, NOT the run-level success_curves.jsonl.
    The jsonl is missing `verifier_value` on every row written before that key entered
    _row_from_curve's whitelist -- on the k=16 control it reads null for steps 10k-50k and
    'armTn' for 60k-100k, same run, same sweep. Filtering the jsonl on the verifier (which
    the docs' warning about mixed verifiers would suggest) therefore silently drops exactly
    the early checkpoints. The per-step JSON records it correctly for all ten.
    """
    out = {}
    d = pathlib.Path(run_dir) / sub
    if not d.is_dir():
        return out
    for f in sorted(d.glob('step_*/success_curve.json')):
        try:
            c = json.loads(f.read_text())
        except json.JSONDecodeError:      # a sweep mid-write; skip rather than crash
            continue
        ns, sr = c.get('n') or [], c.get('success_rate') or []
        if target_n not in ns or len(sr) != len(ns):
            continue
        step = int(f.parent.name.split('_')[1])
        out[step] = (float(sr[ns.index(target_n)]), int(c.get('n_episodes') or 0))
    return out


def pick(cells):
    """(step, rate, n_ep) maximising rate, LATER step breaking ties. None if no cells.

    max() over (rate, step) does both at once: tuples compare left-to-right, so the step
    only ever decides among equal rates.
    """
    if not cells:
        return None
    step = max(cells, key=lambda s: (cells[s][0], s))
    return (step,) + cells[step]


@click.command()
@click.option('--n', 'target_n', default=TARGET_N, show_default=True,
              help='search width the rule reads at.')
@click.option('--sbatch-line', is_flag=True,
              help='emit just the STEP_B=(...) line for dump_slot_wts_grid.sbatch.')
def main(target_n, sbatch_line):
    picks, rows = {}, []
    for label, run, sub in ARMS:
        cells = read_final_pass(BASE / run, sub, target_n)
        rows.append((label, sub, cells, pick(cells)))
        if label not in FALLBACK_ARMS:
            p = pick(cells)
            if p is not None:
                picks[label] = p[0]

    modal = max(set(picks.values()), key=list(picks.values()).count) if picks else 10000

    if not sbatch_line:
        print(f'Rule: highest final_pass success at n={target_n}; later checkpoint breaks '
              f'ties.\nSource: <run>/<sub>/step_*/success_curve.json\n')
        print(f"{'arm':22s} {'step':>7s} {'rate':>6s} {'95% Wilson':>14s} {'eps':>4s}  "
              f"{'all steps (k:rate)'}")
    out = []
    for label, sub, cells, p in rows:
        if label in FALLBACK_ARMS or p is None:
            step, rate, nep, note = modal, None, None, 'no final_pass read-outs -> modal pick'
        else:
            step, rate, nep = p
            note = ''
        out.append(step)
        if sbatch_line:
            continue
        ci = ('%.2f-%.2f' % wilson_interval(int(round((rate or 0) * (nep or 0))), nep or 0)
              if rate is not None else '-')
        series = ' '.join(f'{s//1000}k:{v[0]:.2f}' for s, v in sorted(cells.items())) or '-'
        print(f'{label:22s} {step:>7d} {("%.2f" % rate) if rate is not None else "-":>6s} '
              f'{ci:>14s} {nep if nep else "-":>4}  {series}{"  <- " + note if note else ""}')

    print(('' if sbatch_line else '\n') + 'STEP_B=(' + ' '.join(str(s) for s in out) + ')')
    if not sbatch_line:
        print('\nPaste that into scripts/slurm/dump_slot_wts_grid.sbatch.')
        print('Intervals overlap heavily at 50 episodes -- the rule being fixed in advance '
              'is what makes this defensible, not the margin between arms.')


if __name__ == '__main__':
    main()
