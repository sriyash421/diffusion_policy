"""Collect the `q_fn_sac_all_206_demos` ST sweep into one table. Read-only, re-runnable.

One row per (arm, selection, step), reading each run's own
`bon_search_sel-<sel>_ver-q_sac_all/success_curves.jsonl` -- the file
`eval_search_pusht.py` appends to under flock, so this is safe to run mid-sweep.

REPORTS SUCCESS **AND** MEAN REWARD -- and `mean_reward` is NOT coverage, which is the trap.
`pusht_env.py:133` defines `reward = clip(coverage / 0.95, 0, 1)` and success is `coverage >
0.95`, so reward is an exact affine image of coverage BELOW the threshold and saturates at 1.0
above it. It is therefore right-censored exactly on the solved episodes. Below threshold,
`coverage = reward * 0.95`; at 1.0 all you know is "solved".

Quoting success alone still hides the case where search moves the block most of the way and
stops, which is what these columns are for -- but call them reward, not coverage.
`eval_search_pusht` does not persist raw coverage in the curve row (it prints it and drops it),
so raw coverage for these arms would need a re-run. `sac/eval.py bon-sweep` DOES persist it as
`mean_max_coverage`, which is why the BC rows can show the uncensored quantity.

NOTHING HERE NOMINATES A CHECKPOINT. Every step measured is printed; picking one is the
reader's job, and `--step` exists to quote a step chosen elsewhere, not to find a best one.

    python scripts/q_st_sweep_status.py                 # everything, latest step per arm
    python scripts/q_st_sweep_status.py --all-steps     # the whole 10k grid
    python scripts/q_st_sweep_status.py --step 100000   # one step, both selections
"""
import argparse
import json
import os
import pathlib
import re

ROOT = pathlib.Path(os.environ.get(
    'DP_OUTPUT_ROOT', '/gscratch/robotics/harine/diffusion_policy_outputs'))
BASE = ROOT / 'pusht_search' / 'pusht_image_search_imgonly' / 'outer_inner'
SELECTIONS = ('argmax', 'final_pass')


def arm_label(run: str) -> str:
    """The parts that vary between arms; the rest is constant across this generation."""
    s = re.sub(r'^value_|_ver-q_sac_all|_enc-resnet18|_seed-42$|_demos-\d+', '', run)
    return s.replace('_son-', ' ').replace('_split-', ' ').strip()


def rows_for(run_dir: pathlib.Path, sel: str):
    """-> {step: {n: (success, ci, cov_max, cov_final)}} for one (arm, selection)."""
    # `_bon_subdir` appends `_ver-<value>` ONLY when --verifier-value was passed as an
    # override. These arms carry `verifier_tag=q_sac_all` natively, so nothing is overridden
    # and the directory is the plain `bon_search_sel-<sel>`. Both spellings are accepted:
    # a run swept with an explicit override would land in the longer one.
    for cand in (f'bon_search_sel-{sel}', f'bon_search_sel-{sel}_ver-q_sac_all'):
        path = run_dir / cand / 'success_curves.jsonl'
        if path.exists():
            break
    else:
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue                      # a torn line from a concurrent append; skip it
        # later rows for a step supersede earlier ones (a requeue re-evaluates)
        out[int(r['step'])] = {
            int(n): (r['success_rate'][i],
                     (r.get('success_ci') or [[None, None]] * len(r['n']))[i],
                     (r.get('mean_reward') or [None] * len(r['n']))[i],
                     (r.get('mean_reward_final') or [None] * len(r['n']))[i],
                     # raw coverage: persisted from 2026-09-21, absent before
                     (r.get('mean_coverage') or [None] * len(r['n']))[i])
            for i, n in enumerate(r['n'])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all-steps', action='store_true')
    ap.add_argument('--step', type=int, default=None)
    ap.add_argument('--metric',
                    choices=['success', 'coverage', 'reward_max', 'reward_final'],
                    default='success')
    args = ap.parse_args()

    idx = {'success': 0, 'reward_max': 2, 'reward_final': 3, 'coverage': 4}[args.metric]
    arms = sorted(BASE.glob('*ver-q_sac_all*'))
    if not arms:
        raise SystemExit(f'no q_sac_all arms under {BASE}')

    ns_seen, table = set(), []
    for d in arms:
        for sel in SELECTIONS:
            data = rows_for(d, sel)
            if not data:
                continue
            steps = sorted(data)
            if args.step is not None:
                steps = [s for s in steps if s == args.step]
            elif not args.all_steps:
                steps = steps[-1:]
            for s in steps:
                ns_seen.update(data[s])
                table.append((arm_label(d.name), sel, s, data[s]))

    if not table:
        raise SystemExit('no curves yet -- the sweep has not written any rows')

    ns = sorted(ns_seen)
    head = f"{'arm':<26} {'sel':<11} {'step':>7}" + ''.join(f"{('n='+str(n)):>9}" for n in ns)
    print(f'metric: {args.metric}   (success = coverage > 0.95; '
          f'reward_* = clip(coverage/0.95, 0, 1), censored at 1.0)\n')
    print(head)
    print('-' * len(head))
    for label, sel, step, per_n in sorted(table):
        cells = ''
        for n in ns:
            v = per_n.get(n, (None,) * 4)[idx]
            cells += f'{v:>9.3f}' if v is not None else f"{'-':>9}"
        print(f'{label:<26} {sel:<11} {step:>7}{cells}')

    done = sum(1 for _, _, _, _ in table)
    print(f'\n{done} rows  ·  {len(arms)} arms  ·  '
          f'{len({(l, s) for l, s, _, _ in table})} of {len(arms) * len(SELECTIONS)} '
          f'(arm, selection) pairs have written at least one step')


if __name__ == '__main__':
    main()
