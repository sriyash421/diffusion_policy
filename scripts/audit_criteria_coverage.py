"""Coverage audit for the selection-criteria sweep.

Answers one question exhaustively: for every (run, checkpoint, criterion) the experiment
asked for, is the result actually on disk, at the right width, over the right number of
episodes, with the per-step trace beside it?

Partial coverage is the failure mode that matters here. A missing row is invisible in a
rendered table -- it looks like a blank cell, indistinguishable from a run that has not got
there yet -- so this reports MISSING and PARTIAL separately from NOT-YET-TRAINED.

Usage: python scripts/audit_criteria_coverage.py
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(
    '/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search'
    '/outer_inner')

# The requested grid.
STEPS = [1000, 5000] + list(range(10000, 100001, 10000))
CRITERIA = ['cand-last', 'cand-8th-from-last', 'argmax-all', 'argmax-last8',
            'softmax-all', 'softmax-last8']
WANT_N = 16          # search width
WANT_EPISODES = 50   # test split
WANT_K = 16          # candidates per control step in the trace

TRACE_KEYS = ('scores', 'chosen_idx', 'step_reward', 'valid_len', 'episode_idxs')
ROW_KEYS = ('success_rate', 'mean_reward', 'n', 'n_episodes', 'criterion', 'step')


def main():
    runs = sorted(ROOT.glob('*_swd-*_demos-100_seed-42'))
    if not runs:
        print('no sweep runs found under', ROOT)
        return 1

    problems, totals = [], dict(done=0, missing=0, untrained=0)
    print(f'grid: {len(STEPS)} checkpoints x {len(CRITERIA)} criteria x {len(runs)} runs '
          f'= {len(STEPS) * len(CRITERIA) * len(runs)} results\n')

    for run in runs:
        name = run.name.replace('_corrupt-False', '').replace('_demos-100_seed-42', '')
        jl = run / 'criteria_search' / 'criteria_curves.jsonl'
        rows = {}
        if jl.is_file():
            for line in jl.read_text().splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rows[(r.get('step'), r.get('criterion'))] = r
        trained = {int(p.stem.split('_')[1])
                   for p in (run / 'checkpoints').glob('step_*.ckpt')}

        line_bits = []
        for step in STEPS:
            have = [c for c in CRITERIA if (step, c) in rows]
            if len(have) == len(CRITERIA):
                totals['done'] += 1
                line_bits.append('#')
            elif step not in trained:
                totals['untrained'] += 1
                line_bits.append('.')
            elif have:
                totals['missing'] += 1
                line_bits.append('!')
                problems.append(f'{name} @ {step}: PARTIAL, only {len(have)}/6 criteria '
                                f'({sorted(set(CRITERIA) - set(have))} missing)')
            else:
                totals['missing'] += 1
                line_bits.append('x')
                problems.append(f'{name} @ {step}: checkpoint EXISTS but no criteria '
                                f'results at all')
        print(f'  {name:<30} {"".join(line_bits)}')

        # field / width / episode-count checks on everything that IS there
        for (step, crit), r in sorted(rows.items()):
            for k in ROW_KEYS:
                if r.get(k) is None:
                    problems.append(f'{name} @ {step}/{crit}: missing field {k!r}')
            if r.get('n') != WANT_N:
                problems.append(f'{name} @ {step}/{crit}: n={r.get("n")} not {WANT_N}')
            if r.get('n_episodes') != WANT_EPISODES:
                problems.append(f'{name} @ {step}/{crit}: '
                                f'{r.get("n_episodes")} episodes not {WANT_EPISODES}')
            t = run / 'criteria_search' / f'step_{step:07d}' / 'traces' / f'{crit}.npz'
            if not t.is_file():
                problems.append(f'{name} @ {step}/{crit}: trace npz missing')
                continue
            z = np.load(t)
            for k in TRACE_KEYS:
                if k not in z:
                    problems.append(f'{name} @ {step}/{crit}: trace lacks {k!r}')
            if 'scores' in z:
                sh = z['scores'].shape
                if sh[0] != WANT_EPISODES or sh[2] != WANT_K:
                    problems.append(f'{name} @ {step}/{crit}: scores shape {sh}, '
                                    f'expected ({WANT_EPISODES}, T, {WANT_K})')

    print('\n  legend: # all 6 criteria   ! partial   x trained-but-unevaluated   '
          '. not trained yet')
    print(f'\ncheckpoints complete : {totals["done"]}')
    print(f'checkpoints missing  : {totals["missing"]}   (trained but not fully evaluated)')
    print(f'not trained yet      : {totals["untrained"]}')

    if problems:
        print(f'\n{len(problems)} PROBLEM(S):')
        for p in problems[:40]:
            print('  -', p)
        if len(problems) > 40:
            print(f'  ... and {len(problems) - 40} more')
    else:
        print('\nNo problems: every result present is at n=16 over 50 episodes, carries '
              'success_rate and mean_reward, and has a (50, T, 16) trace beside it.')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
