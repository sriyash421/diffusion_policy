"""Verify the aug26 slot-weight artifacts on disk, and assert the nulls behave.

Modelled on scripts/check_aug19_artifacts.py. The write-up is provenance: it claims a set
of files exist and were produced a particular way, and nothing keeps that true as array
tasks are preempted or partially completed.

The last check is the one that matters most and is not merely bookkeeping:

  THE I.I.D. ARMS MUST COME BACK FLAT. stk1 and unetbc cannot encode slot order by
  construction. If either shows a slot trend or a non-zero excess_records, the pipeline is
  measuring an artifact and the stk16 numbers are not trustworthy -- the null is the
  instrument check, not a baseline to beat.

  python scripts/check_slot_wts_artifacts.py
"""
import json
import pathlib
import sys

import numpy as np

DUMP_DIR = pathlib.Path('analysis/aug26_slot_wts')
ARMS = ('stk16_l2tol1', 'stk16_lin4857', 'stk16_uniform', 'stk1', 'unetbc')
STEPS = (10000, 30000, 100000)
IID = {'stk1': 'iid_no_context', 'unetbc': 'iid_unbounded'}
# Tolerances for "flat". Deliberately loose: these are 50 finite episodes, so an i.i.d. arm
# will not return exactly zero, and a tight bound here would fail on noise rather than on a
# real defect.
MAX_IID_EXCESS = 0.5        # excess_records summed over k>=1
MAX_IID_SLOPE = 0.05        # |OLS slope| of centered value vs slot index


def main():
    fail, warn = [], []
    verifiers = {}
    print(f'{"arm":<16}{"step":>8}  {"slot semantics":<20}{"steps":>7}{"excess":>9}'
          f'{"slope":>9}  {"succ":>5}  files')
    for arm in ARMS:
        for step in STEPS:
            d = DUMP_DIR / f'{arm}_step{step}'
            if not d.is_dir():
                print(f'{arm:<16}{step:>8}  MISSING DIR')
                fail.append(f'{arm}@{step}: no dir')
                continue
            have = {f: (d / f).is_file() for f in (
                'candidate_scores.npz', 'candidate_scores_meta.json',
                'candidate_scores_stats.json')}
            if not all(have.values()):
                print(f'{arm:<16}{step:>8}  INCOMPLETE {have}')
                fail.append(f'{arm}@{step}: {[k for k,v in have.items() if not v]}')
                continue
            st = json.loads((d / 'candidate_scores_stats.json').read_text())
            m = json.loads((d / 'candidate_scores_meta.json').read_text())
            per_step = len(list((d / 'per_step').glob('ep*.json'))) if (d / 'per_step').is_dir() else 0
            print(f'{arm:<16}{step:>8}  {m["slot_semantics"]:<20}'
                  f'{st["n_control_steps"]:>7}{st["excess_records"]:>+9.2f}'
                  f'{st["ols_slope"]:>+9.4f}  {m["success_rate"]:>5.0%}  '
                  f'npz+meta+stats, {per_step} per-step')

            # the invariants every cell must satisfy
            if m['n'] != 16:
                fail.append(f'{arm}@{step}: n={m["n"]} != 16')
            if m['split'] != 'test' or len(m['episode_idxs']) != 50:
                fail.append(f'{arm}@{step}: split={m["split"]} '
                            f'episodes={len(m["episode_idxs"])} (want test/50)')
            if not m.get('paired_seeds'):
                fail.append(f'{arm}@{step}: paired_seeds is False -- not comparable')
            verifiers.setdefault(m['verifier_value'], []).append(f'{arm}@{step}')
            # ALL FIVE ARMS MUST SHARE ONE VALUE FUNCTION. t_goal is flat across candidates
            # until the arm touches the block, so an arm scored under it would have a
            # mechanically flatter slot profile than the armTn arms whether or not it uses
            # context -- the null would look like a null for the wrong reason. unetbc
            # predates the cutover and is re-scored with --verifier-value armTn; its native
            # dumps are kept as `unetbc-native-tgoal_step*` for contrast.
            if m['verifier_value'] != 'armTn':
                fail.append(f'{arm}@{step}: scored with {m["verifier_value"]}, not armTn')
            if m.get('verifier_value_overridden'):
                print(f'{"":<16}{"":>8}  ^ scored under armTn by override; checkpoint\'s '
                      f'own value is {m.get("verifier_value_native")}')

            # THE INSTRUMENT CHECK
            if arm in IID:
                if m['slot_semantics'] != IID[arm]:
                    fail.append(f'{arm}@{step}: slot_semantics {m["slot_semantics"]} '
                                f'!= {IID[arm]}')
                if abs(st['excess_records']) > MAX_IID_EXCESS:
                    fail.append(f'{arm}@{step}: i.i.d. arm has excess_records '
                                f'{st["excess_records"]:+.2f} (|.| > {MAX_IID_EXCESS}) -- '
                                f'the analysis is measuring an artifact')
                if abs(st['ols_slope']) > MAX_IID_SLOPE:
                    warn.append(f'{arm}@{step}: i.i.d. arm slot slope '
                                f'{st["ols_slope"]:+.4f} (|.| > {MAX_IID_SLOPE})')
            elif m['slot_semantics'] != 'trained_staircase':
                fail.append(f'{arm}@{step}: expected trained_staircase, '
                            f'got {m["slot_semantics"]}')

    # every arm must have scored the SAME 50 episodes, or nothing is paired
    seen = {}
    for arm in ARMS:
        for step in STEPS:
            p = DUMP_DIR / f'{arm}_step{step}' / 'candidate_scores_meta.json'
            if p.is_file():
                seen[f'{arm}@{step}'] = tuple(json.loads(p.read_text())['episode_idxs'])
    if len(set(seen.values())) > 1:
        fail.append(f'episode sets differ across cells: {len(set(seen.values()))} distinct')
    elif seen:
        print(f'\nall {len(seen)} cells scored the same 50 test episodes')

    # VERIFIER VALUE IS A COMPARABILITY BOUNDARY, not a pass/fail property of a file.
    #
    # The scoring rule changed on 2026-08-19 (t_goal -> armT/armTn, adding an arm-to-T
    # approach term) and resolved_verifier_value's own docstring records that the two are
    # NOT comparable: t_goal is FLAT across candidates until the arm touches the block,
    # armTn is not. Every artifact is scored on what its checkpoint was trained with, which
    # is correct -- but it means arms in different groups below are in different units, and
    # an "arm minus null" subtraction across the boundary is meaningless.
    print()
    if len(verifiers) > 1:
        print('VERIFIER GROUPS -- magnitudes are NOT comparable across these:')
        for v, cells in sorted(verifiers.items()):
            arms = sorted({c.split('@')[0] for c in cells})
            print(f'  {v:<8} {", ".join(arms)}')
        warn.append(
            f'{len(verifiers)} verifier values present ({", ".join(sorted(verifiers))}); '
            f'only same-group arms may be subtracted from one another')
    elif verifiers:
        print(f'all cells scored with verifier_value={next(iter(verifiers))}')

    for w in warn:
        print(f'WARN  {w}')
    print('\nRESULT:', 'PASS' if not fail else 'FAIL')
    for f in fail:
        print(f'  FAIL  {f}')
    return 0 if not fail else 1


if __name__ == '__main__':
    sys.exit(main())
