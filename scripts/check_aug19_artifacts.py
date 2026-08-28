"""Verify the aug19 artifacts on disk and emit the status tables for aug19_analysis.md.

The doc is provenance: it claims a set of files exist and were produced a particular way.
Nothing keeps that claim true as jobs are re-run, preempted or partially completed, so this
checks it rather than trusting it -- and prints the §2a / §2c tables filled in from what is
actually on disk, so the doc is transcribed from the artifacts rather than from memory.

Checks, in the order they can fail:
  1. every (label, step, n) cell has the expected number of mp4s, and each has a sidecar
  2. the two arms rendered the SAME episodes (the pairing claim in §4f)
  3. every dump dir has its npz / meta / stats / per_step, and the per-step JSON validates
     against the v1 schema
  4. every artifact records the same verifier_value, and it is the one the doc claims
  5. the one episode designated for cross-arm comparison exists in all three arms

  python scripts/check_aug19_artifacts.py
  python scripts/check_aug19_artifacts.py --tables      # markdown only, for pasting
"""
if __name__ == "__main__":
    import sys, os, pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import json
import pathlib
import re
import sys

import click

VIDEO_DIR = pathlib.Path('videos/aug19_bcvSTk1_actions_viz')
DUMP_DIR = pathlib.Path('analysis/aug19_candidate_scores')
VIDEO_ARMS = ('unetbc', 'stk1')
DUMP_ARMS = ('unetbc', 'stk1', 'stk16')
STEPS = (10000, 50000, 100000)
NS = (1, 2, 8, 16, 64)
N_EPISODES = 10
EXPECT_VERIFIER = 't_goal'


def _sidecars():
    return {(m.group(1), int(m.group(2)), int(m.group(3))): p
            for p in sorted(VIDEO_DIR.glob('episodes_*.json'))
            for m in [re.match(r'episodes_(.+)_step(\d+)_n(\d+)\.json$', p.name)] if m}


def check_videos(problems):
    print('## 2a. Videos\n')
    if not VIDEO_DIR.is_dir():
        problems.append(f'{VIDEO_DIR} does not exist')
        print('  (no video directory yet)\n')
        return
    side = _sidecars()
    mp4 = sorted(VIDEO_DIR.glob('*.mp4'))
    print(f'| label | step | ' + ' | '.join(f'n={n}' for n in NS) + ' |')
    print('|---|---|' + '---|' * len(NS))
    episodes_by_arm = {}
    for arm in VIDEO_ARMS:
        for step in STEPS:
            cells = []
            for n in NS:
                got = len([p for p in mp4 if re.match(
                    rf'seed\d+_n{n}_step{step}_{arm}_(succ|trunc)\.mp4$', p.name)])
                sc = side.get((arm, step, n))
                if sc is None and got == 0:
                    cells.append('pending')
                    continue
                if sc is None:
                    cells.append(f'{got}/{N_EPISODES} NO SIDECAR')
                    problems.append(f'{arm} step{step} n={n}: {got} mp4s but no sidecar')
                    continue
                recs = json.loads(sc.read_text())
                episodes_by_arm.setdefault((step, n), {})[arm] = [
                    r['episode_idx'] for r in recs]
                bad = [r for r in recs if not (VIDEO_DIR / r['video']).is_file()]
                if bad:
                    problems.append(f'{arm} step{step} n={n}: sidecar names '
                                    f'{len(bad)} missing file(s), e.g. {bad[0]["video"]}')
                vv = {r.get('verifier_value') for r in recs}
                if vv != {EXPECT_VERIFIER}:
                    problems.append(f'{arm} step{step} n={n}: verifier_value {vv}, '
                                    f'expected {{{EXPECT_VERIFIER!r}}}')
                if got != N_EPISODES:
                    problems.append(f'{arm} step{step} n={n}: {got} mp4s, '
                                    f'expected {N_EPISODES}')
                blind = sum(r['n_blind'] for r in recs)
                dec = sum(r['n_decisions'] for r in recs)
                cells.append(f'{got}/{N_EPISODES}'
                             + (f' ({blind}/{dec} blind)' if dec else ''))
            print(f'| {arm} | {step} | ' + ' | '.join(cells) + ' |')
    print(f'\n{len(mp4)} mp4s, {len(side)} sidecars '
          f'(expect {len(VIDEO_ARMS)*len(STEPS)*len(NS)*N_EPISODES} and '
          f'{len(VIDEO_ARMS)*len(STEPS)*len(NS)}).\n')

    # THE PAIRING CLAIM: both arms must have rendered the same episodes, or the videos are
    # not comparable side by side no matter how identical the seeds were.
    for (step, n), by_arm in sorted(episodes_by_arm.items()):
        if len(by_arm) == len(VIDEO_ARMS):
            vals = list(by_arm.values())
            if any(v != vals[0] for v in vals[1:]):
                problems.append(f'step{step} n={n}: arms rendered DIFFERENT episodes '
                                f'{ {a: v[:3] for a, v in by_arm.items()} }')


def check_dumps(problems):
    print('## 2c. Candidate-score dumps\n')
    if not DUMP_DIR.is_dir():
        print('  (no dump directory yet)\n')
        return
    print('| label | step | control steps | success | blind | in-lead | '
          'argmax unique | argmax first | perm null | verifier |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    per_step_idxs = {}
    for arm in DUMP_ARMS:
        for step in STEPS:
            d = DUMP_DIR / f'{arm}_step{step:07d}'
            sf = d / 'candidate_scores_stats.json'
            if not sf.is_file():
                print(f'| {arm} | {step} | pending | | | | | | | |')
                continue
            st = json.loads(sf.read_text())
            for req in ('candidate_scores.npz', 'candidate_scores_meta.json'):
                if not (d / req).is_file():
                    problems.append(f'{d.name}: missing {req}')
            if st.get('verifier_value') != EXPECT_VERIFIER:
                problems.append(f'{d.name}: verifier_value '
                                f'{st.get("verifier_value")!r} != {EXPECT_VERIFIER!r}')
            if not st.get('paired_seeds'):
                problems.append(f'{d.name}: paired_seeds is false; it cannot be compared '
                                f'with the paired dumps')
            ps = sorted((d / 'per_step').glob('*.json')) if (d / 'per_step').is_dir() else []
            if not ps:
                problems.append(f'{d.name}: no per_step/ JSON')
            per_step_idxs[(arm, step)] = [
                int(re.search(r'idx(\d+)', p.name).group(1)) for p in ps]
            for p in ps[:3]:                      # schema-check a sample, not all 450
                _check_per_step(p, problems)
            print(f"| {arm} | {step} | {st['n_control_steps']} | "
                  f"{st['success_rate']:.2f} | {st['blind_rate_overall']:.1%} | "
                  f"{st['frac_blind_in_leading_run']:.0%} | "
                  f"{st.get('argmax_slot_mean_unique', float('nan')):.2f} | "
                  f"{st.get('argmax_slot_mean_first', float('nan')):.2f} | "
                  f"{st.get('argmax_slot_perm_null', float('nan')):.2f} | "
                  f"{st.get('verifier_value')} |")
    print()
    # the cross-arm episode: only meaningful if all three arms dumped the same one
    common = None
    for v in per_step_idxs.values():
        common = set(v) if common is None else (common & set(v))
    if per_step_idxs and common:
        print(f'Episodes present in every dumped arm: {sorted(common)[:10]}'
              f'{" ..." if len(common) > 10 else ""}\n')
    elif per_step_idxs:
        problems.append('no episode_idx is common to every dumped arm, so §2b has no '
                        'single episode the three arms can be compared on')


_REQUIRED_STEP_KEYS = ('t', 'env_step', 'alive', 'value', 'argmax_first',
                       'argmax_last', 'n_argmax_tied', 'argmax_margin_px', 'degenerate',
                       'executed', 'executed_value', 'final_slot_value', 'spread')

# The negated-value key was renamed in v2: `distance_px` was a misnomer under any value but
# `t_goal` (armTn's negation is a spread-normalized composite of two distances, not pixels).
# Both schemas are accepted because the aug19 dumps on disk are v1 and are not being
# rewritten -- a checker that only knew the new name would report every existing artifact
# as broken.
_NEG_VALUE_KEY = {'pusht_candidate_values/v1': 'distance_px',
                  'pusht_candidate_values/v2': 'neg_value'}


def _check_per_step(path, problems):
    doc = json.loads(path.read_text())
    where = f'{path.parent.parent.name}/{path.name}'
    neg_key = _NEG_VALUE_KEY.get(doc.get('schema'))
    if neg_key is None:
        problems.append(f'{where}: schema {doc.get("schema")!r}')
        return
    n = doc['n']
    for s in doc['steps']:
        missing = [k for k in _REQUIRED_STEP_KEYS + (neg_key,) if k not in s]
        if missing:
            problems.append(f'{where} t={s.get("t")}: missing {missing}')
            return
        if len(s['value']) != n or len(s[neg_key]) != n:
            problems.append(f'{where} t={s["t"]}: {len(s["value"])} values, expected {n}')
            return
        if any(abs(a + b) > 1e-6 for a, b in zip(s['value'], s[neg_key])):
            problems.append(f'{where} t={s["t"]}: {neg_key} != -value')
            return
        if s['argmax_first'] > s['argmax_last']:
            problems.append(f'{where} t={s["t"]}: argmax_first > argmax_last')
            return
        # first == last is exactly the unique-maximizer case; anything else means the tie
        # count and the indices disagree, and the argmax-slot statistic would be wrong
        if (s['n_argmax_tied'] == 1) != (s['argmax_first'] == s['argmax_last']):
            problems.append(f'{where} t={s["t"]}: n_argmax_tied={s["n_argmax_tied"]} '
                            f'disagrees with first/last')
            return
        if s['degenerate'] and s['n_argmax_tied'] != n:
            problems.append(f'{where} t={s["t"]}: degenerate but only '
                            f'{s["n_argmax_tied"]}/{n} tied')
            return


@click.command()
@click.option('--tables', is_flag=True, help='print only the markdown tables')
def main(tables):
    problems = []
    check_videos(problems)
    check_dumps(problems)
    if tables:
        return
    if problems:
        print(f'{len(problems)} PROBLEM(S):')
        for p in problems:
            print(' -', p)
        raise SystemExit(1)
    print('all aug19 artifacts check out')


if __name__ == '__main__':
    main()
