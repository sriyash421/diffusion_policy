"""The `q_fn_sac_all_206_demos` headline table: argmax vs final_pass at n = 1…64.

Emits markdown for docs/reports/q_fn_sac_all_206_demos_2026-09-20.md. Read-only.

TWO SOURCES, TWO FORMATS, ONE TABLE -- and the difference is not cosmetic:

  ST arms  `<run>/bon_search_sel-<sel>/success_curves.jsonl`, written by eval_search_pusht.py.
           The ranker is the arm's OWN `verifier_tag=q_sac_all`, rebuilt from its config, so
           there is exactly one ranker per row and no ranker column to print.
  BC arms  `bon_q_geom_n64/unetbc_<split>_step100k_<sel>/bon_curves.json`, written by
           sac/eval.py bon-sweep. BC carries `verifier_tag=t_goal`, so the Q has to be
           installed over it -- which is why BOTH rankers appear on the same episodes, and
           why `t_goal` is the reference line the Q is read against.

    python scripts/q_n64_table.py                    # success
    python scripts/q_n64_table.py --metric cov_max   # mean max coverage
"""
import argparse
import json
import os
import pathlib
import re

ROOT = pathlib.Path(os.environ.get(
    'DP_OUTPUT_ROOT', '/gscratch/robotics/harine/diffusion_policy_outputs'))
ST_BASE = ROOT / 'pusht_search' / 'pusht_image_search_imgonly' / 'outer_inner'
BC_BASE = pathlib.Path('/gscratch/robotics/harine/value_arms/bon_q_geom_n64')
NS = (1, 2, 4, 8, 16, 32, 64)
SELECTIONS = ('argmax', 'final_pass')
# ST rows come from eval_search_pusht, which persists REWARD, not coverage:
# `reward = clip(coverage / 0.95, 0, 1)` (pusht_env.py:133) and success is `coverage > 0.95`,
# so reward is an exact affine image of coverage below the threshold and saturates at 1.0
# above it -- right-censored precisely on the solved episodes. Below threshold
# `coverage = reward * 0.95`; at 1.0 all it says is "solved". It is NOT coverage and is not
# labelled as such anywhere here.
ST_KEY = {'success': 'success_rate', 'reward_max': 'mean_reward',
          'reward_final': 'mean_reward_final',
          # persisted from 2026-09-21; older curves lack it and print as an em dash rather
          # than silently falling back to the censored reward under a coverage heading
          'coverage': 'mean_coverage'}
# BC rows come from sac/eval.py bon-sweep, which persists BOTH: the same censored reward AND
# the raw uncensored coverage.
BC_KEY = {'success': 'success', 'reward_max': 'mean_max_reward',
          'coverage': 'mean_max_coverage'}


def st_rows(step, key):
    """-> [(order, arm, selection, {n: value})] for the 12 Q-trained ST arms."""
    out = []
    for d in sorted(ST_BASE.glob('*ver-q_sac_all*')):
        arm = re.sub(r'^value_|_ver-q_sac_all|_enc-resnet18|_seed-42$|_demos-\d+', '', d.name)
        arm = arm.replace('_son-', ' ').replace('_split-', ' ').strip()
        arm = arm.replace('blq', 'blq137').replace('brd100', 'brd100')
        for sel in SELECTIONS:
            f = d / f'bon_search_sel-{sel}' / 'success_curves.jsonl'
            if not f.exists():
                continue
            best = None
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue            # torn line from a concurrent append
                if int(r['step']) == step:
                    best = r            # later rows supersede (a requeue re-evaluates)
            if best and best.get(key):
                out.append((1, arm, sel,
                            {int(n): best[key][i] for i, n in enumerate(best['n'])
                             if best[key][i] is not None}))
    return out


def bc_rows(metric):
    """-> [(order, arm, selection, {n: value})], one per (split, selection, ranker)."""
    out = []
    if not BC_BASE.exists():
        return out
    key = BC_KEY.get(metric)
    if key is None:
        return out                      # e.g. reward_final, which bon-sweep does not store
    for d in sorted(BC_BASE.glob('unetbc_*')):
        m = re.match(r'unetbc_(\w+?)_step\d+k_(argmax|final_pass)$', d.name)
        f = d / 'bon_curves.json'
        if not m or not f.exists():
            continue
        split, sel = m.group(1), m.group(2)
        rep = json.loads(f.read_text())
        for ranker, c in rep['curves'].items():
            vals = c.get(key)
            if vals is None:
                continue
            out.append((0, f'unetbc {split} [{ranker}]', sel,
                        {int(n): vals[i] for i, n in enumerate(c['n'])
                         if vals[i] is not None}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric', default='success',
                    choices=['success', 'reward_max', 'reward_final', 'coverage'])
    ap.add_argument('--step', type=int, default=100000)
    args = ap.parse_args()
    st_key = ST_KEY.get(args.metric)
    rows = bc_rows(args.metric)
    if st_key is not None:
        rows += st_rows(args.step, st_key)
    if not rows:
        raise SystemExit('no curves found yet')

    label = {'success': 'Success rate (coverage > 0.95)',
             'reward_max': 'Mean MAX reward  = clip(coverage/0.95, 0, 1), censored at 1.0',
             'reward_final': 'Mean FINAL reward (censored at 1.0)',
             'coverage': 'Mean MAX goal coverage, RAW (uncensored)'}[args.metric]
    print(f'#### {label} — step {args.step:,}, n = 1…64\n')
    if args.metric == 'coverage':
        print('*Raw goal coverage, uncensored. `mean_reward` saturates at 1.0 from coverage\n'
              '0.95 up, so it cannot separate two arms among their solved episodes; this can.\n'
              'A row of em dashes means that curve predates the 2026-09-21 change that began\n'
              'persisting coverage and needs a re-run.*\n')
    print('| arm | sel | ' + ' | '.join(f'n={n}' for n in NS) + ' |')
    print('|---|---|' + '--:|' * len(NS))
    for _, arm, sel, per_n in sorted(rows):
        cells = ' | '.join(f'{per_n[n]:.3f}' if n in per_n else '–' for n in NS)
        print(f'| {arm} | `{sel}` | {cells} |')

    have = {n for _, _, _, p in rows for n in p}
    missing = [n for n in NS if n not in have]
    print(f'\n<!-- {len(rows)} rows'
          + (f'; n not yet measured anywhere: {missing}' if missing else '; full n grid')
          + ' -->')


if __name__ == '__main__':
    main()
