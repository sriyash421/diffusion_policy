"""Backfill test `mean_reward` into every curve already on disk.

`eval_search_pusht.py` has always stored `per_n_rewards` -- the raw per-episode max reward
for every test episode at every n -- but only ever REPORTED the binary success rate derived
from it (reward >= 1.0). Adding `mean_reward` therefore needs no GPU and no re-rollout: it
is the mean of numbers already recorded, and it reproduces exactly what a fresh eval would
now write.

Two files per checkpoint carry it:
  bon_search/step_*/success_curve.json   -- the full curve, holds per_n_rewards
  bon_search/success_curves.jsonl        -- the run-level index, a projection without them

The jsonl row is filled from its matching step_*/success_curve.json, keyed by n rather than
by list position: a merged curve's `n` can differ between the two files when one job per n
wrote them at different times, so index-aligning would silently shift rewards onto the wrong
n. An n present in the jsonl with no reward on disk gets None, not a guess.

Idempotent -- rerunning recomputes the same values. Concurrent eval jobs hold a lock on the
jsonl, so this takes the same lock via eval_search_pusht._locked.

    python scripts/backfill_mean_reward.py [--dry]
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np

from eval_search_pusht import _locked

ROOT = pathlib.Path('/gscratch/robotics/harine/diffusion_policy_outputs/'
                    'pusht_search/pusht_image_search/offline')
DRY = '--dry' in sys.argv


def means_by_n(curve):
    """{n: mean test reward} for every n whose per-episode rewards are on disk."""
    out = {}
    for k, v in (curve.get('per_n_rewards') or {}).items():
        if v:
            out[int(k)] = float(np.mean(v))
    return out


def main():
    n_curve = n_row = n_skip = 0
    for run in sorted(ROOT.glob('*')):
        if run.is_symlink() or not run.is_dir():
            continue
        bs = run / 'bon_search'
        if not bs.is_dir():
            continue

        per_step = {}
        for f in sorted(bs.glob('step_*/success_curve.json')):
            curve = json.loads(f.read_text())
            m = means_by_n(curve)
            step = int(f.parent.name.split('_')[1])
            per_step[step] = m
            want = [m.get(int(n)) for n in curve['n']]
            if curve.get('mean_reward') == want:
                n_skip += 1
                continue
            curve['mean_reward'] = want
            if not DRY:
                tmp = f.with_suffix('.json.tmp')
                tmp.write_text(json.dumps(curve))
                tmp.replace(f)
            n_curve += 1

        jsonl = bs / 'success_curves.jsonl'
        if not jsonl.is_file():
            continue
        with _locked(jsonl):
            rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
            changed = False
            for r in rows:
                m = per_step.get(int(r['step']))
                if m is None:
                    continue
                want = [m.get(int(n)) for n in r['n']]
                if r.get('mean_reward') != want:
                    r['mean_reward'] = want
                    changed = True
                    n_row += 1
            if changed and not DRY:
                tmp = jsonl.with_suffix('.jsonl.tmp')
                tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                tmp.replace(jsonl)

    print(f"{'DRY-RUN: ' if DRY else ''}{n_curve} success_curve.json updated, "
          f"{n_row} jsonl rows updated, {n_skip} already current")


if __name__ == '__main__':
    main()
