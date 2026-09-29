"""Backfill `mean_coverage` into success_curves.jsonl from the per-step success_curve.json.

WHY THIS EXISTS INSTEAD OF A RE-RUN. `_row_from_curve` is an explicit whitelist, so when
`mean_coverage` was added to the curve on 2026-09-21 it reached every
`step_*/success_curve.json` and none of the run-level jsonl rows. The measurement was never
missing -- only its projection was. Re-running 24 GPU jobs to recover a number already on disk
would be waste, so this re-merges the step files through the (now fixed) projection.

Idempotent, and it merges rather than replaces: `append_curve_row` unions over n, so a step
whose jsonl row already carries coverage is rewritten to the same value.

    python scripts/backfill_curve_coverage.py --dry-run
    python scripts/backfill_curve_coverage.py
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from eval_search_pusht import append_curve_row, read_curve_rows   # noqa: E402

ROOT = pathlib.Path('/gscratch/robotics/harine/diffusion_policy_outputs'
                    '/pusht_search/pusht_image_search_imgonly/outer_inner')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='*ver-q_sac_all*')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    fixed = skipped = 0
    for run in sorted(ROOT.glob(args.glob)):
        for sub in sorted(run.glob('bon_search*')):
            have = {r['step'] for r in read_curve_rows(sub)
                    if r.get('mean_coverage') is not None}
            for step_dir in sorted(sub.glob('step_*')):
                f = step_dir / 'success_curve.json'
                if not f.exists():
                    continue
                curve = json.loads(f.read_text())
                step = int(step_dir.name.split('_')[1])
                if curve.get('mean_coverage') is None or step in have:
                    skipped += 1
                    continue
                if args.dry_run:
                    print(f'would fill {run.name}/{sub.name} step {step} '
                          f'({len(curve["mean_coverage"])} n)')
                else:
                    append_curve_row(sub, step, curve.get('checkpoint', ''), curve)
                fixed += 1
    print(f'\n{"would fill" if args.dry_run else "filled"}: {fixed}   '
          f'skipped (already present or no coverage): {skipped}')


if __name__ == '__main__':
    main()
