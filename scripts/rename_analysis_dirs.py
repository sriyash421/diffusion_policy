"""Rename analysis output folders to the dataset-first convention. Dry run unless --apply.

    python scripts/rename_analysis_dirs.py                  # show what would move
    python scripts/rename_analysis_dirs.py --apply

Moves each run folder under the mode_analysis and attention_analysis roots to
scripts/analysis_run_name.canonical(). Only the ANALYSIS folders move; the checkpoint
directories under diffusion_policy_outputs keep their names, because the eval output lives
inside them and build_geometric_splits_doc.py addresses it by exact path.

SKIPS ANYTHING THAT IS NOT A RUN FOLDER -- `expert/` (a policy-independent analysis, not an
arm) and legacy names like `k16_blq/` that predate the convention and cannot be parsed. A
folder already in canonical form is left alone, so this is idempotent and safe to re-run.
Refuses to overwrite: if the target exists, the pair is reported and neither is touched.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analysis_run_name import canonical

ROOTS = ['/gscratch/robotics/harine/mode_analysis',
         '/gscratch/robotics/harine/attention_analysis']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--roots', nargs='*', default=ROOTS)
    args = ap.parse_args()

    moved = skipped = blocked = 0
    for root in args.roots:
        r = pathlib.Path(root)
        if not r.is_dir():
            print(f'{root}: not a directory, skipping'); continue
        print(f'\n=== {root} ===')
        for d in sorted(p for p in r.iterdir() if p.is_dir()):
            new = canonical(d.name)
            if new is None:
                print(f'  skip (not a run folder)  {d.name}'); skipped += 1
                continue
            if new == d.name:
                print(f'  already canonical        {d.name}'); skipped += 1
                continue
            tgt = d.parent / new
            if tgt.exists():
                print(f'  BLOCKED, target exists   {d.name}\n'
                      f'                        -> {new}'); blocked += 1
                continue
            print(f'  {d.name}\n   -> {new}')
            if args.apply:
                d.rename(tgt)
            moved += 1
    verb = 'moved' if args.apply else 'would move'
    print(f'\n{verb}: {moved}   skipped: {skipped}   blocked: {blocked}')
    if not args.apply:
        print('dry run; re-run with --apply')


if __name__ == '__main__':
    main()
