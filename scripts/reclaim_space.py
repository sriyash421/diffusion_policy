"""Reclaim space under diffusion_policy_outputs. Dry run unless --apply.

    python scripts/reclaim_space.py                 # list
    python scripts/reclaim_space.py --apply

THREE CATEGORIES, each opt-in by flag:

  --dupes      `latest.ckpt` where it is BYTE-IDENTICAL to the run's highest step_*.ckpt.
               Verified with a real comparison at delete time, never inferred from size --
               a same-size-different-bytes file would be silent data loss. It is only read
               by `training.resume` to extend a finished run past max_steps, and is
               restorable with `cp step_<max>.ckpt latest.ckpt`.

  --nongeo     Every checkpoint of BC (unet_bc/) and ST k=1 (offline/) runs whose name has
               no `_split-` tag, i.e. the runs NOT in SUCCESS_RATES_GEOMETRIC.md. These are
               the older random-split generation.

               NARROW IT WITH --after/--before. `nongeo` is not uniformly stale: it spans
               2026-08-02..08-30, and the 2026-08-29/30 files are the ResNet18-E2E
               noised-obs runs behind success_rates_noised_obs_resnetE2E.md and
               slot_verifier_scores_noised_obs.md -- the latter still has arms marked "not
               dumped yet", which needs those checkpoints. Dates are on file mtime and both
               bounds are INCLUSIVE.

               DELETING A CHECKPOINT NEVER INVALIDATES A RECORDED NUMBER: the success rates
               live in each run's bon_search_*/success_curves.jsonl and the docs are built
               from those, which are left alone. What is lost is the ability to run any NEW
               evaluation or analysis on these arms, permanently -- the same state four
               obs-noise arms are already in.

Skips any run with a live SLURM job, so nothing is pulled out from under a writer.
"""
import argparse
import datetime
import filecmp
import pathlib
import re
import subprocess

ROOT = pathlib.Path('/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search')


def live_job_names():
    try:
        out = subprocess.run(['squeue', '-u', 'harine', '-h', '-o', '%j'],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return set()
    return {l.strip() for l in out.splitlines() if l.strip()}


def human(n):
    for u in ('B', 'K', 'M', 'G', 'T'):
        if n < 1024:
            return f'{n:.1f}{u}'
        n /= 1024
    return f'{n:.1f}P'


def run_dirs():
    for task in ROOT.glob('*'):
        for sub in task.glob('*'):
            for run in sub.glob('*'):
                if (run / 'checkpoints').is_dir():
                    yield sub.name, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--dupes', action='store_true', help='redundant latest.ckpt')
    ap.add_argument('--nongeo', action='store_true',
                    help='all checkpoints of non-geometric BC and ST k=1 runs')
    ap.add_argument('--after', help='only files with mtime on/after this date (YYYY-MM-DD)')
    ap.add_argument('--before', help='only files with mtime on/before this date (YYYY-MM-DD)')
    args = ap.parse_args()
    if not (args.dupes or args.nongeo):
        ap.error('pick at least one of --dupes / --nongeo')
    after = datetime.date.fromisoformat(args.after) if args.after else None
    before = datetime.date.fromisoformat(args.before) if args.before else None

    def in_window(path):
        if after is None and before is None:
            return True
        d = datetime.date.fromtimestamp(path.stat().st_mtime)
        return (after is None or d >= after) and (before is None or d <= before)

    live = live_job_names()
    victims, freed, skipped = [], 0, []

    for subname, run in sorted(run_dirs(), key=lambda t: str(t[1])):
        ck = run / 'checkpoints'
        if f'tr_{run.name}' in live:
            skipped.append((run.name, 'training job live'))
            continue
        steps = sorted(ck.glob('step_*.ckpt'))

        if args.dupes:
            latest = ck / 'latest.ckpt'
            if latest.is_file() and steps:
                top = steps[-1]
                if latest.stat().st_size == top.stat().st_size:
                    victims.append(('dupe', latest, top))
                    freed += latest.stat().st_size
                else:
                    skipped.append((run.name, 'latest.ckpt differs in size from top step'))

        if args.nongeo and subname in ('unet_bc', 'offline') and '_split-' not in run.name:
            for f in sorted(ck.glob('*.ckpt')):
                if any(f == v[1] for v in victims) or not in_window(f):
                    continue
                victims.append(('nongeo', f, None))
                freed += f.stat().st_size

    print(f'{"KIND":<8} {"SIZE":>8}  PATH')
    for kind, f, _ in victims:
        print(f'{kind:<8} {human(f.stat().st_size):>8}  {f.relative_to(ROOT)}')
    print(f'\ncandidates: {len(victims)}   would free: {human(freed)}')
    for name, why in skipped:
        print(f'  SKIP {name}: {why}')

    if not args.apply:
        print('\ndry run; re-run with --apply')
        return

    removed = rfreed = gone = kept = 0
    for kind, f, twin in victims:
        # A live job can rotate latest.ckpt between the listing pass and this one, so every
        # path is re-checked rather than trusted; a vanished file is not an error.
        if not f.is_file() or (twin is not None and not twin.is_file()):
            gone += 1
            continue
        if kind == 'dupe':
            # Verified here, not at listing time: only a real comparison justifies deleting.
            try:
                if not filecmp.cmp(str(f), str(twin), shallow=False):
                    print(f'  NOT IDENTICAL, keeping {f.relative_to(ROOT)}')
                    kept += 1
                    continue
            except OSError as e:
                print(f'  unreadable, keeping {f.relative_to(ROOT)}: {e}')
                kept += 1
                continue
        try:
            n = f.stat().st_size
            f.unlink()
        except OSError as e:
            print(f'  could not remove {f.relative_to(ROOT)}: {e}')
            kept += 1
            continue
        removed += 1
        rfreed += n
    print(f'\nremoved {removed} files, freed {human(rfreed)}'
          f'   (vanished meanwhile: {gone}, kept: {kept})')


if __name__ == '__main__':
    main()
