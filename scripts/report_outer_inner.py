#!/usr/bin/env python
"""Status + search-quality report for the three outer/inner PushT search runs.

Prints, per run: SLURM state, progress, the search-quality (nRMSE / verifier-value)
metrics, drift across the inner loop, and any best-of-N success rates already evaluated.
Also lists which 10k checkpoints still need a success-rate eval.

    python scripts/report_outer_inner.py              # full report
    python scripts/report_outer_inner.py --pending    # just the un-evaluated 10k ckpts

The nRMSE triplet is the whole point of the search: `first` is candidate 0, generated with
an EMPTY search context (i.e. the no-search baseline), `min` is the best of max_actions
candidates, `avg` is the mean. So (first - min) is the best-of-n gain, and `avg` is the
control that says whether the candidate DISTRIBUTION moved or we merely drew more samples
from an unchanged one. Same logic for action_value_{first,best}, in verifier units.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
from collections import defaultdict

OUT_ROOT = pathlib.Path(os.environ.get(
    'DP_OUTPUT_ROOT', '/gscratch/robotics/harine/diffusion_policy_outputs')) / 'runs'

RUNS = [
    ('value',         'train_pusht_search_outer_inner'),
    ('subgoal',       'train_pusht_search_outer_inner_subgoal'),
    ('subgoal_value', 'train_pusht_search_outer_inner_subgoal_verifier'),
]

EVAL_EVERY = 10000


def queue_summary():
    """One-line-per-job squeue dump. Jobs cannot be attributed to configs from squeue --
    the sbatch Command does not carry the hydra.run.dir argument -- so liveness is
    reported per run from log freshness instead (see `liveness`)."""
    try:
        return subprocess.run(
            ['squeue', '-u', os.environ.get('USER', 'harine'),
             '-o', '%.10i %.9P %.9T %.11M %.8R'],
            capture_output=True, text=True, timeout=30).stdout.rstrip()
    except Exception as e:
        return f'(squeue failed: {e})'


def liveness(run_dir):
    """How long ago this run last wrote to its log -- the per-run 'is it alive' signal."""
    import time
    p = run_dir / 'logs.json.txt'
    if not p.is_file():
        return 'no log yet'
    age = time.time() - p.stat().st_mtime
    if age < 300:
        return f'ALIVE (wrote {age:.0f}s ago)'
    return f'STALE (no write for {age/60:.1f} min)'


def load_logs(run_dir):
    p = run_dir / 'logs.json.txt'
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # torn final line while training is writing
    return rows


def latest_with(rows, key):
    for d in reversed(rows):
        if key in d:
            return d
    return None


def success_rows(run_dir):
    p = run_dir / 'bon_search' / 'success_curves.jsonl'
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def checkpoints(run_dir):
    d = run_dir / 'checkpoints'
    if not d.is_dir():
        return []
    steps = []
    for p in d.glob('step_*.ckpt'):
        m = re.search(r'step_(\d+)\.ckpt', p.name)
        if m:
            steps.append((int(m.group(1)), p))
    return sorted(steps)


def _inflight_ckpts():
    """Checkpoint paths currently being evaluated by a queued/running eval job.

    A result row only appears in success_curves.jsonl when the whole n-sweep finishes
    (~30 min), so without this every polling tick would resubmit the same checkpoints and
    pile up duplicate jobs. squeue cannot be asked which checkpoint a job is on -- the
    sbatch Command does not carry its arguments -- so read it back from each job's log,
    which echoes `evaluating <path>` as its first line.
    """
    paths = set()
    try:
        out = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', 'harine'), '-h',
             '-n', 'pusht_eval_ckpt', '-o', '%i'],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return paths
    logdir = pathlib.Path('/gscratch/robotics/harine/slurm_logs')
    for jid in out.split():
        f = logdir / f'pusht_eval_ckpt-{jid}.out'
        if not f.is_file():
            continue
        try:
            for line in f.read_text(errors='ignore').splitlines():
                if line.startswith('evaluating '):
                    paths.add(line.split(' ', 1)[1].strip())
                    break
        except OSError:
            continue
    return paths


def pending_evals(run_dir, skip_inflight=None):
    """10k-multiple checkpoints with no success-rate row yet and no eval job in flight."""
    done = {r.get('step') for r in success_rows(run_dir)}
    inflight = _inflight_ckpts() if skip_inflight is None else skip_inflight
    return [(s, p) for s, p in checkpoints(run_dir)
            if s % EVAL_EVERY == 0 and s not in done and str(p) not in inflight]


def report(pending_only=False):
    if pending_only:
        for label, name in RUNS:
            for step, path in pending_evals(OUT_ROOT / name):
                print(f'{label}\t{step}\t{path}')
        return

    print('=== slurm queue ===')
    print(queue_summary())
    for label, name in RUNS:
        run_dir = OUT_ROOT / name
        rows = load_logs(run_dir)
        print(f'\n=== {label}  ({name})')
        if not rows:
            print('    no logs yet')
            continue
        last = rows[-1]
        step = max(d['global_step'] for d in rows)
        outer = max(d.get('epoch', 0) for d in rows)
        loss = latest_with(rows, 'train_loss')
        lr_row = latest_with(rows, 'lr') or {}
        print(f'    liveness   : {liveness(run_dir)}')
        print(f'    progress   : step {step}/100000 ({100*step/100000:.1f}%)  '
              f'outer {outer}  lr {lr_row.get("lr", float("nan")):.2e}')
        if loss:
            print(f'    train_loss : {loss["train_loss"]:.4f}')

        m = latest_with(rows, 'val_nrmse_min')
        if m:
            print(f'    -- search quality @ step {m["global_step"]} '
                  f'(first = no-search baseline, min = best-of-n) --')
            for split in ('val', 'test'):
                if f'{split}_nrmse_min' not in m:
                    continue
                f_, mi, av = (m[f'{split}_nrmse_first'], m[f'{split}_nrmse_min'],
                              m[f'{split}_nrmse_avg'])
                gain = 100 * (f_ - mi) / f_ if f_ else float('nan')
                print(f'    {split:>4} nRMSE : first {f_:.4f}  min {mi:.4f}  avg {av:.4f}'
                      f'   -> best-of-n gain {gain:.1f}%')
                vf, vb = m.get(f'{split}_action_value_first'), m.get(f'{split}_action_value_best')
                if vf is not None:
                    print(f'    {split:>4} value : first {vf:.2f}  best {vb:.2f}'
                          f'   -> gain {vb - vf:+.2f}')
        if 'val_loss' in (vl := latest_with(rows, 'val_loss') or {}):
            print(f'    val_loss   : {vl["val_loss"]:.4f} @ step {vl["global_step"]}')

        drift = [(d['train_drift_inner_step'], d['train_drift_mse_eps'])
                 for d in rows if 'train_drift_mse_eps' in d]
        if drift:
            by = defaultdict(list)
            for i, v in drift:
                by[i].append(v)
            recent = defaultdict(list)
            for i, v in drift[-256:]:
                recent[i].append(v)
            keys = sorted(recent)
            means = {i: sum(recent[i]) / len(recent[i]) for i in keys}
            curve = '  '.join(f'{i}:{means[i]:.2e}' for i in keys)
            print(f'    drift      : {curve}')
            # inner_step 0 is measured ONE update after the snapshot, so it is always the
            # smallest and a 0->last ratio overstates staleness. The meaningful number is
            # growth across the REST of the inner loop.
            if len(keys) > 2:
                a, b = means[keys[1]], means[keys[-1]]
                print(f'                 growth inner {keys[1]}->{keys[-1]}: '
                      f'{b/a if a else float("nan"):.2f}x  '
                      f'(flat => inner_epochs could go higher)')

        ro = latest_with(rows, 'test/mean_score') or latest_with(rows, 'test_mean_score')
        if ro:
            k = 'test/mean_score' if 'test/mean_score' in ro else 'test_mean_score'
            print(f'    rollout    : {k} = {ro[k]:.4f} @ step {ro["global_step"]}')

        cks = checkpoints(run_dir)
        if cks:
            print(f'    ckpts      : {len(cks)}, latest step_{cks[-1][0]:07d}')
        srs = success_rows(run_dir)
        if srs:
            print('    -- best-of-N success rate --')
            for r in sorted(srs, key=lambda r: r.get('step', 0)):
                pairs = ' '.join(f'n={n}:{s:.3f}' for n, s in
                                 zip(r.get('n', []), r.get('success_rate', [])))
                print(f'      step {r.get("step")}: {pairs}')
        pend = pending_evals(run_dir)
        if pend:
            print(f'    pending eval: {[s for s, _ in pend]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pending', action='store_true',
                    help='print only un-evaluated 10k checkpoints, tab-separated')
    report(pending_only=ap.parse_args().pending)
