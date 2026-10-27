"""Every best-of-n success_curve.json under pusht_search -> one CSV.

Walks every task tree (pusht_image_search, pusht_image_search_imgonly, ...) and every
trainer under it; `model` names the trainer (MODELS), `split` the run's demos/split tag
(SPLITS gives the short names; an unlisted tag is kept verbatim).

One row per (run, readout dir, step, n). Selection and verifier value are read from the
json, not parsed from the directory name. Coverage: `coverage` is the stored mean
per-episode max coverage; files written before it was recorded leave it blank and give
only `coverage_lo` / `coverage_hi`, derived from the max episode reward
(reward = clip(cov / 0.95, 0, 1), so a success only bounds its coverage to [0.95, 1]).

    python scripts/export_bon_csv.py --out docs/reports/summaries/bon_results.csv
    python scripts/export_bon_csv.py --out tgn.csv --values wp_v3_la_tgn,wp_v3_la_tgd --splits blq137,brd100
"""
import csv
import glob
import json
import os
import pathlib
import re
from functools import lru_cache

import click
import numpy as np
import yaml

ROOT = '/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search'
SPLITS = {'blq137': 'demos-137_split-blq', 'brd100': 'demos-100_split-brd100',
          'brd60': 'demos-60_split-brd60', 'brd30': 'demos-30_split-brd30',
          'trq173': 'demos-173_split-trq', 'mv176': 'demos-176_split-mv',
          'blq50': 'demos-50_split-blq50', 'blq50far': 'demos-50_split-blq50far',
          'train176': 'demos-176', 'train126': 'demos-126', 'train30': 'demos-30'}
MODELS = {'unet_bc': 'bc', 'outer_inner': 'st', 'offline': 'st_offline', 'lstm_bc': 'lstm_bc'}
TAG_TO_SPLIT = {v: k for k, v in SPLITS.items()}
FIELDS = ['task', 'model', 'split', 'run', 'readout', 'step', 'value', 'selection', 'obs_eval',
          'temperature',
          'n', 'n_episodes', 'successes', 'success_rate', 'success_ci_lo', 'success_ci_hi',
          'coverage', 'coverage_lo', 'coverage_hi', 'source']


def coverage_bounds(rewards):
    r = np.asarray(rewards, float)
    return float(np.mean(0.95 * r)), float(np.mean(np.where(r >= 1.0, 1.0, 0.95 * r)))


_RUN_CFG = {}


def run_noise(run_dir):
    """(trained under obs noise, the run's own corrupt_obs_eval) from its hydra config.
    Noise = a per-slot ladder (slot_obs_noise.mode != uniform) or corrupt_obs; a null
    corrupt_obs_eval resolves to False (model/common/obs_corruption.py)."""
    if run_dir not in _RUN_CFG:
        p = pathlib.Path(run_dir) / '.hydra' / 'config.yaml'
        c = yaml.safe_load(p.read_text()) if p.exists() else {}
        son = (c.get('slot_obs_noise') or {}).get('mode', 'uniform')
        _RUN_CFG[run_dir] = (son != 'uniform' or bool(c.get('corrupt_obs')),
                             bool(c.get('corrupt_obs_eval')))
    return _RUN_CFG[run_dir]


def obs_eval_label(curve, run_dir):
    """conditional = rollouts saw the observations the run trained on (clean for a run
    trained clean, noised for a noised one); clean = a noised run evaluated noise-free."""
    trained_noisy, cfg_eval = run_noise(run_dir)
    if not trained_noisy:
        return 'conditional'
    on = curve['corrupt_obs_eval'] if curve.get('corrupt_obs_eval') is not None else cfg_eval
    return 'conditional' if on else 'clean'


def rows_for(path, task, model, split, curve=None):
    d = json.load(open(path)) if curve is None else curve
    p = pathlib.Path(path)
    if not valid_curve(p, d):
        return
    cov = d.get('mean_coverage')
    if cov is not None and len(cov) != len(d['n']):
        cov = None
    by_kind = d.get('per_n_rewards_by_kind', {}).get('max', {})
    for i, n in enumerate(d['n']):
        rate = d['success_rate'][i]
        lo, hi = coverage_bounds(by_kind[str(n)]) if str(n) in by_kind else (None, None)
        if cov:
            lo = hi = None
        yield {
            'task': task, 'model': model, 'split': split, 'run': p.parents[2].name, 'readout': p.parents[1].name,
            'step': int(p.parent.name.split('_')[-1]), 'value': d.get('verifier_value'),
            'selection': d.get('selection'),
            'obs_eval': obs_eval_label(d, p.parents[2]),
            'temperature': d.get('selection_temperature') if d.get('selection') == 'softmax' else '',
            'n': n, 'n_episodes': d['n_episodes'],
            'successes': int(round(rate * d['n_episodes'])), 'success_rate': rate,
            'success_ci_lo': d['success_ci'][i][0], 'success_ci_hi': d['success_ci'][i][1],
            'coverage': cov[i] if cov else '', 'coverage_lo': '' if lo is None else lo,
            'coverage_hi': '' if hi is None else hi, 'source': str(p),
        }


@lru_cache(maxsize=1)
def invalidated_readouts():
    audit = pathlib.Path(ROOT) / 'pusht_ema_audit_20261003' / 'affected_evals.json'
    return {os.path.realpath(r['output']) for r in json.loads(audit.read_text())}


def valid_curve(path, curve):
    """Never re-export invalidated noisy EMA evaluations, even if restored on disk."""
    path = pathlib.Path(path)
    if os.path.realpath(path.parents[1]) in invalidated_readouts():
        return (curve.get('obs_noise_rng') == 'per_episode_v2'
                and curve.get('ema_noise_buffers_repaired') is True)
    return True


@click.command()
@click.option('--out', required=True, type=click.Path())
@click.option('--values', default='', help='comma-separated verifier values to keep (default: all)')
@click.option('--splits', default='', help=f'comma-separated split names, e.g. {sorted(SPLITS)} (default: all)')
def main(out, values, splits):
    keep_v = {v for v in values.split(',') if v}
    keep_s = {s for s in splits.split(',') if s}
    rows = []
    for run in sorted(glob.glob(os.path.join(ROOT, '*', '*', '*'))):
        task, trainer, name = pathlib.Path(run).parts[-3:]
        m = re.search(r'_(demos-\d+(?:_split-[a-z0-9]+)?)_seed-', name)
        if trainer not in MODELS or m is None:
            continue
        split = TAG_TO_SPLIT.get(m.group(1), m.group(1))
        if keep_s and split not in keep_s:
            continue
        for f in sorted(glob.glob(f'{run}/bon_search*/step_*/success_curve.json')):
            rows.extend(r for r in rows_for(f, task, MODELS[trainer], split)
                        if not keep_v or r['value'] in keep_v)
    rows.sort(key=lambda r: (r['task'], r['model'], r['split'], r['run'], r['readout'],
                             r['step'], r['n']))
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} rows -> {out}')


if __name__ == '__main__':
    main()
