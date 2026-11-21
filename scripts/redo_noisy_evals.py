"""Run one cell of a frozen eval manifest (array task = row), resuming completed N cells.

Written for the noisy-EMA redo; also runs the valpool50 held-out-train evals
(scripts/launch_valpool50_evals.py): `split`/`split_file` pass through to
eval_search_pusht.py; a row with `evaluator: dt` runs eval_dt_pusht.py instead, and one with
`evaluator: cand` scripts/candidate_rmse_pusht.py, and one with `evaluator: video`
scripts/render_search_videos.py (widths tiled side by side).
"""
import json
import os
from pathlib import Path
import subprocess
import sys

audit = Path(sys.argv[1])
index = int(os.environ['SLURM_ARRAY_TASK_ID'])
row = json.loads((audit / os.environ.get('MANIFEST', 'redo_manifest.json')).read_text())[index]
if row.get('evaluator') == 'dt':
    for entry in row['entries']:
        if not (Path(row['output']) / entry['step'] / 'metrics.csv').exists():
            subprocess.run([sys.executable, 'eval_dt_pusht.py', entry['checkpoint'],
                            '--output-dir', str(Path(row['output']) / entry['step']),
                            '--split', entry['split'], '--split-file', entry['split_file'],
                            '--n', str(entry['n']), '--n-envs', str(entry['n_envs']),
                            '--seed', str(entry['seed'])], check=True)
    sys.exit(0)
if row.get('evaluator') == 'cand':
    entry = row['entries'][0]
    command = [sys.executable, 'scripts/candidate_rmse_pusht.py', '-c', entry['checkpoint'],
               '--split', entry['split'], '--out', row['output']]
    if entry.get('split_file'):
        command += ['--split-file', entry['split_file']]
    subprocess.run(command, check=True)
    sys.exit(0)
if row.get('evaluator') == 'video':
    entry = row['entries'][0]
    command = [sys.executable, 'scripts/render_search_videos.py', '-c', entry['checkpoint'],
               '--label', entry['label'], '--out-dir', row['output'], '--n-seeds', str(entry['n_seeds']),
               '--seed', str(entry['seed']), '--split', entry['split'], '--terminal-scoring', '--tile']
    command += [a for n in entry['n'] for a in ('--n', str(n))]
    if entry.get('split_file'):
        command += ['--split-file', entry['split_file']]
    subprocess.run(command, check=True)
    sys.exit(0)
for entry in row['entries']:
    remaining_n = list(entry['n'])
    destination = Path(row['output']) / entry['step'] / 'success_curve.json'
    if destination.exists():
        old = json.loads(destination.read_text())
        protocol = 'per_episode_v2' if row.get('corrupt_obs_eval', True) else None
        if (old.get('obs_noise_rng') == protocol
                and old.get('candidate_rng') == 'paired_decision_v3'):
            remaining_n = [n for n in remaining_n if n not in old['n']]
    if not remaining_n:
        continue
    command = [sys.executable, 'eval_search_pusht.py', '--checkpoint', entry['checkpoint'],
               '--run-dir', row['run'], '--device', 'cuda:0',
               '--corrupt-obs-eval' if row.get('corrupt_obs_eval', True) else '--no-corrupt-obs-eval',
               '--n-envs', str(entry['n_envs']), '--seed', str(entry['seed']),
               '--n-list', ','.join(map(str, remaining_n)), '--skip-val']
    if entry.get('selection'):
        command += ['--selection', entry['selection']]
    if entry.get('selection_temperature') is not None:
        command += ['--selection-temperature', str(entry['selection_temperature'])]
    if entry.get('terminal_scoring'):
        command += ['--terminal-scoring']
    if entry.get('test_start_seed') is not None:
        command += ['--test-start-seed', str(entry['test_start_seed'])]
    if entry.get('plan_dir'):
        command += ['--plan-dir', entry['plan_dir']]
    if entry.get('store_scores'):
        command += ['--store-scores']
    if entry.get('split'):
        command += ['--split', entry['split']]
    if entry.get('split_file'):
        command += ['--split-file', entry['split_file']]
    # Explicit so a manifest may point at a directory other than the one _bon_subdir
    # derives (e.g. candidate-stream dumps kept apart from the report's curves).
    command += ['--output_dir', row['output']]
    # Native verifier is baked into the checkpoint; an override is only needed
    # for readout directories explicitly naming a different verifier.
    if '_ver-' in Path(row['output']).name:
        command += ['--verifier-value', entry['verifier_value']]
    subprocess.run(command, check=True)
