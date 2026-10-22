"""Redo a frozen list of invalid noisy-EMA readouts, resuming completed N cells."""
import json
import os
from pathlib import Path
import subprocess
import sys

audit = Path(sys.argv[1])
index = int(os.environ['SLURM_ARRAY_TASK_ID'])
row = json.loads((audit / 'redo_manifest.json').read_text())[index]
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
    # Explicit so a manifest may point at a directory other than the one _bon_subdir
    # derives (e.g. candidate-stream dumps kept apart from the report's curves).
    command += ['--output_dir', row['output']]
    # Native verifier is baked into the checkpoint; an override is only needed
    # for readout directories explicitly naming a different verifier.
    if '_ver-' in Path(row['output']).name:
        command += ['--verifier-value', entry['verifier_value']]
    subprocess.run(command, check=True)
