#!/bin/bash
# asteroid.py run: sbatch --job-name=<tag> --export=ALL,TAG=<tag>,ARGS="<asteroid.py train flags>"[,SMOKE="<smoke flags>"] astrun.sbatch
# Resumes from runs/<tag>/state.pt if it exists (resubmit the same command after a timeout).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_rounds_20261103/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg PYTHONPATH=/gscratch/robotics/harine/L2S
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
if [ -n "${SMOKE+x}" ]; then
  $PY -u asteroid.py smoke $SMOKE 2>&1 | grep -v Warning | grep -E '"iter"|Traceback|Error' | tail -6
  grep -q train_success runs/asteroid_smoke/log.jsonl || { echo 'smoke failed'; exit 1; }
  grep -A9 -E '"feedback"|"obs_noise_slots"' runs/asteroid_smoke/config.json | tr -d ' \n'; echo; echo SMOKE_OK
fi
$PY -u asteroid.py train --tag $TAG $ARGS 2>&1 | grep -v Warning | grep -E "ROUND|Traceback|Error"
