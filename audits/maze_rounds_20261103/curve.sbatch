#!/bin/bash
# rounds.py with the in-training curve (train/test success after every iteration): smoke, then 1-step and 20-step runs.
#SBATCH --account=gpu-a40-robotics
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_rounds_20261103/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=4 MPLBACKEND=Agg PYTHONPATH=/gscratch/robotics/harine/L2S
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
$PY -u rounds.py smoke 2>&1 | grep -v Warning | tail -15
grep -q train_success runs/smoke/log.jsonl || { echo 'smoke: no curve rows'; exit 1; }
for spi in 1 20; do
  [[ -f runs/curve_spi$spi/TRAINED ]] || $PY -u rounds.py train --tag curve_spi$spi --rounds 8 --iters 30 --steps-per-iter $spi --batch 128 --lr 1e-4 --curve 1 2>&1 | grep -v Warning | grep -E "ROUND|Traceback|Error" || true
  [[ -f runs/curve_spi$spi/TRAINED ]] || { echo "curve_spi$spi failed"; exit 1; }
done
echo DONE
