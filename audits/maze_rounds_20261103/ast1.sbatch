#!/bin/bash
# asteroid.py with the 1-step schedule (train + test eval after every iteration).
#SBATCH --account=gpu-a40-robotics
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_rounds_20261103/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=4 MPLBACKEND=Agg PYTHONPATH=/gscratch/robotics/harine/L2S
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python



$PY -u asteroid.py train --tag ast_s1 --rounds 8 --iters 30 --steps-per-iter 1 --batch 128 --lr 1e-4 --train-seeds 1 --test-seeds 8 2>&1 | grep -v Warning | grep -E "ROUND|Traceback|Error"
