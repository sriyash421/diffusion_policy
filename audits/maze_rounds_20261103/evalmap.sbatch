#!/bin/bash
# Per-episode success of every round's weights (8-attempt chains) + BC, for the goal maps: runs/<TAG>/eval_seeds8.npz
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_rounds_20261103/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg PYTHONPATH=/gscratch/robotics/harine/L2S
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
test -f runs/$TAG/TRAINED || { echo "runs/$TAG not TRAINED"; exit 1; }
/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python -u asteroid.py eval --tag $TAG --seeds 8 --attempts 8 2>&1 | grep -v Warning
