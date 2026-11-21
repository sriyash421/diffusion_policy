#!/bin/bash
# Round-based in-context training / eval (rounds.py). Usage: sbatch [--job-name=...] rounds.sbatch <rounds.py args...>
#SBATCH --job-name=rounds
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_rounds_20261103/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=4 MPLBACKEND=Agg PYTHONPATH=/gscratch/robotics/harine/L2S
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python -u ${SCRIPT:-rounds.py} "$@"
