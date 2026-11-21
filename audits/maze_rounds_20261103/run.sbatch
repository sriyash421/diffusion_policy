#!/bin/bash
# train -> eval -> plot for one config. Usage: sbatch --job-name=rounds_<tag> run.sbatch <tag> <steps-per-iter>
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_rounds_20261103/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=4 MPLBACKEND=Agg PYTHONPATH=/gscratch/robotics/harine/L2S
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
[[ -f runs/$1/TRAINED ]] || $PY -u rounds.py train --tag "$1" --rounds 8 --iters 30 --steps-per-iter "$2"
$PY -u rounds.py eval --tag "$1" --seeds 8
$PY -u rounds.py plot --tag "$1" --seeds 8
