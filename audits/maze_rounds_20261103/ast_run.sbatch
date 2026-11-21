#!/bin/bash
# ASTEROID-style rounds (asteroid.py): train (resumes per round from runs/<tag>/state.pt) -> eval -> plot.
# Usage: sbatch --job-name=ast_<tag> ast_run.sbatch <tag> [extra train args]. ckpt-all requeues on preemption.
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40|l40s|a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_rounds_20261103/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=4 MPLBACKEND=Agg PYTHONPATH=/gscratch/robotics/harine/L2S
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
tag=$1; shift
echo "== start $(date) on $(hostname) restart=${SLURM_RESTART_COUNT:-0}"
[[ -f runs/$tag/TRAINED ]] || $PY -u asteroid.py train --tag "$tag" --rounds 8 --updates 10000 --batch 64 --lr 1e-4 --warmup 500 "$@"
[[ -f runs/$tag/eval_seeds8.npz ]] || $PY -u asteroid.py eval --tag "$tag" --seeds 8 --attempts 8
$PY -u asteroid.py plot --tag "$tag" --seeds 8
$PY -u plot_clear.py "$tag"
