#!/bin/bash
# Held-out action error to the expert + how attempts end (mse_eval.py). Usage: sbatch mse_eval.sbatch [--pg] --tags a,b,c [--pairs 100 --seeds 2]
#SBATCH --account=robotics
#SBATCH --partition=ckpt-all
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
unset MAZE_SETTING
A=/gscratch/robotics/harine/codex_audits
export PYTHONPATH=/gscratch/robotics/harine/L2S:$A/maze_rounds_20261103:$A/maze_procgen_ast_20261121
cd $A/maze_rounds_20261103
echo "== start $(date) on $(hostname) restart=${SLURM_RESTART_COUNT:-0} args=$*"
/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python -u $A/maze_interp_20261105/${SCRIPT:-mse_eval.py} "$@" 2>&1 | grep -v Warning
echo "== done $(date)"
