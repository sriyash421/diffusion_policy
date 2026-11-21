#!/bin/bash
# Control eval: STk0..7 and BC of one run on the 50 BC-success control pairs (control_episodes.json), 8 seeds × 8 attempts.
#   sbatch --job-name=ctrl_<tag> control_eval.sbatch <tag> [seeds attempts]   -> runs/<tag>/bon_seeds<seeds>_control.npz
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30 INTERP_TEST=control
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
[[ ${1:-} == c_* ]] && export MAZE_SETTING=interp30c   # c_<tag>: continuous maze (cmaze.py)
echo "== start $(date) on $(hostname) tag=$1"
/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python -u bon_eval.py --tag "$1" --seeds ${2:-8} --attempts ${3:-8} --splits test 2>&1 | grep -v Warning
echo "== done $(date)"
