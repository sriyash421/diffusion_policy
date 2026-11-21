#!/bin/bash
# All held-out-goal pairs: STk0..(R-1) of one run on every (start, held-out goal) pair of bc_scan.npz (2870 pairs, all 41 held-out
# goals), not only the 50 test pairs. BC comes from bc_scan.npz (3 seeds), so BC is skipped here (--no-bc).
#   sbatch --job-name=all_<tag> allpairs_eval.sbatch <tag> [seeds attempts]   -> runs/<tag>/bon_seeds<seeds>_allpairs.npz
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30 INTERP_TEST=all
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
[[ ${1:-} == c_* ]] && export MAZE_SETTING=interp30c   # c_<tag>: continuous maze (cmaze.py)
echo "== start $(date) on $(hostname) tag=$1"
/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python -u bon_eval.py --tag "$1" --seeds ${2:-3} --attempts ${3:-8} --splits test --no-bc 2>&1 | grep -v Warning
echo "== done $(date)"
