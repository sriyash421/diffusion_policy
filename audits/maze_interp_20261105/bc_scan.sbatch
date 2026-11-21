#!/bin/bash
# BC scan over every (start, held-out goal) pair (3 seeds) and the 50-pair test set. Needs bc_selection.json.
#SBATCH --account=robotics
#SBATCH --partition=ckpt-all
#SBATCH --constraint=a40|a100|l40|l40s|h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=interp_bcscan
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30
/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python -u /gscratch/robotics/harine/codex_audits/maze_interp_20261105/bc_scan.py 2>&1 | grep -v Warning
