#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
set -euo pipefail
cd "${DP_REPO:-${SLURM_SUBMIT_DIR}}"
exec /gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python -u \
    scripts/diagnose_pusht_capacity.py --run-dir "$1"
