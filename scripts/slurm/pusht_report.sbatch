#!/bin/bash
#SBATCH --account=ckpt-robotics
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --job-name=pusht_report
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/pusht_report-%j.out
# Rebuild the PushT W&B report (scripts/wandb_fulltraj_report.py); pass --dry-run for figures only.
#   sbatch scripts/slurm/pusht_report.sbatch [--dry-run]
set -eo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone_pushT
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate vae_pushT_l2s
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/wandb_fulltraj_report.py "$@"
