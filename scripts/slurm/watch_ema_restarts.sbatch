#!/bin/bash
#SBATCH --account=ckpt-robotics
#SBATCH --partition=ckpt
#SBATCH --qos=ckpt
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/watch_ema_restarts-%j.out
set -euo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone_pushT
exec /gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python scripts/watch_ema_restarts.py "$1"
