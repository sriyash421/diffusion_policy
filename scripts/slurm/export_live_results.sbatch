#!/bin/bash
#SBATCH --job-name=export_eval_csvs
#SBATCH --account=ckpt-robotics
#SBATCH --partition=ckpt
#SBATCH --qos=ckpt
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/export_results-%j.out
set -eo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone_pushT
/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python -u scripts/export_live_results.py --watch &
PID=$!
trap 'kill -TERM "$PID" 2>/dev/null || true; wait "$PID" || true; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
wait "$PID"
