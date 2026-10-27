#!/bin/bash
#SBATCH --account=ckpt-robotics
#SBATCH --partition=ckpt
#SBATCH --qos=ckpt-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=9:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/redo_noise-%A_%a.out
set -eo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone_pushT
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate vae_pushT_l2s
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python scripts/redo_noisy_evals.py "$1" &
PID=$!
trap 'kill -TERM "$PID" 2>/dev/null || true; wait "$PID" || true; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
wait "$PID"
