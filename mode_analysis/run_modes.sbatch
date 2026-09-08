#!/bin/bash
#SBATCH --job-name=modes
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Candidate-mode analysis. ckpt partition: analysis must not take a training GPU.
#   sbatch mode_analysis/run_modes.sbatch <RUN_DIR> [args]
set -euo pipefail
RUN_DIR="${1:?usage: sbatch run_modes.sbatch <RUN_DIR> [args]}"; shift
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate "${DP_CONDA_ENV:-vae_pushT_l2s}"
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone
GSCRATCH_ROOT="${GSCRATCH_ROOT:-/gscratch/robotics/harine}"
export WANDB_DIR="$GSCRATCH_ROOT/wandb" WANDB_CACHE_DIR="$GSCRATCH_ROOT/wandb/cache"
export WANDB_ARTIFACT_DIR="$GSCRATCH_ROOT/wandb/artifacts" WANDB_CONFIG_DIR="$GSCRATCH_ROOT/wandb/config"
mkdir -p "$WANDB_DIR"
python mode_analysis/candidate_modes.py --run-dir "$RUN_DIR" --device cuda:0 "$@"
