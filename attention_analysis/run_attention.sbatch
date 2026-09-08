#!/bin/bash
#SBATCH --job-name=attn_viz
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Cross-attention capture for a trained search policy. On the preemptible ckpt partition:
# this is analysis, so it must not take a guaranteed training GPU.
#
#   sbatch attention_analysis/run_attention.sbatch <RUN_DIR> [extra args]
set -euo pipefail
RUN_DIR="${1:?usage: sbatch run_attention.sbatch <RUN_DIR> [args]}"; shift
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate "${DP_CONDA_ENV:-vae_pushT_l2s}"
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

# wandb writes its RUN files to the CWD unless told otherwise, and the CWD is the repo --
# which lives under a 10GB HOME quota. train_pusht_search.sbatch already redirects these;
# the eval and analysis jobs did not, and quietly grew repo/wandb to 1.2GB across 426 runs.
GSCRATCH_ROOT="${GSCRATCH_ROOT:-/gscratch/robotics/harine}"
export WANDB_DIR="${WANDB_DIR:-$GSCRATCH_ROOT/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$GSCRATCH_ROOT/wandb/cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$GSCRATCH_ROOT/wandb/artifacts}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$GSCRATCH_ROOT/wandb/config}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR"

python attention_analysis/visualize_attention.py --run-dir "$RUN_DIR" --device cuda:0 "$@"
