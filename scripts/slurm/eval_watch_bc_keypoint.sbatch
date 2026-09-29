#!/bin/bash
#SBATCH --job-name=bc_keypoint_eval
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# logs on /gscratch, not home: home is a 10G hard quota (see README_pusht.md)
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Success-rate watcher for the PushT KEYPOINT BC arm, on the preemptible checkpoint
# partition so the guaranteed robotics/weirdlab GPUs stay free for training. --requeue
# re-launches after preemption, and eval_bc_keypoint_pusht.py seeds its `seen` set from the
# results already on disk, so a preempted watcher resumes instead of re-evaluating.
#
# The keypoint twin of eval_watch_pusht_search.sbatch. It is a SEPARATE script because
# eval_search_pusht.py constructs a PushTImageEnv directly and has no keypoint path at all;
# eval_bc_keypoint_pusht.py instead instantiates the run's OWN task.env_runner out of the
# checkpoint config, so the eval rollout cannot drift from the training rollout.
#
# Usage: sbatch scripts/slurm/eval_watch_bc_keypoint.sbatch <TRAIN_OUTPUT_DIR> [extra args]
#   (the training run's output dir, which contains checkpoints/step_*.ckpt)
# Extra args are forwarded to eval_bc_keypoint_pusht.py -- e.g. --idle-exit-sec 7200.
set -euo pipefail

RUN_DIR="${1:?usage: sbatch eval_watch_bc_keypoint.sbatch <TRAIN_OUTPUT_DIR> [eval args]}"
shift

# conda's deactivate hook reads an unset var, which kills the job under `set -u` if sbatch
# was run from an active conda shell
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
DP_CONDA_ENV="${DP_CONDA_ENV:-vae_pushT_l2s}"
conda activate "$DP_CONDA_ENV"
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

# wandb writes its RUN files to the CWD unless told otherwise, and the CWD is the repo --
# which lives under a 10GB HOME quota.
GSCRATCH_ROOT="${GSCRATCH_ROOT:-/gscratch/robotics/harine}"
export WANDB_DIR="${WANDB_DIR:-$GSCRATCH_ROOT/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$GSCRATCH_ROOT/wandb/cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$GSCRATCH_ROOT/wandb/artifacts}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$GSCRATCH_ROOT/wandb/config}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR"

python eval_bc_keypoint_pusht.py \
  --watch \
  --run-dir "$RUN_DIR" \
  --device cuda:0 \
  "$@"
