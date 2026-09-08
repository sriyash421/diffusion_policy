#!/bin/bash
# a* recall + verifier informativeness for the 5-arm x 3-step grid.
#
#   unset WANDB_API_KEY
#   sbatch --account=robotics --partition=ckpt --array=0-14 scripts/slurm/astar_grid.sbatch
#
# One task per (arm, step). Read-only with respect to every training run: it loads a
# checkpoint and writes only under ANALYSIS_ROOT, so it is safe against a live trainer.
#
# --time is finite and short on purpose: a cluster maintenance reservation can start while
# these are queued, and a job that would run past it never starts at all.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --job-name=astar
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%A_%a.out
set -euo pipefail

set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
# vae_pushT_l2s, NOT robodiff: this imports eval_search_pusht, which is what that env is for.
conda activate "${DP_CONDA_ENV:-vae_pushT_l2s}"
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"
TASK="$ROOT/pusht_search/pusht_image_search_imgonly"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/astar_recall}"

# subdir | run stem | short label
ARMS=(
  "unet_bc|unetbc_ver-t_goal|bc"
  "offline|value_k1_ver-t_goal|stk1"
  "outer_inner|value_k16_ver-t_goal|k16-uniform"
  "outer_inner|value_k16_ver-t_goal_son-flat400|flat400"
  "outer_inner|value_k16_ver-t_goal_son-ramp400to200|ramp400to200"
)
STEPS=(10000 50000 100000)

IDX="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
ARM_SPEC="${ARMS[$((IDX / ${#STEPS[@]}))]}"
STEP="${STEPS[$((IDX % ${#STEPS[@]}))]}"
IFS='|' read -r SUB STEM LABEL <<< "$ARM_SPEC"

RUN="${STEM}_enc-resnet18_demos-137_split-blq_seed-42"
CKPT="$TASK/$SUB/$RUN/checkpoints/$(printf 'step_%07d.ckpt' "$STEP")"
# The canonical analysis name sorts every arm into aligned columns. It falls back to the run
# name itself for a stem the pattern does not cover (the UNet BC arm is not `value_k*`).
CANON="$(python scripts/analysis_run_name.py "$RUN")"
OUT="$ANALYSIS_ROOT/$CANON/$(printf 'step_%07d.json' "$STEP")"

echo "task $IDX: arm=$LABEL step=$STEP"
echo "  ckpt $CKPT"
echo "  out  $OUT"
test -f "$CKPT" || { echo "MISSING CHECKPOINT"; exit 1; }

PYTHONPATH=$PWD python scripts/astar_uniform_walk.py \
    -c "$CKPT" --arm "$LABEL" --device cuda:0 \
    --episodes "${EPISODES:-50}" --n "${NCAND:-16}" \
    --out "$OUT" "$@"
