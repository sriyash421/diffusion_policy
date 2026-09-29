#!/bin/bash
#SBATCH --job-name=unetbc_armTn
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%A_%a.out
#SBATCH --array=0-2
#
# Re-score the UNet BC null under armTn so all five arms of the slot-context grid share one
# value function.
#
# WHY THIS IS NEEDED. unetbc_demos-30 predates the 2026-08-19 scoring cutover, so its config
# resolves to `t_goal`. That is correct for the checkpoint and wrong for this comparison in
# a specific way: t_goal is FLAT ACROSS CANDIDATES until the arm touches the block, so an
# arm scored under it has a mechanically flatter slot profile whether or not it uses
# context -- which would make the null look like a null for the wrong reason.
#
# WHAT CHANGES. The value RANKS candidates, so scoring under armTn also changes which
# candidate argmax executes, and the trajectories differ from the native dump. That is the
# point, not a side effect: this measures the UNet BC policy selecting under armTn, which is
# what makes it a null for the armTn arms. The native t_goal dumps are kept alongside as
# `unetbc-native-tgoal_step*` rather than overwritten.
#
#   sbatch scripts/slurm/dump_unetbc_armTn.sbatch
set -euo pipefail
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search
OUTROOT=/mmfs1/home/harine/diffusion_policy_standalone/analysis/aug26_slot_wts
STEPS=(10000 30000 100000)
STEP=${STEPS[$SLURM_ARRAY_TASK_ID]}
CKPT=$(printf '%s/unet_bc/unetbc_demos-30_seed-42/checkpoints/step_%07d.ckpt' "$ROOT" "$STEP")
OUTDIR="$OUTROOT/unetbc_step$STEP"

echo "=== unetbc step $STEP scored under armTn"
[ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }

PYTHONPATH=$PWD python -u scripts/dump_candidate_scores.py \
  -c "$CKPT" \
  --arm unetbc \
  --out-dir "$OUTDIR" \
  --n 16 \
  --episodes 50 \
  --split test \
  --seed 42 \
  --verifier-value armTn \
  --per-step-json \
  --per-step-episodes 0 \
  --pair-seeds \
  --device cuda:0 \
  "$@"

echo "=== done unetbc step $STEP -> $OUTDIR"
