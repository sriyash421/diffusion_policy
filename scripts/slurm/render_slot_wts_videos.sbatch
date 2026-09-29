#!/bin/bash
#SBATCH --job-name=slotwts_vid
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%A_%a.out
#SBATCH --array=0-6
#
# Candidate-fan videos for the round-2 slot-weight arms, at TIER A (step 10000) -- the
# paired step every arm shares, so the fans are comparable across arms. n=16, 10 episodes.
#
# Slots 0, 7 and 15 are drawn in addition to the argmax and the final candidate, which is
# what makes "does candidate 15 beat candidate 0" visible in a frame rather than only in a
# table. At n=16 slot 15 IS the final candidate, so the two markers are folded into one and
# the legend names both -- see highlight_spec in scripts/render_search_videos.py.
#
# One array task per arm rather than one process over all seven: each loads a checkpoint
# and forks a 32-worker verifier sim pool, and the UNet BC checkpoint alone is 4.7 GB.
#
#   sbatch scripts/slurm/render_slot_wts_videos.sbatch
set -euo pipefail
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search
OUTDIR=/mmfs1/home/harine/diffusion_policy_standalone/videos/aug26_slot_wts
K16=outer_inner/value_k16_ver-armTn%s_corrupt-False_demos-30_seed-42
STEP=10000

LABELS=(lin100-l2 lin100-l2tol1 geo735-l2 geo735-l2tol1 curr-lin100-l2 curr-lin100-l2tol1 unetbc)
RUNS=(
  "$(printf "$K16" _sw-lin100-l2)"
  "$(printf "$K16" _sw-lin100-l2tol1)"
  "$(printf "$K16" _sw-geo735-l2)"
  "$(printf "$K16" _sw-geo735-l2tol1)"
  "$(printf "$K16" _sw-curr-lin100-l2)"
  "$(printf "$K16" _sw-curr-lin100-l2tol1)"
  unet_bc/unetbc_demos-30_seed-42
)
# unetbc carries no verifier_tag and natively resolves to the pre-cutover `t_goal`. Left
# native, its videos would show a DIFFERENT scoring rule deciding the executed candidate
# than every other arm here, and than its own dump. Overridden to match; every other arm is
# armTn-native and must not be.
VERS=("" "" "" "" "" "" armTn)

a=$SLURM_ARRAY_TASK_ID
LABEL="${LABELS[$a]}"
CKPT=$(printf '%s/%s/checkpoints/step_%07d.ckpt' "$ROOT" "${RUNS[$a]}" "$STEP")
VER_ARG=()
[ -n "${VERS[$a]}" ] && VER_ARG=(--verifier-value "${VERS[$a]}")

echo "=== task $a: $LABEL step $STEP ${VER_ARG[*]-}"
[ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }
mkdir -p "$OUTDIR"

# --seed 42 and 10 episodes match the dumps, so a frame can be looked up in
# per_prediction_A.csv by (arm, episode_idx, t).
# --value-strip draws the per-candidate bars: the "does it improve with context" read at a
# glance. Bars are rescaled PER FRAME, so heights compare candidates within one decision
# and never across decisions.
PYTHONPATH=$PWD python -u scripts/render_search_videos.py \
  -c "$CKPT" \
  --label "$LABEL" \
  --out-dir "$OUTDIR" \
  --n 16 \
  --n-seeds 10 \
  --seed 42 \
  --value-strip \
  --highlight-slots 0,7,15 \
  --device cuda:0 \
  ${VER_ARG[@]+"${VER_ARG[@]}"} \
  "$@"

echo "=== done $LABEL -> $OUTDIR"
