#!/bin/bash
#SBATCH --job-name=aug19_cand
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%A_%a.out
#SBATCH --array=0-8
#
# Objective 2 of docs/reports/archive/aug19_analysis.md: candidate distance values for the 3x3 grid
# {unetbc, stk1, stk16} x {10k, 50k, 100k}, at n=16 over the 50 test episodes.
#
# Read-only with respect to every training run: loads a checkpoint and writes only into
# OUTDIR, so it is safe against a directory a live trainer is still writing to.
#
#   sbatch scripts/slurm/dump_aug19_grid.sbatch
set -euo pipefail
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search
OUTROOT=/mmfs1/home/harine/diffusion_policy_standalone/analysis/aug19_candidate_scores

# Both ST arms are the 4/4/256 pair: the big 6x8x1024 k=16 run stops at step 40000 and so
# cannot supply the 50k/100k cells. `bc_demos-30` predates the rename and IS ST k=1.
RUNS=(unet_bc/unetbc_demos-30_seed-42 offline/bc_demos-30_seed-42 outer_inner/value_k16_corrupt-False_demos-30_seed-42)
LABELS=(unetbc stk1 stk16)
STEPS=(10000 50000 100000)

a=$(( SLURM_ARRAY_TASK_ID / 3 ))
s=$(( SLURM_ARRAY_TASK_ID % 3 ))
CKPT=$(printf '%s/%s/checkpoints/step_%07d.ckpt' "$ROOT" "${RUNS[$a]}" "${STEPS[$s]}")
LABEL="${LABELS[$a]}"
OUTDIR=$(printf '%s/%s_step%07d' "$OUTROOT" "$LABEL" "${STEPS[$s]}")

echo "=== task $SLURM_ARRAY_TASK_ID: $LABEL step ${STEPS[$s]}"
[ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }

# --pair-seeds is the default and is what makes the three arms comparable episode-by-
# episode; it is passed explicitly so the command in docs/reports/archive/aug19_analysis.md is self-describing.
# --per-step-json writes all 50 episodes (~1 MB total) rather than a pre-chosen one:
# picking "the" episode before seeing the data would be choosing the example to fit.
PYTHONPATH=$PWD python -u scripts/dump_candidate_scores.py \
  -c "$CKPT" \
  --arm "$LABEL" \
  --out-dir "$OUTDIR" \
  --n 16 \
  --episodes 50 \
  --split test \
  --per-step-json \
  --pair-seeds \
  --device cuda:0 \
  "$@"

echo "=== done $LABEL step ${STEPS[$s]} -> $OUTDIR"
