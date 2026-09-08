#!/bin/bash
#SBATCH --job-name=vrank
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%A_%a.out
#SBATCH --array=0-15
#
# Does the armTn verifier rank the EXPERT action above the policy's own candidates?
# Same 8 arms x 2 tiers as scripts/slurm/dump_slot_wts_grid.sbatch, so a row here lines up
# 1:1 with a row of slot_context_aug27_tier*.md.
#
# The verifier's ORDERING is what best-of-n consumes; every slot statistic measures
# candidates against each other under that ordering and none of them checks the ordering
# itself. This is that check, against the only ground truth available -- the demonstration.
#
#   sbatch scripts/slurm/verifier_ranks_expert_grid.sbatch
set -euo pipefail
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search
OUTROOT=/mmfs1/home/harine/diffusion_policy_standalone/analysis/aug27_verifier_ranks
K16=outer_inner/value_k16_ver-armTn%s_corrupt-False_demos-30_seed-42

LABELS=(lin100-l2 lin100-l2tol1 geo735-l2 geo735-l2tol1 curr-lin100-l2 curr-lin100-l2tol1 stk1 unetbc)
RUNS=(
  "$(printf "$K16" _sw-lin100-l2)"
  "$(printf "$K16" _sw-lin100-l2tol1)"
  "$(printf "$K16" _sw-geo735-l2)"
  "$(printf "$K16" _sw-geo735-l2tol1)"
  "$(printf "$K16" _sw-curr-lin100-l2)"
  "$(printf "$K16" _sw-curr-lin100-l2tol1)"
  offline/value_k1_ver-armTn_corrupt-False_demos-30_seed-42
  unet_bc/unetbc_demos-30_seed-42
)
STEP_B=(60000 30000 40000 50000 60000 60000 60000 60000)
# unetbc is natively t_goal (no verifier_tag); overridden so its candidates are ranked by
# the same rule as every other arm's. See dump_slot_wts_grid.sbatch.
VERS=("" "" "" "" "" "" "" armTn)

a=$(( SLURM_ARRAY_TASK_ID % 8 ))
tier=$(( SLURM_ARRAY_TASK_ID / 8 ))
if [ "$tier" -eq 0 ]; then STEP=10000; TIER=A; else STEP="${STEP_B[$a]}"; TIER=B; fi
LABEL="${LABELS[$a]}"
CKPT=$(printf '%s/%s/checkpoints/step_%07d.ckpt' "$ROOT" "${RUNS[$a]}" "$STEP")
mkdir -p "$OUTROOT"
OUT=$(printf '%s/%s_step%07d.json' "$OUTROOT" "$LABEL" "$STEP")

VER_ARG=()
[ -n "${VERS[$a]}" ] && VER_ARG=(--verifier-value "${VERS[$a]}")

echo "=== task $SLURM_ARRAY_TASK_ID: tier $TIER  $LABEL  step $STEP"
[ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }

# 50 test episodes x 8 decision points. Points are sampled only where a full obs window AND
# a full expert chunk exist, so no padded/repeated action ever lands on the ground-truth
# side of the comparison.
PYTHONPATH=$PWD python -u scripts/verifier_ranks_expert.py \
  -c "$CKPT" --arm "$LABEL" --n 16 \
  --episodes 50 --per-episode 8 --split test \
  --batch 8 --seed 42 --device cuda:0 --out "$OUT" \
  ${VER_ARG[@]+"${VER_ARG[@]}"} \
  "$@"

echo "=== done tier $TIER $LABEL step $STEP -> $OUT"
