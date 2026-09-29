#!/bin/bash
#SBATCH --job-name=slotwts_cand
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%A_%a.out
#SBATCH --array=0-15
#
# Per-candidate verifier values for the round-2 slot-weight arms, at n=16 over the 50 test
# episodes. Answers "is candidate 15 better than candidate 0" -- see the analysis in
# slot_context_aug27.md, produced by scripts/slot_context_analysis.py from these dumps.
#
# TWO TIERS, and they are not interchangeable:
#   A  step 10000, every arm -- the PAIRED tier. One identical step, so a cross-arm
#      comparison of slot effects is not confounded with training duration.
#   B  each arm's own peak `final_pass` success at n=16, later checkpoint breaking ties.
#      A rule fixed in advance, applied to disk as of 2026-08-27 (six arms were still
#      training). Under final_pass at n=16 the deployed candidate IS slot 15, so this
#      selects on the read-out that most directly depends on slot 15 being good --
#      deliberately, but it does make tier B mildly circular, which tier A is not.
#      Tier B steps DIFFER PER ARM, so tier B is a within-arm check only.
#
# Re-running after training completes is a one-line edit: replace STEP_B below with the
# fresh output of `python scripts/pick_final_pass_step.py`.
#
# Read-only with respect to every training run: loads a checkpoint and writes only into
# OUTDIR, so it is safe against a directory a live trainer is still writing to.
#
#   sbatch scripts/slurm/dump_slot_wts_grid.sbatch
set -euo pipefail
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search
OUTROOT=/mmfs1/home/harine/diffusion_policy_standalone/analysis/aug27_slot_wts
K16=outer_inner/value_k16_ver-armTn%s_corrupt-False_demos-30_seed-42

# The six round-2 arms, then the two i.i.d. nulls. The nulls are not baselines OF the
# slot statistic -- they ARE its null distribution, measured on real data with the real tie
# structure: stk1 has max_actions==1 (empty-context branch) and unetbc's predict_action
# discards the search context outright. If a k=16 arm's slot-15-over-slot-0 delta does not
# beat theirs, the context bought nothing.
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
# Tier B, per arm. stk1 has no bon_search_sel-final_pass readouts at all, so the rule is
# undefined for it and it takes 60000 by convention -- the modal tier-B step, which pairs it
# with the other null.
STEP_B=(60000 30000 40000 50000 60000 60000 60000 60000)

# unetbc was trained before the verifier cutover and carries no verifier_tag, so it
# resolves to the pre-cutover default `t_goal`. Scored natively it would rank on a
# DIFFERENT RULE than every other arm here and would be worthless as a null, so it is
# overridden -- the same override its own bon_search_ver-armTn/ eval subdir uses. Every
# other arm is armTn-native and must NOT be overridden.
VERS=("" "" "" "" "" "" "" armTn)

a=$(( SLURM_ARRAY_TASK_ID % 8 ))
tier=$(( SLURM_ARRAY_TASK_ID / 8 ))          # 0 = tier A (10k), 1 = tier B (per-arm peak)
if [ "$tier" -eq 0 ]; then STEP=10000; TIER=A; else STEP="${STEP_B[$a]}"; TIER=B; fi

LABEL="${LABELS[$a]}"
CKPT=$(printf '%s/%s/checkpoints/step_%07d.ckpt' "$ROOT" "${RUNS[$a]}" "$STEP")
OUTDIR=$(printf '%s/%s_step%07d' "$OUTROOT" "$LABEL" "$STEP")

VER_ARG=()
[ -n "${VERS[$a]}" ] && VER_ARG=(--verifier-value "${VERS[$a]}")

echo "=== task $SLURM_ARRAY_TASK_ID: tier $TIER  $LABEL  step $STEP  ${VER_ARG[*]-}"
[ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }

# --n 16 is REQUIRED, not a default: at n <= max_actions the slot index IS the context
# length, and past 16 the rolling window makes the index stop being a slot id.
# --pair-seeds (the default, passed explicitly so the command is self-describing) is what
# makes the arms comparable episode-by-episode.
# --per-step-json writes all 50 episodes rather than a pre-chosen one: picking "the"
# episode before seeing the data would be choosing the example to fit the story.
PYTHONPATH=$PWD python -u scripts/dump_candidate_scores.py \
  -c "$CKPT" \
  --arm "$LABEL" \
  --out-dir "$OUTDIR" \
  --n 16 \
  --episodes 50 \
  --split test \
  --seed 42 \
  --per-step-json \
  --per-step-episodes 0 \
  --pair-seeds \
  --device cuda:0 \
  ${VER_ARG[@]+"${VER_ARG[@]}"} \
  "$@"

echo "=== done tier $TIER $LABEL step $STEP -> $OUTDIR"
