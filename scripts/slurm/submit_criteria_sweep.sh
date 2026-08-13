#!/bin/bash
# Submit the selection-criteria eval matrix: every run x every checkpoint on the eval grid.
#
# One job per (run, checkpoint); all six criteria run inside it. 6 runs x 12 checkpoints
# = 72 jobs on the preemptible `ckpt` partition, which keeps the guaranteed
# robotics/weirdlab GPUs free for the training jobs.
#
# The grid is 1k, 5k, then every 10k to 100k -- exactly what training writes
# (training.checkpoint_every: 10000 plus training.checkpoint_steps: [1000, 5000]), so a
# missing file here means that checkpoint has not been reached yet, not that the grid is
# wrong. Missing checkpoints are SKIPPED with a notice rather than submitted, so the script
# is safe to re-run as training progresses and will fill in only what has appeared.
#
# Usage:
#   bash scripts/slurm/submit_criteria_sweep.sh            # submit
#   DRY=1 bash scripts/slurm/submit_criteria_sweep.sh      # list what would be submitted
set -euo pipefail

ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"
TRAINER_DIR="$ROOT/pusht_search/pusht_image_search/outer_inner"
DRY="${DRY:-}"

# The six cells of the sweep, by run-directory name. These are exactly what
# train_pusht_diffusion_search.yaml's run_name resolves to for each (arm, slot_weight_decay)
# -- `swd-` is part of the name precisely so the two decay variants of an arm cannot land in
# the same directory.
RUNS=(
  "value_corrupt-False_swd-1.0_demos-100_seed-42"
  "value_corrupt-False_swd-0.9_demos-100_seed-42"
  "subgoal-value_corrupt-False_swd-1.0_demos-100_seed-42"
  "subgoal-value_corrupt-False_swd-0.9_demos-100_seed-42"
  "subgoal-chosen4value_corrupt-False_swd-1.0_demos-100_seed-42"
  "subgoal-chosen4value_corrupt-False_swd-0.9_demos-100_seed-42"
)

STEPS=(1000 5000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000)

n_sub=0
n_missing=0
for run in "${RUNS[@]}"; do
  for step in "${STEPS[@]}"; do
    ckpt=$(printf '%s/%s/checkpoints/step_%07d.ckpt' "$TRAINER_DIR" "$run" "$step")
    if [ ! -f "$ckpt" ]; then
      n_missing=$((n_missing + 1))
      continue
    fi
    # Already done? All six criteria present means nothing to redo. Re-running a partially
    # evaluated checkpoint is harmless (rows merge under a lock, keyed on step+criterion),
    # so the check only saves GPU time.
    done_dir=$(printf '%s/%s/criteria_search/step_%07d' "$TRAINER_DIR" "$run" "$step")
    if [ -d "$done_dir" ] && [ "$(ls "$done_dir"/*.json 2>/dev/null | wc -l)" -ge 6 ]; then
      continue
    fi
    if [ -n "$DRY" ]; then
      echo "would submit: $run @ $step"
    else
      jid=$(sbatch --parsable \
              --job-name="crit_${run%%_*}_${step}" \
              scripts/slurm/eval_criteria_pusht_search.sbatch "$ckpt")
      echo "submitted $jid: $run @ $step"
    fi
    n_sub=$((n_sub + 1))
  done
done

echo
echo "$n_sub job(s) $([ -n "$DRY" ] && echo 'would be submitted' || echo submitted); \
$n_missing checkpoint(s) not written yet (re-run this script as training progresses)."
