#!/bin/bash
# Launch the n=16 slot-weight-decay sweep: 3 feedback mechanisms x 2 slot_weight_decay
# values, all CLEAN (corrupt_obs: False), 100 demos, seed 42, 100k gradient steps.
#
#   feedback              search_context   arm
#   distance              value            value
#   subgoal + distance    subgoal_value    subgoal-value
#   subgoal               subgoal          subgoal-chosen4value
#
# crossed with slot_weight_decay in {1.0, 0.9}. 1.0 is uniform -- bit-identical to the
# objective every previous run used. 0.9 weights the per-candidate-slot loss geometrically
# back from the last slot (w_k proportional to 0.9^(K-1-k), renormalized to mean 1), so the
# fully-conditioned slot carries ~1.96 and the first ~0.4.
#
# WHY THAT PAIRS WITH THIS EVAL. slot_weight_decay weights slots; the eval's six criteria
# select by slot (last, 8th-from-last, argmax/softmax over all 16 or the trailing 8). So the
# training axis and the read-out axis are the same axis -- that is the point, not a confound,
# but say so in any writeup.
#
# Everything else is the config default and is NOT overridden here: n_candidates 16,
# corrupt_obs False, n_demos 100, selection argmax, context_decay 1.0, the shared LR
# schedule, use_ema. Overriding a default on the command line is how two runs end up
# differing in something nobody recorded.
#
# NOTE run_name carries `swd-`, so the two decay variants of an arm get DIFFERENT output
# directories. Without that they would share one, and `training.resume: True` would make the
# second launch silently continue the first.
#
# Usage:
#   bash scripts/slurm/launch_swd_sweep.sh          # submit
#   DRY=1 bash scripts/slurm/launch_swd_sweep.sh    # print the commands only
set -euo pipefail

cd /mmfs1/home/harine/diffusion_policy_standalone
DRY="${DRY:-}"
# Finite and under the next maintenance reservation -- an unlimited job is parked with
# Reason=ReqNodeNotAvail and never starts. Check with:
#   scontrol show reservation | grep -A1 Maintenance
TLIM="${TLIM:-5-00:00:00}"

# search_context|arm
CELLS=(
  "value|value"
  "subgoal_value|subgoal-value"
  "subgoal|subgoal-chosen4value"
)
SWDS=(1.0 0.9)

for cell in "${CELLS[@]}"; do
  ctx="${cell%%|*}"; arm="${cell##*|}"
  for swd in "${SWDS[@]}"; do
    # pick_gpu.sh reports a free (account, partition) pair on robotics/weirdlab. The ACCOUNT
    # is e.g. gpu-l40-weirdlab while the PARTITION is gpu-l40 -- passing the account as the
    # partition fails with "invalid partition specified".
    read -r acct part < <(bash scripts/slurm/pick_gpu.sh)
    name="swd_${arm}_${swd}"
    if [ -n "$DRY" ]; then
      echo "would submit [$acct/$part] $name:"
      echo "    search_context=$ctx arm=$arm slot_weight_decay=$swd"
      continue
    fi
    jid=$(sbatch --parsable --account="$acct" --partition="$part" --time="$TLIM" \
            --job-name="$name" \
            scripts/slurm/train_pusht_search.sbatch \
            "search_context=$ctx" "arm=$arm" "slot_weight_decay=$swd")
    echo "submitted $jid [$acct/$part] $name"
  done
done

echo
echo "Submitted. Next:"
echo "  squeue -u \$USER"
echo "  bash scripts/slurm/monitor_pusht_search.sh     # confirms nothing escaped DP_OUTPUT_ROOT"
echo "  bash scripts/slurm/submit_criteria_sweep.sh    # once checkpoints appear (safe to re-run)"
