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

# Per-job ask, from train_pusht_search.sbatch's #SBATCH lines. Both matter.
JOB_GPUS=1
JOB_CPUS=8

# CAPACITY IS ALLOCATED ONCE, UP FRONT, AND DECREMENTED AS WE SUBMIT.
#
# Do NOT call pick_gpu.sh per job. It reports the single best free (account, partition)
# from a fresh `hyakalloc`, and a job submitted a second ago is still PENDING and has not
# reduced anything hyakalloc reports -- so six calls in a loop return the SAME partition
# six times and pile the whole sweep onto one account.
#
# It also considers only GPUs. That is the half that bit: robotics/gpu-a40 showed 4 free
# GPUs but 26 CPUs TOTAL, and at 8 CPUs a job the fourth submission was rejected with
# AssocGrpCpuLimit while GPUs were still nominally free. Track both.
mapfile -t SLOTS < <(hyakalloc | sed 's/│/|/g' | awk -F'|' -v jg="$JOB_GPUS" -v jc="$JOB_CPUS" '
  {
    for (i = 1; i <= NF; i++) gsub(/^ +| +$/, "", $i)
    if ($2 != "") acct = $2
    if ($3 != "") part = $3
    if ($7 == "FREE" && ($6 + 0) >= jg && ($4 + 0) >= jc &&
        (acct == "robotics" || acct == "weirdlab")) {
      # prefer robotics, then whichever slot holds the most jobs
      cap = ($6 + 0); if (int(($4 + 0) / jc) < cap) cap = int(($4 + 0) / jc)
      print (acct == "robotics" ? 0 : 1), cap, part "-" acct, part, cap
    }
  }' | sort -k1,1n -k2,2nr | awk '{print $3, $4, $5}')

if [ ${#SLOTS[@]} -eq 0 ]; then
  echo "FATAL: no robotics/weirdlab partition has both a free GPU and ${JOB_CPUS} free CPUs." >&2
  echo "       Check with hyakalloc; either wait, or lower --cpus-per-task." >&2
  exit 1
fi
echo "capacity found (account partition jobs_it_can_hold):"
printf '  %s\n' "${SLOTS[@]}"
echo

slot_i=0
PICK_ACCT=; PICK_PART=
# Sets PICK_ACCT/PICK_PART and decrements that slot. It assigns to GLOBALS rather than
# echoing, because the caller would otherwise have to read it through `< <(take_slot)` --
# process substitution runs the function in a SUBSHELL, so the decrement would be discarded
# and every job would be handed the same first slot. (Which is the same class of bug as
# calling pick_gpu.sh per job: capacity bookkeeping that silently does not persist.)
take_slot () {
  while [ "$slot_i" -lt "${#SLOTS[@]}" ]; do
    read -r a p cap <<<"${SLOTS[$slot_i]}"
    if [ "$cap" -gt 0 ]; then
      SLOTS[$slot_i]="$a $p $((cap - 1))"
      PICK_ACCT="$a"; PICK_PART="$p"
      return 0
    fi
    slot_i=$((slot_i + 1))
  done
  return 1
}

for cell in "${CELLS[@]}"; do
  ctx="${cell%%|*}"; arm="${cell##*|}"
  for swd in "${SWDS[@]}"; do
    # The ACCOUNT is e.g. gpu-l40-weirdlab while the PARTITION is gpu-l40 -- passing the
    # account as the partition fails with "invalid partition specified".
    if ! take_slot; then
      echo "OUT OF CAPACITY before submitting $arm swd=$swd. Submitting it anyway would" >&2
      echo "  queue it behind a multi-day run. Wait for capacity and re-run -- but check" >&2
      echo "  squeue first: this script does not know which cells are already submitted." >&2
      exit 1
    fi
    acct="$PICK_ACCT"; part="$PICK_PART"
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
