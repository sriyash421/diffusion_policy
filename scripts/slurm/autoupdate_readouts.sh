#!/bin/bash
# Top up the extra read-out evals (argmax / final_pass / cand-8) as the last checkpoints of
# the two k=16 variant arms land.
#
#   setsid nohup bash scripts/slurm/autoupdate_readouts.sh > /dev/null 2>&1 &
#
# A SECOND loop rather than a line added to autoupdate_slot_norm.sh: that one is already
# running, and bash reads a script incrementally, so editing a live one can corrupt its
# execution mid-loop.
#
# submit_slot_norm_readouts.sh is idempotent -- a (run, step, rule) already fully swept on
# disk is skipped -- so this only ever submits the new tail.
#
# BOUNDED (default 24h): training finishes tonight, and the 47 jobs already queued cover
# every checkpoint through step 80k. This exists to catch 90k and 100k.
set -uo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone

INTERVAL=${INTERVAL:-1800}
HOURS=${HOURS:-24}
LOG=/gscratch/robotics/harine/slurm_logs/readout_autoupdate.log

deadline=$(( $(date +%s) + HOURS * 3600 ))
echo "[$(date '+%F %T')] start: every ${INTERVAL}s for ${HOURS}h" >> "$LOG"
while [ "$(date +%s)" -lt "$deadline" ]; do
    out=$(SUBMIT=1 bash scripts/slurm/submit_slot_norm_readouts.sh 2>&1 | tail -3 | tr '\n' ' ')
    n_ro=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c '^ro_' || true)
    echo "[$(date '+%F %T')] $out | ro_ jobs in queue=$n_ro" >> "$LOG"
    sleep "$INTERVAL"
done
echo "[$(date '+%F %T')] done -- re-launch if checkpoints are still landing" >> "$LOG"
