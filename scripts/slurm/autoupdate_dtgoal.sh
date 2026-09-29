#!/bin/bash
# Keep the d_t_goal round moving: top up the eval jobs and regenerate BOTH docs.
#
#   setsid nohup bash scripts/slurm/autoupdate_dtgoal.sh > /dev/null 2>&1 &
#
# Replaces autoupdate_slot_norm.sh + autoupdate_readouts.sh, which ran as two loops. Both
# docs are rebuilt because the armTn evals are still draining on the checkpoints written
# before that round was stopped -- armTn's tables keep filling in though its training ended.
#
# submit_slot_norm_readouts.sh is idempotent on BOTH axes now: it checks the real output
# directory (bon_search_sel-<rule>[_ver-<value>]/, which it previously got wrong for any
# group passing --verifier-value, so those were resubmitted every tick) and it checks the
# live queue by a job name that now includes the verifier.
#
# BOUNDED (default 72h). Re-launch if training is still going.
set -uo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone

INTERVAL=${INTERVAL:-1800}
HOURS=${HOURS:-72}
PY=/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python
LOG=/gscratch/robotics/harine/slurm_logs/dtgoal_autoupdate.log

deadline=$(( $(date +%s) + HOURS * 3600 ))
echo "[$(date '+%F %T')] start: every ${INTERVAL}s for ${HOURS}h" >> "$LOG"
while [ "$(date +%s)" -lt "$deadline" ]; do
    subs=$(SUBMIT=1 bash scripts/slurm/submit_slot_norm_readouts.sh 2>&1 \
           | grep -c '^submitted ' || true)
    SUBMIT=1 bash scripts/slurm/submit_30_100_watchers.sh > /dev/null 2>&1
    d1=$("$PY" scripts/build_slot_norm_doc.py 2>&1 | tail -1)
    d2=$("$PY" scripts/build_dtgoal_doc.py 2>&1 | tail -1)
    tr=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c '^tr_' || true)
    echo "[$(date '+%F %T')] armTn: $d1 | dtgoal: $d2 | readouts+=$subs training=$tr" >> "$LOG"
    sleep "$INTERVAL"
done
echo "[$(date '+%F %T')] done -- re-launch if evals are still landing" >> "$LOG"
