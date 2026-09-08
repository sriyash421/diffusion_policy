#!/bin/bash
# Keep the slot-loss-norm experiment moving without anyone remembering to poke it: every
# INTERVAL seconds, top up the eval watchers and regenerate the doc.
#
#   setsid nohup bash scripts/slurm/autoupdate_slot_norm.sh > /dev/null 2>&1 &
#
# Both halves are idempotent. submit_30_100_watchers.sh skips runs with a live watcher, no
# checkpoints, or a complete sweep -- so this neither doubles up watchers (two on one run
# dir contend for the same success_curve lock) nor resurrects one for a finished arm. The
# doc builder reads only on-disk eval output.
#
# The top-up is what actually delivers "eval every 10k checkpoints": a watcher's wall clock
# is 12h and these arms train for days, so one watcher per run is never enough.
#
# BOUNDED (default 72h) rather than forever -- an unbounded detached loop outlives the
# experiment and keeps sbatching weeks later. Re-launch it if training is still going.
# INTERVAL / HOURS override the defaults. Log: /gscratch/robotics/harine/slurm_logs/.
set -uo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone

INTERVAL=${INTERVAL:-1800}
HOURS=${HOURS:-72}
PY=/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python
LOG=/gscratch/robotics/harine/slurm_logs/slot_norm_autoupdate.log

deadline=$(( $(date +%s) + HOURS * 3600 ))
echo "[$(date '+%F %T')] start: every ${INTERVAL}s for ${HOURS}h" >> "$LOG"
while [ "$(date +%s)" -lt "$deadline" ]; do
    # Only the two l2tol1 arms can be missing evals; the rest of the list is long since
    # complete and reports "nothing to do" in one squeue call, so this is cheap.
    subs=$(SUBMIT=1 bash scripts/slurm/submit_30_100_watchers.sh 2>&1 \
           | grep -c 'submitted job' || true)
    doc=$("$PY" scripts/build_slot_norm_doc.py 2>&1 | tail -1)
    n_tr=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c 'l2tol1' || true)
    echo "[$(date '+%F %T')] $doc | watchers submitted=$subs | l2tol1 jobs=$n_tr" >> "$LOG"
    sleep "$INTERVAL"
done
echo "[$(date '+%F %T')] done -- re-launch if training is still running" >> "$LOG"
