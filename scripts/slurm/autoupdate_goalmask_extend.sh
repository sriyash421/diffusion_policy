#!/bin/bash
# Extend the two goal-mask arms to 300k, once each has finished its current 100k budget.
#
#   setsid nohup bash scripts/slurm/autoupdate_goalmask_extend.sh >/dev/null 2>&1 &
#
# WHY A LOOP. The arms are mid-run with max_gradient_steps=100000 already loaded, so the new
# budget cannot be applied until they stop. This waits, then resubmits; `training.resume:
# True` continues from latest.ckpt and cfg is NOT restored from the payload, so the 300k
# wins. Raising the budget is safe mid-schedule because `decay_then_constant` is a pure
# function of the absolute step and is flat at 0.1x base past 80k.
#
# run_goalmask_geometric.sh is idempotent: it skips an arm whose watcher is live and one that
# already has the step_0300000.ckpt, so this only ever submits the tail.
#
# THE WALL CLOCK IS RECOMPUTED EVERY TICK from the reservation calendar. 300k needs ~4 days
# at the measured k=16 pace and the Sept 8 maintenance falls inside that, so these arms WILL
# need more than one resume cycle -- and a job asking for more time than remains before a
# reservation does not schedule at all, it just pends.
set -uo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone

INTERVAL=${INTERVAL:-1800}
HOURS=${HOURS:-192}
MAX_STEPS=${MAX_STEPS:-300000}
LOG=/gscratch/robotics/harine/slurm_logs/goalmask_extend.log
PY=${DP_PY:-/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python}

deadline=$(( $(date +%s) + HOURS * 3600 ))
echo "[$(date '+%F %T')] start: extend gm arms to ${MAX_STEPS}, every ${INTERVAL}s for ${HOURS}h" >> "$LOG"
while [ "$(date +%s)" -lt "$deadline" ]; do
    tl=$(bash scripts/slurm/safe_time_limit.sh 3-00:00:00)
    out=$(MAX_STEPS="$MAX_STEPS" TIME_LIMIT="$tl" EXCLUDE_NODES="${EXCLUDE_NODES:-g3070}" \
          SUBMIT=1 DP_PY="$PY" bash scripts/run_goalmask_geometric.sh 2>&1 | tail -3 | tr '\n' ' ')
    echo "[$(date '+%F %T')] time-limit=$tl | $out" >> "$LOG"
    sleep "$INTERVAL"
done
echo "[$(date '+%F %T')] done -- relaunch if the arms have not reached ${MAX_STEPS}" >> "$LOG"
