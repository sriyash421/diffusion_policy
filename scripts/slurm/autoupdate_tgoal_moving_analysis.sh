#!/usr/bin/env bash
# Keep the t_goal-moving attention/mode analyses topped up while the seven arms train.
#
#   setsid nohup env SUBMIT=1 bash scripts/slurm/autoupdate_tgoal_moving_analysis.sh >/dev/null 2>&1 &
#
# submit_tgoal_moving_analysis.sh is incremental -- it asks only for steps that have a
# checkpoint but no entry in the arm's analysis json yet, and skips an arm whose job is
# already queued -- so a pass with nothing new costs one squeue and a few file reads.
#
# A SEPARATE loop file rather than an edit to a live one: bash reads a script incrementally,
# so editing a running loop can corrupt its execution mid-iteration. (Same reason
# autoupdate_geometric_readouts.sh is its own file.)
set -uo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone

INTERVAL=${INTERVAL:-1800}
HOURS=${HOURS:-48}          # the slowest arm is ~20h of GPU plus queue; bounded so it
                            # cannot outlive the runs it is watching
LOG=/gscratch/robotics/harine/slurm_logs/tgoal_moving_analysis_autoupdate.log

deadline=$(( $(date +%s) + HOURS * 3600 ))
echo "[$(date '+%F %T')] start: every ${INTERVAL}s for ${HOURS}h" >> "$LOG"
while [ "$(date +%s)" -lt "$deadline" ]; do
    out=$(SUBMIT=1 bash scripts/slurm/submit_tgoal_moving_analysis.sh 2>&1 | tail -2 | tr '\n' ' ')
    # `^tr_` only. The analysis jobs this loop submits are named attn_/mode_<run> and so
    # also contain demos-176_split-mv -- counting those would report the loop's own work as
    # training still to do, and hold the exit condition open.
    tr_left=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c '^tr_.*demos-176_split-mv' || true)
    echo "[$(date '+%F %T')] $out | training arms still queued/running=$tr_left" >> "$LOG"
    # nothing left to analyse once every arm has finished AND no analysis is queued
    if [ "$tr_left" -eq 0 ]; then
        pend=$(bash scripts/slurm/submit_tgoal_moving_analysis.sh 2>/dev/null | grep -c "WOULD SUBMIT" || true)
        if [ "$pend" -eq 0 ]; then
            echo "[$(date '+%F %T')] all arms done and nothing left to analyse; exiting" >> "$LOG"
            break
        fi
    fi
    sleep "$INTERVAL"
done
echo "[$(date '+%F %T')] done -- re-launch if checkpoints are still landing" >> "$LOG"
