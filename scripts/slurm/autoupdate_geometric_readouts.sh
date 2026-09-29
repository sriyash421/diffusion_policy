#!/bin/bash
# Keep the geometric-split eval watchers topped up while the 15 training runs produce
# checkpoints. The watcher has a 12h wall clock and training runs for days, so a single
# submission covers only the first checkpoints.
#
#   setsid nohup env SUBMIT=1 bash scripts/slurm/autoupdate_geometric_readouts.sh >/dev/null 2>&1 &
#
# submit_geometric_readouts.sh is idempotent -- it skips a (run, rule) whose watcher is
# already queued and skips runs with no checkpoints -- so this only ever submits the tail.
# A SEPARATE loop file rather than an edit to a live one: bash reads a script incrementally,
# so editing a running loop can corrupt its execution mid-iteration.
set -uo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone

INTERVAL=${INTERVAL:-3600}
HOURS=${HOURS:-120}                 # 15 runs x 100k steps; bounded so it cannot outlive them
LOG=/gscratch/robotics/harine/slurm_logs/geometric_readout_autoupdate.log

deadline=$(( $(date +%s) + HOURS * 3600 ))
echo "[$(date '+%F %T')] start: every ${INTERVAL}s for ${HOURS}h" >> "$LOG"
while [ "$(date +%s)" -lt "$deadline" ]; do
    out=$(SUBMIT=1 bash scripts/slurm/submit_geometric_readouts.sh 2>&1 | tail -2 | tr '\n' ' ')
    n_ev=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c '^ev_.*split-' || true)
    echo "[$(date '+%F %T')] $out | ev_ split jobs in queue=$n_ev" >> "$LOG"
    sleep "$INTERVAL"
done
echo "[$(date '+%F %T')] done -- re-launch if checkpoints are still landing" >> "$LOG"
