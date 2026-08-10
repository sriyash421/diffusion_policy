#!/bin/bash
# Compact health report for the three PushT search training runs.
# Pinned to this session's job IDs; edit JOBS if a run is requeued with a new ID.
set -uo pipefail

LOGDIR=/gscratch/robotics/harine/slurm_logs
# run dirs live on /gscratch too (see README_pusht.md); override with DP_OUTPUT_ROOT.
OUTROOT=${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}
# id:label  (label has no spaces)
JOBS=(
  "37862498:verifier"
  "37862501:subgoal+verifier"
  "37862600:subgoal-only"
)

echo "===== pusht search monitor @ $(date '+%Y-%m-%d %H:%M:%S') ====="
# Run dirs now live entirely on /gscratch, so nothing training-related should be growing
# on home (10G hard quota). This checks the invariant rather than the old footprint: if
# data/outputs is non-empty, some run escaped $DP_OUTPUT_ROOT and will eat the quota.
# Bounded by timeout so a slow GPFS scan never hangs the monitor.
home_m=$(timeout 20 du -sm data/outputs 2>/dev/null | cut -f1)
if [ -n "$home_m" ] && [ "$home_m" -ge 50 ]; then
  echo "-- *** WARNING: ${home_m}MB in data/outputs on HOME (10240MB quota) --" \
       "a run is not using \$DP_OUTPUT_ROOT=$OUTROOT ***"
fi
out_m=$(timeout 20 du -sm "$OUTROOT" 2>/dev/null | cut -f1)
echo "-- run output on /gscratch: ${out_m:-?}MB  ($OUTROOT)"
echo "-- queue --"
squeue -u harine -o "%.10i %.22j %.9P %.9T %.11M %.11L %R" \
  | grep -E "JOBID|pusht_search" || echo "  (no pusht_search jobs in queue)"

echo "-- per-run progress --"
for entry in "${JOBS[@]}"; do
  id=${entry%%:*}; label=${entry##*:}
  st=$(squeue -h -j "$id" -o "%T %M %R" 2>/dev/null)
  [ -z "$st" ] && st="$(sacct -n -j "$id" -o State%-20 2>/dev/null | head -1 | tr -d ' ')"
  f="$LOGDIR/pusht_search_train-$id.out"
  printf '  [%-16s] job %s  %s\n' "$label" "$id" "$st"
  if [ ! -f "$f" ]; then echo "      (no log file)"; continue; fi
  # collapse tqdm carriage returns to newlines, then pull the last meaningful line
  clean=$(tr '\r' '\n' < "$f")
  # only surface RECENT errors (last ~1500 lines) so resolved/stale errors stop alarming
  err=$(printf '%s\n' "$clean" | tail -1500 | grep -aiE "traceback|error|exception|cuda out of memory|quota" | tail -1)
  prog=$(printf '%s\n' "$clean" | grep -aiE "epoch [0-9]+|step|loss=|nrmse|test_mean_score|rollout" | grep -av '?it/s\]$' | tail -1)
  [ -z "$prog" ] && prog=$(printf '%s\n' "$clean" | grep -av '^$' | tail -1)
  echo "      progress: ${prog:0:180}"
  [ -n "$err" ] && echo "      ERROR:    ${err:0:180}"
done

# --- eval watchers: keep one alive per run (ckpt jobs TIMEOUT at ~9h; resubmit) -------
# Run dirs are DISCOVERED, not hand-listed. eval_watchers.tsv used to be edited by hand and
# went stale: it pointed at three home-relative data/outputs/... dirs that no longer exist,
# and this loop then resubmitted a 13-day watcher against each dead path -- a GPU held
# indefinitely producing nothing. The run dir is now identity-keyed
# ($OUTROOT/<exp>/<task>/<trainer>/ctx-*_corrupt-*_seed-*), so a glob is authoritative and
# the TSV is only a job-id cache.
STATE="$(dirname "$0")/eval_watchers.tsv"
echo "-- eval watchers --"
declare -A KNOWN_JID=()
if [ -f "$STATE" ]; then
  while IFS=$'\t' read -r l d j; do [ -n "$d" ] && KNOWN_JID["$d"]="$j"; done < "$STATE"
fi
tmp=$(mktemp)
shopt -s nullglob
run_dirs=("$OUTROOT"/*/*/*/ctx-*_corrupt-*_seed-*)
if [ ${#run_dirs[@]} -eq 0 ]; then
  echo "  (no identity-keyed run dirs under $OUTROOT)"
fi
for run_dir in "${run_dirs[@]}"; do
    [ -d "$run_dir/checkpoints" ] || continue   # nothing to evaluate yet
    label=$(basename "$run_dir")
    jid=${KNOWN_JID["$run_dir"]:-0}
    st=$(squeue -h -j "${jid:-0}" -o "%T" 2>/dev/null)
    if [ "$st" != "RUNNING" ] && [ "$st" != "PENDING" ]; then
      new=$(sbatch --parsable --time=13-00:00:00 \
              "$(dirname "$0")/eval_watch_pusht_search.sbatch" "$run_dir" 2>/dev/null)
      if [ -n "$new" ]; then
        echo "  [$label] watcher was ${st:-GONE} -> resubmitted as $new"
        jid=$new; st="SUBMITTED"
      else
        echo "  [$label] watcher ${st:-GONE}; resubmit FAILED"
      fi
    fi
    # latest evaluated checkpoint / best-so-far from the run-level index
    bj="$run_dir/bon_search/best.json"
    best=$( [ -f "$bj" ] && python -c "
import json,sys
d=json.load(open('$bj'))
sr=d.get('success_rate');
peak=max(sr) if isinstance(sr,list) else sr
n=d.get('n'); nmax=(max(n) if isinstance(n,list) else n)
print('step=%s peak_success=%.3f @n<=%s' % (d.get('step'), peak, nmax))
" 2>/dev/null )
    echo "  [$label] job $jid $st  ${best:+best: $best}"
    printf '%s\t%s\t%s\n' "$label" "$run_dir" "$jid" >> "$tmp"
done
mv "$tmp" "$STATE"
