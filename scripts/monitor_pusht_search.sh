#!/bin/bash
# Compact health report for the three PushT search training runs.
# Pinned to this session's job IDs; edit JOBS if a run is requeued with a new ID.
set -uo pipefail

LOGDIR=/mmfs1/home/harine/slurm_logs
# id:label  (label has no spaces)
JOBS=(
  "37862498:verifier"
  "37862501:subgoal+verifier"
  "37862600:subgoal-only"
)

echo "===== pusht search monitor @ $(date '+%Y-%m-%d %H:%M:%S') ====="
# home is a 10G hard quota; checkpoints are symlinked to /gscratch (du won't follow
# them), so summing data/outputs = our real on-home footprint (wandb + eval jsons).
# Bounded by timeout so a slow GPFS scan never hangs the monitor.
out_m=$(timeout 20 du -sm data/outputs 2>/dev/null | cut -f1)
if [ -n "$out_m" ]; then
  msg="-- data/outputs on home: ${out_m}MB (10240MB user quota; checkpoints are on /gscratch)"
  [ "$out_m" -ge 8000 ] && msg="$msg  *** WARNING: approaching quota ***"
  echo "$msg"
else
  echo "-- data/outputs usage: (scan skipped/timed out)"
fi
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
STATE="$(dirname "$0")/eval_watchers.tsv"
if [ -f "$STATE" ]; then
  echo "-- eval watchers --"
  tmp=$(mktemp)
  while IFS=$'\t' read -r label run_dir jid; do
    [ -z "$label" ] && continue
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
  done < "$STATE"
  mv "$tmp" "$STATE"
fi
