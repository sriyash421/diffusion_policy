#!/bin/bash
# Clear an eval backlog in PARALLEL on the ckpt partition -- one job per checkpoint --
# instead of letting a single --watch job grind them serially at ~1h each.
#
# Why the watcher has to be replaced rather than supplemented. `--watch` seeds its `seen`
# set from success_curves.jsonl ONCE at startup and then keeps it in memory
# (eval_search_pusht.py:768). It never re-reads the file, so a checkpoint finished by a
# parallel job after the watcher started is invisible to it and gets evaluated a second
# time. Running both at once therefore duplicates most of the backlog. So for each run:
#
#   1. cancel its live watcher AND the chained successor (the successor is
#      --dependency=afterany, so cancelling the live one would otherwise release it
#      immediately and it would re-grind the same backlog)
#   2. submit one job per unevaluated checkpoint, all at once
#   3. submit a fresh watcher that depends on ALL of them, so it starts only once the
#      backlog is gone, re-seeds `seen` from disk, and covers only checkpoints training
#      writes from then on
#
# Backlog means "not covered at every n up to --max-n" -- the same test the watcher uses,
# so a checkpoint with an n<=32 curve still counts as outstanding for an n<=64 sweep.
#
#   bash scripts/slurm/submit_backlog_evals.sh                    # still-training runs
#   DRY=1 bash scripts/slurm/submit_backlog_evals.sh              # print only
#   RUNS='subgoal-only_k4_cd0.9_corrupt-False_demos-100_seed-42' bash scripts/...
#   MAX_N=16 bash scripts/slurm/submit_backlog_evals.sh           # cheaper: cap the curve
set -uo pipefail

ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search/offline
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python
MAX_N="${MAX_N:-64}"
DRY="${DRY:-}"
# Guard against a runaway submit. The BC runs hold ~90 checkpoints evaluated at n=1 only,
# which count as "backlog" for an n<=64 sweep -- a bare glob over every run would submit
# ~120 jobs. Default scope is the runs still training, which is where a backlog actually
# accumulates; pass RUNS= explicitly for anything else.
MAX_JOBS="${MAX_JOBS:-40}"

if [ -n "${RUNS:-}" ]; then
    runs=($RUNS)
else
    runs=($(squeue -u "$USER" -h -o "%j" | grep '^tr_' | sed 's/^tr_//'))
fi
[ ${#runs[@]} -gt 0 ] || { echo "no runs selected"; exit 0; }

total=0
for run in "${runs[@]}"; do
    d="$ROOT/$run"
    [ -d "$d/checkpoints" ] || { echo "  no checkpoints, skipping  $run"; continue; }

    mapfile -t backlog < <("$PY" - "$d" "$MAX_N" <<'EOF'
import json, pathlib, re, sys
d, max_n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
want = {n for n in (1, 2, 4, 8, 16, 32, 64) if n <= max_n}
steps = sorted(int(re.search(r'step_(\d+)\.ckpt', f.name).group(1))
               for f in (d / 'checkpoints').glob('step_*.ckpt'))
done = set()
jl = d / 'bon_search' / 'success_curves.jsonl'
if jl.is_file():
    for line in jl.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if want <= set(r['n']):
                done.add(r['step'])
print('\n'.join(str(s) for s in steps if s not in done))
EOF
)
    # Drop empty elements. An empty '\n'.join() still emits one newline, so mapfile hands
    # back a single empty string and a run with NO backlog reads as one checkpoint -- which
    # cancelled its watchers, submitted nothing, and left the run permanently unwatched.
    filtered=()
    for s in "${backlog[@]}"; do [ -n "$s" ] && filtered+=("$s"); done
    backlog=("${filtered[@]}")
    n=${#backlog[@]}
    [ "$n" -gt 0 ] || { echo "  no backlog  $run"; continue; }
    if [ $((total + n)) -gt "$MAX_JOBS" ]; then
        echo "  SKIPPING $run: $n jobs would exceed MAX_JOBS=$MAX_JOBS (raise it deliberately)"
        continue
    fi
    echo "=== $run: $n checkpoint(s)"

    live=($(squeue -u "$USER" -h -o "%i %j" | awk -v n="ev_$run" '$2==n {print $1}'))
    if [ ${#live[@]} -gt 0 ]; then
        echo "  cancelling watcher(s): ${live[*]}"
        [ -n "$DRY" ] || scancel "${live[@]}"
    fi

    deps=()
    for step in "${backlog[@]}"; do
        ckpt=$(printf '%s/checkpoints/step_%07d.ckpt' "$d" "$step")
        [ -f "$ckpt" ] || continue
        if [ -n "$DRY" ]; then
            echo "  would submit  step=$step  n<=$MAX_N"
            total=$((total+1)); continue
        fi
        jid=$(sbatch --parsable --time=8:00:00 \
                --job-name="bk_$(echo "$run" | sed 's/_demos-100//;s/_seed-42//')_$step" \
                "$HERE/eval_ckpt_pusht_search.sbatch" "$ckpt" --max-n "$MAX_N")
        deps+=("$jid"); total=$((total+1))
    done
    [ -n "$DRY" ] && continue
    echo "  submitted ${#deps[@]} eval job(s)"

    # fresh watcher, gated on the whole backlog so it re-seeds `seen` from a complete jsonl
    dep=$(IFS=:; echo "${deps[*]}")
    wid=$(sbatch --parsable --account=robotics --partition=ckpt \
            --dependency=afterany:"$dep" --job-name="ev_${run}" \
            "$HERE/eval_watch_pusht_search.sbatch" "$d" --max-n "$MAX_N")
    echo "  watcher $wid queued behind them"
done
echo "${DRY:+DRY-RUN: }$total eval job(s) total"
