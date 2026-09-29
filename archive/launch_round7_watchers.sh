#!/bin/bash
# One eval watcher per Round 7 run. Each polls its run dir and evaluates every new
# step_*.ckpt as training writes it, appending to <run>/bon_search/success_curves.jsonl.
#
# Per-checkpoint n differs by arm, because the two answer different questions:
#   BOTH BC and search: n<=64 per checkpoint, matching the legacy runs so every generation
#   stays comparable. BC was briefly run at n=1 only, which was a mistake twice over: it
#   gave no success-vs-n curve to put beside the search arms, and BC scores 0% binary
#   success at n=1 on EVERY checkpoint, so the val selector was a flat tie across the whole
#   run and best.json returned whichever row happened to come first. Sweeping to n=64 makes
#   success non-degenerate and the selector works normally.
# The large-n tail (128..1024) is NOT done here -- it runs per-n on the val-selected best
# checkpoint via submit_large_n_evals.sh once training has produced enough checkpoints.
#
# ALL evals go to the preemptible `ckpt` partition, never the weirdlab L40 training nodes
# and never the guaranteed robotics A40s -- those are a 4-GPU pool that long evals can
# monopolise. This is safe because eval_search_pusht.py persists after every n, so a
# preemption costs at most the level in flight, not the whole sweep.
#
# The ONE exception is a single-n job at large n (submit_large_n_evals.sh at n=1024): it
# writes nothing until that level finishes, so preemption discards ~8h. Those go to a
# guaranteed partition deliberately -- see the comment there.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search/offline
DRY=${1:-}

# run_name|extra eval args|account|partition
WATCHERS="
bc_demos-25_seed-42|--max-n@64@--idle-exit-sec@900|robotics|ckpt
bc_demos-100_seed-42|--max-n@64@--idle-exit-sec@900|robotics|ckpt
ctx-value_corrupt-False_demos-100_seed-42|--max-n@64|robotics|ckpt
ctx-value_corrupt-True_demos-100_seed-42|--max-n@64|robotics|ckpt
ctx-subgoal_corrupt-False_demos-100_seed-42|--max-n@64|robotics|ckpt
ctx-subgoal_corrupt-True_demos-100_seed-42|--max-n@64|robotics|ckpt
ctx-subgoal_value_corrupt-False_demos-100_seed-42|--max-n@64|robotics|ckpt
ctx-subgoal_value_corrupt-True_demos-100_seed-42|--max-n@64|robotics|ckpt
"

for spec in $WATCHERS; do
    IFS='|' read -r run args acct part <<< "$spec"
    # one watcher per run: two would evaluate the same checkpoints twice and race on the
    # shared jsonl (the flock makes that safe, not useful)
    if squeue -u "$USER" -h -o "%j" | grep -qx "ev_${run}"; then
        echo "  already watching: $run"; continue
    fi
    cmd=(sbatch --parsable --account="$acct" --partition="$part"
         --job-name="ev_${run}" "$HERE/eval_watch_pusht_search.sbatch" "$ROOT/$run")
    IFS='@' read -ra extra <<< "$args"
    cmd+=("${extra[@]}")
    if [ "$DRY" = --dry ]; then echo "  DRY ${cmd[*]}"; continue; fi
    jid=$("${cmd[@]}")
    echo "  submitted $jid  $part  watch $run  (${extra[*]})"
done
