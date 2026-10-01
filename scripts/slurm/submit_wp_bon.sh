#!/usr/bin/env bash
# Best-of-n on ONE BC UNet checkpoint under {t_goal, wp_v3, wp_v5}, n = 1,2,4,...,128.
#
# ONE JOB PER (value, n). eval_search_pusht.py has no resume outside --watch, and these run
# on the preemptible ckpt partition with --requeue: a single 8h job holding the whole n
# list restarts from n=1 after every preemption, while a per-n job re-does at most itself.
# Every job merges its row into the same <run>/bon_search_ver-<value>/step_XXXXXXX/
# success_curve.json under a lock, so the curve assembles regardless of finish order.
#
# Time limits scale with n (a chunk at n=8 took ~200s on 50 envs in the Aug-20 t_goal
# evals, ~25s per unit n; the wp values add a tracker copy + update per candidate per
# step, budgeted at 2x, plus an hour for a slow gscratch: the 2026-09-30 smoke read it
# at 8 MB/s and the ~440MB import alone took 15 min). All end inside the ckpt ~9h cap.
#
#   unset WANDB_API_KEY
#   CKPT=/path/to/step_0060000.ckpt bash scripts/slurm/submit_wp_bon.sh            # dry run
#   CKPT=... SUBMIT=1 bash scripts/slurm/submit_wp_bon.sh                          # sbatch
#   CKPT=... VALUES="wp_v5" NS="64 128" SUBMIT=1 bash scripts/slurm/submit_wp_bon.sh
#   CKPT=... VALUES="wp_v3 wp_v5" SELECTION=softmax TEMP=1.0 SUBMIT=1 bash ...   # softmax
#     (rows land in bon_search_sel-softmax_ver-<value>/, never merging with argmax ones)
set -uo pipefail
cd "$(dirname "$0")/../.."

CKPT="${CKPT:?set CKPT=/path/to/step_XXXXXXX.ckpt}"
test -f "$CKPT" || { echo "MISSING CHECKPOINT $CKPT"; exit 1; }
SUBMIT="${SUBMIT:-}"
VALUES="${VALUES:-t_goal wp_v3 wp_v5}"
NS="${NS:-1 2 4 8 16 32 64 128}"
N_ENVS="${N_ENVS:-30}"
SELECTION="${SELECTION:-}"          # empty = the checkpoint's own rule (argmax for BC)
TEMP="${TEMP:-1.0}"
SEL_ARGS=(); SEL_TAG=""
if [[ -n "$SELECTION" ]]; then
  SEL_ARGS=(--selection "$SELECTION")
  [[ "$SELECTION" == softmax ]] && SEL_ARGS+=(--selection-temperature "$TEMP")
  SEL_TAG="_${SELECTION}"
fi

for d in media/veritas_pusht/v5_pusher_and_t media/veritas_pusht/v3_end_outside; do
  echo "plans in $d: $(ls "$d"/ep*.json 2>/dev/null | wc -l)"
done

time_for_n() {
  case "$1" in
    128) echo 7:00:00 ;;
    64)  echo 4:00:00 ;;
    32)  echo 3:00:00 ;;
    *)   echo 2:00:00 ;;
  esac
}

STEP="$(basename "$CKPT" .ckpt)"
LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)
count=0
for v in $VALUES; do
  for n in $NS; do
    name="wpbon_${v}${SEL_TAG}_${STEP}_n${n}"
    if grep -qx "$name" <<< "$LIVE"; then
      echo "skip $name (already queued/running)"; continue
    fi
    cmd=(sbatch --job-name="$name" --time="$(time_for_n "$n")"
         scripts/slurm/eval_ckpt_pusht_search.sbatch "$CKPT"
         --skip-val --n-envs "$N_ENVS" --n-list "$n" --verifier-value "$v" --store-scores "${SEL_ARGS[@]}")
    echo "${cmd[*]}"
    if [[ -n "$SUBMIT" ]]; then "${cmd[@]}"; fi
    count=$((count + 1))
  done
done
echo "$count job(s) $([[ -n "$SUBMIT" ]] && echo submitted || echo 'planned (SUBMIT=1 to sbatch)')"
