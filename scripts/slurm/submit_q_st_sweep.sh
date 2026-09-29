#!/usr/bin/env bash
# THE READOUT FOR THE 12 Q-TRAINED ST ARMS (tag `q_fn_sac_all_206_demos`).
# Every 10k checkpoint x {argmax, final_pass}, each arm on its OWN test split.
#
#   bash scripts/slurm/submit_q_st_sweep.sh            # dry run
#   SUBMIT=1 bash scripts/slurm/submit_q_st_sweep.sh   # ...and sbatch
#   ARMS_FILTER=split-blq SELECTIONS=argmax SUBMIT=1 bash scripts/slurm/submit_q_st_sweep.sh
#
# NO --q FLAG, AND NONE IS POSSIBLE. These arms carry `verifier_tag=q_sac_all` in their own
# config, so `_build_verifier` rebuilds the same PushTQVerifier they trained against and
# `_normalize_q` reproduces the same running z-score from the checkpointed buffers. That is the
# whole point of training them: unlike `install_q_ranker` on a t_goal arm, there is no
# train/eval mismatch to confound the reading. `eval_search_pusht.py` needs no Q-specific flag.
#
# TWO SELECTIONS, AND THE SECOND IS THE CONTROL.
#   argmax     -- the deployed rule: the verifier scalar ranks n candidates, best one executed.
#   final_pass -- the verifier is INERT: candidate n-1 is executed as-is, having been generated
#                 conditioned on the other n-1. It separates "search changed the SAMPLER" from
#                 "search changed the SELECTOR", which a success rate under argmax alone cannot.
#                 `_bon_subdir` gives each its own directory, so the two never merge.
#
# ONE WATCHER PER (arm, selection), not one job per checkpoint: eval_search_pusht.py --watch
# walks every step_*.ckpt in the run dir and appends to one success_curves.jsonl. 12 x 2 = 24
# jobs instead of 240, and the flock-guarded append makes a requeue safe.
#
# ⚠️ NOTHING IS HELD OUT FROM q_sac_all (demo_episodes: all). See
# docs/reports/q_fn_sac_all_206_demos_2026-09-20.md.
set -uo pipefail
cd "$(dirname "$0")/../.."

ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"
B="$ROOT/pusht_search/pusht_image_search_imgonly/outer_inner"
N_LIST="${N_LIST:-1,2,4,8,16}"
SELECTIONS="${SELECTIONS:-argmax final_pass}"
ARMS_FILTER="${ARMS_FILTER:-}"
IDLE="${IDLE:-900}"          # training is done; exit once the grid is covered
TIME_LIMIT="${TIME_LIMIT:-$(bash scripts/slurm/safe_time_limit.sh 12:00:00)}"

# A STALE WANDB_API_KEY shadows ~/.netrc inside the job and kills it at wandb.init.
if [ -n "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "WARNING: WANDB_API_KEY is set and ~/.netrc has no wandb entry; keeping it" >&2
else
    unset WANDB_API_KEY
fi

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)
n=0; skipped=0
for dir in "$B"/*ver-q_sac_all*/; do
    [ -d "$dir" ] || continue
    run=$(basename "$dir")
    [ -n "$ARMS_FILTER" ] && ! grep -q -- "$ARMS_FILTER" <<<"$run" && continue
    # The full 10k grid must exist before a watcher is worth starting; --watch would sit and
    # poll for the rest, burning a GPU on an arm that is still training.
    have=$(ls "$dir"/checkpoints/step_*.ckpt 2>/dev/null | wc -l)
    if [ "$have" -lt 10 ]; then
        printf '%-62s %s\n' "${run:0:62}" "only $have/10 checkpoints, skipping"
        skipped=$((skipped+1)); continue
    fi
    # label: drop the fixed components, keep what varies between arms
    tag=$(sed -e 's/^value_//' -e 's/_ver-q_sac_all//' -e 's/_enc-resnet18//' \
              -e 's/_seed-42$//' -e 's/_son-/_/' -e 's/_demos-[0-9]*//' <<<"$run")
    for sel in $SELECTIONS; do
        name="qst_${tag}_${sel}"
        if grep -qxF "$name" <<<"$LIVE"; then
            printf '%-62s %s\n' "$name" "already running, skipping"; skipped=$((skipped+1)); continue
        fi
        cmd=(sbatch --parsable --job-name="$name" --time="$TIME_LIMIT"
             scripts/slurm/eval_watch_pusht_search.sbatch "$dir"
             --n-list "$N_LIST" --selection "$sel" --skip-val --idle-exit-sec "$IDLE")
        if [ -z "${SUBMIT:-}" ]; then
            printf '%-62s WOULD SUBMIT\n' "$name"
        else
            jid=$("${cmd[@]}")
            printf '%-62s submitted %s\n' "$name" "$jid"
        fi
        n=$((n+1))
    done
done
echo
echo "sweeps handled: $n   skipped: $skipped   n-list: $N_LIST   selections: $SELECTIONS"
[ -z "${SUBMIT:-}" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
