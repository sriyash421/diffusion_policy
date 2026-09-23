#!/usr/bin/env bash
# EXTEND THE `q_fn_sac_all_206_demos` SWEEP TO n = 32 AND 64, at step 100k only.
#
#   bash scripts/slurm/submit_q_n64_extend.sh            # dry run
#   SUBMIT=1 bash scripts/slurm/submit_q_n64_extend.sh   # ...and sbatch
#
# ONLY THE MISSING n ARE COMPUTED. `eval_search_pusht.merge_curves` unions two curves over n
# and keeps the row when seed and episode set agree, so `--n-list 32,64` splices into the
# 1,2,4,8,16 already on disk instead of recomputing them. Re-running the whole grid would cost
# ~4x this and produce identical numbers for the n already measured.
#
# STEP 100k ONLY, deliberately. The 10k grid at n<=16 is already complete (submit_q_st_sweep.sh);
# what is wanted here is one readable table, and 12 arms x 2 selections x 7 n-values is already
# 168 cells. Pass STEPS to widen it.
#
# BC GOES THROUGH sac/eval.py, NOT eval_search_pusht.py. The BC checkpoints carry
# `verifier_tag=t_goal`, so eval_search_pusht would build the SIM verifier from their own config
# and never touch the Q; `bon-sweep --rankers t_goal,q_sac_all` is what puts both rankers on the
# same episodes in one job. It writes its own bon_curves.json rather than merging, so the BC
# rows are recomputed from n=1 -- there is nothing to splice into.
set -uo pipefail
cd "$(dirname "$0")/../.."

ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"
B="$ROOT/pusht_search/pusht_image_search_imgonly"
N_LIST="${N_LIST:-32,64}"          # ST: only what is missing
MAX_N="${MAX_N:-64}"               # BC: full grid, nothing to merge into
SELECTIONS="${SELECTIONS:-argmax final_pass}"
STEPS="${STEPS:-100000}"
OUT="${OUT:-/gscratch/robotics/harine/value_arms/bon_q_geom_n64}"
TIME_LIMIT="${TIME_LIMIT:-$(bash scripts/slurm/safe_time_limit.sh 12:00:00)}"

if [ -n "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "WARNING: WANDB_API_KEY is set and ~/.netrc has no wandb entry; keeping it" >&2
else
    unset WANDB_API_KEY
fi

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)
n=0; skipped=0

submit() {  # $1 name, rest = sbatch args
    local name="$1"; shift
    if grep -qxF "$name" <<<"$LIVE"; then
        printf '%-58s %s\n' "$name" "already running, skipping"; skipped=$((skipped+1)); return
    fi
    if [ -z "${SUBMIT:-}" ]; then
        printf '%-58s WOULD SUBMIT\n' "$name"; n=$((n+1)); return
    fi
    printf '%-58s submitted %s\n' "$name" "$("$@")"
    n=$((n+1))
}

# ---------------------------------------------------------------- ST: 12 arms x 2 selections
for dir in "$B"/outer_inner/*ver-q_sac_all*/; do
    [ -d "$dir" ] || continue
    run=$(basename "$dir")
    tag=$(sed -e 's/^value_//' -e 's/_ver-q_sac_all//' -e 's/_enc-resnet18//' \
              -e 's/_seed-42$//' -e 's/_son-/_/' -e 's/_demos-[0-9]*//' <<<"$run")
    for step in $STEPS; do
        ck=$(printf '%s/checkpoints/step_%07d.ckpt' "$dir" "$step")
        [ -f "$ck" ] || { echo "[SKIP $tag @$step] no checkpoint"; skipped=$((skipped+1)); continue; }
        for sel in $SELECTIONS; do
            submit "qn64_${tag}_${sel}" \
                sbatch --parsable --job-name="qn64_${tag}_${sel}" --time="$TIME_LIMIT" \
                scripts/slurm/eval_ckpt_pusht_search.sbatch "$ck" \
                --n-list "$N_LIST" --selection "$sel" --skip-val
        done
    done
done

# ---------------------------------------------------------------- BC: 2 arms x 2 selections
CONDA=/gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
REPO=/mmfs1/home/harine/diffusion_policy_standalone
for arm in "blq137|unetbc_ver-t_goal_enc-resnet18_demos-137_split-blq_seed-42" \
           "brd100|unetbc_ver-t_goal_enc-resnet18_demos-100_split-brd100_seed-42"; do
    IFS='|' read -r label run <<<"$arm"
    for step in $STEPS; do
        ck=$(printf '%s/unet_bc/%s/checkpoints/step_%07d.ckpt' "$B" "$run" "$step")
        [ -f "$ck" ] || { echo "[SKIP unetbc $label @$step] no checkpoint"; skipped=$((skipped+1)); continue; }
        for sel in $SELECTIONS; do
            # --skip-context-sim: BC ignores the search context, so the sim call is pure
            # overhead for the Q ranker. --allow-contaminated: q_sac_all's buffer holds every
            # episode being scored, and the flag records that in bon_curves.json.
            cmd="python -u sac/eval.py bon-sweep -c $ck --rankers t_goal,q_sac_all \
--max-n $MAX_N --split test --n-envs 8 --selection $sel --skip-context-sim \
--allow-contaminated -o $OUT/unetbc_${label}_step$((step/1000))k_${sel}"
            submit "qn64_unetbc_${label}_${sel}" \
                sbatch --parsable --partition=ckpt --account=robotics --requeue \
                --job-name="qn64_unetbc_${label}_${sel}" --gpus=1 --cpus-per-task=8 \
                --mem=32G --time="$TIME_LIMIT" \
                --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out \
                --wrap "set -u; source $CONDA; conda activate robodiff; cd $REPO; \
                        unset WANDB_API_KEY; PYTHONPATH=\$PWD $cmd"
        done
    done
done

echo
echo "jobs handled: $n   skipped: $skipped   ST n-list: $N_LIST   BC max-n: $MAX_N   steps: $STEPS"
[ -z "${SUBMIT:-}" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
