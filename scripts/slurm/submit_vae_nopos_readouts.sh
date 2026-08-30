#!/usr/bin/env bash
# THE VAE_no_pos EVAL MATRIX. One checkpoint, several rows -- selection and obs-corruption
# are both READOUT-TIME knobs, applied to already-trained weights.
#
# THE QUESTION. Does grading the observation by candidate slot buy anything as the search
# widens? That is read off success-rate-vs-n, and n and slot are the same index: at n = 8
# with K = 16 only slots 0..7 are ever generated -- the NOISY HALF of the ladder -- and
# n = 16 is the first level that reaches the clean end. Under `final_pass` the executed
# action is the ladder's cleanest reachable slot. So n=1, n=8 and n=16 are three different
# conditionals under the ladder, where under the uniform baseline they are the same one
# sampled more times.
#
# THE MATRIX, per checkpoint:
#
#   fixed ladders (linear_t / geometric / linear_signal), and the uniform baselines:
#       {argmax, final_pass} x {--corrupt-obs-eval, --no-corrupt-obs-eval} x n in {1, 8, 16}
#   random_base:
#       {argmax, final_pass} x {--no-corrupt-obs-eval} x n in {1, 8, 16}
#
#   --corrupt-obs-eval  reproduces the slot->level mapping the loss trained under, so the
#                       rollout conditional matches the training one.
#   --no-corrupt-obs-eval  evaluates every slot CLEAN. A legitimate arm -- it measures how
#                       much of the benefit survives without deployment-time corruption --
#                       but it is not the conditional that was trained, which is why
#                       corrupt_obs_eval is in _bon_subdir and in _IDENTITY and the two
#                       never share a curve.
#
# --n-list 1,8,16 buys exactly those three levels. The default powers-of-two rule would also
# pay for n=2 and n=4, and the cost of a level is linear in n.
#
#   bash scripts/slurm/submit_vae_nopos_readouts.sh            # dry run
#   SUBMIT=1 bash scripts/slurm/submit_vae_nopos_readouts.sh   # ...and sbatch
set -uo pipefail
cd "$(dirname "$0")/../.."

SUBMIT="${SUBMIT:-}"
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search
TASK="${TASK_DIR:-pusht_image_search_imgonly}"
N_LIST="${N_LIST:-1,8,16}"
N_ENVS="${N_ENVS:-50}"

# Run dirs, relative to $ROOT/$TASK. The three clean baselines first: a ladder number with
# nothing to compare it against says nothing.
RUNS="
unet_bc/unetbc_ver-t_goal_enc-vae_demos-30_seed-42
offline/value_k1_ver-t_goal_enc-vae_demos-30_seed-42
outer_inner/value_k16_ver-t_goal_enc-vae_demos-30_seed-42
"

# label:flags. The label goes in the job name only; the OUTPUT directory comes from
# eval_search_pusht._bon_subdir, which keys on selection and corrupt_obs_eval both --
# bon_search_sel-{argmax,final_pass}_obs-{corrupt,clean}.
RULES=(
  "argmax-corrupt:--selection argmax --corrupt-obs-eval"
  "argmax-clean:--selection argmax --no-corrupt-obs-eval"
  "final_pass-corrupt:--selection final_pass --corrupt-obs-eval"
  "final_pass-clean:--selection final_pass --no-corrupt-obs-eval"
)
# A clean-trained arm has no ladder, so --corrupt-obs-eval is a no-op on it and the two
# rows would be the same experiment recorded twice. Baselines get the clean rules only.
CLEAN_RULES=(
  "argmax-clean:--selection argmax --no-corrupt-obs-eval"
  "final_pass-clean:--selection final_pass --no-corrupt-obs-eval"
)

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)
n=0

for rel in $RUNS; do
    dir="$ROOT/$TASK/$rel"
    run=$(basename "$rel")
    if [ ! -d "$dir/checkpoints" ]; then
        printf '%-52s %s\n' "$run" "no checkpoints yet, skipping"; continue
    fi
    # Ladder arms carry _son- in the name; only those get the corrupted rows.
    case "$run" in
        *_son-*) rules=("${RULES[@]}") ;;
        *)       rules=("${CLEAN_RULES[@]}") ;;
    esac
    for rule in "${rules[@]}"; do
        label="${rule%%:*}"; flags="${rule#*:}"
        # MUST match eval_search_pusht._bon_subdir exactly. It appends _sel-<mode> and
        # _obs-{corrupt,clean}; a probe that spells either differently stats a path that
        # never exists, so every finished sweep looks un-done and is resubmitted forever.
        # (submit_slot_norm_readouts.sh did exactly that with _corrupt-on/_corrupt-off.)
        sel="${label%%-*}"; obs="${label##*-}"
        sub="bon_search_sel-${sel}_obs-${obs}"
        name="ev_${label}_${run}"
        if grep -qxF "$name" <<<"$LIVE"; then
            printf '%-52s %-20s %s\n' "$run" "$label" "already evaluating, skipping"; continue
        fi
        if [ -z "$SUBMIT" ]; then
            printf '%-52s %-20s WOULD SUBMIT -> %s\n' "$run" "$label" "$sub"; n=$((n+1)); continue
        fi
        read -r A P < <(bash scripts/slurm/pick_gpu.sh) || { echo "no free GPU for $name" >&2; continue; }
        jid=$(sbatch --parsable --account="$A" --partition="$P" --job-name="$name" \
              scripts/slurm/eval_watch_pusht_search.sbatch "$dir" \
              --n-list "$N_LIST" --n-envs "$N_ENVS" --skip-val --wandb $flags)
        printf '%-52s %-20s submitted %s\n' "$run" "$label" "$jid"
        n=$((n+1))
    done
done
echo
echo "readouts handled: $n"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
