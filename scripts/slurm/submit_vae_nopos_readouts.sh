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
# n differs BY RULE. argmax is the success-vs-width curve and wants the whole sweep;
# final_pass costs the same per level and only three points are asked for. Cost is linear
# in n per level, so a full argmax sweep is ~2x its own largest level.
N_LIST_ARGMAX="${N_LIST_ARGMAX:-1,2,4,8,16,32,64}"
N_LIST_FINAL="${N_LIST_FINAL:-1,8,16}"
N_ENVS="${N_ENVS:-50}"

# Run dirs, relative to $ROOT/$TASK. The three clean baselines first: a ladder number with
# nothing to compare it against says nothing. Then the five ladders, matching the eight arms
# scripts/run_vae_nopos_30demo.sh trains.
LADDER_RUNS="
unet_bc/unetbc_ver-t_goal_enc-vae_demos-30_seed-42
offline/value_k1_ver-t_goal_enc-vae_demos-30_seed-42
outer_inner/value_k16_ver-t_goal_enc-vae_demos-30_seed-42
outer_inner/value_k16_ver-t_goal_son-lint999_enc-vae_demos-30_seed-42
outer_inner/value_k16_ver-t_goal_son-lint400_enc-vae_demos-30_seed-42
outer_inner/value_k16_ver-t_goal_son-linsig_enc-vae_demos-30_seed-42
outer_inner/value_k16_ver-t_goal_son-geo7_enc-vae_demos-30_seed-42
outer_inner/value_k16_ver-t_goal_son-rndlinsig_enc-vae_demos-30_seed-42
"

# The ENCODER DEBUG set: 3 encoders x {BC UNet, ST k=1}, no ladder on any of them, so the
# case below gives them the clean rules only -- argmax n=1..64 and final_pass n=1,8,16.
DEBUG_RUNS="
unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-30_seed-42
offline/value_k1_ver-t_goal_enc-resnet18_demos-30_seed-42
unet_bc/unetbc_ver-t_goal_enc-vae_demos-30_seed-42
offline/value_k1_ver-t_goal_enc-vae_demos-30_seed-42
unet_bc/unetbc_ver-t_goal_enc-vae-ft_demos-30_seed-42
offline/value_k1_ver-t_goal_enc-vae-ft_demos-30_seed-42
unet_bc/unetbc_ver-t_goal_enc-resnet18-frozen_demos-30_seed-42
offline/value_k1_ver-t_goal_enc-resnet18-frozen_demos-30_seed-42
unet_bc/unetbc_ver-t_goal_enc-resnet18-frozen_demos-126_seed-42
offline/value_k1_ver-t_goal_enc-resnet18-frozen_demos-126_seed-42
unet_bc/unetbc_ver-t_goal_enc-vae_demos-126_seed-42
offline/value_k1_ver-t_goal_enc-vae_demos-126_seed-42
unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-126_seed-42
offline/value_k1_ver-t_goal_enc-resnet18_demos-126_seed-42
unet_bc/unetbc_ver-t_goal_enc-vae-ft_demos-126_seed-42
offline/value_k1_ver-t_goal_enc-vae-ft_demos-126_seed-42
unet_bc/unetbc_ver-t_goal_enc-resnet18-frozen-bn_demos-30_seed-42
offline/value_k1_ver-t_goal_enc-resnet18-frozen-bn_demos-30_seed-42
unet_bc/unetbc_ver-t_goal_enc-resnet18-frozen-bn_demos-126_seed-42
offline/value_k1_ver-t_goal_enc-resnet18-frozen-bn_demos-126_seed-42
"

# The PAPER-90 set: the original Diffusion Policy protocol (90 train / 4 val / 112
# discarded). These MUST be evaluated on fresh env seeds, not on our held-out episodes --
# the paper's 90 training episodes overlap our 50 test episodes by 23, so a held-out-episode
# readout on these checkpoints would be scoring 23 episodes the model trained on. The
# manifest carries no test split at all for that reason.
PAPER90_RUNS="
unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-90_seed-42
unet_bc/unetbc_ver-t_goal_enc-resnet18-frozen_demos-90_seed-42
unet_bc/unetbc_ver-t_goal_enc-vae_demos-90_seed-42
unet_bc/unetbc_ver-t_goal_enc-vae-ft_demos-90_seed-42
offline/value_k1_ver-t_goal_enc-resnet18_demos-90_seed-42
offline/value_k1_ver-t_goal_enc-resnet18-frozen_demos-90_seed-42
offline/value_k1_ver-t_goal_enc-vae_demos-90_seed-42
offline/value_k1_ver-t_goal_enc-vae-ft_demos-90_seed-42
"

case "${RUNS_SET:-ladder}" in
  debug)   RUNS="$DEBUG_RUNS" ;;
  paper90) RUNS="$PAPER90_RUNS" ;;
  *)       RUNS="$LADDER_RUNS" ;;
esac

# TEST_START_SEED switches the episode source from held-out dataset episodes to fresh
# environment seeds (the paper's protocol). REQUIRED for paper90; optional elsewhere, where
# it gives every budget a common, contamination-free readout -- fresh seeds come from the
# env, never from the dataset, so no split can leak into them.
SEED_FLAGS=""; SEED_SUB=""
if [ "${RUNS_SET:-ladder}" = "paper90" ] || [ -n "${TEST_START_SEED:-}" ]; then
  _tss="${TEST_START_SEED:-100000}"
  SEED_FLAGS="--test-start-seed ${_tss} --n-test-seeds ${N_TEST_SEEDS:-50}"
  SEED_SUB="_seeds${_tss}"
fi

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
#
# `random_base` also takes the clean rules, for a different reason: its level is drawn per
# sample, so "eval with the same noise as training" is not a defined condition -- there is no
# single ladder to reproduce.
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
    # FIXED ladder arms carry _son- and get the corrupted rows too. Baselines have no
    # ladder; random_base has no reproducible one. Both take the clean rules only.
    case "$run" in
        *_son-rndlinsig*) rules=("${CLEAN_RULES[@]}") ;;
        *_son-*)          rules=("${RULES[@]}") ;;
        *)                rules=("${CLEAN_RULES[@]}") ;;
    esac
    for rule in "${rules[@]}"; do
        label="${rule%%:*}"; flags="${rule#*:}"
        # MUST match eval_search_pusht._bon_subdir exactly. It appends _sel-<mode> and
        # _obs-{corrupt,clean}; a probe that spells either differently stats a path that
        # never exists, so every finished sweep looks un-done and is resubmitted forever.
        # (submit_slot_norm_readouts.sh did exactly that with _corrupt-on/_corrupt-off.)
        sel="${label%%-*}"; obs="${label##*-}"
        # _bon_subdir puts the seed key FIRST, before _sel-. Mirror that exactly, or the
        # fresh-seed sweeps stat a path that never exists and resubmit forever.
        sub="bon_search${SEED_SUB}_sel-${sel}_obs-${obs}"
        # The job name must differ too, otherwise the LIVE check below treats a running
        # held-out-episode sweep as though it were this fresh-seed one and skips it.
        name="ev_${label}${SEED_SUB}_${run}"
        if grep -qxF "$name" <<<"$LIVE"; then
            printf '%-52s %-20s %s\n' "$run" "$label" "already evaluating, skipping"; continue
        fi
        if [ -z "$SUBMIT" ]; then
            case "$sel" in
                argmax) nl="$N_LIST_ARGMAX" ;;
                *)      nl="$N_LIST_FINAL"  ;;
            esac
            printf '%-52s %-18s n=%-16s -> %s\n' "$run" "$label" "$nl" "$sub"
            n=$((n+1)); continue
        fi
        # NO pick_gpu.sh here. eval_watch_pusht_search.sbatch declares
        # `--partition=ckpt --account=robotics` itself, and eval belongs on the preemptible
        # checkpoint partition so the guaranteed robotics/weirdlab GPUs stay free for
        # training. Overriding it with pick_gpu (which is the TRAINING allocator) put eval
        # on those very GPUs.
        case "$sel" in
            argmax) nl="$N_LIST_ARGMAX" ;;
            *)      nl="$N_LIST_FINAL"  ;;
        esac
        jid=$(sbatch --parsable --job-name="$name" \
              scripts/slurm/eval_watch_pusht_search.sbatch "$dir" \
              --n-list "$nl" --n-envs "$N_ENVS" --skip-val --wandb $flags $SEED_FLAGS)
        printf '%-52s %-20s submitted %s\n' "$run" "$label" "$jid"
        n=$((n+1))
    done
done
echo
echo "readouts handled: $n"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
