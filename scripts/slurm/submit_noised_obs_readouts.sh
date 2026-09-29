#!/usr/bin/env bash
# Eval watchers for the noised-obs ladder grid (scripts/slurm/submit_noised_obs_rnE2E.sh).
#
# Idempotent top-up, exactly like submit_30_100_watchers.sh: each watcher polls its run's
# checkpoints/ and scores every new step_*.ckpt. A watcher's wall clock is 12h and the search
# arms train for days, so this is meant to be re-run (by hand or from /loop); it skips any
# (run, readout) that already has a live watcher or is already fully scored.
#
# FOUR READOUTS, all on the SAME weights -- selection and corrupt_obs_eval are readout rules,
# not trained state, so each checkpoint is read four ways rather than trained four times:
#
#   argmax   n = 1 2 4 8 16 32 64   the test-time-compute sweep
#   fpass    n = 1 8 16             selection: final_pass -- execute the LAST generation,
#                                   i.e. the ladder's cleanest reachable slot
#   x obs-corrupt   rollouts see the SAME ladder the run trained under
#   x obs-clean     rollouts see clean observations
#
# WHICH ARMS GET THE CORRUPT ROLLOUT. The three fixed ladders (linear_t, linear_signal,
# geometric) get both; `random_base` gets obs-clean ONLY, as asked. The three baselines have
# no ladder at all, so corrupt_obs_eval is the identity for them and the flag is omitted --
# passing it would only fork their curves into directories holding identical numbers.
#
# Reading obs-clean: the slot -> corruption-level mapping the model trained under does not
# hold at rollout, so it is a legitimate arm but not the conditional the loss trained.
#
# Each readout lands in its OWN bon_search subdirectory (eval_search_pusht._bon_subdir), so
# no two ever merge into one curve.
#
#   bash scripts/slurm/submit_noised_obs_readouts.sh          # dry run
#   SUBMIT=1 bash scripts/slurm/submit_noised_obs_readouts.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search/pusht_image_search_imgonly
SUBMIT="${SUBMIT:-}"
PY_BIN="${PY_BIN:-/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python}"
# --skip-val: the ask is the 50 TEST episodes; the 30 val ones would roughly double the
# per-checkpoint cost for a number nothing here reports.
BASE_ARGS="--skip-val"

# run subdir/run_name | ladder class
#   none  = no ladder; corrupt_obs_eval is the identity, so no obs-* fork
#   both  = fixed ladder; score under corrupt AND clean rollouts
#   clean = random_base; clean rollouts only
ARMS=(
  "unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-30_seed-42|none"
  "offline/value_k1_ver-t_goal_enc-resnet18_demos-30_seed-42|none"
  "outer_inner/value_k16_ver-t_goal_enc-resnet18_demos-30_seed-42|none"
  "outer_inner/value_k16_ver-t_goal_son-lint-cap999_enc-resnet18_demos-30_seed-42|both"
  "outer_inner/value_k16_ver-t_goal_son-lint-cap400_enc-resnet18_demos-30_seed-42|both"
  "outer_inner/value_k16_ver-t_goal_son-linsig-cap999_enc-resnet18_demos-30_seed-42|both"
  "outer_inner/value_k16_ver-t_goal_son-linsig-cap400_enc-resnet18_demos-30_seed-42|both"
  "outer_inner/value_k16_ver-t_goal_son-geo85-cap999_enc-resnet18_demos-30_seed-42|both"
  "outer_inner/value_k16_ver-t_goal_son-geo85-cap400_enc-resnet18_demos-30_seed-42|both"
  "outer_inner/value_k16_ver-t_goal_son-rndlinsig-cap999_enc-resnet18_demos-30_seed-42|clean"
)

# key | eval args | bon_search subdir | expected n grid
# The subdir MUST match what _bon_subdir builds from those same args, or the completeness
# check reads a directory the watcher never writes and resubmits forever.
sel_readouts() {   # $1 = ladder class -> one "key|args|subdir|grid" line per readout
    local cls="$1"
    local obs=()
    case "$cls" in
        none)  obs=("|") ;;
        both)  obs=("--corrupt-obs-eval|_obs-corrupt" "--no-corrupt-obs-eval|_obs-clean") ;;
        clean) obs=("--no-corrupt-obs-eval|_obs-clean") ;;
    esac
    local o oarg osuf
    for o in "${obs[@]}"; do
        oarg="${o%%|*}"; osuf="${o#*|}"
        # --selection argmax is passed EXPLICITLY even though it is the native rule, so the
        # directory names its readout instead of being the unlabelled `bon_search/` default.
        # Same convention as the vae-debug grid; it also keeps these curves from merging with
        # any native-mode sweep an older watcher may have written into the same run.
        echo "argmax${osuf}|--selection argmax ${oarg}|bon_search_sel-argmax${osuf}|1,2,4,8,16,32,64"
        echo "fpass${osuf}|--selection final_pass --n-list 1,8,16 ${oarg}|bon_search_sel-final_pass${osuf}|1,8,16"
    done
}

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)
n=0
for arm in "${ARMS[@]}"; do
    rel="${arm%%|*}"; cls="${arm#*|}"
    run="${rel##*/}"; dir="$ROOT/$rel"
    if ! [ -d "$dir" ]; then
        printf '%-70s %s\n' "$run" "no run dir yet -- training has not started it"; continue
    fi
    n_ckpt=$(ls "$dir"/checkpoints/step_*.ckpt 2>/dev/null | wc -l)
    # A watcher started against an empty checkpoints/ just idle-polls until its 12h wall
    # expires, holding a ckpt GPU for no work. The next top-up picks the run up instead.
    if [ "$n_ckpt" -eq 0 ]; then
        printf '%-70s %s\n' "$run" "no checkpoints yet -- nothing to evaluate"; continue
    fi
    while IFS='|' read -r key args sub grid; do
        [ -z "$key" ] && continue
        job="ev_${key}_${run}"
        if grep -qxF "$job" <<<"$LIVE"; then
            printf '  %-14s %-58s %s\n' "$key" "$run" "watcher live, skipping"; continue
        fi
        # Already fully scored for THIS readout? Without this the script resurrects a
        # watcher for a finished arm on every invocation -- "has checkpoints, has no live
        # watcher" stays true forever once training ends.
        if "$PY_BIN" scripts/bon_curve_complete.py "$dir/$sub" "$n_ckpt" "$grid"; then
            printf '  %-14s %-58s %s\n' "$key" "$run" "fully scored -- nothing to do"; continue
        fi
        if [ -z "$SUBMIT" ]; then
            printf '  %-14s %-58s WOULD SUBMIT (%s ckpt) [%s]\n' "$key" "$run" "$n_ckpt" "$args"
            n=$((n+1)); continue
        fi
        jid=$(sbatch --parsable --job-name="$job" \
              scripts/slurm/eval_watch_pusht_search.sbatch "$dir" $BASE_ARGS $args)
        printf '  %-14s %-58s submitted %s (%s ckpt)\n' "$key" "$run" "$jid" "$n_ckpt"
        n=$((n+1))
    done < <(sel_readouts "$cls")
done
echo
echo "readouts handled: $n"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
