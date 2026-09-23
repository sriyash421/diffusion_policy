#!/usr/bin/env bash
# THE t_goal-MOVING GENERATION. 6 ST arms (2 widths x 3 obs-noise ladders) + 1 UNet BC,
# all trained only on the transitions where the demonstrator's T actually moves.
#
# WHY. `t_goal` scores a candidate by where the T ends up and nothing else, so on a decision
# that moves the block nowhere every candidate ties and the verifier expresses no preference
# at all -- 15-25% of decisions, measured in docs/reports/ppo_sac_lstm-bc_eval_2026-09-17.md.
# Every arm so far trained on those windows anyway. These do not.
#
#   DATA. All 206 episodes: 176 train / 0 val / 30 test, the 30 chosen so the MOVING-
#   transition ratio is 100:20 (scripts/make_moving_ratio_split.py). Filtered, that is
#   12,972 train and 2,595 test windows.
#
#   NO VAL, BY DESIGN. Nothing selects a checkpoint here; the analysis steps are the whole
#   10k checkpoint grid, named in advance. Both workspaces skip validation when the val split
#   is empty, so this needs no flag.
#
#   THE WIDTHS. k=16 and k=4. The ladders span the same 400->200 at both, but at 4 points
#   versus 16 -- endpoints matched, granularity not. A K-length profile is REQUIRED
#   (_slot_obs_profile raises if len(timesteps) != max_actions), which is why mk_flat/mk_ramp
#   take K here rather than hardcoding 16 as run_obsnoise_geometric.sh does.
#
#   split_suffix IS LOAD-BEARING. run_name has no transition component, so a filtered arm
#   with an empty suffix resolves to the SAME hydra.run.dir as an unfiltered one at the same
#   n_demos and `training.resume: True` continues it. check_transition_filter_labels refuses
#   to start if the two disagree, but the suffix is what makes them agree.
#
#   RE-RUNNING IS THE REPAIR. The launcher skips any arm already training or already at
#   step 100k, so `SUBMIT=1` a second time resubmits exactly the ones that died and touches
#   nothing else. Worth knowing because of a wandb flake that has now killed four runs in
#   this project's history: two jobs starting on the SAME node at the same instant race in
#   wandb's service startup and one dies with `assert ports_found` in
#   wandb/sdk/service/service.py, minutes in, AFTER the dataset has loaded. It is not a
#   config error and nothing about the arm is wrong -- just resubmit it.
#
#   # the blq137 generation: same four ladders, a different parent split
#   SPLIT_FILE=diffusion_policy/config/splits/pusht_blockquad_bottomleft_train137.json \
#   TRANS_FILE=diffusion_policy/config/splits/transitions/pusht_blockquad_bottomleft_train137_moving_ta8.json \
#   N_DEMOS=137 N_TEST=50 N_VAL=19 SPLIT_SUFFIX=_split-blq-mv \
#   INCLUDE_BC=0 WIDTHS="16 4" LADDERS="flat600 ramp600to200" SUBMIT=1 \
#     bash scripts/run_tgoal_moving.sh
#
#   bash scripts/run_tgoal_moving.sh            # dry run
#   SUBMIT=1 bash scripts/run_tgoal_moving.sh   # ...and sbatch (also: resubmit what died)
#   INCLUDE_BC=0 LADDERS="flat600 ramp600to200" SUBMIT=1 bash scripts/run_tgoal_moving.sh
set -uo pipefail
cd "$(dirname "$0")/.."

SUBMIT="${SUBMIT:-}"
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search
PY="${DP_PY:-python}"
SPL=diffusion_policy/config/splits
# THE PARENT SPLIT, overridable as a set. Every field below is a property of one episode
# manifest, so they move together or not at all -- a transition manifest is pinned to its
# parent's checksum and load_transition_manifest raises if the two are mixed. The blq137
# generation (2026-09-21) is this block with five different values, not a second script.
SPLIT_FILE="${SPLIT_FILE:-$SPL/pusht_seed42_train176.json}"
TRANS_FILE="${TRANS_FILE:-$SPL/transitions/pusht_seed42_train176_moving_ta8.json}"
N_DEMOS="${N_DEMOS:-176}"; N_TEST="${N_TEST:-30}"; N_VAL="${N_VAL:-0}"
SPLIT_SUFFIX="${SPLIT_SUFFIX:-_split-mv}"
# Cap the wall clock so the job ends before the next maintenance reservation; SLURM parks a
# job that would run past one on the far side of the window. See safe_time_limit.sh.
TIME_LIMIT="${TIME_LIMIT:-$(bash scripts/slurm/safe_time_limit.sh 3-00:00:00)}"
# A GPU with failing memory takes a job down minutes in with an uncorrectable ECC error and
# SLURM does not drain the node for it. Comma-separated, e.g. EXCLUDE_NODES=g3070,g3081
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
EXCL_FLAG=""; [ -n "$EXCLUDE_NODES" ] && EXCL_FLAG="--exclude=$EXCLUDE_NODES"

for f in "$SPLIT_FILE" "$TRANS_FILE"; do
    [ -f "$f" ] || { echo "MISSING: $f -- run scripts/make_moving_ratio_split.py and " \
                          "scripts/make_moving_transitions.py first" >&2; exit 1; }
done

# Built here rather than pasted, so a ladder cannot drift from the name it is filed under.
mk_flat() { $PY -c "print('['+','.join(['$1']*$2)+']')"; }                     # t, K
mk_ramp() { $PY -c "a,b,k=$1,$2,$3; print('['+','.join(str(int(round(a-(a-b)*i/(k-1)))) for i in range(k))+']')"; }

# WHICH ARMS. Parameterised so a follow-up generation is an env var rather than an edited
# file -- the 2026-09-19 flat600 / ramp600to200 pair was added this way. A ladder token is
# `uniform`, `flat<t>` or `ramp<a>to<b>`; the timesteps are built from the token at the arm's
# own K, so a name can never disagree with the profile it is filed under.
WIDTHS="${WIDTHS:-16 4}"
LADDERS="${LADDERS:-uniform flat400 ramp400to200}"
INCLUDE_BC="${INCLUDE_BC:-1}"   # 0 when extending a generation whose BC already exists

ladder_override() {              # <token> <K> -> hydra overrides, or '' if the token is bad
    local tok="$1" K="$2"
    case "$tok" in
      uniform) echo "slot_obs_noise.mode=uniform" ;;
      flat*)   local t="${tok#flat}"
               echo "slot_obs_noise.mode=list +slot_obs_noise.timesteps=$(mk_flat "$t" "$K")" \
                    "son_suffix=_son-${tok} son_tag=flat-${t} obs_noise_tag=obs_noised" ;;
      ramp*to*) local ab="${tok#ramp}" a b
               a="${ab%%to*}"; b="${ab##*to}"
               echo "slot_obs_noise.mode=list +slot_obs_noise.timesteps=$(mk_ramp "$a" "$b" "$K")" \
                    "son_suffix=_son-${tok} son_tag=ramp-${a}-${b} obs_noise_tag=obs_noised" ;;
      *) return 1 ;;
    esac
}

# config | overrides | label.  BC first: its checkpoint is ~4.4GB and it is the cheapest arm,
# so a config or quota problem surfaces on it rather than 30 hours into a k=16 run.
ARMS=()
[ "$INCLUDE_BC" = "1" ] && ARMS+=("train_pusht_unet_bc||BC")
for K in $WIDTHS; do
    for tok in $LADDERS; do
        ov=$(ladder_override "$tok" "$K") || { echo "unknown ladder token: $tok" >&2; exit 1; }
        ARMS+=("train_pusht_diffusion_search|n_candidates=$K $ov|k$K $tok")
    done
done

DOV="task.dataset.split_file=$SPLIT_FILE transition_file=$TRANS_FILE n_demos=$N_DEMOS"
DOV="$DOV task.dataset.n_test_episodes=$N_TEST task.dataset.n_val_episodes=$N_VAL"
DOV="$DOV split_suffix=$SPLIT_SUFFIX"
# NO IN-TRAINING ROLLOUTS. The runner resets to the held-out val+test episodes, so the
# default `rollout_every_steps: 20000` would score a run on its own held-out set five times
# per run and log it -- whether or not a val split exists, since test is in that set either
# way. Success is measured after training by eval_search_pusht at named checkpoints, so the
# in-training rollouts buy nothing and are switched off rather than ignored. Must stay a
# multiple of checkpoint_every; 10^8 is.
DOV="$DOV training.rollout_every_steps=100000000"

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)

# A STALE WANDB_API_KEY in the submitting shell overrides ~/.netrc inside the job and kills
# it at wandb.init with 401, after the GPU has already been allocated.
if [ -n "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "WARNING: WANDB_API_KEY is set and ~/.netrc has no wandb entry; keeping it" >&2
else
    unset WANDB_API_KEY
fi

if ! $PY -c 'import hydra' 2>/dev/null; then
    echo "ERROR: '$PY' cannot import hydra. conda activate \${DP_CONDA_ENV:-vae_pushT_l2s}" >&2
    exit 1
fi

# WALK the allocator's list rather than calling it per job. hyakalloc reports the scheduler's
# view, which does not reflect a job submitted seconds ago -- so N calls in a row return the
# same answer N times and pile every job onto one partition. pick_gpu.sh prints every viable
# account/partition, best first, and says so in its own header.
mapfile -t TARGETS < <(bash scripts/slurm/pick_gpu.sh 2>/dev/null)
if [ -n "$SUBMIT" ] && [ "${#TARGETS[@]}" -eq 0 ]; then
    echo "no account/partition can currently run this job; try later" >&2; exit 1
fi

n=0; skipped=0
for arm in "${ARMS[@]}"; do
    IFS='|' read -r cfg ov label <<<"$arm"
    all_ov="$ov $DOV"
    err=$(mktemp)
    resolved=$($PY train.py --config-name="$cfg" $all_ov --cfg job --resolve 2>"$err") || true
    run=$(sed -n 's/^run_name: //p'   <<<"$resolved")
    sub=$(sed -n 's/^trainer: //p'    <<<"$resolved")
    task=$(sed -n 's/^task_name: //p' <<<"$resolved")
    if [ -z "$run" ] || [ -z "$sub" ] || [ -z "$task" ]; then
        printf '%-78s %s\n' "$label" "CONFIG FAILED TO RESOLVE, skipping"
        sed -e 's/^/    | /' "$err" | tail -5 >&2; rm -f "$err"; skipped=$((skipped+1)); continue
    fi
    rm -f "$err"
    dir="$ROOT/$task/$sub/$run"; name="tr_$run"
    if grep -qxF "$name" <<<"$LIVE"; then
        printf '%-78s %s\n' "$run" "already training, skipping"; skipped=$((skipped+1)); continue
    fi
    if [ -f "$dir/checkpoints/step_0100000.ckpt" ]; then
        printf '%-78s %s\n' "$run" "already finished, skipping"; skipped=$((skipped+1)); continue
    fi
    if [ -z "$SUBMIT" ]; then
        printf '%-78s WOULD SUBMIT  (%s)\n' "$run" "$label"
        n=$((n+1)); continue
    fi
    read -r A P <<<"${TARGETS[$((n % ${#TARGETS[@]}))]}"
    jid=$(sbatch --parsable --account="$A" --partition="$P" --job-name="$name" \
          --time="$TIME_LIMIT" $EXCL_FLAG \
          --export=ALL,CONFIG_NAME="$cfg" \
          scripts/slurm/train_pusht_search.sbatch $all_ov)
    printf '%-78s submitted %s on %s/%s\n' "$run" "$jid" "$A" "$P"
    n=$((n+1))
done
echo
echo "arms handled: $n   skipped: $skipped   time limit: $TIME_LIMIT"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
