#!/usr/bin/env bash
# THE LEARNED-Q GENERATION. 12 ST arms: 2 widths x 3 obs-noise ladders x 2 geometric splits,
# every one trained with a trained SAC chunk-Q as the verifier instead of the `t_goal`
# distance heuristic.
#
# WHY. Every ST arm before this trained on a `t_goal`-shaped search context, which is exactly
# why `sac/score.py` will only swap the Q into the RANKING at eval and leaves the context
# alone -- handing a t_goal-trained arm a Q-shaped context confounds "Q ranks better" with
# "ST was given an input it has never seen". The only way out of that confound is to train
# arms whose context is Q from the start. These are those arms.
#
#   THE VERIFIER IS NAMED, AND THE NAME IS THE CHECKPOINT. `verifier_tag=q_sac_all` resolves
#   through pusht_verifier.Q_VERIFIERS to a specific SAC checkpoint, and lands in run_name as
#   `ver-q_sac_all`. There is no path override: a run directory that says which Q scored it
#   and a config key that could say otherwise are two things that drift.
#
#   NO SIM POOL. PushTQVerifier scores (state, chunk) in one forward pass, so the 32-process
#   PushT pool is never forked. `trainer: outer_inner` is kept anyway -- its amortization is
#   now moot, but the 3 existing t_goal arms on blq137 use it and matching them is what makes
#   the comparison about the verifier rather than about the trainer.
#
#   ⚠️ THE Q IS NOT HELD OUT. sac_keypoint ran with `demo_episodes: all`, so all 206 episodes
#   -- every test episode of both splits included -- seeded its replay buffer. This is a
#   caveat on every number these arms produce, not something the launcher can fix.
#
#   CONTROLS ARE PARTIAL. Under t_goal only blq137 k16 {uniform, flat400, ramp400to200} exist
#   (100k) plus brd100 k16 uniform. The k=4 arms on both splits and the brd100 ladders have
#   NO matched twin; those comparisons are across generations and must be labelled unpaired.
#
#   RE-RUNNING IS THE REPAIR, as in run_tgoal_moving.sh: the launcher skips any arm already
#   training or already at step 100k, so `SUBMIT=1` a second time resubmits exactly what died.
#
#   bash scripts/run_q_geometric.sh                      # dry run
#   SUBMIT=1 bash scripts/run_q_geometric.sh             # ...and sbatch
#   DATASET_TAGS=_split-blq SUBMIT=1 bash scripts/run_q_geometric.sh   # one split
#   VERIFIER=q_sac_106 SUBMIT=1 bash scripts/run_q_geometric.sh        # the leak control
set -uo pipefail
cd "$(dirname "$0")/.."

SUBMIT="${SUBMIT:-}"
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search
PY="${DP_PY:-python}"
SPL=diffusion_policy/config/splits
VERIFIER="${VERIFIER:-q_sac_all}"
# Cap the wall clock so the job ends before the next maintenance reservation; SLURM parks a
# job that would run past one on the far side of the window. See safe_time_limit.sh.
TIME_LIMIT="${TIME_LIMIT:-$(bash scripts/slurm/safe_time_limit.sh 3-00:00:00)}"
# A GPU with failing memory takes a job down minutes in with an uncorrectable ECC error and
# SLURM does not drain the node for it. Comma-separated, e.g. EXCLUDE_NODES=g3070,g3081
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
EXCL_FLAG=""; [ -n "$EXCLUDE_NODES" ] && EXCL_FLAG="--exclude=$EXCLUDE_NODES"

# tag | split_file | n_demos | n_test | n_val -- must match run_geometric_splits.sh exactly.
# brd100's n_val is 0 because the border band of 100 leaves no ring over; the dataset skips
# the count check when n_val_episodes is falsy and both workspaces guard their val loop on a
# non-empty split, so that is a supported configuration rather than a hole.
DATASETS=(
  "_split-blq|$SPL/pusht_blockquad_bottomleft_train137.json|137|50|19"
  "_split-brd100|$SPL/pusht_blockborder_train100_core50.json|100|50|0"
)
DATASET_TAGS="${DATASET_TAGS:-}"   # optional filter, space-separated tags

# Resolved by the registry, so a bad name fails here rather than after the GPU is allocated.
if ! $PY -c "from diffusion_policy.env.pusht.pusht_verifier import q_verifier_spec; \
             print('[verifier] %s -> %s (rung %d)' % (('$VERIFIER',) + q_verifier_spec('$VERIFIER')))"; then
    echo "ERROR: verifier '$VERIFIER' does not resolve; see pusht_verifier.Q_VERIFIERS" >&2
    exit 1
fi

# Built here rather than pasted, so a ladder cannot drift from the name it is filed under.
mk_flat() { $PY -c "print('['+','.join(['$1']*$2)+']')"; }                     # t, K
mk_ramp() { $PY -c "a,b,k=$1,$2,$3; print('['+','.join(str(int(round(a-(a-b)*i/(k-1)))) for i in range(k))+']')"; }

WIDTHS="${WIDTHS:-16 4}"
LADDERS="${LADDERS:-uniform flat400 ramp400to200}"

# Token grammar, identical to run_tgoal_moving.sh. The timesteps are built at the arm's own
# K, so a name can never disagree with the profile it is filed under; _slot_obs_profile
# raises if len(timesteps) != n_candidates, which is why these take K.
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

# NO IN-TRAINING ROLLOUTS. The runner resets to the held-out val+test episodes, and this
# generation's readout is the offline best-of-n sweep, so the default rollout_every_steps
# would score the arms on their own held-out set five times a run and log it for nothing.
# Must stay a multiple of checkpoint_every; 10^8 is.
COMMON="verifier_tag=$VERIFIER training.rollout_every_steps=100000000"

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

# WALK the allocator's list rather than calling it per job: hyakalloc reports the scheduler's
# view, which does not reflect a job submitted seconds ago, so N calls in a row return the
# same answer N times and pile every job onto one partition.
mapfile -t TARGETS < <(bash scripts/slurm/pick_gpu.sh 2>/dev/null)
if [ -n "$SUBMIT" ] && [ "${#TARGETS[@]}" -eq 0 ]; then
    echo "no account/partition can currently run this job; try later" >&2; exit 1
fi

n=0; skipped=0
for ds in "${DATASETS[@]}"; do
    IFS='|' read -r tag file demos ntest nval <<<"$ds"
    if [ -n "$DATASET_TAGS" ] && ! grep -qw -- "$tag" <<<"$DATASET_TAGS"; then continue; fi
    [ -f "$file" ] || { echo "MISSING SPLIT: $file" >&2; exit 1; }
    dov="task.dataset.split_file=$file n_demos=$demos split_suffix=$tag"
    dov="$dov task.dataset.n_test_episodes=$ntest task.dataset.n_val_episodes=$nval"
    for K in $WIDTHS; do
        for tok in $LADDERS; do
            ov=$(ladder_override "$tok" "$K") || { echo "unknown ladder token: $tok" >&2; exit 1; }
            all_ov="n_candidates=$K $ov $dov $COMMON"
            label="${tag#_split-} k$K $tok"
            err=$(mktemp)
            resolved=$($PY train.py --config-name=train_pusht_diffusion_search $all_ov \
                       --cfg job --resolve 2>"$err") || true
            run=$(sed -n 's/^run_name: //p'   <<<"$resolved")
            sub=$(sed -n 's/^trainer: //p'    <<<"$resolved")
            task=$(sed -n 's/^task_name: //p' <<<"$resolved")
            if [ -z "$run" ] || [ -z "$sub" ] || [ -z "$task" ]; then
                printf '%-84s %s\n' "$label" "CONFIG FAILED TO RESOLVE, skipping"
                sed -e 's/^/    | /' "$err" | tail -5 >&2; rm -f "$err"
                skipped=$((skipped+1)); continue
            fi
            rm -f "$err"
            dir="$ROOT/$task/$sub/$run"; name="tr_$run"
            if grep -qxF "$name" <<<"$LIVE"; then
                printf '%-84s %s\n' "$run" "already training, skipping"
                skipped=$((skipped+1)); continue
            fi
            if [ -f "$dir/checkpoints/step_0100000.ckpt" ]; then
                printf '%-84s %s\n' "$run" "already finished, skipping"
                skipped=$((skipped+1)); continue
            fi
            if [ -z "$SUBMIT" ]; then
                printf '%-84s WOULD SUBMIT  (%s)\n' "$run" "$label"
                n=$((n+1)); continue
            fi
            read -r A P <<<"${TARGETS[$((n % ${#TARGETS[@]}))]}"
            jid=$(sbatch --parsable --account="$A" --partition="$P" --job-name="$name" \
                  --time="$TIME_LIMIT" $EXCL_FLAG \
                  --export=ALL,CONFIG_NAME=train_pusht_diffusion_search \
                  scripts/slurm/train_pusht_search.sbatch $all_ov)
            printf '%-84s submitted %s on %s/%s\n' "$run" "$jid" "$A" "$P"
            n=$((n+1))
        done
    done
done
echo
echo "arms handled: $n   skipped: $skipped   verifier: $VERIFIER   time limit: $TIME_LIMIT"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
