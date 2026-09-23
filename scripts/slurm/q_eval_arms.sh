#!/usr/bin/env bash
# Q as the best-of-N verifier, on episodes the Q was NEVER SEEDED FROM.
#
# WHAT CHANGED, AND WHY. The previous generation trained ONE Q seeded from all 206 demos and
# said so beside every number: uniform and declared beat partial and hidden, because seed-42's
# 106 train episodes covered 27 of blq137's 50 test episodes and 24 of brd60's, so restricting
# to that manifest bought a half-clean hold-out -- bias landing unevenly across strata, and only
# on the learned side, since t_goal has no training set at all. The fix is not a better excuse
# but a different partition: train the Q on the GEOMETRIC manifest the sweep scores on, and the
# overlap is zero rather than uneven. Those runs are sac_keypoint_blq137 / sac_keypoint_brd100.
#
# WHICH Q SCORES WHICH CHECKPOINT. Three groups, and the third is the one that needs an argument:
#
#   blq137 ckpts <- blq137 Q     0 of 50 test episodes in the buffer
#   brd100 ckpts <- brd100 Q     0 of 50
#   brd60  ckpts <- brd100 Q     0 of 50, because brd60 and brd100 are the same blockborder
#                                family: their TEST sets are the identical 50 interior episodes,
#                                and brd60.train is a strict SUBSET of brd100.train. So the
#                                brd100 Q holds out brd60's test set exactly as cleanly as its
#                                own, and no separate brd60 Q is needed.
#
# `--split test` IS NOT A DEFAULT HERE, IT IS THE PRECONDITION. brd60's VAL split is 40 episodes
# and every one of them is inside brd100.train. The brd60 rows are sound on test and contaminated
# on val, so the split is not a knob. `sac/score.py:assert_held_out` enforces all of this and
# refuses rather than warns; it is the guard, and this comment is only the reason.
#
# THE THREE MEASUREMENTS, as asked for:
#
#   1. CONTACT -- on candidate sets where some candidate moved the T, does Q's ranking match
#      t_goal's?  diag_ranker_agreement.py: Spearman over the candidate set (order, not
#      magnitude -- selection reads rank) and argmax agreement against 1/n chance, split
#      contact vs no-contact on t_goal's own BLIND_EPS. Never pooled: with no contact t_goal
#      has no ordering, so "agreement" there is agreement with a tie-break.
#
#   2. RANK + SEARCH -- on the step-30k checkpoints:
#        rank-expert : where the expert chunk a* lands among n candidates, Q vs t_goal, reported
#                      both by candidate informativeness and by whether the DEMO's own T moved
#                      over the window (block_moving / block_still). The second split is a
#                      property of the state, so it is byte-identical across every arm compared.
#        bon-sweep   : does best-of-N on Q solve more episodes, n = 1..16
#      Neither implies the other. The V arm ranked a* better than t_goal on the pre-contact
#      half and still lost every best-of-N curve (report section 6).
#
#   3. NO CONTACT -- 10 frames where t_goal is blind, every candidate chunk drawn on the
#      decision-state frame and coloured by its Q.  diag_q_frames.py. Read it for whether Q
#      prefers candidates that move the arm TOWARD the T; that is visible by eye long before
#      it is significant in a success rate.
#
# CADENCE IS A SUBMIT-TIME DECISION, not a default baked in here. bon-sweep is ~12h against 11
# rows, so SWEEP=0 plus an explicit STEPS is the cheap read and SWEEP=1 is the whole budget.
# Likewise --episodes 20 scores a PREFIX of the 50 held-out episodes, and _stats bootstraps
# clustered on episode, so raising it tightens the CI at a proportional cost in wall clock --
# raise EP_TIME with it.
#
#   bash scripts/slurm/q_eval_arms.sh                            # dry run
#   SUBMIT=1 bash scripts/slurm/q_eval_arms.sh                   # every new checkpoint at EVERY
#   SUBMIT=1 SWEEP=0 STEPS=10000000 bash scripts/slurm/q_eval_arms.sh
#   SUBMIT=1 EPISODES=50 EP_TIME=4:00:00 bash scripts/slurm/q_eval_arms.sh
set -uo pipefail

ROOT=/mmfs1/home/harine/diffusion_policy_standalone
ARMS_ROOT="${ARMS_ROOT:-/gscratch/robotics/harine/value_arms}"
OUT="${OUT:-$ARMS_ROOT/qeval}"
EVERY="${EVERY:-100000}"
CPUS="${CPUS:-8}"
MAX_N="${MAX_N:-16}"
EPISODES="${EPISODES:-20}"     # a PREFIX of the 50 held-out episodes; see the cadence note
EP_TIME="${EP_TIME:-2:00:00}"  # raise with EPISODES
SWEEP="${SWEEP:-1}"            # SWEEP=0 skips the ~12h best-of-N run, keeping the cheap metrics

# qgroup -> the SAC run whose Q that group is scored with
declare -A QRUN=(
  [blq137]="$ARMS_ROOT/sac_keypoint_blq137"
  [brd100]="$ARMS_ROOT/sac_keypoint_brd100"
)

B=ckpts/pusht_search/pusht_image_search_imgonly
# qgroup | label | run.  The four arms are STk16-uniform, son-flat400, unet BC and STk1; brd100
# has no son-flat400 run on disk, so that cell is absent rather than substituted.
ROWS="
blq137 | k16uniform_blq137 | $B/outer_inner/value_k16_ver-t_goal_enc-resnet18_demos-137_split-blq_seed-42
blq137 | flat400_blq137    | $B/outer_inner/value_k16_ver-t_goal_son-flat400_enc-resnet18_demos-137_split-blq_seed-42
blq137 | unetbc_blq137     | $B/unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-137_split-blq_seed-42
blq137 | k1_blq137         | $B/offline/value_k1_ver-t_goal_enc-resnet18_demos-137_split-blq_seed-42
brd100 | k16uniform_brd100 | $B/outer_inner/value_k16_ver-t_goal_enc-resnet18_demos-100_split-brd100_seed-42
brd100 | unetbc_brd100     | $B/unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-100_split-brd100_seed-42
brd100 | k1_brd100         | $B/offline/value_k1_ver-t_goal_enc-resnet18_demos-100_split-brd100_seed-42
brd100 | k16uniform_brd60  | $B/outer_inner/value_k16_ver-t_goal_enc-resnet18_demos-60_split-brd60_seed-42
brd100 | flat400_brd60     | $B/outer_inner/value_k16_ver-t_goal_son-flat400_enc-resnet18_demos-60_split-brd60_seed-42
brd100 | unetbc_brd60      | $B/unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-60_split-brd60_seed-42
brd100 | k1_brd60          | $B/offline/value_k1_ver-t_goal_enc-resnet18_demos-60_split-brd60_seed-42
"

submit() {   # name, timelimit, command -- skips anything already queued
    if [ "${SUBMIT:-0}" = "1" ]; then
        squeue -u "$USER" -h -o '%j' | grep -qx "$1" && { echo "[QUEUED $1] already"; return; }
        echo "[SUBMIT $1]"
        sbatch --partition=ckpt --account=robotics --requeue --job-name="$1" \
               --gpus=1 --cpus-per-task="$CPUS" --mem=32G --time="$2" \
               --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out \
               --wrap "set -u; source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh; \
                       conda activate robodiff; cd $ROOT; unset WANDB_API_KEY; $3" >/dev/null
    else
        echo "[$1] $3"
    fi
}

# (0) the SAC arm's OWN success rate on the episodes it holds out. Not a Q measurement, but the
# thing every Q number has to be read against: a Q from a policy that never solves the task is
# a different object from one that does, and the two are not comparable.
for g in "${!QRUN[@]}"; do
    run="${QRUN[$g]}"
    [ -d "$run" ] || { echo "[SKIP policy_$g] no $run"; continue; }
    submit "qp_$g" 12:00:00 \
        "python sac/eval.py policy $run --watch --split test --idle-exit-sec 7200 \
                --out-dir $OUT/$g/policy"
done

for g in "${!QRUN[@]}"; do
    SAC="${QRUN[$g]}"
    [ -d "$SAC" ] || { echo "[SKIP $g] no $SAC"; continue; }

    # Default: every checkpoint on disk at the EVERY cadence that has no result yet, so this is
    # re-runnable as a watcher and never repeats work.
    if [ -z "${STEPS:-}" ]; then
        GSTEPS=""
        for f in "$SAC"/model_*_steps.zip; do
            [ -e "$f" ] || continue
            st="${f##*model_}"; st="${st%%_steps.zip}"
            [ $((st % EVERY)) -eq 0 ] && GSTEPS="$GSTEPS $st"
        done
    else
        GSTEPS="$STEPS"
    fi

    for step in $GSTEPS; do
        q="$SAC/model_${step}_steps.zip"
        [ -f "$q" ] || { echo "[SKIP $g ${step}] no $q"; continue; }
        m="$(printf '%d.%dM' $((step / 1000000)) $(((step % 1000000) / 100000)))"
        o="$OUT/$g"
        [ "${SUBMIT:-0}" = "1" ] && mkdir -p "$o"

        while IFS='|' read -r qg label run; do
            qg="$(echo "$qg" | xargs)"; label="$(echo "$label" | xargs)"; run="$(echo "$run" | xargs)"
            [ -z "$qg" ] || [ "$qg" != "$g" ] && continue
            ck="$run/checkpoints/step_0030000.ckpt"
            [ -f "$ROOT/$ck" ] || { echo "[SKIP $label] no $ck"; continue; }

            # (1) contact: does Q's order match t_goal's?
            [ -f "$o/agree_${label}_${m}.json" ] || submit "qa_${label}_${m}" 2:00:00 \
                "python diag_ranker_agreement.py --ckpt $ck --q $q --n $MAX_N \
                        --episodes $EPISODES --split test --out $o/agree_${label}_${m}.json"

            # (2a) where a* ranks -- reported by informativeness AND by block_moving/block_still
            [ -f "$o/rank_${label}_${m}.json" ] || submit "qr_${label}_${m}" "$EP_TIME" \
                "python sac/eval.py rank-expert -c $ck --arm $label --q $q --n $MAX_N \
                        --episodes $EPISODES --split test --out $o/rank_${label}_${m}.json"

            # (2b) does search on Q solve more episodes
            [ "$SWEEP" = "1" ] && [ ! -f "$o/bon_${label}_${m}/bon_curves.json" ] && \
                submit "qs_${label}_${m}" 12:00:00 \
                "python sac/eval.py bon-sweep -c $ck --q $q --rankers t_goal,q --max-n $MAX_N \
                        --split test --n-envs $CPUS -o $o/bon_${label}_${m}"

            # (3) no-contact frames -- one arm per group is enough to look at, and this is for
            # looking
            [ "$label" = "k16uniform_$g" ] && [ ! -f "$o/frames_${m}.png" ] && \
                submit "qf_${g}_${m}" 2:00:00 \
                "python diag_q_frames.py --ckpt $ck --q $q --frames 10 --n $MAX_N \
                        --episodes $EPISODES --split test --out $o/frames_${m}.png"
        done <<< "$ROWS"
    done
done

[ "${SUBMIT:-0}" = "1" ] || echo $'\nDry run. SUBMIT=1 to sbatch.'
