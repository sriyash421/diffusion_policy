#!/usr/bin/env bash
# Q as the best-of-N verifier. THE THREE MEASUREMENTS, as asked for:
#
#   1. CONTACT -- on candidate sets where some candidate moved the T, does Q's ranking match
#      t_goal's?  diag_ranker_agreement.py: Spearman over the candidate set (order, not
#      magnitude -- selection reads rank) and argmax agreement against 1/n chance, split
#      contact vs no-contact on t_goal's own BLIND_EPS. Never pooled: with no contact t_goal
#      has no ordering, so "agreement" there is agreement with a tie-break.
#
#   2. RANK + SEARCH -- on the step-30k BC and ST-k16 checkpoints:
#        rank-expert : where the expert chunk a* lands among n candidates, Q vs t_goal
#        bon-sweep   : does best-of-N on Q solve more episodes, n = 1..16
#      Neither implies the other. The V arm ranked a* better than t_goal on the pre-contact
#      half and still lost every best-of-N curve (report section 6).
#
#   3. NO CONTACT -- 10 frames where t_goal is blind, every candidate chunk drawn on the
#      decision-state frame and coloured by its Q.  diag_q_frames.py. Read it for whether Q
#      prefers candidates that move the arm TOWARD the T; that is visible by eye long before
#      it is significant in a success rate.
#
# NOT HELD OUT, and that is deliberate. SAC is demo-seeded from ALL 206 episodes, so every eval
# episode is in its buffer. The earlier 106-episode filter held out only the seed-42 manifest
# while the sweep scores on the geometric ones -- 27 of blq137's 50 test episodes and 24 of
# brd60's 50 were in the buffer anyway. Uniform and declared beats partial and hidden. Say it
# beside every Q number; t_goal has no training set at all.
#
#   bash scripts/slurm/q_eval_arms.sh                       # dry run
#   SUBMIT=1 bash scripts/slurm/q_eval_arms.sh              # every new 100k checkpoint
#   SUBMIT=1 STEPS="1000000 2000000" bash scripts/slurm/q_eval_arms.sh
#   SUBMIT=1 EVERY=1000000 bash scripts/slurm/q_eval_arms.sh    # coarser cadence
set -uo pipefail

ROOT=/mmfs1/home/harine/diffusion_policy_standalone
SAC="${SAC:-/gscratch/robotics/harine/value_arms/sac_keypoint}"
OUT="${OUT:-/gscratch/robotics/harine/value_arms/qeval}"
EVERY="${EVERY:-100000}"
SPLIT="${SPLIT:-test}"
CPUS="${CPUS:-8}"
MAX_N="${MAX_N:-16}"
SWEEP="${SWEEP:-1}"        # SWEEP=0 skips the ~1.5h best-of-N run, keeping the cheap metrics

# unetbc and ST-k16, both geometric split families -- the same four the V sweep covered, so the
# Q and V numbers sit on identical episodes and are directly comparable.
B=ckpts/pusht_search/pusht_image_search_imgonly
CKPTS="
unetbc_blq137     | $B/unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-137_split-blq_seed-42
unetbc_brd60      | $B/unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-60_split-brd60_seed-42
value_k16_blq137  | $B/outer_inner/value_k16_ver-t_goal_enc-resnet18_demos-137_split-blq_seed-42
value_k16_brd60   | $B/outer_inner/value_k16_ver-t_goal_enc-resnet18_demos-60_split-brd60_seed-42
"

# Default: every checkpoint on disk at the EVERY cadence that has no result yet, so this is
# re-runnable as a watcher and never repeats work.
if [ -z "${STEPS:-}" ]; then
    STEPS=""
    for f in "$SAC"/model_*_steps.zip; do
        [ -e "$f" ] || continue
        st="${f##*model_}"; st="${st%%_steps.zip}"
        [ $((st % EVERY)) -eq 0 ] && STEPS="$STEPS $st"
    done
fi

submit() {   # name, timelimit, command -- skips anything already recorded
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

for step in $STEPS; do
    q="$SAC/model_${step}_steps.zip"
    [ -f "$q" ] || { echo "[SKIP ${step}] no $q"; continue; }
    m="$(printf '%d.%dM' $((step / 1000000)) $(((step % 1000000) / 100000)))"
    while IFS='|' read -r label run; do
        label="$(echo "$label" | xargs)"; run="$(echo "$run" | xargs)"
        [ -z "$label" ] && continue
        ck="$run/checkpoints/step_0030000.ckpt"
        [ -f "$ROOT/$ck" ] || { echo "[SKIP $label] no $ck"; continue; }

        # (1) contact: does Q's order match t_goal's?
        [ -f "$OUT/agree_${label}_${m}.json" ] || submit "qa_${label}_${m}" 2:00:00 \
            "python diag_ranker_agreement.py --ckpt $ck --q $q --n $MAX_N --episodes 20 \
                    --split $SPLIT --out $OUT/agree_${label}_${m}.json"

        # (2a) where a* ranks
        [ -f "$OUT/rank_${label}_${m}.json" ] || submit "qr_${label}_${m}" 2:00:00 \
            "python sac/eval.py rank-expert -c $ck --arm $label --q $q --n $MAX_N \
                    --episodes 20 --split $SPLIT --out $OUT/rank_${label}_${m}.json"

        # (2b) does search on Q solve more episodes
        [ "$SWEEP" = "1" ] && [ ! -f "$OUT/bon_${label}_${m}/bon_curves.json" ] && \
            submit "qs_${label}_${m}" 12:00:00 \
            "python sac/eval.py bon-sweep -c $ck --q $q --rankers t_goal,q --max-n $MAX_N \
                    --split $SPLIT --n-envs $CPUS -o $OUT/bon_${label}_${m}"

        # (3) no-contact frames -- one arm is enough to look at, and this is for looking
        [ "$label" = "value_k16_blq137" ] && [ ! -f "$OUT/frames_${m}.png" ] && \
            submit "qf_${m}" 2:00:00 \
            "python diag_q_frames.py --ckpt $ck --q $q --frames 10 --n $MAX_N \
                    --episodes 20 --split $SPLIT --out $OUT/frames_${m}.png"
    done <<< "$CKPTS"
done

[ "${SUBMIT:-0}" = "1" ] || echo $'\nDry run. SUBMIT=1 to sbatch.'
