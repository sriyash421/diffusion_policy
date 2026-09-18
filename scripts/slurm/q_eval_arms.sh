#!/usr/bin/env bash
# Q as the best-of-N verifier: how it ranks a*, and whether it improves success through search.
#
#   bash scripts/slurm/q_eval_arms.sh                        # dry run
#   SUBMIT=1 bash scripts/slurm/q_eval_arms.sh               # existing ckpts, every 1M
#   SUBMIT=1 STEPS="4900000 5000000" bash scripts/slurm/q_eval_arms.sh    # upcoming, every 100k
#
# TWO evaluations per (SAC checkpoint, ST checkpoint) pair:
#   rank-expert  -- where the expert chunk a* ranks among n candidates under Q vs the heuristic
#   bon-sweep    -- does best-of-N on Q solve more episodes, n = 1..16
# They answer different questions and neither implies the other: the V arm ranked a* better
# than t_goal on the pre-contact half and still lost every best-of-N curve (report section 6).
set -uo pipefail

ROOT=/mmfs1/home/harine/diffusion_policy_standalone
SAC="${SAC:-/gscratch/robotics/harine/value_arms/sac_keypoint}"
OUT="${OUT:-/gscratch/robotics/harine/value_arms/qeval}"
# Existing checkpoints at every 1M. The upcoming ones are evaluated every 100k by passing STEPS.
STEPS="${STEPS:-1000000 2000000 3000000 4000000}"
SPLIT="${SPLIT:-test}"
CPUS="${CPUS:-8}"
MAX_N="${MAX_N:-16}"
SWEEP="${SWEEP:-1}"            # SWEEP=0 for the cheap rank-only pass at a fast cadence

# unetbc and ST-k16, both geometric split families -- the same four the V sweep covered, so the
# Q and V numbers sit on identical episodes and are directly comparable.
B=ckpts/pusht_search/pusht_image_search_imgonly
CKPTS="
unetbc_blq137     | $B/unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-137_split-blq_seed-42
unetbc_brd60      | $B/unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-60_split-brd60_seed-42
value_k16_blq137  | $B/outer_inner/value_k16_ver-t_goal_enc-resnet18_demos-137_split-blq_seed-42
value_k16_brd60   | $B/outer_inner/value_k16_ver-t_goal_enc-resnet18_demos-60_split-brd60_seed-42
"

submit() {   # name, command
    if [ "${SUBMIT:-0}" = "1" ]; then
        echo "[SUBMIT $1]"
        sbatch --partition=ckpt --account=robotics --requeue --job-name="$1" \
               --gpus=1 --cpus-per-task="$CPUS" --mem=32G --time=12:00:00 \
               --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out \
               --wrap "set -u; source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh; \
                       conda activate robodiff; cd $ROOT; unset WANDB_API_KEY; $2" >/dev/null
    else
        echo "[$1] $2"
    fi
}

for step in $STEPS; do
    q="$SAC/model_${step}_steps.zip"
    [ -f "$q" ] || { echo "[SKIP ${step}] no $q"; continue; }
    m=$((step / 1000000)).$(( (step % 1000000) / 100000 ))M
    while IFS='|' read -r label run; do
        label="$(echo "$label" | xargs)"; run="$(echo "$run" | xargs)"
        [ -z "$label" ] && continue
        ck="$run/checkpoints/step_0030000.ckpt"
        [ -f "$ROOT/$ck" ] || { echo "[SKIP $label] no $ck"; continue; }

        submit "qr_${label}_${m}" \
            "python sac/eval.py rank-expert -c $ck --arm $label --q $q --n $MAX_N \
                    --episodes 20 --split $SPLIT --out $OUT/rank_${label}_${m}.json"

        [ "$SWEEP" = "1" ] && submit "qs_${label}_${m}" \
            "python sac/eval.py bon-sweep -c $ck --q $q --rankers t_goal,q --max-n $MAX_N \
                    --split $SPLIT --n-envs $CPUS -o $OUT/bon_${label}_${m}"
    done <<< "$CKPTS"
done

[ "${SUBMIT:-0}" = "1" ] || echo $'\nDry run. SUBMIT=1 to sbatch.'
