#!/usr/bin/env bash
# Launch the value-function generation: two BC stages, four PPO arms, two SAC arms.
#
# THIS SCRIPT EXISTS BECAUSE THE LAST GENERATION DID NOT HAVE ONE. Eighteen runs were launched
# by hand, each recording only its own command.txt, and when they were read months later the
# grid could be described but not re-launched. Every run below is reproducible from this file.
#
#   bash scripts/slurm/train_value_arms.sh            # dry run: print what would be submitted
#   SUBMIT=1 bash scripts/slurm/train_value_arms.sh   # ...and sbatch it
#   SUBMIT=1 ARMS="bc_keypoint ppo_lstm_keypoint_bc" bash scripts/slurm/train_value_arms.sh
#
# ORDER MATTERS for two of them: ppo_lstm_*_bc consume the bc_* checkpoints, so the BC stages
# must finish first. They are cheap (48k gradient steps against 10M env steps), so run them,
# check their val curves, and only then launch the fine-tunes.
set -uo pipefail

ROOT=/mmfs1/home/harine/diffusion_policy_standalone
LOGS="${LOGS:-$ROOT/logs/value_arms}"
SPLIT=diffusion_policy/config/splits/pusht_seed42_train106_val50.json
STEPS="${STEPS:-10000000}"
SEED="${SEED:-42}"

# --reward delta throughout: its return telescopes to total progress, so V estimates "coverage
# still to gain", which is what a ranker needs. A level-valued reward once paid a do-nothing
# policy 92.7.
#
# --video-freq 0 on every arm: the default is 24 and hard-errors without --wandb.
#
# NO --split-file ON THE PPO ARMS, and that is not an omission. They train on procedurally
# sampled starts -- there is no demonstration split to train on -- and their in-training
# evaluation is a monitoring signal, seeded but not manifest-pinned. The numbers that get
# COMPARED with the offline arms come from scoring the saved checkpoints afterwards:
#
#   python -m recurrent_ppo.scripts.eval_checkpoints <run> --split-file $SPLIT --split test \
#          --out <run>/eval_test.json
#
# which rolls the manifest's own episodes and keys every row to its episode index. SAC and the
# BC stage DO take --split-file, because both read demonstrations: SAC seeds its buffer from
# them and BC trains on them.
COMMON="--reward delta --seed $SEED --video-freq 0"

# label | entry | args
ARMS_ALL="
bc_keypoint            | -m recurrent_ppo.bc         | --obs keypoint
bc_image               | -m recurrent_ppo.bc         | --obs image --lstm-hidden-size 256
ppo_plain_keypoint     | -m recurrent_ppo.ppo.train  | --obs keypoint --n-stack 1
ppo_plain_image        | -m recurrent_ppo.ppo.train  | --obs image --n-stack 1
ppo_lstm_keypoint      | -m recurrent_ppo.train      | --obs keypoint
ppo_lstm_image         | -m recurrent_ppo.train      | --obs image --lstm-hidden-size 256
ppo_lstm_keypoint_bc   | -m recurrent_ppo.train      | --obs keypoint --bc-init BC/bc_keypoint/bc_best.zip
ppo_lstm_image_bc      | -m recurrent_ppo.train      | --obs image --lstm-hidden-size 256 --bc-init BC/bc_image/bc_best.zip
sac_keypoint           | sac/runner.py               | --obs keypoint
sac_image              | sac/runner.py               | --obs image
"

WANT="${ARMS:-}"
SCORE=""
while IFS='|' read -r label entry extra; do
    label="$(echo "$label" | xargs)"; entry="$(echo "$entry" | xargs)"; extra="$(echo "$extra" | xargs)"
    [ -z "$label" ] && continue
    if [ -n "$WANT" ] && ! grep -qw "$label" <<<"$WANT"; then continue; fi

    out="$LOGS/$label"
    extra="${extra//BC\//$LOGS/}"                    # --bc-init resolves against this LOGS root

    case "$entry" in
        *recurrent_ppo.bc)
            # the BC stage takes no --reward/--video-freq; it trains on demonstrations
            cmd="python $entry $extra --split-file $SPLIT --seed $SEED --out $out"
            ;;
        *sac/runner.py)
            # SAC's action is an absolute chunk, so --reward delta does not apply to it
            cmd="python $entry $extra --seed $SEED --split-file $SPLIT --total-timesteps $STEPS --log-dir $out"
            ;;
        *)
            cmd="python $entry $extra $COMMON --total-timesteps $STEPS --log-dir $out"
            SCORE="$SCORE$out"$'\n'
            ;;
    esac

    if [ "${SUBMIT:-0}" = "1" ]; then
        read A P < <(bash "$ROOT/scripts/slurm/pick_gpu.sh")
        echo "[SUBMIT $label] account=$A partition=$P"
        sbatch --account="$A" --partition="$P" --job-name="va_$label" \
               --gpus=1 --cpus-per-task=8 --mem=64G --time=2-00:00:00 \
               --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out \
               --wrap "set -u; source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh; \
                       conda activate robodiff; cd $ROOT; unset WANDB_API_KEY; $cmd"
    else
        echo "[$label] $cmd"
    fi
done <<< "$ARMS_ALL"

[ "${SUBMIT:-0}" = "1" ] || echo $'\nDry run. SUBMIT=1 to sbatch.'

if [ -n "$SCORE" ]; then
    echo $'\nWhen the PPO arms finish, score them on the SHARED episodes -- this is what makes'
    echo "their numbers comparable with ST / BC-UNet, and it is a separate step on purpose:"
    while read -r run; do
        [ -z "$run" ] && continue
        echo "  python -m recurrent_ppo.scripts.eval_checkpoints $run \\"
        echo "         --split-file $SPLIT --split test --out $run/eval_test.json"
    done <<< "$SCORE"
fi
