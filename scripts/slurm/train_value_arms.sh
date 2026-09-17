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

# Asked for, and filtered on, the SAME numbers -- pick_gpu.sh's whole point is that a partition
# can show a free GPU while its CPU or memory allowance is exhausted, and a job sent there sits
# in AssocGrpCpuLimit forever rather than failing. Modest on purpose: PushT stepping is cheap,
# the rollout buffer holds uint8 frames, and 8 CPUs against the 128 free on gpu-l40 leaves the
# node usable by everyone else.
CPUS="${CPUS:-8}"
MEM_G="${MEM_G:-32}"
# MUST be finite and MUST end before the next maintenance window -- an unlimited job can never
# be proven to finish before one, so SLURM parks it with Reason=ReqNodeNotAvail and it silently
# never starts. Next window: 2026-10-13 09:00. Re-check with
#   scontrol show reservation | grep -A1 Maintenance
TIME="${TIME:-5-00:00:00}"

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

# ONE call, then round-robin. hyakalloc reports the SCHEDULER's view, which does not include a
# job submitted seconds ago -- so calling this once per arm returns the same answer every time
# and piles all ten onto one partition. pick_gpu.sh prints every candidate best-first precisely
# so a multi-job caller can walk it, which is what this does.
TARGETS=()
if [ "${SUBMIT:-0}" = "1" ]; then
    while read -r acct part; do
        [ -n "$acct" ] && TARGETS+=("$acct $part")
    done < <(NEED_CPUS=$CPUS NEED_MEM_G=$MEM_G NEED_GPUS=1 bash "$ROOT/scripts/slurm/pick_gpu.sh")
    [ ${#TARGETS[@]} -eq 0 ] && { echo "no partition has ${CPUS} CPUs + ${MEM_G}G + 1 GPU free" >&2; exit 1; }
    echo "[INFO] ${#TARGETS[@]} candidate partition(s): ${TARGETS[*]}"
fi
NEXT=0
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
        read A P <<< "${TARGETS[$((NEXT % ${#TARGETS[@]}))]}"
        NEXT=$((NEXT + 1))
        echo "[SUBMIT $label] account=$A partition=$P"
        sbatch --account="$A" --partition="$P" --job-name="va_$label" \
               --gpus=1 --cpus-per-task="$CPUS" --mem="${MEM_G}G" --time="$TIME" \
               --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out \
               --wrap "set -u; source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh; \
                       conda activate robodiff; cd $ROOT; unset WANDB_API_KEY; $cmd"
    else
        echo "[$label] $cmd"
    fi
done <<< "$ARMS_ALL"

[ "${SUBMIT:-0}" = "1" ] || echo $'\nDry run. SUBMIT=1 to sbatch.'

# Scoring the finished arms on the SHARED episodes. A separate step on purpose: PPO's
# in-training evaluation is seeded procedural resets, a monitoring signal, while the number
# comparable with ST / BC-UNet comes from rolling the manifest's own episodes.
#
#   EVAL=1 bash scripts/slurm/train_value_arms.sh
#
# ON THE ckpt PARTITION, preemptible and free, so the guaranteed robotics/weirdlab GPUs stay
# available for training. --requeue re-runs after preemption; eval_checkpoints re-seeds per
# checkpoint, so a requeued job repeats work rather than corrupting it.
if [ -n "$SCORE" ]; then
    if [ "${EVAL:-0}" = "1" ]; then
        while read -r run; do
            [ -z "$run" ] && continue
            [ -d "$run" ] || { echo "[SKIP $(basename "$run")] not trained yet"; continue; }
            echo "[EVAL $(basename "$run")] -> ckpt"
            sbatch --partition=ckpt --account=robotics --requeue \
                   --job-name="ve_$(basename "$run")" --gpus=1 --cpus-per-task="$CPUS" \
                   --mem="${MEM_G}G" --time=8:00:00 \
                   --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out \
                   --wrap "set -u; source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh; \
                           conda activate robodiff; cd $ROOT; unset WANDB_API_KEY; \
                           python -m recurrent_ppo.scripts.eval_checkpoints $run \
                                  --split-file $SPLIT --split test --num-envs $CPUS \
                                  --out $run/eval_test.json"
        done <<< "$SCORE"
    else
        echo $'\nWhen the PPO arms finish, score them on the SHARED episodes:'
        echo "  EVAL=1 bash scripts/slurm/train_value_arms.sh    # submits to the ckpt partition"
    fi
fi
