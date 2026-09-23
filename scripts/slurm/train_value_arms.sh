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
# ON /gscratch, NOT under the repo. $HOME is a 10 GB HARD quota and these write checkpoints:
# one 10M-step image arm produced 1.7 GB on its own, and four of six arms died on
# `OSError: [Errno 122] Disk quota exceeded` when logs/ was the default. The repo's sbatch files
# already send SLURM stdout here for the same reason; the model checkpoints are the larger half.
LOGS="${LOGS:-/gscratch/robotics/harine/value_arms}"
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

# ppo_plain_keypoint_near re-runs the one arm that learned anything (+7.8 eval reward, and still
# success_rate 0 at 10M) from a start distribution that hands it the last push: the agent 16-25px
# from the T's surface and the T 63px from the goal, median, against a uniform start's 167px.
#
# --block-coverage-max 0.2 is NOT a slackening of the no-overlap rule for its own sake. "T near
# the goal" and "T covering none of it" are geometrically incompatible -- the T must be displaced
# by roughly its own size to clear the goal -- so a ceiling is the only way to have both. The
# number is set by what is samplable: at offset 60 a ceiling of 0.05 accepts 0% of draws and 0.20
# accepts 34%, and SPAWN_TRIES is 20, so an acceptance rate much below ~0.3 silently falls back
# to an UNFILTERED start on (1-p)^20 of resets. Nothing starts solved: 0 of 400 draws had
# coverage >= 0.95. Under --reward delta start coverage is not paid per step anyway; that hazard
# was specific to the level-valued reward.
#
# --eval-curriculum off so eval/ stays the real uniform task, directly comparable to the four
# arms already finished, and best_model tracks it rather than the curriculum. The curriculum
# number is still logged, as eval_train/.
# sac_keypoint_blq137 / _brd100 are the same arm as sac_keypoint with its demo seeding
# restricted to ONE geometric manifest's train episodes, which is what makes that manifest's 50
# test episodes genuinely held out from the Q. sac_keypoint itself seeds from all 206 on purpose
# -- seed-42's 106 train episodes cover 27 of blq137's test episodes and 24 of brd60's, so the
# old restriction bought a half-clean hold-out, and uniform-and-declared beat partial-and-hidden.
# Restricting to the geometric manifest instead removes the overlap rather than spreading it.
# No brd60 arm: brd60.train is a strict subset of brd100.train and their test sets are the SAME
# 50 episodes, so the brd100 Q holds out brd60's test set too and a third run would add nothing.
# label | entry | args
ARMS_ALL="
bc_keypoint            | -m recurrent_ppo.bc         | --obs keypoint
bc_image               | -m recurrent_ppo.bc         | --obs image --lstm-hidden-size 256
ppo_plain_keypoint     | -m recurrent_ppo.ppo.train  | --obs keypoint --n-stack 1
ppo_plain_image        | -m recurrent_ppo.ppo.train  | --obs image --n-stack 1
ppo_plain_keypoint_near| -m recurrent_ppo.ppo.train  | --obs keypoint --n-stack 1 --agent-near-block-prob 1.0 --block-near-goal-prob 1.0 --block-goal-offset 60 0.5 --block-coverage-max 0.2 --eval-curriculum off
ppo_lstm_keypoint      | -m recurrent_ppo.train      | --obs keypoint
ppo_lstm_image         | -m recurrent_ppo.train      | --obs image --lstm-hidden-size 256
ppo_lstm_keypoint_bc   | -m recurrent_ppo.train      | --obs keypoint --bc-init BC/bc_keypoint/bc_best.zip
ppo_lstm_image_bc      | -m recurrent_ppo.train      | --obs image --lstm-hidden-size 256 --bc-init BC/bc_image/bc_best.zip
sac_keypoint           | sac/runner.py               | --obs keypoint
sac_image              | sac/runner.py               | --obs image
sac_keypoint_blq137    | sac/runner.py               | --obs keypoint --demo-episodes train --split-file diffusion_policy/config/splits/pusht_blockquad_bottomleft_train137.json
sac_keypoint_brd100    | sac/runner.py               | --obs keypoint --demo-episodes train --split-file diffusion_policy/config/splits/pusht_blockborder_train100_core50.json
"

WANT="${ARMS:-}"
SCORE=""

# ONE call, then round-robin. hyakalloc reports the SCHEDULER's view, which does not include a
# job submitted seconds ago -- so calling this once per arm returns the same answer every time
# and piles all ten onto one partition. pick_gpu.sh prints every candidate best-first precisely
# so a multi-job caller can walk it, which is what this does.
TARGETS=()
declare -A FREE_GPUS FREE_CPUS FREE_MEM_G
if [ "${SUBMIT:-0}" = "1" ]; then
    while read -r acct part; do
        [ -n "$acct" ] && TARGETS+=("$acct $part")
    done < <(NEED_CPUS=$CPUS NEED_MEM_G=$MEM_G NEED_GPUS=1 bash "$ROOT/scripts/slurm/pick_gpu.sh")
    [ ${#TARGETS[@]} -eq 0 ] && { echo "no partition has ${CPUS} CPUs + ${MEM_G}G + 1 GPU free" >&2; exit 1; }

    # WHAT EACH PARTITION CAN STILL TAKE, decremented as we submit. pick_gpu.sh reports the
    # scheduler's view at ONE instant, and a job submitted a second ago is not in it -- so
    # round-robining its list still over-commits: two jobs to one partition, the second parked
    # on AssocGrpCpuLimit. Reading the free counts once and spending them here is the only way
    # to know what is left without asking the scheduler between every sbatch.
    #
    # hyakalloc PRINTS ACCOUNT AND PARTITION ON THE `TOTAL` ROW ONLY. The `USED` and `FREE` rows
    # that follow leave both columns blank:
    #
    #   | weirdlab | gpu-a40 | 52 | 1007G | 8 | TOTAL |
    #   |          |         | 39 |  368G | 6 | USED  |
    #   |          |         | 13 |  639G | 2 | FREE  |
    #
    # so requiring a non-empty partition on the FREE row -- as this did until 2026-09-17 --
    # skipped EVERY row and left the table empty. Each lookup then fell through to its `:-0`
    # default, next_target found no partition with room, and all ten arms went to TARGETS[0] via
    # the "every candidate partition is spent" path. That is not a hypothetical: it is how the
    # 2026-09-17 wave put five jobs on one partition and parked sac_image on AssocGrpCpuLimit.
    # Carry the labels forward across the triplet instead, and spend MEMORY as well -- a
    # partition can have GPUs and CPUs to spare and still not fit two 32G jobs.
    _acct="" _part=""
    while IFS='|' read -r _ c1 c2 cpus mem gpus tag; do
        c1="${c1// /}"; c2="${c2// /}"; tag="${tag// /}"
        [ -n "$c1" ] && _acct="$c1"
        [ -n "$c2" ] && _part="$c2"
        [ "$tag" = "FREE" ] || continue
        [ -n "$_part" ] || continue
        FREE_GPUS["$_acct/$_part"]=$((gpus + 0))
        FREE_CPUS["$_acct/$_part"]=$((cpus + 0))
        FREE_MEM_G["$_acct/$_part"]=$(( ${mem//[!0-9]/} + 0 ))
    done < <(hyakalloc | sed 's/│/|/g')

    # LOUD, not silent. An empty table is indistinguishable from "everything is busy" at the
    # call site, and the difference is a parse bug versus a real cluster state.
    [ ${#FREE_GPUS[@]} -eq 0 ] && {
        echo "capacity table is EMPTY -- hyakalloc's format changed; fix the parse above" >&2
        echo "                          (refusing to submit: every arm would pile onto one partition)" >&2
        exit 1
    }
    echo "[INFO] ${#TARGETS[@]} candidate partition(s): ${TARGETS[*]}"
    for k in "${!FREE_GPUS[@]}"; do
        echo "[INFO]   $k free: ${FREE_GPUS[$k]} gpu / ${FREE_CPUS[$k]} cpu / ${FREE_MEM_G[$k]}G"
    done
fi
NEXT=0

# Sets TARGET to the next partition with room for one more job; returns 1 if all are spent.
#
# SETS A GLOBAL rather than echoing, because `t="$(next_target)"` would run this in a SUBSHELL
# and every decrement below would be discarded -- the function would hand out the same
# partition forever, which is exactly the bug it exists to fix. Caught by the self-test at the
# bottom of this file, not by reading it.
next_target() {
    local i key k
    for ((i = 0; i < ${#TARGETS[@]}; i++)); do
        key="${TARGETS[$(((NEXT + i) % ${#TARGETS[@]}))]}"
        k="${key// //}"
        # ALL THREE, because sbatch requires all three. Dropping the memory test was how a
        # partition with spare GPUs and CPUs could still be handed a job it could not fit.
        if [ "${FREE_GPUS[$k]:-0}" -ge 1 ] \
        && [ "${FREE_CPUS[$k]:-0}" -ge "$CPUS" ] \
        && [ "${FREE_MEM_G[$k]:-0}" -ge "$MEM_G" ]; then
            NEXT=$(((NEXT + i + 1) % ${#TARGETS[@]}))
            FREE_GPUS[$k]=$((FREE_GPUS[$k] - 1))
            FREE_CPUS[$k]=$((FREE_CPUS[$k] - CPUS))
            FREE_MEM_G[$k]=$((FREE_MEM_G[$k] - MEM_G))
            TARGET="$key"
            return 0
        fi
    done
    return 1
}
while IFS='|' read -r label entry extra; do
    label="$(echo "$label" | xargs)"; entry="$(echo "$entry" | xargs)"; extra="$(echo "$extra" | xargs)"
    [ -z "$label" ] && continue
    if [ -n "$WANT" ] && ! grep -qw "$label" <<<"$WANT"; then continue; fi

    out="$LOGS/$label"
    extra="${extra//BC\//$LOGS/}"                    # --bc-init resolves against this LOGS root

    # $SPLIT BEFORE $extra on both demonstration-reading entries. argparse is last-wins, so with
    # the default after it an arm naming its own --split-file was accepted, printed in the dry
    # run, and then silently trained on the wrong manifest. This way $SPLIT is the default and a
    # per-arm manifest overrides it, which is what the printed command says happened.
    case "$entry" in
        *recurrent_ppo.bc)
            # the BC stage takes no --reward/--video-freq; it trains on demonstrations
            cmd="python $entry --split-file $SPLIT $extra --seed $SEED --out $out"
            ;;
        *sac/runner.py)
            # SAC's action is an absolute chunk, so --reward delta does not apply to it
            cmd="python $entry --split-file $SPLIT $extra --seed $SEED --total-timesteps $STEPS --log-dir $out"
            ;;
        *)
            cmd="python $entry $extra $COMMON --total-timesteps $STEPS --log-dir $out"
            SCORE="$SCORE$out"$'\n'
            ;;
    esac

    if [ "${SUBMIT:-0}" = "1" ]; then
        if ! next_target; then
            echo "[QUEUE $label] every candidate partition is spent; submitting to the first"
            echo "               anyway -- it will pend until one frees"
            TARGET="${TARGETS[0]}"
        fi
        read A P <<< "$TARGET"
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


# ------------------------------------------------------------------ self-test
# SELFTEST=1 bash scripts/slurm/train_value_arms.sh
# Spends a fake two-partition budget and asserts the allocator stops when it is gone. This
# caught the subshell bug above, where the decrements were discarded and every job was handed
# the same partition -- which looks identical to working until the jobs pile up and pend.
if [ "${SELFTEST:-0}" = "1" ]; then
    fail() { echo "FAIL: $1" >&2; exit 1; }

    # (1) GPUs are the binding resource. 3 GPUs of capacity must serve 3 jobs, not 4.
    TARGETS=("robotics gpu-x" "weirdlab gpu-y")
    declare -A FREE_GPUS=(  ["robotics/gpu-x"]=2  ["weirdlab/gpu-y"]=1 )
    declare -A FREE_CPUS=(  ["robotics/gpu-x"]=16 ["weirdlab/gpu-y"]=8 )
    declare -A FREE_MEM_G=( ["robotics/gpu-x"]=64 ["weirdlab/gpu-y"]=32 )
    CPUS=8; MEM_G=32; NEXT=0; got=()
    for n in 1 2 3 4; do
        if next_target; then got+=("$TARGET"); else got+=("EXHAUSTED"); fi
    done
    printf '  gpu-bound: %s\n' "${got[@]}"
    [ "${got[3]}" = "EXHAUSTED" ] || fail "3 GPUs of capacity served 4 jobs"
    [ "${got[0]}" = "${got[1]}" ] && fail "did not alternate partitions"

    # (2) MEMORY is binding on its own. Plenty of GPUs and CPUs, room for one 32G job.
    # Without the FREE_MEM_G test this hands out four and three of them never start.
    TARGETS=("robotics gpu-x")
    FREE_GPUS=(  ["robotics/gpu-x"]=4 ); FREE_CPUS=( ["robotics/gpu-x"]=64 )
    FREE_MEM_G=( ["robotics/gpu-x"]=40 )
    NEXT=0; got=()
    for n in 1 2; do
        if next_target; then got+=("$TARGET"); else got+=("EXHAUSTED"); fi
    done
    printf '  mem-bound: %s\n' "${got[@]}"
    [ "${got[1]}" = "EXHAUSTED" ] || fail "40G of capacity served two 32G jobs"

    # (3) THE PARSE, against real hyakalloc output. This is the test that would have caught the
    # 2026-09-17 wave: the table came back empty, so every arm looked unplaceable and all five
    # went to TARGETS[0]. Asserting "not empty" is the whole point -- an empty table and a busy
    # cluster are indistinguishable downstream.
    unset FREE_GPUS FREE_CPUS FREE_MEM_G
    declare -A FREE_GPUS FREE_CPUS FREE_MEM_G
    _acct="" _part=""
    while IFS='|' read -r _ c1 c2 cpus mem gpus tag; do
        c1="${c1// /}"; c2="${c2// /}"; tag="${tag// /}"
        [ -n "$c1" ] && _acct="$c1"
        [ -n "$c2" ] && _part="$c2"
        [ "$tag" = "FREE" ] || continue
        [ -n "$_part" ] || continue
        FREE_GPUS["$_acct/$_part"]=$((gpus + 0))
        FREE_CPUS["$_acct/$_part"]=$((cpus + 0))
        FREE_MEM_G["$_acct/$_part"]=$(( ${mem//[!0-9]/} + 0 ))
    done < <(hyakalloc | sed 's/│/|/g')
    echo "  parsed ${#FREE_GPUS[@]} partition(s) from live hyakalloc"
    [ ${#FREE_GPUS[@]} -gt 0 ] || fail "parsed NOTHING from hyakalloc -- the format moved again"
    for k in "${!FREE_GPUS[@]}"; do
        [[ "$k" == */* && "$k" != /* && "$k" != */ ]] || fail "malformed key '$k'"
        echo "    $k -> ${FREE_GPUS[$k]} gpu / ${FREE_CPUS[$k]} cpu / ${FREE_MEM_G[$k]}G"
    done

    echo "self-test OK"
    exit 0
fi