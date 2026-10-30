#!/bin/bash
# Diffusion-policy BC + Veritas verifier best-of-N on the step-2 50 eval seeds of one task
# (the "BC on the offline demos" curve of the N sweep), through the existing
# scripts/simpler/eval_search/eval_veritas_bon.sbatch.
#
#   bash scripts/simpler/veritas_pi0/submit_dp_sweep.sh <eggplant|stack|carrot> <step_XXXX.ckpt> [SUBMIT=1]
#
# Settings follow the position-holdout campaign's DP evals (H=16 UNet executing 8 steps, ddim,
# EMA weights, scored horizon 8, plan-done cost 0) with three choices for parity with the pi0
# arms: plans from gemini-3.8-flash with Grounded-SAM-2 object hints, the task's native step
# limit (120 eggplant, 60 otherwise), and ONE plan per seed shared by every N: stage 1 (N=1)
# plans live into <task>/dp_plans, stage 2 (N=2..64) starts after it and reads that cache.
# Cells go to <task>/dp_eval/<run>_<step>/n<N>; eval_search resumes per episode.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
SHORT="${1:?eggplant|stack|carrot}"; CKPT="$(realpath "${2:?step ckpt}")"
case "$SHORT" in
  eggplant) TASK=widowx_put_eggplant_in_basket; MAX_STEPS=120
            SPLIT=/gscratch/robotics/harine/RoboMonkey_data/experiments/position_holdout_20261009/splits/eggplant/position_holdout_train100.json ;;
  stack)    TASK=widowx_stack_cube; MAX_STEPS=60
            SPLIT=/gscratch/robotics/harine/RoboMonkey_data/experiments/position_holdout_20261009/splits/stack/position_holdout_train100.json ;;
  carrot)   TASK=widowx_carrot_on_plate; MAX_STEPS=60
            SPLIT=$REPO/diffusion_policy/config/splits/simpler/carrot/random_train100.json ;;
  *) echo "unknown task $SHORT" >&2; exit 1 ;;
esac
S2=/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2/$SHORT
SEEDS="$(python3 -c "import json,sys; print(','.join(str(s) for v in json.load(open(sys.argv[1])).values() for s in v))" "$S2/eval_seeds_50.json")"
run="$(basename "$(dirname "$(dirname "$CKPT")")")"; step="$(basename "$CKPT" .ckpt)"
OUT_ROOT="$S2/dp_eval/${run}_${step}"; PLAN_DIR="$S2/dp_plans"
mkdir -p "$OUT_ROOT" "$PLAN_DIR"
NS="${NS:-1 2 4 8 16 32 64}"
COMMON=(--account=ckpt-robotics --partition=ckpt,ckpt-all --qos=ckpt-gpu --requeue --constraint=a40|l40s|l40|a100|h200
        --output=/gscratch/robotics/harine/slurm_logs/pi0_veritas/%x-%j.out --error=/gscratch/robotics/harine/slurm_logs/pi0_veritas/%x-%j.out)
ENV="CKPT=$CKPT,SPLIT_FILE=$SPLIT,PLAN_DIR=$PLAN_DIR,CAMERA_FILE=$REPO/diffusion_policy/config/veritas/camera_${TASK}.json,TASK=$TASK"
ENV="$ENV,MAX_STEPS=$MAX_STEPS,N_ACTION_STEPS=8,BC_SAMPLER=ddim,USE_EMA=1,MODE=argmax,ACTION_POS_SCALE=null"
export SEEDS   # --export splits at commas: the seed list goes through the environment (ALL)
dep=""
for n in $NS; do
  out="$OUT_ROOT/n$n"
  if [ -f "$out/eval_log.json" ]; then echo "done: $out"; continue; fi
  hrs=$(( n >= 32 ? 12 : (n >= 8 ? 8 : 4) ))
  if [ "$n" = 1 ]; then live="VERITAS_ALLOW_LIVE=1,VERITAS_HINTS=gsam2"; else live="VERITAS_ALLOW_LIVE=0"; fi
  cmd=(sbatch "${COMMON[@]}" --job-name="dp_bon_${SHORT}_${step}_n${n}" --time="${hrs}:00:00" ${dep:+--dependency=afterok:$dep}
       --export="ALL,N=$n,OUT_DIR=$out,$live,$ENV" "$REPO/scripts/simpler/eval_search/eval_veritas_bon.sbatch")
  echo "${cmd[*]}" | cut -c1-200
  if [ "${SUBMIT:-0}" = 1 ]; then
    jid="$("${cmd[@]}" | grep -oE '[0-9]+$')"; echo "  -> $jid"
    [ "$n" = 1 ] && dep="$jid"
  fi
done
