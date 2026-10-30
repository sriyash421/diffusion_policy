#!/bin/bash
# Submit one seeded step-2 eval arm (70 seeds: 50 held-out-position test + 20 train) through
# eval_pi0.sbatch. Plans/objects come from the shared Gemini cache so every arm sees the same
# ones.
#
#   bash scripts/simpler/veritas_pi0/submit_step2_eval.sh raw                      # pi0, N=1
#   bash scripts/simpler/veritas_pi0/submit_step2_eval.sh base_bon                 # official verifier, N=5
#   bash scripts/simpler/veritas_pi0/submit_step2_eval.sh ctx_seq  <ckpt.pt> [interp_lambda]
#   bash scripts/simpler/veritas_pi0/submit_step2_eval.sh ctx_tworound <ckpt.pt> [interp_lambda]
#   bash scripts/simpler/veritas_pi0/submit_step2_eval.sh ckpt_bon <ckpt.pt>
#   N=16 bash scripts/simpler/veritas_pi0/submit_step2_eval.sh ctx_sliding <ckpt.pt>   # all N sequential
#   N=16 bash scripts/simpler/veritas_pi0/submit_step2_eval.sh ctx_sliding            # untrained context model
# ARM (run name) defaults to <mode>[_<run>_<step>][_interpL]; N defaults to 5; SUBMIT=1 submits.
set -euo pipefail
cd "$(dirname "$0")/../../.."
MODE="${1:?mode}"; CKPT="${2:-}"; INTERP="${3:-0}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"
SHORT="${TASK#widowx_}"; case "$SHORT" in put_eggplant_in_basket) SHORT=eggplant;; stack_cube) SHORT=stack;; carrot_on_plate) SHORT=carrot;; spoon_on_towel) SHORT=spoon;; esac
S2=/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2/$SHORT
OUT_ROOT="${OUT_ROOT:-$S2/eval}"
N="${N:-5}"
SEEDS="${SEEDS:-$S2/eval_seeds_50.json}"
export GEMINI_CACHE_DIR="${GEMINI_CACHE_DIR:-/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2/gemini_cache}"

tag=""
if [ -n "$CKPT" ]; then
  run=$(basename "$(dirname "$CKPT")"); step=$(basename "$CKPT" .pt | sed 's/ckpt_//')
  tag="_${run}_${step}"
fi
[ "$INTERP" != 0 ] && tag="${tag}_interp${INTERP}"
[ -z "$CKPT" ] && case "$MODE" in ctx_*|ckpt_bon|mix_bon) tag="_untrained";; esac   # context model at init
case "$MODE" in
  raw)      VERIFIER=0; N=1; ARM="${ARM:-raw_n1_seeded}"; SCRIPT=run_pi0_eval.py; EXTRA="" ;;
  collect)  # pi0 + verifier rollouts with every decision stored; ATTEMPTS (3) retries per seed
    VERIFIER=1; ARM="${ARM:-collect_n${N}}"; SCRIPT=eval_pi0_context.py
    SEEDS="${COLLECT_SEEDS:-$S2/collect_seeds_300.json}"; OUT_ROOT="$S2/collect_runs"
    EXTRA="+ctx_mode=base_bon +collect_decisions=$S2/collect +attempts=${ATTEMPTS:-3}" ;;
  base_bon) VERIFIER=1; ARM="${ARM:-veritas_n${N}_seeded}"; SCRIPT=run_pi0_eval.py; EXTRA="" ;;
  ctx_seq|ctx_tworound|ctx_sliding|ckpt_bon|mix_bon)
    VERIFIER=1; ARM="${ARM:-${MODE}_n${N}${tag}}"; SCRIPT=eval_pi0_context.py
    CTX_MODE=$MODE; FORCE=""
    [ "$MODE" = mix_bon ] && { CTX_MODE=ctx_seq; FORCE="+force_n_ctx=0"; }   # 4 base + 1 ft, no context
    EXTRA="+ctx_mode=$CTX_MODE +interp_lambda=$INTERP $FORCE ${VALUE_STATS:+ +value_stats=$VALUE_STATS}"
    [ -n "$CKPT" ] && EXTRA="$EXTRA +ctx_ckpt=$(realpath "$CKPT")" ;;
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac

if [ "${PARTITION:-}" = ckpt ]; then
  SLURM_OPTS=(--account=ckpt-robotics --partition=ckpt,ckpt-all --qos=ckpt-gpu --requeue)
elif [ -n "${PARTITION:-}" ]; then   # a named lab partition, e.g. PARTITION=gpu-l40
  SLURM_OPTS=(--account="${ACCOUNT:-weirdlab}" --partition="$PARTITION" --qos=normal --no-requeue)
else
  read -r A P < <(NEED_CPUS=6 NEED_MEM_G=48 NEED_GPUS=1 bash scripts/slurm/pick_gpu.sh 2>/dev/null || echo "ckpt-robotics ckpt")
  if [ "$P" = ckpt ]; then SLURM_OPTS=(--account=ckpt-robotics --partition=ckpt,ckpt-all --qos=ckpt-gpu --requeue)
  else SLURM_OPTS=(--account="$A" --partition="$P" --qos=normal --no-requeue); fi
fi
cmd=(sbatch "${SLURM_OPTS[@]}" --job-name="pi0_ev_${ARM}" --time="${TIME:-6:00:00}"
     --export=ALL,OUT_DIR="$OUT_ROOT/$ARM",TASK="$TASK",VERIFIER="$VERIFIER",N="$N",ARM="$ARM",EVAL_SEEDS="$SEEDS",EVAL_SCRIPT="$SCRIPT",COLLECT_DEMOS=0,EXTRA="$EXTRA"
     scripts/simpler/veritas_pi0/eval_pi0.sbatch)
echo "${cmd[*]}"
[ "${SUBMIT:-0}" = 1 ] && "${cmd[@]}" || echo "(dry run; SUBMIT=1 to submit)"
