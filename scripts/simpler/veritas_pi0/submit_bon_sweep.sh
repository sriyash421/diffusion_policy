#!/bin/bash
# Best-of-N sweep of the Veritas verifier over both pi0 Bridge policies, all 4 tasks, on the
# 50 eval seeds (<task>/eval_seeds_50.json), one eval_pi0.sbatch arm per (task, policy, N).
# N=1 is raw pi0 (verifier off). Plans come from the shared Gemini cache, so every arm of a
# task sees the same plan per seed (PLAN_PROMPT picks the prompt; it is part of the cache key).
#
#   PLAN_PROMPT=gemini_written SUBMIT=1 bash scripts/simpler/veritas_pi0/submit_bon_sweep.sh
# Env: PLAN_PROMPT (official), NS ("1 2 4 8 16 64"), POLICIES ("pi0 beta"), TASKS, SWEEP (dir
# under <task>/eval, default bon_<PLAN_PROMPT>), SUBMIT=1 to submit (else dry run).
set -euo pipefail
cd "$(dirname "$0")/../../.."
S=/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2
export PLAN_PROMPT="${PLAN_PROMPT:-official}"
export GEMINI_CACHE_DIR="${GEMINI_CACHE_DIR:-$S/gemini_cache}"
SWEEP="${SWEEP:-bon_$PLAN_PROMPT}"
declare -A ENV=([eggplant]=widowx_put_eggplant_in_basket [stack]=widowx_stack_cube [carrot]=widowx_carrot_on_plate [spoon]=widowx_spoon_on_towel)
declare -A CKPTS=([pi0]=models/bridge_uniform_step19296_2024-12-26_22-31_42.pt [beta]=models/bridge_beta_step19296_2024-12-26_22-30_42.pt)
declare -A BF16=([pi0]=False [beta]=True)
for t in ${TASKS:-eggplant stack carrot spoon}; do for p in ${POLICIES:-pi0 beta}; do for n in ${NS:-1 2 4 8 16 64}; do
  if [ "$n" = 1 ]; then VERIFIER=0; ARM="${p}_raw_n1"; else VERIFIER=1; ARM="${p}_veritas_n$n"; fi
  OUT="$S/$t/eval/$SWEEP/$ARM"
  if [ -f "$OUT/episodes.jsonl" ] && [ "$(wc -l < "$OUT/episodes.jsonl")" -ge 50 ]; then echo "done: $t/$ARM"; continue; fi
  cmd=(sbatch --account=ckpt-robotics --partition=ckpt,ckpt-all --qos=ckpt-gpu --requeue --time=4:00:00
       --job-name="pi0_ev_${SWEEP}_${t}_${ARM}"
       --export=ALL,OUT_DIR="$OUT",TASK="${ENV[$t]}",VERIFIER="$VERIFIER",N="$n",ARM="${SWEEP}_$ARM",EVAL_SEEDS="$S/$t/eval_seeds_50.json",CKPT="${CKPTS[$p]}",USE_BF16="${BF16[$p]}",COLLECT_DEMOS=0,PLAN_PROMPT="$PLAN_PROMPT",GEMINI_CACHE_DIR="$GEMINI_CACHE_DIR"
       scripts/simpler/veritas_pi0/eval_pi0.sbatch)
  if [ "${SUBMIT:-0}" = 1 ]; then "${cmd[@]}"; else echo "${cmd[*]}"; fi
done; done; done
