#!/usr/bin/env bash
# Full checkpoint x selection x n grid for the 30-demo pair.
#
# Every step_*.ckpt (10k..100k) x {argmax,softmax,final_pass} x n in {1,2,4,8,16},
# both arms. Removes the need to pre-select a checkpoint: selection can then be done
# on val and read off test, rather than by maximising test.
set -u
cd "$(dirname "$0")/.."
export DP_OUTPUT_ROOT="${DP_OUTPUT_ROOT:-/home/harine/diffusion_policy_outputs}"
PY=/home/harine/miniconda3/envs/robodiff2/bin/python
ROOT=$DP_OUTPUT_ROOT/pusht_search/pusht_image_search
declare -A ARMS=(
  [search]=$ROOT/outer_inner/value_k16_corrupt-False_demos-30_seed-42
  [bc]=$ROOT/offline/bc_demos-30_seed-42
)
for arm in search bc; do
  d=${ARMS[$arm]}
  for ckpt in $(ls $d/checkpoints/step_*.ckpt | sort); do
    step=$(basename $ckpt .ckpt)
    for sel in argmax softmax final_pass; do
      echo "=== [$(date -Is)] $arm $step $sel ==="
      $PY -u eval_search_pusht.py -c "$ckpt" \
        -o "$ROOT/bon_grid_30demo/$arm" --selection "$sel" \
        --min-n 1 --max-n 16 --n-envs 16 --store-scores 2>&1 \
        | grep -E "success_rate=|\[scores\]|Traceback|Error"
    done
  done
done
echo "=== [$(date -Is)] grid done ==="
