#!/usr/bin/env bash
# NAMING: the on-disk arm key `bc` is HISTORICAL and means the search transformer at
# k=1 (max_actions 1), NOT the UNet. It is kept because the eval outputs under
# bon_grid_30demo/bc/ and the run dir offline/bc_demos-30_seed-42 already exist under
# it; renaming the key would orphan them. Everything a human reads says ST k=1.
# The UNet baseline is the separate `unetbc` arm.
# Best-of-n curves for the 30-demo pair across all three selection rules.
#
# Both arms are evaluated at the SAME n grid (1,2,4,8,16) so the search-trained
# k=16 policy and the k=1 policy are compared at matched eval-time search budget --
# the in-training rollouts could not do this (k=16 ran at n=16, k=1 at n=1).
set -u
cd "$(dirname "$0")/.."
export DP_OUTPUT_ROOT="${DP_OUTPUT_ROOT:-/home/harine/diffusion_policy_outputs}"
PY=/home/harine/miniconda3/envs/robodiff2/bin/python
ROOT=$DP_OUTPUT_ROOT/pusht_search/pusht_image_search
S=$ROOT/outer_inner/value_k16_corrupt-False_demos-30_seed-42/checkpoints/step_0060000.ckpt
B=$ROOT/offline/bc_demos-30_seed-42/checkpoints/step_0080000.ckpt

for spec in "search:$S" "bc:$B"; do
  arm="${spec%%:*}"; ckpt="${spec#*:}"
  for sel in argmax softmax final_pass; do
    echo "=== [$(date -Is)] $arm / $sel ==="
    $PY -u eval_search_pusht.py -c "$ckpt" \
      -o "$ROOT/bon_30demo/$arm" --selection "$sel" \
      --min-n 1 --max-n 16 --n-envs 16 2>&1 \
      | grep -E "success_rate=|Error|Traceback"
    echo "    rc=$?"
  done
done
echo "=== [$(date -Is)] sweep done ==="
