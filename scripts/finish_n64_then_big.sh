#!/usr/bin/env bash
# Fill the one combo missing from the n=32,64 grid, then hand off to the big-ST run.
#
# bc/step_0030000/final_pass was lost when a deadlock timed out under the pre-fix launcher,
# whose `if ! cmd; then rc=$?` always read 0 and so never retried. Everything else is done.
set -u
cd "$(dirname "$0")/.."
export DP_OUTPUT_ROOT="${DP_OUTPUT_ROOT:-/home/harine/diffusion_policy_outputs}"
PY=/home/harine/miniconda3/envs/robodiff2/bin/python
ROOT=$DP_OUTPUT_ROOT/pusht_search/pusht_image_search
CKPT=$ROOT/offline/bc_demos-30_seed-42/checkpoints/step_0030000.ckpt

for attempt in 1 2; do
  echo "=== [$(date -Is)] bc step_0030000 final_pass (attempt $attempt) ==="
  timeout 45m $PY -u eval_search_pusht.py -c "$CKPT" \
    -o "$ROOT/bon_grid_30demo/bc" --selection final_pass \
    --min-n 32 --max-n 64 --n-envs 16 --seed 42 --store-scores 2>&1 \
    | grep -E "success_rate=|\[scores\]|Traceback|Error"
  rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] && break
  echo "=== [$(date -Is)] attempt $attempt failed rc=$rc ==="
  pkill -f 'multiprocessing.forkserver' 2>/dev/null
done
echo "=== [$(date -Is)] n64 done ==="
