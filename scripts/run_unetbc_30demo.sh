#!/usr/bin/env bash
# Train the PushT UNet BC baseline at 30 demos, then eval every 10k checkpoint.
# Queued behind the n=64 grid extension so nothing contends for the GPU.
#
# This is a DIFFERENT baseline from ST-diffusion-k1: that one is the search transformer at
# max_actions=1 (same class as k16, one slot). This is the diffusion UNet, a separate
# policy class with no search interface at all -- so it has no verifier and best-of-n is
# undefined for it. Its eval is therefore n=1 only, which eval_search_pusht asserts.
set -u
cd "$(dirname "$0")/.."
export DP_OUTPUT_ROOT="${DP_OUTPUT_ROOT:-/home/harine/diffusion_policy_outputs}"
PY=/home/harine/miniconda3/envs/robodiff2/bin/python
ROOT=$DP_OUTPUT_ROOT/pusht_search/pusht_image_search
N64_LOG=logs/bon_grid_30demo_n64.log
mkdir -p logs

if [ -f "$N64_LOG" ] && ! grep -q 'n64 done' "$N64_LOG"; then
  echo "=== [$(date -Is)] waiting for the n=64 extension to finish ==="
  while ! grep -q 'n64 done' "$N64_LOG"; do
    if ! ps -eo args --no-headers | awk '/bon_grid_30demo_n64\.sh/ && !/awk/' | grep -q .; then
      echo "n64 launcher gone without writing 'n64 done' -- refusing to start" >&2; exit 1
    fi
    sleep 60
  done
fi

echo "=== [$(date -Is)] UNET BC train (30 demos, 100k steps) ==="
$PY -u train.py --config-name=train_pusht_unet_bc \
    n_demos=30 training.max_gradient_steps=100000
rc=$?
echo "=== [$(date -Is)] unet bc train exited rc=$rc ==="
[ $rc -ne 0 ] && exit $rc

# Same deterministic protocol as the grid: fixed split manifest, seed 42, torch+numpy
# re-seeded to the same base before every n (eval_search_pusht.py:311-320), 50 test
# episodes / 30 val. n=1 only -- see the header.
D=$ROOT/offline/unetbc_demos-30_seed-42
for ckpt in $(ls $D/checkpoints/step_*.ckpt | sort); do
  step=$(basename "$ckpt" .ckpt)
  echo "=== [$(date -Is)] unetbc $step argmax ==="
  timeout 45m $PY -u eval_search_pusht.py -c "$ckpt" \
    -o "$ROOT/bon_grid_30demo/unetbc" --min-n 1 --max-n 1 --n-envs 16 --seed 42 2>&1 \
    | grep -E "success_rate=|Traceback|Error"
  [ "${PIPESTATUS[0]}" = 124 ] && echo "=== [$(date -Is)] TIMEOUT: unetbc $step -- skipped ==="
done
echo "=== [$(date -Is)] unetbc done ==="
