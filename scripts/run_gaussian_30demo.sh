#!/usr/bin/env bash
# Train the PushT GAUSSIAN search arm at 30 demos, queued behind the best-of-n grid.
#
# The grid saturates the CPU with 32-worker verifier pools and holds the GPU; this arm
# needs both (its training loop generates max_actions-1 candidates per step and simulates
# each). Running them concurrently would slow both and make the grid's per-combo timings
# incomparable to the ones already recorded, so this waits for the grid's own done marker
# rather than starting immediately.
#
# Third arm of the 30-demo generation: same task, encoder, verifier, search procedure,
# demo budget and optimizer as ST-diffusion-k16 -- the ONLY difference is how a candidate
# is produced (one rsample from a Normal vs an 8-step DDIM loop).
set -u
cd "$(dirname "$0")/.."
export DP_OUTPUT_ROOT="${DP_OUTPUT_ROOT:-/home/harine/diffusion_policy_outputs}"
PY=/home/harine/miniconda3/envs/robodiff2/bin/python
# wait on the n=64 EXTENSION, which itself waits on the n<=16 grid -- so this is
# last in the chain and never contends with either sweep
GRID_LOG=logs/bon_grid_30demo_n64.log
mkdir -p logs

if [ -f "$GRID_LOG" ] && ! grep -q 'n64 done' "$GRID_LOG"; then
  echo "=== [$(date -Is)] waiting for the n=64 grid extension to finish before starting ==="
  # poll the marker, not the process: the launcher can die without writing it, and in that
  # case we should still not start on top of a half-finished sweep someone may resume
  while ! grep -q 'n64 done' "$GRID_LOG"; do
    if ! ps -eo args --no-headers | awk '/bon_grid_30demo_n64\.sh/ && !/awk/' | grep -q .; then
      echo "n64 launcher is gone and 'n64 done' was never written -- refusing to start" >&2
      exit 1
    fi
    sleep 60
  done
  echo "=== [$(date -Is)] n=64 extension finished, starting gaussian ==="
fi

echo "=== [$(date -Is)] GAUSSIAN search (n_candidates=16, 30 demos) ==="
$PY -u train.py --config-name=train_pusht_gaussian_search \
    n_demos=30 training.max_gradient_steps=100000
echo "=== [$(date -Is)] gaussian exited rc=$? ==="
