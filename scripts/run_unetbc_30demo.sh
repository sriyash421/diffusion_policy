#!/usr/bin/env bash
# Train the PushT UNet BC baseline at 30 demos. TRAINING ONLY -- see EVAL below.
# Queued behind the n=64 grid extension so nothing contends for the GPU.
#
# A DIFFERENT baseline from ST-diffusion-k1: that one is the search transformer at
# max_actions=1 (same class as k16, one slot). This is the diffusion UNet -- a separate
# architecture, on the same encoder, optimizer, LR schedule, scheduler (DDIM at 8) and
# split.
#
# Best-of-n here is I.I.D. SAMPLING: PushTUNetSearchPolicy draws n independent samples,
# scores each with the PushT sim verifier, and executes the best. No learned selection is
# involved, which is what makes it the baseline for the search arms' test-time budget.
#
# The config pins `trainer: unet_bc`, so this lands in
# unet_bc/unetbc_demos-30_seed-42 -- NOT offline/, which is where the ST arms live. That
# path is what scripts/slurm/submit_30_100_watchers.sh polls for this arm.
#
# EVAL IS NOT DONE HERE. The watcher owns it -- eval_watch_pusht_search.sbatch sweeps
# n = 1 2 4 8 16 32 64 over the 50 test episodes with --skip-val, and numbers from any
# other protocol are not comparable to it. After training:
#   SUBMIT=1 bash scripts/slurm/submit_30_100_watchers.sh
# (idempotent: it skips arms that already have a live watcher or are fully scored).
set -u
cd "$(dirname "$0")/.."
export DP_OUTPUT_ROOT="${DP_OUTPUT_ROOT:-/home/harine/diffusion_policy_outputs}"
PY=/home/harine/miniconda3/envs/robodiff2/bin/python
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

# `unetbc done` was the sentinel scripts/run_st_big_30demo.sh grepped for to chain itself.
# That launcher was removed with the big-ST arm on 2026-08-29; the sentinel is kept because
# it is also this script's own completion marker
# behind this stage -- keep the string.
echo "=== [$(date -Is)] unetbc done -- now: SUBMIT=1 bash scripts/slurm/submit_30_100_watchers.sh ==="
