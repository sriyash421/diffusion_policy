#!/usr/bin/env bash
# 30-demo ST k16-vs-k1 pair, sequential: k16 first (it is the slower arm and gets a clean
# GPU), then k1. ONE config, one override apart: `n_candidates` is the only difference, and
# `n_search_actions` and the k in run_name all derive from it. "BC" is NOT this pair --
# BC means the UNet (train_pusht_unet_bc, scripts/run_unetbc_30demo.sh).
#
#   nohup bash scripts/run_30demo_2x2.sh > logs/run_30demo_2x2.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export DP_OUTPUT_ROOT="${DP_OUTPUT_ROOT:-/home/harine/diffusion_policy_outputs}"
PY=/home/harine/miniconda3/envs/robodiff2/bin/python
mkdir -p logs

# rollout_every_steps 20000 is inherited, so success rates land at 20k/40k/60k/80k/100k.
# n_demos defaults to 30 in pusht_base.yaml; stated here so the log records it.
COMMON="n_demos=30 training.max_gradient_steps=100000"

echo "=== [$(date -Is)] ST k=16 ==="
$PY -u train.py --config-name=train_pusht_diffusion_search $COMMON n_candidates=16
search_rc=$?
echo "=== [$(date -Is)] search exited rc=$search_rc ==="

if [ $search_rc -ne 0 ]; then
  echo "k16 failed; NOT starting k1 so the GPU stays free for diagnosis" >&2
  exit $search_rc
fi

echo "=== [$(date -Is)] ST k=1 ==="
$PY -u train.py --config-name=train_pusht_diffusion_search $COMMON n_candidates=1
echo "=== [$(date -Is)] ST k=1 exited rc=$? ==="
