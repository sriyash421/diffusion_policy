#!/usr/bin/env bash
# 30-demo search-vs-BC pair, sequential: search first (it is the slower arm and gets a
# clean GPU), then BC. Both are the same policy class -- PushTDiffusionSearchPolicy -- so
# the only difference is search width 16 vs 1.
#
#   nohup bash scripts/run_30demo_2x2.sh > logs/run_30demo_2x2.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export DP_OUTPUT_ROOT="${DP_OUTPUT_ROOT:-/home/harine/diffusion_policy_outputs}"
# Interpreter, overridable: the paths below are per-machine, and a wrong one fails here
# with a clear message rather than 100k steps later. On Hyak:
#   PY=/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python \
#   DP_OUTPUT_ROOT=/gscratch/robotics/harine/diffusion_policy_outputs bash $0
PY="${PY:-/home/harine/miniconda3/envs/robodiff2/bin/python}"
if [ ! -x "$PY" ]; then
  echo "no interpreter at $PY -- set PY=/path/to/env/bin/python" >&2
  exit 1
fi
mkdir -p logs

# rollout_every_steps 20000 is inherited, so success rates land at 20k/40k/60k/80k/100k.
COMMON="n_demos=30 training.max_gradient_steps=100000"

echo "=== [$(date -Is)] SEARCH (n_candidates=16) ==="
$PY -u train.py --config-name=train_pusht_diffusion_search $COMMON
search_rc=$?
echo "=== [$(date -Is)] search exited rc=$search_rc ==="

if [ $search_rc -ne 0 ]; then
  echo "search failed; NOT starting BC so the GPU stays free for diagnosis" >&2
  exit $search_rc
fi

echo "=== [$(date -Is)] BC (max_actions=1) ==="
$PY -u train.py --config-name=train_pusht_bc $COMMON
echo "=== [$(date -Is)] BC exited rc=$? ==="
