#!/usr/bin/env bash
# Extend the 30-demo grid to n in {32, 64}, argmax only, queued behind the n<=16 grid.
#
# Run as a SEPARATE SLICE (--min-n 32 --max-n 64) rather than by re-running the whole
# sweep at --max-n 64: eval_search_pusht merges slices into the same success_curves.jsonl,
# so the n<=16 cells already measured are not recomputed. Cost is linear in n per level,
# so {32,64} is 96 generations per policy call against 31 for {1,2,4,8,16} -- 3.1x the
# original sweep even though it adds only two columns.
#
# argmax only. It wins ~75% of cells at n>=4 and is the rule a headline would quote;
# extending softmax and final_pass too would triple this to ~21h to fill in rules already
# established as weaker.
#
# READ THE n>16 COLUMNS WITH CARE. The search arm trained at K=16, so above that
# predict_n_actions switches to the rolling window: every candidate past the 16th
# conditions on the last 15 rather than the true prefix. Context LENGTH stays in
# distribution, content does not. BC (k=1) has no context at all, so best-of-64 is exactly
# what it was trained for. These columns therefore test extrapolation for the search arm
# and nothing unusual for BC -- they are not a like-for-like comparison the way n<=16 is.
set -u
cd "$(dirname "$0")/.."
export DP_OUTPUT_ROOT="${DP_OUTPUT_ROOT:-/home/harine/diffusion_policy_outputs}"
PY=/home/harine/miniconda3/envs/robodiff2/bin/python
ROOT=$DP_OUTPUT_ROOT/pusht_search/pusht_image_search
GRID_LOG=logs/bon_grid_30demo.log

if [ -f "$GRID_LOG" ] && ! grep -q 'grid done' "$GRID_LOG"; then
  echo "=== [$(date -Is)] waiting for the n<=16 grid to finish ==="
  while ! grep -q 'grid done' "$GRID_LOG"; do
    if ! ps -eo args --no-headers | awk '/bon_grid_30demo\.sh/ && !/awk/' | grep -q .; then
      echo "grid launcher gone without writing 'grid done' -- refusing to start" >&2
      exit 1
    fi
    sleep 60
  done
fi

declare -A ARMS=(
  [search]=$ROOT/outer_inner/value_k16_corrupt-False_demos-30_seed-42
  [bc]=$ROOT/offline/bc_demos-30_seed-42
)
for arm in search bc; do
  d=${ARMS[$arm]}
  for ckpt in $(ls $d/checkpoints/step_*.ckpt | sort); do
    step=$(basename $ckpt .ckpt)
    echo "=== [$(date -Is)] $arm $step argmax ==="
    # 45m cap: at n=64 the async verifier pool has deadlocked (0 CPU, blocked on a unix
    # socket) at least once, wedging the whole chain behind one checkpoint. A timeout skips
    # the stuck checkpoint instead; eval_search_pusht resumes per (checkpoint, n_list), so
    # a skipped one is picked up by re-running this script.
    timeout 45m $PY -u eval_search_pusht.py -c "$ckpt" \
      -o "$ROOT/bon_grid_30demo/$arm" --selection argmax \
      --min-n 32 --max-n 64 --n-envs 16 --store-scores 2>&1 \
      | grep -E "success_rate=|\[scores\]|Traceback|Error"
    rc=${PIPESTATUS[0]}
    [ "$rc" = 124 ] && echo "=== [$(date -Is)] TIMEOUT after 45m: $arm $step -- skipped ==="
    pkill -f 'pusht_verifier' 2>/dev/null   # a timed-out eval leaves its worker pool behind
  done
done
echo "=== [$(date -Is)] n64 done ==="
