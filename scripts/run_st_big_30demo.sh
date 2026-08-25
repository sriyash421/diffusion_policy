#!/usr/bin/env bash
# ST-diffusion-k16 at 6/8/1024 (137.8M), 30 demos, then the full eval grid.
# Queued behind the n=32,64 sweep so nothing contends for the GPU.
#
# NOT param-matched to the UNet BC (137.8M vs 293.4M) -- this width was chosen to see how
# the search arm behaves at ~8x the 4/4/256 model before committing to a wider one.
#
# run_name is overridden and that is load-bearing: the width overrides alone resolve to the
# SAME run_name as the 4/4/256 run, so hydra.run.dir would collide and this would overwrite
# the checkpoints the entire existing grid is built on.
#
# EXPECT THIS TO BE SLOW. Measured fwd+bwd at batch 32: 267.9 ms at 6/8/1024 vs 25.3 ms at
# 4/4/256 -- 10.6x. Gradient steps alone are ~7.4h for 100k, and on the search arm
# candidate generation dominates and scales the same way. The 4/4/256 k16 run took 12h26m,
# so budget 1.5-5 days.
set -u
cd "$(dirname "$0")/.."
export DP_OUTPUT_ROOT="${DP_OUTPUT_ROOT:-/home/harine/diffusion_policy_outputs}"
PY=/home/harine/miniconda3/envs/robodiff2/bin/python
ROOT=$DP_OUTPUT_ROOT/pusht_search/pusht_image_search
N64_LOG=logs/bon_grid_30demo_n64.log
RUN=value_k16_arch-6x8x1024_corrupt-False_demos-30_seed-42
ARM=st-k16-big
mkdir -p logs

if [ -f "$N64_LOG" ] && ! grep -q 'n64 done' "$N64_LOG"; then
  echo "=== [$(date -Is)] waiting for the n=32,64 sweep to finish ==="
  while ! grep -q 'n64 done' "$N64_LOG"; do
    # either the main sweep or the finisher that fills its last cell may be the writer
    if ! ps -eo args --no-headers \
         | awk '/bon_grid_30demo_n64\.sh|finish_n64_then_big\.sh/ && !/awk/' | grep -q .; then
      echo "n64 launcher gone without writing 'n64 done' -- refusing to start" >&2; exit 1
    fi
    sleep 60
  done
fi

echo "=== [$(date -Is)] TRAIN ST-diffusion-k16 6x8x1024 (30 demos, 100k steps) ==="
$PY -u train.py --config-name=train_pusht_diffusion_search \
    n_demos=30 training.max_gradient_steps=100000 \
    policy.n_layer=6 policy.n_head=8 policy.n_emb=1024 "run_name=$RUN"
rc=$?
echo "=== [$(date -Is)] train exited rc=$rc ==="
[ $rc -ne 0 ] && exit $rc

# Fill the SAME grid the 4/4/256 arms occupy: every 10k checkpoint x
# {argmax, softmax, final_pass} x n in {1..64}. Two slices because cost is linear in n per
# level: n<=16 is 31 generations per policy call, n in {32,64} is 96.
for ckpt in $(ls "$ROOT/outer_inner/$RUN/checkpoints/step_"*.ckpt | sort); do
  step=$(basename "$ckpt" .ckpt)
  for sel in argmax softmax final_pass; do
    for slice in "1 16" "32 64"; do
      set -- $slice
      if [ "$1" = 32 ] && $PY scripts/_curve_has_n.py \
           "$ROOT/bon_grid_30demo/$ARM/success_curves.jsonl" "$step" "$sel"; then
        echo "=== [$(date -Is)] $ARM $step $sel -- already has n=32,64, skipping ==="
        continue
      fi
      echo "=== [$(date -Is)] $ARM $step $sel ==="
      timeout 45m $PY -u eval_search_pusht.py -c "$ckpt" \
        -o "$ROOT/bon_grid_30demo/$ARM" --selection "$sel" \
        --min-n "$1" --max-n "$2" --n-envs 16 --seed 42 --store-scores 2>&1 \
        | grep -E "success_rate=|\[scores\]|Traceback|Error"
      if [ "${PIPESTATUS[0]}" = 124 ]; then
        echo "=== [$(date -Is)] TIMEOUT: $ARM $step $sel n=$1..$2 -- skipped ==="
        pkill -f 'multiprocessing.forkserver' 2>/dev/null
      fi
    done
  done
done
echo "=== [$(date -Is)] st big done ==="
