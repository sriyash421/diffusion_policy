#!/usr/bin/env bash
# ST at 6/8/1024 (137.8M), k=1 then k=16, 30 demos, then eval every 10k checkpoint.
# Queued behind the UNet BC run.
#
# NOT param-matched to the UNet BC (137.8M vs 293.4M); this width was chosen to see how
# the search arms behave at ~8x their current size before committing to a bigger one.
#
# run_name is overridden explicitly. Without it the width overrides resolve to the SAME
# run_name as the 4/4/256 runs, so hydra.run.dir would collide and these would overwrite
# the checkpoints the entire best-of-n grid is built on.
set -u
cd "$(dirname "$0")/.."
export DP_OUTPUT_ROOT="${DP_OUTPUT_ROOT:-/home/harine/diffusion_policy_outputs}"
PY=/home/harine/miniconda3/envs/robodiff2/bin/python
ROOT=$DP_OUTPUT_ROOT/pusht_search/pusht_image_search
UNET_LOG=logs/run_unetbc_30demo.log
BIG="policy.n_layer=6 policy.n_head=8 policy.n_emb=1024"
mkdir -p logs

if [ -f "$UNET_LOG" ] && ! grep -q 'unetbc done' "$UNET_LOG"; then
  echo "=== [$(date -Is)] waiting for the UNet BC stage to finish ==="
  while ! grep -q 'unetbc done' "$UNET_LOG"; do
    if ! ps -eo args --no-headers | awk '/run_unetbc_30demo\.sh/ && !/awk/' | grep -q .; then
      echo "unetbc launcher gone without writing 'unetbc done' -- refusing to start" >&2; exit 1
    fi
    sleep 60
  done
fi

# k=1 first: it is the cheap one (no candidate generation, offline trainer), so a result
# lands before the k=16 run's much longer wall clock.
run_one () {   # $1 config  $2 run_name  $3 out-subdir  $4 trainer-dir
  echo "=== [$(date -Is)] TRAIN $1 ($2) ==="
  $PY -u train.py --config-name="$1" n_demos=30 training.max_gradient_steps=100000 \
      $BIG "run_name=$2"
  rc=$?; echo "=== [$(date -Is)] $1 train exited rc=$rc ==="
  [ $rc -ne 0 ] && return $rc
  for ckpt in $(ls "$ROOT/$4/$2/checkpoints/step_"*.ckpt | sort); do
    step=$(basename "$ckpt" .ckpt)
    echo "=== [$(date -Is)] $3 $step argmax ==="
    timeout 45m $PY -u eval_search_pusht.py -c "$ckpt" \
      -o "$ROOT/bon_grid_30demo/$3" --min-n 1 --max-n 16 --n-envs 16 --seed 42 \
      --store-scores 2>&1 | grep -E "success_rate=|Traceback|Error"
    [ "${PIPESTATUS[0]}" = 124 ] && echo "=== [$(date -Is)] TIMEOUT: $3 $step -- skipped ==="
  done
}

run_one train_pusht_bc               bc_arch-6x8x1024_demos-30_seed-42               st-k1-big  offline
run_one train_pusht_diffusion_search value_k16_arch-6x8x1024_corrupt-False_demos-30_seed-42 st-k16-big outer_inner
echo "=== [$(date -Is)] st big done ==="
