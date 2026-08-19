#!/usr/bin/env bash
# NAMING: the on-disk arm key `bc` is HISTORICAL and means the search transformer at
# k=1 (max_actions 1), NOT the UNet. It is kept because the eval outputs under
# bon_grid_30demo/bc/ and the run dir offline/bc_demos-30_seed-42 already exist under
# it; renaming the key would orphan them. Everything a human reads says ST k=1.
# The UNet baseline is the separate `unetbc` arm.
# Extend every row of the 30-demo grid to n in {32, 64}: both arms x
# {argmax, softmax, final_pass} x all ten checkpoints = 60 combos.
#
# Run as a SEPARATE SLICE (--min-n 32 --max-n 64). eval_search_pusht merges slices into one
# success_curves.jsonl, so the n<=16 cells are not recomputed. Cost is linear in n per
# level: {32,64} is 96 generations per policy call against 31 for {1..16}, so this is ~3x
# the original sweep despite adding only two columns. Budget ~20h.
#
# READ THE n>16 COLUMNS WITH CARE. ST-diffusion-k16 trained at K=16, so above that
# predict_n_actions switches to the rolling window: candidates past the 16th condition on
# the last 15 rather than the true prefix -- context LENGTH stays in distribution, content
# does not. ST-diffusion-k1 has no context at all (max_context_actions 0), so best-of-64 is
# exactly what it was trained for. The two arms are therefore NOT in the same regime here.
#
# Two reliability guards, both earned:
#   * timeout 45m per combo. The async verifier pool deadlocks: main blocks in
#     unix_stream_data_wait for workers that died, and because the pool uses
#     context='forkserver' the forkserver holds duplicate fds so main never sees EOF and
#     hangs instead of raising. Seen twice on step_0060000 at n=64.
#   * ONE retry after a timeout. The two observed hangs stalled at different points
#     (after n=32 the first time, inside test n=64 the second), so it is not a fixed bad
#     step and a retry has a real chance of getting through.
set -u
cd "$(dirname "$0")/.."
export DP_OUTPUT_ROOT="${DP_OUTPUT_ROOT:-/home/harine/diffusion_policy_outputs}"
PY=/home/harine/miniconda3/envs/robodiff2/bin/python
ROOT=$DP_OUTPUT_ROOT/pusht_search/pusht_image_search
declare -A ARMS=(
  [search]=$ROOT/outer_inner/value_k16_corrupt-False_demos-30_seed-42
  [bc]=$ROOT/offline/bc_demos-30_seed-42
)

run_combo () {   # $1 arm  $2 ckpt path  $3 step  $4 selection
  timeout 45m $PY -u eval_search_pusht.py -c "$2" \
    -o "$ROOT/bon_grid_30demo/$1" --selection "$4" \
    --min-n 32 --max-n 64 --n-envs 16 --seed 42 --store-scores 2>&1 \
    | grep -E "success_rate=|\[scores\]|Traceback|Error"
  return "${PIPESTATUS[0]}"
}

for arm in search bc; do
  d=${ARMS[$arm]}
  for ckpt in $(ls "$d"/checkpoints/step_*.ckpt | sort); do
    step=$(basename "$ckpt" .ckpt)
    for sel in argmax softmax final_pass; do
      # eval_search_pusht resumes only in its --watch path; with -c it re-runs
      # unconditionally, so a relaunch would redo everything already finished.
      if $PY scripts/_curve_has_n.py "$ROOT/bon_grid_30demo/$arm/success_curves.jsonl" \
             "$step" "$sel"; then
        echo "=== [$(date -Is)] $arm $step $sel -- already has n=32,64, skipping ==="
        continue
      fi
      echo "=== [$(date -Is)] $arm $step $sel ==="
      # Capture the status from a PLAIN call. `if ! run_combo; then rc=$?` sets rc to 0,
      # not the command's status -- $? there is the status of the negation, which is 0
      # precisely because the branch was taken. That silently disabled both the TIMEOUT
      # log and the retry.
      run_combo "$arm" "$ckpt" "$step" "$sel"
      rc=$?
      if [ $rc -ne 0 ]; then
        pkill -f 'multiprocessing.forkserver' 2>/dev/null   # reap the orphaned sim pool
        if [ "$rc" = 124 ]; then
          echo "=== [$(date -Is)] TIMEOUT: $arm $step $sel -- retrying once ==="
          echo "=== [$(date -Is)] $arm $step $sel (retry) ==="
          run_combo "$arm" "$ckpt" "$step" "$sel" || {
            echo "=== [$(date -Is)] TIMEOUT again: $arm $step $sel -- skipped ==="
            pkill -f 'multiprocessing.forkserver' 2>/dev/null; }
        fi
      fi
    done
  done
done
echo "=== [$(date -Is)] n64 done ==="
