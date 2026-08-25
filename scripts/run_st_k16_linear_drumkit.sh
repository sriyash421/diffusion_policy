#!/usr/bin/env bash
# ST-diffusion k16 (4/4/256) with LINEAR per-slot loss weights, 30 demos, 100k steps,
# then the full every-10k-checkpoint eval grid. Runs on drumkit (local RTX 5000 Ada).
#
#   nohup bash scripts/run_st_k16_linear_drumkit.sh > logs/run_st_k16_linear_drumkit.log 2>&1 &
#
# WHAT THIS ARM IS. One forward decodes all K=16 candidate slots against the same expert
# action, slot k attending to the first k scored candidates. `slot_weights` scales each of
# those K loss terms. This arm sets mode=linear, so w_k is affine in the slot index:
#
#   w_k ∝ 1 + (ratio-1)*k/(K-1), renormalized to mean 1
#   -> 0.341 0.429 0.517 0.605 0.693 0.780 0.868 0.956
#      1.044 1.132 1.220 1.307 1.395 1.483 1.571 1.659
#
# WHY ratio=4.857 AND NOT SOMETHING ROUNDER. `linear` has no default -- the resolver raises
# without a ratio -- so the number is a choice, and this is the only one that makes the
# comparison mean anything. 4.857 = 0.9^-(K-1) at K=16, which is exactly the endpoint spread
# of the legacy `slot_weight_decay: 0.9` geometric profile (w_last/w_first = 4.857 for both,
# verified). Holding the spread fixed is what isolates CURVATURE -- geometric front-loads the
# down-weighting into the low-context slots, linear spreads it evenly -- from the much larger
# effect of simply tilting harder. Any other ratio confounds the two. It is also the ratio the
# config's own sw_suffix example is written for ('_sw-lin4857').
#
# WHY sw_suffix IS NOT OPTIONAL. hydra.run.dir is a pure function of run_name and
# training.resume is on, so a slot_weights override with no suffix resolves to the SAME
# directory as the uniform arm, finds its latest.ckpt, and silently resumes it under a
# different objective. The suffix is what keeps the two runs separable on disk.
#
# ⚠️ NO MATCHED CONTROL EXISTS YET. verifier_tag defaults to armTn, and the slot-weight code
# path requires it. The 4/4/256 uniform k16 run this would naturally be read against
# (outer_inner/value_k16_corrupt-False_demos-30_seed-42, the `search` arm of the existing
# grid) has no verifier_value in its .hydra/config.yaml at all -- it predates the cutover and
# trained under t_goal, a DIFFERENT scoring rule that also feeds the training search context.
# pusht_base.yaml says runs across the two are not comparable, which is why ver- is in
# run_name. So the tables this produces stand alone until a uniform armTn arm is trained:
#   bash scripts/run_st_k16_linear_drumkit.sh --uniform-control
# queues that arm instead (same everything, slot_weights untouched).
set -u
cd "$(dirname "$0")/.."
export DP_OUTPUT_ROOT="${DP_OUTPUT_ROOT:-/home/harine/diffusion_policy_outputs}"
# Interpreter, overridable: these paths are per-machine and a wrong one should fail here
# rather than 100k steps later.
PY="${PY:-/home/harine/miniconda3/envs/robodiff2/bin/python}"
if [ ! -x "$PY" ]; then
  echo "no interpreter at $PY -- set PY=/path/to/env/bin/python" >&2
  exit 1
fi
ROOT=$DP_OUTPUT_ROOT/pusht_search/pusht_image_search
mkdir -p logs

# --uniform-control trains the matched armTn baseline: byte-identical overrides minus the
# slot_weights block, so the pair differs in the weighting and nothing else.
if [ "${1:-}" = "--uniform-control" ]; then
  RUN=value_k16_ver-armTn_corrupt-False_demos-30_seed-42
  ARM=st-k16-unif-armTn
  SW=()
else
  RUN=value_k16_ver-armTn_sw-lin4857_corrupt-False_demos-30_seed-42
  ARM=st-k16-lin4857
  SW=(slot_weights.mode=linear slot_weights.ratio=4.857 sw_suffix=_sw-lin4857)
fi

# n_demos and max_gradient_steps match every arm in the 30-demo grid. checkpoint_every is
# already 10000 in pusht_base, which is what "eval every 10k" below iterates over; it is not
# restated here so the two cannot drift apart -- the eval loop reads whatever was written.
echo "=== [$(date -Is)] TRAIN $ARM -- $RUN ==="
echo "    overrides: n_demos=30 max_gradient_steps=100000 ${SW[*]:-<none>}"
$PY -u train.py --config-name=train_pusht_diffusion_search \
    n_demos=30 training.max_gradient_steps=100000 "${SW[@]}"
rc=$?
echo "=== [$(date -Is)] train exited rc=$rc ==="
if [ $rc -ne 0 ]; then
  echo "train failed; NOT starting the eval grid so the GPU stays free for diagnosis" >&2
  exit $rc
fi

# The same grid the 4/4/256 arms occupy: every 10k checkpoint x {argmax, softmax,
# final_pass} x n in 1..64. Two slices because cost per policy call is linear in the number
# of generations: n<=16 is 31 per call, n in {32,64} is 96. The 45m timeout and the
# forkserver reap are the pattern from run_st_big_30demo.sh -- an AsyncVectorEnv that wedges
# otherwise strands the whole grid, and PIPESTATUS is read because the pipe into grep would
# otherwise mask the exit status (see commit 776170d).
for ckpt in $(ls "$ROOT/outer_inner/$RUN/checkpoints/step_"*.ckpt 2>/dev/null | sort); do
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
      $PY scripts/build_linear_weights_doc.py >/dev/null 2>&1
    done
  done
done
$PY scripts/build_linear_weights_doc.py
echo "=== [$(date -Is)] $ARM done ==="
