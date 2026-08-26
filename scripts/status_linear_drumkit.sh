#!/usr/bin/env bash
# One-screen status for the st-k16-lin4857 arm (scripts/run_st_k16_linear_drumkit.sh).
#
# WHY THIS EXISTS RATHER THAN `tail`. The driver's stdout carries only the "context buffer"
# tqdm -- the outer/inner trainer's pool regeneration -- and NO gradient-step counter, so a
# tail of the log cannot tell you the step, the rate, or the loss, and cannot distinguish
# "stepping fine" from "wedged mid-pool". global_step lives in wandb-summary.json, which the
# trainer rewrites as it goes; that file is the actual progress signal and is what this reads.
#
# ETA is computed from _runtime/global_step -- the run's own average since start, not a
# recent-window rate -- so it is stable but lags a genuine slowdown. Cross-check against the
# process etime if a number looks wrong.
set -u
cd "$(dirname "$0")/.."
export DP_OUTPUT_ROOT="${DP_OUTPUT_ROOT:-/home/harine/diffusion_policy_outputs}"
PY="${PY:-/home/harine/miniconda3/envs/robodiff2/bin/python}"
ROOT=$DP_OUTPUT_ROOT/pusht_search/pusht_image_search
ARM=${ARM:-st-k16-lin4857}
RUN=${RUN:-value_k16_ver-armTn_sw-lin4857_corrupt-False_demos-30_seed-42}
D=$ROOT/outer_inner/$RUN
LOG=logs/run_st_k16_linear_drumkit.log

alive=$(ps -eo args --no-headers | grep -c "[t]rain.py --config-name=train_pusht_diffusion_search")
echo "run       : $RUN"
echo "training  : $([ "$alive" -gt 0 ] && echo "alive ($alive procs)" || echo "NOT RUNNING")"

$PY - "$D" "$LOG" <<'PYEOF'
import glob, json, os, sys, time
d, log = sys.argv[1], sys.argv[2]
TARGET = 100000
f = sorted(glob.glob(os.path.join(d, 'wandb', '*', 'files', 'wandb-summary.json')))
if f:
    s = json.load(open(f[-1]))
    step, rt = s.get('global_step', 0), s.get('_runtime', 0) or 1
    rate = step / rt
    left = (TARGET - step) / rate if rate else 0
    stale = time.time() - os.path.getmtime(f[-1])
    print(f"step      : {step:,} / {TARGET:,}  ({100*step/TARGET:.1f}%)  @ {rate:.2f} steps/s")
    print(f"eta       : {left/3600:.1f}h to 100k  (summary written {stale:.0f}s ago)")
    print(f"loss      : train {s.get('train_loss', float('nan')):.4f}   "
          f"val {s.get('val_loss', float('nan')):.4f}   epoch {s.get('epoch', '?')}")
    # A summary that has not been touched in minutes while the process is alive is the
    # signature of a wedged verifier pool, which is the failure this whole check is for.
    if stale > 600:
        print(f"  !! STALLED: wandb summary untouched for {stale/60:.0f} min")
else:
    print("step      : no wandb summary yet (still building the first context pool)")

ck = sorted(glob.glob(os.path.join(d, 'checkpoints', 'step_*.ckpt')))
print(f"ckpts     : {len(ck)}/10" + (f"  latest {os.path.basename(ck[-1])}" if ck else ""))
if os.path.exists(log):
    bad = [l for l in open(log, errors='ignore')
           if any(k in l for k in ('Traceback', 'CUDA out of memory', 'No space left',
                                   'Killed', 'TIMEOUT:')) or ('train exited rc=' in l and 'rc=0' not in l)]
    print(f"errors    : {len(bad)}" + ("" if not bad else "  <-- see below"))
    for l in bad[-5:]:
        print("            " + l.rstrip()[:140])
PYEOF

curves=$ROOT/bon_grid_30demo/$ARM/success_curves.jsonl
if [ -f "$curves" ]; then
  echo "eval grid : $(wc -l < "$curves") rows in bon_grid_30demo/$ARM"
  $PY scripts/build_linear_weights_doc.py
else
  echo "eval grid : not started (runs after training exits 0)"
fi
echo "disk      : $(df -h --output=avail / | tail -1 | tr -d ' ') free on /"
