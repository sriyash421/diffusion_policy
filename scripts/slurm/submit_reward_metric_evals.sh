#!/bin/bash
# Re-evaluate the val-selected checkpoint of every arm so the FINAL and DISCOUNTED reward
# tables (SUCCESS_RATES.md 1d/1e/2d/2e) have data.
#
# Why a re-run is unavoidable. Both new series are reductions of the per-STEP reward
# sequence, and every eval before 2026-08-05 called np.max() on that sequence inside the
# rollout and discarded the rest. The episode MAX was backfillable from what was on disk
# (scripts/backfill_mean_reward.py); these are not -- the numbers they reduce no longer
# exist. Nothing is wasted otherwise: the eval re-seeds from the same base per (split, n),
# so the success rates it recomputes reproduce the existing ones exactly and merge into the
# same curve rather than replacing it.
#
# Scope is one checkpoint per arm -- the val-selected best from best.json, which is the row
# every headline number is read off. Re-running all ~397 evaluated checkpoints would be
# ~400 GPU-hours for table cells nothing is quoted from.
#
#   bash scripts/slurm/submit_reward_metric_evals.sh          # submit
#   DRY=1 bash scripts/slurm/submit_reward_metric_evals.sh    # print only
set -uo pipefail

ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search/offline
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python
MAX_N="${MAX_N:-64}"
DRY="${DRY:-}"

shopt -s nullglob
submitted=0
for arm in "$ROOT"/*_seed-42; do
    r=$(basename "$arm")
    [ -L "$arm" ] && continue                  # back-symlink from the ctx-* rename
    case "$r" in ctx-*) continue;; esac
    [ -d "$arm/checkpoints" ] || continue

    step=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['step'])" \
           "$arm/bon_search/best.json" 2>/dev/null || echo "")
    [ -n "$step" ] || { echo "  no best.json, skipping  $r"; continue; }
    ckpt=$(printf '%s/checkpoints/step_%07d.ckpt' "$arm" "$step")
    [ -f "$ckpt" ] || { echo "  missing $ckpt, skipping"; continue; }

    # already has the new series at every n it was evaluated at?
    have=$("$PY" - "$arm/bon_search/step_$(printf '%07d' "$step")/success_curve.json" <<'EOF' || echo no
import json, sys
try:
    c = json.load(open(sys.argv[1]))
    f = c.get('mean_reward_final') or []
    print('yes' if len(f) == len(c['n']) and all(v is not None for v in f) else 'no')
except Exception:
    print('no')
EOF
)
    if [ "$have" = yes ]; then
        echo "  have final/discounted  step=$step  $r"
        continue
    fi
    if [ -n "$DRY" ]; then
        echo "  would submit  step=$step  n<=$MAX_N  $r"
        submitted=$((submitted+1))
        continue
    fi
    # ckpt partition, per the standing rule that all success-rate evals go there
    jid=$(sbatch --parsable --time=8:00:00 \
            --job-name="rew_$(echo "$r" | sed 's/_demos-100//;s/_seed-42//')" \
            "$HERE/eval_ckpt_pusht_search.sbatch" "$ckpt" --max-n "$MAX_N")
    echo "  submitted $jid  step=$step  n<=$MAX_N  $r"
    submitted=$((submitted+1))
done
echo "${DRY:+DRY-RUN: }$submitted job(s)"
