#!/bin/bash
# Submit the large-n best-of-N sweep as ONE JOB PER n, on the preemptible ckpt partition.
#
# Why split. Cost is linear in n and the levels sum to ~2*max_n, so a single n<=1024 sweep
# runs ~10h per checkpoint (both splits; val's 10 episodes are padded out to n_envs=50, so
# val and test cost the SAME -- measured 505s vs 503s at n=32). Six such jobs were killed at
# a 9h walltime having written nothing, because the old eval persisted only after the whole
# sweep returned. eval_search_pusht.py now writes after every n and merges under a lock, so
# the levels can be run independently and land in one success_curve.json.
#
# n<=64 already exists for all 20 checkpoints of all 6 arms, so only the tail is submitted;
# --min-n == --max-n makes each job exactly one level.
#
# Wall time per job, from measured single-split times x2 splits, ~1.6x margin:
#   n=128 ~53m   n=256 ~100m   n=512 ~135m   n=1024 ~270m
#
#   bash scripts/slurm/submit_large_n_evals.sh 1000 2000  # explicit steps, all arms
#
# Steps are REQUIRED. This script used to default to the winner recorded in best.json, which
# silently spent the tail budget according to one selection rule (val success at the largest
# common n). Nothing records a winner any more -- read success_curves.jsonl, decide which
# steps you want, and pass them.
set -euo pipefail

ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search/offline
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python

declare -A TIME=( [128]=2:00:00 [256]=3:30:00 [512]=5:00:00 [1024]=8:00:00 )
# n=1024 is NOT in the default sweep. It needs 8-12h for a single level and writes nothing
# until that level finishes, so on the preemptible ckpt partition it is the one eval shape
# that cannot survive a preemption -- three attempts died that way. n<=512 covers the tail
# (the arms that have it are flat or saturated from 256 on), so 1024 is not worth a
# guaranteed-partition exception. Pass NS=1024 explicitly if you want it anyway.
NS="${NS:-128 256 512}"
STEPS=("$@")
if [ ${#STEPS[@]} -eq 0 ]; then
    echo "usage: $(basename "$0") <step> [step...]" >&2
    echo "no default: pick the steps from each run's bon_search/success_curves.jsonl" >&2
    exit 2
fi

submitted=0
# Arm-label globs (post-rename, AUDIT.md 9.9). Deliberately NOT `ctx-*`: those paths now
# exist only as back-symlinks, so globbing them would visit every run twice -- once by each
# name -- and the second visit reads as a separate arm rather than as a duplicate.
shopt -s nullglob
# `bc_*` is in the list because the BC baseline is the comparison every search number is
# read against; leaving it out meant a large-n sweep produced a tail for the search arms
# and no tail for the thing they are being compared to.
for arm in "$ROOT"/value_*_seed-42 "$ROOT"/subgoal-chosen4value_*_seed-42 \
           "$ROOT"/subgoal-value_*_seed-42 "$ROOT"/subgoal-only_*_seed-42 \
           "$ROOT"/bc_*_seed-42; do
    [ -d "$arm/checkpoints" ] || continue
    arm_steps=("${STEPS[@]}")
    for step in "${arm_steps[@]}"; do
        ckpt=$(printf '%s/checkpoints/step_%07d.ckpt' "$arm" "$step")
        [ -f "$ckpt" ] || { echo "missing $ckpt, skipping"; continue; }
        for n in $NS; do
            # skip levels already on disk -- reruns are idempotent, not wasteful
            have=$("$PY" - "$arm/bon_search/step_$(printf '%07d' "$step")/success_curve.json" "$n" <<'EOF' || echo no
import json, sys
try:
    print('yes' if int(sys.argv[2]) in json.load(open(sys.argv[1]))['n'] else 'no')
except Exception:
    print('no')
EOF
)
            if [ "$have" = yes ]; then
                echo "  have n=$n step=$step $(basename "$arm")"
                continue
            fi
            # --partition/--account are ckpt/robotics from the sbatch header;
            # ALL success-rate evals run on the checkpoint partition.
            jid=$(sbatch --parsable --time="${TIME[$n]}" \
                    --job-name="bon_n${n}_$(basename "$arm" | sed 's/ctx-//;s/_seed-42//')" \
                    "$HERE/eval_ckpt_pusht_search.sbatch" "$ckpt" \
                    --min-n "$n" --max-n "$n")
            echo "  submitted $jid  n=$n  step=$step  $(basename "$arm")"
            submitted=$((submitted+1))
        done
    done
done
echo "submitted $submitted job(s)"
