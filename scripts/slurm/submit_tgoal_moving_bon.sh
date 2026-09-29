#!/usr/bin/env bash
# BEST-OF-N SUCCESS RATES for the t_goal-moving arms, on the held-out episodes.
#
# ⚠️ THIRTY EPISODES, NOT FIFTY. This generation holds out 30 episodes (176 train / 0 val /
# 30 test) because the 30 were chosen to put the MOVING-transition ratio at 100:20. The
# canonical seed-42 50-episode test set is NOT usable here: 43 of its 50 episodes are inside
# this generation's TRAINING set, so scoring on it would be the same train leak
# docs/reports/RESULTS.md records for the retired eval_bon geometric numbers. At n=30 a 95%
# Wilson interval is about +/-18 points at p=0.5, so treat small differences as noise.
#
# eval_search_pusht reads the split from the checkpoint's own cfg and cross-checks it against
# <run>/splits.json, so it cannot silently score a different episode set.
#
# TWO READOUTS FOR THE LADDER ARMS. `--corrupt-obs-eval` reproduces the slot->level mapping
# the loss trained under; `--no-corrupt-obs-eval` evaluates every slot clean. The clean run is
# what makes the ladder arms comparable with uniform and BC; the corrupt run is the only one
# that measures them under the conditional they were fitted to. Both are keyed into the output
# directory, so they can never merge into one curve. The flag is a no-op on an arm with no
# registered ladder, so uniform and BC get the clean row only.
#
#   bash scripts/slurm/submit_tgoal_moving_bon.sh            # dry run
#   SUBMIT=1 bash scripts/slurm/submit_tgoal_moving_bon.sh   # ...and sbatch
#   FORCE=1 ...                                              # re-measure what is already done
#   CLEAN_ONLY=1 ...    # clean readout only        CORRUPT_ONLY=1 ...  # corrupt readout only
#   ARM_FILTER='son-|unetbc' ...                             # restrict to matching run names
#   DEMOS=137 SPLIT_TAG=split-blq-mv ARM_FILTER='son-(flat|ramp)600' ...   # another generation
set -uo pipefail
cd "$(dirname "$0")/../.."

SUBMIT="${SUBMIT:-}"
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search/pusht_image_search_imgonly
STEP="${STEP:-100000}"
N_LIST="${N_LIST:-1,4,8,16}"
SELECTION="${SELECTION:-argmax}"
# WHICH GENERATION. The run-name tail, so this launcher serves any parent split rather than
# being copied per generation. blq137-filtered: DEMOS=137 SPLIT_TAG=split-blq-mv, with
# ARM_FILTER='son-(flat|ramp)600' to select the four arms that exist there.
DEMOS="${DEMOS:-176}"
SPLIT_TAG="${SPLIT_TAG:-split-mv}"
PY="${DP_PY:-/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python}"
PY="${DP_PY:-/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python}"

# subdir | run stem | has a ladder?
ARMS=("unet_bc|unetbc_ver-t_goal|0"
      "outer_inner|value_k16_ver-t_goal|0"
      "outer_inner|value_k16_ver-t_goal_son-flat400|1"
      "outer_inner|value_k16_ver-t_goal_son-ramp400to200|1"
      "outer_inner|value_k4_ver-t_goal|0"
      "outer_inner|value_k4_ver-t_goal_son-flat400|1"
      "outer_inner|value_k4_ver-t_goal_son-ramp400to200|1"
      # the 2026-09-19 t=600 pair. An arm with no step-100k checkpoint yet is skipped with a
      # message, so this script can be re-run as they finish rather than held until all four.
      "outer_inner|value_k16_ver-t_goal_son-flat600|1"
      "outer_inner|value_k16_ver-t_goal_son-ramp600to200|1"
      "outer_inner|value_k4_ver-t_goal_son-flat600|1"
      "outer_inner|value_k4_ver-t_goal_son-ramp600to200|1")

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)
n=0; skipped=0
for a in "${ARMS[@]}"; do
    IFS='|' read -r sub stem lad <<<"$a"
    run="${stem}_enc-resnet18_demos-${DEMOS}_${SPLIT_TAG}_seed-42"
    # ARM_FILTER is an extended regex on the run name. Added for the n=64 sweep, where a level
    # costs ~4x n=16: under "each arm's own conditional" only the three unladdered arms need a
    # clean row, and without a filter the clean pass would submit all eleven.
    if [ -n "${ARM_FILTER:-}" ] && ! grep -qE "$ARM_FILTER" <<<"$run"; then
        skipped=$((skipped+1)); continue
    fi
    ckpt="$ROOT/$sub/$run/checkpoints/$(printf 'step_%07d.ckpt' "$STEP")"
    if [ ! -f "$ckpt" ]; then
        printf '%-74s %s\n' "$run" "no step $STEP, skipping"; skipped=$((skipped+1)); continue
    fi
    rules=()
    # CLEAN_ONLY=1 drops the corrupt readout. Used for the wide n sweep, where n=64 costs ~4x
    # n=16 per level and the corrupt rows are a secondary comparison. CORRUPT_ONLY=1 is its
    # mirror: it backfills the corrupt rows for the ladder arms without re-running the clean
    # ones, which FORCE=1 alone cannot express. An arm with no ladder has no corrupt readout,
    # so under CORRUPT_ONLY it drops out with a message rather than submitting nothing.
    [ -z "${CORRUPT_ONLY:-}" ] && rules+=("--no-corrupt-obs-eval|clean")
    [ "$lad" = "1" ] && [ -z "${CLEAN_ONLY:-}" ] && rules+=("--corrupt-obs-eval|corrupt")
    if [ "${#rules[@]}" -eq 0 ]; then
        printf '%-74s %s\n' "$run" "no corrupt readout (no ladder), skipping"
        skipped=$((skipped+1)); continue
    fi
    for r in "${rules[@]}"; do
        IFS='|' read -r flag tag <<<"$r"
        # SELECTION **and STEP** are in the job name. Each (arm, rule, selection, step) is its
        # own readout; leaving either out makes the "already queued" check below treat an
        # unrelated queued job as covering this one. Measured: without STEP, a sweep over
        # 20k/40k/60k/80k submitted only the first step and silently skipped the other three.
        name="bon_${SELECTION}_${STEP}_${tag}_${run}"
        # ALREADY MEASURED? eval_search_pusht merges into <run>/bon_search_sel-<sel>_obs-<tag>/
        # keyed by the readout rule, so a finished (arm, rule, step) is visible on disk. Without
        # this the script only skips a QUEUED job and re-running it to pick up one new arm
        # resubmits every finished one -- 11 wasted GPU jobs on the first such re-run.
        curve="$ROOT/$sub/$run/bon_search_sel-${SELECTION}_obs-${tag}/success_curves.jsonl"
        # N-AWARE. eval_search_pusht MERGES new n levels into the same file, so "the step is
        # present" is not the same as "every requested n is present" -- checking only the step
        # made a re-run for extra levels (n=2,32,64) skip every arm that already had 1,4,8,16.
        if [ -z "${FORCE:-}" ] && [ -f "$curve" ] && $PY - "$curve" "$STEP" "$N_LIST" <<'PYEOF'
import json, sys
path, step, want = sys.argv[1], int(sys.argv[2]), {int(x) for x in sys.argv[3].split(',')}
have = set()
for line in open(path):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    if int(d.get('step', -1)) == step:
        have |= set(d.get('n', []))
sys.exit(0 if want <= have else 1)
PYEOF
        then
            printf '%-74s %s\n' "$run [$tag]" "already evaluated at $STEP for n=$N_LIST"
            skipped=$((skipped+1)); continue
        fi
        if grep -qxF "$name" <<<"$LIVE"; then
            printf '%-74s %s\n' "$run [$tag]" "already queued"; skipped=$((skipped+1)); continue
        fi
        if [ -z "$SUBMIT" ]; then
            printf '%-74s WOULD SUBMIT [%s] n=%s\n' "$run" "$tag" "$N_LIST"; n=$((n+1)); continue
        fi
        jid=$(sbatch --parsable --job-name="$name" scripts/slurm/eval_ckpt_pusht_search.sbatch \
              "$ckpt" --n-list "$N_LIST" --selection "$SELECTION" "$flag" --skip-val)
        printf '%-74s submitted %s [%s]\n' "$run" "$jid" "$tag"; n=$((n+1))
    done
done
echo
# The episode count is NOT set here -- eval_search_pusht reads it from the checkpoint's own
# cfg and cross-checks <run>/splits.json. Printing a literal would have said "30" for the
# blq137 generation, whose test split is 50.
echo "eval jobs handled: $n   skipped: $skipped   step=$STEP  n-list=$N_LIST" \
     "  (episodes: each run's own test split)"
[ -z "$SUBMIT" ] && [ "$n" -gt 0 ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
