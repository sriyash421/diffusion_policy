#!/usr/bin/env bash
# Analyse the t_goal-moving arms AS THEIR CHECKPOINTS LAND, not once at the end.
#
# The seven arms finish days apart -- k=16 is ~20h of GPU and two arms spent hours queued
# behind other work -- so waiting for a matched step means waiting for the slowest arm. This
# submits whatever is newly available, per arm, and the comparison plots are read with
# `--latest` (each arm at its own step, printed on its label).
#
# ONLY the `demos-176_split-mv` arms. attention_analysis/submit_all.sh and
# mode_analysis/submit_all.sh glob every `*split-*` run, which would re-analyse the whole
# blq137/brd60 generation on every pass.
#
# INCREMENTAL, which is the point. candidate_modes and visualize_attention MERGE into their
# json (see merge_summary) instead of overwriting, so this only ever asks for the steps that
# are missing -- re-running costs nothing when nothing new has landed.
#
#   bash scripts/slurm/submit_tgoal_moving_analysis.sh            # dry run
#   SUBMIT=1 bash scripts/slurm/submit_tgoal_moving_analysis.sh   # ...and sbatch
set -uo pipefail
cd "$(dirname "$0")/../.."

SUBMIT="${SUBMIT:-}"
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search/pusht_image_search_imgonly
ATTN_OUT="${ATTN_OUT:-/gscratch/robotics/harine/attention_analysis}"
MODE_OUT="${MODE_OUT:-/gscratch/robotics/harine/mode_analysis}"
PY="${DP_PY:-/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python}"
TAG="${TAG:-demos-176_split-mv}"
REPEATS="${REPEATS:-32}"; N_STATES="${N_STATES:-3}"; N_EPISODES="${N_EPISODES:-8}"
SLOTS="${SLOTS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"   # filtered to < K by candidate_modes
MODE_TIME="${MODE_TIME:-$(bash scripts/slurm/safe_time_limit.sh 0-03:00:00)}"

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)

# steps present as checkpoints, minus steps already recorded in the analysis json
missing() {  # <run_dir> <json path>
    $PY - "$1" "$2" <<'PYEOF'
import json, pathlib, sys
run, js = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
have = {int(p.name[5:-5]) for p in (run/'checkpoints').glob('step_*.ckpt')}
done = set()
if js.is_file():
    try:
        done = {int(k) for k in json.loads(js.read_text()).get('steps', {})}
    except Exception:
        pass
print(','.join(str(s) for s in sorted(have - done)))
PYEOF
}

n=0; skipped=0
for sub in outer_inner offline unet_bc; do
    for dir in "$ROOT/$sub"/*"$TAG"*/; do
        [ -d "$dir" ] || continue
        run=$(basename "$dir")
        canon=$($PY scripts/analysis_run_name.py "$run")
        is_bc=0; case "$run" in unetbc*) is_bc=1;; esac

        # ---- attention. BC has no cross-attention at all (ConditionalUnet1D, FiLM), so it
        # is skipped rather than run and left empty.
        if [ "$is_bc" -eq 0 ]; then
            steps=$(missing "$dir" "$ATTN_OUT/$canon/attention.json")
            name="attn_${run}"
            if [ -z "$steps" ]; then
                : # nothing new
            elif grep -qxF "$name" <<<"$LIVE"; then
                printf '%-70s %s\n' "attn $canon" "already queued"; skipped=$((skipped+1))
            elif [ -z "$SUBMIT" ]; then
                printf '%-70s WOULD SUBMIT steps=%s\n' "attn $canon" "$steps"; n=$((n+1))
            else
                jid=$(sbatch --parsable --job-name="$name" attention_analysis/run_attention.sbatch \
                      "$dir" --steps "$steps" --mode batch --n-episodes "$N_EPISODES" \
                      --outdir "$ATTN_OUT/$canon")
                printf '%-70s submitted %s steps=%s\n' "attn $canon" "$jid" "$steps"; n=$((n+1))
            fi
        fi

        # ---- modes. BC IS included -- its candidates are independent draws with no search
        # context, which is the baseline the ST slot curves are read against -- but its
        # max_actions is None, so the width has to be named explicitly.
        steps=$(missing "$dir" "$MODE_OUT/$canon/modes.json")
        name="mode_${run}"
        extra=""; [ "$is_bc" -eq 1 ] && extra="--n-actions 16"
        if [ -z "$steps" ]; then
            : # nothing new
        elif grep -qxF "$name" <<<"$LIVE"; then
            printf '%-70s %s\n' "mode $canon" "already queued"; skipped=$((skipped+1))
        elif [ -z "$SUBMIT" ]; then
            printf '%-70s WOULD SUBMIT steps=%s %s\n' "mode $canon" "$steps" "$extra"; n=$((n+1))
        else
            jid=$(sbatch --parsable --job-name="$name" --time="$MODE_TIME" \
                  mode_analysis/run_modes.sbatch "$dir" --steps "$steps" \
                  --repeats "$REPEATS" --n-states "$N_STATES" --slots "$SLOTS" $extra \
                  --outdir "$MODE_OUT/$canon")
            printf '%-70s submitted %s steps=%s\n' "mode $canon" "$jid" "$steps"; n=$((n+1))
        fi
    done
done
echo
echo "analysis jobs handled: $n   skipped: $skipped"
[ -z "$SUBMIT" ] && [ "$n" -gt 0 ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
