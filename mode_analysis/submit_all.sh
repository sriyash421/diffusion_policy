#!/usr/bin/env bash
# Candidate-mode analysis for every ST k=16 arm EXCEPT flat800, at 10k/50k/100k.
#
# WHY k=16 ONLY. The question is whether the candidate set gains modes as the slot index
# grows. ST k=1 has max_actions=1 -- one candidate, no slot axis -- and UNet BC has no
# search at all, so neither has anything to plot.
#
# WHY NOT flat800. Excluded on request: at t=800 the observation is unidentifiable
# (sqrt(alpha_bar)=0.039, frame discriminability at chance) and the arm reads 0.00 at every
# width under its own conditional.
#
# STEPS ARE FILTERED TO WHAT EXISTS. Several arms were stopped early (ramp800to400 at
# 83k/89k) or are still training, so a fixed 10k,50k,100k would be a hard error on them.
# Re-running therefore also backfills: an arm that had only 10k when it was last analysed
# picks up 50k/100k as soon as those checkpoints land.
#
# EVERY SLOT, NOT A SAMPLE. candidate_modes.py defaults to --slots 0,1,3,7,15; the mode
# panels then show 5 of the 16 slots and the ladder has to be read across gaps. SLOTS below
# asks for all of them, which is what makes the per-slot figure comparable to the 16-point
# dispersion curve beside it. Override with e.g. SLOTS=0,1,3,7,15 for the narrow figure.
#
#   bash mode_analysis/submit_all.sh            # dry run
#   SUBMIT=1 bash mode_analysis/submit_all.sh   # ...and sbatch
set -uo pipefail
cd "$(dirname "$0")/.."
SUBMIT="${SUBMIT:-}"
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search/pusht_image_search_imgonly/outer_inner
OUT="${MODE_OUT:-/gscratch/robotics/harine/mode_analysis}"
WANT="${WANT_STEPS:-10000,50000,100000}"
REPEATS="${REPEATS:-32}"; N_STATES="${N_STATES:-3}"
SLOTS="${SLOTS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
# Cap the wall clock so it ends before the next maintenance reservation; SLURM otherwise
# parks the job on the far side of the window. See scripts/slurm/safe_time_limit.sh.
TIME_LIMIT="${TIME_LIMIT:-$(bash scripts/slurm/safe_time_limit.sh 0-03:00:00)}"

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)
n=0; skipped=0
for dir in "$ROOT"/*split-*/; do
    [ -d "$dir" ] || continue
    run=$(basename "$dir")
    case "$run" in *son-flat800*) printf '%-74s %s\n' "$run" "flat800, excluded"; skipped=$((skipped+1)); continue;; esac
    have=""
    for s in ${WANT//,/ }; do
        [ -f "$dir/checkpoints/$(printf 'step_%07d.ckpt' "$s")" ] && have="${have:+$have,}$s"
    done
    if [ -z "$have" ]; then
        printf '%-74s %s\n' "$run" "none of $WANT on disk, skipping"; skipped=$((skipped+1)); continue
    fi
    name="mode_${run}"
    if grep -qxF "$name" <<<"$LIVE"; then
        printf '%-74s %s\n' "$run" "already running"; skipped=$((skipped+1)); continue
    fi
    if [ -z "$SUBMIT" ]; then
        printf '%-74s WOULD SUBMIT steps=%s\n' "$run" "$have"; n=$((n+1)); continue
    fi
    jid=$(sbatch --parsable --job-name="$name" --time="$TIME_LIMIT" \
          mode_analysis/run_modes.sbatch "$dir" \
          --steps "$have" --repeats "$REPEATS" --n-states "$N_STATES" --slots "$SLOTS" \
          --outdir "$OUT/$(python scripts/analysis_run_name.py "$run")")
    printf '%-74s submitted %s steps=%s\n' "$run" "$jid" "$have"
    n=$((n+1))
done
echo; echo "runs handled: $n   skipped: $skipped"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1"
exit 0
