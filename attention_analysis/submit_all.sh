#!/usr/bin/env bash
# Cross-attention analysis for every arm that HAS attention, one SLURM job each.
#
# WHICH ARMS QUALIFY. The memory is [obs tokens | action_value tokens] and the second block
# is the search context, so "does it use its context" is only a question for arms that have
# one:
#   ST k=16   max_context_actions = 15  -> the real subject, geometric AND obs-noise arms
#   ST k=1    max_context_actions = 0   -> obs-only memory; included for the obs-token
#                                          comparison, but its slot plot is empty by
#                                          construction and the script says so
#   UNet BC   ConditionalUnet1D, FiLM   -> NO attention at all; skipped, not "run and empty"
#
# STEPS ARE DISCOVERED, not assumed: an arm mid-training has fewer checkpoints, and asking
# for a step that does not exist is a hard error. Nothing here nominates a best checkpoint --
# it takes an evenly spaced sample of what is on disk (see --n-steps).
#
# Figures go to /gscratch, NOT $HOME: home is a 10GB hard quota and was at 90% when this was
# written, with writes already failing.
#
#   bash attention_analysis/submit_all.sh            # dry run
#   SUBMIT=1 bash attention_analysis/submit_all.sh   # ...and sbatch
set -uo pipefail
cd "$(dirname "$0")/.."

SUBMIT="${SUBMIT:-}"
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search/pusht_image_search_imgonly
OUT="${ATTN_OUT:-/gscratch/robotics/harine/attention_analysis}"
N_STEPS="${N_STEPS:-3}"          # how many checkpoints to sample per run
N_EPISODES="${N_EPISODES:-8}"
MODE="${MODE:-batch}"
MIN_CKPT="${MIN_CKPT:-2}"        # skip arms too early to say anything

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)
n=0; skipped=0
# offline == ST k=1, outer_inner == ST k=16. unet_bc is deliberately absent.
for sub in outer_inner offline; do
    for dir in "$ROOT/$sub"/*split-*/; do
        [ -d "$dir" ] || continue
        run=$(basename "$dir")
        mapfile -t ck < <(ls "$dir"checkpoints/step_*.ckpt 2>/dev/null | sed 's/.*step_0*//;s/\.ckpt//' | sort -n)
        if [ "${#ck[@]}" -lt "$MIN_CKPT" ]; then
            printf '%-74s %s\n' "$run" "only ${#ck[@]} checkpoint(s), skipping"; skipped=$((skipped+1)); continue
        fi
        # evenly spaced sample INCLUDING the newest, which is the one people ask about
        steps=$(python - "$N_STEPS" "${ck[@]}" <<'PY'
import sys
n=int(sys.argv[1]); ck=[int(x) for x in sys.argv[2:]]
if n>=len(ck): pick=ck
else:
    idx=sorted({round(i*(len(ck)-1)/(n-1)) for i in range(n)}) if n>1 else [len(ck)-1]
    pick=[ck[i] for i in idx]
if ck[-1] not in pick: pick.append(ck[-1])
print(','.join(str(x) for x in sorted(set(pick))))
PY
)
        name="attn_${run}"
        if grep -qxF "$name" <<<"$LIVE"; then
            printf '%-74s %s\n' "$run" "already analysing, skipping"; skipped=$((skipped+1)); continue
        fi
        if [ -z "$SUBMIT" ]; then
            printf '%-74s WOULD SUBMIT steps=%s (%d ckpts)\n' "$run" "$steps" "${#ck[@]}"
            n=$((n+1)); continue
        fi
        jid=$(sbatch --parsable --job-name="$name" attention_analysis/run_attention.sbatch \
              "$dir" --steps "$steps" --mode "$MODE" --n-episodes "$N_EPISODES" \
              --outdir "$OUT/$(python "$(dirname "$0")/../scripts/analysis_run_name.py" "$run")")
        printf '%-74s submitted %s  steps=%s\n' "$run" "$jid" "$steps"
        n=$((n+1))
    done
done
echo
echo "runs handled: $n   skipped: $skipped"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
