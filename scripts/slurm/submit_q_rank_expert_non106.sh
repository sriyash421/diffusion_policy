#!/usr/bin/env bash
# Where does the expert chunk a* rank under Q, on the demos SAC was never seeded from?
#
#   bash scripts/slurm/submit_q_rank_expert_non106.sh           # dry run
#   SUBMIT=1 bash scripts/slurm/submit_q_rank_expert_non106.sh  # ...and sbatch it
#
# THE EPISODE SET IS Q'S, NOT THE POLICY'S. rank-expert normally resolves episodes through the
# ST/BC checkpoint's own manifest, which answers "what did this policy hold out" -- unrelated to
# what a learned verifier was fitted on. --episodes-from points it at the val+test halves of
# pusht_seed42_train106_val50.json instead: the 100 episodes outside the 106 that
# `demo_episodes=train` seeding preloads. All 100 are clean for sac106; NONE is clean for
# sac206, which seeded from all 206 by design.
#
# Both verifiers score the SAME decisions, the SAME candidates and the SAME a*, and the output
# carries the block_still / block_moving split -- block_still is where t_goal is blind by
# construction and a learned Q has to earn its place.
set -uo pipefail

source "$(dirname "$0")/q_arms.sh"

OUT="${OUT:-/gscratch/robotics/harine/value_arms/qeval_non106}"
SPLITFILE=diffusion_policy/config/splits/pusht_seed42_train106_val50.json
N="${N:-16}"
EPISODES="${EPISODES:-100}"
PER_EPISODE="${PER_EPISODE:-8}"

while IFS='|' read -r label run _flags; do
    label="$(echo "$label" | xargs)"; run="$(echo "$run" | xargs)"
    [ -z "$label" ] && continue
    ck="$run/$STEP"
    [ -f "$ROOT/$ck" ] || { echo "[SKIP $label] no $ck"; continue; }

    while IFS='|' read -r qlabel qck contam; do
        qlabel="$(echo "$qlabel" | xargs)"; qck="$(echo "$qck" | xargs)"
        contam="$(echo "$contam" | xargs)"
        [ -z "$qlabel" ] && continue
        [ -f "$qck" ] || { echo "[SKIP $label/$qlabel] no $qck"; continue; }
        # PushTQVerifier reads params/args.yaml from beside the checkpoint for obs type and the
        # tau ladder; without it the Q is built against the wrong observation width.
        [ -f "$(dirname "$qck")/params/args.yaml" ] || {
            echo "[SKIP $label/$qlabel] no params/args.yaml beside $qck"; continue; }

        # These 100 episodes are outside sac106's 106, so only sac206 is contaminated here.
        allow=""; [ "$contam" = "always" ] && allow="--allow-contaminated"
        cmd="python -u sac/eval.py rank-expert -c $ck --arm ${label}_${qlabel} --q $qck $allow \
--n $N --episodes $EPISODES --per-episode $PER_EPISODE \
--episodes-from $SPLITFILE --episodes-split val,test \
--out $OUT/rank_${label}_${qlabel}.json"
        submit_or_echo "qrank_${label}_${qlabel}" "$cmd"
    done <<< "$QS"
done <<< "$ARMS"

[ "${SUBMIT:-0}" = "1" ] || echo $'\nDry run. SUBMIT=1 to sbatch.'
