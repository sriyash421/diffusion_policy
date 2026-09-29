#!/usr/bin/env bash
# Does substituting the learned Q for t_goal make best-of-N solve more episodes?
#
#   bash scripts/slurm/submit_q_bon_mv.sh           # dry run
#   SUBMIT=1 bash scripts/slurm/submit_q_bon_mv.sh  # ...and sbatch it
#
# rank-expert asks whether Q ORDERS a* well; this asks the deployable question instead -- run
# the policy, let each verifier pick, count solved episodes. A verifier can rank a* well and
# still not help.
#
# EPISODES COME FROM THE CHECKPOINT'S OWN MANIFEST here, unlike the rank-expert sweep, and that
# is correct: best-of-N must be scored on episodes the POLICY held out. For these arms that is
# pusht_seed42_train176's test-30. Note 20 of those 30 are inside the 106, so even sac106 is
# clean on only 10 of 30; the clean Q comparison is the rank-expert sweep, not this one.
# 30 episodes puts the 95% Wilson interval near +-0.17, so read the trend in n, not a single
# cell -- the curves are paired episode by episode under both rankers.
#
# t_goal is requested in BOTH jobs of an arm rather than once. It costs one extra sim sweep and
# buys the check that matters: the two t_goal curves must come out identical, and at n=1 every
# ranker must agree because there is nothing to rank.
set -uo pipefail

source "$(dirname "$0")/q_arms.sh"

OUT="${OUT:-/gscratch/robotics/harine/value_arms/bon_q_mv}"
RANKERS="${RANKERS:-t_goal,q}"
MAX_N="${MAX_N:-16}"
SPLIT="${SPLIT:-test}"
# SELECTION=final_pass is the verifier-off control: the n'th generation conditioned on the other
# n-1, nothing selected. It answers "did searching wider change the SAMPLER or the SELECTOR",
# which a success rate under argmax alone cannot. Under it the ranker is inert, so one job per
# arm with RANKERS=t_goal is the whole sweep.
SELECTION="${SELECTION:-}"
TAG="${TAG:-}"   # appended to the output dir, so two selections never overwrite each other

while IFS='|' read -r label run flags; do
    label="$(echo "$label" | xargs)"; run="$(echo "$run" | xargs)"; flags="$(echo "$flags" | xargs)"
    [ -z "$label" ] && continue
    ck="$run/$STEP"
    [ -f "$ROOT/$ck" ] || { echo "[SKIP $label] no $ck"; continue; }

    # WHEN `q` IS NOT A RANKER THE Q IS NEVER BUILT, so iterating the Q list would submit the
    # same job twice under two different names. Collapse to one nameless run instead: under
    # --selection final_pass the ranker is inert, and a directory called `..._sac206_...` for a
    # curve no Q took part in is a mislabel waiting to be quoted.
    if [ "${RANKERS#*q}" = "$RANKERS" ]; then
        cmd="python -u sac/eval.py bon-sweep -c $ck --rankers $RANKERS \
--max-n $MAX_N --split $SPLIT --n-envs ${CPUS:-8} ${SELECTION:+--selection $SELECTION} $flags \
-o $OUT/${label}${TAG}"
        submit_or_echo "qbon_${label}${TAG}" "$cmd"
        continue
    fi

    while IFS='|' read -r qlabel qck contam; do
        qlabel="$(echo "$qlabel" | xargs)"; qck="$(echo "$qck" | xargs)"
        contam="$(echo "$contam" | xargs)"
        [ -z "$qlabel" ] && continue
        [ -f "$qck" ] || { echo "[SKIP $label/$qlabel] no $qck"; continue; }
        [ -f "$(dirname "$qck")/params/args.yaml" ] || {
            echo "[SKIP $label/$qlabel] no params/args.yaml beside $qck"; continue; }

        # --skip-context-sim is BC-ONLY and comes from ARMS, not from a flag here: it is a
        # property of the policy (BC ignores the search context), not of this sweep.
        # 20 of the mv test-30 are inside the 106, so BOTH Qs are contaminated on this split --
        # sac206 wholly, sac106 on 20 of 30. The flag is required either way and the report
        # records which.
        sel=""; [ -n "$SELECTION" ] && sel="--selection $SELECTION"
        cmd="python -u sac/eval.py bon-sweep -c $ck --q $qck --rankers $RANKERS \
--max-n $MAX_N --split $SPLIT --n-envs ${CPUS:-8} --allow-contaminated $sel $flags \
-o $OUT/${label}_${qlabel}${TAG}"
        submit_or_echo "qbon_${label}_${qlabel}${TAG}" "$cmd"
    done <<< "$QS"
done <<< "$ARMS"

[ "${SUBMIT:-0}" = "1" ] || echo $'\nDry run. SUBMIT=1 to sbatch.'
