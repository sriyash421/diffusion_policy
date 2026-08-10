#!/bin/bash
# Round 8: the six 29-demo argmax arms, retrained at parity with the 100-demo generation.
#
# Same 29 TRAINING EPISODES as the legacy 29-demo runs (verified byte-identical -- see
# scripts/make_29demo_parity_split.py), so old vs new isolates exactly the three changes:
#   use_ema False -> True (0.995)   the legacy numbers came from the live weights
#   encoder-only crop -> policy-level crop_shape [76,76]   obs and its derived subgoals now
#                                   share ONE crop offset instead of being translated apart
#   20k -> 100k gradient steps
#
# Run names carry `-r8`. They MUST: the inherited template resolves to the legacy directory
# name, and with training.resume: True that would continue the legacy 20k no-EMA checkpoint
# instead of starting fresh -- and overwrite it.
#
#   bash scripts/slurm/launch_round8_29demo.sh          # submit
#   DRY=1 bash scripts/slurm/launch_round8_29demo.sh    # print only
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search/offline
CONFIG=train_pusht_diffusion_search_29
DRY="${DRY:-}"
# MUST fit before the next maintenance reservation. A request that would still be running
# when the window opens is not queued behind it -- it sits at ReqNodeNotAvail indefinitely,
# which is how all six of these launched and then never started. Check with
# `scontrol show reservation` and shorten to fit; an overrun costs nothing because
# training.resume finds latest.ckpt and a relaunch continues (verified end to end by
# scripts/slurm/test_resume.sbatch, 12/12 assertions on this exact config family).
TLIM="${TLIM:-2-18:00:00}"

# search_context|arm|corrupt
ARMS="
value|value|False
value|value|True
subgoal|subgoal-chosen4value|False
subgoal|subgoal-chosen4value|True
subgoal_value|subgoal-value|False
subgoal_value|subgoal-value|True
"

# Alternate L40 / L40S so six jobs do not all queue behind one partition. ACCOUNT and
# PARTITION are different namespaces: `gpu-l40-weirdlab` is the ACCOUNT, `gpu-l40` the
# partition -- passing the account as --partition fails with "invalid partition specified".
ACCTS=(gpu-l40-weirdlab gpu-l40s-weirdlab)
PARTS=(gpu-l40 gpu-l40s)
i=0
n=0
for spec in $ARMS; do
    sc="${spec%%|*}"; rest="${spec#*|}"; arm="${rest%%|*}"; cor="${rest##*|}"
    run="${arm}_corrupt-${cor}_demos-29-r8_seed-42"

    # never submit a second trainer for a run that already has one: the run dir is
    # identity-keyed, so two jobs would resume the same checkpoint and fight over it
    if squeue -u "$USER" -h -o "%j" | grep -qx "tr_${run}"; then
        echo "  already training  $run"; continue
    fi
    acct="${ACCTS[$((i % 2))]}"; part="${PARTS[$((i % 2))]}"; i=$((i+1))

    if [ -n "$DRY" ]; then
        echo "  would submit  $run  ($part)  search_context=$sc arm=$arm corrupt_obs=$cor"
        n=$((n+1)); continue
    fi
    jid=$(sbatch --parsable --account="$acct" --partition="$part" --time="$TLIM" \
            --job-name="tr_${run}" --export=ALL,CONFIG_NAME="$CONFIG" \
            "$HERE/train_pusht_search.sbatch" \
            "search_context=$sc" "arm=$arm" "corrupt_obs=$cor")
    echo "  $jid  $run  ($part)"
    n=$((n+1))

    # Watcher: --skip-val, so every checkpoint is scored on the SAME 50 test episodes and
    # nothing is spent on val. That makes best.json's marker test-selected rather than
    # val-selected -- fine here because the doc reports every checkpoint and the star is
    # only a convenience, but it means the starred number must not be quoted as if it were
    # held out.
    [ -n "$DRY" ] || wid=$(sbatch --parsable --account=robotics --partition=ckpt \
            --job-name="ev_${run}" \
            "$HERE/eval_watch_pusht_search.sbatch" "$ROOT/$run" --max-n 64 --skip-val)
    echo "        watcher $wid"
done
echo "${DRY:+DRY-RUN: }$n arm(s)"
