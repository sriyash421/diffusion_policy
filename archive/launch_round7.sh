#!/bin/bash
# Round 7: the data-budget experiment.
#
#   6 search arms @ 100 demos   {value, subgoal, subgoal_value} x {clean, corrupt}
#   2 BC baselines  @ 100 and 25 demos   (max_actions=1 -> empty context -> plain BC)
#
# All eight use the committed split manifests, so test (50) and val (30) are IDENTICAL
# across every arm and across budgets, and the 25-demo train set is a strict subset of the
# 100-demo one. The six pre-existing runs under offline/ctx-*_seed-42/ are NOT these: they
# predate the manifest and trained on 29 episodes against a 10-episode val split.
#
# Training goes to weirdlab L40/L40S; evals are kept off those nodes (see
# submit_large_n_evals.sh / the watchers) so a long eval never blocks a training slot.
#
#   bash scripts/launch_round7.sh          # launch everything not already running
#   bash scripts/launch_round7.sh --dry    # print what would be submitted
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search/offline
DRY=${1:-}

# config                                                run_name                                    time
JOBS="
train_pusht_bc_25|bc_demos-25_seed-42|1-00:00:00
train_pusht_bc|bc_demos-100_seed-42|1-00:00:00
train_pusht_diffusion_search|ctx-value_corrupt-False_demos-100_seed-42|3-00:00:00
train_pusht_diffusion_search_corrupt|ctx-value_corrupt-True_demos-100_seed-42|3-00:00:00
train_pusht_diffusion_search_subgoal|ctx-subgoal_corrupt-False_demos-100_seed-42|3-00:00:00
train_pusht_diffusion_search_subgoal_corrupt|ctx-subgoal_corrupt-True_demos-100_seed-42|3-00:00:00
train_pusht_diffusion_search_subgoal_verifier|ctx-subgoal_value_corrupt-False_demos-100_seed-42|3-00:00:00
train_pusht_diffusion_search_subgoal_verifier_corrupt|ctx-subgoal_value_corrupt-True_demos-100_seed-42|3-00:00:00
train_pusht_diffusion_search_subgoal_only|subgoal-only_corrupt-False_demos-100_seed-42|3-00:00:00
train_pusht_diffusion_search_subgoal_only_corrupt|subgoal-only_corrupt-True_demos-100_seed-42|3-00:00:00
"
# The two subgoal-only rows differ from the six above in three ways, all deliberate:
#   * run_name is already the post-rename `${arm}_...` form -- these runs have no directory
#     to resume, so they take the new naming now while the others wait (AUDIT.md 9.9).
#   * they train to 100k steps rather than 20k, and the six arms run at roughly 1k
#     steps/hour, so ~4 days. The walltime is still 3 days: a longer request cannot be
#     scheduled ahead of a maintenance reservation (a 7-day one sat at
#     ReqNodeNotAvail behind the 2026-08-11 window rather than starting), and an overrun is
#     cheap anyway -- training.resume finds latest.ckpt, so a relaunch continues.
#   * `selection: final_pass` -- the executed action is a further sample conditioned on the
#     n scored candidates, not the argmax over them.

# Alternate L40 / L40S so eight jobs do not all queue behind one partition.
#
# ACCOUNT and PARTITION are different namespaces here and are easy to confuse:
# `gpu-l40-weirdlab` is the ACCOUNT (that is what `sacctmgr show assoc` lists); the
# PARTITION is plain `gpu-l40`. Passing the account name as --partition fails with
# "invalid partition specified".
ACCTS=(gpu-l40-weirdlab gpu-l40s-weirdlab)
PARTS=(gpu-l40 gpu-l40s)
i=0
for spec in $JOBS; do
    cfg="${spec%%|*}"; rest="${spec#*|}"; run="${rest%%|*}"; tlim="${rest##*|}"
    # never submit a second trainer for a run that already has one: the run dir is
    # identity-keyed, so two jobs would resume the same checkpoint and fight over it
    if squeue -u "$USER" -h -o "%j" | grep -qx "tr_${run}"; then
        echo "  already queued/running: $run"; continue
    fi
    k=$((i % 2)); acct=${ACCTS[$k]}; part=${PARTS[$k]}; i=$((i+1))
    cmd=(sbatch --parsable --account="$acct" --partition="$part" --time="$tlim"
         --job-name="tr_${run}" --export=ALL,CONFIG_NAME="$cfg"
         "$HERE/train_pusht_search.sbatch")
    if [ "$DRY" = --dry ]; then
        echo "  DRY ${cmd[*]}"; continue
    fi
    jid=$("${cmd[@]}")
    echo "  submitted $jid  $part  $cfg -> $run"
done

echo
echo "run dirs: $ROOT/<run_name>/"
