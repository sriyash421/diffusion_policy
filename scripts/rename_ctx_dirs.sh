#!/bin/bash
# Rename the run directories from the pre-`selection` `ctx-<search_context>` template to the
# arm labels. See the 2026-08-05 run-directory rename for why the names had to change: `selection` became a second
# axis, so `ctx-subgoal` no longer identifies an arm -- it is `subgoal-chosen4value` under
# argmax and `subgoal-only` under final_pass, which are different experiments.
#
#   search_context   selection    arm label
#   value            argmax       value
#   subgoal          argmax       subgoal-chosen4value
#   subgoal_value    argmax       subgoal-value
#   subgoal          final_pass   subgoal-only        (already on the target name)
#
# The 29-demo runs additionally gain an explicit `demos-29`, which they never carried --
# their budget was implicit in the ABSENCE of a demos- component, which is exactly the kind
# of thing that gets misread later.
#
# SAFETY. Every rename is `mv old new && ln -s new old`, so the old absolute path keeps
# resolving to the same inode. That matters because jobs hold absolute paths: at the time of
# writing six `cand64_*` jobs read `ctx-*/checkpoints/step_*.ckpt` through paths baked into
# their --export, and they requeue on preemption and re-resolve. Without the back-symlink
# those would fail on restart. Drop the symlinks only once nothing resolves through them.
#
#   bash scripts/rename_ctx_dirs.sh --dry     # print the map, touch nothing
#   bash scripts/rename_ctx_dirs.sh           # execute
set -euo pipefail

ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search/offline
DRY=${1:-}

# old|new
MAP="
ctx-value_corrupt-False_demos-100_seed-42|value_corrupt-False_demos-100_seed-42
ctx-value_corrupt-True_demos-100_seed-42|value_corrupt-True_demos-100_seed-42
ctx-subgoal_corrupt-False_demos-100_seed-42|subgoal-chosen4value_corrupt-False_demos-100_seed-42
ctx-subgoal_corrupt-True_demos-100_seed-42|subgoal-chosen4value_corrupt-True_demos-100_seed-42
ctx-subgoal_value_corrupt-False_demos-100_seed-42|subgoal-value_corrupt-False_demos-100_seed-42
ctx-subgoal_value_corrupt-True_demos-100_seed-42|subgoal-value_corrupt-True_demos-100_seed-42
ctx-value_corrupt-False_seed-42|value_corrupt-False_demos-29_seed-42
ctx-value_corrupt-True_seed-42|value_corrupt-True_demos-29_seed-42
ctx-subgoal_corrupt-False_seed-42|subgoal-chosen4value_corrupt-False_demos-29_seed-42
ctx-subgoal_corrupt-True_seed-42|subgoal-chosen4value_corrupt-True_demos-29_seed-42
ctx-subgoal_value_corrupt-False_seed-42|subgoal-value_corrupt-False_demos-29_seed-42
ctx-subgoal_value_corrupt-True_seed-42|subgoal-value_corrupt-True_demos-29_seed-42
"

n=0
for spec in $MAP; do
    old="${spec%%|*}"; new="${spec##*|}"
    if [ ! -e "$ROOT/$old" ]; then echo "  MISSING  $old"; continue; fi
    if [ -L "$ROOT/$old" ]; then echo "  done already (symlink)  $old"; continue; fi
    if [ -e "$ROOT/$new" ]; then echo "  TARGET EXISTS, skipping  $new"; continue; fi
    if [ "$DRY" = --dry ]; then
        printf '  %-50s -> %s\n' "$old" "$new"
    else
        mv "$ROOT/$old" "$ROOT/$new"
        ln -s "$new" "$ROOT/$old"
        printf '  renamed %-50s -> %s  (+ back-symlink)\n' "$old" "$new"
    fi
    n=$((n+1))
done
echo "${DRY:+DRY-RUN: }$n director$([ $n -eq 1 ] && echo y || echo ies)"
