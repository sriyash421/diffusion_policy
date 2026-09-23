#!/usr/bin/env bash
# Gather every figure for the t_goal-moving generation into ONE folder.
#
# The per-arm outputs live under <mode_analysis>/<canonical run name>/ and
# <attention_analysis>/<canonical run name>/, which is right for generating them and wrong for
# reading them as a set. This copies the ones worth looking at into a flat folder with names
# that sort into arm order.
#
#   bash scripts/collect_mv_figures.sh [DEST]
set -uo pipefail
M="${MODE_OUT:-/gscratch/robotics/harine/mode_analysis}"
A="${ATTN_OUT:-/gscratch/robotics/harine/attention_analysis}"
DEST="${1:-$M/figures_tgoal_moving}"
STEP="${STEP:-100000}"
RUNS="${RUNS:-*split-mv*}"
mkdir -p "$DEST"

# cross-arm: every arm in one panel each, per decision state
for f in "$M"/slot_grid_step*.png; do [ -f "$f" ] && cp -f "$f" "$DEST/"; done
# cross-arm figures are GENERATED here, not copied from the analysis roots: those roots hold
# every generation's figures under the same names, and the copy silently pulled in the blq137
# ones. --runs restricts the axes to this generation.
python mode_analysis/compare_arms.py --dir "$M" --runs "$RUNS" --step "$STEP" \
    --out "$DEST/compare_arms_$((STEP/1000))k.png" >/dev/null
python attention_analysis/compare_arms.py --dir "$A" --runs "$RUNS" --step "$STEP" \
    --states test-windows --out "$DEST/compare_arms_d7_$((STEP/1000))k.png" >/dev/null

# per-arm: dispersion curve + the per-slot cloud panels at STEP
for d in "$M"/*demos-176*/ "$M"/split-mv_*/; do
    [ -d "$d" ] || continue
    run=$(basename "$d")
    short=$(sed -E 's/split-mv_demos-176_//; s/_ver-t_goal//; s/_enc-resnet18//; s/_seed-42//;
                    s/^son-none_value_/uniform_/; s/^son-(flat|ramp)([0-9to]+)_value_/\1\2_/;
                    s/unetbc.*/BC/' <<<"$run")
    for f in "$d/dispersion_step${STEP}.png" "$d"/modes_step${STEP}_state*.png; do
        [ -f "$f" ] && cp -f "$f" "$DEST/${short}__$(basename "$f")"
    done
done
n=$(ls -1 "$DEST" | wc -l)
echo "collected $n figures -> $DEST"
ls -1 "$DEST" | head -12
[ "$n" -gt 12 ] && echo "  ... and $((n-12)) more"
