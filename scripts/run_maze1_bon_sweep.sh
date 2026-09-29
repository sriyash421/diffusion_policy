#!/bin/bash
# Best-of-N sweep for the three maze-1 arms, on the 50 innermost test episodes.
#
# Every arm is scored on the SAME episodes: eval_bon_maze resolves them from each run's
# own split_file, and the manifest checksum lands in each bon.json, so a silent
# repartition between arms shows up as a mismatch rather than as a difference in results.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/robodiff2/bin/python
N_SAMPLES=${N_SAMPLES:-16}
MAX_STEPS=${MAX_STEPS:-600}

for arm in maze1_bc_unet maze1_st_k16 maze1_st_k1; do
    ckpt="data/outputs/$arm/checkpoints/latest.ckpt"
    if [ ! -f "$ckpt" ]; then echo "SKIP $arm (no checkpoint)"; continue; fi
    echo "=== $arm ==="
    $PY eval_bon_maze.py -c "$ckpt" -o "data/outputs/$arm/bon" \
        --n-samples "$N_SAMPLES" --max-steps "$MAX_STEPS" 2>&1 \
        | tee "data/outputs/${arm}_bon.log" | tail -3
done
