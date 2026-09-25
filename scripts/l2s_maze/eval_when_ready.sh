#!/bin/bash
# For each split, wait until all of its training runs are .done, then run its eval sweep.
#   bash scripts/l2s_maze/eval_when_ready.sh [splits...]
set -uo pipefail
cd "$(dirname "$0")/../.."
if [ $# -gt 0 ]; then SPLITS=("$@"); else SPLITS=(as_is inner_outer quadrant); fi
for s in "${SPLITS[@]}"; do
    for m in A B U UF E; do
        until [ -f "data/outputs/l2s_maze/$s/l2s_$m/.done" ]; do sleep 60; done
    done
    echo "$(date '+%F %T') $s: all checkpoints ready"
    bash scripts/l2s_maze/launch_eval.sh "$s"
done
/home/harine/miniconda3/envs/robodiff2/bin/python scripts/l2s_maze/summarize.py | tail -1
