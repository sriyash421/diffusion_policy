#!/bin/bash
# Train the L2S maze models on each split (1 seed, config-default budgets).
#   A  l2s_maze_search   corrupt_obs=false  ST gaussian: BoN, Ours
#   B  l2s_maze_search   corrupt_obs=true   ST gaussian: FuzzyBoN
#   U  l2s_maze_bc_unet  corrupt_obs=false  BC UNet: BoN
#   UF l2s_maze_bc_unet  corrupt_obs=true   BC UNet: FuzzyBoN
#   E  l2s_maze_dt_search                   ST diffusion: Ours
# (C/D, the transformer BC in l2s_maze_dt.yaml, are no longer part of the report.)
# E (slow) runs in its own queue, concurrently with the rest.
# Outputs: data/outputs/l2s_maze/<split>/l2s_<X>/  (logs: data/outputs/l2s_maze/logs/)
# Finished runs leave a .done marker and are skipped.
#
#   [MODELS="U UF"] bash scripts/l2s_maze/launch_train.sh [--dry-run] [splits...]
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/harine/miniconda3/envs/robodiff2/bin/python}
MODELS=${MODELS:-A B U UF E}
DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
if [ $# -gt 0 ]; then SPLITS=("$@"); else SPLITS=(as_is inner_outer quadrant); fi
LOGS=data/outputs/l2s_maze/logs
mkdir -p "$LOGS"

train() {  # split model config corrupt
    local split=$1 model=$2 config=$3 corrupt=$4
    [[ " $MODELS " == *" $model "* ]] || return 0
    local name=l2s_$model
    local cmd=("$PY" train.py --config-name="$config" maze_split="$split" name="$name")
    [ -n "$corrupt" ] && cmd+=(policy.corrupt_obs="$corrupt")
    if [ -f "data/outputs/l2s_maze/$split/$name/.done" ]; then
        echo "skip $split/$name (done)"; return
    fi
    echo "$(date '+%F %T') start $split/$name: ${cmd[*]}"
    [ $DRY = 1 ] && return
    if "${cmd[@]}" > "$LOGS/${split}_${name}.log" 2>&1; then
        touch "data/outputs/l2s_maze/$split/$name/.done"
        echo "$(date '+%F %T') done  $split/$name"
    else
        echo "$(date '+%F %T') FAIL  $split/$name (see $LOGS/${split}_${name}.log)"
    fi
}

queue_main() {
    for s in "${SPLITS[@]}"; do
        train "$s" A  l2s_maze_search false
        train "$s" B  l2s_maze_search true
        train "$s" U  l2s_maze_bc_unet false
        train "$s" UF l2s_maze_bc_unet true
    done
}

queue_e() {
    for s in "${SPLITS[@]}"; do
        train "$s" E l2s_maze_dt_search ""
    done
}

queue_main &
queue_e &
wait
