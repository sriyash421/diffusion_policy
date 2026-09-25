#!/bin/bash
# Eval sweep: every split x (model, arm) x selection rule x N, plus the expert.
# Outputs: data/l2s_maze_eval/<split>/<model>_<arm>/<select>/N<n>.json
#
#   [NS=1,2,4] [SELECTS="argmax last"] bash scripts/l2s_maze/launch_eval.sh [--dry-run] [splits...]
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/harine/miniconda3/envs/robodiff2/bin/python}
NS=${NS:-1,2,4,8,16,32,64}
SELECTS=${SELECTS:-argmax last}
DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
if [ $# -gt 0 ]; then SPLITS=("$@"); else SPLITS=(as_is inner_outer quadrant); fi
LOGS=data/l2s_maze_eval/logs
mkdir -p "$LOGS"

run() {  # split tag args...
    local split=$1 tag=$2; shift 2
    local log="$LOGS/${split}_${tag}.log"
    echo "$(date '+%F %T') $split/$tag: $*"
    [ $DRY = 1 ] && return
    "$PY" scripts/l2s_maze/eval_arms.py --split "$split" "$@" > "$log" 2>&1 \
        || echo "FAIL $split/$tag (see $log)"
    grep "N=\|expert:" "$log"
}

for s in "${SPLITS[@]}"; do
    ck() { echo "data/outputs/l2s_maze/$s/l2s_$1/checkpoints/latest.ckpt"; }
    run "$s" expert --arm expert
    for sel in $SELECTS; do
        common=(--select "$sel" --ns "$NS")
        run "$s" "st_gaussian_bon_$sel"       --model st_gaussian  --arm bon       --ckpt "$(ck A)"  "${common[@]}"
        run "$s" "st_gaussian_fuzzy_bon_$sel" --model st_gaussian  --arm fuzzy_bon --ckpt "$(ck B)"  "${common[@]}"
        run "$s" "st_gaussian_ours_$sel"      --model st_gaussian  --arm ours      --ckpt "$(ck A)"  "${common[@]}"
        run "$s" "bc_unet_bon_$sel"           --model bc_unet      --arm bon       --ckpt "$(ck U)"  "${common[@]}"
        run "$s" "bc_unet_fuzzy_bon_$sel"     --model bc_unet      --arm fuzzy_bon --ckpt "$(ck UF)" "${common[@]}"
        run "$s" "st_diffusion_ours_$sel"     --model st_diffusion --arm ours      --ckpt "$(ck E)"  "${common[@]}"
    done
done
