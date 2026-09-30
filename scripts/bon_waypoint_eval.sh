#!/usr/bin/env bash
# Best-of-n on ONE BC UNet checkpoint under the VLM-waypoint verifier values, next to
# the t_goal baseline row.
#
#   CKPT=/path/to/step_0100000.ckpt bash scripts/bon_waypoint_eval.sh
#   CKPT=... VALUES="wp_v5" MAX_N=16 bash scripts/bon_waypoint_eval.sh
#
# Intended for the 176-demo expert arm (train_pusht_unet_bc_expert), whose split is the
# ONLY one the plans cover: media/veritas_pusht/{v5_pusher_and_t,v3_end_outside}/ hold one
# ep{idx}.json per train176 TEST episode (scripts/veritas_pusht_overlays.py). The eval
# refuses to start if any evaluated episode has no plan, so a checkpoint from another
# split fails fast rather than scoring a different signal than the row name claims.
#
# --skip-val: the expert split has no val episodes, and the wp rows are REPORTS on a
# checkpoint chosen elsewhere (the watcher's t_goal curve), not something selected here.
# Results land in the run's own bon_search_ver-<value>/ so no row can merge with another.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${CKPT:?set CKPT=/path/to/step_XXXXXXX.ckpt}"
PY="${PY:-/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python}"
VALUES="${VALUES:-t_goal wp_v5 wp_v3}"
MAX_N="${MAX_N:-64}"
N_ENVS="${N_ENVS:-30}"

for d in media/veritas_pusht/v5_pusher_and_t media/veritas_pusht/v3_end_outside; do
  n=$(ls "$d"/ep*.json 2>/dev/null | wc -l)
  echo "plans in $d: $n"
done

for v in $VALUES; do
  echo "=== [$(date -Is)] verifier-value $v ==="
  $PY -u eval_search_pusht.py -c "$CKPT" --skip-val \
    --verifier-value "$v" --min-n 1 --max-n "$MAX_N" --n-envs "$N_ENVS" \
    --store-scores 2>&1 | grep -E "success_rate=|verifier value|waypoint plans|Traceback|Error"
done
