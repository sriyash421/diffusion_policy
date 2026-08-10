#!/bin/bash
# Echo "ACCOUNT PARTITION" for a robotics/weirdlab partition that currently has FREE GPUs
# (per `hyakalloc`), preferring robotics, then the partition with the most free GPUs.
# Used to target training at whatever is actually available instead of hardcoding.
#
#   read A P < <(bash scripts/slurm/pick_gpu.sh)
#   sbatch --account=$A --partition=$P scripts/slurm/train_pusht_search.sbatch
set -euo pipefail

line=$(hyakalloc | sed 's/│/|/g' | awk -F'|' '
  {
    for (i = 1; i <= NF; i++) { gsub(/^ +| +$/, "", $i) }
    if ($2 != "") acct = $2
    if ($3 != "") part = $3
    tag = $7; gpus = $6 + 0
    if (tag == "FREE" && gpus > 0 && (acct == "robotics" || acct == "weirdlab")) {
      pri = (acct == "robotics" ? 0 : 1)
      print pri, gpus, acct, part
    }
  }' | sort -k1,1n -k2,2nr | head -1)

if [ -z "$line" ]; then
  echo "pick_gpu: no free robotics/weirdlab GPUs found" >&2
  exit 1
fi
echo "$line" | awk '{print $3, $4}'
