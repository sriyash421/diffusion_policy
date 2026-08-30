#!/bin/bash
# Echo "ACCOUNT PARTITION" for robotics/weirdlab partitions that can actually RUN the job
# right now (per `hyakalloc`), best first: robotics before weirdlab, then most free GPUs.
#
#   read A P < <(bash scripts/slurm/pick_gpu.sh)
#   sbatch --account=$A --partition=$P scripts/slurm/train_pusht_search.sbatch
#
# A FREE GPU IS NOT ENOUGH. The account quotas are per-resource, so a partition can show a
# free GPU while its CPU or memory allowance is exhausted -- and a job sent there sits in
# AssocGrpCpuLimit indefinitely instead of failing. Filter on all three against what
# train_pusht_search.sbatch actually requests; override with NEED_CPUS / NEED_MEM_G / NEED_GPUS
# if a caller's sbatch asks for something else.
#
# Every candidate is printed, best first, not just the winner. `read A P` still takes the
# best line, so single-job callers are unchanged; a caller submitting SEVERAL jobs at once
# should walk the list instead of calling this once per job. hyakalloc reports the
# scheduler's view, which does not reflect a job submitted seconds ago, so N calls in a row
# return the same answer N times and pile every job onto one partition.
set -euo pipefail

NEED_CPUS="${NEED_CPUS:-8}"
NEED_MEM_G="${NEED_MEM_G:-96}"
NEED_GPUS="${NEED_GPUS:-1}"

lines=$(hyakalloc | sed 's/│/|/g' | awk -F'|' -v nc="$NEED_CPUS" -v nm="$NEED_MEM_G" -v ng="$NEED_GPUS" '
  {
    for (i = 1; i <= NF; i++) { gsub(/^ +| +$/, "", $i) }
    if ($2 != "") acct = $2
    if ($3 != "") part = $3
    cpus = $4 + 0; mem = $5 + 0; gpus = $6 + 0; tag = $7
    if (tag == "FREE" && gpus >= ng && cpus >= nc && mem >= nm &&
        (acct == "robotics" || acct == "weirdlab")) {
      pri = (acct == "robotics" ? 0 : 1)
      print pri, gpus, acct, part
    }
  }' | sort -k1,1n -k2,2nr)

if [ -z "$lines" ]; then
  echo "pick_gpu: no robotics/weirdlab partition has ${NEED_GPUS} GPU + ${NEED_CPUS} CPUs + ${NEED_MEM_G}G free" >&2
  exit 1
fi
echo "$lines" | awk '{print $3, $4}'
