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
# ACCOUNTS (space-separated, default "robotics weirdlab") picks which accounts are
# considered, in priority order, e.g. ACCOUNTS=cse. A non-default account may carry its own
# per-job wall limit (cse: 1 day), so pair it with a TIME_LIMIT and the USR1 requeue flags.
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
ACCOUNTS="${ACCOUNTS:-robotics weirdlab}"

lines=$(hyakalloc | sed 's/│/|/g' | awk -F'|' -v nc="$NEED_CPUS" -v nm="$NEED_MEM_G" -v ng="$NEED_GPUS" -v accts="$ACCOUNTS" '
  BEGIN { na = split(accts, al, " "); for (k = 1; k <= na; k++) rank[al[k]] = k - 1 }
  {
    for (i = 1; i <= NF; i++) { gsub(/^ +| +$/, "", $i) }
    if ($2 != "") acct = $2
    if ($3 != "") part = $3
    cpus = $4 + 0; mem = $5 + 0; gpus = $6 + 0; tag = $7
    if (tag == "FREE" && gpus >= ng && cpus >= nc && mem >= nm &&
        (acct in rank)) {
      pri = rank[acct]
      print pri, gpus, acct, part
    }
  }' | sort -k1,1n -k2,2nr)

if [ -z "$lines" ]; then
  echo "pick_gpu: no $ACCOUNTS partition has ${NEED_GPUS} GPU + ${NEED_CPUS} CPUs + ${NEED_MEM_G}G free" >&2
  exit 1
fi
echo "$lines" | awk '{print $3, $4}'
