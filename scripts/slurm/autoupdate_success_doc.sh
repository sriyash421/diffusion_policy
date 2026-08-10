#!/bin/bash
# Regenerate SUCCESS_RATES.md every INTERVAL seconds, so the doc tracks the watchers
# without anyone having to remember. Bounded (default 24h) rather than forever -- an
# unbounded detached loop outlives the experiment and silently rewrites the file weeks
# later. Idempotent: the generator reads only on-disk eval output.
#
#   setsid nohup bash scripts/slurm/autoupdate_success_doc.sh > /dev/null 2>&1 &
#
# INTERVAL / HOURS override the defaults. Log: /gscratch/robotics/harine/slurm_logs/.
set -uo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone

INTERVAL=${INTERVAL:-1800}
HOURS=${HOURS:-24}
PY=/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python
LOG=/gscratch/robotics/harine/slurm_logs/success_doc_autoupdate.log

deadline=$(( $(date +%s) + HOURS * 3600 ))
echo "[$(date '+%F %T')] start: every ${INTERVAL}s for ${HOURS}h" >> "$LOG"
while [ "$(date +%s)" -lt "$deadline" ]; do
    out=$("$PY" scripts/build_success_rates_doc.py 2>&1 | tail -1)
    n_tr=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "^tr_" || true)
    # sel_ (the argmax/softmax selection sweep) and fill_ (fill_eval_gaps.sh) count too:
    # they write the same success_curves.jsonl files this doc is generated from, so a log
    # line that omitted them read as "no evals running" while 57 were.
    n_ev=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -cE "^(ev_|bon_|sel_|fill_)" || true)
    echo "[$(date '+%F %T')] $out | training=$n_tr evals=$n_ev" >> "$LOG"
    sleep "$INTERVAL"
done
echo "[$(date '+%F %T')] done" >> "$LOG"
