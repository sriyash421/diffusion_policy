#!/bin/bash
#SBATCH --job-name=pusht_keepalive
#SBATCH --account=ckpt-robotics
#SBATCH --partition=ckpt
#SBATCH --qos=ckpt
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=9:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --export=NONE
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/pusht_keepalive-%j.out
# One CPU job for the PushT results pipeline, self-requeueing every ~9 h:
#   export_live_results --watch  -> live_reports/*.csv from new success_curve.json files
#   monitor_blq_report.py        -> republishes the W&B report when those results change
#   pusht_keepalive.py REGISTRY  -> resubmits tracked training runs / eval watchers that died
#
#   sbatch scripts/slurm/pusht_keepalive.sbatch   (from the repo root)
set -eo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB__SERVICE_WAIT=300
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
REPORTS=/mmfs1/home/harine/Documents/Codex/2026-10-04/for-push-t-st-k1-and/work
REGISTRY=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/keepalive/registry.json
cd /mmfs1/home/harine/diffusion_policy_standalone_pushT

$PY -u scripts/export_live_results.py --watch & P1=$!
$PY -u "$REPORTS/monitor_blq_report.py" & P2=$!
$PY -u scripts/pusht_keepalive.py "$REGISTRY" & P3=$!
trap 'echo "[$(date -Is)] USR1: requeueing ${SLURM_JOB_ID}"; kill -TERM $P1 $P2 $P3 2>/dev/null || true; wait || true;
      scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
wait
