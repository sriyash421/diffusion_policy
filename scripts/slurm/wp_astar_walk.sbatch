#!/bin/bash
# a* rank + NO-TOUCH discrimination of ONE checkpoint under the three PushT verifier
# values {t_goal, wp_v3, wp_v5}, one array task per value.
#
#   unset WANDB_API_KEY
#   CKPT=/path/to/step_0060000.ckpt sbatch --account=robotics --partition=ckpt \
#       --array=0-2 scripts/slurm/wp_astar_walk.sbatch
#
# Read-only against every training run: it loads the checkpoint and writes only under
# ANALYSIS_ROOT. Meant for the 176-demo BC expert (train 176 / val 0 / test 30): the wp
# values need a plan per walked episode and the committed plans cover exactly that test
# split (media/veritas_pusht/{v3_end_outside,v5_pusher_and_t}/ep*.json).
#
# --time is finite on purpose: a maintenance reservation can start while these queue, and
# a job that would run past it never starts.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --job-name=wp_astar
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%A_%a.out
set -euo pipefail

set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate "${DP_CONDA_ENV:-vae_pushT_l2s}"
set -u
# The repo to run from: the sbatch submit dir (every submit path runs from the repo root;
# $0 is useless under sbatch, it is the spooled copy in /var/spool), else this file's own
# repo when run outside SLURM. DP_REPO overrides both.
cd "${DP_REPO:-${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")/../..}}"

CKPT="${CKPT:?set CKPT=/path/to/step_XXXXXXX.ckpt}"
VALUES=(t_goal wp_v3 wp_v5)
IDX="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
VALUE="${VALUES[$IDX]}"

RUN="$(basename "$(dirname "$(dirname "$CKPT")")")"
STEP="$(basename "$CKPT" .ckpt)"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/wp_verifier}"
OUT="$ANALYSIS_ROOT/$RUN/${STEP}_${VALUE}.json"
mkdir -p "$(dirname "$OUT")"

echo "task $IDX: value=$VALUE run=$RUN $STEP"
echo "  ckpt $CKPT"
echo "  out  $OUT"
test -f "$CKPT" || { echo "MISSING CHECKPOINT"; exit 1; }

PYTHONPATH=$PWD python scripts/astar_uniform_walk.py \
    -c "$CKPT" --arm "${ARM:-bc-176}" --device cuda:0 --split test \
    --episodes "${EPISODES:-30}" --n "${NCAND:-16}" --viz "${VIZ:-10}" \
    --verifier-value "$VALUE" --out "$OUT" "$@"
