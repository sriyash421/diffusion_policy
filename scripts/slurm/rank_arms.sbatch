#!/bin/bash
# Pool all five arms' predictions into ONE verifier ranking, split by whether the demo's
# chunk moves the block. One task per checkpoint step.
#
#   unset WANDB_API_KEY
#   sbatch --account=ckpt-robotics --partition=ckpt --array=0-2 scripts/slurm/rank_arms.sbatch
#
# Read-only against every training run: loads checkpoints, writes only under OUTDIR.
# --mem is high because five policies are resident at once and the BC checkpoint alone is
# ~4.4GB. --time is finite so a job cannot be left straddling a maintenance reservation.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --job-name=rankarms
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%A_%a.out
set -euo pipefail

set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate "${DP_CONDA_ENV:-vae_pushT_l2s}"
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

STEPS=(10000 50000 100000)
# STEP_OVERRIDE names the step directly, for sweeps that are not one of the three below.
# Without it the OUT filename is built from the array entry, so a job asked for step 20000 via
# an extra --step arg still wrote step_0100000.json -- a file whose name contradicts its
# contents, which is worse than no file.
STEP="${STEP_OVERRIDE:-${STEPS[${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}]}}"
OUTDIR="${OUTDIR:-analysis/arm_ranking}"
OUT="$OUTDIR/$(printf 'step_%07d.json' "$STEP")"

echo "step=$STEP -> $OUT"
PYTHONPATH=$PWD python -u scripts/rank_arms_by_blockmotion.py \
    --step "$STEP" --device cuda:0 --episodes "${EPISODES:-50}" \
    --batch "${BATCH:-32}" --out "$OUT" "$@"
