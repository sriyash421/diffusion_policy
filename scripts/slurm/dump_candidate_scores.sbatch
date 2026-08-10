#!/bin/bash
# Dump per-candidate verifier scores for ONE checkpoint.
#
#   sbatch --account=robotics --partition=ckpt \
#          --job-name=cand_<arm> \
#          --export=ALL,CKPT=<...>,ARM=<label>,OUTDIR=<...> \
#          scripts/slurm/dump_candidate_scores.sbatch [extra flags...]
#
# Read-only with respect to every training run: it loads a checkpoint and writes only into
# OUTDIR, so it is safe to run against a directory a live trainer is still writing to.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
set -euo pipefail

source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
cd /mmfs1/home/harine/diffusion_policy_standalone

: "${CKPT:?set CKPT}"
: "${ARM:?set ARM}"
: "${OUTDIR:?set OUTDIR}"

echo "arm=$ARM ckpt=$CKPT out=$OUTDIR"
PYTHONPATH=$PWD python scripts/dump_candidate_scores.py \
    -c "$CKPT" --arm "$ARM" --out-dir "$OUTDIR" --device cuda:0 "$@"
