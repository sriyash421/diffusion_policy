#!/bin/bash
#SBATCH --job-name=pusht_search_eval
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# logs on /gscratch, not home: home is a 10G hard quota (see README_pusht.md)
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Eval watcher on the preemptible checkpoint partition (idle GPUs), so the guaranteed
# robotics/weirdlab GPUs stay free for training. --requeue re-launches after preemption.
#
# Usage: sbatch scripts/slurm/eval_watch_pusht_search.sbatch <TRAIN_OUTPUT_DIR> [extra eval args]
#   (the training run's output dir, which contains checkpoints/step_*.ckpt)
#
# Extra args are forwarded to eval_search_pusht.py -- e.g. `--min-n 1 --max-n 1` for the BC
# baselines, whose per-checkpoint number is the n=1 rollout. Without the `shift` + "$@"
# below they were silently DROPPED, which is how six large-n evals once ran at the default
# n<=64 and wasted ~1h of GPU each.
set -euo pipefail

RUN_DIR="${1:?usage: sbatch eval_watch_pusht_search.sbatch <TRAIN_OUTPUT_DIR> [eval args]}"
shift

# see eval_ckpt_pusht_search.sbatch: conda's deactivate hook reads an unset var,
# which kills the job under `set -u` if sbatch was run from an active robodiff shell
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
# The environment. `vae_pushT_l2s` (conda_environment_vae_pusht.yaml), NOT `robodiff`:
# robodiff cannot import AutoencoderKL at all -- it has diffusers 0.36 against accelerate
# 0.13.2 -- so every SD-VAE arm dies at startup there with 0 steps. Both envs carry the same
# diffusers, so the DDIM/DDPM sampling is unchanged and ResNet-era checkpoints evaluate
# identically under either. Override DP_CONDA_ENV to pin a different one.
DP_CONDA_ENV="${DP_CONDA_ENV:-vae_pushT_l2s}"
conda activate "$DP_CONDA_ENV"
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

python eval_search_pusht.py \
  --watch \
  --run-dir "$RUN_DIR" \
  --device cuda:0 \
  "$@"
