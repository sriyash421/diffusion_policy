#!/bin/bash
#SBATCH --job-name=pusht_search_eval
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/mmfs1/home/harine/slurm_logs/%x-%j.out
# Eval watcher on the preemptible checkpoint partition (idle GPUs), so the guaranteed
# robotics/weirdlab GPUs stay free for training. --requeue re-launches after preemption.
#
# Usage: sbatch scripts/eval_watch_pusht_search.sbatch <TRAIN_OUTPUT_DIR>
#   (the training run's output dir, which contains checkpoints/step_*.ckpt)
set -euo pipefail

RUN_DIR="${1:?usage: sbatch eval_watch_pusht_search.sbatch <TRAIN_OUTPUT_DIR>}"

source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
cd /mmfs1/home/harine/diffusion_policy_standalone

python eval_search_pusht.py \
  --watch \
  --run-dir "$RUN_DIR" \
  --device cuda:0 \
  --wandb --wandb-entity l2sml --wandb-project pushT_diffusion_search
