#!/bin/bash
#SBATCH --job-name=pusht_search_train
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0
#SBATCH --output=/mmfs1/home/harine/slurm_logs/%x-%j.out
# --time=0 == no limit (partitions are MaxTime=UNLIMITED). Do NOT omit --time: omitting
# it applies the partition DefaultTime (1h) and the run gets killed at 1h.
# --account / --partition are NOT hardcoded -- pass them at submit time from pick_gpu.sh
# so the job lands on whatever robotics/weirdlab GPU is free (and 1 GPU + 8 CPUs + 96G
# leaves the node's other GPUs usable):
#
#   read A P < <(bash scripts/pick_gpu.sh)
#   sbatch --account=$A --partition=$P scripts/train_pusht_search.sbatch
#
# CONFIG_NAME selects the run (default = the scalar-verifier baseline). The two
# search-context ablations need --export so the variable reaches the job:
#
#   CONFIG_NAME=train_pusht_diffusion_search_subgoal \
#     sbatch --account=$A --partition=$P --export=ALL,CONFIG_NAME \
#            scripts/train_pusht_search.sbatch
#   CONFIG_NAME=train_pusht_diffusion_search_subgoal_verifier \
#     sbatch --account=$A --partition=$P --export=ALL,CONFIG_NAME \
#            scripts/train_pusht_search.sbatch
set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-train_pusht_diffusion_search}"

source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
cd /mmfs1/home/harine/diffusion_policy_standalone

python train.py \
  --config-name="$CONFIG_NAME" \
  training.device=cuda:0 \
  "$@"
