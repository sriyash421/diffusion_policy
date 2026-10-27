#!/bin/bash
#SBATCH --job-name=dt_resume_check
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Resume correctness for the GPT-2 DT arms, via scripts/slurm/test_resume.sbatch's assertions:
#   1. a k=8 debug run (2 fresh-context updates, checkpoint every step) under _dt_smoke
#   2. resume that run for 2 more updates
#   3. resume the k=1 debug run (from smoke_dt_gpt2.sbatch, 6 steps) for 6 more
set -uo pipefail
REPO=/mmfs1/home/harine/diffusion_policy_standalone_pushT
cd "$REPO"
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate "${DP_CONDA_ENV:-vae_pushT_l2s}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export WANDB_MODE=offline HYDRA_FULL_ERROR=1
export DP_OUTPUT_ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/_dt_smoke
ROOT=$DP_OUTPUT_ROOT/pusht_search/pusht_image_search_imgonly/dt_search

echo "=== 1. k=8 debug run, 2 updates"
python train.py --config-name=train_pusht_dt_search training.debug=True training.max_gradient_steps=2 \
  training.checkpoint_every=1 training.val_every_steps=1000 dataloader.num_workers=2 val_dataloader.num_workers=2 \
  logging.mode=offline \
  && echo "VERDICT 1: k=8 debug OK" || echo "VERDICT 1: k=8 debug FAILED"

echo "=== 2. resume k=8 for 2 more updates"
REPO=$REPO SRC=$(ls -d $ROOT/dt-gpt2_k8_*) EXTRA_STEPS=2 CKPT_EVERY=1 CONFIG_NAME=train_pusht_dt_search \
  bash scripts/slurm/test_resume.sbatch training.debug=True training.val_every_steps=1000 dataloader.num_workers=2 \
  && echo "VERDICT 2: k=8 resume OK" || echo "VERDICT 2: k=8 resume FAILED"

echo "=== 3. resume k=1 for 6 more steps"
REPO=$REPO SRC=$(ls -d $ROOT/dt-gpt2_k1_*) EXTRA_STEPS=6 CKPT_EVERY=3 CONFIG_NAME=train_pusht_dt_bc \
  bash scripts/slurm/test_resume.sbatch training.debug=True training.val_every_steps=1000 dataloader.num_workers=2 \
  && echo "VERDICT 3: k=1 resume OK" || echo "VERDICT 3: k=1 resume FAILED"
