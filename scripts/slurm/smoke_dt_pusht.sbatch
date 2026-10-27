#!/bin/bash
#SBATCH --job-name=dt_smoke
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Smoke tests for the full-trajectory arms, on a compute node:
#   1. eval_dt_pusht.py on the finished l2s_online reference checkpoint (2 attempts, 4 episodes)
#   2. train_pusht_dt_bc.yaml for a few debug steps
#   3. train_pusht_online_search.yaml for a few debug steps
# Each step runs even if the previous failed; read the log for the three VERDICT lines.
set -uo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone_pushT
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate "${DP_CONDA_ENV:-vae_pushT_l2s}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export WANDB_MODE=offline HYDRA_FULL_ERROR=1
export DP_OUTPUT_ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/_dt_smoke

echo "=== 1. eval harness on l2s_online reference ckpt"
python eval_dt_pusht.py /gscratch/robotics/harine/l2s_online_ref/data/outputs/online_search_train/checkpoints/latest.ckpt \
  --n 2 --n-envs 4 --max-episodes 4 --split test --output-dir "$DP_OUTPUT_ROOT/eval_smoke_online" \
  && echo "VERDICT 1: eval harness OK" || echo "VERDICT 1: eval harness FAILED"

echo "=== 2. DT k=1 debug training"
python train.py --config-name=train_pusht_dt_bc training.debug=True training.max_gradient_steps=6 \
  training.val_every_steps=3 training.checkpoint_every=6 dataloader.num_workers=2 val_dataloader.num_workers=2 \
  logging.mode=offline \
  && echo "VERDICT 2: DT k=1 debug OK" || echo "VERDICT 2: DT k=1 debug FAILED"

echo "=== 3. online-search debug training"
python train.py --config-name=train_pusht_online_search training.debug=True dataloader.num_workers=2 \
  val_dataloader.num_workers=2 logging.mode=offline \
  && echo "VERDICT 3: online debug OK" || echo "VERDICT 3: online debug FAILED"
