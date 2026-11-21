#!/bin/bash
#SBATCH --job-name=ev_dt_rounds
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Per-start eval of a context-rounds DT run (replay + own, train + test), then the per-start plot
# against the U-Net BC full300 at its test-best step.
#   sbatch --account=ckpt-robotics --partition=ckpt-all --export=ALL,RUN_DIR=<run> scripts/slurm/eval_dt_rounds.sbatch
set -eo pipefail
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate vae_pushT_l2s
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/diffusion_policy_standalone_pushT}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
BC_RUN=${BC_RUN:-/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search_imgonly/unet_bc/unetbc-full300_ver-t_goal_enc-resnet18_demos-50_split-blq50far_seed-42}
BC_STEP=${BC_STEP:-50000}
python eval_dt_rounds_pusht.py "$RUN_DIR"
python scripts/plot_dt_rounds.py "$RUN_DIR" --bc-run "$BC_RUN" --bc-step "$BC_STEP"
