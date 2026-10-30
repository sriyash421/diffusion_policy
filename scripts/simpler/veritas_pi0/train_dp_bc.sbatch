#!/bin/bash
#SBATCH --job-name=dp_bc
#SBATCH --account=ckpt-robotics
#SBATCH --partition=ckpt,ckpt-all
#SBATCH --qos=ckpt-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40s|l40|a100|h200
#SBATCH --requeue
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/pi0_veritas/%x-%j.out
#SBATCH --error=/gscratch/robotics/harine/slurm_logs/pi0_veritas/%x-%j.out
# Diffusion-policy BC (UNet, H=16, n_obs_steps=1, 20k steps) on one task's 100-demo split: the
# exact recipe of position_holdout_20261009/eggplant100/train/ph_eggplant_100_bc, for the
# "BC on the offline demos" curve of the N sweep. Resumable (training.resume=true, --requeue).
#   TASK=carrot|stack|eggplant sbatch --export=ALL,TASK=carrot scripts/simpler/veritas_pi0/train_dp_bc.sbatch
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/diffusion_policy_standalone_simpler}"
cd "$REPO_ROOT"
TASK="${TASK:?carrot|stack|eggplant}"
PH=/gscratch/robotics/harine/RoboMonkey_data/experiments/position_holdout_20261009
case "$TASK" in
  carrot)   DATASET_DIR=/gscratch/robotics/harine/data/carrot_on_plate_strict
            SPLIT_FILE="$REPO_ROOT/diffusion_policy/config/splits/simpler/carrot/random_train100.json" ;;
  stack)    DATASET_DIR=/gscratch/robotics/harine/data/stack_cube_strict_split
            SPLIT_FILE="$PH/splits/stack/position_holdout_train100.json" ;;
  eggplant) DATASET_DIR=/gscratch/robotics/harine/data/strict_distance_success_split
            SPLIT_FILE="$PH/splits/eggplant/position_holdout_train100.json" ;;
  *) echo "unknown TASK $TASK" >&2; exit 1 ;;
esac
RUN_NAME="${RUN_NAME:-dp_${TASK}_100_bc}"
RUN_DIR="/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2/$TASK/dp_train/$RUN_NAME"
mkdir -p "$RUN_DIR" /gscratch/robotics/harine/slurm_logs/pi0_veritas
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dqc_jax
export PYTHONPATH="$PWD:${PYTHONPATH:-}" DIFFUSION_POLICY_ROOT="$PWD"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DISPLAY=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "node=$(hostname -s) job=$SLURM_JOB_ID task=$TASK run=$RUN_DIR"
python -u diffusion_policy/workspace/train_diffusion_unet_image_workspace.py \
    --config-name=robomonkey_eggplant_bc \
    horizon=16 n_obs_steps=1 \
    training.num_epochs=100000 +training.max_gradient_steps=20000 \
    training.lr_scheduler=cosine_then_constant +training.lr_decay_steps=40000 +training.lr_min_ratio=0.1 \
    training.lr_warmup_steps=500 training.checkpoint_every=1000 training.resume=true \
    hydra.run.dir="$RUN_DIR" \
    task.dataset.dataset_dir="$DATASET_DIR" task.dataset.split_file="$SPLIT_FILE" task.dataset.use_cache=true \
    training.seed=42 dataloader.num_workers=4 val_dataloader.num_workers=2 \
    logging.name="$RUN_NAME" logging.id="$RUN_NAME" logging.project=simpler_veritas_h16 \
    logging.group=step2_n_sweep_dp +logging.entity=l2sml logging.resume=allow
echo "[sbatch] done: $RUN_DIR/checkpoints/"
