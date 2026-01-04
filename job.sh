#!/bin/bash
#SBATCH --job-name=pg_pomdp_transformer
#SBATCH --qos=normal
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/projects/weirdlab/sriyash/slurm_logs/slurm-%x-%j.out

cd /gpfs/projects/weirdlab/sriyash/diffusion_policy
source .venv/bin/activate

# export SCRUBBED_PATH="/gpfs/scrubbed/sriyash"
# export UV_CACHE_DIR="$SCRUBBED_PATH/.cache/uv"
# export OPENPI_DATA_HOME=$SCRUBBED_PATH
export HF_HOME="$SCRUBBED_PATH/huggingface"
export HF_DATASETS_CACHE="$SCRUBBED_PATH/hf_datasets_cache"

module load conda
conda activate robodiff

dataset="/gpfs/scrubbed/sriyash/pg_no_randomization_5k"

python train.py \
    --config-name $1 \
    --config-dir diffusion_policy/config \
    task.dataset.dataset_dir=$dataset


