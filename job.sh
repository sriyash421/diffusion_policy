#!/bin/bash
#SBATCH --job-name=pg_iter2
#SBATCH --qos=normal
#SBATCH --account=socialrl
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=8:00:00
#SBATCH --output=/gpfs/projects/weirdlab/sriyash/slurm_logs/%x-%j.out
#SBATCH --array=0-0


cd /gpfs/projects/weirdlab/sriyash/diffusion_policy


# export SCRUBBED_PATH="/gpfs/scrubbed/sriyash"
# export UV_CACHE_DIR="$SCRUBBED_PATH/.cache/uv"
# export OPENPI_DATA_HOME=$SCRUBBED_PATH
export HF_HOME="$SCRUBBED_PATH/huggingface"
export HF_DATASETS_CACHE="$SCRUBBED_PATH/hf_datasets_cache"

module load conda
conda activate robodiff2

# task=$1
# iteration=0
# dataset="/gpfs/scrubbed/sriyash/assembly_datasets/${task}_step${iteration}"
# output_dir="data/${task}-seed$SLURM_ARRAY_TASK_ID-iteration${iteration}"
# mkdir -p $output_dir
# python train.py \
#     --config-name in_context_exploration_rgb_base.yaml \
#     --config-dir diffusion_policy/config \
#     output_dir=$output_dir \
#     "task.dataset.dataset_config=[{dataset_dir: $dataset, sampling_ratio: 1.0}]" \
#     name=${task} \
#     exp_name=${task} \
#     logging.project=incontext_exploration_assembly \
#     logging.group=train \
#     optimizer.lr=0.00001 \
#     seed=$SLURM_ARRAY_TASK_ID \
#     iteration=$iteration


# python train.py \
#     --config-name $1 \
#     --config-dir diffusion_policy/config $2

# mkdir -p data/pg_exp_final-seed$SLURM_ARRAY_TASK_ID-iteration0
# mkdir -p data/pg_exp_final-seed$SLURM_ARRAY_TASK_ID-iteration0
# python train.py \
#     --config-name in_context_exploration_rgb_base.yaml \
#     --config-dir diffusion_policy/config \
#     output_dir=data/pg_exp_final-seed$SLURM_ARRAY_TASK_ID-iteration0 \
#     "task.dataset.dataset_config=[{dataset_dir: /gpfs/scrubbed/sriyash/peg_insertion_exploration_step0, sampling_ratio: 1.0}]" \
#     name=pg_exp_final \
#     exp_name=pg_exp_final \
#     logging.project=peg_insertion_exploration_final \
#     logging.group=train \
#     optimizer.lr=0.0001 \
#     seed=$SLURM_ARRAY_TASK_ID \
#     iteration=0

# iteration=1
# task=peg_insertion
# output_dir="data/${task}-seed$SLURM_ARRAY_TASK_ID-iteration${iteration}"
# mkdir -p $output_dir
# python train.py \
#     --config-name in_context_exploration_rgb_base.yaml \
#     --config-dir diffusion_policy/config \
#     output_dir=$output_dir \
#     "task.dataset.dataset_config=[{dataset_dir:/gpfs/scrubbed/sriyash/assembly_datasets/peg_insertion_step0,sampling_ratio:0.25},{dataset_dir:/gpfs/scrubbed/sriyash/assembly_datasets/peg_insertion-seed$SLURM_ARRAY_TASK_ID-step1,sampling_ratio:0.75}]" \
#     name=${task} \
#     exp_name=${task} \
#     logging.project=incontext_exploration_assembly \
#     logging.group=train \
#     optimizer.lr=0.00001 \
#     seed=$SLURM_ARRAY_TASK_ID \
#     iteration=$iteration \
#     checkpoint.pretrained_ckpt_path=data/${task}-seed$SLURM_ARRAY_TASK_ID-iteration0/checkpoints/step_0050000.ckpt

iteration=2
task=peg_insertion
output_dir="data/${task}-seed$SLURM_ARRAY_TASK_ID-iteration${iteration}"
mkdir -p $output_dir
python train.py \
    --config-name in_context_exploration_rgb_base.yaml \
    --config-dir diffusion_policy/config \
    output_dir=$output_dir \
    "task.dataset.dataset_config=[{dataset_dir:/gpfs/scrubbed/sriyash/assembly_datasets/peg_insertion_step0,sampling_ratio:0.2},{dataset_dir:/gpfs/scrubbed/sriyash/assembly_datasets/peg_insertion-seed$SLURM_ARRAY_TASK_ID-step1,sampling_ratio:0.3},{dataset_dir:/gpfs/scrubbed/sriyash/assembly_datasets/peg_insertion-seed$SLURM_ARRAY_TASK_ID-step2,sampling_ratio:0.5}]" \
    name=${task} \
    exp_name=${task} \
    logging.project=incontext_exploration_assembly \
    logging.group=train \
    optimizer.lr=0.00001 \
    seed=$SLURM_ARRAY_TASK_ID \
    iteration=$iteration \
    checkpoint.pretrained_ckpt_path=data/${task}-seed$SLURM_ARRAY_TASK_ID-iteration1/checkpoints/step_0050000.ckpt

# iteration=2
# mkdir -p data/peg_insertion_exploration_final-seed$SLURM_ARRAY_TASK_ID-iteration$iteration

# python train.py \
#     --config-name in_context_exploration_rgb_base.yaml \
#     --config-dir diffusion_policy/config \
#     output_dir=data/peg_insertion_exploration_final-seed$SLURM_ARRAY_TASK_ID-iteration$iteration \
#     "task.dataset.dataset_config=[{dataset_dir: /gpfs/scrubbed/sriyash/pg_no_randomization_5k, sampling_ratio: 0.2},{dataset_dir: /gpfs/scrubbed/sriyash/peg_insertion_exploration_final-iteration1-seed$SLURM_ARRAY_TASK_ID, sampling_ratio: 0.3},{dataset_dir: /gpfs/scrubbed/sriyash/peg_insertion_exploration_final-iteration2-seed$SLURM_ARRAY_TASK_ID, sampling_ratio: 0.5}]" \
#     name=peg_insertion_exploration_final \
#     exp_name=peg_insertion_exploration_final \
#     logging.project=peg_insertion_exploration_final \
#     logging.group=train \
#     optimizer.lr=0.00001 \
#     seed=$SLURM_ARRAY_TASK_ID \
#     iteration=$iteration \
#     checkpoint.pretrained_ckpt_path=data/peg_insertion_exploration_final-seed$SLURM_ARRAY_TASK_ID-iteration1/checkpoints/latest.ckpt
    


