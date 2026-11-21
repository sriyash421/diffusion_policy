#!/bin/bash
#SBATCH --account=robotics
#SBATCH --partition=ckpt-all
#SBATCH --constraint=a40|a100|l40|l40s|h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
export OMP_NUM_THREADS=4
ROOT=/mmfs1/gscratch/robotics/harine/codex_audits/maze_quadrants_delta_20261019
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
train_task=$((SLURM_ARRAY_TASK_ID / 10))
step=$(((SLURM_ARRAY_TASK_ID % 10 + 1) * 1000))
read -r setting algo seed variant < <(sed -n "$((train_task+1))p" "$ROOT/train.tsv")
printf -v step_tag '%07d' "$step"
checkpoint="$ROOT/runs/$setting/${algo}__${variant}/seed_${seed}/step_${step_tag}.pt"
if [[ ! -f "$checkpoint" ]]; then echo "Checkpoint missing: $checkpoint"; exit 42; fi
"$PY" -u "$ROOT/evaluate.py" --checkpoint "$checkpoint"
