#!/bin/bash
#SBATCH --account=robotics
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=leak_probe
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30c
cd /gscratch/robotics/harine/codex_audits/maze_interp_20261105
/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python -u leak_audit.py probe 2>&1 | grep -v Warning
