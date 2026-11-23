#!/bin/bash
# Stage 2 (different mazes): pg_dagger.py train then evalctx. Resumes per round from state.pt after preemption.
#   sbatch --job-name=<tag> pg_run.sbatch <tag> <fb: reward|bfs|none> [--obs window | --obs ddpm --obs-t 400] [--no-context]
#SBATCH --account=robotics
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40|l40s|a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_procgen_ast_20261121/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg PYTHONPATH=/gscratch/robotics/harine/L2S
unset MAZE_SETTING
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
PG=/gscratch/robotics/harine/codex_audits/maze_procgen_ast_20261121
tag=$1; fb=$2; shift 2; cd $PG
echo "== start $(date) on $(hostname) restart=${SLURM_RESTART_COUNT:-0} tag=$tag fb=$fb flags=$*"
[[ -f runs/$tag/TRAINED ]] || $PY -u pg_dagger.py train --tag "$tag" --fb "$fb" "$@" 2>&1 | grep -v Warning | grep -E "ROUND|resumed|Traceback|Error"
[[ -f runs/$tag/TRAINED ]]
[[ -f runs/$tag/eval_ctx.npz ]] || $PY -u pg_dagger.py evalctx --tag "$tag" --seeds 3 --attempts 8 2>&1 | grep -v Warning
echo "== done $(date)"
