#!/bin/bash
# Smoke for pg_dagger.py (stage 2, different mazes): levels (disjoint 800 / 200 layouts), expert reaches the goal on 300 random levels,
# tiny train -> evalctx for window + reward (context) and ddpm + bfs (no context).
#SBATCH --account=robotics
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --job-name=pg_smoke
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_procgen_ast_20261121/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg PYTHONPATH=/gscratch/robotics/harine/L2S
unset MAZE_SETTING
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
cd /gscratch/robotics/harine/codex_audits/maze_procgen_ast_20261121
echo "== window, reward, context"
$PY -u pg_dagger.py smoke --obs window --fb reward 2>&1 | grep -v Warning | tail -12
echo "== ddpm t=400, bfs, no context"
$PY -u pg_dagger.py smoke --obs ddpm --obs-t 400 --fb bfs --no-context 2>&1 | grep -v Warning | tail -8
rm -rf runs/pg_smoke
echo PG_SMOKE_ALL_OK
