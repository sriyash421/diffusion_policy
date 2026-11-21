#!/bin/bash
# smoke of the new asteroid.py, then the context-length eval (train + test starts, 8 attempts, full window) of the given round-design tags.
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40|l40s|a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_rounds_20261103/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=4 MPLBACKEND=Agg PYTHONPATH=/gscratch/robotics/harine/L2S
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
$PY -u asteroid.py smoke 2>&1 | grep -v Warning | tail -25
for t in "$@"; do
  [[ -f runs/$t/ctx_eval_seeds8.npz ]] || $PY -u asteroid.py eval --tag "$t" --splits train,test --seeds 8 --attempts 8 --out ctx_eval_seeds8.npz
  $PY -u plot_clear.py "$t"
done
