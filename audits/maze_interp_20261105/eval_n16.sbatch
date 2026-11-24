#!/bin/bash
# 16 attempts in a row from each (start, goal), final-round policy; context = the run's eval window (r8c2 runs: the last 2 attempts,
# a sliding window; no-context runs: none). -> runs/<tag>/eval_ctx_n16.npz.  Usage: sbatch eval_n16.sbatch <tag> <interp30c|11brc|pg>
#SBATCH --account=robotics
#SBATCH --partition=ckpt-all
#SBATCH --constraint=a40|l40|l40s|a100|h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
A=/gscratch/robotics/harine/codex_audits; R=$A/maze_rounds_20261103
export PYTHONPATH=/gscratch/robotics/harine/L2S:$R
tag=$1; export MAZE_SETTING=$2
if [[ $MAZE_SETTING = pg ]]; then   # new mazes: 640 train levels / 1000 held-out-maze levels x 3 seeds
  unset MAZE_SETTING; cd $A/maze_procgen_ast_20261121; echo "== start $(date) on $(hostname) restart=${SLURM_RESTART_COUNT:-0} tag=$tag setting=pg"
  /gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python -u pg_dagger.py evalctx --tag "$tag" --seeds 3 --attempts 16 --eval-rounds -1 --out _n16 2>&1 | grep -v Warning
  echo "== done $(date)"; exit 0
fi
case $MAZE_SETTING in interp30c) TEST="--test-set all --eval-test-seeds 3";; 11brc) TEST="";; *) echo bad setting; exit 1;; esac
cd $R
echo "== start $(date) on $(hostname) restart=${SLURM_RESTART_COUNT:-0} tag=$tag setting=$MAZE_SETTING"
/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python -u asteroid.py evalctx --tag "$tag" --seeds 8 --attempts 16 \
    --eval-rounds -1 --splits train,test --out _n16 $TEST 2>&1 | grep -v Warning
echo "== done $(date)"
