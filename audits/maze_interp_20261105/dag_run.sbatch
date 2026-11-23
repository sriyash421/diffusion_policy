#!/bin/bash
# DAgger (asteroid.py --target relabel: the student plays attempt n+1, the continuous expert labels every point it visited) with partial /
# noised observations, continuous maze (2026-11-22 plan). 8 rounds × 30 iterations × batch 128 × 30 steps, constant lr 1e-4, test every
# iteration; resumes per round from state.pt after preemption. Then evalctx: rounds 0 and 7 make 8 attempts in a row (x = candidates in
# context) on train starts / 100 unseen pairs of train goals (8 seeds) / held-out goals (interp30c: all 2870 pairs, 3 seeds; 11brc: the
# 100 held-out-quadrant episodes, 8 seeds).
#   sbatch --job-name=<tag> dag_run.sbatch <tag> <setting: interp30c|11brc> <fb: reward|bfs|none> [train flags: --obs window | --obs ddpm --obs-t 400 | --no-context]
#SBATCH --account=robotics
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S
R=/gscratch/robotics/harine/codex_audits/maze_rounds_20261103
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
tag=$1; export MAZE_SETTING=$2; fb=$3; shift 3
case $MAZE_SETTING in interp30c) RUN=$I/runs/$tag; TEST="--test-set all --eval-test-seeds 3";; 11brc) RUN=$R/runs/$tag; TEST="";;
  *) echo "setting must be interp30c or 11brc"; exit 1;; esac
cd $R
echo "== start $(date) on $(hostname) restart=${SLURM_RESTART_COUNT:-0} tag=$tag setting=$MAZE_SETTING fb=$fb flags=$*"
[[ -f $RUN/TRAINED ]] || $PY -u asteroid.py train --tag "$tag" --target relabel --fb "$fb" --rounds 8 --batch 128 --micro 32 --lr 1e-4 \
    --steps-per-iter 30 --iters 30 --test-every 30 --test-seeds 8 --train-seeds 1 "$@" 2>&1 | grep -v Warning | grep -E "ROUND|resumed|Traceback|Error"
[[ -f $RUN/TRAINED ]]
[[ -f $RUN/eval_ctx.npz ]] || $PY -u asteroid.py evalctx --tag "$tag" --seeds 8 --attempts ${EVAL_ATTEMPTS:-8} --splits train,train_heldout,test $TEST 2>&1 | grep -v Warning
echo "== done $(date)"
