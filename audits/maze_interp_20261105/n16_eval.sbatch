#!/bin/bash
# n = 1..16 attempts on train starts and on all held-out pairs (2870), for the n16 figure (plots.py n16).
#   sbatch --job-name=n16_<tag>_r<k> n16_eval.sbatch st <tag> <round>
#       -> runs/<tag>/bon_seeds8_n16_r<k>.npz (train) and bon_seeds8_n16_r<k>_allpairs.npz; 8 seeds x 16 attempts in a row
#          (attempt j sees the last min(R-1, j-1) attempts; R = 1 runs have no context, so their 16 attempts are iid)
#   sbatch --job-name=n16_<out> n16_eval.sbatch bc <out> <run> <step>
#   sbatch --job-name=bcnoise n16_eval.sbatch bcnoise <out> <run> <step> <t,t,...>  -> runs/<out>/bc_noisy.npz
#       -> runs/<out>/bc_n16.npz: closed-loop BC (re-plan every step), 128 iid rollouts per pair = 8 seeds x 16 candidates
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30
[[ ${2:-} == c_* ]] && export MAZE_SETTING=interp30c   # c_<tag>: continuous maze (cmaze.py)
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
echo "== start $(date) on $(hostname) $*"
if [[ $1 == st ]]; then
  cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
  A=(--tag $2 --only-round $3 --suffix _n16_r$3 --seeds 8 --attempts 16 --no-bc)
  [[ -f $I/runs/$2/bon_seeds8_n16_r$3.npz ]] || $PY -u bon_eval.py "${A[@]}" --splits train 2>&1 | grep -v Warning
  [[ -f $I/runs/$2/bon_seeds8_n16_r$3_allpairs.npz ]] || INTERP_TEST=all $PY -u bon_eval.py "${A[@]}" --splits test 2>&1 | grep -v Warning
elif [[ $1 == bcnoise ]]; then   # bcnoise <out> <run> <step> <t,t,...> [seeds]: closed-loop BC (exec 1) clean + with observation-feature noise
  cd $I
  $PY -u bc_closed.py --tag $2 --run $3 --step $4 --exec 1 --seeds ${6:-8} --splits train,allpairs --obs-noise-t $5 --out bc_noisy.npz 2>&1 | grep -v Warning
elif [[ $1 == bcsel ]]; then   # bcsel <out> <run> <exec> <train_heldout|test>: pick the BC checkpoint by that success, then the n16 eval
  cd $I
  $PY -u bc_closed.py --tag $2 --run $3 --select --select-on $5 --exec $4 --seeds 128 --splits train,allpairs --out bc_n16.npz 2>&1 | grep -v Warning
else
  cd $I
  $PY -u bc_closed.py --tag $2 --run $3 --step $4 --exec 1 --seeds 128 --splits train,allpairs --out bc_n16.npz 2>&1 | grep -v Warning
fi
echo "== done $(date)"
