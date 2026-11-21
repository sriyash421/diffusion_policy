#!/bin/bash
# Short-horizon BC UNet (horizon 4; the same UNet / demos / 10k updates as the interp30 BC, which predicts 64), run closed loop:
# re-plan from the current cell every step, execute the first action (bc_closed.py --exec 1). The checkpoint is picked by
# train_heldout under that closed-loop rollout, then evaluated on train starts / 50 test / 50 control / all held-out pairs.
#   sbatch bc_h4.sbatch                -> runs/interp30/bc_unet__clean__h4/seed_0, runs/bc_h4_closed/{selection.json,bc_closed.npz}
#   sbatch bc_h4.sbatch bc64           -> only the closed-loop eval of the existing 64-step BC (step 7000) -> runs/bc64_closed/
#   sbatch bc_h4.sbatch cont           -> the same on the continuous maze (MAZE_SETTING=interp30c, smoothed-A* demos, real-valued actions)
#                                         -> runs/interp30c/bc_unet__clean__h4/seed_0, runs/c_bc_h4/{selection.json,bc_closed.npz}
#SBATCH --account=robotics
#SBATCH --partition=ckpt-all
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=bc_h4
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
D=/gscratch/robotics/harine/codex_audits/maze_quadrants_delta_20261019
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
echo "== start $(date) on $(hostname) restart=${SLURM_RESTART_COUNT:-0} mode=${1:-h4}"
cd $I
if [[ ${1:-h4} == bc64 ]]; then
  [[ -f runs/bc64_closed/bc_closed.npz ]] || $PY -u bc_closed.py --tag bc64_closed --run runs/interp30/bc_unet__clean/seed_0 --step 7000 --exec 1 --seeds 8 2>&1 | grep -v Warning
  echo "== done $(date)"; exit 0
fi
ST=interp30; OUT=bc_h4_closed
if [[ ${1:-h4} == cont ]]; then export MAZE_SETTING=interp30c; ST=interp30c; OUT=c_bc_h4; fi
"$PY" -u $D/run_experiment.py --setting $ST --algo bc_unet --seed 0 --variant clean --out-root $I --horizon 4 --tag h4 2>&1 | grep -v Warning | grep -E "VALIDATION|Traceback|Error" || true
[[ -f runs/$ST/bc_unet__clean__h4/seed_0/TRAINED ]]
[[ -f runs/$OUT/bc_closed.npz ]] || $PY -u bc_closed.py --tag $OUT --run runs/$ST/bc_unet__clean__h4/seed_0 --select --exec 1 --seeds 8 2>&1 | grep -v Warning
echo "== done $(date)"
