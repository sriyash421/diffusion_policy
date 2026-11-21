#!/bin/bash
# BC UNet on the interp30 demos: smoke (2 updates + smoke eval), 10k updates, evaluate.py on the 10 checkpoints, pick by train_heldout.
#SBATCH --account=robotics
#SBATCH --partition=ckpt-all
#SBATCH --constraint=a40|a100|l40|l40s|h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=interp_bc
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
D=/gscratch/robotics/harine/codex_audits/maze_quadrants_delta_20261019
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
echo "== start $(date) on $(hostname) restart=${SLURM_RESTART_COUNT:-0}"
RUN=$I/runs/interp30/bc_unet__clean/seed_0
if [[ ! -f $I/smoke/interp30/bc_unet__clean/seed_0/smoke_evals/step_0000002/COMPLETED ]]; then
  "$PY" -u $D/run_experiment.py --setting interp30 --algo bc_unet --seed 0 --variant clean --out-root $I --smoke
  "$PY" -u $D/evaluate.py --checkpoint $I/smoke/interp30/bc_unet__clean/seed_0/step_0000002.pt --smoke
  echo SMOKE_OK
fi
"$PY" -u $D/run_experiment.py --setting interp30 --algo bc_unet --seed 0 --variant clean --out-root $I
for s in 1 2 3 4 5 6 7 8 9 10; do
  "$PY" -u $D/evaluate.py --checkpoint $RUN/step_$(printf '%07d' $((s*1000))).pt
done
"$PY" -u $I/bc_select.py
