#!/bin/bash
# BC baseline for an ASTEROID / DAgger relabel run: BC UNet retrained on the data the run collected (the BFS expert's demos from every
# state the learner visited, all rounds aggregated; dagger_bc_data.py), same UNet / 10k updates as the interp30 BC. Two horizons:
#   64 (the original BC): evaluated open loop (one chunk from the start) and closed loop (re-plan every step)
#   4  (as bc_h4):         evaluated closed loop
# Each picks its checkpoint by train_heldout under its own rollout mode (bc_closed.py --select), then train / test / control / all pairs.
#   sbatch --job-name=dagbc_<tag> dagger_bc.sbatch <tag>     (after the relabel run's TRAINED)
#SBATCH --account=robotics
#SBATCH --partition=ckpt-all
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
D=/gscratch/robotics/harine/codex_audits/maze_quadrants_delta_20261019
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
tag=$1; Z=$I/runs/$tag/bc_data.zarr
echo "== start $(date) on $(hostname) restart=${SLURM_RESTART_COUNT:-0} tag=$tag"
cd $I
[[ -f $I/runs/$tag/TRAINED ]]
[[ -f $Z/.built ]] || { $PY -u dagger_bc_data.py --tag $tag 2>&1 | grep -v Warning; touch $Z/.built; }
for h in 64 4; do
  $PY -u $D/run_experiment.py --setting interp30 --algo bc_unet --seed 0 --variant clean --out-root $I --dataset $Z --val-ratio .02 \
      --horizon $h --tag dag_${tag}_h$h 2>&1 | grep -v Warning | grep -E "VALIDATION|Traceback|Error" || true
  [[ -f runs/interp30/bc_unet__clean__dag_${tag}_h$h/seed_0/TRAINED ]]
done
R=runs/interp30/bc_unet__clean__dag_${tag}
[[ -f runs/dagbc_${tag}_h64_open/bc_closed.npz ]]   || $PY -u bc_closed.py --tag dagbc_${tag}_h64_open   --run ${R}_h64/seed_0 --select --exec 64 --seeds 8 2>&1 | grep -v Warning
[[ -f runs/dagbc_${tag}_h64_closed/bc_closed.npz ]] || $PY -u bc_closed.py --tag dagbc_${tag}_h64_closed --run ${R}_h64/seed_0 --select --exec 1 --seeds 8 2>&1 | grep -v Warning
[[ -f runs/dagbc_${tag}_h4_closed/bc_closed.npz ]]  || $PY -u bc_closed.py --tag dagbc_${tag}_h4_closed  --run ${R}_h4/seed_0 --select --exec 1 --seeds 8 2>&1 | grep -v Warning
echo "== done $(date)"
