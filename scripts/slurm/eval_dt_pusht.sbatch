#!/bin/bash
#SBATCH --job-name=ev_dt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# eval_dt_pusht.py --watch over one run directory: every step_*.ckpt gets the 8-attempt
# in-context eval (val + test) into <run>/bon_search/dt_eval/<step>/. Submit with
#   sbatch --export=ALL,RUN_DIR=<run dir> [--job-name=...] scripts/slurm/eval_dt_pusht.sbatch
set -uo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone_pushT
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate "${DP_CONDA_ENV:-vae_pushT_l2s}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
: "${RUN_DIR:?set RUN_DIR}"
python eval_dt_pusht.py --watch --run-dir "$RUN_DIR" --n "${N_CANDIDATES:-8}" --n-envs "${N_ENVS:-50}" \
  --split "${SPLIT:-both}" --poll-sec "${POLL_SEC:-600}"
