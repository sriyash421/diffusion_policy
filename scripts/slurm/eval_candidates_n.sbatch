#!/bin/bash
#SBATCH --job-name=ev_cands_n
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# N candidates (default 32) from every train and test start for one checkpoint (eval_dt_pusht.py) ->
# $OUT/{train,test}_candidates.npz; MAX_ACTIONS=1 draws every candidate without context. Skips a finished
# split, so a requeue resumes.
#   sbatch --account=ckpt-robotics --partition=ckpt-all --requeue --export=ALL,CK=<ckpt>,OUT=<dir>[,MAX_ACTIONS=1] scripts/slurm/eval_candidates_n.sbatch
set -eo pipefail
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate vae_pushT_l2s
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/diffusion_policy_standalone_pushT}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONPATH=$PWD
: "${CK:?set CK}" "${OUT:?set OUT}"
for split in train test; do
  [[ -f "$OUT/${split}_candidates.npz" ]] && continue
  python eval_dt_pusht.py "$CK" --n "${N:-32}" --split $split --output-dir "$OUT" ${MAX_ACTIONS:+--max-actions $MAX_ACTIONS}
done
