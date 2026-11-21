#!/bin/bash
#SBATCH --job-name=ev_stage_cands
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# 8 in-context candidates (eval_dt_pusht.py) from every train and test start for each stage checkpoint
# stages/pi_k<n>.ckpt -> <run>/bon_search/stage_eval/pi_k<n>/{train,test}_candidates.npz. Skips
# finished ones, so a requeue resumes. MAX_EP=<k>: first k starts only (smoke).
#   sbatch --account=ckpt-robotics --partition=ckpt-all --requeue --export=ALL,RUN_DIR=<run> scripts/slurm/eval_stage_candidates.sbatch
set -eo pipefail
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate vae_pushT_l2s
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/diffusion_policy_standalone_pushT}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONPATH=$PWD
for ck in $(ls "$RUN_DIR"/stages/pi_k*.ckpt | sort -V); do
  name=$(basename "$ck" .ckpt)
  # complete once items_k<n>.pt is written after it: training may still be saving this checkpoint
  [[ -f "$(dirname "$ck")/items_${name#pi_}.pt" ]] || continue
  out="$RUN_DIR/bon_search/stage_eval/$name"
  for split in train test; do
    [[ -f "$out/${split}_candidates.npz" ]] && continue
    python eval_dt_pusht.py "$ck" --n 8 --split $split --output-dir "$out" ${MAX_EP:+--max-episodes $MAX_EP}
  done
done
