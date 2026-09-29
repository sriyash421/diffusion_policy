#!/bin/bash
#SBATCH --job-name=diag_bc_n1
#SBATCH --partition=gpu-a40
#SBATCH --account=gpu-a40-robotics
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Why does BC score 0% at n=1 when it should solve PushT?
#
# Two candidate causes, and this separates them:
#   (a) UNDERTRAINED -- 52 epochs against upstream's 3050. Tested by comparing an early
#       checkpoint with the last one: if both are equally bad, training length is not the
#       whole story.
#   (b) SAMPLER -- num_inference_steps is 8, inherited from the search config where each
#       gradient step draws 15 candidates; upstream PushT uses 100. Purely a sampling-time
#       knob, so sweeping it on FIXED weights isolates sampler error from policy quality.
#
# Read-only: evaluates existing checkpoints, writes nothing into bon_search/.
set -euo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
set +u; conda activate robodiff; set -u

RUN=/gscratch/robotics/harine/diffusion_policy_outputs/_diag/bc_preserved
OUT=/gscratch/robotics/harine/diffusion_policy_outputs/_diag/bc_n1
mkdir -p "$OUT"

for STEP in 0011000 0020000; do
  for NIS in 8 16 32 100; do
    echo "=== step $STEP  num_inference_steps=$NIS ==="
    python eval_search_pusht.py \
      -c "$RUN/step_${STEP}.ckpt" \
      -o "$OUT/step${STEP}_nis${NIS}" \
      --min-n 1 --max-n 1 \
      --num-inference-steps "$NIS" \
      --device cuda:0
  done
done
