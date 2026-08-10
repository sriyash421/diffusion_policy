#!/bin/bash
#SBATCH --job-name=pusht_vid
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Per-decision search videos for one (run, checkpoint) pair, at both n=16 and n=64.
# Runs on the preemptible ckpt partition so training GPUs stay free.
#
# Usage: sbatch scripts/slurm/render_videos.sbatch <RUN_NAME> <LABEL> <STEP> [EXTRA...]
#   RUN_NAME  directory under $DP_OUTPUT_ROOT/runs
#   LABEL     value | subgoal | subgoal_value   (also the output subfolder)
#   STEP      checkpoint step, e.g. 10000
set -euo pipefail

RUN_NAME="${1:?usage: sbatch render_videos.sbatch <RUN_NAME> <LABEL> <STEP>}"
LABEL="${2:?missing LABEL}"
STEP="${3:?missing STEP}"
shift 3

# see eval_ckpt_pusht_search.sbatch: conda's deactivate hook reads an unset var,
# which kills the job under `set -u` if sbatch was run from an active robodiff shell
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"
CKPT=$(printf '%s/runs/%s/checkpoints/step_%07d.ckpt' "$ROOT" "$RUN_NAME" "$STEP")
OUT="videos/jul31_debug/$LABEL"

# `value` feeds back only the verifier scalar, so there is no subgoal to show.
SUB="--subgoals"
[ "$LABEL" = "value" ] && SUB="--no-subgoals"

for N in 16 64; do
  echo "########## $LABEL step=$STEP n=$N ##########"
  python scripts/render_search_videos.py \
    -c "$CKPT" --label "$LABEL" --n "$N" --out-dir "$OUT" \
    --device cuda:0 $SUB "$@"
done
echo "########## DONE $LABEL step=$STEP ##########"
