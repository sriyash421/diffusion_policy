#!/bin/bash
#SBATCH --job-name=bcvstk1_vid
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%A_%a.out
#SBATCH --array=0-5
#
# Objective 1 of docs/reports/archive/aug19_analysis.md: UNet BC vs ST k=1 action-sample videos.
#
# 6 array tasks = 2 policies x 3 checkpoints. Each task loads its policy ONCE, spawns the
# 32-worker verifier pool ONCE, and sweeps every n internally -- the pools cost more to
# build than a search costs to run, so splitting the n sweep across tasks would be slower
# and would also re-pay the ResNet warmup five times.
#
# Preemptible ckpt partition so training GPUs stay free; --requeue because a rendered
# video is cheap to redo and nothing downstream reads a partial one (the .tmp_* name is
# only renamed to its final form after the episode completes).
#
#   sbatch scripts/slurm/render_bcvstk1.sbatch
set -euo pipefail

# see eval_ckpt_pusht_search.sbatch: conda's deactivate hook reads an unset var, which
# kills the job under `set -u` if sbatch was run from an active robodiff shell
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search
OUT=/mmfs1/home/harine/diffusion_policy_standalone/videos/aug19_bcvSTk1_actions_viz

# Both ST arms in docs/reports/archive/aug19_analysis.md are the 4/4/256 pair: the big 6x8x1024 k=16 run stops
# at step 40000 and so cannot supply the 50k/100k cells. stk1 is `bc_demos-30` -- that
# directory name predates the rename and IS ST k=1, not the UNet BC baseline.
RUNS=(unet_bc/unetbc_demos-30_seed-42 offline/bc_demos-30_seed-42)
LABELS=(unetbc stk1)
STEPS=(10000 50000 100000)

p=$(( SLURM_ARRAY_TASK_ID / 3 ))
s=$(( SLURM_ARRAY_TASK_ID % 3 ))
CKPT=$(printf '%s/%s/checkpoints/step_%07d.ckpt' "$ROOT" "${RUNS[$p]}" "${STEPS[$s]}")
LABEL="${LABELS[$p]}"

echo "=== task $SLURM_ARRAY_TASK_ID: $LABEL step ${STEPS[$s]}"
echo "=== $CKPT"
[ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }

# --seed 42 pins each episode's noise to (episode index, draw index) via set_sample_seeds,
# so the two arms are paired episode-by-episode rather than merely both random.
python -u scripts/render_search_videos.py \
  -c "$CKPT" \
  --label "$LABEL" \
  --n 1 --n 2 --n 8 --n 16 --n 64 \
  --out-dir "$OUT" \
  --n-seeds 10 \
  --seed 42 \
  --no-subgoals \
  --max-fan 24 \
  --zoom 384 \
  --skip-blind \
  --closeup \
  --no-value-strip \
  --device cuda:0 \
  "$@"

echo "=== done $LABEL step ${STEPS[$s]}"
