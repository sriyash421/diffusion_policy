#!/bin/bash
#SBATCH --job-name=sac_policy_eval
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# logs on /gscratch, not home: home is a 10G hard quota (see README_pusht.md)
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Success-rate watcher for a SAC arm, on the MANIFEST's held-out episodes.
#
# WHAT THIS MEASURES THAT THE RUN DOES NOT. `FixedEval` logs eval/success_rate on seeded
# PROCEDURAL resets -- fixed in ST's STYLE, not ST's SET. This rolls the recorded first frame of
# each episode the split manifest holds out, which is what every offline arm is scored on, so
# these numbers are comparable across families and the run's own are not.
#
# It REFUSES a run whose replay buffer was seeded from the episodes being scored (sac/score.py's
# assert_held_out). That is why it is pointed at the per-split arms: sac_keypoint_blq137 and
# sac_keypoint_brd100 each hold out their own manifest's 50 test episodes. The all-206
# sac_keypoint / sac_image runs need --allow-contaminated, and the flag is then recorded in
# every row so their numbers stay distinguishable from the held-out ones.
#
# Preemptible on purpose, so the guaranteed robotics/weirdlab GPUs stay free for training.
# --requeue re-launches after preemption, and `policy` seeds its done-set from the rows already
# in success_rates.jsonl, so a preempted watcher resumes instead of re-scoring.
#
# Usage: sbatch scripts/slurm/eval_watch_sac_policy.sbatch <SAC_RUN_DIR> [extra args]
#   (the SAC run dir, which contains model_*_steps.zip and params/args.yaml)
# Extra args go to `sac/eval.py policy` -- e.g. --idle-exit-sec 7200, --stochastic.
set -euo pipefail

RUN_DIR="${1:?usage: sbatch eval_watch_sac_policy.sbatch <SAC_RUN_DIR> [eval args]}"
shift

# conda's deactivate hook reads an unset var, which kills the job under `set -u` if sbatch
# was run from an active conda shell
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
DP_CONDA_ENV="${DP_CONDA_ENV:-robodiff}"
conda activate "$DP_CONDA_ENV"
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

# the shell's WANDB_API_KEY on klone is stale and shadows the valid ~/.netrc; nothing here logs
# to wandb, but an inherited bad key has killed jobs at import time before
unset WANDB_API_KEY || true

python sac/eval.py policy "$RUN_DIR" \
  --watch \
  --split test \
  --device cuda:0 \
  "$@"
