#!/bin/bash
#SBATCH --job-name=pusht_search_train
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10-00:00:00
# logs on /gscratch, not home: home is a 10G hard quota and these capture raw tqdm output
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# --time MUST be finite and MUST fit before the next cluster maintenance reservation.
# The partitions are MaxTime=UNLIMITED, so `--time=0` is legal -- but an unlimited job can
# never be proven to finish before a maintenance window, so SLURM parks it with
# `Reason=ReqNodeNotAvail,_Reserved_for_maintenance` and it silently never starts. Check
# the window with `scontrol show reservation | grep -A1 Maintenance` and keep --time under
# the gap. Do NOT omit --time either: that applies the partition DefaultTime (1h).
#
# A run that outlives its --time is fine -- it checkpoints every 2000 gradient steps and
# resumes exactly, but ONLY if you resubmit with the same hydra.run.dir (see below).
# --account / --partition are NOT hardcoded -- pass them at submit time from pick_gpu.sh
# so the job lands on whatever robotics/weirdlab GPU is free (and 1 GPU + 8 CPUs + 96G
# leaves the node's other GPUs usable):
#
#   read A P < <(bash scripts/slurm/pick_gpu.sh)
#   sbatch --account=$A --partition=$P scripts/slurm/train_pusht_search.sbatch
#
# CONFIG_NAME selects the run (default = the scalar-verifier baseline). The other configs
# need --export so the variable reaches the job:
#
#   CONFIG_NAME=train_pusht_diffusion_search_subgoal \
#     sbatch --account=$A --partition=$P --export=ALL,CONFIG_NAME \
#            scripts/slurm/train_pusht_search.sbatch
#   CONFIG_NAME=train_pusht_diffusion_search_subgoal_verifier \
#     sbatch --account=$A --partition=$P --export=ALL,CONFIG_NAME \
#            scripts/slurm/train_pusht_search.sbatch
#   CONFIG_NAME=train_pusht_search_outer_inner \
#     sbatch --account=$A --partition=$P --export=ALL,CONFIG_NAME \
#            scripts/slurm/train_pusht_search.sbatch
#
# Run output goes to $DP_OUTPUT_ROOT (default /gscratch/robotics/harine/diffusion_policy_outputs),
# never home.
#
# RESUMING: just resubmit the same command. `hydra.run.dir` is
# ${output_root}/${exp_name}/${task_name}/${trainer}/${run_name} -- derived from the arm,
# not from a timestamp -- so the same config resolves to the same directory, finds its own
# latest.ckpt, and continues. Verified end to end by scripts/slurm/test_resume.sbatch.
#
# (This block used to say the opposite: that hydra mints a new timestamped directory and a
# plain resubmit silently restarts. That was true of the upstream template's
# `dir: data/outputs/${now:...}` and stopped being true when run_name became the path. Do
# NOT pass an explicit hydra.run.dir "to be safe" -- a path that differs from the derived
# one by so much as a trailing slash starts a second run in a second directory.)
#
# Two things a resume does NOT restore, neither of which changes the result:
#   - RNG state. Noise/timestep draws and the dataloader shuffle restart from
#     training.seed. They are i.i.d. by design and each epoch still covers the full
#     dataset, so this is not the biased-resampling hazard it would be for a run that
#     draws a random POOL per step (the outer/inner trainer, which seeds from position).
#   - The epoch loop counter, which restarts at 0 while self.epoch continues. Harmless:
#     num_epochs (1000, ~385k steps) is far above max_gradient_steps, which is the real
#     bound, and every eval cadence is keyed on gradient steps via last_*_step.
set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-train_pusht_diffusion_search}"

# see eval_ckpt_pusht_search.sbatch: conda's deactivate hook reads an unset var,
# which kills the job under `set -u` if sbatch was run from an active robodiff shell
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

# ---------------------------------------------------------------------------
# STORAGE. Home is a 10G hard quota; a single run's checkpoints exceed it on their own.
# Everything a run writes must land on /gscratch.
# ---------------------------------------------------------------------------
GSCRATCH_ROOT=/gscratch/robotics/harine

# The config already defaults output_root to $GSCRATCH_ROOT/diffusion_policy_outputs, but
# DP_OUTPUT_ROOT overrides it -- and README_pusht/build_success_rates_doc tell you to point
# that at a LOCAL COPY of the results tree for off-cluster work. A training job that
# inherited such a value would write the entire run there, silently, and blow the quota.
# So: default it explicitly, and refuse to start if it points anywhere but /gscratch.
export DP_OUTPUT_ROOT="${DP_OUTPUT_ROOT:-$GSCRATCH_ROOT/diffusion_policy_outputs}"
case "$DP_OUTPUT_ROOT" in
  /gscratch/*) ;;
  *) echo "FATAL: DP_OUTPUT_ROOT=$DP_OUTPUT_ROOT is not on /gscratch. A training run writes"
     echo "       tens of GB; on home that hits the 10G quota mid-run. Unset it or point it"
     echo "       at /gscratch/..." >&2
     exit 1 ;;
esac

# wandb writes its RUN files to `dir=` (the hydra output dir, already on /gscratch) but its
# cache, artifacts and config default to $HOME. Redirect all four.
export WANDB_DIR="${WANDB_DIR:-$GSCRATCH_ROOT/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$GSCRATCH_ROOT/wandb/cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$GSCRATCH_ROOT/wandb/artifacts}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$GSCRATCH_ROOT/wandb/config}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR"

# Preflight free space. /gscratch and home are the SAME gpfs filesystem (/mmfs1), so this is
# not a formality -- the UNet arms already died once with "No space left on device" at step
# 115k. A 100k-step search run writes ~3 GB of checkpoints; refuse to start under 50G so a
# doomed job fails in the first second rather than at step 90k.
avail_g=$(df -BG --output=avail "$DP_OUTPUT_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "${avail_g:-}" ] && [ "$avail_g" -lt 50 ]; then
  echo "FATAL: only ${avail_g}G free on the filesystem holding $DP_OUTPUT_ROOT (need >=50G)." >&2
  exit 1
fi
echo "output root : $DP_OUTPUT_ROOT (${avail_g:-?}G free)"
echo "wandb dir   : $WANDB_DIR"

python train.py \
  --config-name="$CONFIG_NAME" \
  training.device=cuda:0 \
  "$@"
