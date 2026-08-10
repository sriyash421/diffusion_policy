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

python train.py \
  --config-name="$CONFIG_NAME" \
  training.device=cuda:0 \
  "$@"
