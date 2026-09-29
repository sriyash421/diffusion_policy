#!/bin/bash
#SBATCH --job-name=vae_preflight
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
# logs on /gscratch, not home: home is a 10G hard quota (see README_pusht.md)
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Preflight for the VAE_no_pos generation: the invariant checks, then every ladder panel.
#
# WHY NOT THE LOGIN NODE. Each step holds the whole zarr, a policy, and (for the panels) a
# second full AutoencoderKL -- the decoder SDVAEEncoder drops. The login node is one shared
# ~10G cgroup, and every attempt there was OOM-killed partway: of 15 panels exactly one
# survived, and the dataset build alone died repeatedly. Preemptible `ckpt` so the
# guaranteed robotics/weirdlab GPUs stay free for training.
#
# WHAT IT PRINTS THAT MATTERS: `obs_feature_std (measured over N windows)`. That is the
# per-dimension scale the corruption ladder multiplies its noise by. The first generation of
# panels was rendered before decode_obs_latents.py seeded it, i.e. at sigma = 1 on all 324
# dimensions, which is NOT what training applies -- so read that line before reading the
# images, and do not choose a ladder floor from a panel whose sidecar says otherwise.
#
#   sbatch scripts/slurm/vae_preflight.sbatch
#   SHAPES=linear_signal CAPS=99 sbatch --export=ALL,SHAPES,CAPS scripts/slurm/vae_preflight.sbatch
set -euo pipefail

# see eval_ckpt_pusht_search.sbatch: conda's deactivate hook reads an unset var,
# which kills the job under `set -u` if sbatch was run from an active conda shell
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
# vae_pushT_l2s, not robodiff: robodiff cannot import AutoencoderKL at all (diffusers 0.36
# against accelerate 0.13.2). See README_pusht.md section 1.
DP_CONDA_ENV="${DP_CONDA_ENV:-vae_pushT_l2s}"
conda activate "$DP_CONDA_ENV"
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

echo "############ invariants ############"
python scripts/vae_nopos_smoke.py

echo
echo "############ ladder panels ############"
# --device cpu deliberately: the VAE forward on a handful of samples is seconds either way,
# the cost is load time, and this keeps the GPU free for the encode in the smoke above.
DEV="${DEV:-cpu}" bash scripts/render_ladder_panels.sh

echo
echo "############ done ############"
