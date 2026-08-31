#!/bin/bash
#SBATCH --job-name=rndbase_smoke
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# A few real outer steps of the `random_base` ladder arm, through the REAL trainer.
#
# WHY THIS MODE SPECIFICALLY. random_base is the one ladder whose fixed-timestep buffer is
# None by construction -- it registers `slot_obs_shape` instead, and its levels do not exist
# until a base is drawn per sample. _compute_loss used to dispatch on `slot_obs_t is None`
# alone, which sent exactly this mode down the FLAT corruption path, where corrupt_obs is
# False (the two are mutually exclusive) and the corruption is the identity: the arm trained
# entirely clean while its startup print announced the ladder. Fixed by dispatching on
# `slot_ladder_on`, which both call sites now share.
#
# The unit tests pin the predicate; they do not run the trainer. This does, because the
# things that could still fail here -- the per-decision base pinned by _corrupt_scope, the
# running obs_feature_std, the (B,K,To,D) conditioning the inner loop then indexes -- all
# fail quietly rather than loudly. Expect a startup line reading
# `slot_obs_noise random_base shape=linear_signal base_range=[0, 999]`.
#
# The unit tests cover the ladder maths and the crop contract, and vae_nopos_smoke covers the
# obs contract -- none of them run the TRAINER. This does, because the per-inner-epoch feature
# cache changed the fill's return shape, the inner loop's indexing and the drift metric's
# inputs all at once, and every one of those fails silently rather than loudly: a wrong epoch
# row is just a different augmentation, and a drift metric on the wrong features is just a
# different number.
#
# Deliberately NOT training.debug: that pins inner_epochs=1, which is exactly the case the
# multi-crop cache does not exercise. The pool is shrunk instead so the verifier stays cheap.
#
#   sbatch scripts/slurm/outer_inner_smoke.sbatch
set -euo pipefail
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate "${DP_CONDA_ENV:-vae_pushT_l2s}"
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

export DP_OUTPUT_ROOT=/gscratch/robotics/harine/diffusion_policy_smoke
python train.py --config-name=train_pusht_diffusion_search \
  n_candidates=16 \
  slot_obs_noise.mode=random_base \
  +slot_obs_noise.shape=linear_signal \
  "+slot_obs_noise.base_range=[0,999]" \
  son_suffix=_son-rndlinsig son_tag=random_base-linsig obs_noise_tag=obs_noised \
  run_name=RNDBASE_SMOKE_delete_me \
  training.seed=42 \
  training.max_gradient_steps=64 \
  training.outer_batch_size=32 \
  training.inner_batch_size=8 \
  training.inner_epochs=4 \
  training.drift_every=4 \
  training.rollout_every_steps=1000000 \
  training.val_every_steps=1000000 \
  training.sample_every_steps=1000000 \
  training.checkpoint_every=1000000 \
  training.resume=False \
  logging.mode=disabled
echo "OI SMOKE DONE"
