#!/usr/bin/env bash
# THE VAE_no_pos BASELINE GENERATION, 30 demos: the three headline arms retrained on the
# frozen SD-VAE encoder with an image-only observation.
#
# WHAT CHANGED UNDER THEM, and why the previous numbers do not carry over:
#   * obs backbone     ResNet18 (trainable) -> SD VAE encoder, FROZEN and always in eval
#   * observation      {image, agent_pos, feedback} -> {image}. feedback is an exact
#                      invertible transform of the block pose, so the old arms were handed
#                      the ground-truth T pose in closed form.
#   * crop             76x76 -> 72x72, on BOTH arms (the UNet arm was on 76 alone)
#   * task_name        pusht_image_search -> pusht_image_search_imgonly, so these land in
#                      their own output tree and cannot resume anything older
# The run_name also gains _enc-vae, which is what stops a VAE run from resuming a ResNet
# run's checkpoints when the two agree on every other identity key (AUDIT 9.9).
#
# These three are the CLEAN BASELINE. The slot_obs_noise ladder arms are read against them,
# so they go first -- a ladder number with nothing to compare it to says nothing.
#
# run_name is DERIVED, not overridden. Every axis that distinguishes these runs is now in
# run_name by construction (arm, k, verifier, slot weights, obs noise, corrupt, arch,
# encoder, demos, seed), so hardcoding a name here would just be a second place for it to
# drift. The script asks hydra what the name resolves to and uses that for its guards.
#
#   bash scripts/run_vae_nopos_30demo.sh            # dry run: show what would be submitted
#   SUBMIT=1 bash scripts/run_vae_nopos_30demo.sh   # ...and sbatch them
#
# EVAL IS NOT DONE HERE. See scripts/slurm/submit_vae_nopos_readouts.sh.
set -uo pipefail
cd "$(dirname "$0")/.."

SUBMIT="${SUBMIT:-}"
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search
PY="${DP_PY:-python}"

# config | extra overrides
#
# n_candidates=1 IS LOAD-BEARING on the k=1 arm: train_pusht_diffusion_search_single pins the
# single-step TRAINER, not width 1, and inherits n_candidates: 16 from the search master.
# Without it the k=1 arm trains at width 16 under a name that says k1.
# THE EIGHT ARMS. Three baselines, then five ladders -- every ladder is ST k=16 with uniform
# slot weights, so arm 3 is their control and the only thing that varies is the ladder.
#
# EVERY LADDER SETS THREE LABELS TOGETHER: son_suffix (the run DIRECTORY, since run_name IS
# hydra.run.dir and two ladders sharing a name would resume each other), son_tag and
# obs_noise_tag (the wandb tags). _check_obs_noise_labels refuses to start if any of them
# disagrees with policy.slot_obs_noise, so a forgotten one is a startup error, not a silently
# mislabelled run.
#
# The obs corruption schedule is TMRL's VLA one (T=1000, beta 1e-4->0.02) and lives in
# train_pusht_diffusion_search.yaml, so `max_t` and `base_range` below are absolute timesteps
# of THAT schedule -- they must be re-derived if it ever changes.
LADDER_TAGS='son_suffix=%s son_tag=%s obs_noise_tag=obs_noised'
LADDER_ARMS=(
  # --- baselines: no ladder ---
  "train_pusht_unet_bc|"
  "train_pusht_diffusion_search_single|n_candidates=1"
  "train_pusht_diffusion_search|n_candidates=16"
  # --- 2a linear in t, slot 0 at the full 999 and compressed to 400 ---
  "train_pusht_diffusion_search|n_candidates=16 slot_obs_noise.mode=linear_t son_suffix=_son-lint999 son_tag=linear_t-999 obs_noise_tag=obs_noised"
  "train_pusht_diffusion_search|n_candidates=16 slot_obs_noise.mode=linear_t +slot_obs_noise.max_t=400 son_suffix=_son-lint400 son_tag=linear_t-400 obs_noise_tag=obs_noised"
  # --- 2b linear in retained signal (sqrt(alpha_bar)) ---
  "train_pusht_diffusion_search|n_candidates=16 slot_obs_noise.mode=linear_signal son_suffix=_son-linsig son_tag=linear_signal obs_noise_tag=obs_noised"
  # --- 2c geometric in t. Degenerate on T=1000 (6/15 adjacent pairs indistinguishable);
  #     it is the shape-comparison control, not a contender. The startup print warns.
  "train_pusht_diffusion_search|n_candidates=16 slot_obs_noise.mode=geometric +slot_obs_noise.decay=0.7 son_suffix=_son-geo7 son_tag=geometric-0.7 obs_noise_tag=obs_noised"
  # --- 2d slot 0 drawn per sample over the WHOLE range, including 0 (sometimes no noise
  #     at all), with the linear-in-signal shape rescaled into [0, base] ---
  "train_pusht_diffusion_search|n_candidates=16 slot_obs_noise.mode=random_base +slot_obs_noise.shape=linear_signal +slot_obs_noise.base_range=[0,999] son_suffix=_son-rndlinsig son_tag=random_base-linsig obs_noise_tag=obs_noised"
)

# The ENCODER DEBUG set: 3 encoders x {BC UNet, ST k=1}, no ladder on any of them.
# Isolates the low-n collapse -- run 2 vs the ResNet curve already on disk tests the
# speedup revert; ResNet->vae isolates the encoder; vae->vae-ft isolates the freeze.
#
# The ResNet override reaches INSIDE obs_encoder because `policy.obs_encoder` is an
# interpolation of that block, so hydra cannot address it any other way. encoder_tag drives
# enc_suffix, so all six land in distinct run dirs and cannot resume each other.
RESNET="obs_encoder.rgb_model._target_=diffusion_policy.model.vision.model_getter.get_resnet +obs_encoder.rgb_model.name=resnet18 +obs_encoder.rgb_model.weights=IMAGENET1K_V1 ~obs_encoder.rgb_model.model_name ~obs_encoder.rgb_model.scaling_factor obs_encoder.use_group_norm=True crop_shape=[76,76] encoder_tag=resnet18"
VAE_FT="+obs_encoder.rgb_model.trainable=True encoder_tag=vae-ft"
RESNET_FROZEN="+training.freeze_encoder=True encoder_tag=resnet18-frozen"

DEBUG_ARMS=(
  "train_pusht_unet_bc|$RESNET"
  "train_pusht_diffusion_search_single|n_candidates=1 $RESNET"
  "train_pusht_unet_bc|"
  "train_pusht_diffusion_search_single|n_candidates=1"
  "train_pusht_unet_bc|$VAE_FT"
  "train_pusht_diffusion_search_single|n_candidates=1 $VAE_FT"
  # Frozen ResNet -- the missing cell of the encoder x freeze 2x2. No obs_encoder override:
  # the default IS ResNet18 now, so this differs from the trainable ResNet arm ONLY by the
  # freeze. `+` because freeze_encoder is not declared in the composed PushT config.
  #
  # ST k=1 MUST use _single: `train_pusht_diffusion_search n_candidates=1` selects
  # TrainSearchOuterInnerWorkspace, which has no freeze_encoder handler, so the freeze would
  # be a SILENT no-op and the arm a duplicate of the trainable one.
  "train_pusht_unet_bc|$RESNET_FROZEN"
  "train_pusht_diffusion_search_single|n_candidates=1 $RESNET_FROZEN"
)

# ARMS_SET=debug runs the encoder debug set; anything else runs the ladder matrix.
if [ "${ARMS_SET:-ladder}" = "debug" ]; then
  ARMS=("${DEBUG_ARMS[@]}")
else
  ARMS=("${LADDER_ARMS[@]}")
fi

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)

# Fail loudly on the wrong interpreter rather than once per arm. $PY defaults to bare
# `python`, so running this outside the conda env makes every --cfg job below fail on the
# hydra import and every arm skip with "CONFIG FAILED TO RESOLVE" -- indistinguishable
# from a genuinely broken config, and it looks like the launcher ran and found nothing.
if ! $PY -c 'import hydra' 2>/dev/null; then
    echo "ERROR: '$PY' cannot import hydra. Activate the env first:" >&2
    echo "         conda activate \${DP_CONDA_ENV:-vae_pushT_l2s}" >&2
    echo "       or set DP_PY to an interpreter that has it." >&2
    exit 1
fi

n=0
for arm in "${ARMS[@]}"; do
    IFS='|' read -r cfg ov <<<"$arm"
    # Ask the config, rather than repeating it. Also acts as a config check: a resolution
    # error here stops the arm instead of surfacing 30 seconds into a SLURM job.
    err=$(mktemp); resolved=$($PY train.py --config-name="$cfg" $ov --cfg job --resolve 2>"$err") || true
    # Extracted BY NAME, not by position: --cfg job prints the keys in config order, which
    # is not the order of the path.
    run=$(sed -n 's/^run_name: //p'  <<<"$resolved")
    sub=$(sed -n 's/^trainer: //p'   <<<"$resolved")
    task=$(sed -n 's/^task_name: //p' <<<"$resolved")
    if [ -z "$run" ] || [ -z "$sub" ] || [ -z "$task" ]; then
        printf '%-56s %s\n' "$cfg" "CONFIG FAILED TO RESOLVE, skipping"
        sed -e 's/^/    | /' "$err" | tail -5 >&2
        rm -f "$err"; continue
    fi
    rm -f "$err"
    dir="$ROOT/$task/$sub/$run"
    name="tr_$run"
    # Already training, or already finished? Resubmitting a live run is how two jobs end up
    # writing one hydra.run.dir; resubmitting a finished one silently continues from
    # latest.ckpt and appends steps past max_gradient_steps.
    if grep -qxF "$name" <<<"$LIVE"; then
        printf '%-56s %s\n' "$run" "already training, skipping"; continue
    fi
    if [ -f "$dir/checkpoints/step_0100000.ckpt" ]; then
        printf '%-56s %s\n' "$run" "already finished (step_0100000.ckpt present)"; continue
    fi
    if [ -z "$SUBMIT" ]; then
        printf '%-56s WOULD SUBMIT  %s [%s]\n' "$run" "$cfg" "${ov:-no overrides}"
        printf '%-56s   -> %s\n' "" "$dir"
        n=$((n+1)); continue
    fi
    # One pick_gpu call per arm: it reports whichever robotics/weirdlab partition has free
    # GPUs right now, and three submissions in a row would otherwise pile onto the first.
    read -r A P < <(bash scripts/slurm/pick_gpu.sh) || { echo "no free GPU for $run" >&2; continue; }
    jid=$(sbatch --parsable --account="$A" --partition="$P" --job-name="$name" \
          --export=ALL,CONFIG_NAME="$cfg" scripts/slurm/train_pusht_search.sbatch $ov)
    printf '%-56s submitted %s on %s/%s\n' "$run" "$jid" "$A" "$P"
    n=$((n+1))
done
echo
echo "arms handled: $n"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
