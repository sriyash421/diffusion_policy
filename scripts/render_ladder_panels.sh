#!/usr/bin/env bash
# Render one decoder panel per ladder shape into media/obs_latent_<shape>/.
#
# WHAT CHANGED. This used to sweep a `cap` -- an artificial ceiling on the noisiest slot --
# because the obs schedule could not reach the noise floor and the question was where to put
# a floor instead. That question is closed: the PushT arms now run TMRL's VLA schedule
# (T=1000, beta 1e-4 -> 0.02), whose sqrt(alpha_bar) bottoms out at 0.0064, so slot 0 really
# is the marginal. Each shape is therefore rendered at its FULL range, which is what the
# training arms will actually use.
#
# It also no longer computes the timesteps itself. It used to build a `mode: list` profile
# from its own hardcoded DDPMScheduler(T=100) -- which, once the policy moved to T=1000,
# would have handed it t<=99 out of 1000, i.e. an almost-clean ladder, and produced panels
# that looked nothing like training. The shape is now passed by NAME and the policy derives
# the ladder from its own scheduler, so the panel cannot disagree with the run.
#
# ONE AT A TIME, deliberately: each render holds the zarr, the policy and a second full
# AutoencoderKL (the decoder the policy drops) at once, and two of them beside an editor hit
# the shared ~10GB login cgroup and get Killed. Run it under scripts/slurm/vae_preflight.sbatch.
#
# Output is TEED, not redirected to /tmp: on a compute node /tmp is node-local, so the
# `obs_feature_std` line -- the one that says what magnitude the panel was rendered at -- was
# being written somewhere the SLURM log could never see.
#
#   bash scripts/render_ladder_panels.sh
#   SHAPES=linear_signal bash scripts/render_ladder_panels.sh
set -uo pipefail
cd "$(dirname "$0")/.."

SHAPES="${SHAPES:-linear_t geometric linear_signal}"
# geometric only. 0.7 is the decay the configs and unit tests use as their example. On the
# 1000-step schedule it is badly degenerate (1099x spread in retained signal, 6 of 15 adjacent
# slot pairs indistinguishable) -- rendered anyway, because seeing that is the point of
# keeping it as a shape comparison.
DECAY="${DECAY:-0.7}"
NS="${NS:-4}"
DEV="${DEV:-cpu}"

for shape in $SHAPES; do
    # `geometric` renders into obs_latent_geometric_t/ -- the folder name is pre-existing.
    case "$shape" in
        geometric) dir=media/obs_latent_geometric_t ;;
        *)         dir=media/obs_latent_$shape ;;
    esac
    mkdir -p "$dir"
    out="$dir/obs_latents"
    if [ -f "$out.png" ] && [ -f "$out.json" ]; then
        printf '%-40s %s\n' "$out" "already rendered, skipping"; continue
    fi
    ov=(-o "slot_obs_noise.mode=$shape")
    [ "$shape" = geometric ] && ov+=(-o "+slot_obs_noise.decay=$DECAY")
    printf '%-40s %s\n' "$out" "mode=$shape"
    python scripts/decode_obs_latents.py "${ov[@]}" \
        --n-samples "$NS" --device "$DEV" --out "$dir" --name "obs_latents" 2>&1 \
        | sed 's/^/    /' \
        || printf '%-40s %s\n' "" "FAILED (see above)"
done

# ---------------------------------------------------------------- random_base base sweep
# `random_base` draws slot 0's level per sample and rescales the shape into [0, base], so it
# has no single ladder to show. Pinning `base_range: [N, N]` makes the draw deterministic at
# N, which renders exactly the ladder that base produces -- one panel per candidate, so the
# noise_cap can be read off the images.
#
# The timesteps are NOT computed here. Passing base_range and letting the policy derive the
# ladder from its own scheduler is what stops this script from disagreeing with the run --
# the failure the `mode: list` version of this driver had when the schedule moved to T=1000.
BASES="${BASES:-999 800 600 400}"
RB_SHAPE="${RB_SHAPE:-linear_signal}"
if [ -n "${BASES// /}" ]; then
    dir=media/obs_latent_random_base
    mkdir -p "$dir"
    for base in $BASES; do
        out="$dir/obs_latents$base"
        if [ -f "$out.png" ] && [ -f "$out.json" ]; then
            printf '%-40s %s\n' "$out" "already rendered, skipping"; continue
        fi
        printf '%-40s %s\n' "$out" "random_base shape=$RB_SHAPE base pinned at $base"
        python scripts/decode_obs_latents.py \
            -o slot_obs_noise.mode=random_base \
            -o "+slot_obs_noise.shape=$RB_SHAPE" \
            -o "+slot_obs_noise.base_range=[$base,$base]" \
            --n-samples "$NS" --device "$DEV" \
            --out "$dir" --name "obs_latents$base" 2>&1 \
            | sed 's/^/    /' \
            || printf '%-40s %s\n' "" "FAILED (see above)"
    done
fi

echo "PANELS DONE"
