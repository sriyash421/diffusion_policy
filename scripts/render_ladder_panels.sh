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
# Slot-0 base per shape. 999 = full extent, which is the arm every shape is trained at;
# linear_t additionally has the 400 variant (arm 2a), and linear_signal carries the sweep
# that the random_base range is read off.
DEFAULT_BASES="${DEFAULT_BASES:-999}"
declare -A BASES_FOR_SHAPE=(
  [linear_t]="999 400"
  [linear_signal]="999 800 600 400"
)
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
    # One panel per slot-0 base, TAGGED BY IT: obs_latents<BASE>.{png,json}. 999 is the
    # shape at full extent; a lower base is the same shape compressed by `max_t`, which is
    # how arm 2a's 400 variant is expressed. `random_base` gets no folder of its own -- it is
    # not a shape, it is one of these shapes with the base drawn per sample, so its panels
    # belong with the shape it borrows.
    for base in ${BASES_FOR_SHAPE[$shape]:-$DEFAULT_BASES}; do
        out="$dir/obs_latents$base"
        if [ -f "$out.png" ] && [ -f "$out.json" ]; then
            printf '%-40s %s\n' "$out" "already rendered, skipping"; continue
        fi
        ov=(-o "slot_obs_noise.mode=$shape")
        [ "$shape" = geometric ] && ov+=(-o "+slot_obs_noise.decay=$DECAY")
        # 999 is the full extent, i.e. no ceiling at all -- max_t would be a no-op and the
        # policy rejects max_t >= T anyway.
        [ "$base" != 999 ] && ov+=(-o "+slot_obs_noise.max_t=$base")
        printf '%-40s %s\n' "$out" "mode=$shape slot0=$base"
        python scripts/decode_obs_latents.py "${ov[@]}" \
            --n-samples "$NS" --device "$DEV" --out "$dir" --name "obs_latents$base" 2>&1 \
            | sed 's/^/    /' \
            || printf '%-40s %s\n' "" "FAILED (see above)"
    done
done

# NO separate random_base sweep. Its panels are the linear_signal ones above: pinning the
# base to N renders exactly the ladder `random_base` produces when it draws N, and `max_t=N`
# on the borrowed shape gives the identical profile (unit_tests/test_slot_obs_noise.py
# asserts that equality). One folder per shape, tagged by slot-0 base.

echo "PANELS DONE"
