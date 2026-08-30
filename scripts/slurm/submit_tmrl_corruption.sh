#!/bin/bash
# TMRL-style graded observation corruption: three ladder shapes, one arm each.
#
# Slot k's obs features are noised by the obs DDPM at a FIXED timestep t_k that falls across
# the slot index -- slot 0 (no search context) most corrupted, slot K-1 (full context)
# clean. The premise is that a slot which cannot see well must explore while a
# fully-conditioned slot sharpens, so the search context becomes something the model must
# USE rather than may ignore. See SearchProcedureMixin._resolve_slot_obs_noise.
#
#   linear_t       even in the timestep index
#   geometric      the shape slot_weights already offers; needs `decay`
#   linear_signal  even in sqrt(alpha_bar) -- the only shape giving all K slots a distinct,
#                  equally spaced corruption level, since alpha_bar is a cumulative product
#                  and equal steps in t are NOT equal steps in corruption. The default.
#
# Everything else is the config default and is NOT overridden: k=16, 30 demos, seed 42,
# t_goal, 100k steps, checkpoint every 10k, uniform slot weights, plain L2, and the SD-VAE
# obs encoder (the default since 2026-08-28 -- encoder_tag: vae). Overriding a default here
# is how two runs end up differing in something nobody recorded.
#
# son_suffix is NOT decoration: run_name carries it and run_name IS hydra.run.dir, so
# without it three ladder shapes share one directory and `training.resume: True` has them
# resume each other. corrupt-${corrupt_obs} is a bool and cannot tell three shapes apart.
#
# wandb goes to the pushT_tmrl project, so these are separable from the slot-weighting
# history in one place rather than by tag archaeology.
#
#   bash scripts/slurm/submit_tmrl_corruption.sh          # dry run
#   SUBMIT=1 bash scripts/slurm/submit_tmrl_corruption.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

SUBMIT="${SUBMIT:-}"
CFG=train_pusht_diffusion_search
PROJECT="${PROJECT:-pushT_tmrl}"
# geometric needs a parameter; 0.7 is the config's documented example. The startup print
# warns if a shape collapses adjacent slots into indistinguishable corruption levels.
GEO_DECAY="${GEO_DECAY:-0.7}"

# suffix | overrides
ARMS=(
  "_son-lint|slot_obs_noise.mode=linear_t"
  "_son-geo7|slot_obs_noise.mode=geometric +slot_obs_noise.decay=$GEO_DECAY"
  "_son-linsig|slot_obs_noise.mode=linear_signal"
)

# Alternate the account: pick_gpu.sh reports what is free RIGHT NOW, but a job submitted
# seconds earlier has not registered against the association quota yet, so calling it once
# per arm piles them all onto the first account and the tail lands in PD AssocGrpCpuLimit.
ACCTS=(robotics weirdlab gpu-l40-weirdlab)
PARTS=(gpu-a40  gpu-a40  gpu-l40)

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search/pusht_image_search

n=0
for arm in "${ARMS[@]}"; do
    suffix="${arm%%|*}"; ov="${arm#*|}"
    run="value_k16_ver-t_goal${suffix}_demos-30_seed-42"
    name="tr_$run"
    if grep -qxF "$name" <<<"$LIVE"; then
        printf '%-14s %s\n' "$suffix" "already training, skipping"; continue
    fi
    if [ -f "$ROOT/outer_inner/$run/checkpoints/step_0100000.ckpt" ]; then
        printf '%-14s %s\n' "$suffix" "already finished"; continue
    fi
    if [ -z "$SUBMIT" ]; then
        printf '%-14s WOULD SUBMIT  %s\n' "$suffix" "$ov"; n=$((n+1)); continue
    fi
    A=${ACCTS[$((n % ${#ACCTS[@]}))]}; P=${PARTS[$((n % ${#PARTS[@]}))]}
    jid=$(sbatch --parsable --account="$A" --partition="$P" --job-name="$name" \
          --export=ALL,CONFIG_NAME=$CFG scripts/slurm/train_pusht_search.sbatch \
          $ov "son_suffix=$suffix" "logging.project=$PROJECT")
    printf '%-14s submitted %s on %s/%s\n' "$suffix" "$jid" "$A" "$P"
    n=$((n+1))
done
echo
echo "arms handled: $n   (wandb project: $PROJECT)"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
