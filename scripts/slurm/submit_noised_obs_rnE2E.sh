#!/bin/bash
# Per-slot OBSERVATION-NOISE ladders on the ResNet18-end-to-end backbone. TRAINING ONLY.
#
# Four noise schedules over the K=16 candidate slots, against three uncorrupted baselines.
# Slot k conditions on the first k scored candidates, so the ladder always puts the MOST
# corrupted observation at slot 0 (no context) and the cleanest at slot 15: a slot that
# cannot see well has to explore, a slot with full context sharpens.
#
#   1  linear_t       t_k falls linearly in the TIMESTEP index
#   2  linear_signal  t_k chosen so sqrt(alpha_bar) -- the factor the obs is actually
#                     scaled by -- falls in equal steps. "Linear in a_bar."
#   3  geometric      t_k = cap * 0.85^k
#   4  random_base    slot 0's timestep is drawn per sample from [0, 999] and the
#                     linear_signal curve is rescaled into [0, that draw], so what varies
#                     between samples is the ladder's EXTENT, not its shape.
#
# 1-3 at max_cap 999 and 400; 4 at 999 only. `max_t: 999` is the full extent, i.e. bit-
# identical to omitting it -- it is written out anyway so every arm's config states its cap
# rather than leaving one of the two implicit.
#
# THE DECAY IS 0.85, NOT 0.7. At K=16 on this schedule decay 0.7 leaves 6 of 15 adjacent
# slots within 0.005 of each other in sqrt(alpha_bar) (8 of 15 at cap 400) -- most of the
# ladder is then the same observation, and "geometric lost" would not be separable from
# "geometric collapsed". 0.85 keeps all 16 levels distinct at cap 999. The trade is that
# slot 15 lands at t=137 rather than 0, so the geometric arm's cleanest slot is not fully
# clean; the startup print reports the realized ladder for every arm.
#
# BASELINES ARE RETRAINED, not read off disk. The older BC / k=1 / k16 runs predate
# ${enc_suffix} in run_name and predate the random_base dispatch fix in this same policy
# file, so retraining is what puts every arm in the table on one code state. They land in
# NEW directories (`_ver-t_goal_enc-resnet18`), so the existing runs are untouched.
#
# Everything not named here is the config default and is deliberately NOT overridden:
# k=16, 30 demos, seed 42, t_goal, 100k gradient steps, checkpoint every 10k, 4/4/256,
# uniform slot weights, l2 slot loss, corrupt_obs False, selection argmax.
#
# son_suffix / son_tag / obs_noise_tag are NOT decoration. run_name IS hydra.run.dir and
# `training.resume: True` finds latest.ckpt there, so two ladders sharing a name would
# resume each other; the tags are how these runs are found in wandb. The workspace's
# _check_obs_noise_labels refuses to start if any of the three disagrees with the ladder.
#
# EVAL IS NOT DONE HERE -- see scripts/slurm/submit_noised_obs_readouts.sh.
#
#   bash scripts/slurm/submit_noised_obs_rnE2E.sh          # dry run
#   SUBMIT=1 bash scripts/slurm/submit_noised_obs_rnE2E.sh
# -f: $ov is expanded UNQUOTED at the sbatch call (it is many words), which would let
# `+slot_obs_noise.base_range=[0,999]` be read as a glob. Nothing here needs globbing.
set -ufo pipefail
cd "$(dirname "$0")/../.."

SUBMIT="${SUBMIT:-}"
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search/pusht_image_search_imgonly
CFG=train_pusht_diffusion_search

# The ladder overrides go on the TOP-LEVEL slot_obs_noise block, not on policy.*: the policy
# block is `slot_obs_noise: ${slot_obs_noise}`, and hydra cannot reach inside an
# interpolation (it refuses with "Could not override policy.slot_obs_noise.mode").
# `max_t` / `shape` / `base_range` are commented out in that block, so they need a leading +.
LADDER="slot_obs_noise.mode"

# config | trainer subdir | run_name | overrides
#
# The two ST baselines use DIFFERENT configs and it is not cosmetic -- `trainer` is a path
# component of hydra.run.dir. k=1 goes through ..._single (trainer: offline) because there
# is no search cost to amortize at width 1; k=16 uses the search default (outer_inner).
ARMS=(
  # ---- baselines, no ladder ------------------------------------------------------------
  "train_pusht_unet_bc|unet_bc|unetbc_ver-t_goal_enc-resnet18_demos-30_seed-42|"
  "train_pusht_diffusion_search_single|offline|value_k1_ver-t_goal_enc-resnet18_demos-30_seed-42|n_candidates=1"
  "train_pusht_diffusion_search|outer_inner|value_k16_ver-t_goal_enc-resnet18_demos-30_seed-42|"
  # ---- 1. linear in t ------------------------------------------------------------------
  "$CFG|outer_inner|value_k16_ver-t_goal_son-lint-cap999_enc-resnet18_demos-30_seed-42|$LADDER=linear_t +slot_obs_noise.max_t=999 son_suffix=_son-lint-cap999 son_tag=linear_t obs_noise_tag=obs_noised"
  "$CFG|outer_inner|value_k16_ver-t_goal_son-lint-cap400_enc-resnet18_demos-30_seed-42|$LADDER=linear_t +slot_obs_noise.max_t=400 son_suffix=_son-lint-cap400 son_tag=linear_t obs_noise_tag=obs_noised"
  # ---- 2. linear in a_bar --------------------------------------------------------------
  "$CFG|outer_inner|value_k16_ver-t_goal_son-linsig-cap999_enc-resnet18_demos-30_seed-42|$LADDER=linear_signal +slot_obs_noise.max_t=999 son_suffix=_son-linsig-cap999 son_tag=linear_signal obs_noise_tag=obs_noised"
  "$CFG|outer_inner|value_k16_ver-t_goal_son-linsig-cap400_enc-resnet18_demos-30_seed-42|$LADDER=linear_signal +slot_obs_noise.max_t=400 son_suffix=_son-linsig-cap400 son_tag=linear_signal obs_noise_tag=obs_noised"
  # ---- 3. geometric in t ---------------------------------------------------------------
  "$CFG|outer_inner|value_k16_ver-t_goal_son-geo85-cap999_enc-resnet18_demos-30_seed-42|$LADDER=geometric +slot_obs_noise.decay=0.85 +slot_obs_noise.max_t=999 son_suffix=_son-geo85-cap999 son_tag=geometric-d0.85 obs_noise_tag=obs_noised"
  "$CFG|outer_inner|value_k16_ver-t_goal_son-geo85-cap400_enc-resnet18_demos-30_seed-42|$LADDER=geometric +slot_obs_noise.decay=0.85 +slot_obs_noise.max_t=400 son_suffix=_son-geo85-cap400 son_tag=geometric-d0.85 obs_noise_tag=obs_noised"
  # ---- 4. random base, decaying in a_bar -----------------------------------------------
  "$CFG|outer_inner|value_k16_ver-t_goal_son-rndlinsig-cap999_enc-resnet18_demos-30_seed-42|$LADDER=random_base +slot_obs_noise.shape=linear_signal +slot_obs_noise.base_range=[0,999] son_suffix=_son-rndlinsig-cap999 son_tag=random_base-linsig obs_noise_tag=obs_noised"
)

# ROUND-ROBIN the account/partition explicitly rather than calling pick_gpu.sh per arm.
# pick_gpu.sh reports what is free RIGHT NOW, but a job submitted seconds earlier has not
# registered against the association quota yet -- so ten calls in a row all returned
# robotics/gpu-a40 (3 free GPUs) and all ten arms queued behind each other. Alternating over
# the guaranteed robotics/weirdlab pools puts two arms on each. Same fix, and the same
# reason, as submit_slot_weight_dtgoal.sh.
ACCTS=(robotics weirdlab weirdlab weirdlab robotics)
PARTS=(gpu-a40  gpu-a40  gpu-l40  gpu-l40s gpu-a100)

# WALL CLOCK, and why it is not the sbatch's own 10 days. A job is only scheduled if it fits
# before the next cluster MAINTENANCE reservation (`scontrol show reservation`), and those
# land every few weeks -- so a 10-day request submitted 9 days out from one sits at
# `ReqNodeNotAvail, Reserved for maintenance` until the window PASSES, holding no GPU and
# making no progress. All ten arms did exactly that on 2026-08-30 against the September8
# window. A shorter limit costs nothing here: `training.resume: True` plus a checkpoint every
# 10k means an arm that hits the wall continues from latest.ckpt on the next submission, so
# the wall is a resume point rather than a loss. Raise it when the next window is further out
#   scontrol show reservation | grep -A1 Maintenance
TRAIN_TIME="${TRAIN_TIME:-7-12:00:00}"

# A STALE WANDB_API_KEY IN THE ENVIRONMENT KILLS THE WHOLE GRID. `--export=ALL` snapshots
# the submitting shell, and wandb prefers $WANDB_API_KEY over ~/.netrc -- so an expired key
# exported from an interactive/agent session reaches every node and each arm dies at
# wandb.init with "Invalid or missing api_key", about a minute in. That is how 9 of these 10
# arms failed on 2026-08-30 while an identical run submitted from a plain shell trained fine.
# Drop it here (only when the netrc fallback actually exists, so a machine that authenticates
# by env var alone is left alone) and let the node authenticate the way it always has.
if [ -n "${WANDB_API_KEY:-}" ] && grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "note: dropping WANDB_API_KEY from the exported env; ~/.netrc is the credential"
    unset WANDB_API_KEY
fi

# The wandb run name, so the board shows the schedule rather than a timestamp. run_name
# already carries arm/k/verifier/ladder/encoder/demos/seed, and ${gen_tag} appends rnE2E.
# Quoted at the sbatch call: hydra must receive the ${...} literally.
NAME_FMT='logging.name=${now:%Y.%m.%d-%H.%M.%S}_${run_name}_${gen_tag}'
# The board these runs belong to (wandb.ai/l2sml/tmrl_ST). Override to put them elsewhere.
PROJECT="${WANDB_PROJECT:-tmrl_ST}"

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)

n=0
for arm in "${ARMS[@]}"; do
    IFS='|' read -r cfg sub run ov <<<"$arm"
    name="tr_$run"
    # Already training, or already finished? Resubmitting a live run is how two jobs end up
    # writing one hydra.run.dir; resubmitting a finished one silently continues from
    # latest.ckpt and appends steps past max_gradient_steps.
    if grep -qxF "$name" <<<"$LIVE"; then
        printf '%-62s %s\n' "$run" "already training, skipping"; continue
    fi
    if [ -f "$ROOT/$sub/$run/checkpoints/step_0100000.ckpt" ]; then
        printf '%-62s %s\n' "$run" "already finished (step_0100000.ckpt present)"; continue
    fi
    if [ -z "$SUBMIT" ]; then
        printf '%-62s WOULD SUBMIT  %s\n' "$run" "$cfg${ov:+ [$ov]}"; n=$((n+1)); continue
    fi
    A=${ACCTS[$((n % ${#ACCTS[@]}))]}; P=${PARTS[$((n % ${#PARTS[@]}))]}
    jid=$(sbatch --parsable --account="$A" --partition="$P" --job-name="$name" \
          --time="$TRAIN_TIME" \
          --export=ALL,CONFIG_NAME="$cfg" scripts/slurm/train_pusht_search.sbatch \
          $ov "run_name=$run" "logging.project=$PROJECT" "$NAME_FMT")
    printf '%-62s submitted %s on %s/%s\n' "$run" "$jid" "$A" "$P"
    n=$((n+1))
done
echo
echo "arms handled: $n"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
