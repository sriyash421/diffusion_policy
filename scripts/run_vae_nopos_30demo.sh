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
ARMS=(
  "train_pusht_unet_bc|"
  "train_pusht_diffusion_search_single|n_candidates=1"
  "train_pusht_diffusion_search|n_candidates=16"
)

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)

n=0
for arm in "${ARMS[@]}"; do
    IFS='|' read -r cfg ov <<<"$arm"
    # Ask the config, rather than repeating it. Also acts as a config check: a resolution
    # error here stops the arm instead of surfacing 30 seconds into a SLURM job.
    resolved=$($PY train.py --config-name="$cfg" $ov --cfg job --resolve 2>/dev/null) || true
    # Extracted BY NAME, not by position: --cfg job prints the keys in config order, which
    # is not the order of the path.
    run=$(sed -n 's/^run_name: //p'  <<<"$resolved")
    sub=$(sed -n 's/^trainer: //p'   <<<"$resolved")
    task=$(sed -n 's/^task_name: //p' <<<"$resolved")
    if [ -z "$run" ] || [ -z "$sub" ] || [ -z "$task" ]; then
        printf '%-56s %s\n' "$cfg" "CONFIG FAILED TO RESOLVE, skipping"; continue
    fi
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
