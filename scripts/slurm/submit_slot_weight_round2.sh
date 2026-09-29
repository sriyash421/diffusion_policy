#!/bin/bash
# Round 2 of the slot-weighting sweep: six k=16 arms at a 100:1 endpoint spread.
#
# Round 1 (linear r=4.857 and slot_loss_norm l2tol1) moved nothing -- both variants sat
# within ~0.03 of the uniform control in every cell of a 70-cell n-sweep, against a +/-0.13
# per-cell CI. The hypothesis here is that 4.857:1 across 16 slots was simply too gentle a
# tilt, so this round goes to 100:1 and asks whether SHAPE (straight ramp vs sharp late
# spike) or a uniform warm-up matters at that spread.
#
#   linear r=100      w_k prop 1 + 99k/15        slot 0 = 0.0198 -> slot 15 = 1.9802
#   geometric d=0.735 w_k prop 0.735^(15-k)      slot 0 = 0.0422 -> slot 15 = 4.271
#                     (stays under 1.0 until slot 10, then spikes; endpoint ratio 101.3:1)
#   curriculum        30k steps on UNIFORM, then linear r=100 for the remaining 70k
#
# crossed with slot_loss_norm in {l2, l2tol1}, except the curriculum which is linear-only.
# Everything else is the config default and is NOT overridden: k=16, 30 demos, seed 42,
# 100k gradient steps, checkpoint every 10k, selection argmax, corrupt_obs False.
# Overriding a default here is how two runs end up differing in something nobody recorded.
#
# armTn IS THE ONE EXCEPTION, and it is pinned rather than inherited (see VER below). These
# arms' directories are `..._ver-armTn_sw-...`, and on 2026-08-28 the pusht_base default
# flipped to t_goal. Inheriting it would resolve every arm here to a `ver-t_goal_sw-...`
# path that holds no latest.ckpt, so a re-run would silently start eight fresh runs instead
# of resuming the eight on disk -- the AUDIT.md 9.9 failure. run_name is hydra.run.dir.
#
# sw_suffix is NOT decoration: hydra.run.dir is a pure function of run_name and
# `training.resume: True` finds latest.ckpt there, so two profiles sharing a name would
# silently resume each other's run. Each suffix names the profile, its ratio, and the norm.
#
#   bash scripts/slurm/submit_slot_weight_round2.sh          # dry run
#   SUBMIT=1 bash scripts/slurm/submit_slot_weight_round2.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

SUBMIT="${SUBMIT:-}"
CFG=train_pusht_diffusion_search
# Pinned, NOT inherited -- see the armTn note in the header block above.
VER=armTn
WP='slot_weights.waypoints=[{step:0,mode:uniform},{step:30000,mode:linear,ratio:100}]'

# suffix | overrides.  The suffix IS the run identity; see above.
ARMS=(
  "_sw-lin100-l2|slot_weights.mode=linear slot_weights.ratio=100"
  "_sw-lin100-l2tol1|slot_weights.mode=linear slot_weights.ratio=100 slot_loss_norm.mode=l2tol1"
  "_sw-geo735-l2|slot_weights.mode=geometric slot_weights.decay=0.735"
  "_sw-geo735-l2tol1|slot_weights.mode=geometric slot_weights.decay=0.735 slot_loss_norm.mode=l2tol1"
  "_sw-curr-lin100-l2|slot_weights.mode=curriculum slot_weights.interp=step $WP"
  "_sw-curr-lin100-l2tol1|slot_weights.mode=curriculum slot_weights.interp=step $WP slot_loss_norm.mode=l2tol1"
)

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search/pusht_image_search

n=0
for arm in "${ARMS[@]}"; do
    suffix="${arm%%|*}"; ov="${arm#*|}"
    run="value_k16_ver-armTn${suffix}_corrupt-False_demos-30_seed-42"
    name="tr_$run"
    # Already training, or already finished? Resubmitting a live run is how two jobs end up
    # writing one hydra.run.dir; resubmitting a finished one silently continues from
    # latest.ckpt and appends steps past max_gradient_steps.
    if grep -qxF "$name" <<<"$LIVE"; then
        printf '%-24s %s\n' "$suffix" "already training, skipping"; continue
    fi
    if [ -f "$ROOT/outer_inner/$run/checkpoints/step_0100000.ckpt" ]; then
        printf '%-24s %s\n' "$suffix" "already finished (step_0100000.ckpt present)"; continue
    fi
    if [ -z "$SUBMIT" ]; then
        printf '%-24s WOULD SUBMIT  %s\n' "$suffix" "$ov"; n=$((n+1)); continue
    fi
    # One pick_gpu call per arm: it reports whichever robotics/weirdlab partition has free
    # GPUs right now, and six submissions in a row would otherwise all pile onto the first.
    read -r A P < <(bash scripts/slurm/pick_gpu.sh) || { echo "no free GPU for $suffix" >&2; continue; }
    jid=$(sbatch --parsable --account="$A" --partition="$P" --job-name="$name" \
          --export=ALL,CONFIG_NAME=$CFG scripts/slurm/train_pusht_search.sbatch \
          $ov "verifier_tag=$VER" "sw_suffix=$suffix")
    printf '%-24s submitted %s on %s/%s\n' "$suffix" "$jid" "$A" "$P"
    n=$((n+1))
done
echo
echo "arms handled: $n"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
