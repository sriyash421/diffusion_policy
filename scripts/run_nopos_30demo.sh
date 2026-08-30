#!/usr/bin/env bash
# RETIRED 2026-08-29. Kept as the record of a finished experiment; DO NOT RELAUNCH.
#
# The two hydra deletions below (`~task.shape_meta.obs.agent_pos` / `.feedback`) now target
# keys that no longer exist: the image-only observation stopped being an ablation and became
# the task. config/task/pusht_image_search_imgonly.yaml is the only PushT search task, its
# shape_meta declares `image` alone, and the policy asserts that no low_dim obs key is
# declared -- so this script raises rather than reproducing anything.
#
# Its successor is scripts/run_vae_nopos_30demo.sh, which is the same three arms on the
# frozen SD-VAE encoder. The runs this script produced are under
# .../pusht_image_search/ (with-pos task tree) and are untouched.
# IMAGE-ONLY ABLATION, 30 demos: retrain the three headline arms with `agent_pos` and
# `feedback` REMOVED from the policy observation. TRAINING ONLY -- see EVAL below.
#
# WHY. Every PushT arm to date conditions on shape_meta.obs = {image, agent_pos, feedback}.
# `feedback` is an exact, invertible transform of `block_pos` (feedback_util), so the
# ground-truth T pose has been handed to the policy in closed form rather than extracted
# from pixels -- a stronger observation than the standard PushT-image setup, and one that
# makes these success rates non-comparable to published image baselines. These three runs
# measure what those two low_dim keys were actually buying.
#
# THE VERIFIER STILL SEES EVERYTHING. It resets a pymunk sim and cannot work otherwise:
# PushTSearchMixin._verifier_inputs slices `agent_pos`/`feedback` straight off the obs dict,
# which the dataset and the env wrapper still emit. This ablation changes what the POLICY
# sees, not what the SEARCH can score. `_normalize_value` likewise still finds
# normalizer['feedback'].
#
# HOW IT IS EXPRESSED. Two hydra key deletions, and nothing else:
#
#   '~task.shape_meta.obs.agent_pos' '~task.shape_meta.obs.feedback'
#
# shape_meta is read by exactly two things -- policy.shape_meta and
# policy.obs_encoder.shape_meta, both via the top-level `shape_meta: ${task.shape_meta}` in
# pusht_base. The dataset and the env runner never look at it. So the deletions shrink
# MultiImageObsEncoder's output from 530 to 512 and change NOTHING else: the dataset still
# emits all three keys and the normalizer still holds params for all three, so
# normalizer.normalize(obs_dict) cannot KeyError; the ST encoder simply never reads them and
# the UNet's _select_obs drops them. (There is no YAML-alias hazard: `&shape_meta` is
# declared in pusht_image.yaml but never aliased.)
#
# ARCHITECTURE-CHANGING: obs_emb becomes 512->256 and the UNet's global_cond 1060->1024, so
# these checkpoints are NOT loadable against the with-pos controls. Hence new run dirs.
#
# run_name IS OVERRIDDEN EXPLICITLY on every arm. Without it the ST arms resolve to
# value_k{1,16}_ver-t_goal_corrupt-False_demos-30_seed-42 -- which is what a with-pos run
# launched today also resolves to, so hydra.run.dir would collide and one would resume into
# the other (AUDIT.md 9.9).
#
# MATCHED CONTROLS, already trained and already fully scored under t_goal at every 10k step
# across n = 1..64. Nothing needs re-evaluating on the with-pos side:
#   outer_inner/value_k16_corrupt-False_demos-30_seed-42
#   offline/value_k1_demos-30_seed-42          <- ST k=1, legacy name
#   unet_bc/unetbc_demos-30_seed-42
#
# EVAL IS NOT DONE HERE. The watcher owns it -- eval_watch_pusht_search.sbatch sweeps
# n = 1 2 4 8 16 32 64 over the 50 test episodes with --skip-val for every step_*.ckpt,
# which is the every-10k-checkpoint eval this ablation asks for and the same protocol the
# controls were measured under. After the first checkpoints land:
#   SUBMIT=1 bash scripts/slurm/submit_30_100_watchers.sh
# then: python scripts/build_nopos_doc.py   ->  success_rates_no_pos.md
#
#   bash scripts/run_nopos_30demo.sh            # dry run: show what would be submitted
#   SUBMIT=1 bash scripts/run_nopos_30demo.sh   # ...and sbatch them
set -uo pipefail
cd "$(dirname "$0")/.."

SUBMIT="${SUBMIT:-}"
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search/pusht_image_search

# The whole ablation, in two tokens. Quoted at every use: `~` is tilde-expansion bait.
NOPOS="~task.shape_meta.obs.agent_pos ~task.shape_meta.obs.feedback"

# config | trainer subdir | run_name | extra overrides
#
# The two ST arms use DIFFERENT configs and it is not cosmetic -- `trainer` is a path
# component of hydra.run.dir. k=1 goes through ..._single (trainer: offline) because there
# is no search cost to amortize at width 1, matching where the k=1 controls live; k=16 uses
# the search default (trainer: outer_inner). Everything else is the config default and is
# NOT overridden: 4/4/256, seed 42, t_goal, 100k gradient steps, checkpoint every 10k,
# selection argmax, corrupt_obs False, uniform slot weights.
ARMS=(
  "train_pusht_diffusion_search_single|offline|value_k1_ver-t_goal_nopos_corrupt-False_demos-30_seed-42|n_candidates=1"
  "train_pusht_diffusion_search|outer_inner|value_k16_ver-t_goal_nopos_corrupt-False_demos-30_seed-42|n_candidates=16"
  "train_pusht_unet_bc|unet_bc|unetbc_ver-t_goal_nopos_demos-30_seed-42|"
)

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)

n=0
for arm in "${ARMS[@]}"; do
    IFS='|' read -r cfg sub run ov <<<"$arm"
    name="tr_$run"
    # Already training, or already finished? Resubmitting a live run is how two jobs end up
    # writing one hydra.run.dir; resubmitting a finished one silently continues from
    # latest.ckpt and appends steps past max_gradient_steps.
    if grep -qxF "$name" <<<"$LIVE"; then
        printf '%-58s %s\n' "$run" "already training, skipping"; continue
    fi
    if [ -f "$ROOT/$sub/$run/checkpoints/step_0100000.ckpt" ]; then
        printf '%-58s %s\n' "$run" "already finished (step_0100000.ckpt present)"; continue
    fi
    if [ -z "$SUBMIT" ]; then
        printf '%-58s WOULD SUBMIT  %s [%s]\n' "$run" "$cfg" "$ov"; n=$((n+1)); continue
    fi
    # One pick_gpu call per arm: it reports whichever robotics/weirdlab partition has free
    # GPUs right now, and three submissions in a row would otherwise all pile onto the first.
    read -r A P < <(bash scripts/slurm/pick_gpu.sh) || { echo "no free GPU for $run" >&2; continue; }
    jid=$(sbatch --parsable --account="$A" --partition="$P" --job-name="$name" \
          --export=ALL,CONFIG_NAME="$cfg" scripts/slurm/train_pusht_search.sbatch \
          $NOPOS $ov "run_name=$run")
    printf '%-58s submitted %s on %s/%s\n' "$run" "$jid" "$A" "$P"
    n=$((n+1))
done
echo
echo "arms handled: $n"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
