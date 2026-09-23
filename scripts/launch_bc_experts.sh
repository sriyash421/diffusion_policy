#!/bin/bash
# Launch the two PushT BC expert arms and their success-rate watchers.
#
# WHAT THIS TRAINS. Behaviour cloning on ALL 176 demonstrations -- every episode except the
# 30 held out -- at two observations:
#   image     train_pusht_unet_bc_expert           -> .../pusht_image_search_imgonly/unet_bc/
#   keypoint  train_pusht_unet_bc_keypoint_expert  -> .../pusht_keypoint_manifest/unet_bc/
# Both write a checkpoint every 10k gradient steps to a 200k budget, i.e. 20 checkpoints,
# and a watcher scores each one on the 30 held-out episodes as it appears. The expert
# checkpoint is then chosen BY HAND off that curve -- nothing here nominates one.
#
#   bash scripts/launch_bc_experts.sh          # show what would be submitted
#   SUBMIT=1 bash scripts/launch_bc_experts.sh # actually submit
#   SUBMIT=1 ARMS=image bash scripts/launch_bc_experts.sh
#   SUBMIT=1 TRAIN_ONLY=1 bash scripts/launch_bc_experts.sh   # skip the watchers
set -euo pipefail
cd "$(dirname "$0")/.."

SUBMIT="${SUBMIT:-0}"
ARMS="${ARMS:-image keypoint}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
OUT_ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}/pusht_search"

# THE SHELL'S WANDB_API_KEY IS STALE ON KLONE and shadows the valid ~/.netrc, so a job that
# inherits it dies at wandb.init. --export=ALL is the sbatch default, so it must be unset
# HERE, in the submitting shell, not inside the job.
unset WANDB_API_KEY || true

# --time must be finite AND must end before the next maintenance reservation, or SLURM parks
# the job with Reason=ReqNodeNotAvail and it silently never starts.
# 1 day, against a measured ~0.1 s/step on the existing UNet BC runs -- so 200k steps is
# ~6h and this is ~4x headroom. Deliberately not larger: --time is what SLURM backfills on,
# so an inflated request just sits in the queue. A run that does outlive it resumes cleanly
# from latest.ckpt on resubmission (hydra.run.dir is derived from the config, not the clock).
TRAIN_TIME="${TRAIN_TIME:-1-00:00:00}"
next_maint=$(scontrol show reservation 2>/dev/null \
  | awk '/Maintenance/ {for (i=1;i<=NF;i++) if ($i ~ /^StartTime=/) {sub("StartTime=","",$i); print $i}}' \
  | sort | head -1)
echo "next maintenance reservation : ${next_maint:-none found}"
echo "requested --time             : $TRAIN_TIME"
echo

arm_config () { case "$1" in
  image)    echo train_pusht_unet_bc_expert ;;
  keypoint) echo train_pusht_unet_bc_keypoint_expert ;;
esac }
arm_rundir () { case "$1" in
  image)    echo "$OUT_ROOT/pusht_image_search_imgonly/unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-176_seed-42" ;;
  keypoint) echo "$OUT_ROOT/pusht_keypoint_manifest/unet_bc/unetbckp_demos-176_seed-42" ;;
esac }
# The image arm is scored by the established watcher, pinned to n=1 -- this is plain BC, so
# the per-checkpoint number is the single-sample rollout, not a best-of-n curve. --skip-val
# because this manifest has NO val split: the 30 held-out episodes are `test`.
arm_watcher () { case "$1" in
  image)    echo "scripts/slurm/eval_watch_pusht_search.sbatch|--n-list 1 --skip-val --idle-exit-sec 7200" ;;
  keypoint) echo "scripts/slurm/eval_watch_bc_keypoint.sbatch|--idle-exit-sec 7200" ;;
esac }

for arm in $ARMS; do
  cfg=$(arm_config "$arm"); run_dir=$(arm_rundir "$arm")
  IFS='|' read -r wsb wargs <<<"$(arm_watcher "$arm")"

  # A free GPU is not enough -- the account quotas are per-resource, so pick_gpu filters on
  # CPU and memory too. Called once PER ARM: hyakalloc reports the scheduler's view, which
  # does not reflect a job submitted seconds ago, so two calls in a row would pile both jobs
  # onto one partition. Walk the list instead.
  mapfile -t cands < <(bash scripts/slurm/pick_gpu.sh)
  idx=0; for a in $ARMS; do [ "$a" = "$arm" ] && break; idx=$((idx+1)); done
  pick="${cands[$((idx % ${#cands[@]}))]}"
  read -r ACCT PART <<<"$pick"

  echo "=== $arm ==="
  echo "  config   : $cfg"
  echo "  run dir  : $run_dir"
  echo "  train    : sbatch --account=$ACCT --partition=$PART --time=$TRAIN_TIME \\"
  echo "               --export=ALL,CONFIG_NAME=$cfg scripts/slurm/train_pusht_search.sbatch"
  echo "  watcher  : sbatch $wsb $run_dir $wargs"

  if [ "$SUBMIT" = "1" ]; then
    jid=$(sbatch --parsable --account="$ACCT" --partition="$PART" --time="$TRAIN_TIME" \
      --job-name="bc_expert_$arm" \
      --export=ALL,CONFIG_NAME="$cfg" scripts/slurm/train_pusht_search.sbatch)
    echo "  -> train job $jid"
    if [ "$TRAIN_ONLY" != "1" ]; then
      # The watcher polls for step_*.ckpt and tolerates the directory not existing yet, so
      # it does NOT need to wait on the training job -- and must not, since it is meant to
      # score checkpoints while training continues.
      wjid=$(sbatch --parsable --job-name="bc_expert_eval_$arm" "$wsb" "$run_dir" $wargs)
      echo "  -> watcher job $wjid"
    fi
  fi
  echo
done

if [ "$SUBMIT" != "1" ]; then
  echo "dry run -- re-run with SUBMIT=1 to submit"
fi
