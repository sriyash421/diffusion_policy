#!/usr/bin/env bash
# PART 3 -- GOAL-ONLY OBSERVATION CORRUPTION, on the geometric splits. ST k=16, 2 runs.
#
# Noise is added to the observation IMAGE inside a mask around the goal T, before the
# encoder. This is NOT slot_obs_noise: that corrupts the ENCODED vector, which has no
# spatial structure, so "noise only where the goal is" cannot be expressed there. The two
# are different mechanisms and their numbers do not compare.
#
# THE MASK IS FIXED. PushTEnv pins goal_pose = [256, 256, pi/4] for every episode, so the
# goal T occupies the same pixels in every frame; the mask is built once and only the noise
# is redrawn, fresh per sample per epoch (torch's RNG, which the dataloader re-seeds per
# worker -- numpy's is not, and workers would share one stream).
#
# margin_px IS IN IMAGE PIXELS of the 96px observation, not the 512px arena. The goal T
# spans only ~26 image px, so the two readings differ by 5.3x. Coverage: 2.1% at margin 0,
# 5.8% at 3, 8.2% at 5. Renders: media/goal_mask/goal_mask_variants.png.
#
# TRAIN SPLIT ONLY. PushTImageDataset._split_copy clears the flag, so val and test are clean
# and the env runner never sees it -- this is a training-time ablation, and every rollout
# number is measured on an uncorrupted observation. Verified by scripts/goal_mask_smoke.py.
#
#   bash scripts/run_goalmask_geometric.sh            # dry run
#   SUBMIT=1 bash scripts/run_goalmask_geometric.sh   # ...and sbatch
set -uo pipefail
cd "$(dirname "$0")/.."

SUBMIT="${SUBMIT:-}"
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search
PY="${DP_PY:-python}"
SPL=diffusion_policy/config/splits
# Capped so the job ends before the next maintenance reservation -- SLURM parks a job whose
# wall clock runs past one, which has cost this project three delays. training.resume picks
# up from latest.ckpt, so a short window costs a requeue, not progress.
TIME_LIMIT="${TIME_LIMIT:-$(bash scripts/slurm/safe_time_limit.sh)}"
# Nodes to keep off. A GPU with failing memory takes a job down minutes in with
# `CUDA error: uncorrectable ECC error encountered`, and SLURM does not drain the node for
# it -- g3070 stayed State=MIXED and kept accepting work after killing a run on 2026-09-04.
# Comma-separated, e.g. EXCLUDE_NODES=g3070,g3081
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
EXCL_FLAG=""; [ -n "$EXCLUDE_NODES" ] && EXCL_FLAG="--exclude=$EXCLUDE_NODES"
T="${GM_T:-800}"
MARGIN="${GM_MARGIN:-3}"
# THE THREE REPAIRS to the original _gm-t800-T3px arm, which scored ~0.00 everywhere. Its
# mask sits at the fixed goal pose, and since the task is to push the block ONTO the goal,
# at t=800 the block was destroyed exactly during the endgame -- the policy could never see
# the configuration it had to solve.
#   GM_EXCLUDE_BLOCK=1                subtract the block's own pixels per frame (the fix)
#   GM_T=400                          block stays legible through the noise
#   GM_MARGIN=0                       smallest footprint, T pixels only (2.1% vs 5.8%)
XB="${GM_EXCLUDE_BLOCK:-0}"
if [ "$XB" = "1" ]; then
    GM_CFG="{t:${T},margin_px:${MARGIN},exclude_block:true}"
    GM_SUF="_gm-t${T}-T${MARGIN}px-xb"
else
    GM_CFG="{t:${T},margin_px:${MARGIN}}"
    GM_SUF="_gm-t${T}-T${MARGIN}px"
fi
# Training budget. Overriding this is how an arm is EXTENDED past its original run: cfg is
# NOT restored from the checkpoint (BaseWorkspace.include_keys carries only the step
# counters), so the new value wins and `training.resume: True` continues from latest.ckpt.
# Safe to raise at any time because the LR schedule is `decay_then_constant`, a pure
# function of the absolute step -- warmup 3k, cosine to 80k, then flat at 0.1x base
# forever. A cosine schedule would climb back to peak past its horizon and silently start a
# second cycle on the extended run (see pusht_base.yaml:279).
MAX_STEPS="${MAX_STEPS:-100000}"

DATASETS=(
  "_split-blq|$SPL/pusht_blockquad_bottomleft_train137.json|137|50|19"
  "_split-brd60|$SPL/pusht_blockborder_train60_core50.json|60|50|40"
)

# WALK THE LIST, DO NOT RE-ASK. hyakalloc reports the scheduler's view, which does not
# reflect a job submitted seconds ago, so calling pick_gpu.sh once per arm returns the same
# answer every time and piles every job onto one partition -- where they sit in
# AssocGrpCpuLimit rather than failing. pick_gpu.sh's own header asks multi-job callers to
# do this. GPU_OFFSET staggers successive invocations of this script so three back-to-back
# variants do not all start at the same partition.
mapfile -t GPU_TARGETS < <(bash scripts/slurm/pick_gpu.sh 2>/dev/null || true)
gpu_i="${GPU_OFFSET:-0}"
if [ "${#GPU_TARGETS[@]}" -gt 0 ]; then
    printf 'free now: %s\n' "$(printf '%s; ' "${GPU_TARGETS[@]}")"
fi

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)
if [ -n "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "WARNING: WANDB_API_KEY is set and ~/.netrc has no wandb entry; keeping it" >&2
else
    unset WANDB_API_KEY
fi
if ! $PY -c 'import hydra' 2>/dev/null; then
    echo "ERROR: '$PY' cannot import hydra. conda activate \${DP_CONDA_ENV:-vae_pushT_l2s}" >&2; exit 1
fi

n=0; skipped=0
for ds in "${DATASETS[@]}"; do
    IFS='|' read -r tag file demos ntest nval <<<"$ds"
    [ -f "$file" ] || { echo "MISSING MANIFEST: $file" >&2; exit 1; }
    # gm_suffix is LOAD-BEARING: two levels sharing a run name would resolve to one
    # hydra.run.dir and `training.resume: True` would continue one under the other's data.
    ov="n_candidates=16 +goal_mask_noise=${GM_CFG}"
    ov="$ov training.max_gradient_steps=${MAX_STEPS}"
    ov="$ov gm_suffix=${GM_SUF}"
    ov="$ov task.dataset.split_file=$file n_demos=$demos"
    ov="$ov task.dataset.n_test_episodes=$ntest task.dataset.n_val_episodes=$nval"
    ov="$ov split_suffix=$tag"
    err=$(mktemp)
    resolved=$($PY train.py --config-name=train_pusht_diffusion_search $ov --cfg job --resolve 2>"$err") || true
    run=$(sed -n 's/^run_name: //p'   <<<"$resolved")
    sub=$(sed -n 's/^trainer: //p'    <<<"$resolved")
    task=$(sed -n 's/^task_name: //p' <<<"$resolved")
    if [ -z "$run" ] || [ -z "$sub" ] || [ -z "$task" ]; then
        printf '%-74s %s\n' "$tag" "CONFIG FAILED TO RESOLVE, skipping"
        sed -e 's/^/    | /' "$err" | tail -5 >&2; rm -f "$err"; skipped=$((skipped+1)); continue
    fi
    rm -f "$err"
    dir="$ROOT/$task/$sub/$run"; name="tr_$run"
    if grep -qxF "$name" <<<"$LIVE"; then
        printf '%-74s %s\n' "$run" "already training, skipping"; skipped=$((skipped+1)); continue
    fi
    # Keyed on MAX_STEPS, not a literal 100000: with the old constant an extension to 300k
    # was skipped as "already finished" the moment the 100k checkpoint existed.
    if [ -f "$dir/checkpoints/$(printf 'step_%07d.ckpt' "$MAX_STEPS")" ]; then
        printf '%-74s %s\n' "$run" "already finished, skipping"; skipped=$((skipped+1)); continue
    fi
    if [ -z "$SUBMIT" ]; then
        printf '%-74s WOULD SUBMIT  t=%s margin=%spx\n' "$run" "$T" "$MARGIN"; n=$((n+1)); continue
    fi
    # pick_gpu reports partitions with capacity FREE RIGHT NOW. When the queue is full it
    # reports nothing, and refusing to submit would mean these arms never enter the queue at
    # all -- the other geometric arms all queued at Reason=Priority and started as capacity
    # freed. So fall back to an explicit account/partition and let SLURM queue it.
    if [ -n "${FORCE_ACCOUNT:-}" ] && [ -n "${FORCE_PARTITION:-}" ]; then
        A="$FORCE_ACCOUNT"; P="$FORCE_PARTITION"
    elif [ "${#GPU_TARGETS[@]}" -gt 0 ]; then
        read -r A P <<<"${GPU_TARGETS[$(( gpu_i % ${#GPU_TARGETS[@]} ))]}"
        gpu_i=$((gpu_i + 1))
    else
        A="${FALLBACK_ACCOUNT:-robotics}"; P="${FALLBACK_PARTITION:-gpu-a40}"
        echo "  (no free GPU now; queueing on $A/$P)"
    fi
    jid=$(sbatch --parsable --account="$A" --partition="$P" --job-name="$name" \
          --time="$TIME_LIMIT" $EXCL_FLAG --export=ALL,CONFIG_NAME=train_pusht_diffusion_search \
          scripts/slurm/train_pusht_search.sbatch $ov)
    printf '%-74s submitted %s on %s/%s\n' "$run" "$jid" "$A" "$P"
    n=$((n+1))
done
echo
echo "arms handled: $n   skipped: $skipped"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
