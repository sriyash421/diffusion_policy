#!/usr/bin/env bash
# THE GEOMETRIC-SPLIT SWEEP: 3 arms x 5 datasets, where train and eval differ by WHERE THE
# T STARTS rather than by a random draw. The five manifests are README_pusht.md 2.2.
#
#   arms      UNet BC | ST k=1 | ST k=16 with UNIFORM slot weights
#   datasets  blockquad {bottomleft, topright} | blockborder {30, 60, 100}
#
# ENCODER. No obs_encoder override: the default is ResNet18 trained END TO END (reverted
# from the frozen SD VAE on 2026-08-30 -- pusht_base.yaml:66 has the measurement), so these
# are `enc-resnet18` / gen_tag rnE2E runs. Do not read them against the enc-vae numbers.
#
# UNIFORM SLOT WEIGHTS AND NO LADDER are the DEFAULTS, which is why no arm below sets
# slot_weights or slot_obs_noise: sw_suffix and son_suffix are empty exactly when those are
# uniform, so an empty suffix in the run name IS the uniform claim.
#
# THE FOUR DATASET OVERRIDES GO TOGETHER, always:
#   task.dataset.split_file       which manifest
#   n_demos                       its train count -- PushTImageDataset VALIDATES this
#                                 against the manifest and raises on disagreement, so a
#                                 typo is a startup error rather than a silent mis-budget
#   n_test_episodes / n_val_episodes   likewise validated
#   split_suffix                  the run DIRECTORY. WITHOUT IT, blockborder_train30 and
#                                 the seed-42 train30 resolve to ONE hydra.run.dir and
#                                 `training.resume: True` continues one under the other's
#                                 data. n_demos alone does not separate them.
# env_runner needs nothing: it interpolates all four from task.dataset.
#
#   bash scripts/run_geometric_splits.sh            # dry run: show what would be submitted
#   SUBMIT=1 bash scripts/run_geometric_splits.sh   # ...and sbatch them
#
# EVAL IS NOT DONE HERE. See scripts/slurm/submit_geometric_readouts.sh.
set -uo pipefail
cd "$(dirname "$0")/.."

SUBMIT="${SUBMIT:-}"
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search
PY="${DP_PY:-python}"
SPL=diffusion_policy/config/splits
# WALL CLOCK. train_pusht_search.sbatch asks for 10 days, which SLURM will not schedule
# ahead of a maintenance reservation: on 2026-09-03 all 15 jobs sat at
# `ReqNodeNotAvail, Reserved for maintenance` with StartTime pushed 5.5 days out, past the
# Sept 8 window. Measured, these arms finish well inside 3 days -- BC UNet ~3h, ST k=1
# ~1.5h, ST k=16 ~29h -- so the 10-day ask buys nothing and costs the queue slot. Raise it
# if an arm ever grows; `training.resume: True` means a timed-out run continues from
# latest.ckpt on resubmission rather than restarting.
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
# Nodes to keep off. A GPU with failing memory takes a job down minutes in with
# `CUDA error: uncorrectable ECC error encountered`, and SLURM does not drain the node for
# it -- g3070 stayed State=MIXED and kept accepting work after killing a run on 2026-09-04.
# Comma-separated, e.g. EXCLUDE_NODES=g3070,g3081
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
EXCL_FLAG=""; [ -n "$EXCLUDE_NODES" ] && EXCL_FLAG="--exclude=$EXCLUDE_NODES"

# tag | split_file | n_demos | n_test | n_val
# n_val is 0 on two of them (topright's quadrant holds only 33 episodes, all of them test;
# blockborder-100 has no ring left over). The dataset skips the count check when
# n_val_episodes is falsy and the workspace's val loop is guarded on len(val_losses) > 0,
# so an empty val split is a supported configuration, not a hole.
DATASETS=(
  "_split-blq|$SPL/pusht_blockquad_bottomleft_train137.json|137|50|19"
  "_split-trq|$SPL/pusht_blockquad_topright_train173.json|173|33|0"
  "_split-brd30|$SPL/pusht_blockborder_train30_core50.json|30|50|70"
  "_split-brd60|$SPL/pusht_blockborder_train60_core50.json|60|50|40"
  "_split-brd100|$SPL/pusht_blockborder_train100_core50.json|100|50|0"
)

# config | extra overrides.
# n_candidates=1 IS LOAD-BEARING on the k=1 arm: train_pusht_diffusion_search_single pins
# the single-step TRAINER, not width 1, and inherits n_candidates: 16 from the search
# master. Without it the k=1 arm trains at width 16 under a name that says k1.
ARM_SPECS=(
  "train_pusht_unet_bc|"
  "train_pusht_diffusion_search_single|n_candidates=1"
  "train_pusht_diffusion_search|n_candidates=16"
)

# WANDB CREDENTIALS. `--export=ALL` copies the submitting shell's environment into every
# job, so a STALE WANDB_API_KEY there overrides ~/.netrc inside the job and wandb.init dies
# with `401 Unauthorized` / `Invalid or missing api_key` after the GPU is already allocated.
# That killed the first three arms of this sweep on 2026-09-03. ~/.netrc is the credential
# that is actually maintained, so drop the env var and let wandb read it.
if [ -n "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "WARNING: WANDB_API_KEY is set and ~/.netrc has no wandb entry; keeping the env var" >&2
else
    unset WANDB_API_KEY
fi

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)

# Fail loudly on the wrong interpreter rather than once per arm -- otherwise every --cfg job
# below dies on the hydra import and every arm skips with "CONFIG FAILED TO RESOLVE", which
# is indistinguishable from a genuinely broken config.
if ! $PY -c 'import hydra' 2>/dev/null; then
    echo "ERROR: '$PY' cannot import hydra. Activate the env first:" >&2
    echo "         conda activate \${DP_CONDA_ENV:-vae_pushT_l2s}" >&2
    echo "       or set DP_PY to an interpreter that has it." >&2
    exit 1
fi

n=0; skipped=0
for ds in "${DATASETS[@]}"; do
    IFS='|' read -r tag file demos ntest nval <<<"$ds"
    [ -f "$file" ] || { echo "MISSING MANIFEST: $file (run scripts/make_geometric_splits.py)" >&2; exit 1; }
    dov="task.dataset.split_file=$file n_demos=$demos"
    dov="$dov task.dataset.n_test_episodes=$ntest task.dataset.n_val_episodes=$nval"
    dov="$dov split_suffix=$tag"
    for arm in "${ARM_SPECS[@]}"; do
        IFS='|' read -r cfg aov <<<"$arm"
        ov="$aov $dov"
        # Ask the config rather than repeating it; a resolution error stops the arm here
        # instead of 30 seconds into a SLURM job.
        err=$(mktemp); resolved=$($PY train.py --config-name="$cfg" $ov --cfg job --resolve 2>"$err") || true
        # BY NAME, not by position: --cfg job prints keys in config order, not path order.
        run=$(sed -n 's/^run_name: //p'   <<<"$resolved")
        sub=$(sed -n 's/^trainer: //p'    <<<"$resolved")
        task=$(sed -n 's/^task_name: //p' <<<"$resolved")
        if [ -z "$run" ] || [ -z "$sub" ] || [ -z "$task" ]; then
            printf '%-64s %s\n' "$cfg $tag" "CONFIG FAILED TO RESOLVE, skipping"
            sed -e 's/^/    | /' "$err" | tail -5 >&2
            rm -f "$err"; skipped=$((skipped+1)); continue
        fi
        rm -f "$err"
        dir="$ROOT/$task/$sub/$run"
        name="tr_$run"
        # Resubmitting a live run is how two jobs end up writing one hydra.run.dir;
        # resubmitting a finished one silently continues past max_gradient_steps.
        if grep -qxF "$name" <<<"$LIVE"; then
            printf '%-64s %s\n' "$run" "already training, skipping"; skipped=$((skipped+1)); continue
        fi
        if [ -f "$dir/checkpoints/step_0100000.ckpt" ]; then
            printf '%-64s %s\n' "$run" "already finished, skipping"; skipped=$((skipped+1)); continue
        fi
        if [ -z "$SUBMIT" ]; then
            printf '%-64s WOULD SUBMIT  %s\n' "$run" "$cfg"
            printf '%-64s   -> %s\n' "" "$dir"
            n=$((n+1)); continue
        fi
        # One pick_gpu call per arm: it reports whichever robotics/weirdlab partition has
        # free GPUs right now, and N submissions in a row would pile onto the first.
        read -r A P < <(bash scripts/slurm/pick_gpu.sh) || { echo "no free GPU for $run" >&2; skipped=$((skipped+1)); continue; }
        jid=$(sbatch --parsable --account="$A" --partition="$P" --job-name="$name" \
              --time="$TIME_LIMIT" $EXCL_FLAG \
              --export=ALL,CONFIG_NAME="$cfg" scripts/slurm/train_pusht_search.sbatch $ov)
        printf '%-64s submitted %s on %s/%s\n' "$run" "$jid" "$A" "$P"
        n=$((n+1))
    done
done
echo
echo "arms handled: $n   skipped: $skipped"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
