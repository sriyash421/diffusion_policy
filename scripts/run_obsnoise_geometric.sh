#!/usr/bin/env bash
# OBS-NOISE ARMS ON THE GEOMETRIC SPLITS. ST k=16 only, 2 datasets x 5 ladders = 10 runs.
#
#   PART 1 -- FLAT. The same corruption level on EVERY candidate slot, at t = 800/400/200.
#     `mode: list` with 16 identical timesteps. This is deliberately not a ladder: it asks
#     what a uniformly degraded observation costs, as the control the graded arms are read
#     against. The startup print will warn "15/15 adjacent slot pairs differ by < 0.005 in
#     sqrt(alpha_bar)" -- for a flat profile that is the POINT, not a misconfiguration.
#
#   PART 2 -- HALF-RAMP. Slot 0 at the base timestep, slot 15 at HALF THAT TIMESTEP, linear
#     in t across the 16 slots. Note this is not half the corruption: alpha_bar is a
#     cumulative product, so t 800->400 moves sqrt(alpha_bar) 0.039 -> 0.440.
#
# LEVELS TRANSFER FROM THE SD-VAE UNCHANGED. _corrupt_obs_features scales the noise by
# `obs_feature_std`, a running per-dimension EMA, so t fixes an SNR rather than a magnitude.
# Measured on 32 real frames, mean-centred cosine at t = 800/400/200 is 0.041/0.448/0.792
# for ResNet18 against 0.003/0.426/0.810 for the VAE -- the same corruption. (Raw cosine
# disagrees wildly, 0.97 vs 0.28 at t=800, but that is the ReLU mean offset: ResNet features
# are non-negative with mean 0.77*sigma, and that DC component carries no per-sample
# information.) So the VAE-calibrated 800/400/200 are used as-is.
#
# ENCODER is the default ResNet18 trained end to end, matching scripts/run_geometric_splits.sh.
#
#   bash scripts/run_obsnoise_geometric.sh            # dry run
#   SUBMIT=1 bash scripts/run_obsnoise_geometric.sh   # ...and sbatch
set -uo pipefail
cd "$(dirname "$0")/.."

SUBMIT="${SUBMIT:-}"
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search
PY="${DP_PY:-python}"
SPL=diffusion_policy/config/splits
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
# Nodes to keep off. A GPU with failing memory takes a job down minutes in with
# `CUDA error: uncorrectable ECC error encountered`, and SLURM does not drain the node for
# it -- g3070 stayed State=MIXED and kept accepting work after killing a run on 2026-09-04.
# Comma-separated, e.g. EXCLUDE_NODES=g3070,g3081
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
EXCL_FLAG=""; [ -n "$EXCLUDE_NODES" ] && EXCL_FLAG="--exclude=$EXCLUDE_NODES"     # see run_geometric_splits.sh: 10d will not schedule

# tag | split_file | n_demos | n_test | n_val   -- must match run_geometric_splits.sh exactly
DATASETS=(
  "_split-blq|$SPL/pusht_blockquad_bottomleft_train137.json|137|50|19"
  "_split-brd60|$SPL/pusht_blockborder_train60_core50.json|60|50|40"
)

# Built here rather than pasted, so the ramp cannot drift from what the name claims.
mk_flat() { $PY -c "print('['+','.join(['$1']*16)+']')"; }
mk_ramp() { $PY -c "a,b=$1,$2; print('['+','.join(str(int(round(a-(a-b)*k/15))) for k in range(16))+']')"; }

ARMS=()
for t in 800 400 200; do
  ARMS+=("$(mk_flat $t)|_son-flat${t}|flat-${t}")
done
for base in 800 400; do
  half=$((base/2))
  ARMS+=("$(mk_ramp $base $half)|_son-ramp${base}to${half}|ramp-${base}-${half}")
done

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)

# See run_geometric_splits.sh: a STALE WANDB_API_KEY in the submitting shell overrides
# ~/.netrc inside the job and kills it at wandb.init with 401, after the GPU is allocated.
if [ -n "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "WARNING: WANDB_API_KEY is set and ~/.netrc has no wandb entry; keeping it" >&2
else
    unset WANDB_API_KEY
fi

if ! $PY -c 'import hydra' 2>/dev/null; then
    echo "ERROR: '$PY' cannot import hydra. conda activate \${DP_CONDA_ENV:-vae_pushT_l2s}" >&2
    exit 1
fi

n=0; skipped=0
for ds in "${DATASETS[@]}"; do
    IFS='|' read -r tag file demos ntest nval <<<"$ds"
    [ -f "$file" ] || { echo "MISSING MANIFEST: $file" >&2; exit 1; }
    dov="task.dataset.split_file=$file n_demos=$demos"
    dov="$dov task.dataset.n_test_episodes=$ntest task.dataset.n_val_episodes=$nval"
    dov="$dov split_suffix=$tag"
    for arm in "${ARMS[@]}"; do
        IFS='|' read -r ts sfx stag <<<"$arm"
        ov="n_candidates=16 slot_obs_noise.mode=list +slot_obs_noise.timesteps=$ts"
        ov="$ov son_suffix=$sfx son_tag=$stag obs_noise_tag=obs_noised $dov"
        err=$(mktemp)
        resolved=$($PY train.py --config-name=train_pusht_diffusion_search $ov --cfg job --resolve 2>"$err") || true
        run=$(sed -n 's/^run_name: //p'   <<<"$resolved")
        sub=$(sed -n 's/^trainer: //p'    <<<"$resolved")
        task=$(sed -n 's/^task_name: //p' <<<"$resolved")
        if [ -z "$run" ] || [ -z "$sub" ] || [ -z "$task" ]; then
            printf '%-72s %s\n' "$sfx $tag" "CONFIG FAILED TO RESOLVE, skipping"
            sed -e 's/^/    | /' "$err" | tail -5 >&2; rm -f "$err"; skipped=$((skipped+1)); continue
        fi
        rm -f "$err"
        dir="$ROOT/$task/$sub/$run"; name="tr_$run"
        if grep -qxF "$name" <<<"$LIVE"; then
            printf '%-72s %s\n' "$run" "already training, skipping"; skipped=$((skipped+1)); continue
        fi
        if [ -f "$dir/checkpoints/step_0100000.ckpt" ]; then
            printf '%-72s %s\n' "$run" "already finished, skipping"; skipped=$((skipped+1)); continue
        fi
        if [ -z "$SUBMIT" ]; then
            printf '%-72s WOULD SUBMIT  t=%s\n' "$run" "$ts"
            n=$((n+1)); continue
        fi
        read -r A P < <(bash scripts/slurm/pick_gpu.sh) || { echo "no free GPU for $run" >&2; skipped=$((skipped+1)); continue; }
        jid=$(sbatch --parsable --account="$A" --partition="$P" --job-name="$name" \
              --time="$TIME_LIMIT" $EXCL_FLAG \
              --export=ALL,CONFIG_NAME=train_pusht_diffusion_search \
              scripts/slurm/train_pusht_search.sbatch $ov)
        printf '%-72s submitted %s on %s/%s\n' "$run" "$jid" "$A" "$P"
        n=$((n+1))
    done
done
echo
echo "arms handled: $n   skipped: $skipped"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
