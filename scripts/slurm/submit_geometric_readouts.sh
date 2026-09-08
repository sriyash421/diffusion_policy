#!/usr/bin/env bash
# EVAL FOR THE GEOMETRIC-SPLIT SWEEP (scripts/run_geometric_splits.sh): 15 runs x 2 readout
# rules = 30 watcher jobs on the PREEMPTIBLE ckpt partition.
#
# THE QUESTION these numbers answer: does a policy trained on one region of T start poses
# work on a region it never saw? Every run's `test` split is held-out BY GEOMETRY, not by a
# random draw, so success-vs-n here is an out-of-region readout.
#
# WIDTHS. n = 1,2,4,8,16,32,64 for BOTH rules and ALL THREE arms -- the same levels on every
# arm, because the comparison being made is across arms at matched n. Cost is linear in n
# per level and the levels sum to 127 units, so one checkpoint x one rule is ~127 n=1
# sweeps; at checkpoint_every=10000 that is 10 checkpoints per run.
#
# RULES. Clean only. None of these three arms trains under a slot_obs_noise ladder, so
# --corrupt-obs-eval is a NO-OP on them and the corrupt row would be the same experiment
# recorded twice. --no-corrupt-obs-eval is still passed EXPLICITLY: it is what keys the
# output into bon_search_sel-<rule>_obs-clean, which is where the doc builder reads.
#
# WHY NO pick_gpu.sh: eval_watch_pusht_search.sbatch declares `--partition=ckpt
# --account=robotics` itself. Eval belongs on the preemptible checkpoint partition so the
# guaranteed robotics/weirdlab GPUs stay free for the 15 training jobs.
#
# RE-RUNNABLE. The watcher has a 12h wall clock and training runs for days, so this must be
# re-run as checkpoints land. It skips any (run, rule) whose watcher is already queued, and
# eval_search_pusht merges into the same success_curves.jsonl, so a re-submitted watcher
# picks up where the last one stopped. To automate:
#   setsid nohup env SUBMIT=1 bash scripts/slurm/autoupdate_geometric_readouts.sh >/dev/null 2>&1 &
#
#   bash scripts/slurm/submit_geometric_readouts.sh            # dry run
#   SUBMIT=1 bash scripts/slurm/submit_geometric_readouts.sh   # ...and sbatch
set -uo pipefail
cd "$(dirname "$0")/../.."

SUBMIT="${SUBMIT:-}"
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search
TASK="${TASK_DIR:-pusht_image_search_imgonly}"
N_LIST="${N_LIST:-1,2,4,8,16,32,64}"
N_ENVS="${N_ENVS:-50}"
ENC="${ENC_TAG:-enc-resnet18}"

# tag | n_demos -- must match scripts/run_geometric_splits.sh exactly. A tag spelled
# differently here stats a directory that never exists, so every run looks un-trained and
# nothing is ever submitted.
DATASETS=("_split-blq|137" "_split-trq|173" "_split-brd30|30" "_split-brd60|60" "_split-brd100|100")
# trainer subdir | run-name stem
ARMS=("unet_bc|unetbc_ver-t_goal" "offline|value_k1_ver-t_goal" "outer_inner|value_k16_ver-t_goal")

# The OBS-NOISE arms (scripts/run_obsnoise_geometric.sh, run_goalmask_geometric.sh) live on
# a subset of the datasets and carry an extra suffix between ver-t_goal and enc-.
# dataset-tag | n_demos | run-name infix
NOISE_DATASETS=("_split-blq|137" "_split-brd60|60")
NOISE_INFIXES=(
  "_son-flat800" "_son-flat400" "_son-flat200"          # part 1: flat ladder
  "_son-ramp800to400" "_son-ramp400to200"               # part 2: half-ramp
  "_gm-t800-T3px"                                       # part 3: goal-only image mask
)

# label:flags. The label is the job name only; the OUTPUT directory comes from
# eval_search_pusht._bon_subdir and MUST be mirrored exactly below, or a finished sweep
# looks un-done and is resubmitted forever.
CLEAN_RULES=(
  "argmax-clean:--selection argmax --no-corrupt-obs-eval"
  "final_pass-clean:--selection final_pass --no-corrupt-obs-eval"
)
# A run with a slot ladder gets the corrupted rows too: --corrupt-obs-eval reproduces the
# slot -> level mapping the loss trained under, so the rollout conditional matches the
# training one. On a run with NO ladder the flag is a no-op and the two rows would be the
# same experiment recorded twice -- which is why the case below keys on _son-.
#
# The _gm- (goal-mask) arms take the CLEAN rules only for that reason: their slot ladder is
# uniform (off) and their corruption is image-space and TRAIN-ONLY, so there is no
# eval-time condition to reproduce.
LADDER_RULES=(
  "argmax-clean:--selection argmax --no-corrupt-obs-eval"
  "argmax-corrupt:--selection argmax --corrupt-obs-eval"
  "final_pass-clean:--selection final_pass --no-corrupt-obs-eval"
  "final_pass-corrupt:--selection final_pass --corrupt-obs-eval"
)

# Build the full (trainer-subdir, run-name) work list once.
RUNS=()
for ds in "${DATASETS[@]}"; do
    IFS='|' read -r tag demos <<<"$ds"
    for arm in "${ARMS[@]}"; do
        IFS='|' read -r sub stem <<<"$arm"
        RUNS+=("$sub|${stem}_${ENC}_demos-${demos}${tag}_seed-42")
    done
done
for ds in "${NOISE_DATASETS[@]}"; do
    IFS='|' read -r tag demos <<<"$ds"
    for infix in "${NOISE_INFIXES[@]}"; do
        RUNS+=("outer_inner|value_k16_ver-t_goal${infix}_${ENC}_demos-${demos}${tag}_seed-42")
    done
done

# WANDB CREDENTIALS. sbatch exports the submitting shell's environment by default, so a
# STALE WANDB_API_KEY there overrides ~/.netrc inside the job and `--wandb` dies with
# `401 Unauthorized` after the GPU is allocated. That killed the first three TRAINING arms
# of this sweep on 2026-09-03, and this script passes --wandb too.
if [ -n "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "WARNING: WANDB_API_KEY is set and ~/.netrc has no wandb entry; keeping it" >&2
else
    unset WANDB_API_KEY
fi

LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)
n=0; skipped=0

for entry in "${RUNS[@]}"; do
    IFS='|' read -r sub run <<<"$entry"
    dir="$ROOT/$TASK/$sub/$run"
    if [ ! -d "$dir/checkpoints" ] || ! compgen -G "$dir/checkpoints/step_*.ckpt" >/dev/null; then
        printf '%-74s %s\n' "$run" "no checkpoints yet, skipping"; skipped=$((skipped+1)); continue
    fi
    ncp=$(ls "$dir"/checkpoints/step_*.ckpt 2>/dev/null | wc -l)
    # A slot ladder is registered only by the _son- arms; everything else (baselines and the
    # image-space _gm- arms) has no eval-time corruption to reproduce.
    case "$run" in
        *_son-*) rules=("${LADDER_RULES[@]}") ;;
        *)       rules=("${CLEAN_RULES[@]}") ;;
    esac
    for rule in "${rules[@]}"; do
        label="${rule%%:*}"; flags="${rule#*:}"
        sel="${label%%-*}"; obs="${label##*-}"
        out="bon_search_sel-${sel}_obs-${obs}"       # mirrors _bon_subdir
        name="ev_${label}_${run}"
        if grep -qxF "$name" <<<"$LIVE"; then
            printf '%-74s %-20s %s\n' "$run" "$label" "already evaluating, skipping"
            skipped=$((skipped+1)); continue
        fi
        done_n=0
        [ -f "$dir/$out/success_curves.jsonl" ] && done_n=$(wc -l < "$dir/$out/success_curves.jsonl")
        if [ -z "$SUBMIT" ]; then
            printf '%-74s %-20s %2d ckpt, %2d done\n' "$run" "$label" "$ncp" "$done_n"
            n=$((n+1)); continue
        fi
        jid=$(sbatch --parsable --job-name="$name" \
              scripts/slurm/eval_watch_pusht_search.sbatch "$dir" \
              --n-list "$N_LIST" --n-envs "$N_ENVS" --skip-val --wandb $flags)
        printf '%-74s %-20s submitted %s\n' "$run" "$label" "$jid"
        n=$((n+1))
    done
done
echo
echo "readouts handled: $n   skipped: $skipped"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
