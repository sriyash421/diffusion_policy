#!/usr/bin/env bash
# Phase 5(c): does substituting a LEARNED value for the t_goal heuristic make best-of-N solve
# more episodes? Six checkpoints x {t_goal, v}, n = 1..16, paired episode by episode.
#
#   bash scripts/slurm/bon_sweep_arms.sh              # dry run
#   SUBMIT=1 bash scripts/slurm/bon_sweep_arms.sh     # ...and sbatch it
#   SUBMIT=1 RANKERS=t_goal,v,q Q=<sac.zip> bash scripts/slurm/bon_sweep_arms.sh
#
# Phase 5(a) asked where the expert action a* RANKS under each verifier. This asks the
# deployable question instead: run the policy, let each verifier pick, count solved episodes.
# A verifier can rank a* well and still not help -- ranking a* is necessary, not sufficient.
set -uo pipefail

ROOT=/mmfs1/home/harine/diffusion_policy_standalone
OUT="${OUT:-/gscratch/robotics/harine/value_arms/bon}"
V="${V:-/gscratch/robotics/harine/value_arms/ppo_plain_keypoint/model.zip}"
Q="${Q:-}"
RANKERS="${RANKERS:-t_goal,v}"
MAX_N="${MAX_N:-16}"
SPLIT="${SPLIT:-test}"
CPUS="${CPUS:-8}"

# ckpt is preemptible and free, so the guaranteed robotics/weirdlab GPUs stay on training.
# --requeue re-runs after preemption; each sweep re-seeds per n, so a requeue repeats work
# rather than corrupting a partial curve.
TIME="${TIME:-12:00:00}"

# The six ranked checkpoints, step 30k: three policy classes x two geometric split families.
# The SAME six Phase 5(a) scored, so the two evaluations are about the same models.
B=ckpts/pusht_search/pusht_image_search_imgonly
CKPTS="
unetbc_blq137     | $B/unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-137_split-blq_seed-42
unetbc_brd60      | $B/unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-60_split-brd60_seed-42
value_k16_blq137  | $B/outer_inner/value_k16_ver-t_goal_enc-resnet18_demos-137_split-blq_seed-42
value_k16_brd60   | $B/outer_inner/value_k16_ver-t_goal_enc-resnet18_demos-60_split-brd60_seed-42
value_k1_blq137   | $B/offline/value_k1_ver-t_goal_enc-resnet18_demos-137_split-blq_seed-42
value_k1_brd60    | $B/offline/value_k1_ver-t_goal_enc-resnet18_demos-60_split-brd60_seed-42
"

[ -f "$ROOT/$V" ] || [ -f "$V" ] || { echo "no V checkpoint at $V" >&2; exit 1; }

while IFS='|' read -r label run; do
    label="$(echo "$label" | xargs)"; run="$(echo "$run" | xargs)"
    [ -z "$label" ] && continue
    ck="$run/checkpoints/step_0030000.ckpt"
    [ -f "$ROOT/$ck" ] || { echo "[SKIP $label] no $ck"; continue; }

    # --skip-context-sim is Q-ONLY and is deliberately absent here: V takes no action, so the
    # sim rollout IS its input. Skipping it would leave V nothing to evaluate.
    args="-c $ck --rankers $RANKERS --v $V --max-n $MAX_N --split $SPLIT --n-envs $CPUS"
    [ -n "$Q" ] && args="$args --q $Q"
    cmd="python sac/eval.py bon-sweep $args -o $OUT/$label"

    if [ "${SUBMIT:-0}" = "1" ]; then
        echo "[SUBMIT $label]"
        sbatch --partition=ckpt --account=robotics --requeue --job-name="vs_$label" \
               --gpus=1 --cpus-per-task="$CPUS" --mem=32G --time="$TIME" \
               --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out \
               --wrap "set -u; source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh; \
                       conda activate robodiff; cd $ROOT; unset WANDB_API_KEY; $cmd"
    else
        echo "[$label] $cmd"
    fi
done <<< "$CKPTS"

[ "${SUBMIT:-0}" = "1" ] || echo $'\nDry run. SUBMIT=1 to sbatch.'
