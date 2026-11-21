#!/bin/bash
#SBATCH --job-name=pi0_carrot_demos
#SBATCH --account=ckpt-robotics
#SBATCH --partition=ckpt
#SBATCH --qos=ckpt-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40s|l40|a100
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/pi0_veritas/%x-%j.out
#SBATCH --error=/gscratch/robotics/harine/slurm_logs/pi0_veritas/%x-%j.out
# Arm-B (expert-target) data for carrot (TASK=spoon for spoon), every step skipped when its output exists:
#   1 success-filtered dataset   2 seed gate   3 random_train100 split   4 Gemini plans (train+val)
#   5 replay (frames, proprio)   6 H=4 verifier targets + value stats   7 base-pi0 candidates
# Steps 1-4,6 use this repo's conda env, 5 and 7 the veritas venv.
# DEMO_SRC=pi0_demos|pi0beta_demos (any TASK): recorded pi0 demos in $S2/$DEMO_SRC instead of the
# OpenVLA dataset; steps 1-2 are skipped, the split is the strict-passing demos
# (make_simpler_splits.py --demo-dir, after strict_stats.py) and outputs go under $S2/$DEMO_SRC.
set -euo pipefail
DP_ROOT="${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/diffusion_policy_standalone_simpler}"; cd "$DP_ROOT"
CONDA=/gscratch/robotics/harine/miniconda3/envs/simpler_env/bin/python
TASK="${TASK:-carrot}"
case "$TASK" in
  eggplant) ENV_TASK=widowx_put_eggplant_in_basket; DATA_NAME=strict_distance_success_split; INSTR="put eggplant into yellow basket" ;;
  stack)  ENV_TASK=widowx_stack_cube; DATA_NAME=stack_cube_strict_split; INSTR="stack the green block on the yellow block" ;;
  carrot) ENV_TASK=widowx_carrot_on_plate; DATA_NAME=carrot_on_plate_strict; INSTR="put carrot on plate" ;;
  spoon)  ENV_TASK=widowx_spoon_on_towel; DATA_NAME=spoon_on_towel_strict; INSTR="put the spoon on the towel" ;;
  *) echo "unknown TASK $TASK" >&2; exit 1 ;;
esac
S2=/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2/$TASK
DATA=/gscratch/robotics/harine/data/$DATA_NAME
SPLIT=diffusion_policy/config/splits/simpler/$TASK/random_train100.json
if [ -n "${DEMO_SRC:-}" ]; then
  S2="$S2/$DEMO_SRC"; DATA="$S2"; SPLIT="$S2/split_strict.json"
  [ -f "$SPLIT" ] || { echo "$SPLIT missing: run strict_stats.py, then make_simpler_splits.py --demo-dir $S2" >&2; exit 1; }
fi
CAM=diffusion_policy/config/veritas/camera_$ENV_TASK.json
export GEMINI_API_KEY="$(tr -d '[:space:]' < "$HOME/.config/gemini/api_key")" GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.8-flash}"
mkdir -p "$S2"
echo "node=$(hostname -s) job=$SLURM_JOB_ID"

if [ -z "${DEMO_SRC:-}" ]; then
  ls -d "$DATA"/*.zarr >/dev/null 2>&1 || $CONDA scripts/simpler/data/filter_success_episodes.py --task "$TASK"
  [ -f "scripts/simpler/eval_search/demoseeds/${TASK}_strict_seed_gate.json" ] || $CONDA scripts/simpler/veritas/seed_gate.py --task "$TASK"
  [ -f "$SPLIT" ] || $CONDA scripts/simpler/splits/make_simpler_splits.py --task "$TASK"
fi
mkdir -p "$S2/plans_h4"
$CONDA scripts/simpler/veritas/plan_simpler_episodes.py --task "$ENV_TASK" --split-file "$SPLIT" \
  --splits train val --outdir "$S2/plans_h4" --camera-file "$CAM" --skip-planned

export DP_ROOT; source scripts/simpler/veritas_pi0/step2_env.sh
[ -f "$S2/replay/replay_summary.json" ] || "$PY" scripts/simpler/veritas_pi0/precompute_proprio_replay.py \
  --split-file "$SPLIT" --out "$S2/replay" --episodes train,val
[ -f "$S2/plans_h4/veritas_targets.npy" ] || $CONDA scripts/simpler/veritas/build_veritas_targets.py \
  --task "$ENV_TASK" --split-file "$SPLIT" --plan-dir "$S2/plans_h4" --camera-file "$CAM" \
  --horizon 4 --require-splits train val
"$PY" scripts/simpler/veritas_pi0/precompute_context_candidates.py --replay-dir "$S2/replay" --split-file "$SPLIT" \
  --targets "$S2/plans_h4/veritas_targets.npy" --out "$S2/candidates_ct" --episodes train,val \
  --camera-file "$CAM" --instruction "$INSTR" --value cost_over_tol
echo "$TASK demo pipeline done"
