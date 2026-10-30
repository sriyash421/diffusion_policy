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
# Arm-B (expert-target) data for carrot, every step skipped when its output exists:
#   1 success-filtered dataset   2 seed gate   3 random_train100 split   4 Gemini plans (train+val)
#   5 replay (frames, proprio)   6 H=4 verifier targets + value stats   7 base-pi0 candidates
# Steps 1-4,6 use this repo's conda env, 5 and 7 the veritas venv.
set -euo pipefail
DP_ROOT="${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/diffusion_policy_standalone_simpler}"; cd "$DP_ROOT"
CONDA=/gscratch/robotics/harine/miniconda3/envs/simpler_env/bin/python
S2=/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2/carrot
DATA=/gscratch/robotics/harine/data/carrot_on_plate_strict
SPLIT=diffusion_policy/config/splits/simpler/carrot/random_train100.json
CAM=diffusion_policy/config/veritas/camera_widowx_carrot_on_plate.json
export GEMINI_API_KEY="$(tr -d '[:space:]' < "$HOME/.config/gemini/api_key")" GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.8-flash}"
mkdir -p "$S2"
echo "node=$(hostname -s) job=$SLURM_JOB_ID"

[ -d "$DATA/carrot_on_plate_strict.zarr" ] || $CONDA scripts/simpler/data/filter_success_episodes.py --task carrot
[ -f scripts/simpler/eval_search/demoseeds/carrot_strict_seed_gate.json ] || $CONDA scripts/simpler/veritas/seed_gate.py --task carrot
[ -f "$SPLIT" ] || $CONDA scripts/simpler/splits/make_simpler_splits.py --task carrot
mkdir -p "$S2/plans_h4"
$CONDA scripts/simpler/veritas/plan_simpler_episodes.py --task widowx_carrot_on_plate --split-file "$SPLIT" \
  --splits train val --outdir "$S2/plans_h4" --camera-file "$CAM" --skip-planned

export DP_ROOT; source scripts/simpler/veritas_pi0/step2_env.sh
[ -f "$S2/replay/replay_summary.json" ] || "$PY" scripts/simpler/veritas_pi0/precompute_proprio_replay.py \
  --split-file "$SPLIT" --out "$S2/replay" --episodes train,val
[ -f "$S2/plans_h4/veritas_targets.npy" ] || $CONDA scripts/simpler/veritas/build_veritas_targets.py \
  --task widowx_carrot_on_plate --split-file "$SPLIT" --plan-dir "$S2/plans_h4" --camera-file "$CAM" \
  --horizon 4 --require-splits train val
"$PY" scripts/simpler/veritas_pi0/precompute_context_candidates.py --replay-dir "$S2/replay" --split-file "$SPLIT" \
  --targets "$S2/plans_h4/veritas_targets.npy" --out "$S2/candidates_ct" --episodes train,val \
  --camera-file "$CAM" --instruction "put carrot on plate" --value cost_over_tol
echo "carrot demo pipeline done"
