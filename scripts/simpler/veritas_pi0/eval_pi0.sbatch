#!/bin/bash
#SBATCH --job-name=pi0_veritas
#SBATCH --account=ckpt-robotics
#SBATCH --partition=ckpt
#SBATCH --qos=ckpt-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40s|l40|a100
#SBATCH --requeue
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/pi0_veritas/%x-%j.out
#SBATCH --error=/gscratch/robotics/harine/slurm_logs/pi0_veritas/%x-%j.out
# One eval arm of open-pi-zero in SimplerEnv through the official Veritas EvalAgent
# (scripts/simpler/veritas_pi0/run_pi0_eval.py). Submit from the repo root.
#
# Required env: OUT_DIR (log_dir; episodes.jsonl there makes a restart resume).
# Optional: TASK (widowx_put_eggplant_in_basket), VERIFIER (0|1, default 0), N (1; candidates
#   per decision), COLLECT_DEMOS (0|1, default = VERIFIER), N_EPISODES (240), ARM (raw_n1 /
#   veritas_n5), EVAL_SEEDS (json list -> seeded protocol), CKPT (bridge_uniform), USE_BF16 (False),
#   GEMINI_MODEL (gemini-3.8-flash), GEMINI_KEY_FILE, VERITAS_ROOT, WANDB_MODE (online),
#   EXTRA (further hydra overrides, space separated).
set -euo pipefail
VERITAS_ROOT="${VERITAS_ROOT:-/gscratch/robotics/harine/veritas}"
DP_ROOT="${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/diffusion_policy_standalone_simpler}"
mkdir -p /gscratch/robotics/harine/slurm_logs/pi0_veritas

# Gemini key: from the environment, else a key file (never via --export, which would expose it
# in the job's recorded environment).
if [ -n "${GEMINI_KEY_FILE:-}" ]; then
  export GEMINI_API_KEY="$(tr -d '[:space:]' < "$GEMINI_KEY_FILE")"
elif [ -z "${GEMINI_API_KEY:-}" ] && [ -f "$HOME/.config/gemini/api_key" ]; then
  export GEMINI_API_KEY="$(tr -d '[:space:]' < "$HOME/.config/gemini/api_key")"
fi
# gemini-2.5-flash (official default) is retired on this key (404, 2026-10-27); 3.8-flash is the
# current flash model and what the repo's own plan caches used.
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.8-flash}"

TASK="${TASK:-widowx_put_eggplant_in_basket}"
VERIFIER="${VERIFIER:-0}"
N="${N:-1}"
COLLECT_DEMOS="${COLLECT_DEMOS:-$VERIFIER}"
N_EPISODES="${N_EPISODES:-240}"
CKPT="${CKPT:-models/bridge_uniform_step19296_2024-12-26_22-31_42.pt}"
USE_BF16="${USE_BF16:-False}"
if [ "$VERIFIER" = 1 ]; then VERIFIER_BOOL=true; ARM="${ARM:-veritas_n$N}"; else VERIFIER_BOOL=false; ARM="${ARM:-raw_n$N}"; fi
if [ "$COLLECT_DEMOS" = 1 ]; then COLLECT_BOOL=true; else COLLECT_BOOL=false; fi
OUT_DIR="$(realpath -m "${OUT_DIR:?}")"
mkdir -p "$OUT_DIR"

cd "$VERITAS_ROOT"
# torch.compile's inductor shells out to nvcc (and its crash reporter does too); the CUDA module
# puts an executable nvcc on PATH. USE_COMPILE=False falls back to eager (official default True).
module load cuda/12.8.1 || true
export TORCHDYNAMO_VERBOSE="${TORCHDYNAMO_VERBOSE:-1}"
USE_COMPILE="${USE_COMPILE:-True}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NVIDIA_TF32_OVERRIDE=1
export GROUNDED_SAM2_PATH="$VERITAS_ROOT/Grounded-SAM-2"
export PYTHONPATH="$GROUNDED_SAM2_PATH:$DP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_CACHE="$VERITAS_ROOT/transformers_cache"   # paligemma dir + linked hub caches
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export VLA_LOG_DIR="$OUT_DIR"
export VERITAS_ROOT WANDB_MODE="${WANDB_MODE:-online}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# every cache on gscratch, never home (10 GB quota): torch.compile / triton kernels, wandb
CACHE=/gscratch/robotics/harine/.cache
mkdir -p "$CACHE/torchinductor" "$CACHE/triton" "$CACHE/wandb"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor" TRITON_CACHE_DIR="$CACHE/triton"
export WANDB_CACHE_DIR="$CACHE/wandb" WANDB_DATA_DIR="$CACHE/wandb" WANDB_DIR="$OUT_DIR"
export PYTHONWARNINGS="ignore:pkg_resources is deprecated:UserWarning,ignore::FutureWarning:google.api_core._python_version_support,ignore:TensorFloat32 tensor cores:UserWarning"

echo "arm=$ARM task=$TASK verifier=$VERIFIER_BOOL N=$N collect=$COLLECT_BOOL episodes=$N_EPISODES out=$OUT_DIR"
echo "veritas @ $(git -C "$VERITAS_ROOT" rev-parse --short HEAD)  gemini=$GEMINI_MODEL  node=$(hostname -s)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# tiny compile check so a toolchain problem shows up before the model loads
"$VERITAS_ROOT/.venv/bin/python" -c 'import shutil, torch
print("nvcc:", shutil.which("nvcc"), "| g++:", shutil.which("g++"))
f = torch.compile(lambda x: (x * 2).sin().sum())
print("compile check:", float(f(torch.ones(8, device="cuda"))))'

SEED_ARG=""
if [ -n "${EVAL_SEEDS:-}" ]; then SEED_ARG="+eval_seeds=$(realpath "$EVAL_SEEDS")"; fi

"$VERITAS_ROOT/.venv/bin/python" "$DP_ROOT/scripts/simpler/veritas_pi0/run_pi0_eval.py" \
  env.task="$TASK" \
  checkpoint_path="$CKPT" \
  log_dir="$OUT_DIR" \
  n_eval_episode="$N_EPISODES" \
  use_bf16="$USE_BF16" \
  use_torch_compile="$USE_COMPILE" \
  +use_veritas_verifier="$VERIFIER_BOOL" \
  +verifier_samples="$N" \
  +collect_demos="$COLLECT_BOOL" \
  +visualize_positions=true +visualize_action_trace=true +visualize_waypoint_curve=true +visualize_ee=false \
  +arm="$ARM" \
  $SEED_ARG ${EXTRA:-} \
  2> >(grep -v -E "svulkan2.*(GLFW error: X11|Continue without GLFW)" >&2)
