#!/bin/bash
# Environment for the step-2 (pi0 + context) scripts: the official veritas checkout's venv with
# this repo importable. Source from an sbatch or an salloc shell; never on a login node for
# anything heavier than --help. Mirrors the exports of eval_pi0.sbatch.
export VERITAS_ROOT="${VERITAS_ROOT:-/gscratch/robotics/harine/veritas}"
export DP_ROOT="${DP_ROOT:-/mmfs1/home/harine/diffusion_policy_standalone_simpler}"
export STEP2_ROOT="${STEP2_ROOT:-/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2}"
export PY="$VERITAS_ROOT/.venv/bin/python"
export VLA_LOG_DIR="${VLA_LOG_DIR:-$STEP2_ROOT/logs}"   # config/eval/bridge.yaml interpolates it
mkdir -p "$VLA_LOG_DIR"
module load cuda/12.8.1 2>/dev/null || true
export CUDA_DEVICE_ORDER=PCI_BUS_ID NVIDIA_TF32_OVERRIDE=1
export PYTHONPATH="$DP_ROOT:$VERITAS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_CACHE="$VERITAS_ROOT/transformers_cache" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false
CACHE=/gscratch/robotics/harine/.cache   # every cache on gscratch, never home
mkdir -p "$CACHE/torchinductor" "$CACHE/triton" "$CACHE/wandb"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor" TRITON_CACHE_DIR="$CACHE/triton"
export WANDB_CACHE_DIR="$CACHE/wandb" WANDB_DATA_DIR="$CACHE/wandb"
export PYTHONWARNINGS="ignore:pkg_resources is deprecated:UserWarning,ignore::FutureWarning"
