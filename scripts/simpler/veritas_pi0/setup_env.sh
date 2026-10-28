#!/bin/bash
# Build the official veritas checkout's uv environment. Run INSIDE a GPU salloc, never on a
# login node:
#   salloc --account=robotics --partition=gpu-a40 --gpus=1 -c 6 --mem=32G --time=2:00:00
#   bash scripts/simpler/veritas_pi0/setup_env.sh
#
# pyproject.toml pins torch>=2.9 on the cu130 index. If the node's driver cannot run CUDA 13
# wheels (torch.cuda.is_available() False after sync), torch/torchvision/torchaudio are
# reinstalled from the cu128 index into the same venv; pyproject/uv.lock stay untouched.
set -euo pipefail
VERITAS_ROOT="${VERITAS_ROOT:-/gscratch/robotics/harine/veritas}"
DP_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"   # this repo, resolved before any cd
export UV_CACHE_DIR="${UV_CACHE_DIR:-/gscratch/robotics/harine/.uv_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/gscratch/robotics/harine/.uv_python}"
export SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}"   # tracking does not need the SAM2 CUDA extension
export TMPDIR="${TMPDIR:-/tmp}"
[ -n "${SLURM_JOB_ID:-}" ] || { echo "run this inside a Slurm job" >&2; exit 1; }
module load cuda/12.8.1 || true
cd "$VERITAS_ROOT"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "uv $(uv --version)  python target $(grep requires-python pyproject.toml)"

# ruckig 0.17.3 (ManiSkill2 dep, sdist only) does not build with scikit-build-core>=0.10.
# `uv sync` takes build constraints only from pyproject [tool.uv], so this is the ONE local
# edit to the official checkout (idempotent; uv re-locks the manifest line, versions unchanged).
if ! grep -q "build-constraint-dependencies" pyproject.toml; then
  sed -i 's/^\[tool.uv\]$/[tool.uv]\nbuild-constraint-dependencies = ["scikit-build-core<0.10"]  # klone: ruckig sdist build/' pyproject.toml
  grep -n -A1 "^\[tool.uv\]" pyproject.toml
fi
uv sync --no-progress
PY="$VERITAS_ROOT/.venv/bin/python"
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "cu130 torch cannot see the GPU on this driver; reinstalling torch from cu128"
  TV="$("$PY" -c 'import torch; print(torch.__version__.split("+")[0])')"
  uv pip install --no-progress --index-url https://download.pytorch.org/whl/cu128 \
    "torch==$TV" torchvision torchaudio
fi
"$PY" - <<'EOF'
import torch, simpler_env, sam2, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__, "simpler_env ok, sam2 ok")
EOF
echo "env ready: $PY"
