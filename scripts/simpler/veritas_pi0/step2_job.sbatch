#!/bin/bash
#SBATCH --job-name=pi0_step2
#SBATCH --account=ckpt-robotics
#SBATCH --partition=ckpt,ckpt-all
#SBATCH --qos=ckpt-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40s|l40|a100|h200
#SBATCH --requeue
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/pi0_veritas/%x-%j.out
#SBATCH --error=/gscratch/robotics/harine/slurm_logs/pi0_veritas/%x-%j.out
# Generic step-2 job: runs "$PY $CMD" inside the veritas venv with this repo on PYTHONPATH.
#   export CMD="scripts/simpler/veritas_pi0/precompute_proprio_replay.py --split-file ... --out ..."
#   sbatch --job-name=pi0_replay --export=ALL scripts/simpler/veritas_pi0/step2_job.sbatch
# (CMD goes through the environment: `--export=ALL,CMD=...` splits CMD at its first comma.)
# Every step-2 script resumes from its output directory, so --requeue on ckpt is safe.
set -euo pipefail
mkdir -p /gscratch/robotics/harine/slurm_logs/pi0_veritas
DP_ROOT="${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/diffusion_policy_standalone_simpler}"
export DP_ROOT
source "$DP_ROOT/scripts/simpler/veritas_pi0/step2_env.sh"
cd "$DP_ROOT"
echo "node=$(hostname -s) job=$SLURM_JOB_ID cmd=$CMD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# eval keeps quoted arguments (e.g. --instruction '...') intact
eval "\"$PY\" $CMD" 2> >(grep -v -E "svulkan2.*(GLFW error: X11|Continue without GLFW)" >&2)
