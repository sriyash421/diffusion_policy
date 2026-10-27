#!/bin/bash
#SBATCH --account=ckpt-robotics
#SBATCH --partition=ckpt
#SBATCH --qos=ckpt-gpu
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40s|l40|a100
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/gscratch/robotics/harine/RoboMonkey_data/experiments/veritas_undertrained_bon_20261006/logs/%x-%j.out
# Usage: sbatch -J vub_<task>_<arm>_n<N> veritas_undertrained_bon.sbatch <task> <arm> <N>
set -eo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone_simpler
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate simpler_env
# Veritas runs use Gemini key 2 only; never passed via --export.
if [ "$2" != "n1" ]; then
  KEYF="$HOME/.config/gemini/api_key2"
  export GEMINI_API_KEY="$(tr -d '[:space:]' < "$KEYF")" GEMINI_KEY_GLOB="$KEYF"
fi
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
exec python -u scripts/simpler/eval_search/veritas_undertrained_bon.py "$@"
