#!/bin/bash
#SBATCH --job-name=pusht_eval_bon
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# eval_bon.py on ONE checkpoint, on the preemptible ckpt partition so the guaranteed
# robotics/weirdlab GPUs stay free for training. --requeue re-runs after preemption.
#
# THE SIBLING OF eval_ckpt_pusht_search.sbatch, not a replacement for it. That one drives
# eval_search_pusht.py, whose best-of-n is a VERIFIER-RANKED search over candidates within a
# control step. This one drives eval_bon.py, whose best-of-n is an oracle max over whole
# EPISODES and needs no verifier -- which is why it is the one the LSTM BC arms can use at
# all (`verifier_tag: null`, no policy.verifier_value).
#
# N=1 IS THE DEFAULT HERE, and that is a statement about the policy, not a shortcut.
# eval_bon draws its spread entirely from the policy; `LSTMBCPolicy.predict_action` returns
# the distribution MEAN and the env is deterministic, so every extra sample is a byte-identical
# rollout. At n=1 the reported `mean_success_single_sample` IS the test success rate and no
# curve is written. Pass --n-samples N explicitly for a policy that actually samples.
#
# The test resets come from the CHECKPOINT'S OWN cfg.task.dataset, so each arm is scored on
# the split it trained against; there is nothing to pass here.
#
# Usage:
#   sbatch scripts/slurm/eval_bon_ckpt.sbatch <path/to/step_XXXXXXX.ckpt> [extra eval_bon args]
set -euo pipefail

CKPT="${1:?usage: sbatch eval_bon_ckpt.sbatch <path/to/step_XXXXXXX.ckpt> [extra args...]}"
shift

# see eval_ckpt_pusht_search.sbatch: conda's deactivate hook reads an unset var, which kills
# the job under `set -u` if sbatch was run from an active env
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
DP_CONDA_ENV="${DP_CONDA_ENV:-vae_pushT_l2s}"
conda activate "$DP_CONDA_ENV"
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

GSCRATCH_ROOT=/gscratch/robotics/harine
export DP_OUTPUT_ROOT="${DP_OUTPUT_ROOT:-$GSCRATCH_ROOT/diffusion_policy_outputs}"
case "$DP_OUTPUT_ROOT" in
  /gscratch/*) ;;
  *) echo "FATAL: DP_OUTPUT_ROOT=$DP_OUTPUT_ROOT is not on /gscratch." >&2; exit 1 ;;
esac

# Results land beside the run, keyed by step, so re-evaluating one step cannot overwrite
# another's and the set of evaluated steps is readable from the directory listing.
STEP="$(basename "$CKPT" .ckpt)"
RUN_DIR="$(dirname "$(dirname "$CKPT")")"
OUT="$RUN_DIR/bon_episodes/$STEP"
mkdir -p "$OUT"
echo "checkpoint : $CKPT"
echo "output     : $OUT"

python eval_bon.py -c "$CKPT" -o "$OUT" -d cuda:0 --n-samples 1 --n-envs 50 "$@"
