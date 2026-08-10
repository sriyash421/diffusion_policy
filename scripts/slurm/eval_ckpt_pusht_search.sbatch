#!/bin/bash
#SBATCH --job-name=pusht_eval_ckpt
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Best-of-N success-rate eval of ONE checkpoint, on the preemptible checkpoint partition
# so the guaranteed robotics/weirdlab GPUs stay free for training. --requeue re-runs after
# preemption. Results merge into <run>/bon_search/success_curves.jsonl -- the same
# run-level index the --watch watcher maintains, so one-off evals and watched evals share
# one history.
#
# Usage: sbatch scripts/slurm/eval_ckpt_pusht_search.sbatch <CHECKPOINT.ckpt>
set -euo pipefail

CKPT="${1:?usage: sbatch eval_ckpt_pusht_search.sbatch <path/to/step_XXXXXXX.ckpt> [extra eval args...]}"
shift   # everything after the checkpoint is forwarded to eval_search_pusht.py

# `set +u` around conda: activating robodiff first DEACTIVATES whatever env is live, and
# this install's deactivate.d/libxml2_deactivate.sh reads an unset variable. SLURM exports
# the submitting shell's environment, so if you sbatch from a shell that already has
# robodiff active, the job dies in ~10s with "xml_catalog_files_libxml2: unbound variable"
# before running anything. Nounset is restored immediately after.
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

echo "evaluating $CKPT"
# "$@" matters: without it extra flags (e.g. --max-n 1024) are silently dropped and
# the job runs the DEFAULT sweep while looking like it honoured the request.
python eval_search_pusht.py -c "$CKPT" --device cuda:0 "$@"
