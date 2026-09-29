#!/bin/bash
#SBATCH --job-name=pusht_eval_criteria
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# SELECTION-CRITERIA eval of ONE checkpoint: n held fixed (default 16) while the six
# read-out rules are swept on the same weights. On the preemptible checkpoint partition so
# the guaranteed robotics/weirdlab GPUs stay free for training; --requeue re-runs after
# preemption, and eval_search_pusht persists after EVERY criterion, so a requeued job
# re-does at most one of the six rather than all of them.
#
# All six criteria run in ONE job on purpose: they share a policy load and an env pool, and
# they are six rollouts of ~5-6 min each rather than six jobs of one.
#
# Results go to <run>/criteria_search/, NOT bon_search/ -- these rows are keyed by criterion
# at fixed n, so merging them with the n-sweep curves would put two different experiments in
# one file.
#
# Usage: sbatch scripts/slurm/eval_criteria_pusht_search.sbatch <CHECKPOINT.ckpt> [extra args...]
set -euo pipefail

CKPT="${1:?usage: sbatch eval_criteria_pusht_search.sbatch <path/to/step_XXXXXXX.ckpt> [extra eval args...]}"
shift   # everything after the checkpoint is forwarded to eval_search_pusht.py

# `set +u` around conda: activating robodiff first DEACTIVATES whatever env is live, and
# this install's deactivate.d/libxml2_deactivate.sh reads an unset variable. SLURM exports
# the submitting shell's environment, so if you sbatch from a shell that already has
# robodiff active, the job dies in ~10s with "xml_catalog_files_libxml2: unbound variable".
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

# Traces (~125 KB per criterion) land next to the run on /gscratch along with everything
# else; nothing here writes to home.
echo "criteria eval: $CKPT"
# --skip-val: the 50 TEST episodes only. Note the consequence -- with no val curve, a
# checkpoint later picked off these numbers is picked on test, so quote whole curves rather
# than a chosen step, or re-run without this flag if you need a held-out estimate.
# "$@" last so extra flags actually reach the script instead of being silently dropped.
python eval_search_pusht.py \
  -c "$CKPT" \
  --device cuda:0 \
  --criteria-sweep \
  --criteria-n 16 \
  --n-envs 50 \
  --skip-val \
  "$@"
