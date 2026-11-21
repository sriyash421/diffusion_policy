#!/bin/bash
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
export OMP_NUM_THREADS=4
ROOT=/mmfs1/gscratch/robotics/harine/codex_audits/maze_quadrants_delta_20261019
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
read -r setting algo seed variant < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$ROOT/${TASK_LIST:-train.tsv}")
extra=(); if [[ "${TASK_LIST:-train.tsv}" == smoke.tsv ]]; then extra+=(--smoke); fi
"$PY" -u "$ROOT/run_experiment.py" --setting "$setting" --algo "$algo" --seed "$seed" --variant "$variant" "${extra[@]}"
if [[ "${TASK_LIST:-train.tsv}" == smoke.tsv ]]; then
 checkpoint="$ROOT/smoke/$setting/${algo}__${variant}/seed_${seed}/step_0000002.pt"
 if [[ "$variant" == delta ]]; then "$PY" -u "$ROOT/test_delta.py" "$checkpoint"; fi
 "$PY" -u "$ROOT/evaluate.py" --checkpoint "$checkpoint" --smoke
fi
