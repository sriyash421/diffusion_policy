#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o /gscratch/robotics/harine/codex_audits/maze_rounds_20261103/logs/%x-%j.out
set -e; cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg PYTHONPATH=/gscratch/robotics/harine/L2S
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
$PY bon_eval.py --tag $TAG --seeds 1 --attempts 2 --splits test ${EXTRA:-}
$PY - <<PYEOF
import numpy as np
z = np.load('runs/$TAG/bon_seeds1${SUF:-}.npz')
for k in ('st', 'bc'):
    s, b = z[f'test_{k}_success'].ravel(), z[f'test_{k}_bfs'].ravel()
    print(k, 'success', s.mean().round(3), 'bfs of successes', np.bincount(b[s])[:4], 'min bfs of failures', b[~s].min() if (~s).any() else None)
    assert (b[s] <= 1).all(), 'successful candidates must end at (or next to) the goal'
print('BON_SMOKE_OK')
PYEOF
$PY bon_eval.py --tag $TAG --seeds 8 --attempts 8 ${EXTRA:-}
