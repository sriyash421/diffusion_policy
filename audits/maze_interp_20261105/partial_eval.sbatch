#!/bin/bash
# Partial eval while a run trains: STk0..(saved rounds) on the 50 test pairs, 8 seeds × 8 attempts (bon_eval --partial --splits test);
# runs trained with observation noise also get the noisy-observation eval. Writes runs/<tag>/bon_seeds8[_noisyobs]_partial.npz and
# prints STk on attempt k+1 and STk(last) attempt 1..8. Submitted by monitor.sh.
#   sbatch --job-name=pe_<tag> partial_eval.sbatch <tag>
#SBATCH --account=robotics
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
[[ ${1:-} == c_* ]] && export MAZE_SETTING=interp30c   # c_<tag>: continuous maze (cmaze.py)
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
tag=$1
echo "== start $(date) on $(hostname) tag=$tag"
$PY -u bon_eval.py --tag "$tag" --seeds 8 --attempts 8 --splits test --partial 2>&1 | grep -v Warning | tail -3
noisy=0; grep -q '"obs_noise": "\(flat\|ramp\)"' $I/runs/$tag/config.json && noisy=1
[[ $noisy == 1 ]] && $PY -u bon_eval.py --tag "$tag" --seeds 8 --attempts 8 --splits test --partial --eval-obs-noise 2>&1 | grep -v Warning | tail -3
$PY - "$I/runs/$tag" <<'PYEOF'
import sys, numpy as np
from pathlib import Path
d = Path(sys.argv[1])
for f in ('bon_seeds8_partial.npz', 'bon_seeds8_noisyobs_partial.npz'):
    if not (d / f).exists():
        continue
    z = np.load(d / f); S = z['test_st_success']; R, A = S.shape[:2]
    diag = [S[m, min(m, A - 1)].mean() for m in range(R)]
    print(f'SUMMARY {d.name} {f}: STk0..{R - 1} on attempt k+1 {np.round(diag, 2).tolist()} | STk{R - 1} attempt 1..8 '
          f'{np.round(S[R - 1].mean((-1, -2)), 2).tolist()} | BC {z["test_bc_success"][..., 0].mean() if "test_bc_success" in z.files else float("nan"):.2f}', flush=True)
PYEOF
echo "== done $(date)"
