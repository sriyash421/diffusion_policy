#!/bin/bash
# Smoke for --fb none and INTERP_TEST=all: feedback is all zero; a tiny asteroid smoke trains with --fb none; the all-pairs set loads
# 2870 pairs and one round of int_k0long runs on it (1 seed, 2 attempts, timed). Output removed afterwards.
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
$PY - <<'PYEOF'
import torch, rounds
from asteroid import set_fb
set_fb('none'); b = torch.rand(3, 5); d = torch.rand(3, 5, 2)
assert rounds.fb_step(b, d).shape == (3, 5, 4) and rounds.fb_step(b, d).abs().sum() == 0
assert rounds.fb_end(b[:, :2], d[:, :2]).shape == (3, 2, 4) and rounds.fb_end(b[:, :2], d[:, :2]).abs().sum() == 0
set_fb('bfs_dir'); assert rounds.fb_step(b, d).abs().sum() > 0
print('FB_NONE_OK')
PYEOF
$PY -u asteroid.py smoke --fb none 2>&1 | grep -v Warning | tail -4
grep -q '"fb": "none"' $I/runs/asteroid_smoke/config.json && echo SMOKE_CONFIG_FB_NONE_OK
export INTERP_TEST=all
$PY -c "import rounds; e = rounds.test_episodes(); print('all pairs', len(e)); assert len(e) == 2870"
t0=$(date +%s)
$PY -u bon_eval.py --tag int_k0long --seeds 1 --attempts 2 --splits test --no-bc 2>&1 | grep -v Warning | tail -3
echo "ALLPAIRS_SECONDS $(( $(date +%s) - t0 )) for 1 round x 2 attempts x 2870 rows"
$PY -c "
import numpy as np; z = np.load('$I/runs/int_k0long/bon_seeds1_allpairs.npz'); print({k: z[k].shape for k in z.files})
assert z['test_st_success'].shape == (1, 2, 2870, 1)"
rm -f $I/runs/int_k0long/bon_seeds1_allpairs.npz
echo SMOKE_ALLPAIRS_OK
