#!/bin/bash
# Smoke: asteroid --save-every (mid-round ckpts), select_ckpt.py, bon_eval --ckpt on a tiny 2-round run (runs/int_cksmoke, removed after).
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=smoke_ck
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python; I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105; D=$I/runs/int_cksmoke
rm -rf $D
$PY -u asteroid.py train --tag int_cksmoke --fb bfs_dir --rounds 2 --batch 32 --micro 32 --lr 1e-4 --steps-per-iter 10 --iters 4 \
    --test-every 10 --test-seeds 1 --train-seeds 1 --save-every 10 2>&1 | grep -v Warning | grep -E "ROUND|Traceback|Error"
ls $D/ckpt; [[ $(ls $D/ckpt | wc -l) == 6 ]] && echo CKPT_COUNT_OK
$PY -u $I/select_ckpt.py --tag int_cksmoke --rounds all --seeds 1 2>&1 | grep -v Warning
$PY - <<PYEOF
import json, torch
s = json.load(open('$D/selection_r1.json')); print(s['pick'])
assert len(s['by_update']) == 4 and [r['update'] for r in s['by_update']] == [10, 20, 30, 40]
a = torch.load('$D/ckpt/round_1_u00030.pt', weights_only=False)['model']; b = torch.load('$D/round_1.pt', weights_only=False)['model']
assert any((a[k] != b[k]).any() for k in a), 'mid-round ckpt equals the end of the round'
print('SELECT_OK')
PYEOF
f=$($PY -c "import json; print(json.load(open('$D/selection_r1.json'))['pick']['test']['file'])")
$PY -u bon_eval.py --tag int_cksmoke --only-round 1 --ckpt $f --suffix _sm --seeds 1 --attempts 3 --no-bc --splits train,test 2>&1 | grep -v Warning | tail -3
$PY -c "
import numpy as np; z = np.load('$D/bon_seeds1_sm.npz'); print(z['ckpt'], z['test_st_success'].shape, z['train_st_success'].shape)
assert z['test_st_success'].shape[:2] == (1, 3) and str(z['ckpt']) == '$f'; print('BON_CKPT_OK')"
rm -rf $D; echo SMOKE_CK_ALL_OK
