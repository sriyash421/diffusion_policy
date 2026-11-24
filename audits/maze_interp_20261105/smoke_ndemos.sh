#!/bin/bash
# Smoke (2026-11-24): asteroid.py --n-demos (DAgger on a subset of the interp30c train pairs) and pg_dagger.py --max-ctx / --out.
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
A=/gscratch/robotics/harine/codex_audits; PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
export PYTHONPATH=/gscratch/robotics/harine/L2S:$A/maze_rounds_20261103:$A/maze_procgen_ast_20261121
cd $A/maze_rounds_20261103; export MAZE_SETTING=interp30c
rm -rf $A/maze_interp_20261105/runs/smoke_nd $A/maze_procgen_ast_20261121/runs/smoke_pgc2
$PY -u asteroid.py train --tag smoke_nd --target relabel --fb reward --rounds 4 --batch 16 --micro 16 --steps-per-iter 2 --iters 1 \
  --test-every 0 --train-seeds 0 --obs ddpm --obs-t 400 --max-ctx 2 --n-demos 40 2>&1 | grep -v Warning | grep -E 'ROUND|Error|Trace' | cut -c1-200
$PY - <<'P'
import json, torch
r = '/gscratch/robotics/harine/codex_audits/maze_interp_20261105/runs/smoke_nd/'
c = json.load(open(r + 'config.json')); st = torch.load(r + 'state.pt', map_location='cpu', weights_only=False)
print('config n_demos', c['n_demos'], 'train_starts', c['train_starts'], 'items', len(st['items']), 'ctx lens', sorted({len(i[1]) for i in st['items']}),
      'distinct starts', len({i[0] for i in st['items']}))
assert c['train_starts'] == 40 and len(st['items']) == 160 and sorted({len(i[1]) for i in st['items']}) == [0, 1, 2]
P
$PY -u asteroid.py evalctx --tag smoke_nd --seeds 1 --attempts 4 --eval-rounds -1 --splits train --out _n4 2>&1 | grep -v Warning | tail -2
$PY -c "import numpy as np; z=np.load('$A/maze_interp_20261105/runs/smoke_nd/eval_ctx_n4.npz'); print('window', z['window'], 'train', z['train_success'].shape); assert z['train_success'].shape[2]==40 and int(z['window'])==2"
unset MAZE_SETTING; cd $A/maze_procgen_ast_20261121
$PY -u pg_dagger.py train --tag smoke_pgc2 --fb reward --obs ddpm --obs-t 400 --rounds 4 --iters 1 --steps-per-iter 2 --batch 16 --micro 16 --max-levels 20 --max-ctx 2 2>&1 | grep -v Warning | grep -E 'ROUND|Error|Trace' | cut -c1-200
$PY -u pg_dagger.py evalctx --tag smoke_pgc2 --seeds 1 --attempts 4 --eval-rounds -1 --out _n4 2>&1 | grep -v Warning | tail -2
$PY - <<'P'
import numpy as np, torch
r = '/gscratch/robotics/harine/codex_audits/maze_procgen_ast_20261121/runs/smoke_pgc2/'
st = torch.load(r + 'state.pt', map_location='cpu', weights_only=False); z = np.load(r + 'eval_ctx_n4.npz')
print('pg ctx lens', sorted({len(i[1]) for i in st['items']}), 'window', z['window'], z['train_success'].shape)
assert sorted({len(i[1]) for i in st['items']}) == [0, 1, 2] and int(z['window']) == 2
P
echo SMOKE_OK
