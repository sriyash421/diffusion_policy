#!/bin/bash
# Smoke for --no-stop (2026-11-23): tiny DAgger train + evalctx + mse_eval on interp30c (window, reward fb) and on the new mazes.
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
A=/gscratch/robotics/harine/codex_audits; PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
export PYTHONPATH=/gscratch/robotics/harine/L2S:$A/maze_rounds_20261103:$A/maze_procgen_ast_20261121
cd $A/maze_rounds_20261103
export MAZE_SETTING=interp30c
rm -rf $A/maze_interp_20261105/runs/smoke_nostop $A/maze_procgen_ast_20261121/runs/smoke_nostop_pg
$PY -u asteroid.py train --tag smoke_nostop --target relabel --fb reward --rounds 2 --batch 16 --micro 16 --steps-per-iter 2 --iters 1 \
  --test-every 0 --obs window --no-stop 2>&1 | grep -v Warning | tail -3
$PY -c "import json;print('config no_stop =', json.load(open('$A/maze_interp_20261105/runs/smoke_nostop/config.json'))['no_stop'])"
$PY -u asteroid.py evalctx --tag smoke_nostop --seeds 1 --attempts 2 --splits train 2>&1 | grep -v Warning | tail -2
unset MAZE_SETTING
$PY -u $A/maze_interp_20261105/mse_eval.py --tags smoke_nostop --pairs 20 --seeds 1 2>&1 | grep -v Warning | grep success
cd $A/maze_procgen_ast_20261121
$PY -u pg_dagger.py train --tag smoke_nostop_pg --fb reward --rounds 2 --iters 1 --steps-per-iter 2 --batch 16 --micro 16 --max-levels 20 --no-stop 2>&1 | grep -v Warning | tail -2
$PY -u $A/maze_interp_20261105/mse_eval.py --pg --tags smoke_nostop_pg --pairs 20 --seeds 1 2>&1 | grep -v Warning | grep success
