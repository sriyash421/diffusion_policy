#!/bin/bash
# ASTEROID-style ST on interp30 with the ast_s30 schedule (8 rounds × 30 iterations × batch 128 × 30 steps, constant lr 1e-4),
# then eval (8 seeds × 8 attempts) and best-of-n (bon_eval). Resumes per round from runs/<tag>/state.pt after preemption.
#   sbatch --job-name=int_s30_<fb> st_run.sbatch <tag> <fb> [extra train flags]   fb in bfs | bfs_dir | dir
#   e.g. ablation: st_run.sbatch int_s30_nofb bfs --no-ctx-feedback   (feedback of the attempts in context zeroed)
#   obs noise: st_run.sbatch int_s30_on400 bfs_dir --obs-noise flat --obs-noise-t 400   (also writes bon_seeds8_noisyobs.npz)
#   matched-steps STk0: st_run.sbatch int_k0long bfs_dir --rounds 1 --iters 240   (later flags override the defaults)
#   sbatch --job-name=int_smoke st_run.sbatch smoke                 (asteroid smoke + bon_eval smoke on interp30)
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40|l40s|a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
[[ ${1:-} == c_* ]] && export MAZE_SETTING=interp30c   # c_<tag>: continuous maze (cmaze.py)
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
tag=$1
echo "== start $(date) on $(hostname) restart=${SLURM_RESTART_COUNT:-0} tag=$tag"
if [[ $tag == smoke ]]; then
  $PY -u asteroid.py smoke --fb bfs_dir 2>&1 | grep -v Warning | tail -12
  grep -q train_success $I/runs/asteroid_smoke/log.jsonl
  $PY -u bon_eval.py --tag asteroid_smoke --seeds 1 --attempts 2 --splits test 2>&1 | grep -v Warning | tail -4
  $PY - <<'PYEOF'
import numpy as np
z = np.load('/gscratch/robotics/harine/codex_audits/maze_interp_20261105/runs/asteroid_smoke/bon_seeds1.npz')
assert z['test_st_success'].shape[2] == 50 and z['test_bc_success'].shape[0] == 50, (z['test_st_success'].shape, z['test_bc_success'].shape)
for k in ('st', 'bc'):
    s, b = z[f'test_{k}_success'].ravel(), z[f'test_{k}_bfs'].ravel()
    assert (b[s] <= 1).all(); print(k, 'success', s.mean().round(3))
print('INTERP_SMOKE_OK')
PYEOF
  exit 0
fi
fb=$2; shift 2
[[ -f $I/runs/$tag/TRAINED ]] || $PY -u asteroid.py train --tag "$tag" --fb "$fb" --rounds 8 --batch 128 --micro 32 --lr 1e-4 \
    --steps-per-iter 30 --iters 30 --test-every 30 --test-seeds 8 --train-seeds 1 "$@" 2>&1 | grep -v Warning | grep -E "ROUND|resumed|Traceback|Error"
[[ -f $I/runs/$tag/eval_seeds8.npz ]] || $PY -u asteroid.py eval --tag "$tag" --seeds 8 --attempts 8 2>&1 | grep -v Warning
$PY -u asteroid.py plot --tag "$tag" --seeds 8 2>&1 | grep -v Warning | tail -2
$PY -c "import numpy as np, sys; sys.exit(0 if 'train_st_success' in np.load('$I/runs/$tag/bon_seeds8.npz').files else 1)" 2>/dev/null || $PY -u bon_eval.py --tag "$tag" --seeds 8 --attempts 8 2>&1 | grep -v Warning | tail -6
if grep -q '"obs_noise": "\(flat\|ramp\)"' $I/runs/$tag/config.json; then   # trained with observation noise: also eval with it
  [[ -f $I/runs/$tag/bon_seeds8_noisyobs.npz ]] || $PY -u bon_eval.py --tag "$tag" --seeds 8 --attempts 8 --eval-obs-noise 2>&1 | grep -v Warning | tail -6
fi
echo "== done $(date)"
