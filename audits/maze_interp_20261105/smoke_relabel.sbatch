#!/bin/bash
# Smoke for ASTEROID / DAgger relabelling (asteroid.py --target relabel) and its BC baseline (dagger_bc_data.py + run_experiment --dataset).
#SBATCH --account=robotics
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=smoke_relabel
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
D=/gscratch/robotics/harine/codex_audits/maze_quadrants_delta_20261019
echo "== 1. BFS expert tables vs the demo generator; relabel loss == demo loss where the expert path is the demo"
$PY - <<'PYEOF' 2>&1 | grep -v Warning
import numpy as np, torch, rounds
from asteroid import set_fb
from rounds import (make_policy, load_data, train_starts, get_collate_fn, dict_apply, loss_with_context, loss_relabel, expert_rollout,
                    expert_labels, expert_tables, maze)
set_fb('bfs_dir'); device = 'cuda'; torch.manual_seed(0)
model, pc = make_policy(4, device); ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer()); model.to(device).train()
S0, G0, free_np = train_starts(ds); mv, dist = expert_tables(free_np)
same, opt = [], 0
for i in range(len(ds)):
    d = dict_apply(get_collate_fn()([ds[i]]), lambda x: x[0]); m = np.asarray(d['attention_mask']).astype(bool); cells = np.asarray(d['obs']['agent_pos'])[m].astype(int)
    demo = np.concatenate([cells, cells[-1:] + np.asarray(d['action'])[m][-1:].astype(int)])
    e = expert_rollout(free_np, S0[i], G0[i]); opt += len(e) == len(demo); same.append(len(e) == len(demo) and np.array_equal(e, demo))
    a, st = expert_labels(free_np, G0[i], e[:-1]); assert np.array_equal(a, np.diff(e, axis=0)) and st[-1] and not st[:-1].any()
print(f'expert path length == demo length: {opt}/{len(ds)}; identical path: {sum(same)}/{len(ds)}'); assert opt == len(ds)
rows = [i for i in range(len(ds)) if same[i]][:32]
b = dict_apply(get_collate_fn()([ds[i] for i in rows]), lambda x: x.to(device))
free = torch.as_tensor(free_np, device=device)[None].repeat(len(rows), 1, 1); gl = torch.as_tensor(G0[rows], device=device)
model.eval(); torch.manual_seed(1); ld = loss_with_context(model, b, None).item()
lr, nt = loss_relabel(model, free, gl, [[] for _ in rows], [expert_rollout(free_np, S0[i], G0[i]) for i in rows])
print('demo loss', ld, 'relabel loss', lr.item(), 'tokens', nt, int(b['attention_mask'].sum())); assert abs(ld - lr.item()) < 1e-4 and nt == int(b['attention_mask'].sum())
# relabel with context: learner attempts from the (untrained) policy, then gradient flows
from asteroid import chain
free1 = torch.as_tensor(free_np, device=device)[None]
mem, succ, steps = chain(model, free1, S0[:16], G0[:16], 3, 2, 512, device)
model.train(); loss, nt = loss_relabel(model, free1.repeat(16, 1, 1), torch.as_tensor(G0[:16], device=device), [m[:2] for m in mem], [m[2] for m in mem])
loss.backward(); gn = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** .5
print('relabel loss with 2 context attempts', loss.item(), 'tokens', nt, 'steps of attempt 3', steps[2].tolist()[:8], 'grad norm', float(gn))
assert torch.isfinite(loss) and gn > 0 and nt == int(steps[2].sum())
print('RELABEL_CHECKS_OK')
PYEOF
echo "== 2. asteroid smoke --target relabel --fb bfs_dir"
$PY -u asteroid.py smoke --fb bfs_dir --target relabel 2>&1 | grep -v Warning | tail -5
grep -q '"target": "relabel"' $I/runs/asteroid_smoke/config.json && echo SMOKE_CONFIG_RELABEL_OK
echo "== 3. BC data from the smoke run, 2-update BC on it, bc_closed"
cd $I
$PY -u dagger_bc_data.py --tag asteroid_smoke 2>&1 | grep -v Warning
$PY -u $D/run_experiment.py --setting interp30 --algo bc_unet --seed 0 --variant clean --out-root $I --smoke --dataset $I/runs/asteroid_smoke/bc_data.zarr \
    --val-ratio .02 --tag dagsmoke 2>&1 | grep -v Warning | tail -2
grep -q 'asteroid_smoke/bc_data.zarr' $I/smoke/interp30/bc_unet__clean__dagsmoke/seed_0/config.json && echo SMOKE_BC_DATASET_OK
$PY -u bc_closed.py --tag smoke_dagbc --run smoke/interp30/bc_unet__clean__dagsmoke/seed_0 --select --exec 64 --seeds 1 --splits test 2>&1 | grep -v Warning | tail -2
rm -rf $I/runs/smoke_dagbc $I/runs/asteroid_smoke $I/smoke/interp30/bc_unet__clean__dagsmoke
echo SMOKE_RELABEL_ALL_OK
