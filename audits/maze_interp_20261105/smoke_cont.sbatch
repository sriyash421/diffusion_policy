#!/bin/bash
# Smoke for MAZE_SETTING=interp30c (continuous positions, smoothed-A* demos, real-valued actions): the ST rollout stream equals
# build_context on real-valued trajectories (1 and 2 attempts), trajectories leave the grid, actions are not snapped; asteroid smoke
# + bon_eval; BC UNet horizon 4 smoke; bc_closed --select on it. Smoke outputs removed afterwards.
#SBATCH --account=robotics
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=smoke_cont
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30c
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
D=/gscratch/robotics/harine/codex_audits/maze_quadrants_delta_20261019
echo "== 1. continuous ST stream vs build_context"
$PY - <<'PYEOF' 2>&1 | grep -v Warning
import numpy as np, torch, rounds
from asteroid import set_fb
from rounds import make_policy, load_data, train_starts, get_collate_fn, dict_apply, build_context, loss_with_context, rollout, TOKEN_KEYS
assert rounds.CONT and rounds.STEP.__module__ == 'cmaze'
set_fb('bfs_dir'); device = 'cuda'; torch.manual_seed(0); model, pc = make_policy(4, device); assert model.snap_cardinal is False
ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer()); model.to(device).eval()
S0, G0, free_np = train_starts(ds); print('train starts', S0.shape, 'non-integer starts', float((np.abs(S0 - np.rint(S0)) > 1e-4).any(1).mean()))
b = dict_apply(get_collate_fn()([ds[i] for i in range(16)]), lambda x: x.to(device)); a = b['action'][b['attention_mask'].bool()]
print('demo action norms', a.norm(dim=-1).min().item(), a.norm(dim=-1).max().item())
B = 16; free = torch.as_tensor(free_np, device=device)[None].repeat(B, 1, 1); st = torch.as_tensor(S0[:B], device=device); gl = torch.as_tensor(G0[:B], device=device)
orig = model._step; cap = {}
def step(obs, fr):
    out = orig(obs, fr); cap['S'] = model._st; return out
model._step = step
mem = [[] for _ in range(B)]; worst = {}
for att in range(2):
    ctx = build_context(model, free, gl, mem)
    trajs, ok, ns = rollout(model, free, st, gl, ctx)
    S = cap['S']
    for r in range(B):
        mem[r].append(trajs[r])
    full = build_context(model, free, gl, mem)
    for k in TOKEN_KEYS:
        for r in range(B):
            L = int(full['len'][r]); worst[k] = max(worst.get(k, 0.), float((full[k][r, :L].float() - S[k][r, :L].float()).abs().max()))
    print('attempt', att + 1, 'success', ok.mean(), 'mean steps', ns.mean())
print('stream vs build_context, max abs diff:', worst); assert all(v < 1e-4 for v in worst.values()), worst
allp = np.concatenate([t for m in mem for t in m]); print('traj dtype', allp.dtype, 'off-grid positions', float((np.abs(allp - np.rint(allp)) > 1e-3).any(1).mean()))
mv = np.concatenate([np.diff(t, axis=0) for m in mem for t in m]); print('executed move norms: max', np.linalg.norm(mv, axis=1).max(), 'non-cardinal', float(((np.abs(mv) > 1e-3).all(1)).mean()))
assert allp.dtype == np.float32 and np.linalg.norm(mv, axis=1).max() <= 1 + 1e-4
print('loss n=0', loss_with_context(model, b, None).item())
print('CONT_STREAM_OK')
PYEOF
echo "== 2. asteroid smoke (interp30c, bfs_dir) + bon_eval"
$PY -u asteroid.py smoke --fb bfs_dir 2>&1 | grep -v Warning | tail -3
$PY -u bon_eval.py --tag asteroid_smoke --seeds 1 --attempts 2 --splits test 2>&1 | grep -v Warning | tail -3
$PY -c "import numpy as np; z = np.load('$I/runs/asteroid_smoke/bon_seeds1.npz'); print(z.files); assert 'test_bc_success' not in z.files" && echo BON_NO_BC_OK
echo "== 3. BC UNet horizon 4 on interp30c (smoke)"
$PY -u $D/run_experiment.py --setting interp30c --algo bc_unet --seed 0 --variant clean --out-root $I --smoke --horizon 4 --tag h4 2>&1 | grep -v Warning | tail -3
SM=$I/smoke/interp30c/bc_unet__clean__h4/seed_0; ls $SM
cd $I
$PY -u bc_closed.py --tag smoke_cbc --run smoke/interp30c/bc_unet__clean__h4/seed_0 --select --seeds 1 --splits train,test,control 2>&1 | grep -v Warning | tail -5
rm -rf $I/runs/smoke_cbc $I/runs/asteroid_smoke
echo SMOKE_CONT_ALL_OK
