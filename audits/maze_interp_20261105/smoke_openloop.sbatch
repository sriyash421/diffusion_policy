#!/bin/bash
# Smoke for (1) open-loop ST (asteroid.py --open-loop: obs features of the attempt start cell on every token, feedback per step) and
# (2) the short-horizon closed-loop BC (run_experiment.py --horizon 4, bc_closed.py). Smoke outputs go to smoke/ and runs/*smoke*.
#SBATCH --account=robotics
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=smoke_openloop
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
D=/gscratch/robotics/harine/codex_audits/maze_quadrants_delta_20261019
echo "== 1. open-loop stream: build_context == the policy's own rollout stream; obs features constant within an attempt"
$PY - <<'PYEOF' 2>&1 | grep -v Warning
import torch, rounds
from asteroid import set_fb
from rounds import make_policy, load_data, get_collate_fn, dict_apply, build_context, loss_with_context, TOKEN_KEYS
set_fb('bfs_dir', True); assert rounds.OBS_OPEN and rounds.FB_MODE == 'bfs_dir'
device = 'cuda'; torch.manual_seed(0); model, pc = make_policy(4, device)
ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer()); model.to(device).eval()
b = dict_apply(get_collate_fn()([ds[i] for i in range(16)]), lambda x: x.to(device)); o = b['obs']
free, st, gl = o['image'][:, 0, 0] < 0.5, o['agent_pos'][:, 0], o['goal_pos'][:, 0]
c = model.rollout_candidates_torch(free, st, gl, 3)
trajs = [[c['traj'][r, k, :int(c['n_steps'][r, k]) + 1].round().long().cpu().numpy() for k in range(3)] for r in range(16)]
ctx = build_context(model, free, gl, trajs); worst = {}
for k in TOKEN_KEYS:
    for r in range(16):
        L = int(ctx['len'][r]); worst[k] = max(worst.get(k, 0.), float((ctx[k][r, :L].float() - c['stream'][k][r, :L].float()).abs().max()))
print('open loop: context vs rollout stream, max abs diff:', worst); assert all(v < 1e-4 for v in worst.values()), worst
spread = 0.; moved = 0
for r in range(16):
    s = [int(v[r]) for v in ctx['att_starts']] + [int(ctx['len'][r])]
    for j in range(3):
        f = ctx['feat'][r, s[j]:s[j + 1]]; spread = max(spread, float((f - f[:1]).abs().max()))
        moved += int(len(set(map(tuple, trajs[r][j].tolist()))) > 1)
fbv = ctx['fb'][..., 2:].abs().sum().item()
print('max feat change within an attempt', spread, '| attempts that moved', moved, '| |dx,dy fb| sum', round(fbv, 3)); assert spread == 0 and moved > 0 and fbv > 0
l_open = loss_with_context(model, b, ctx).item(); rounds.OBS_OPEN = False; l_closed = loss_with_context(model, b, ctx).item(); rounds.OBS_OPEN = True
print('loss open', l_open, 'closed', l_closed); assert l_open != l_closed
print('OPENLOOP_STREAM_OK')
PYEOF
echo "== 2. asteroid smoke --fb bfs_dir --open-loop, then bon_eval on it"
$PY -u asteroid.py smoke --fb bfs_dir --open-loop 2>&1 | grep -v Warning | tail -4
grep -q '"open_loop": true' $I/runs/asteroid_smoke/config.json && echo SMOKE_CONFIG_OPEN_LOOP_OK
$PY -u bon_eval.py --tag asteroid_smoke --seeds 1 --attempts 2 --splits test --no-bc 2>&1 | grep -v Warning | tail -3
echo "== 3. BC UNet horizon 4: 2-update smoke, closed-loop validation"
$PY -u $D/run_experiment.py --setting interp30 --algo bc_unet --seed 0 --variant clean --out-root $I --smoke --horizon 4 --tag h4 2>&1 | grep -v Warning | tail -4
SM=$I/smoke/interp30/bc_unet__clean__h4/seed_0
grep -q '"horizon": 4' $SM/config.json && grep -q '"validation_closed_loop_steps": 64' $SM/config.json && echo SMOKE_H4_CONFIG_OK
echo "== 4. bc_closed: image builder == env image; smoke h4 select; BC64 open loop (exec 64) vs the scan; BC64 closed loop (exec 1)"
$PY - <<'PYEOF' 2>&1 | grep -v Warning
import os, sys, numpy as np, torch
sys.path.insert(0, '/gscratch/robotics/harine/codex_audits/maze_interp_20261105'); import bc_closed, rounds
from rounds import BatchedRollout, build_images_torch
eps = rounds.interp_test_episodes()
roll = BatchedRollout([e['env'] for e in eps], [e['start'] for e in eps], [e['goal'] for e in eps], 1., 1); o = roll.obs_tensor('cpu')
free = torch.as_tensor(eps[0]['nav'].astype(bool))[None].repeat(len(eps), 1, 1)
img = build_images_torch(free, torch.as_tensor(np.stack([e['start'] for e in eps])), torch.as_tensor(np.stack([e['goal'] for e in eps])))
print('image diff', float((o['image'][:, 0].float() - img).abs().max()), o['agent_pos'].shape, o['agent_pos'].dtype); assert torch.allclose(o['image'][:, 0].float(), img)
print('IMAGE_OK')
PYEOF
cd $I
$PY -u bc_closed.py --tag smoke_bc_h4 --run smoke/interp30/bc_unet__clean__h4/seed_0 --select --seeds 1 --splits test,control 2>&1 | grep -v Warning | tail -4
$PY -u bc_closed.py --tag smoke_bc64_open --run runs/interp30/bc_unet__clean/seed_0 --step 7000 --exec 64 --seeds 3 --splits test,control 2>&1 | grep -v Warning | tail -3
$PY -u bc_closed.py --tag smoke_bc64_closed --run runs/interp30/bc_unet__clean/seed_0 --step 7000 --exec 1 --seeds 3 --splits test,control 2>&1 | grep -v Warning | tail -3
$PY -c "
import json, numpy as np
s = json.load(open('$I/test_episodes.json'))['episodes']; c = json.load(open('$I/control_episodes.json'))['episodes']
print('scan BC on these pairs: test', np.mean([e.get('bc_successes', e.get('bc_scan', 0)) for e in s]) / 3, 'control', np.mean([e.get('bc_successes', e.get('bc_scan', 3)) for e in c]) / 3, list(s[0].keys()))"
rm -rf $I/runs/smoke_bc_h4 $I/runs/smoke_bc64_open $I/runs/asteroid_smoke
echo SMOKE_OPENLOOP_ALL_OK
