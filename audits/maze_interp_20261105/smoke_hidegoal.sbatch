#!/bin/bash
# Smoke for the hidden-goal ASTEROID runs (asteroid.py --hide-goal --fb reward; rounds.HIDE_GOAL, FB_MODE 'reward'):
#   1. the encoder never sees the goal (features and the demo loss are unchanged when the goal moves); with HIDE_GOAL off they change
#   2. build_context == the policy's own rollout stream (hidden goal + reward feedback)
#   3. reward feedback: step tokens all zero, the next attempt's first token [r,r,0,0] with r = the attempt ended within 1 of the goal
#   4. BFS / dx,dy feedback with a hidden goal is refused
#   5. asteroid smoke, --target demo (the repo's ASTEROID: expert path after own attempts) + bon_eval, and --target relabel (DAgger)
#SBATCH --account=robotics
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=smoke_hidegoal
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
echo "== 1-4. hidden goal + reward feedback"
$PY - <<'PYEOF' 2>&1 | grep -v Warning
import numpy as np, torch, rounds
from asteroid import set_fb
from rounds import (make_policy, load_data, train_starts, get_collate_fn, dict_apply, build_context, loss_with_context, TOKEN_KEYS,
                    expert_rollout, build_images_torch)
set_fb('reward', False, True); assert rounds.HIDE_GOAL and rounds.FB_MODE == 'reward'
device = 'cuda'; torch.manual_seed(0); model, pc = make_policy(4, device)
ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer()); model.to(device).eval()
S0, G0, free_np = train_starts(ds)
b = dict_apply(get_collate_fn()([ds[i] for i in range(16)]), lambda x: x.to(device)); o = b['obs']
free, st, gl = o['image'][:, 0, 0] < 0.5, o['agent_pos'][:, 0], o['goal_pos'][:, 0]
# 1. features / loss do not depend on the goal
g2 = gl.flip(0)
def feats(goal):
    return model._obs_features({'image': build_images_torch(free, st, goal), 'agent_pos': st, 'goal_pos': goal})
d_hidden = float((feats(gl) - feats(g2)).abs().max())
b2 = {**b, 'obs': {**o}}; img = o['image'].clone(); img[:, :, 1] = 0
for r in range(16):
    gy, gx = int(g2[r, 1]), int(g2[r, 0]); img[r, :, 1, gy, gx] = 1.
b2['obs']['image'] = img; b2['obs']['goal_pos'] = g2[:, None].expand_as(o['goal_pos']).clone()
torch.manual_seed(1); l1 = loss_with_context(model, b, None).item(); torch.manual_seed(1); l2 = loss_with_context(model, b2, None).item()
rounds.HIDE_GOAL = False; d_visible = float((feats(gl) - feats(g2)).abs().max()); rounds.HIDE_GOAL = True
print(f'feature change when the goal moves: hidden {d_hidden:.2e}, visible {d_visible:.2e}; demo loss {l1:.6f} vs moved goal {l2:.6f}')
assert d_hidden == 0 and d_visible > 1e-3 and abs(l1 - l2) < 1e-6
# 2. context == rollout stream
c = model.rollout_candidates_torch(free, st, gl, 3)
trajs = [[c['traj'][r, k, :int(c['n_steps'][r, k]) + 1].round().long().cpu().numpy() for k in range(3)] for r in range(16)]
ctx = build_context(model, free, gl, trajs); worst = {}
for k in TOKEN_KEYS:
    for r in range(16):
        L = int(ctx['len'][r]); worst[k] = max(worst.get(k, 0.), float((ctx[k][r, :L].float() - c['stream'][k][r, :L].float()).abs().max()))
worst['last_fb'] = float((ctx['last_fb'] - c['snaps'][2]['last_fb']).abs().max())
print('hidden goal: context vs rollout stream, max abs diff:', worst); assert all(v < 1e-4 for v in worst.values()), worst
# 3. reward feedback: attempt 1 = the BFS expert (ends on the goal), attempt 2 = 3 cells of attempt 1 (does not reach it), attempt 3 any
rows = [i for i in range(len(S0)) if len(expert_rollout(free_np, S0[i], G0[i])) >= 4][:16]; tr = []
for r in rows:
    e = expert_rollout(free_np, S0[r], G0[r]).astype(np.int64)
    tr.append([e, e[:3], e[:2]])
free = free[:1].repeat(16, 1, 1)
ctx = build_context(model, free, torch.as_tensor(G0[rows], device=device), tr)
ar = torch.arange(16, device=device); s1, s2 = ctx['att_starts'][1], ctx['att_starts'][2]
reach = torch.tensor([float(np.linalg.norm(t[1][-1] - G0[r]) <= 1) for r, t in zip(rows, tr)], device=device)
step_fb = ctx['fb'].clone(); step_fb[ar, s1] = 0; step_fb[ar, s2] = 0
print('boundary fb attempt 2 (row 0):', ctx['fb'][0, s1[0]].tolist(), '| attempt 3 (row 0):', ctx['fb'][0, s2[0]].tolist(),
      '| attempt 2 reached:', int(reach.sum()), '/ 16 | max |step fb|', float(step_fb.abs().max()))
assert torch.equal(ctx['fb'][ar, s1], torch.tensor([1., 1., 0, 0], device=device).expand(16, -1))
assert torch.equal(ctx['fb'][ar, s2], torch.stack([reach, reach, 0 * reach, 0 * reach], -1)) and float(step_fb.abs().max()) == 0
# 4. leaking feedback refused
for m in ('bfs', 'bfs_dir', 'dir'):
    try:
        set_fb(m, False, True); raise SystemExit(f'{m} accepted with a hidden goal')
    except AssertionError:
        pass
print('HIDEGOAL_CHECKS_OK')
PYEOF
echo "== 5a. asteroid smoke --hide-goal --fb reward --target demo (the repo's ASTEROID), bon_eval"
$PY -u asteroid.py smoke --fb reward --hide-goal --target demo 2>&1 | grep -v Warning | tail -4
grep -q '"hide_goal": true' $I/runs/asteroid_smoke/config.json && grep -q '"fb": "reward"' $I/runs/asteroid_smoke/config.json && echo SMOKE_DEMO_CONFIG_OK
$PY -u bon_eval.py --tag asteroid_smoke --seeds 1 --attempts 2 --splits test --no-bc 2>&1 | grep -v Warning | tail -3
echo "== 5b. asteroid smoke --hide-goal --fb reward --target relabel (DAgger)"
$PY -u asteroid.py smoke --fb reward --hide-goal --target relabel 2>&1 | grep -v Warning | tail -4
grep -q '"hide_goal": true' $I/runs/asteroid_smoke/config.json && grep -q '"target": "relabel"' $I/runs/asteroid_smoke/config.json && echo SMOKE_RELABEL_CONFIG_OK
rm -rf $I/runs/asteroid_smoke
echo SMOKE_HIDEGOAL_ALL_OK
