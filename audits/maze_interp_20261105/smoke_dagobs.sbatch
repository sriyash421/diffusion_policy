#!/bin/bash
# Smoke for the DAgger runs with partial / noised observations (2026-11-22 plan, Parts A + B):
#   A. the continuous default + run-setting resolution (old discrete runs re-evaluate in their own setting, identical numbers)
#   1. window mask (-1 outside the 7x7, goal plane only in view, goal_pos 0) + feature invariance to goal moves outside the window
#   2. ddpm noise statistics (eps ~ N(0, 1) at t = 400), goal_pos 0, fresh noise per call
#   3. context rebuilt from stored attempts == the rollout stream (continuous, window obs, reward and BFS feedback)
#   4. continuous expert: reaches the goal from every training start and from random points; its labels replay to its own path
#   5. tiny train -> eval -> plot -> evalctx: interp30c relabel window reward; 11brc relabel ddpm bfs --no-context (items have no context)
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --job-name=smoke_dagobs
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S
unset MAZE_SETTING
R=/gscratch/robotics/harine/codex_audits/maze_rounds_20261103
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
cd $R
if [[ -z ${SKIP_A:-} ]]; then
echo "== A. run-setting resolution + regression (int_s30_bfs, discrete)"
$PY -c "
import run_setting as r
for t in ('int_s30_bfs', 'int_s30_hg_relabel', 'c_s30_bfsdir', 'c_k0long', 'ast_s30'):
    print(t, r.infer(t))
assert r.infer('int_s30_bfs') == 'interp30' and r.infer('c_s30_bfsdir') == 'interp30c' and r.infer('ast_s30') == '11br'
"
$PY -u bon_eval.py --tag int_s30_bfs --seeds 1 --attempts 1 --splits test --no-bc --only-round 0 --suffix _regA 2>&1 | grep -v Warning | tail -2
MAZE_SETTING=interp30 $PY -u bon_eval.py --tag int_s30_bfs --seeds 1 --attempts 1 --splits test --no-bc --only-round 0 --suffix _regB 2>&1 | grep -v Warning | tail -2
$PY -c "
import numpy as np
a = np.load('$I/runs/int_s30_bfs/bon_seeds1_regA.npz', allow_pickle=True); b = np.load('$I/runs/int_s30_bfs/bon_seeds1_regB.npz', allow_pickle=True)
keys = [k for k in a.files if a[k].dtype != object]
assert set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in keys), [k for k in keys if not np.array_equal(a[k], b[k])]
print('REGRESSION_OK: no MAZE_SETTING == MAZE_SETTING=interp30 for the discrete run;', {k: a[k].shape for k in keys if 'success' in k})
"
rm -f $I/runs/int_s30_bfs/bon_seeds1_regA.npz $I/runs/int_s30_bfs/bon_seeds1_regB.npz
if MAZE_SETTING=interp30c $PY -u bon_eval.py --tag int_s30_bfs --seeds 1 --attempts 1 --splits test --no-bc --only-round 0 --suffix _regC > /dev/null 2>&1; then
  echo 'a discrete run evaluated with MAZE_SETTING=interp30c'; exit 1; else echo 'MISMATCH_REFUSED_OK'; fi
rm -f $I/runs/int_s30_bfs/bon_seeds1_regC.npz
fi

if [[ -z ${SKIP_14:-} ]]; then
echo "== 1-4. obs modes, context == stream, continuous expert"
$PY - <<'PYEOF' 2>&1 | grep -v Warning
import numpy as np, torch, rounds
from asteroid import set_fb, set_obs
from rounds import (make_policy, load_data, train_starts, build_context, rollout, TOKEN_KEYS, expert_rollout, expert_labels,
                    build_images_torch, STEP, CONT, SETTING)
from obs_modes import apply_obs, window_mask, abar
assert SETTING == 'interp30c' and CONT, (SETTING, CONT)
dev = 'cuda'; torch.manual_seed(0)
model, pc = make_policy(4, dev); ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer()); model.to(dev).eval()
S0, G0, free_np = train_starts(ds); free1 = torch.as_tensor(free_np, device=dev)[None]
rng = np.random.default_rng(0); B = 64; idx = rng.choice(len(S0), B, replace=False)
st = torch.as_tensor(S0[idx], device=dev); gl = torch.as_tensor(G0[idx], device=dev); free = free1.repeat(B, 1, 1)
cells = np.argwhere(free_np)[:, ::-1].astype(np.float32)        # (x, y) of free cells
# 1. window
img = build_images_torch(free, st, gl); o = apply_obs({'image': img, 'agent_pos': st, 'goal_pos': gl}, 'window')
m = window_mask(st, 11, 11); mm = m[:, None].expand_as(img)
assert bool((o['image'][~mm] == -1).all()) and bool((o['image'][mm] == img[mm]).all()) and float(o['goal_pos'].abs().max()) == 0
g = torch.round(gl).long(); inwin = m[torch.arange(B), g[:, 1], g[:, 0]]
gp = (o['image'][:, 1] == 1).flatten(1).sum(1)
assert bool((gp == inwin.long()).all()), 'goal plane must show the goal exactly when it is in the window'
set_fb('reward'); set_obs('window')
def feats(goal):
    return model._obs_features({'image': build_images_torch(free, st, goal), 'agent_pos': st, 'goal_pos': goal})
f0 = feats(gl); out_d, in_d = [], []
for b in range(B):
    p = torch.round(st[b]).cpu().numpy(); win = (np.abs(cells - p).max(1) <= 3)
    gb = torch.round(gl[b]).cpu().numpy(); outside = cells[~win & ~np.all(cells == gb, 1)]; inside = cells[win & ~np.all(cells == p, 1)]
    for pool, acc in ((outside, out_d), (inside, in_d)):
        if len(pool) and not bool(inwin[b]):
            g2 = gl.clone(); g2[b] = torch.as_tensor(pool[rng.integers(len(pool))], device=dev)
            acc.append(float((feats(g2)[b] - f0[b]).abs().max()))
print(f'window: rows with the goal in view {int(inwin.sum())}/{B}; feature change, goal moved outside the window: max {max(out_d):.2e} '
      f'({len(out_d)} rows); moved inside: min {min(in_d):.2e}')
assert max(out_d) == 0 and min(in_d) > 0
# 2. ddpm
set_obs('ddpm', 400); ab = abar(400); y = apply_obs({'image': img, 'agent_pos': st, 'goal_pos': gl}, 'ddpm', 400)
z = (y['image'] - ab ** .5 * (img * 2 - 1)) / (1 - ab) ** .5; y2 = apply_obs({'image': img, 'agent_pos': st, 'goal_pos': gl}, 'ddpm', 400)
print(f'ddpm t=400: abar {ab:.4f}, eps mean {float(z.mean()):.4f} std {float(z.std()):.4f}, fresh noise per call: {bool((y2["image"] != y["image"]).any())}')
assert abs(float(z.mean())) < .03 and abs(float(z.std()) - 1) < .03 and float(y['goal_pos'].abs().max()) == 0 and bool((y2['image'] != y['image']).any())
# 3. context == stream (window obs is deterministic)
set_obs('window'); rounds.KEEP_STREAM = True
for fb in ('reward', 'bfs'):
    set_fb(fb); mem = [[] for _ in range(B)]; okk = []
    for a in range(3):
        ctx = build_context(model, free, gl, mem); trajs, ok, n = rollout(model, free, st, gl, ctx); okk.append(ok)
        for r, t in enumerate(trajs):
            mem[r].append(t)
    S = rounds.LAST_STREAM; ctx = build_context(model, free, gl, mem); worst = {}
    for k in TOKEN_KEYS:
        for r in range(B):
            L = int(ctx['len'][r]); worst[k] = max(worst.get(k, 0.), float((ctx[k][r, :L].float() - S[k][r, :L].float()).abs().max()))
    assert bool((ctx['len'] == S['len']).all())
    if fb == 'reward':   # end-of-attempt flag of attempt 2 (on attempt 3's first token) == attempt 2 reached the goal
        s2 = ctx['att_starts'][2]; r2 = ctx['fb'][torch.arange(B), s2, 0].cpu().numpy()
        assert np.array_equal(r2 > .5, okk[1]), 'reward flag != success'
    print(f'context vs rollout stream ({fb} feedback, window obs, continuous), max abs diff:', {k: f'{v:.1e}' for k, v in worst.items()},
          '| success by attempt', [round(float(o.mean()), 3) for o in okk])
    assert all(v < 1e-4 for v in worst.values()), worst
rounds.KEEP_STREAM = False
# 4. continuous expert
lens = [len(expert_rollout(free_np, S0[i], G0[i])) - 1 for i in range(len(S0))]
P = cells[rng.integers(len(cells), size=300)] + rng.uniform(-.45, .45, (300, 2)).astype(np.float32); Gr = cells[rng.integers(len(cells), size=300)]
keep = np.linalg.norm(P - Gr, axis=1) > 1; P, Gr = P[keep], Gr[keep]
lr = [len(expert_rollout(free_np, p, g_)) - 1 for p, g_ in zip(P, Gr)]
pts = expert_rollout(free_np, S0[0], G0[0]); a, stop = expert_labels(free_np, G0[0], pts[:-1])
rep = [torch.as_tensor(pts[:1])]
for x in a:
    rep.append(STEP(free1.cpu(), rep[-1], torch.as_tensor(x[None])))
err = float(np.abs(torch.cat(rep).numpy() - pts).max())
print(f'continuous expert: all {len(S0)} training starts reach the goal (steps mean {np.mean(lens):.2f}, max {max(lens)}); '
      f'{len(P)} random points reach it (steps mean {np.mean(lr):.2f}, max {max(lr)}); label replay err {err:.1e}; stop flags {stop.astype(int).tolist()}')
assert err < 1e-4 and stop[-1] and not stop[:-1].any()
print('DAGOBS_CHECKS_OK')
PYEOF
fi

echo "== 5a. smoke interp30c: relabel, window, reward"
$PY -u asteroid.py smoke --target relabel --obs window --fb reward 2>&1 | grep -v Warning | grep -E "ROUND|success by attempt|saved|Error|error" | tail -14
$PY -c "
import json; c = json.load(open('$I/runs/asteroid_smoke/config.json'))
assert c['setting'] == 'interp30c' and c['actions'] == 'continuous' and c['obs'] == 'window' and c['fb'] == 'reward' and not c['no_context']
assert not c['feedback_leaks_goal_distance']; print('SMOKE_A_CONFIG_OK', c['obs_desc'])
import numpy as np; z = np.load('$I/runs/asteroid_smoke/eval_ctx.npz'); print({k: z[k].shape for k in z.files if 'success' in k})
assert {'train_success', 'train_heldout_success', 'test_success'} <= set(z.files)
"
rm -rf $I/runs/asteroid_smoke
echo "== 5b. smoke 11brc: relabel, ddpm t=400, bfs, no context"
MAZE_SETTING=11brc $PY -u asteroid.py smoke --target relabel --obs ddpm --obs-t 400 --fb bfs --no-context 2>&1 | grep -v Warning | grep -E "ROUND|success by attempt|saved|Error|error" | tail -14
MAZE_SETTING=11brc $PY -c "
import json, torch; c = json.load(open('$R/runs/asteroid_smoke/config.json'))
assert c['setting'] == '11brc' and c['obs'] == 'ddpm' and c['obs_t'] == 400 and c['no_context'] and c['feedback_leaks_goal_distance']
st = torch.load('$R/runs/asteroid_smoke/state.pt', map_location='cpu', weights_only=False)
assert all(len(it[1]) == 0 for it in st['items']), 'no-context run has items with context'
print('SMOKE_B_OK: items', len(st['items']), 'all with empty context; leak flag', c['feedback_leaks_goal_distance'])
"
rm -rf $R/runs/asteroid_smoke
echo SMOKE_DAGOBS_ALL_OK
