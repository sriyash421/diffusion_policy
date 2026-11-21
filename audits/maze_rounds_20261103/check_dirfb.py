"""--fb dir / bfs_dir check (GPU): python check_dirfb.py <mode>.: build_context reproduces the policy's own rollout stream, and every feedback token is [0, 0, (goal - cell)/10]."""
import sys
import torch
import torch.nn.functional as F
import rounds
from rounds import make_policy, load_data, get_collate_fn, dict_apply, build_context, TOKEN_KEYS, loss_with_context
MODE = sys.argv[1] if len(sys.argv) > 1 else 'dir'; rounds.FB_MODE = MODE; device = 'cuda'; torch.manual_seed(0)
model, pc = make_policy(4, device); ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer()); model.to(device).eval()
b = dict_apply(get_collate_fn()([ds[i] for i in range(16)]), lambda x: x.to(device)); o = b['obs']
free, st, gl = o['image'][:, 0, 0] < 0.5, o['agent_pos'][:, 0], o['goal_pos'][:, 0]
c = model.rollout_candidates_torch(free, st, gl, 3)
trajs = [[c['traj'][r, k, :int(c['n_steps'][r, k]) + 1].round().long().cpu().numpy() for k in range(3)] for r in range(16)]
ctx = build_context(model, free, gl, trajs); worst = {}
for k in TOKEN_KEYS:
    for r in range(16):
        L = int(ctx['len'][r]); worst[k] = max(worst.get(k, 0.), float((ctx[k][r, :L].float() - c['stream'][k][r, :L].float()).abs().max()))
worst['last_fb'] = float((ctx['last_fb'] - c['snaps'][2]['last_fb']).abs().max())
print('context vs rollout stream, max abs diff:', worst, flush=True); assert all(v < 1e-4 for v in worst.values()), worst
bad = 0
field, unreach = model.verifier._goal_distance_field_torch(free, gl); scale = model._scale(free)
def nbfs(r, q):
    return -model.verifier._sample_goal_distance_torch(free[r:r + 1].repeat(len(q), 1, 1), field[r:r + 1].repeat(len(q), 1, 1), q, unreach) / scale
def step_tok(r, q):
    d = (gl[r] - q) / 10; z = torch.zeros(len(q), 1, device=device); b = nbfs(r, q)[:, None]
    return torch.cat([z, z, d], -1) if MODE == 'dir' else torch.cat([b, z, d], -1)
def end_tok(r, q):
    d = (gl[r] - q[None]) / 10; z = torch.zeros(1, 1, device=device); b = nbfs(r, q[None])[:, None]
    return (torch.cat([z, z, d], -1) if MODE == 'dir' else torch.cat([b, b, d], -1))[0]
for r in range(16):
    cells = [torch.as_tensor(t[:-1], device=device).float() for t in trajs[r]]; ends = [torch.as_tensor(t[-1], device=device).float() for t in trajs[r]]
    exp = step_tok(r, torch.cat(cells)); s0 = 0
    for j, t in enumerate(cells):
        exp[s0] = 0 if j == 0 else end_tok(r, ends[j - 1]); s0 += len(t)
    bad += int((ctx['fb'][r, :len(exp)] - exp).abs().max() > 1e-5)
print(MODE, 'rows with fb != expected tokens:', bad); assert bad == 0
print('row 0 fb tokens 0..3:', ctx['fb'][0, :4].tolist(), 'boundary', ctx['fb'][0, int(ctx['att_starts'][1][0])].tolist())
print('loss n=0', loss_with_context(model, b, None).item(), 'n=3', loss_with_context(model, b, ctx).item()); print('CHECK_OK', flush=True)
