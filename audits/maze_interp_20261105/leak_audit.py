"""Leakage audit for the DAgger runs with partial / noised observations (2026-11-22 plan). How much does the policy's input tell it about
the goal, per observation mode and feedback?

  probe      a small MLP is trained to name the goal cell (71 free cells, chance 1/71) from what the policy gets:
               obs1      one step's encoder input (image after the obs mode, agent_pos, goal_pos = 0)
               obs1+bfs  the same + that step's BFS feedback (-BFS / 32 of the agent's cell)
               walk+bfs  a 16-step random walk's positions + their BFS feedback (what one attempt in context carries)
               walk+rew  the same walk's positions + the reward flag (1 if the walk ended within 1 of the goal)
             for obs in none (goal visible), window, ddpm t = 200 / 400 / 600; interp30c layout; 40k train / 10k test samples
  behaviour  (after training) attempt-1 success of a run with the goal in the OBSERVATION swapped for a random other free cell (the env,
             success and feedback keep the true goal): a drop means the policy reads the goal from its observation
  split      no held-out goal appears in any DAgger item (state.pt) of a run

  python leak_audit.py probe                       (GPU or CPU job)  -> report/leak_probe.json
  python leak_audit.py behaviour --tags a,b,c      (GPU job)         -> report/leak_behaviour.json
  python leak_audit.py split --tags a,b,c
"""
import argparse, json, os, sys
from pathlib import Path
I = Path(__file__).resolve().parent; REP = I / 'report'
sys.path.insert(0, str(I.parent / 'maze_rounds_20261103'))
import numpy as np
import torch
import torch.nn as nn


def walk(free, S, steps, rng):
    """Random walks of cardinal unit moves (blocked moves stay) from S (N, 2) cells; (N, steps+1, 2)."""
    H, W = free.shape; P = [S.copy()]; p = S.copy()
    mv = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    for _ in range(steps):
        q = p + mv[rng.integers(4, size=len(p))]; ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
        ok[ok] = free[q[ok, 1], q[ok, 0]]; p = np.where(ok[:, None], q, p); P.append(p.copy())
    return np.stack(P, 1)


def fit_probe(Xtr, ytr, Xte, yte, ncls, device, epochs=30):
    torch.manual_seed(0); X = torch.as_tensor(Xtr, device=device).float(); y = torch.as_tensor(ytr, device=device).long()
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, ncls)).to(device)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=device)
        for j in range(0, len(X), 512):
            b = perm[j:j + 512]; loss = nn.functional.cross_entropy(net((X[b] - mu) / sd), y[b]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Xt = (torch.as_tensor(Xte, device=device).float() - mu) / sd
        return float((net(Xt).argmax(1).cpu().numpy() == yte).mean())


def probe(args):
    os.environ.setdefault('MAZE_SETTING', 'interp30c')
    import rounds
    from rounds import load_data, train_starts, make_policy, build_images_torch
    from obs_modes import apply_obs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = load_data(make_policy(2, 'cpu')[1]['shape_meta']); _, _, free = train_starts(ds)
    cells = np.argwhere(free)[:, ::-1].astype(np.int64); C = len(cells); rng = np.random.default_rng(0); n = 50000
    gi = rng.integers(C, size=n); G = cells[gi]; S = cells[rng.integers(C, size=n)]
    free_t = torch.as_tensor(free)
    from collections import deque
    def bfs_table(g):
        d = np.full(free.shape, 10 ** 6, np.int64); d[g[1], g[0]] = 0; q = deque([g])
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < free.shape[1] and 0 <= ny < free.shape[0] and free[ny, nx] and d[ny, nx] > d[y, x] + 1:
                    d[ny, nx] = d[y, x] + 1; q.append((nx, ny))
        return d
    dist = {c: bfs_table(c) for c in map(tuple, cells)}   # BFS distance field per goal cell (H, W)
    bfs1 = np.array([dist[tuple(g)][s[1], s[0]] for s, g in zip(S, G)], np.float32) / 32.
    Wk = walk(free, S, 16, rng); bfsw = np.array([[dist[tuple(g)][q[1], q[0]] for q in w] for w, g in zip(Wk, G)], np.float32) / 32.
    rew = (np.linalg.norm(Wk[:, -1] - G, axis=1) <= 1).astype(np.float32)
    ntr = 40000; res = dict(chance=round(1 / C, 4), cells=C, samples=dict(train=ntr, test=n - ntr))
    for mode, t in (('none', 0), ('window', 0), ('ddpm', 200), ('ddpm', 400), ('ddpm', 600)):
        key = mode if mode != 'ddpm' else f'ddpm{t}'
        img = build_images_torch(free_t[None].expand(n, -1, -1), torch.as_tensor(S).float(), torch.as_tensor(G).float())
        o = apply_obs({'image': img, 'agent_pos': torch.as_tensor(S).float(), 'goal_pos': torch.as_tensor(G).float()}, mode, t,
                      torch.Generator().manual_seed(1))
        obs1 = torch.cat([o['image'].flatten(1), o['agent_pos'], o['goal_pos']], 1).numpy()
        feats = {'obs1': obs1, 'obs1+bfs': np.concatenate([obs1, bfs1[:, None]], 1)}
        res[key] = {k: round(fit_probe(v[:ntr], gi[:ntr], v[ntr:], gi[ntr:], C, device), 4) for k, v in feats.items()}
        print(key, res[key], flush=True)
    wf = Wk.reshape(n, -1).astype(np.float32)
    res['walk'] = {'walk+bfs': round(fit_probe(np.concatenate([wf, bfsw], 1)[:ntr], gi[:ntr], np.concatenate([wf, bfsw], 1)[ntr:], gi[ntr:], C, device), 4),
                   'walk+rew': round(fit_probe(np.concatenate([wf, rew[:, None]], 1)[:ntr], gi[:ntr], np.concatenate([wf, rew[:, None]], 1)[ntr:], gi[ntr:], C, device), 4)}
    print('walk', res['walk']); REP.mkdir(exist_ok=True); (REP / 'leak_probe.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res))


def behaviour(args):
    import run_setting
    out = {}
    for tag in args.tags.split(','):
        os.environ.pop('MAZE_SETTING', None); s = run_setting.infer(tag)
        cmd = (f'MAZE_SETTING={s} {sys.executable} -u {__file__} behaviour1 --tags {tag}')
        r = os.popen(cmd).read().strip().splitlines(); out[tag] = json.loads(r[-1]); print(tag, out[tag], flush=True)
    (REP / 'leak_behaviour.json').write_text(json.dumps(out, indent=1))


@torch.no_grad()
def behaviour1(args):
    """One run (MAZE_SETTING set by the caller): attempt-1 success with the true vs a swapped observation goal, round 0 and the last."""
    import rounds
    from rounds import make_policy, load_data, train_starts, rollout, HERE
    from asteroid import apply_cfg, test_set
    tag = args.tags; out = HERE / 'runs' / tag; cfg = json.loads((out / 'config.json').read_text()); apply_cfg(cfg)
    device = 'cuda'; R = cfg['rounds']; model, pc = make_policy(R, device); ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer())
    tr_s, tr_g, free_np = train_starts(ds); _, te_s, te_g = test_set(free_np); cells = np.argwhere(free_np)[:, ::-1].astype(np.float32)
    rng = np.random.default_rng(0); res = {}
    orig = rounds.RoundsPolicy._obs_features
    for split, S0, G0 in (('train', tr_s, tr_g), ('test', te_s, te_g)):
        S = np.repeat(S0, 4, 0); G = np.repeat(G0, 4, 0); Gs = cells[rng.integers(len(cells), size=len(G))]
        same = np.all(np.rint(Gs) == np.rint(G), 1); Gs[same] = cells[(rng.integers(len(cells), size=same.sum()))]
        free = torch.as_tensor(free_np, device=device)[None].repeat(len(S), 1, 1); St = torch.as_tensor(S, device=device); Gt = torch.as_tensor(G, device=device)
        Gst = torch.as_tensor(Gs, device=device)
        for m in sorted({0, R - 1}):
            model.load_state_dict(torch.load(out / f'round_{m}.pt', map_location='cpu', weights_only=False)['model']); model.to(device).eval()
            torch.manual_seed(1); rounds.RoundsPolicy._obs_features = orig; _, ok, _ = rollout(model, free, St, Gt, None)
            def swapped(self, obs, _orig=orig):   # the goal the policy observes -> the swapped cell (image goal plane + goal_pos)
                from rounds import build_images_torch   # rollout without context: every call is the whole batch, one row per (start, goal)
                assert obs['agent_pos'].shape[0] == len(Gst)
                return _orig(self, {**obs, 'image': build_images_torch(free, obs['agent_pos'], Gst), 'goal_pos': Gst})
            torch.manual_seed(1); rounds.RoundsPolicy._obs_features = swapped; _, ok2, _ = rollout(model, free, St, Gt, None)
            rounds.RoundsPolicy._obs_features = orig
            res[f'{split}_round{m}'] = dict(true_goal=round(float(ok.mean()), 4), swapped_obs_goal=round(float(ok2.mean()), 4))
    print(json.dumps(res))


def split(args):
    import run_setting
    for tag in args.tags.split(','):
        os.environ.pop('MAZE_SETTING', None); s = run_setting.infer(tag)
        r = os.popen(f'MAZE_SETTING={s} {sys.executable} -u {__file__} split1 --tags {tag}').read().strip().splitlines(); print(r[-1], flush=True)


def split1(args):
    import rounds
    from rounds import make_policy, load_data, train_starts, HERE, test_episodes
    tag = args.tags; st = torch.load(HERE / 'runs' / tag / 'state.pt', map_location='cpu', weights_only=False)
    ds = load_data(make_policy(2, 'cpu')[1]['shape_meta']); _, G0, _ = train_starts(ds)
    used = {tuple(np.rint(G0[it[0]]).astype(int)) for it in st['items']}
    test_goals = {tuple(np.rint(e['goal']).astype(int)) for e in test_episodes()}
    bad = used & test_goals
    print(json.dumps(dict(tag=tag, items=len(st['items']), goals_in_items=len(used), held_out_goals_in_items=len(bad))))
    assert not bad, bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['probe', 'behaviour', 'behaviour1', 'split', 'split1']); ap.add_argument('--tags', default='')
    a = ap.parse_args(); {'probe': probe, 'behaviour': behaviour, 'behaviour1': behaviour1, 'split': split, 'split1': split1}[a.mode](a)
