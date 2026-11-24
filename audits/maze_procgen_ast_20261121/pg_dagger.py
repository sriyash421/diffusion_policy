"""Stage 2 of the 2026-11-22 plan: the DAgger runs of stage 1 (asteroid.py --target relabel, continuous actions, --obs window | ddpm,
arms fbR / fbB / nofb / noctx) on DIFFERENT MAZES.

Levels (levels.npz): the L2S procgen layouts (data/procgen_mazes/layouts.npz, its train / eval split): train = the 800 train layouts x 4
goals, test = the 200 held-out layouts x 5 goals. Start = the bottom-left free cell centre (procgen), goal = a free cell more than 1 away,
drawn with seed = layout index. Every row of a batch has its own maze (`free`).

Same model (rounds.RoundsPolicy), schedule and rounds as stage 1:
  round 0: the continuous expert's rollout from every train level (no context)
  round n: the current policy makes n + 1 attempts in a row per train level (attempt j sees attempts 1..j-1; --no-context: 1 attempt from
           an empty stream); the continuous expert labels every point of the last one; data aggregated over rounds
  every round: fresh AdamW, --iters iterations x batch --batch x --steps-per-iter steps, constant lr
Continuous expert (as rounds.cont_expert_action, per layout): the BFS cell path from round(p) (BFS distance field from the goal, neighbour
order +x, -x, +y, -y), string-pulled from p (rounds.first_waypoint), the first resampled step (<= 1). BFS feedback is scaled by the fixed
11 x 11 diameter 32 (the same for every layout).

  python pg_dagger.py smoke
  python pg_dagger.py train   --tag pg_dag_window_fbR --fb reward --obs window [--no-context]
  python pg_dagger.py evalctx --tag pg_dag_window_fbR        -> runs/<tag>/eval_ctx.npz (train / test success, rounds x attempts x levels x seeds)
"""
import argparse, json, os, sys, time
from collections import deque
from pathlib import Path
os.environ.setdefault('MAZE_SETTING', 'interp30c')   # continuous env step + the interp30c normalizer (same 11 x 11 coordinate ranges)
PG = Path(__file__).resolve().parent; AUD = PG.parent
sys.path[:0] = [str(AUD / 'maze_rounds_20261103')]
import numpy as np
import torch
import torch.nn.functional as F
import rounds
from rounds import RoundsPolicy, make_policy, load_data, build_context, rollout, TOKEN_KEYS, _expand_idx, STEP, first_waypoint, STOP_R, maze, SPEC
from asteroid import set_fb, set_obs
assert rounds.CONT and rounds.SETTING == 'interp30c', rounds.SETTING
RoundsPolicy._scale = lambda self, free: 32.0          # BFS feedback scale: the 11 x 11 diameter, the same for every layout
RUNS = PG / 'runs'


# ---------------------------------------------------------------- levels
def levels():
    f = PG / 'levels.npz'
    if f.exists():
        return dict(np.load(f))
    d = np.load(SPEC['layouts']); nav = d['nav'].astype(bool); out = {k: [] for k in ('nav', 'start', 'goal', 'layout', 'split')}
    for split, idx, ng in ((0, d['train_indices'], 4), (1, d['eval_indices'], 5)):
        for li in idx:
            n = nav[li]; s = maze.first_free_bottom_left(d['nav'][li]).astype(np.float32)
            cells = np.argwhere(n)[:, ::-1].astype(np.float32); far = cells[np.linalg.norm(cells - s, axis=1) > 1]
            rng = np.random.default_rng(int(li))
            for g in far[rng.choice(len(far), ng, replace=False)]:
                out['nav'].append(n); out['start'].append(s); out['goal'].append(g); out['layout'].append(int(li)); out['split'].append(split)
    L = {k: np.stack(v) if k in ('nav', 'start', 'goal') else np.array(v) for k, v in out.items()}
    np.savez_compressed(f, **L); return L


# ---------------------------------------------------------------- continuous expert, per layout
_DF, _CE = {}, {}
MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1))


def dist_field(nav, g):
    key = (nav.tobytes(), int(g[0]), int(g[1]))
    if key not in _DF:
        H, W = nav.shape; d = -np.ones((H, W), np.int64); d[g[1], g[0]] = 0; q = deque([(int(g[0]), int(g[1]))])
        while q:
            x, y = q.popleft()
            for dx, dy in MOVES:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and nav[ny, nx] and d[ny, nx] < 0:
                    d[ny, nx] = d[y, x] + 1; q.append((nx, ny))
        _DF[key] = d
    return _DF[key]


def expert_action(nav, p, goal):
    """(action (2,), stop flag) of the continuous expert at point p (see the module docstring); cached per (layout, point, goal)."""
    nav = np.asarray(nav, bool); p = np.asarray(p, np.float32); goal = np.asarray(goal, np.float32)
    key = (nav.tobytes(), p.tobytes(), goal.tobytes())
    if key not in _CE:
        g = np.rint(goal).astype(int); c = np.rint(p).astype(int); d = dist_field(nav, g); H, W = nav.shape; cells = [c.copy()]
        assert d[c[1], c[0]] >= 0, 'point not connected to the goal'
        while d[c[1], c[0]] > 0:
            for dx, dy in MOVES:
                nx, ny = c[0] + dx, c[1] + dy
                if 0 <= nx < W and 0 <= ny < H and d[ny, nx] == d[c[1], c[0]] - 1:
                    c = np.array([nx, ny]); break
            cells.append(c.copy())
        p64 = p.astype(np.float64)
        nxt = goal.astype(np.float64) if len(cells) == 1 else first_waypoint(nav, p64, cells)
        seg = nxt - p64; n = max(1, int(np.ceil(np.linalg.norm(seg) - 1e-9))); a = (seg / n).astype(np.float32)
        _CE[key] = (a, bool(np.linalg.norm(p64 + a - goal) <= STOP_R + 1e-6))
    return _CE[key]


def expert_rollout(nav, start, goal):
    free1 = torch.as_tensor(np.asarray(nav, bool))[None]; p = np.asarray(start, np.float32); g = np.asarray(goal, np.float32); out = [p]
    while (np.linalg.norm(p - g) > STOP_R or len(out) == 1) and len(out) <= 200:
        a, _ = expert_action(nav, p, g)
        p = STEP(free1, torch.as_tensor(p)[None], torch.as_tensor(a)[None])[0].numpy().astype(np.float32); out.append(p)
    assert np.linalg.norm(p - g) <= STOP_R, 'continuous expert did not reach the goal'
    return np.stack(out)


def expert_labels(nav, goal, pts):
    lab = [expert_action(nav, q, goal) for q in np.asarray(pts, np.float32)]
    return np.stack([a for a, _ in lab]).astype(np.float32), np.array([s for _, s in lab])


# ---------------------------------------------------------------- rollouts and loss (per-row mazes)
@torch.no_grad()
def chain(model, nav_t, S, G, attempts, window, chunk, device):
    """`attempts` consecutive attempts per row (row r in maze nav_t[r]); attempt a sees the row's last min(window, a) attempts."""
    N = len(S); mem = [[] for _ in range(N)]; succ = np.zeros((attempts, N), bool); steps = np.zeros((attempts, N), int)
    St = torch.as_tensor(S, device=device); Gt = torch.as_tensor(G, device=device)
    for a in range(attempts):
        for j in range(0, N, chunk):
            rows = range(j, min(N, j + chunk)); free = nav_t[j:j + chunk]
            ctx = build_context(model, free, Gt[j:j + chunk], [mem[r][len(mem[r]) - min(window, len(mem[r])):] for r in rows])
            t, o, n = rollout(model, free, St[j:j + chunk], Gt[j:j + chunk], ctx)
            for r, tr in zip(rows, t):
                mem[r].append(tr)
            succ[a, j:j + chunk] = o; steps[a, j:j + chunk] = n
    return mem, succ, steps


def loss_relabel(model, free, goal, ctxs, labs, navs):
    """rounds.loss_relabel with a maze per row: the student's last attempt labs[b] (after its attempts ctxs[b], unsupervised) is scored at
    every point on the continuous expert's move and stop flag. Returns (loss, labelled tokens)."""
    B = len(labs); dev = goal.device; was = model.training; model.eval()
    with torch.no_grad():
        S = build_context(model, free, goal, [list(c) + [l] for c, l in zip(ctxs, labs)])
    model.train(was)
    T = np.array([len(l) - 1 for l in labs]); Tm = int(T.max()); g = goal.cpu().numpy()
    act = np.zeros((B, Tm, 2), np.float32); stop = np.zeros((B, Tm), np.float32); m = np.zeros((B, Tm), np.float32)
    for b, l in enumerate(labs):
        a, st = expert_labels(navs[b], g[b], np.asarray(l)[:T[b]]); act[b, :T[b]] = a; stop[b, :T[b]] = st; m[b, :T[b]] = 1
    idx = S['att_starts'][-1][:, None] + torch.arange(Tm, device=dev)[None]
    assert bool((S['att_starts'][-1] + torch.as_tensor(T, device=dev) == S['len']).all())
    idx = torch.minimum(idx, S['len'][:, None] - 1)
    h = model._hidden({**{k: S[k] for k in TOKEN_KEYS}, 'len': S['len']}); h = torch.gather(h, 1, _expand_idx(idx, h))
    dist, stop_logit = model._heads(h)
    na = model.normalizer['action'].normalize(torch.as_tensor(act, device=dev)); mm = torch.as_tensor(m, device=dev)
    nll = -(dist.log_prob(na).sum(-1) * mm).sum() / mm.sum()
    bce = (F.binary_cross_entropy_with_logits(stop_logit, torch.as_tensor(stop, device=dev), reduction='none') * mm).sum() / mm.sum()
    return (nll if rounds.NO_STOP else nll + bce), int(m.sum())


def batch_loss(model, L, tr, nav_all, items, rows, micro, device):
    """Backward of the token-weighted loss of one batch; rows with different context lengths go in separate groups."""
    by = {}
    for r in rows:
        by.setdefault(len(items[r][1]), []).append(r)
    denom = sum(len(items[r][2]) - 1 for r in rows); tot = 0.
    for _, rs in sorted(by.items()):
        for j in range(0, len(rs), micro):
            sub = rs[j:j + micro]; lv = tr[[items[r][0] for r in sub]]
            loss, ntok = loss_relabel(model, nav_all[lv], torch.as_tensor(L['goal'][lv], device=device), [items[r][1] for r in sub],
                                      [items[r][2] for r in sub], [L['nav'][v] for v in lv])
            assert torch.isfinite(loss)
            w = ntok / denom; (loss * w).backward(); tot += loss.item() * w
    return tot


# ---------------------------------------------------------------- train / eval
def setup(args, device):
    L = levels(); tr = np.where(L['split'] == 0)[0]; te = np.where(L['split'] == 1)[0]
    if getattr(args, 'max_levels', 0):
        tr, te = tr[:args.max_levels], te[:args.max_levels]
    model, pc = make_policy(args.rounds, device); ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer()); model.to(device)
    return L, tr, te, model, pc, torch.as_tensor(L['nav'], device=device)


def train(args):
    device = 'cuda'; torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    out = RUNS / args.tag; out.mkdir(parents=True, exist_ok=True); noctx = bool(args.no_context)
    L, tr, te, model, pc, nav_all = setup(args, device); N = len(tr)
    S0, G0 = L['start'][tr], L['goal'][tr]
    tsub = te[::max(1, len(te) // 250)]; trsub = tr[::max(1, len(tr) // 250)]
    config = dict(vars(args), policy=pc, setting='procgen_multi', actions='continuous', obs=rounds.OBS_MODE, obs_t=rounds.OBS_T,
                  no_context=noctx, fb=args.fb, train_levels=int(N), test_levels=int(len(te)),
                  levels='800 procgen train layouts x 4 goals / 200 held-out layouts x 5 goals; start = bottom-left free cell',
                  feedback_leaks_goal_distance=bool(rounds.FB_MODE in ('bfs', 'bfs_dir', 'dir')),
                  design='DAgger (continuous expert labels the student\'s last attempt), aggregated data, per-row mazes')
    state_f = out / 'state.pt'; items, first = [], 0
    if state_f.exists():
        st = torch.load(state_f, map_location='cpu', weights_only=False)
        model.load_state_dict(st['model']); model.to(device); items, first = st['items'], st['round'] + 1
        torch.set_rng_state(st['rng']); np.random.set_state(st['np_rng']); print('resumed after round', st['round'], flush=True)
    else:
        (out / 'config.json').write_text(json.dumps(config, indent=1))
    log = (out / 'log.jsonl').open('a')
    for n in range(first, args.rounds):
        t0 = time.time(); rec = dict(round=n); k = 0 if noctx else n
        if n == 0:
            new = [(i, [], expert_rollout(L['nav'][tr[i]], S0[i], G0[i])) for i in range(N)]
        else:
            model.eval(); mem, succ, steps = chain(model, nav_all[tr], S0, G0, k + 1, k, args.chunk, device)
            new = [(i, mem[i][:k], mem[i][k]) for i in range(N)]
            rec.update(collect_success_by_attempt=[round(float(v), 4) for v in succ.mean(1)],
                       collect_mean_steps=[round(float(v), 2) for v in steps.mean(1)])
        items += new; rec.update(items=len(items), collect_seconds=round(time.time() - t0))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.95, .999), eps=1e-8, weight_decay=1e-6)
        rng = np.random.default_rng(args.seed * 1000 + n); order = []; window = []; t1 = time.time()
        U = args.iters * args.steps_per_iter
        for u in range(1, U + 1):
            if (u - 1) % args.steps_per_iter == 0:
                while len(order) < args.batch:
                    order += list(rng.permutation(len(items)))
                rows, order = order[:args.batch], order[args.batch:]
            opt.zero_grad(); window.append(batch_loss(model, L, tr, nav_all, items, rows, args.micro, device))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            if u % args.steps_per_iter == 0:
                model.eval(); r_t = {}
                for name, lv in (('test', tsub), ('train', trsub)):
                    _, s_, _ = chain(model, nav_all[lv], L['start'][lv], L['goal'][lv], k + 1, k, args.chunk, device)
                    r_t[f'{name}_by_attempt'] = [round(float(v), 4) for v in s_.mean(1)]
                model.train()
                log.write(json.dumps(dict(round=n, iter=u // args.steps_per_iter, update=u, loss=float(np.mean(window)), **r_t)) + '\n'); log.flush(); window = []
        rec.update(train_seconds=round(time.time() - t1))
        torch.save(dict(model=model.state_dict(), round=n, config=config), out / f'round_{n}.pt')
        torch.save(dict(model=model.state_dict(), round=n, items=items, rng=torch.get_rng_state(), np_rng=np.random.get_state()), state_f)
        log.write(json.dumps(rec) + '\n'); log.flush(); print('ROUND', json.dumps(rec), flush=True)
    (out / 'TRAINED').write_text('ok\n')


@torch.no_grad()
def eval_ctx(args):
    """Rounds 0 and the last make --attempts attempts in a row (attempt a sees its a-1 earlier attempts; no-context runs: none) on a fixed
    subset of train levels (every 5th: 640) and all 1000 test levels (held-out mazes), --seeds seeds."""
    device = 'cuda'; out = RUNS / args.tag; cfg = json.loads((out / 'config.json').read_text())
    set_fb(cfg['fb']); set_obs(cfg.get('obs', 'none'), cfg.get('obs_t', 400)); rounds.NO_STOP = bool(cfg.get('no_stop')); args.rounds = cfg['rounds']; args.max_levels = cfg.get('max_levels', 0)
    L, tr, te, model, pc, nav_all = setup(args, device); model.eval(); R = cfg['rounds']; W = 0 if cfg['no_context'] else R - 1
    evr = sorted({int(x) % R for x in args.eval_rounds.split(',')}); res = dict(rounds=np.array(evr), window=W); torch.manual_seed(20261122)
    for split, lv in (('train', tr[::5]), ('test', te)):
        lvr = np.repeat(lv, args.seeds); succ = np.zeros((len(evr), args.attempts, len(lvr)), bool)
        for j, m in enumerate(evr):
            model.load_state_dict(torch.load(out / f'round_{m}.pt', map_location='cpu', weights_only=False)['model']); model.to(device).eval()
            _, succ[j], _ = chain(model, nav_all[lvr], L['start'][lvr], L['goal'][lvr], args.attempts, W, args.chunk, device)
            print(split, 'round', m, 'success by attempt', succ[j].mean(1).round(3).tolist(), flush=True)
        res[f'{split}_success'] = succ.reshape(len(evr), args.attempts, len(lv), args.seeds); res[f'{split}_levels'] = lv
    np.savez_compressed(out / 'eval_ctx.npz', **res); print('saved', out / 'eval_ctx.npz')


def smoke(args):
    import shutil
    L = levels(); trl, tel = set(L['layout'][L['split'] == 0].tolist()), set(L['layout'][L['split'] == 1].tolist())
    assert not (trl & tel) and len(trl) == 800 and len(tel) == 200 and (L['split'] == 0).sum() == 3200 and (L['split'] == 1).sum() == 1000
    assert all(L['nav'][i][int(L['goal'][i][1]), int(L['goal'][i][0])] and L['nav'][i][int(L['start'][i][1]), int(L['start'][i][0])] for i in range(len(L['split'])))
    rng = np.random.default_rng(0); pick = rng.choice(len(L['split']), 300, replace=False); t = time.time()
    lens = [len(expert_rollout(L['nav'][i], L['start'][i], L['goal'][i])) - 1 for i in pick]
    print(f'levels: 800 / 200 disjoint layouts, 3200 / 1000 levels; expert reaches the goal on 300 random levels (steps mean {np.mean(lens):.1f}, '
          f'max {max(lens)}, {time.time() - t:.0f}s)')
    a = argparse.Namespace(**{**vars(args), 'tag': 'pg_smoke', 'rounds': 3, 'iters': 2, 'steps_per_iter': 2, 'batch': 32, 'micro': 16,
                              'max_levels': 64, 'eval_rounds': '0,-1', 'attempts': 3, 'seeds': 1})
    shutil.rmtree(RUNS / 'pg_smoke', ignore_errors=True); train(a); eval_ctx(a)
    st = torch.load(RUNS / 'pg_smoke' / 'state.pt', map_location='cpu', weights_only=False); ns = sorted({len(it[1]) for it in st['items']})
    print('context lengths in the items:', ns); assert ns == ([0] if a.no_context else [0, 1, 2])
    print(open(RUNS / 'pg_smoke' / 'log.jsonl').read()[-600:]); print('PG_SMOKE_OK')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['smoke', 'train', 'evalctx', 'levels'])
    ap.add_argument('--tag', default='pg_dag'); ap.add_argument('--rounds', type=int, default=8); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch', type=int, default=128); ap.add_argument('--micro', type=int, default=32); ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--iters', type=int, default=30); ap.add_argument('--steps-per-iter', type=int, default=30); ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--fb', default='reward', choices=['reward', 'bfs', 'none']); ap.add_argument('--obs', default='window', choices=['none', 'window', 'ddpm'])
    ap.add_argument('--obs-t', type=int, default=400); ap.add_argument('--no-context', action='store_true')
    ap.add_argument('--no-stop', action='store_true', help='stop head not trained, never ends an attempt (rounds.NO_STOP)')
    ap.add_argument('--max-levels', type=int, default=0, help='smoke: use only the first N train / test levels')
    ap.add_argument('--eval-rounds', default='0,-1'); ap.add_argument('--attempts', type=int, default=8); ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args(); set_fb(a.fb); set_obs(a.obs, a.obs_t); rounds.NO_STOP = a.no_stop
    {'smoke': smoke, 'train': train, 'evalctx': eval_ctx, 'levels': lambda _: print({k: v.shape for k, v in levels().items()})}[a.mode](a)
