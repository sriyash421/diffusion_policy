"""Best-of-n with the verifier, for BC and for STk0..STk7 of one train_asteroid run.

Candidates per episode (start, goal), n = 1..A:
  BC   n iid samples of the UNet (open-loop H-step chunk each).
  STk  the round-k weights make A attempts in a row from the start; attempt j has attempts 1..j-1 in context (window R-1),
       so the first n attempts are the n candidates.
Verifier = -BFS(final cell of the candidate) (true maze distance to the goal, the same quantity as the feedback; no noise).
Best-of-n executes the candidate with the highest verifier value (first one on ties) and scores its success.
Saves runs/<tag>/bon_seeds<seeds>.npz: {split}_st_success / _st_bfs (R, A, starts, seeds), {split}_bc_success / _bc_bfs
(starts, seeds, A).
  python bon_eval.py --tag ast_s30 --seeds 8 --attempts 8
INTERP_TEST=all (interp30): every (start, held-out goal) pair of bc_scan.npz -> bon_seeds<seeds>_allpairs.npz; use --no-bc --splits test.
--eval-obs-noise (runs trained with --obs-noise): the ST attempts see the training-time observation noise -- every step's encoded
observation of the attempt being rolled out gets the DDPM noise of its slot (t = obs_noise_t for min(window, attempts in context)),
the attempts in context are rebuilt clean, as in training; saves bon_seeds<seeds>_noisyobs.npz. BC stays clean.
"""
import argparse, json, os
from collections import deque
import numpy as np
import torch
from rounds import HERE, make_policy, load_data, train_starts, BC_RUN, BC_CKPT, SPEC, maze, hydra
from rounds import BatchedRollout, snap_cardinal_torch
from asteroid import chain, test_set, drop_context_feedback, set_fb


def bfs_table(free, goal):
    """(H, W) maze distance of every free cell to goal (x, y); unreachable = H*W."""
    H, W = free.shape; d = np.full((H, W), H * W, int); gx, gy = int(goal[0]), int(goal[1]); d[gy, gx] = 0; q = deque([(gx, gy)])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            u, v = x + dx, y + dy
            if 0 <= u < W and 0 <= v < H and free[v, u] and d[v, u] > d[y, x] + 1:
                d[v, u] = d[y, x] + 1; q.append((u, v))
    return d


def bfs_of(free, goals, cells, cache):
    out = np.empty(len(cells), int)
    for i, (g, c) in enumerate(zip(goals, cells)):
        key = (int(round(g[0])), int(round(g[1])))
        if key not in cache:
            cache[key] = bfs_table(free, key)
        x, y = np.clip(np.round(c).astype(int), 0, [free.shape[1] - 1, free.shape[0] - 1]); out[i] = cache[key][y, x]
    return out


def bc_candidates(step, S0, G0, episodes, n, device, free, cache):
    """(starts, n) success and verifier BFS of n iid BC samples per episode."""
    payload = torch.load(BC_RUN / f'step_{step:07d}.pt', map_location='cpu', weights_only=False); cfg = payload['config']; H = cfg['data']['horizon']
    model = hydra.utils.instantiate(cfg['policy']); model.load_state_dict(payload['model']); model.to(device).eval()
    if episodes is None:
        episodes = []
        for s, g in zip(S0, G0):
            env = maze.MazeEnv(SPEC['layouts'], seed=0, split='train', layout_index=0, goal_split=SPEC['goal_split']); env.reset()
            env.goal = g.astype(np.int32).copy(); episodes.append(dict(env=env, start=s, goal=g))
    succ = np.zeros((len(episodes), n), bool); bfs = np.zeros((len(episodes), n), int)
    for k in range(n):
        roll = BatchedRollout([e['env'] for e in episodes], [e['start'] for e in episodes], [e['goal'] for e in episodes], 1., 1)
        obs = roll.obs_tensor(device); aa = []
        with torch.no_grad():
            for j in range(0, len(episodes), 64):
                pred = model.predict_action({key: v[j:j + 64] for key, v in obs.items()})['action']; aa.append(snap_cardinal_torch(pred).cpu().numpy())
        a = np.concatenate(aa)
        for t in range(H):
            roll.step(a[:, t])
        succ[:, k] = roll.done_step >= 0; bfs[:, k] = bfs_of(free, [e['goal'] for e in episodes], roll.pos, cache)
    return succ, bfs


def noisy_chain(model, free1, S, G, attempts, window, chunk, device, ns):
    """chain() with the training-time observation noise on the attempt being rolled out (see the module docstring)."""
    from rounds import build_context, rollout
    from asteroid import noisy_obs, obs_noise_t
    mem = [[] for _ in range(len(S))]; succ = np.zeros((attempts, len(S)), bool)
    for a in range(attempts):
        t = obs_noise_t(ns, min(window, a)); trajs, ok = [], []
        for j in range(0, len(S), chunk):
            rows = range(j, min(len(S), j + chunk)); free = free1.repeat(len(rows), 1, 1); gl = torch.as_tensor(G[j:j + chunk], device=device)
            ctx = build_context(model, free, gl, [mem[r][len(mem[r]) - min(window, len(mem[r])):] for r in rows])     # clean, as chain()
            with noisy_obs(model, t):
                tr, o, _ = rollout(model, free, torch.as_tensor(S[j:j + chunk], device=device), gl, ctx)
            trajs += tr; ok.append(o)
        succ[a] = np.concatenate(ok)
        for r, tr in enumerate(trajs):
            mem[r].append(tr)
    return mem, succ, None


def main(args):
    import rounds
    if rounds.CONT:   # interp30c: no grid BC (the continuous BC is bc_closed.py)
        args.no_bc = True
    device = 'cuda'; out = HERE / 'runs' / args.tag; cfg = json.loads((out / 'config.json').read_text())
    set_fb(cfg.get('fb'), cfg.get('open_loop'), cfg.get('hide_goal'))
    if cfg.get('no_ctx_feedback'):
        drop_context_feedback()
    R = cfg['rounds']; A = args.attempts; W = R - 1; model, pc = make_policy(R, device)
    RE = R if not args.partial else next(m for m in range(R + 1) if m == R or not (out / f'round_{m}.pt').exists())   # --partial: saved cycles only
    assert RE > 0, 'no saved cycle yet'
    MS = list(range(RE)) if args.only_round < 0 else [args.only_round]; assert all(m < RE for m in MS), (MS, RE)
    assert not args.ckpt or (len(MS) == 1 and f'round_{MS[0]}' in args.ckpt), '--ckpt needs --only-round of the same round'
    print('evaluating STk', MS, flush=True)
    ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer()); model.to(device).eval()
    tr_s, tr_g, free_np = train_starts(ds); free1 = torch.as_tensor(free_np, device=device)[None]; free = free_np.astype(bool)
    te, te_s, te_g = test_set(free_np); cache = {}; res = dict(window=W, attempts=A, seeds=args.seeds); torch.manual_seed(20261105)
    ns = None
    if args.eval_obs_noise:
        assert cfg.get('obs_noise', 'none') != 'none', 'run was trained without observation noise'
        ns = argparse.Namespace(obs_noise=cfg['obs_noise'], obs_noise_t=cfg['obs_noise_t'], obs_noise_ramp=cfg['obs_noise_ramp'], rounds=R)
        res['eval_obs_noise_t'] = [int(__import__('asteroid').obs_noise_t(ns, min(W, a))) for a in range(A)]; print('eval obs noise t by attempt', res['eval_obs_noise_t'])
    fname = f'bon_seeds{args.seeds}' + ('_noisyobs' if ns else '') + ('_partial' if args.partial else '') + args.suffix + {'control': '_control', 'all': '_allpairs'}.get(os.environ.get('INTERP_TEST', ''), '') + '.npz'; res['cycles_evaluated'] = RE; res['rounds_evaluated'] = np.array(MS); res['ckpt'] = (args.ckpt or f'round_{MS[0]}.pt') if len(MS) == 1 else 'round_<k>.pt'
    for split, (S0, G0) in {'test': (te_s, te_g), 'train': (tr_s, tr_g)}.items():   # test first: it is small
        if split not in args.splits.split(','):
            continue
        S = np.repeat(S0, args.seeds, 0); G = np.repeat(G0, args.seeds, 0)
        succ = np.zeros((len(MS), A, len(S)), bool); bfs = np.zeros((len(MS), A, len(S)), int)
        for i, m in enumerate(MS):
            model.load_state_dict(torch.load(out / (args.ckpt if args.ckpt else f'round_{m}.pt'), map_location='cpu', weights_only=False)['model']); model.to(device).eval()
            model._feat_std = None
            with torch.no_grad():
                mem, succ[i], _ = (noisy_chain(model, free1, S, G, A, W, args.chunk, device, ns) if ns else chain(model, free1, S, G, A, W, args.chunk, device))
            for a in range(A):
                bfs[i, a] = bfs_of(free, G, [mem[r][a][-1] for r in range(len(S))], cache)
            print(split, 'ST round', m, 'success by attempt', succ[i].mean(1).round(3).tolist(), flush=True)
        res[f'{split}_st_success'] = succ.reshape(len(MS), A, len(S0), args.seeds); res[f'{split}_st_bfs'] = bfs.reshape(len(MS), A, len(S0), args.seeds)
        if args.no_bc:
            np.savez_compressed(out / fname, **res); continue
        bs, bb = bc_candidates(BC_CKPT[split], S, G, None if split == 'train' else [e for e in te for _ in range(args.seeds)], A, device, free, cache)
        res[f'{split}_bc_success'] = bs.reshape(len(S0), args.seeds, A); res[f'{split}_bc_bfs'] = bb.reshape(len(S0), args.seeds, A)
        print(split, 'BC success per candidate', bs.mean(0).round(3).tolist(), flush=True)
        np.savez_compressed(out / fname, **res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='ast_s30'); ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--attempts', type=int, default=8); ap.add_argument('--chunk', type=int, default=512); ap.add_argument('--splits', default='train,test')
    ap.add_argument('--partial', action='store_true', help='training still running: evaluate the saved cycles only, save *_partial.npz')
    ap.add_argument('--only-round', type=int, default=-1, help='evaluate only STk of this round (axis 0 of the arrays has length 1)')
    ap.add_argument('--ckpt', default='', help='weights file relative to the run dir (e.g. ckpt/round_7_u00600.pt) instead of round_<k>.pt; with --only-round k')
    ap.add_argument('--suffix', default='', help='inserted into the output name, e.g. _n16_r7 -> bon_seeds8_n16_r7[_allpairs].npz')
    ap.add_argument('--no-bc', action='store_true', help='skip the BC candidates (e.g. INTERP_TEST=all, where bc_scan.npz has BC)')
    ap.add_argument('--eval-obs-noise', action='store_true', help='training-time observation noise on the ST attempts (see the docstring)')
    main(ap.parse_args())
