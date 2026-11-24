"""Action error to the expert vs candidates in context, and how attempts end (2026-11-23).
For each run (final round): the policy makes A attempts in a row per (start, goal) x seeds exactly as eval_ctx (attempt k sees its last
min(k, W) attempts; no-context runs: none). Then, for every attempt k, the stream (context + attempt k) is rebuilt as in the DAgger
loss and the policy's MEAN action at every visited point p_t is compared with the expert's move from p_t (rounds / pg expert labels):
  mse  = mean over the attempt's points of ||mean_action(p_t) - expert_move(p_t)||^2   (raw action units, |a| <= 1)
  stop = fraction of points where the stop head (sigmoid > .5) agrees with the expert's stop flag
  end  = 0 success (came within 1 of the goal) | 1 stopped elsewhere (stop head) | 2 ran out of steps (budget = free cells)
Splits: train and test (held out: held-out goals / quadrant / mazes), --pairs random pairs (levels) each, --seeds seeds.
-> runs/<tag>/mse_eval.npz  {split}_{mse,stop,end,steps,success} (A, pairs*seeds), {split}_exp_steps (pairs*seeds)
  one maze:  python mse_eval.py --tags c_dag_window_fbR,c_dag_window_fbB,...   (all tags of ONE setting; MAZE_SETTING from the runs)
  new mazes: python mse_eval.py --pg --tags pg_dag_window_fbR,...
"""
import argparse, json, os, sys
from multiprocessing import Pool
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--tags', required=True); ap.add_argument('--pg', action='store_true')
ap.add_argument('--pairs', type=int, default=100); ap.add_argument('--seeds', type=int, default=2); ap.add_argument('--chunk', type=int, default=256)
ap.add_argument('--force', action='store_true'); args = ap.parse_args(); TAGS = args.tags.split(',')
if not args.pg:
    import run_setting
    S0 = run_setting.resolve(TAGS[0]); assert all(run_setting.infer(t) == S0 for t in TAGS), 'tags span several settings'
    if S0 in ('interp30', 'interp30c'):
        os.environ['INTERP_TEST'] = 'all'
    import rounds, asteroid
    from rounds import make_policy, load_data, train_starts, test_episodes, build_context, TOKEN_KEYS, _expand_idx
    ROOT = rounds.HERE / 'runs'
else:
    import pg_dagger as P, rounds
    from rounds import build_context, TOKEN_KEYS, _expand_idx
    ROOT = P.RUNS
import torch

NCPU = int(os.environ.get('SLURM_CPUS_PER_TASK', 1)); dev = 'cuda'; torch.set_num_threads(NCPU)


def _label(job):
    free_np, goal, pts = job
    a, st = (P.expert_labels(free_np, goal, pts) if args.pg else rounds.expert_labels(free_np, goal, pts))
    return a, st


@torch.no_grad()
def mean_actions(model, free, goal, ctxs, labs):
    """Mean action and stop prob at every token of the last attempt (rows with >= 1 step), stream rebuilt as in loss_relabel."""
    S = build_context(model, free, goal, [list(c) + [l] for c, l in zip(ctxs, labs)])
    T = np.array([len(l) - 1 for l in labs]); Tm = int(T.max())
    idx = torch.minimum(S['att_starts'][-1][:, None] + torch.arange(Tm, device=dev)[None], S['len'][:, None] - 1)
    h = model._hidden({**{k: S[k] for k in TOKEN_KEYS}, 'len': S['len']}); h = torch.gather(h, 1, _expand_idx(idx, h))
    dist, stop_logit = model._heads(h)
    return model.normalizer['action'].unnormalize(dist.mean).cpu().numpy(), torch.sigmoid(stop_logit).cpu().numpy(), T


def evaluate(model, frees, free_rows, S, G, A, W, guard):
    """frees: per-row free maps (numpy) for labels; free_rows: (N, H, W) tensor for rollouts. Returns dict of (A, N) arrays."""
    N = len(S); mem = [[] for _ in range(N)]; succ = np.zeros((A, N), bool); steps = np.zeros((A, N), int)
    St = torch.as_tensor(S, device=dev); Gt = torch.as_tensor(G, device=dev)
    for a in range(A):
        for j in range(0, N, args.chunk):
            rows = range(j, min(N, j + args.chunk))
            ctx = build_context(model, free_rows[j:j + args.chunk], Gt[j:j + args.chunk], [mem[r][len(mem[r]) - min(W, len(mem[r])):] for r in rows])
            t, o, n = rounds.rollout(model, free_rows[j:j + args.chunk], St[j:j + args.chunk], Gt[j:j + args.chunk], ctx)
            for r, tr in zip(rows, t):
                mem[r].append(tr)
            succ[a, j:j + args.chunk] = o; steps[a, j:j + args.chunk] = n
    jobs = [(frees[r], G[r], np.asarray(mem[r][a])[:len(mem[r][a]) - 1]) for a in range(A) for r in range(N)]
    keep = [i for i, jb in enumerate(jobs) if len(jb[2])]
    with Pool(NCPU) as pool:
        labs = dict(zip(keep, pool.map(_label, [jobs[i] for i in keep], chunksize=8)))
    mse = np.full((A, N), np.nan); stop = np.full((A, N), np.nan)
    for a in range(A):
        rows = [r for r in range(N) if a * N + r in labs]
        for j in range(0, len(rows), args.chunk):
            sub = rows[j:j + args.chunk]
            ctxs = [mem[r][max(0, a - W):a] if W else [] for r in sub]
            pred, ps, T = mean_actions(model, free_rows[sub], Gt[sub], ctxs, [mem[r][a] for r in sub])
            for b, r in enumerate(sub):
                ea, es = labs[a * N + r]
                mse[a, r] = float(((pred[b, :T[b]] - ea) ** 2).sum(-1).mean()); stop[a, r] = float(((ps[b, :T[b]] > .5) == es).mean())
    end = np.where(succ, 0, np.where(steps >= guard, 2, 1))
    return dict(mse=mse, stop=stop, end=end, steps=steps, success=succ)


def main():
    rng = np.random.default_rng(0)
    if args.pg:
        L = P.levels(); tr = np.where(L['split'] == 0)[0]; te = np.where(L['split'] == 1)[0]
        splits = {s: np.sort(rng.choice(v, min(args.pairs, len(v)), replace=False)) for s, v in (('train', tr), ('test', te))}
        nav_all = torch.as_tensor(L['nav'], device=dev)
    for tag in TAGS:
        out = ROOT / tag; f = out / 'mse_eval.npz'
        if f.exists() and not args.force:
            print('exists', f); continue
        cfg = json.loads((out / 'config.json').read_text()); R = cfg['rounds']; A = min(8, R)
        res = dict(attempts=A)
        if args.pg:
            P.set_fb(cfg['fb']); P.set_obs(cfg.get('obs', 'none'), cfg.get('obs_t', 400)); noctx = bool(cfg['no_context']); rounds.NO_STOP = bool(cfg.get('no_stop'))
            model, pc = P.make_policy(R, dev); ds = P.load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer())
        else:
            noctx = asteroid.apply_cfg(cfg); model, pc = make_policy(R, dev); ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer())
            tr_s, tr_g, free_np = train_starts(ds); te = test_episodes()
            te_s = np.stack([e['start'] for e in te]).astype(np.float32); te_g = np.stack([e['goal'] for e in te]).astype(np.float32)
            pick = {s: np.sort(np.random.default_rng(0).choice(len(v), min(args.pairs, len(v)), replace=False)) for s, v in (('train', tr_s), ('test', te_s))}
        model.load_state_dict(torch.load(out / f'round_{R - 1}.pt', map_location='cpu', weights_only=False)['model']); model.to(dev).eval()
        W = 0 if noctx else asteroid.eval_window(cfg) if not args.pg else R - 1; torch.manual_seed(20261123)
        if cfg.get('max_ctx', -1) is not None and cfg.get('max_ctx', -1) >= 0:
            A = res['attempts'] = min(A, cfg['max_ctx'] + 1)
        for split in ('train', 'test'):
            if args.pg:
                lv = np.repeat(splits[split], args.seeds); frees = [L['nav'][i].astype(bool) for i in lv]; fr = nav_all[lv]
                S, G = L['start'][lv], L['goal'][lv]; guard = L['nav'][lv].reshape(len(lv), -1).sum(1)
                res[f'{split}_levels'] = splits[split]
            else:
                idx = np.repeat(pick[split], args.seeds); S = (tr_s if split == 'train' else te_s)[idx]; G = (tr_g if split == 'train' else te_g)[idx]
                frees = [free_np] * len(idx); fr = torch.as_tensor(free_np, device=dev)[None].repeat(len(idx), 1, 1); guard = int(free_np.sum())
                rounds.expert_tables(free_np); res[f'{split}_pairs'] = pick[split]
            ER = P.expert_rollout if args.pg else rounds.expert_rollout   # expert path length per row: success within k x it (no-stop runs)
            res[f'{split}_exp_steps'] = np.array([len(ER(f_, s_, g_)) - 1 for f_, s_, g_ in zip(frees, S, G)])
            r = evaluate(model, frees, fr, S, G, A, W, guard)
            for k, v in r.items():
                res[f'{split}_{k}'] = v
            print(tag, split, 'success', r['success'].mean(1).round(3).tolist(), 'mse', np.nanmean(r['mse'], 1).round(4).tolist(),
                  'end(succ/stop/budget)', [np.round([(r['end'][a] == e).mean() for e in range(3)], 2).tolist() for a in (0, A - 1)], flush=True)
        np.savez_compressed(f, **res); print('saved', f, flush=True)


if __name__ == '__main__':
    main()
