"""ASTEROID-style rounds for the full-trajectory ST policy (11×11, bottom-right quadrant held out).

Same policy, feedback and A* target as rounds.py; what changes follows github.com/WEIRDLabUW/asteroid (train_asteroid.py):
  round 0:  data = the A* demo of every training start, empty context
  round n:  the CURRENT policy (trained through round n-1) rolls out a chain of n trajectories from every training start
            (trajectory j conditioned on its trajectories 1..j-1); the item is (those n trajectories, A* demo)
  data is aggregated over rounds (round n trains on context lengths 0..n)
  every round: fresh AdamW + the maze ST schedule (lr 1e-4, 500-update linear warmup, cosine to 0), --updates of batch 64,
               grad clip 1; the weights carry over (ASTEROID resets optimizer + schedule each iteration too)
Loss: NLL of the A* actions + stop BCE on the demo only, context tokens are not supervised (as ASTEROID's last-episode mask).

Eval (as ASTEROID's: one policy, its own context): round m's policy makes --attempts consecutive attempts from each start, each
conditioned on all its earlier attempts (up to R-1, what the stream holds; --window to cap). Round m trained with at most m.
During training (--test-every updates, and at update 0) the test-start success of attempt n+1 with n own attempts is logged.
Round m's score = its attempt m+1 (exactly m of its own trajectories in context, as in training).

  python asteroid.py smoke
  python asteroid.py train --tag <tag> [--rounds 8 --updates 10000 --batch 64 --micro 32 --lr 1e-4 --warmup 500]
  python asteroid.py eval  --tag <tag> [--seeds 8 --attempts 8]
  python asteroid.py plot  --tag <tag> [--seeds 8]
"""
import argparse, json, sys, time
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] in ('eval', 'plot', 'evalctx'):
    import run_setting; run_setting.resolve(run_setting.tag_from_argv(sys.argv))   # evaluate in the setting the run was trained in
import numpy as np
import torch
from rounds import (HERE, make_policy, load_data, train_starts, test_episodes, build_context, slice_ctx, rollout_all,
                    loss_with_context, bc_eval, BC_CKPT, get_collate_fn, dict_apply, loss_relabel, expert_rollout)


def set_fb(mode, open_loop=False, hide_goal=False):
    """Feedback channels for train / eval / collection (rounds.FB_MODE); eval scripts call it with config['fb'] and
    config['open_loop'] (rounds.OBS_OPEN: obs features frozen at the attempt's start cell, feedback still per step)."""
    import rounds
    rounds.FB_MODE = mode or 'bfs'; rounds.OBS_OPEN = bool(open_loop); rounds.HIDE_GOAL = bool(hide_goal)
    assert not (rounds.HIDE_GOAL and rounds.FB_MODE in ('bfs', 'dir', 'bfs_dir')), 'hidden goal: BFS / dx, dy feedback reveals the goal'
    assert rounds.FB_MODE != 'reward' or rounds.SETTING in ('11br', 'interp30', 'interp30c', '11brc'), 'reward feedback assumes the 11 x 11 maze'


def set_obs(mode='none', t=400):
    """Observation mode of the policy's encoder input (rounds.OBS_MODE / OBS_T; obs_modes.py): none | window | ddpm."""
    import rounds
    rounds.OBS_MODE = mode or 'none'; rounds.OBS_T = int(t if t is not None else 400)
    assert rounds.OBS_MODE in ('none', 'window', 'ddpm'), rounds.OBS_MODE
    assert not (rounds.OBS_MODE != 'none' and rounds.HIDE_GOAL), 'obs modes already hide goal_pos; do not combine with --hide-goal'


def apply_cfg(cfg):
    """Re-apply a run's feedback / observation settings (eval scripts); returns True for a no-context run."""
    set_fb(cfg.get('fb'), cfg.get('open_loop'), cfg.get('hide_goal')); set_obs(cfg.get('obs', 'none'), cfg.get('obs_t', 400))
    if cfg.get('no_ctx_feedback'):
        drop_context_feedback()
    return bool(cfg.get('no_context'))


def drop_context_feedback():
    """--no-ctx-feedback: zero every feedback channel of the earlier attempts in context (their per-step -BFS and the end-of-attempt
    [-BFS, -BFS, dx, dy] on the next attempt's first token). The cells (positions, moves) stay; the current attempt keeps its own
    per-step -BFS, as in the no-context policy. Applies to training, data collection and eval (all go through build_context)."""
    import rounds
    if getattr(rounds.build_context, 'no_fb', False):
        return
    orig = rounds.build_context
    def build(*a, **k):
        ctx = orig(*a, **k)
        if ctx is not None:
            ctx['fb'] = torch.zeros_like(ctx['fb']); ctx['last_fb'] = torch.zeros_like(ctx['last_fb'])
        return ctx
    build.no_fb = True; rounds.build_context = build; globals()['build_context'] = build


_ABAR = None


def obs_noise_t(args, n):
    """DDPM timestep of the observation noise for an item with n own attempts in context (slot n = its (n+1)-th prediction):
    flat = --obs-noise-t for every slot; ramp = linear from --obs-noise-ramp[0] at slot 0 to [1] at slot rounds-1. None = off."""
    mode = getattr(args, 'obs_noise', 'none')
    if mode == 'flat':
        return int(args.obs_noise_t)
    if mode == 'ramp':
        t0, t1 = [int(v) for v in str(args.obs_noise_ramp).replace(':', ',').split(',')]
        return int(round(t0 + (t1 - t0) * n / max(1, args.rounds - 1)))
    return None


class noisy_obs:
    """Training only (the slot observation noise of the pushT search policy, maze version): inside this scope the policy's
    encoded observation features x of the predicted attempt (the A* demo) become sqrt(abar_t) x + sqrt(1 - abar_t) eps * std,
    with abar from DDPM (T = 1000, beta 1e-4 -> 0.02 linear) and std an EMA (0.99) of the per-dimension feature std. Context
    attempts (built before the scope) and all rollouts / evals stay clean."""
    def __init__(self, model, t):
        self.model, self.t = model, t

    def __enter__(self):
        global _ABAR
        if self.t is None:
            return
        if _ABAR is None:
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
            _ABAR = DDPMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule='linear').alphas_cumprod
        m = self.model; orig = type(m)._obs_features; ab = float(_ABAR[self.t])
        def f(obs):
            x = orig(m, obs)
            with torch.no_grad():
                sd = x.reshape(-1, x.shape[-1]).std(0).clamp_min(1e-6)
                m._feat_std = sd if getattr(m, '_feat_std', None) is None else .99 * m._feat_std + .01 * sd
            return ab ** .5 * x + (1 - ab) ** .5 * torch.randn_like(x) * m._feat_std
        m.__dict__['_obs_features'] = f

    def __exit__(self, *exc):
        self.model.__dict__.pop('_obs_features', None)


@torch.no_grad()
def chain(model, free1, S, G, attempts, window, chunk, device):
    """`attempts` consecutive attempts per row; attempt a sees the row's last min(window, a) attempts. Returns the trajectories
    (row -> list), success (attempts, rows) and steps (attempts, rows)."""
    mem = [[] for _ in range(len(S))]; succ = np.zeros((attempts, len(S)), bool); steps = np.zeros((attempts, len(S)), int)
    for a in range(attempts):
        ctx = [m[len(m) - min(window, len(m)):] for m in mem]
        trajs, succ[a], steps[a] = rollout_all(model, free1, S, G, ctx, chunk, device)
        for r, t in enumerate(trajs):
            mem[r].append(t)
    return mem, succ, steps


def test_set(free_np):
    te = test_episodes(); te_s = np.stack([e['start'] for e in te]).astype(np.float32); te_g = np.stack([e['goal'] for e in te]).astype(np.float32)
    assert np.array_equal(te[0]['nav'].astype(bool), free_np), 'test layout differs from the training layout'
    return te, te_s, te_g


def test_point(model, free1, te_s, te_g, n, seeds, chunk, device):
    """Test-start success of the current weights: n own attempts in context, then attempt n+1 (as round n trains)."""
    was = model.training; model.eval()
    _, succ, _ = chain(model, free1, np.repeat(te_s, seeds, 0), np.repeat(te_g, seeds, 0), n + 1, n, chunk, device)
    model.train(was); return [round(float(v), 4) for v in succ.mean(1)]


def lr_at(u, args):
    if args.steps_per_iter > 0:          # rounds-style schedule: constant lr
        return args.lr
    return args.lr * (u / args.warmup if u <= args.warmup else .5 * (1 + np.cos(np.pi * (u - args.warmup) / (args.updates - args.warmup))))


def batch_loss(model, ds, collate, free1, G0, items, rows, micro, device, args=None):
    """Backward of the token-weighted loss of one batch; rows with different context lengths go in separate groups.
    Items (start, context) are supervised on the start's A* demo (loss_with_context); items (start, context, attempt) on the BFS
    expert's labels at the cells of that learner attempt (--target relabel, loss_relabel)."""
    rel = [r for r in rows if len(items[r]) > 2]; rows = [r for r in rows if len(items[r]) == 2]
    by_n = {}
    for r in rows:
        by_n.setdefault(len(items[r][1]), []).append(r)
    rel_by = {}
    for r in rel:
        rel_by.setdefault(len(items[r][1]), []).append(r)
    demos = {r: ds[items[r][0]] for r in rows}
    parts, tot = [], 0.
    rel_tokens = sum(len(items[r][2]) - 1 for r in rel)
    for n, rs in sorted(by_n.items()):
        b = dict_apply(collate([demos[r] for r in rs]), lambda x: x.to(device))
        model.eval()
        ctx = build_context(model, free1.repeat(len(rs), 1, 1), torch.as_tensor(G0[[items[r][0] for r in rs]], device=device),
                            [items[r][1] for r in rs])
        model.train(); parts.append((b, ctx, len(rs)))
    denom = sum(b['attention_mask'].sum().item() for b, _, _ in parts) + rel_tokens
    for n, rs in sorted(rel_by.items()):
        for j in range(0, len(rs), micro):
            sub = rs[j:j + micro]
            loss, ntok = loss_relabel(model, free1.repeat(len(sub), 1, 1), torch.as_tensor(G0[[items[r][0] for r in sub]], device=device),
                                      [items[r][1] for r in sub], [items[r][2] for r in sub])
            assert torch.isfinite(loss)
            w = ntok / denom; (loss * w).backward(); tot += loss.item() * w
    for b, ctx, B in parts:
        n = 0 if ctx is None else len(ctx['att_starts'])
        for j in range(0, B, micro):
            sl = slice(j, j + micro); part = dict_apply(b, lambda x: x[sl])
            with noisy_obs(model, obs_noise_t(args, n) if args is not None else None):
                loss = loss_with_context(model, part, slice_ctx(ctx, sl, B))
            assert torch.isfinite(loss)
            w = part['attention_mask'].sum().item() / denom; (loss * w).backward(); tot += loss.item() * w
    return tot


def train(args):
    device = 'cuda'; torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(4)
    out = HERE / 'runs' / args.tag; out.mkdir(parents=True, exist_ok=True)
    if getattr(args, 'no_ctx_feedback', False):
        drop_context_feedback()
    model, pc = make_policy(args.rounds, device)
    ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer()); model.to(device)
    collate = get_collate_fn(); N = len(ds)
    S0, G0, free_np = train_starts(ds); free1 = torch.as_tensor(free_np, device=device)[None]
    _, te_s, te_g = test_set(free_np)
    relabel = getattr(args, 'target', 'demo') == 'relabel'
    if relabel:
        assert getattr(args, 'obs_noise', 'none') == 'none', 'relabel: no training-only feature noise (use --obs ddpm)'
    if args.steps_per_iter > 0:
        args.updates = args.iters * args.steps_per_iter; args.test_every = args.steps_per_iter
    import rounds as _r
    noctx = bool(getattr(args, 'no_context', False))
    config = dict(vars(args), policy=pc, train_starts=N, design='asteroid: current-policy context chains, aggregated data',
                  setting=_r.SETTING, actions=_r.ACTIONS, obs=_r.OBS_MODE, obs_t=_r.OBS_T, no_context=noctx,
                  feedback_leaks_goal_distance=bool(_r.FB_MODE in ('bfs', 'bfs_dir', 'dir') and (_r.OBS_MODE != 'none' or _r.HIDE_GOAL)),
                  obs_desc={'none': 'full observation', 'window': 'ASTEROID 7x7 window around the agent (-1 outside), agent_pos, goal_pos 0',
                            'ddpm': f'full image DDPM-noised at t={_r.OBS_T} (train and test), agent_pos, goal_pos 0'}[_r.OBS_MODE],
                  context_desc=('NO context: every item / rollout starts an empty stream (DAgger: one student attempt, expert-labelled)'
                                if noctx else 'own earlier attempts in context'),
                  feedback=('context: none (zeroed); current attempt: step [-BFS,0,0,0]' if getattr(args, 'no_ctx_feedback', False)
                            else 'none: step and trajectory feedback all zero' if getattr(args, 'fb', 'bfs') == 'none'
                            else 'step [0,0,dx,dy]; trajectory [0,0,dx,dy]; dx, dy = goal - cell' if getattr(args, 'fb', 'bfs') == 'dir'
                            else 'step [-BFS,0,dx,dy]; trajectory [-BFS, -BFS, dx, dy]; dx, dy = goal - cell' if getattr(args, 'fb', 'bfs') == 'bfs_dir'
                            else 'sparse reward: step [0,0,0,0]; trajectory [r,r,0,0], r = attempt ended at the goal' if getattr(args, 'fb', 'bfs') == 'reward'
                            else 'step [-BFS,0,0,0]; trajectory [-BFS, -BFS, dx, dy]'),
                  goal_obs=('hidden: goal plane cleared and goal_pos = 0 in the policy\'s observations (rounds.HIDE_GOAL)'
                            if getattr(args, 'hide_goal', False) else 'visible: goal plane + goal_pos'),
                  observations=('open loop: every token carries the attempt start cell\'s obs features; prev_a and feedback follow the rollout'
                                if getattr(args, 'open_loop', False) else 'closed loop: the current cell\'s obs features'),
                  target_desc=('relabel: ASTEROID, round 0 = BFS expert rollouts; round n = the policy\'s attempt n+1 after its n own attempts, '
                          'labelled at every visited cell by the BFS expert (move + stop); context unsupervised; data aggregated'
                          if getattr(args, 'target', 'demo') == 'relabel' else 'demo: the start\'s A* demo after the context attempts'),
                  obs_noise_slots=[obs_noise_t(args, n) for n in range(args.rounds)],
                  schedule=(f'per round: fresh AdamW, constant lr, {args.iters} iterations × batch {args.batch} × {args.steps_per_iter} steps'
                            if args.steps_per_iter > 0 else f'per round: fresh AdamW, linear warmup {args.warmup}, cosine to 0 at {args.updates}'))
    state_f = out / 'state.pt'; items, first = [], 0
    if state_f.exists():
        st = torch.load(state_f, map_location='cpu', weights_only=False)
        model.load_state_dict(st['model']); model.to(device); items, first = st['items'], st['round'] + 1
        torch.set_rng_state(st['rng']); np.random.set_state(st['np_rng']); print('resumed after round', st['round'], flush=True)
    else:
        (out / 'config.json').write_text(json.dumps(config, indent=1))
    log = (out / 'log.jsonl').open('a')
    for n in range(first, args.rounds):
        t0 = time.time(); rec = dict(round=n)
        if n == 0 and relabel:   # ASTEROID step 0: expert rollouts (the BFS expert's own path from every training start)
            new = [(i, [], expert_rollout(free_np, S0[i], G0[i])) for i in range(N)]
        elif relabel:            # the current policy makes n+1 attempts per start; the expert relabels the last, the first n are context
            model.eval(); new = []; k = 0 if noctx else n   # --no-context: one attempt from an empty stream, expert-labelled
            for rep in range(args.ctx_per_start):
                mem, succ, steps = chain(model, free1, S0, G0, k + 1, k, args.chunk, device)
                new += [(i, mem[i][:k], mem[i][k]) for i in range(N)]
            rec.update(collect_success_by_attempt=[round(float(v), 4) for v in succ.mean(1)],
                       collect_mean_steps=[round(float(v), 2) for v in steps.mean(1)])
        elif n == 0 or noctx:
            new = [(i, []) for i in range(N)]
        else:
            model.eval(); new = []
            for rep in range(args.ctx_per_start):
                mem, succ, steps = chain(model, free1, S0, G0, n, n, args.chunk, device)
                new += [(i, mem[i]) for i in range(N)]
            rec.update(collect_success_by_attempt=[round(float(v), 4) for v in succ.mean(1)],
                       collect_mean_steps=[round(float(v), 2) for v in steps.mean(1)])
        items += new; rec.update(items=len(items), collect_seconds=round(time.time() - t0))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.95, .999), eps=1e-8, weight_decay=1e-6)
        rng = np.random.default_rng(args.seed * 1000 + n); order = []; window = []; t1 = time.time()
        def test_log(u):
            if args.test_every > 0:
                by = test_point(model, free1, te_s, te_g, 0 if noctx else n, args.test_seeds, args.chunk, device)
                rec_t = dict(round=n, update=u, test_success=by[-1], test_by_attempt=by)
                if args.steps_per_iter > 0:
                    rec_t['iter'] = u // args.steps_per_iter
                if args.train_seeds > 0:
                    bt = test_point(model, free1, S0, G0, 0 if noctx else n, args.train_seeds, args.chunk, device)
                    rec_t.update(train_success=bt[-1], train_by_attempt=bt)
                log.write(json.dumps(rec_t) + '\n'); log.flush()
        test_log(0)
        for u in range(1, args.updates + 1):
            if args.steps_per_iter <= 0 or (u - 1) % args.steps_per_iter == 0:     # rounds-style: the same batch for steps_per_iter steps
                while len(order) < args.batch:
                    order += list(rng.permutation(len(items)))
                rows, order = order[:args.batch], order[args.batch:]
            for g in opt.param_groups:
                g['lr'] = lr_at(u, args)
            opt.zero_grad(); window.append(batch_loss(model, ds, collate, free1, G0, items, rows, args.micro, device, args))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            if u % args.log_every == 0 or u == args.updates:
                log.write(json.dumps(dict(round=n, update=u, lr=lr_at(u, args), loss=float(np.mean(window)),
                                          sec_per_update=round((time.time() - t1) / u, 4))) + '\n'); log.flush(); window = []
            if args.test_every > 0 and (u % args.test_every == 0 or u == args.updates):
                test_log(u)
            if getattr(args, 'save_every', 0) > 0 and u % args.save_every == 0 and u < args.updates:   # mid-round weights for checkpoint selection
                (out / 'ckpt').mkdir(exist_ok=True); torch.save(dict(model=model.state_dict(), round=n, update=u), out / 'ckpt' / f'round_{n}_u{u:05d}.pt')
        rec.update(train_seconds=round(time.time() - t1), epochs=round(args.updates * args.batch / len(items), 1))
        torch.save(dict(model=model.state_dict(), round=n, config=config), out / f'round_{n}.pt')
        torch.save(dict(model=model.state_dict(), round=n, items=items, rng=torch.get_rng_state(), np_rng=np.random.get_state()), state_f)
        log.write(json.dumps(rec) + '\n'); log.flush(); print('ROUND', json.dumps(rec), flush=True)
    (out / 'TRAINED').write_text('ok\n')


@torch.no_grad()
def eval_rounds(args):
    device = 'cuda'; out = HERE / 'runs' / args.tag; cfg = json.loads((out / 'config.json').read_text()); noctx = apply_cfg(cfg)
    R = cfg['rounds']; A = args.attempts; W = 0 if noctx else R - 1 if args.window < 0 else args.window; model, pc = make_policy(R, device)
    ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer()); model.to(device).eval()
    tr_s, tr_g, free_np = train_starts(ds); free1 = torch.as_tensor(free_np, device=device)[None]
    te, te_s, te_g = test_set(free_np)
    res = dict(window=W); torch.manual_seed(20261104)
    for split, (S0, G0) in {'train': (tr_s, tr_g), 'test': (te_s, te_g)}.items():
        if split not in args.splits.split(','):
            continue
        S = np.repeat(S0, args.seeds, 0); G = np.repeat(G0, args.seeds, 0)
        succ = np.zeros((R, A, len(S)), bool); steps = np.zeros((R, A, len(S)), int)
        for m in range(R):
            model.load_state_dict(torch.load(out / f'round_{m}.pt', map_location='cpu', weights_only=False)['model']); model.to(device).eval()
            _, succ[m], steps[m] = chain(model, free1, S, G, A, W, args.chunk, device)
            print(split, 'round', m, 'success by attempt', succ[m].mean(1).round(3).tolist(), flush=True)
        res[f'{split}_success'] = succ.reshape(R, A, len(S0), args.seeds); res[f'{split}_steps'] = steps.reshape(R, A, len(S0), args.seeds)
        if split not in BC_CKPT:   # interp30c: no grid BC here
            continue
        res[f'{split}_bc_success'] = bc_eval(BC_CKPT[split], S0, G0, te if split == 'test' else None, args.seeds, device)
        print(split, 'BC success', res[f'{split}_bc_success'].mean().round(3), flush=True)
    np.savez_compressed(out / (args.out or f'eval_seeds{args.seeds}.npz'), **res)


@torch.no_grad()
def eval_ctx(args):
    """Success vs candidates in context (the simple plots): the policy of each round in --eval-rounds (default: round 0 and the last)
    makes --attempts attempts in a row per (start, goal) x seeds; attempt a sees its a-1 earlier attempts (all of them; no-context runs:
    none, so every attempt starts an empty stream). Splits: train (the training starts), train_heldout (100 unseen pairs of TRAIN goals),
    test (held-out goals: --test-set '' = the default test episodes, 'all' = every (start, held-out goal) pair (interp30 settings)).
    -> runs/<tag>/eval_ctx<suffix>.npz with <split>_success (rounds, attempts, pairs, seeds)."""
    import os, rounds
    device = 'cuda'; out = HERE / 'runs' / args.tag; cfg = json.loads((out / 'config.json').read_text()); noctx = apply_cfg(cfg)
    if args.test_set:
        os.environ['INTERP_TEST'] = args.test_set
    R = cfg['rounds']; A = args.attempts; W = 0 if noctx else R - 1
    model, pc = make_policy(R, device); ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer()); model.to(device).eval()
    tr_s, tr_g, free_np = train_starts(ds); free1 = torch.as_tensor(free_np, device=device)[None]
    _, te_s, te_g = test_set(free_np); th = rounds.heldout_train_episodes()
    th_s = np.stack([e['start'] for e in th]).astype(np.float32); th_g = np.stack([e['goal'] for e in th]).astype(np.float32)
    evr = sorted({int(x) % R for x in args.eval_rounds.split(',')}); res = dict(rounds=np.array(evr), window=W, test_set=args.test_set or 'default')
    ts = args.eval_test_seeds or args.seeds; torch.manual_seed(20261122)
    for split, (S0, G0, sd) in {'train': (tr_s, tr_g, args.seeds), 'train_heldout': (th_s, th_g, args.seeds), 'test': (te_s, te_g, ts)}.items():
        if split not in args.splits.split(','):
            continue
        S = np.repeat(S0, sd, 0); G = np.repeat(G0, sd, 0); succ = np.zeros((len(evr), A, len(S)), bool)
        for j, m in enumerate(evr):
            model.load_state_dict(torch.load(out / f'round_{m}.pt', map_location='cpu', weights_only=False)['model']); model.to(device).eval()
            _, succ[j], _ = chain(model, free1, S, G, A, W, args.chunk, device)
            print(split, 'round', m, 'success by attempt', succ[j].mean(1).round(3).tolist(), flush=True)
        res[f'{split}_success'] = succ.reshape(len(evr), A, len(S0), sd)
    np.savez_compressed(out / f'eval_ctx{args.out}.npz', **res); print('saved', out / f'eval_ctx{args.out}.npz')


def plot(args):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    out = HERE / 'runs' / args.tag; z = np.load(out / f'eval_seeds{args.seeds}.npz'); cfg = json.loads((out / 'config.json').read_text())
    R = cfg['rounds']; cols = plt.cm.viridis(np.linspace(0, .95, R)); summary = {}
    names = [('train', 'training start points'), ('test', 'test start points (held-out quadrant)')]
    fig, axs = plt.subplots(2, 2, figsize=(18, 8), gridspec_kw=dict(width_ratios=[4, 1]))
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, (split, name) in enumerate(names):
        full = z[f'{split}_success'].mean(-1); A = full.shape[1]                    # (R, A, starts)
        s = np.stack([full[m, min(m, A - 1)] for m in range(R)]); bc = z[f'{split}_bc_success'].mean(-1) if f'{split}_bc_success' in z.files else np.full(s.shape[1], np.nan); x = np.arange(s.shape[1])
        ax = axs[i, 0]
        for m in range(R):
            ax.plot(x, s[m], color=cols[m], lw=.9, label=f'round {m} ({m} own traj. in context): {s[m].mean():.2f}')
        ax.plot(x, bc, color='#D55E00', lw=1.2, ls='--', label=f'BC (UNet): {bc.mean():.2f}')
        ax.set_xlabel(f'{name}: index'); ax.set_ylabel(f'success over {args.seeds} seeds'); ax.set_ylim(-.03, 1.03); ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=3, loc='lower right')
        ax = axs[i, 1]; ax.plot(range(R), s.mean(1), 'o-', color='#0072B2', label='round policy'); ax.axhline(bc.mean(), color='#D55E00', ls='--', label='BC (UNet)')
        ax.set_xlabel('round n'); ax.set_ylabel('mean success'); ax.set_ylim(-.03, 1.03); ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_title(name, fontsize=9)
        ax = axs2[i]
        for m in range(R):
            ax.plot(np.arange(1, A + 1), full[m].mean(1), 'o-', ms=3, color=cols[m], label=f'round {m}')
        ax.axhline(bc.mean(), color='#D55E00', ls='--', label='BC (UNet)')
        ax.set_xlabel('attempt (own earlier attempts in context, at most n)'); ax.set_ylabel('mean success'); ax.set_ylim(-.03, 1.03)
        ax.grid(alpha=.3); ax.set_title(name, fontsize=9); ax.legend(fontsize=7, ncol=2)
        summary[split] = dict(rounds=[round(float(v), 4) for v in s.mean(1)], bc=round(float(bc.mean()), 4),
                              by_attempt={m: [round(float(v), 4) for v in full[m].mean(1)] for m in range(R)})
    fig.suptitle(f"{args.tag}: ASTEROID-style, {R} rounds × {cfg['updates']} updates of batch {cfg['batch']}; round n scored on its attempt n+1", fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'success_by_start.png', dpi=120); plt.close(fig)
    fig2.suptitle(f'{args.tag}: success vs attempt index, per round policy', fontsize=10)
    fig2.tight_layout(); fig2.savefig(out / 'success_by_attempt.png', dpi=120); plt.close(fig2)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1)); print(json.dumps({k: {'rounds': v['rounds'], 'bc': v['bc']} for k, v in summary.items()}))


def smoke(args):
    a = argparse.Namespace(tag='asteroid_smoke', rounds=3, updates=40, warmup=10, batch=64, micro=32, lr=1e-4, seed=0, ctx_per_start=1,
                           chunk=512, log_every=10, seeds=2, attempts=4, test_every=20, test_seeds=2, window=-1, splits='train,train_heldout,test', out='',
                           iters=3, steps_per_iter=2, train_seeds=1, no_ctx_feedback=getattr(args, 'no_ctx_feedback', False),
                           obs_noise=getattr(args, 'obs_noise', 'none'), obs_noise_t=getattr(args, 'obs_noise_t', 400),
                           obs_noise_ramp=getattr(args, 'obs_noise_ramp', '600,200'), fb=getattr(args, 'fb', 'bfs'),
                           open_loop=getattr(args, 'open_loop', False), target=getattr(args, 'target', 'demo'),
                           hide_goal=getattr(args, 'hide_goal', False), obs=getattr(args, 'obs', 'none'), obs_t=getattr(args, 'obs_t', 400),
                           no_context=getattr(args, 'no_context', False), eval_rounds='0,-1', test_set='', eval_test_seeds=0)
    import shutil; shutil.rmtree(HERE / 'runs' / a.tag, ignore_errors=True)
    train(a); eval_rounds(a); plot(a); eval_ctx(a)
    print(open(HERE / 'runs' / a.tag / 'log.jsonl').read())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['smoke', 'train', 'eval', 'plot', 'evalctx'])
    ap.add_argument('--tag', default='asteroid'); ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--updates', type=int, default=10000); ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--batch', type=int, default=64); ap.add_argument('--micro', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ctx-per-start', type=int, default=1); ap.add_argument('--log-every', type=int, default=250)
    ap.add_argument('--save-every', type=int, default=0, help='also save the weights every N updates of each round to ckpt/round_<n>_u<u>.pt '
                    '(the end of the round stays round_<n>.pt); for checkpoint selection (select_ckpt.py)')
    ap.add_argument('--seeds', type=int, default=8); ap.add_argument('--attempts', type=int, default=8); ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--test-every', type=int, default=500, help='updates between test-start evals during training (0 = off)')
    ap.add_argument('--test-seeds', type=int, default=8); ap.add_argument('--window', type=int, default=-1, help='eval: earlier attempts kept in context (-1 = all the stream holds, R-1)')
    ap.add_argument('--splits', default='train,test'); ap.add_argument('--out', default='')
    ap.add_argument('--steps-per-iter', type=int, default=0, help='>0: rounds-style schedule, --iters iterations × the same batch for this '
                    'many steps, constant lr, eval after every iteration (overrides --updates / --test-every)')
    ap.add_argument('--obs-noise', default='none', choices=['none', 'flat', 'ramp'], help='training-only noise on the predicted attempt\'s '
                    'observation features (noisy_obs)'); ap.add_argument('--obs-noise-t', type=int, default=400)
    ap.add_argument('--obs-noise-ramp', default='600,200', help='ramp: timestep at slot 0 (no context), timestep at the last slot')
    ap.add_argument('--no-ctx-feedback', action='store_true', help='zero the feedback of the attempts in context (drop_context_feedback)')
    ap.add_argument('--fb', default='bfs', choices=['bfs', 'dir', 'bfs_dir', 'none', 'reward'], help='feedback: none = all zero; reward = step 0, trajectory [r,r,0,0]; bfs = step [-BFS,0,0,0], trajectory [-BFS,-BFS,dx,dy]; '
                    'dir = step [0,0,dx,dy], trajectory [0,0,dx,dy]; bfs_dir = step [-BFS,0,dx,dy], trajectory [-BFS,-BFS,dx,dy] (rounds.FB_MODE)')
    ap.add_argument('--target', default='demo', choices=['demo', 'relabel'], help='demo: supervise the A* demo of the start after the context; '
                    'relabel: ASTEROID / DAgger, the BFS expert labels the cells of the policy\'s own last attempt (rounds.loss_relabel)')
    ap.add_argument('--open-loop', action='store_true', help='obs features of the attempt start cell on every token (rounds.OBS_OPEN); '
                    'prev_a and the per-step feedback still come from the rollout')
    ap.add_argument('--hide-goal', action='store_true', help='the policy never observes the goal (rounds.HIDE_GOAL); needs --fb reward or none')
    ap.add_argument('--iters', type=int, default=30); ap.add_argument('--train-seeds', type=int, default=0, help='also eval train starts')
    ap.add_argument('--obs', default='none', choices=['none', 'window', 'ddpm'], help='policy observation (obs_modes.py): window = ASTEROID 7x7 '
                    'partial view; ddpm = full image DDPM-noised at --obs-t (train and test); both zero goal_pos')
    ap.add_argument('--obs-t', type=int, default=400, help='DDPM timestep of --obs ddpm (0..999)')
    ap.add_argument('--no-context', action='store_true', help='no earlier attempts in context, ever (DAgger: one student attempt per start '
                    'and round, expert-labelled; eval: every attempt starts an empty stream)')
    ap.add_argument('--eval-rounds', default='0,-1', help='evalctx: rounds to evaluate (negative = from the end)')
    ap.add_argument('--test-set', default='', choices=['', 'all', 'control'], help='evalctx: held-out test pairs (interp30 settings: INTERP_TEST)')
    ap.add_argument('--eval-test-seeds', type=int, default=0, help='evalctx: seeds for the test split (0 = --seeds)')
    a = ap.parse_args(); set_fb(a.fb, a.open_loop, a.hide_goal); set_obs(a.obs, a.obs_t)
    {'smoke': smoke, 'train': train, 'eval': eval_rounds, 'plot': plot, 'evalctx': eval_ctx}[a.mode](a)
