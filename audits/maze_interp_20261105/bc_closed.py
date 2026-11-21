"""BC UNet on interp30 re-planning from the CURRENT cell: every --exec env steps the policy sees the current observation (image,
agent_pos, goal_pos; n_obs_steps = 1) and predicts a new chunk, of which the first --exec actions run (snapped to cardinal moves).
--exec 1 = closed loop (one action per observation, like ST); --exec = horizon = the usual open-loop chunk (rounds.bc_eval).
64 env steps per episode, success = within 1 of the goal (stop radius of rounds.bc_eval / ST); a row freezes once it succeeds.

  --select: pick the checkpoint by train_heldout success (100 unseen pairs of TRAIN goals, 3 seeds; held-out goals never used),
            ties to the later step, as bc_select.py does for the open-loop BC  -> runs/<tag>/selection.json
  then:     train starts / 50 test (BC-fail) / 50 control / all 2870 held-out pairs (bc_scan order) x --seeds -> runs/<tag>/bc_closed.npz

  python bc_closed.py --tag bc_h4_closed --run runs/interp30/bc_unet__clean__h4/seed_0 --select --exec 1
  python bc_closed.py --tag bc64_closed  --run runs/interp30/bc_unet__clean/seed_0 --step 7000 --exec 1
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault('MAZE_SETTING', 'interp30')
sys.path.insert(0, '/gscratch/robotics/harine/codex_audits/maze_rounds_20261103')
import numpy as np
import torch
import rounds
from bon_eval import bfs_of
from rounds import SPEC, maze, hydra, build_images_torch, STEP, CONT, snap_cardinal_torch, build_episodes, load_data, train_starts

I = Path(__file__).resolve().parent; T_ENV = 64


def load(run, step, device):
    payload = torch.load(run / f'step_{step:07d}.pt', map_location='cpu', weights_only=False); pc = payload['config']['policy']
    m = hydra.utils.instantiate(pc); m.load_state_dict(payload['model']); m.to(device).eval()
    return m, int(pc['horizon']), pc['shape_meta']


@torch.no_grad()
def rollout(model, H, free1, S, G, exec_k, chunk, device):
    """Success (rows,) of one rollout per row: re-plan every exec_k steps from the current cell, T_ENV steps."""
    pos = torch.as_tensor(np.asarray(S), device=device).float(); goal = torch.as_tensor(np.asarray(G), device=device).float()
    B = len(pos); free = free1.repeat(B, 1, 1); done = torch.full((B,), -1, dtype=torch.long, device=device); t = 0
    while t < T_ENV and bool((done < 0).any()):
        img = build_images_torch(free, pos, goal)
        a = torch.cat([model.predict_action({'image': img[j:j + chunk, None], 'agent_pos': pos[j:j + chunk, None],
                                             'goal_pos': goal[j:j + chunk, None]})['action'] for j in range(0, B, chunk)])
        assert a.shape == (B, H, 2) and torch.isfinite(a).all()
        if not CONT:   # grid: cardinal unit moves; interp30c: real-valued actions (STEP clips |a| to 1)
            a = snap_cardinal_torch(a)
        for i in range(min(exec_k, H, T_ENV - t)):
            active = done < 0
            pos = torch.where(active[:, None], STEP(free, pos, a[:, i]), pos); t += 1
            done = torch.where(active & ((pos - goal).norm(dim=-1) <= 1.), torch.full_like(done, t), done)
    return (done >= 0).cpu().numpy(), pos.cpu().numpy()


def evaluate(model, H, free1, S, G, seeds, exec_k, chunk, device, seed, final=False):
    """Success (pairs, seeds); final=True also returns the final cells (pairs, seeds, 2)."""
    torch.manual_seed(seed)
    ok, pos = rollout(model, H, free1, np.repeat(S, seeds, 0), np.repeat(G, seeds, 0), exec_k, chunk, device)
    return (ok.reshape(len(S), seeds), pos.reshape(len(S), seeds, 2)) if final else ok.reshape(len(S), seeds)


class noisy_features:
    """BC eval with the observation noise of the ST noise runs (asteroid.noisy_obs): the UNet's encoded observation features x
    become sqrt(abar_t) x + sqrt(1 - abar_t) eps * std (DDPM T = 1000, linear 1e-4 -> 0.02; std = per-dimension std of the batch)."""
    def __init__(self, model, t):
        self.m, self.t = model, t

    def __enter__(self):
        if self.t is None:
            return
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        ab = float(DDPMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule='linear').alphas_cumprod[self.t])
        enc = self.m.obs_encoder; orig = type(enc).forward
        def f(obs):
            x = orig(enc, obs); sd = x.std(0, keepdim=True).clamp_min(1e-6)
            return ab ** .5 * x + (1 - ab) ** .5 * torch.randn_like(x) * sd
        enc.forward = f

    def __exit__(self, *exc):
        self.m.obs_encoder.__dict__.pop('forward', None)


def held_out_set(mode):
    os.environ['INTERP_TEST'] = mode; eps = rounds.interp_test_episodes(); os.environ.pop('INTERP_TEST')
    return np.stack([e['start'] for e in eps]).astype(np.float32), np.stack([e['goal'] for e in eps]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--run', required=True)
    ap.add_argument('--step', type=int, default=0); ap.add_argument('--select', action='store_true')
    ap.add_argument('--select-on', default='train_heldout', choices=['train_heldout', 'test'])
    ap.add_argument('--exec', type=int, default=1, help='actions executed per prediction (1 = closed loop)')
    ap.add_argument('--seeds', type=int, default=8); ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--splits', default='train,test,control,allpairs'); ap.add_argument('--out', default='bc_closed.npz')
    ap.add_argument('--obs-noise-t', default='', help='comma list of DDPM timesteps: also evaluate each split with that feature noise '
                    '(noisy_features) -> keys <split>_t<t>_success / _bfs')
    args = ap.parse_args(); device = 'cuda'; torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    run = Path(args.run); run = run if run.is_absolute() else I / run; out = I / 'runs' / args.tag; out.mkdir(parents=True, exist_ok=True)
    pairs = maze.load_train_pairs(SPEC['dataset'])
    steps = sorted(int(p.stem.split('_')[1]) for p in run.glob('step_*.pt'))
    ds = load_data(load(run, steps[-1], 'cpu')[2]); S0, G0, free_np = train_starts(ds); free1 = torch.as_tensor(free_np, device=device)[None]
    if args.select:
        th = build_episodes(SPEC['layouts'], 'train_heldout', 0, 100, SPEC['goal_split'], 0, pairs)
        ts = np.stack([e['start'] for e in th]).astype(np.float32); tg = np.stack([e['goal'] for e in th]).astype(np.float32)
        by, by_test = {}, {}; te_s, te_g = held_out_set('')
        for s in steps:
            model, H, _ = load(run, s, device)
            by[s] = float(evaluate(model, H, free1, ts, tg, 3, args.exec, args.chunk, device, 20261116).mean())
            by_test[s] = float(evaluate(model, H, free1, te_s, te_g, 8, args.exec, args.chunk, device, 20261117).mean())
            print('train_heldout', s, by[s], 'test', by_test[s], flush=True)
        crit = by if args.select_on == 'train_heldout' else by_test
        args.step = max(crit, key=lambda s: (crit[s], s))
        (out / 'selection.json').write_text(json.dumps(dict(step=args.step, select_on=args.select_on,
                                                            rule=f'max {args.select_on} success (train_heldout: 100 episodes x 3 seeds; test: the 50 test pairs x 8 seeds), ties: later step',
                                                            exec=args.exec, run=str(run), by_step=by, by_step_test=by_test), indent=1)); print('selected', args.step, flush=True)
    model, H, _ = load(run, args.step, device)
    res = dict(run=str(run), step=args.step, horizon=H, exec=args.exec, env_steps=T_ENV, seeds=args.seeds)
    sets = {'train': lambda: (S0, G0), 'test': lambda: held_out_set(''), 'control': lambda: held_out_set('control'),
            'allpairs': lambda: held_out_set('all')}
    for k, split in enumerate(args.splits.split(',')):
        t0 = time.time(); S, G = sets[split]()
        res[f'{split}_success'], fin = evaluate(model, H, free1, S, G, args.seeds, args.exec, args.chunk, device, 20261116 + k, final=True)
        if True:   # maze distance of the final (rounded) cell to the goal: the best-of-n verifier of bon_eval.py (also interp30c)
            res[f'{split}_bfs'] = bfs_of(free_np.astype(bool), np.repeat(G, args.seeds, 0), fin.reshape(-1, 2), {}).reshape(len(S), args.seeds)
        res[f'{split}_pairs'] = np.concatenate([S, G], 1)
        print(f'{split} ({len(S)} pairs x {args.seeds} seeds): success {res[f"{split}_success"].mean():.3f}  [{time.time() - t0:.0f}s]', flush=True)
        for t in [int(v) for v in args.obs_noise_t.split(',') if v]:
            with noisy_features(model, t):
                ok, fin = evaluate(model, H, free1, S, G, args.seeds, args.exec, args.chunk, device, 20261116 + k + 7 * t, final=True)
            res[f'{split}_t{t}_success'] = ok
            res[f'{split}_t{t}_bfs'] = bfs_of(free_np.astype(bool), np.repeat(G, args.seeds, 0), fin.reshape(-1, 2), {}).reshape(len(S), args.seeds)
            print(f'  {split} obs noise t={t}: success {ok.mean():.3f}', flush=True)
    np.savez_compressed(out / args.out, **res); print('wrote', out / args.out)


if __name__ == '__main__':
    main()
