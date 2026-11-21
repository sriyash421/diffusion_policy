"""Round-based in-context training for the full-trajectory ST policy (11×11, bottom-right quadrant held out).

  for n = 0 .. R-1:                     (round n: n earlier trajectories from the same start in context)
      repeat ITERS times (each training start ~ITERS*128/637 times per round):
          take the next 128 training starts (cycling random permutations, so coverage is even)
          update: STEPS_PER_ITER optimizer steps on the A* demo of those starts, teacher-forced after the start's n stored
                  trajectories (NLL of the A* actions + BCE of the A* stop flag)
      save the round-n policy; it rolls out one trajectory from every training start (conditioned on that start's n stored
      trajectories), which is appended to the start's stored trajectories for round n+1

Feedback (4 channels per token, RoundsPolicy):
  per action    [-BFS(cell after the action), 0, 0, 0]                                 (verifier heuristic)
  per trajectory, on the first token of the next attempt:
                [-BFS(final cell), -BFS(final cell), dx, dy]   dx, dy = goal - final cell (what is left to reach the goal)
  -BFS is divided by the layout's BFS diameter (32); dx, dy by the grid size - 1 (10), as goal_delta_policy.
Policy: DTSearchPolicy (the 11br dt_st_step architecture, maze_quadrants_delta_20261019) with the 4-channel feedback,
trained from scratch, max_actions = R (the stream never slides). Context tokens are rebuilt from the stored cells with the
current encoder, exactly as the policy's own rollout writes them (checked by `smoke`).

Eval (chain, as in training): per start and seed, round 0's policy makes trajectory 1 from an empty stream, round 1's policy
makes trajectory 2 with trajectory 1 in context, ... Success of round m = round m's own trajectory reaches the goal.
Training starts = the 637 training episodes; test starts = the 100 held-out-quadrant eval episodes of every report.
BC = the 11br UNet (bc_unet seed 3, checkpoint per split as in the main report), open loop, same number of seeds.

  python rounds.py smoke                      # checks + 2 tiny rounds (GPU job)
  python rounds.py train --tag <tag> [--rounds 8 --iters 30 --steps-per-iter 1 --batch 128 --lr 1e-4]
  python rounds.py eval  --tag <tag> [--seeds 8]
  python rounds.py plot  --tag <tag> [--seeds 8]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent; AUD = HERE.parent; DELTA = AUD / 'maze_quadrants_delta_20261019'
sys.path[:0] = [str(DELTA), str(AUD / 'unet64_20261008'), '/gscratch/robotics/harine/L2S']
from setup_data import maze
import hydra
from diffusion_policy.dataset.sim2real_image_dataset import Sim2RealImageDataset
from diffusion_policy.common.sampler import get_collate_fn
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.maze_verifier import build_images_torch, rollout_step_torch, snap_cardinal_torch
from diffusion_policy.policy.dt_search_policy import DTSearchPolicy, TOKEN_KEYS, _expand_idx
from l2s.eval_openloop import BatchedRollout, build_episodes

SPEC = json.loads((DELTA / 'datasets.json').read_text())['11br']
ST_RUN = DELTA / 'runs/11br/dt_st_step__clean/seed_0'
BC_RUN = DELTA / 'runs/11br/bc_unet__clean/seed_3'
BC_CKPT = {'train': 6000, 'test': 7000}   # main-report BC selection: training goals (train_heldout) / held-out quadrant (eval)
# MAZE_SETTING=interp30 (maze_interp_20261105): 30 random train goals / 41 held out, the 50 BC-failure-first test pairs,
# the interp BC UNet at its train_heldout-selected step, and runs written to INTERP/runs (HERE is only used for runs/).
import os
SETTING = os.environ.get('MAZE_SETTING', '11br'); INTERP = AUD / 'maze_interp_20261105'
if SETTING == 'interp30':
    SPEC = json.loads((DELTA / 'datasets.json').read_text())['interp30']
    BC_RUN = INTERP / 'runs/interp30/bc_unet__clean/seed_0'
    if (INTERP / 'bc_selection.json').exists():
        _s = json.loads((INTERP / 'bc_selection.json').read_text())['step']; BC_CKPT = {'train': _s, 'test': _s}
    else:
        BC_CKPT = {}
    HERE = INTERP
elif SETTING == 'interp30c':
    # continuous positions + smoothed-A* demos (maze_interp_20261105/cmaze.py): real-valued actions, no cardinal snap, the env step is
    # cmaze.cont_step_torch (|a| clipped to 1, walls block / slide); same goals, pairs and test sets as interp30; no BC here (bc_closed.py)
    SPEC = json.loads((DELTA / 'datasets.json').read_text())['interp30c']; BC_CKPT = {}; HERE = INTERP
elif SETTING != '11br':
    raise ValueError(f'MAZE_SETTING must be 11br, interp30 or interp30c; got {SETTING!r}')
CONT = SETTING == 'interp30c'
if CONT:
    sys.path.insert(0, str(INTERP)); from cmaze import cont_step_torch as STEP
else:
    STEP = rollout_step_torch
FB = 4
FB_MODE = 'bfs'   # 'bfs': step [-BFS,0,0,0], trajectory [-BFS,-BFS,dx,dy];  'dir': step [0,0,dx,dy], trajectory [0,0,dx,dy]
                  # 'bfs_dir': step [-BFS,0,dx,dy], trajectory [-BFS,-BFS,dx,dy]
                  # 'none': every feedback channel zero (step and trajectory)
                  # 'reward': ASTEROID's sparse reward: step [0,0,0,0]; trajectory [r,r,0,0], r = 1 if the attempt ended within the success
                  #           radius of the goal (an attempt ends there, so a positive step reward is never seen inside an attempt)
                  #  (dx, dy = goal - cell, / 10; step: the cell after the move, trajectory: the final cell). Set via asteroid.py --fb.
OBS_OPEN = False  # open-loop observations (asteroid.py --open-loop): every token of an attempt carries the obs features of the attempt's
                  # START cell (image, agent_pos), not of its current cell; prev_a and the feedback channels still follow the rollout.
HIDE_GOAL = False # hidden goal (asteroid.py --hide-goal; ASTEROID's partially observed setting): the policy's observation encoder gets
                  # the image with the goal plane cleared and goal_pos = 0, so the goal is known only through the context / feedback.
                  # The env, the success test, the BFS expert and the best-of-n verifier still use the true goal.


def fb_step(bfs, d):
    """Per-action feedback channels from -BFS/32 (...) and (goal - cell)/10 (..., 2). FB_MODE 'none': all zero."""
    if FB_MODE in ('none', 'reward'):
        return bfs.new_zeros(bfs.shape + (FB,))
    if FB_MODE == 'bfs_dir':
        return torch.cat([bfs[..., None], torch.zeros_like(bfs[..., None]), d], -1)
    return F.pad(d, (FB - 2, 0)) if FB_MODE == 'dir' else F.pad(bfs[..., None], (0, FB - 1))


def fb_end(bfs, d):
    """Per-trajectory feedback (first token of the next attempt) from the final cell. FB_MODE 'none': all zero."""
    if FB_MODE == 'none':
        return bfs.new_zeros(bfs.shape + (FB,))
    if FB_MODE == 'reward':   # d = (goal - cell) / (grid - 1); interp30 / 11br are 11 x 11 -> 10; success radius 1 (stop_radius)
        r = ((d * 10).norm(dim=-1) <= 1 + 1e-4).to(bfs.dtype)
        return F.pad(torch.stack([r, r], -1), (0, FB - 2))
    return F.pad(d, (FB - 2, 0)) if FB_MODE == 'dir' else torch.cat([torch.stack([bfs, bfs], -1), d], -1)


# ---------------------------------------------------------------- policy
class RoundsPolicy(DTSearchPolicy):
    """DTSearchPolicy with feedback [step -BFS, trajectory -BFS, trajectory dx, trajectory dy] (see the module docstring)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.fb_emb = nn.Linear(FB, self.n_emb)

    @staticmethod
    def delta(pos, goal, free):
        return (goal - pos) / pos.new_tensor([free.shape[-1] - 1, free.shape[-2] - 1])

    def _mask_fb(self, fb):
        return fb

    def _obs_features(self, obs):
        """HIDE_GOAL: the encoder never sees the goal (goal plane of the (..., 3, H, W) wall/goal/agent image cleared, goal_pos = 0)."""
        if HIDE_GOAL:
            img = obs['image'].clone(); img[..., 1, :, :] = 0
            obs = {**obs, 'image': img, 'goal_pos': torch.zeros_like(obs['goal_pos'])}
        return super()._obs_features(obs)

    @staticmethod
    def _empty_stream(B, Fo, Da, device):
        S = DTSearchPolicy._empty_stream(B, Fo, Da, device)
        S['fb'] = torch.zeros((B, 0, FB), device=device); S['last_fb'] = torch.zeros(B, FB, device=device)
        return S

    def _step(self, obs, free):
        """DTSearchPolicy._step with the 4-channel feedback."""
        pos = obs['agent_pos'].float(); goal = obs['goal_pos'].float(); B = pos.shape[0]; device = pos.device
        feat = self._obs_features(obs)
        if self._st is None:
            self._st = self._empty_stream(B, feat.shape[1], self.action_dim, device)
            self._st['att_starts'].append(self._st['len'].clone())
        S = self._st
        first = S['t'] is None
        if first:
            S['field'], S['unreach'] = self.verifier._goal_distance_field_torch(free, goal)
            S['scale'] = self._scale(free); S['guard'] = free.flatten(1).sum(1)
            S['t'] = torch.zeros(B, dtype=torch.long, device=device); S['active'] = torch.ones(B, dtype=torch.bool, device=device)
            S['pos'] = pos; S['stop'] = torch.zeros(B, dtype=torch.bool, device=device); S['feat0'] = feat
        if OBS_OPEN:
            feat = S['feat0']
        bfs = -self.verifier._sample_goal_distance_torch(free, S['field'], pos, S['unreach']) / S['scale']
        realized = self.normalizer['action'].normalize(pos - S['pos'])
        S['pos'] = pos
        if not first:
            reached = (pos - goal).norm(dim=-1) <= self.stop_radius
            ended = S['active'] & (reached | S['stop'] | (S['t'] >= S['guard']))
            S['last_a'] = torch.where(ended[:, None], realized, S['last_a'])
            end_fb = fb_end(bfs, self.delta(pos, goal, free))
            S['last_fb'] = torch.where(ended[:, None], end_fb, S['last_fb'])
            S['active'] = S['active'] & ~ended
        active = S['active']; zeros = torch.zeros(B, 1, device=device)
        if first and S['has_prev']:
            tok = {'prev_a': S['last_a'], 'fb': S['last_fb'], 'prev_d': torch.ones(B, 1, device=device)}
        elif first:
            tok = {'prev_a': torch.zeros(B, self.action_dim, device=device), 'fb': torch.zeros(B, FB, device=device), 'prev_d': zeros}
        else:
            tok = {'prev_a': realized, 'fb': fb_step(bfs, self.delta(pos, goal, free)), 'prev_d': zeros}
        tok.update(feat=feat, step=S['t'])
        self._append(S, active, tok)
        h = self._hidden(S)[torch.arange(B, device=device), (S['len'] - 1).clamp_min(0)]
        dist, stop_logit = self._heads(h)
        action = self.normalizer['action'].unnormalize(dist.sample())
        if self.snap_cardinal:
            action = snap_cardinal_torch(action)
        action = torch.where(active[:, None], action, torch.zeros_like(action))
        S['stop'] = torch.sigmoid(stop_logit) > 0.5
        S['t'] = S['t'] + active.long()
        return action, active.clone()


def make_policy(rounds, device):
    pc = json.loads((ST_RUN / 'config.json').read_text())['policy']
    pc.update(max_actions=rounds, device=device, kv_cache=False)
    if CONT:
        pc['snap_cardinal'] = False
    kw = {k: v for k, v in pc.items() if k not in ('_target_', 'obs_encoder')}
    return RoundsPolicy(obs_encoder=hydra.utils.instantiate(pc['obs_encoder']), **kw), pc


# ---------------------------------------------------------------- data
def load_data(shape):
    ds = Sim2RealImageDataset(SPEC['dataset'], shape, horizon=max(100, SPEC['expert_max']), pad_before=0, pad_after=0,
                              n_obs_steps=None, seed=0, val_ratio=.2, use_cache=False, use_disk=False, return_sequences=True,
                              zero_pad_action_after=False, val_split='pair')
    assert len(ds) == int(ds.train_mask.sum()), (len(ds), int(ds.train_mask.sum()))   # one sample per training episode
    return ds


def train_starts(ds):
    s = np.stack([np.asarray(ds[i]['obs']['agent_pos'][0]) for i in range(len(ds))]).astype(np.float32)
    g = np.stack([np.asarray(ds[i]['obs']['goal_pos'][0]) for i in range(len(ds))]).astype(np.float32)
    free = np.asarray(ds[0]['obs']['image'][0, 0]) < 0.5
    return s, g, free


def test_episodes():
    if SETTING in ('interp30', 'interp30c'):   # interp30c: the same held-out pairs (integer cell-centre starts)
        return interp_test_episodes()
    pairs = maze.load_train_pairs(SPEC['dataset'])
    return build_episodes(SPEC['layouts'], 'eval', 0, 100, SPEC['goal_split'], 0, pairs)


def interp_test_episodes():
    """The 50 interp30 test pairs (INTERP/test_episodes.json, BC-failures first), in build_episodes' format.
    INTERP_TEST=control: the 50 BC-success control pairs; INTERP_TEST=all: every (start, held-out goal) pair of the BC scan (bc_scan.npz)."""
    eps = []; mode = os.environ.get('INTERP_TEST', '')
    if mode == 'all':
        items = [dict(start=p[:2].tolist(), goal=p[2:].tolist()) for p in np.load(INTERP / 'bc_scan.npz')['pairs']]
    else:
        items = json.loads((INTERP / ('control_episodes.json' if mode == 'control' else 'test_episodes.json')).read_text())['episodes']
    for e in items:
        env = maze.MazeEnv(SPEC['layouts'], seed=0, split='eval', layout_index=0, goal_split=SPEC['goal_split']); env.reset()
        env.pos = np.asarray(e['start'], np.int32).copy(); env.goal = np.asarray(e['goal'], np.int32).copy()
        eps.append(dict(env=env, start=np.asarray(e['start'], np.float32), goal=np.asarray(e['goal'], np.float32), nav=env.maze))
    return eps


# ---------------------------------------------------------------- context from stored trajectories
@torch.no_grad()
def build_context(model, free, goal, trajs):
    """Stream for rows with n stored trajectories each: trajs[b] = list of n int arrays (T+1, 2) of cells (start .. end).
    Returns the dict `_step` continues from (tokens, len, att_starts, last_a, last_fb), or None when n = 0."""
    B = len(trajs); n = len(trajs[0]); dev = goal.device
    if n == 0:
        return None
    assert all(len(t) == n for t in trajs)
    norm = model.normalizer['action']
    rows = []
    for b in range(B):
        pos, prev, step, starts, cells, L = [], [], [], [], [], 0
        for c in trajs[b]:
            c = np.asarray(c, np.float32); T = len(c) - 1; assert T >= 1
            starts.append(L); L += T
            pos.append(np.repeat(c[:1], T, 0) if OBS_OPEN else c[:T])   # OBS_OPEN: obs features of the attempt's start cell
            prev.append(np.concatenate([c[:1], c[:T - 1]])); step.append(np.arange(T)); cells.append(c[:T])
        rows.append(dict(cells=np.concatenate(cells), pos=np.concatenate(pos), prev=np.concatenate(prev), step=np.concatenate(step), starts=starts,
                         ends=np.stack([np.asarray(c, np.float32)[-2:] for c in trajs[b]]), L=L))
    Lmax = max(r['L'] for r in rows)
    Po = torch.zeros(B, Lmax, 2, device=dev); P = torch.zeros(B, Lmax, 2, device=dev); Pv = torch.zeros(B, Lmax, 2, device=dev); ST = torch.zeros(B, Lmax, dtype=torch.long, device=dev)
    for b, r in enumerate(rows):
        Po[b, :r['L']] = torch.as_tensor(r['pos'], device=dev); P[b, :r['L']] = torch.as_tensor(r['cells'], device=dev); Pv[b, :r['L']] = torch.as_tensor(r['prev'], device=dev)
        ST[b, :r['L']] = torch.as_tensor(r['step'], device=dev)
    lens = torch.tensor([r['L'] for r in rows], device=dev)
    valid = torch.arange(Lmax, device=dev)[None] < lens[:, None]
    gl = goal.repeat_interleave(Lmax, 0); flat = Po.reshape(-1, 2)
    feat = model._obs_features({'image': build_images_torch(free.repeat_interleave(Lmax, 0), flat, gl), 'agent_pos': flat,
                                'goal_pos': gl}).reshape(B, Lmax, -1)
    field, unreach = model.verifier._goal_distance_field_torch(free, goal); scale = model._scale(free)
    bfs_of = lambda q: -model.verifier._sample_goal_distance_torch(free.repeat_interleave(q.shape[1], 0), field.repeat_interleave(q.shape[1], 0),
                                                                   q.reshape(-1, 2), unreach).reshape(B, q.shape[1]) / scale
    bfs = bfs_of(P)
    E = torch.as_tensor(np.stack([r['ends'] for r in rows]), device=dev)                       # (B, n, 2 [before last move, final], 2)
    last_move = norm.normalize((E[:, :, 1] - E[:, :, 0]).reshape(-1, 2)).reshape(B, n, 2)
    fbf = bfs_of(E[:, :, 1])
    end_fb = fb_end(fbf, model.delta(E[:, :, 1], goal[:, None], free))   # (B, n, 4)
    prev_a = norm.normalize((P - Pv).reshape(-1, 2)).reshape(B, Lmax, 2)
    fb = fb_step(bfs, model.delta(P, goal[:, None], free))
    prev_d = torch.zeros(B, Lmax, 1, device=dev); ar = torch.arange(B, device=dev)
    for j in range(n):
        s = torch.tensor([r['starts'][j] for r in rows], device=dev)
        if j == 0:
            prev_a[ar, s] = 0; fb[ar, s] = 0
        else:
            prev_a[ar, s] = last_move[:, j - 1]; fb[ar, s] = end_fb[:, j - 1]; prev_d[ar, s] = 1
    m = valid[..., None].float()
    return {'feat': feat * m, 'prev_a': prev_a * m, 'fb': fb * m, 'prev_d': prev_d * m, 'step': ST * valid, 'len': lens,
            'att_starts': [torch.tensor([r['starts'][j] for r in rows], device=dev) for j in range(n)], 'has_prev': True, 't': None,
            'last_a': last_move[:, -1], 'last_fb': end_fb[:, -1].clone()}


def slice_ctx(ctx, sl, B):
    if ctx is None:
        return None
    return {k: (v[sl] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B else [a[sl] for a in v] if k == 'att_starts' else v)
            for k, v in ctx.items()}


@torch.no_grad()
def rollout(model, free, start, goal, ctx):
    """One trajectory per row continuing the stream ctx (None = empty), as DTSearchPolicy.rollout_candidates_torch."""
    B = start.shape[0]; dev = start.device
    model._st = None if ctx is None else {**ctx, 'att_starts': list(ctx['att_starts'])}
    model.begin_attempt()
    pos = start.float(); traj = [pos]; ds = torch.full((B,), -1, dtype=torch.long, device=dev); t = 0
    while True:
        action, active = model._step({'image': build_images_torch(free, pos, goal), 'agent_pos': pos, 'goal_pos': goal}, free)
        if not bool(active.any()):
            break
        pos = torch.where(active[:, None], STEP(free, pos, action), pos); traj.append(pos); t += 1
        reached = active & (ds < 0) & ((pos - goal).norm(dim=-1) <= model.stop_radius)
        ds = torch.where(reached, torch.full_like(ds, t), ds)
    n_steps = model._st['t'].cpu().numpy(); model._st = None
    traj = torch.stack(traj, 1).cpu().numpy() if CONT else torch.stack(traj, 1).round().long().cpu().numpy()
    return [traj[b, :n_steps[b] + 1].astype(np.float32 if CONT else np.int8) for b in range(B)], (ds >= 0).cpu().numpy(), n_steps


@torch.no_grad()
def rollout_all(model, free1, S0, G0, memory, chunk, device):
    """rollout() over all rows in chunks; memory[r] = the stored trajectories of row r."""
    trajs, ok, ns = [], [], []
    for j in range(0, len(S0), chunk):
        rows = range(j, min(len(S0), j + chunk)); free = free1.repeat(len(rows), 1, 1)
        gl = torch.as_tensor(G0[j:j + chunk], device=device)
        ctx = build_context(model, free, gl, [memory[r] for r in rows])
        t, o, n = rollout(model, free, torch.as_tensor(S0[j:j + chunk], device=device), gl, ctx)
        trajs += t; ok.append(o); ns.append(n)
    return trajs, np.concatenate(ok), np.concatenate(ns)


def loss_with_context(model, batch, ctx):
    """NLL + stop BCE of the A* demo, teacher-forced after the stored trajectories in ctx (None = no context)."""
    obs = batch['obs']; nactions = model.normalizer['action'].normalize(batch['action'])
    B, T, Da = nactions.shape; dev = nactions.device
    mask = batch['attention_mask'].bool()
    feat = model._obs_features(dict_apply(obs, lambda x: x.reshape(-1, *x.shape[2:]))).reshape(B, T, -1)
    if OBS_OPEN:   # every step sees the demo's start observation
        feat = feat[:, :1].expand(-1, T, -1)
    free = obs['image'][:, 0, 0] < 0.5; goal = obs['goal_pos'][:, 0]
    field, unreach = model.verifier._goal_distance_field_torch(free, goal)
    bfs = -model.verifier._sample_goal_distance_torch(free.repeat_interleave(T, 0), field.repeat_interleave(T, 0),
                                                      obs['agent_pos'].reshape(-1, 2), unreach).reshape(B, T) / model._scale(free)
    prev_a = torch.cat([torch.zeros_like(nactions[:, :1]), nactions[:, :-1]], 1)
    fb = fb_step(bfs, model.delta(obs['agent_pos'].float(), goal[:, None].float(), free)); fb[:, 0] = 0
    prev_d = torch.zeros(B, T, 1, device=dev); steps = torch.arange(T, device=dev)[None].expand(B, -1)
    stop_target = mask & ~torch.cat([mask[:, 1:], torch.zeros_like(mask[:, :1])], 1)
    S = model._empty_stream(B, feat.shape[-1], Da, dev)
    if ctx is not None:
        L = int(ctx['len'].max())
        for k in TOKEN_KEYS:
            S[k] = ctx[k][:, :L]
        S['len'] = ctx['len'].clone()
        prev_a = torch.cat([ctx['last_a'][:, None], prev_a[:, 1:]], 1)
        fb = torch.cat([ctx['last_fb'][:, None], fb[:, 1:]], 1)
        prev_d = torch.cat([torch.ones(B, 1, 1, device=dev), prev_d[:, 1:]], 1)
    off = S['len']; idx = off[:, None] + steps; Ltot = int((off + T).max())
    for k, v in (('feat', feat), ('prev_a', prev_a), ('fb', fb), ('prev_d', prev_d), ('step', steps)):
        full = torch.zeros((B, Ltot) + tuple(v.shape[2:]), dtype=v.dtype, device=dev)
        full[:, :S[k].shape[1]] = S[k]
        S[k] = full.scatter(1, _expand_idx(idx, v), v)
    S['len'] = off + mask.sum(1)
    h = model._hidden(S); h = torch.gather(h, 1, _expand_idx(idx, h))
    dist, stop_logit = model._heads(h)
    m = mask.float()
    nll = -(dist.log_prob(nactions).sum(-1) * m).sum() / m.sum().clamp_min(1.)
    bce = (F.binary_cross_entropy_with_logits(stop_logit, stop_target.float(), reduction='none') * m).sum() / m.sum().clamp_min(1.)
    return nll + bce


# ---------------------------------------------------------------- BFS expert relabeling (asteroid.py --target relabel)
_EXPERT = {}


def expert_tables(free_np):
    """The BFS expert for every (goal, cell) of the layout: its first move (l2s.maze.shortest_path from the cell, the demo generator:
    BFS with neighbour order +x, -x, +y, -y) and the BFS distance. (H, W, H, W, 2) float / (H, W, H, W) int, indexed [gy, gx, cy, cx]."""
    free_np = np.asarray(free_np, bool); key = free_np.tobytes()
    if key not in _EXPERT:
        from types import SimpleNamespace
        Hh, Ww = free_np.shape; env = SimpleNamespace(maze=free_np.astype(np.uint8), w=Ww, h=Hh)
        mv = np.zeros((Hh, Ww, Hh, Ww, 2), np.float32); dist = -np.ones((Hh, Ww, Hh, Ww), np.int64)
        cells = [(x, y) for y in range(Hh) for x in range(Ww) if free_np[y, x]]
        for gx, gy in cells:
            for cx, cy in cells:
                path = maze.shortest_path(env, (cx, cy), (gx, gy)); dist[gy, gx, cy, cx] = len(path) - 1
                if len(path) > 1:
                    mv[gy, gx, cy, cx] = path[1] - path[0]
        _EXPERT[key] = (mv, dist)
    return _EXPERT[key]


def expert_rollout(free_np, start, goal):
    """Cells (T+1, 2) int8 of the BFS expert from start until it stands on the goal (as the demos: the path ends on the goal cell)."""
    mv, dist = expert_tables(free_np); g = np.rint(goal).astype(int); c = np.rint(start).astype(int); out = [c.copy()]
    while dist[g[1], g[0], c[1], c[0]] > 0:
        c = c + mv[g[1], g[0], c[1], c[0]].astype(int); out.append(c.copy())
    assert len(out) > 1, 'start on the goal'
    return np.stack(out).astype(np.int8)


def expert_labels(free_np, goal, cells):
    """BFS expert labels at cells (T, 2): its move (T, 2) and its stop flag (T,) = the move lands on the goal (the demos' last token)."""
    mv, dist = expert_tables(free_np); g = np.rint(goal).astype(int); c = np.asarray(cells).astype(int)
    d = dist[g[1], g[0], c[:, 1], c[:, 0]]; assert (d > 0).all(), 'a labelled cell is the goal'
    return mv[g[1], g[0], c[:, 1], c[:, 0]], d == 1


def loss_relabel(model, free, goal, ctxs, labs):
    """ASTEROID loss: the expert relabels the states the learner visited. Row b = the learner's earlier attempts ctxs[b] (context, not
    supervised) + its last attempt labs[b] (cells, T+1); every token of that last attempt (the token the policy itself saw when it
    acted, rebuilt by build_context) is scored on the BFS expert's move and stop flag at its cell. Returns (loss, labelled tokens)."""
    B = len(labs); dev = goal.device; free_np = free[0].cpu().numpy()
    was = model.training; model.eval()
    with torch.no_grad():
        S = build_context(model, free, goal, [list(c) + [l] for c, l in zip(ctxs, labs)])
    model.train(was)
    T = np.array([len(l) - 1 for l in labs]); Tm = int(T.max())
    act = np.zeros((B, Tm, 2), np.float32); stop = np.zeros((B, Tm), np.float32); m = np.zeros((B, Tm), np.float32)
    g = goal.cpu().numpy()
    for b, l in enumerate(labs):
        a, st = expert_labels(free_np, g[b], np.asarray(l)[:T[b]]); act[b, :T[b]] = a; stop[b, :T[b]] = st; m[b, :T[b]] = 1
    idx = S['att_starts'][-1][:, None] + torch.arange(Tm, device=dev)[None]
    assert bool((S['att_starts'][-1] + torch.as_tensor(T, device=dev) == S['len']).all())
    idx = torch.minimum(idx, S['len'][:, None] - 1)
    h = model._hidden({**{k: S[k] for k in TOKEN_KEYS}, 'len': S['len']}); h = torch.gather(h, 1, _expand_idx(idx, h))
    dist, stop_logit = model._heads(h)
    na = model.normalizer['action'].normalize(torch.as_tensor(act, device=dev)); mm = torch.as_tensor(m, device=dev)
    nll = -(dist.log_prob(na).sum(-1) * mm).sum() / mm.sum()
    bce = (F.binary_cross_entropy_with_logits(stop_logit, torch.as_tensor(stop, device=dev), reduction='none') * mm).sum() / mm.sum()
    return nll + bce, int(m.sum())


# ---------------------------------------------------------------- train
def train(args):
    device = 'cuda'; torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(4)
    out = HERE / 'runs' / args.tag; out.mkdir(parents=True, exist_ok=True)
    model, pc = make_policy(args.rounds, device)
    ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer()); model.to(device)
    collate = get_collate_fn(); N = len(ds)
    S0, G0, free_np = train_starts(ds); free1 = torch.as_tensor(free_np, device=device)[None]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.95, .999), eps=1e-8, weight_decay=1e-6)
    memory = [[] for _ in range(N)]; rng = np.random.default_rng(args.seed); order = []
    config = dict(vars(args), policy=pc, feedback='step [-BFS,0,0,0]; trajectory [-BFS, -BFS, dx, dy]', train_starts=N,
                  schedule='constant lr, no warmup')
    (out / 'config.json').write_text(json.dumps(config, indent=1))
    log = (out / 'log.jsonl').open('w')
    if args.curve:   # in-training eval: train starts × curve_train_seeds, test starts × curve_test_seeds, each with its own chain
        te = test_episodes(); te_s = np.stack([e['start'] for e in te]).astype(np.float32); te_g = np.stack([e['goal'] for e in te]).astype(np.float32)
        cur = {'train': (np.repeat(S0, args.curve_train_seeds, 0), np.repeat(G0, args.curve_train_seeds, 0)),
               'test': (np.repeat(te_s, args.curve_test_seeds, 0), np.repeat(te_g, args.curve_test_seeds, 0))}
        cmem = {k: [[] for _ in range(len(v[0]))] for k, v in cur.items()}

    def curve(n, it, append=False):
        """Success of the current weights from every eval start, conditioned on that start's n chain trajectories (made by the
        final policies of rounds 0..n-1, as in eval_rounds). append=True (end of round): store this rollout in the chain."""
        model.eval(); rec = dict(round=n, iter=it)
        for k, (S, G) in cur.items():
            trajs, ok, _ = rollout_all(model, free1, S, G, cmem[k], args.chunk, device); rec[f'{k}_success'] = float(ok.mean())
            if append:
                for r, tr in enumerate(trajs):
                    cmem[k][r].append(tr)
        log.write(json.dumps(rec) + '\n'); log.flush(); print('CURVE', json.dumps(rec), flush=True); model.train()

    for n in range(args.rounds):
        seen = np.zeros(N, int); t0 = time.time()
        if args.curve:
            curve(n, 0)
        for it in range(args.iters):
            while len(order) < args.batch:
                order += list(rng.permutation(N))
            idx, order = order[:args.batch], order[args.batch:]
            seen[idx] += 1
            batch = dict_apply(collate([ds[i] for i in idx]), lambda x: x.to(device))
            model.eval()
            ctx = build_context(model, free1.repeat(len(idx), 1, 1), torch.as_tensor(G0[idx], device=device), [memory[i] for i in idx])
            model.train(); losses = []
            for _ in range(args.steps_per_iter):
                opt.zero_grad(); denom = batch['attention_mask'].sum().item(); tot = 0.
                for j in range(0, len(idx), args.micro):
                    sl = slice(j, j + args.micro); part = dict_apply(batch, lambda x: x[sl])
                    loss = loss_with_context(model, part, slice_ctx(ctx, sl, len(idx))); assert torch.isfinite(loss)
                    w = part['attention_mask'].sum().item() / denom; (loss * w).backward(); tot += loss.item() * w
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); losses.append(tot)
            log.write(json.dumps(dict(round=n, iter=it, loss=float(np.mean(losses)))) + '\n'); log.flush()
            if args.curve and (it + 1) % args.curve == 0:
                curve(n, it + 1, append=(it + 1 == args.iters))
        torch.save(dict(model=model.state_dict(), round=n, config=config), out / f'round_{n}.pt')
        model.eval()
        trajs, ok, ns = rollout_all(model, free1, S0, G0, memory, args.chunk, device)
        for i in range(N):
            memory[i].append(trajs[i])
        rec = dict(round=n, loss_last_iter=float(np.mean(losses)), visits_min=int(seen.min()), visits_max=int(seen.max()),
                   train_start_success=float(ok.mean()), mean_steps=float(ns.mean()), seconds=round(time.time() - t0))
        log.write(json.dumps(rec) + '\n'); log.flush(); print('ROUND', json.dumps(rec), flush=True)
    with (out / 'memory.npz').open('wb') as f:   # per round: the cells of each start's stored trajectory, concatenated, + offsets
        np.savez_compressed(f, **{k: v for n in range(args.rounds) for k, v in
                                  ((f'r{n}_cells', np.concatenate([memory[i][n] for i in range(N)])),
                                   (f'r{n}_offsets', np.cumsum([0] + [len(memory[i][n]) for i in range(N)])))})
    (out / 'TRAINED').write_text('ok\n')


# ---------------------------------------------------------------- eval
@torch.no_grad()
def eval_rounds(args):
    device = 'cuda'; out = HERE / 'runs' / args.tag; cfg = json.loads((out / 'config.json').read_text())
    R = cfg['rounds']; model, pc = make_policy(R, device)
    ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer()); model.to(device).eval()
    tr_s, tr_g, free_np = train_starts(ds); free1 = torch.as_tensor(free_np, device=device)[None]
    te = test_episodes(); te_s = np.stack([e['start'] for e in te]).astype(np.float32); te_g = np.stack([e['goal'] for e in te]).astype(np.float32)
    assert np.array_equal(te[0]['nav'].astype(bool), free_np), 'test layout differs from the training layout'
    states = [torch.load(out / f'round_{m}.pt', map_location='cpu', weights_only=False)['model'] for m in range(R)]
    res = {}; torch.manual_seed(20261103)
    for split, (S0, G0) in {'train': (tr_s, tr_g), 'test': (te_s, te_g)}.items():
        S = np.repeat(S0, args.seeds, 0); G = np.repeat(G0, args.seeds, 0)
        succ = np.zeros((R, len(S)), bool); steps = np.zeros((R, len(S)), int); memory = [[] for _ in range(len(S))]
        for m in range(R):
            model.load_state_dict(states[m]); model.to(device).eval()   # the normalizer's load rebuilds its tensors on the state dict's device
            trajs, succ[m], steps[m] = rollout_all(model, free1, S, G, memory, args.chunk, device)
            for r, tr in enumerate(trajs):
                memory[r].append(tr)
            print(split, 'round', m, 'success', succ[m].mean().round(3), flush=True)
        res[f'{split}_success'] = succ.reshape(R, len(S0), args.seeds); res[f'{split}_steps'] = steps.reshape(R, len(S0), args.seeds)
        res[f'{split}_bc_success'] = bc_eval(BC_CKPT[split], S0, G0, te if split == 'test' else None, args.seeds, device)
        print(split, 'BC success', res[f'{split}_bc_success'].mean().round(3), flush=True)
    np.savez_compressed(out / f'eval_seeds{args.seeds}.npz', **res)


@torch.no_grad()
def bc_eval(step, S0, G0, episodes, seeds, device):
    """(starts, seeds) success of the BC UNet: open-loop H-step chunk, iid per seed (as evaluate.py)."""
    payload = torch.load(BC_RUN / f'step_{step:07d}.pt', map_location='cpu'); cfg = payload['config']; H = cfg['data']['horizon']
    model = hydra.utils.instantiate(cfg['policy']); model.load_state_dict(payload['model']); model.to(device).eval()
    if episodes is None:
        episodes = []
        for s, g in zip(S0, G0):
            env = maze.MazeEnv(SPEC['layouts'], seed=0, split='train', layout_index=0, goal_split=SPEC['goal_split']); env.reset()
            env.goal = g.astype(np.int32).copy(); episodes.append(dict(env=env, start=s, goal=g))
    out = np.zeros((len(episodes), seeds), bool)
    for k in range(seeds):
        roll = BatchedRollout([e['env'] for e in episodes], [e['start'] for e in episodes], [e['goal'] for e in episodes], 1., 1)
        obs = roll.obs_tensor(device); aa = []
        for j in range(0, len(episodes), 64):
            pred = model.predict_action({key: v[j:j + 64] for key, v in obs.items()})['action']; aa.append(snap_cardinal_torch(pred).cpu().numpy())
        a = np.concatenate(aa)
        for t in range(H):
            roll.step(a[:, t])
        out[:, k] = roll.done_step >= 0
    return out


# ---------------------------------------------------------------- plot
def plot(args):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    out = HERE / 'runs' / args.tag; z = np.load(out / f'eval_seeds{args.seeds}.npz'); cfg = json.loads((out / 'config.json').read_text())
    R = cfg['rounds']; cols = plt.cm.viridis(np.linspace(0, .95, R)); summary = {}
    fig, axs = plt.subplots(2, 2, figsize=(18, 8), gridspec_kw=dict(width_ratios=[4, 1]))
    for i, (split, name) in enumerate([('train', 'training start points'), ('test', 'test start points (held-out quadrant)')]):
        s = z[f'{split}_success'].mean(-1); bc = z[f'{split}_bc_success'].mean(-1); x = np.arange(s.shape[1])
        ax = axs[i, 0]
        for m in range(R):
            ax.plot(x, s[m], color=cols[m], lw=.9, label=f'round {m} ({m} traj. in context): {s[m].mean():.2f}')
        ax.plot(x, bc, color='#D55E00', lw=1.2, ls='--', label=f'BC (UNet): {bc.mean():.2f}')
        ax.set_xlabel(f'{name}: index'); ax.set_ylabel(f'success over {args.seeds} seeds'); ax.set_ylim(-.03, 1.03); ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=3, loc='lower right')
        ax = axs[i, 1]; ax.plot(range(R), s.mean(1), 'o-', color='#0072B2', label='round policy'); ax.axhline(bc.mean(), color='#D55E00', ls='--', label='BC (UNet)')
        ax.set_xlabel('round n'); ax.set_ylabel('mean success'); ax.set_ylim(-.03, 1.03); ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_title(name, fontsize=9)
        summary[split] = dict(rounds=[round(float(v), 4) for v in s.mean(1)], bc=round(float(bc.mean()), 4))
    fig.suptitle(f"{args.tag}: {R} rounds × {cfg['iters']} iterations × {cfg['steps_per_iter']} step(s) of batch {cfg['batch']}", fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'success_by_start.png', dpi=120); plt.close(fig)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary))


# ---------------------------------------------------------------- smoke
def smoke(args):
    """build_context reproduces the policy's own rollout stream; then 2 tiny rounds + eval + plot."""
    device = 'cuda'; torch.manual_seed(0)
    model, pc = make_policy(4, device)
    ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer()); model.to(device).eval()
    collate = get_collate_fn(); b = dict_apply(collate([ds[i] for i in range(16)]), lambda x: x.to(device)); o = b['obs']
    free, st, gl = o['image'][:, 0, 0] < 0.5, o['agent_pos'][:, 0], o['goal_pos'][:, 0]
    c = model.rollout_candidates_torch(free, st, gl, 3)
    trajs = [[c['traj'][r, k, :int(c['n_steps'][r, k]) + 1].round().long().cpu().numpy() for k in range(3)] for r in range(16)]
    ctx = build_context(model, free, gl, trajs); worst = {}
    for k in TOKEN_KEYS:
        for r in range(16):
            L = int(ctx['len'][r]); worst[k] = max(worst.get(k, 0.), float((ctx[k][r, :L].float() - c['stream'][k][r, :L].float()).abs().max()))
    worst['len'] = int((ctx['len'] != c['snaps'][2]['len']).sum())
    worst['last_a'] = float((ctx['last_a'] - c['snaps'][2]['last_a']).abs().max()); worst['last_fb'] = float((ctx['last_fb'] - c['snaps'][2]['last_fb']).abs().max())
    print('context vs rollout stream, max abs diff:', worst, flush=True)
    assert all(v < 1e-4 for v in worst.values()), worst
    bnd = ctx['fb'][torch.arange(16), ctx['att_starts'][1]]
    print('boundary feedback of trajectory 2 (row 0):', bnd[0].tolist(), 'trajectory 1 final cell', trajs[0][0][-1].tolist(), 'goal', gl[0].tolist())
    print('loss n=0', loss_with_context(model, b, None).item(), 'n=3', loss_with_context(model, b, ctx).item(), flush=True)
    a = argparse.Namespace(tag='smoke', rounds=2, iters=2, steps_per_iter=1, batch=128, micro=32, lr=1e-4, seed=0, seeds=2, chunk=512,
                           curve=1, curve_train_seeds=1, curve_test_seeds=1)
    train(a); eval_rounds(a); plot(a)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['smoke', 'train', 'eval', 'plot'])
    ap.add_argument('--tag', default='main'); ap.add_argument('--rounds', type=int, default=8); ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--steps-per-iter', type=int, default=1); ap.add_argument('--batch', type=int, default=128); ap.add_argument('--micro', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seeds', type=int, default=8); ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--curve', type=int, default=0, help='eval train/test success every this many iterations (0 = off)')
    ap.add_argument('--curve-train-seeds', type=int, default=2); ap.add_argument('--curve-test-seeds', type=int, default=8)
    a = ap.parse_args(); {'smoke': smoke, 'train': train, 'eval': eval_rounds, 'plot': plot}[a.mode](a)
