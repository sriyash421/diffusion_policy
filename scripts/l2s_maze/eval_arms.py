"""Evaluate one arm on a split's eval episodes, sweeping the number of candidates N.

At every replan the arm draws N candidate chunks and the verifier scores each. The executed
chunk is chosen by --select:
  argmax  the highest-value candidate
  last    the final (N-th) candidate drawn
and `chunk[To-1 : To-1+n_action_steps]` is executed. Episodes run in lockstep; only
unfinished ones are sampled.

Arms (model letters as in scripts/l2s_maze/launch_train.sh):
  st_gaussian/bon        N i.i.d. token-0 samples from SearchPolicy A
  st_gaussian/fuzzy_bon  same from SearchPolicy B (corrupt_obs=True)
  st_gaussian/ours       SearchPolicy A predict_n_actions (sequential, sliding window > 16)
  bc_unet/bon            N samples from DiffusionUnetImagePolicy U (EMA)
  bc_unet/fuzzy_bon      same from UF (corrupt_obs=True)
  st_diffusion/ours      DiffusionTransformerSearchPolicy E predict_n_actions
  expert                 waypoint expert, closed loop (harness sanity check; no --ckpt)

Metrics per episode: success (within max_steps = 2 * expert solve length) and
progress = 1 if success else clip(1 - d_final / d0, 0, 1), d = BFS cells to the goal.

    python scripts/l2s_maze/eval_arms.py --split as_is --model st_gaussian --arm bon \
        --ckpt data/outputs/l2s_maze/as_is/l2s_A/checkpoints/latest.ckpt --select argmax
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

from expert_utils import CachedExpert, make_env
from maze_layout import DEFAULT_LAYOUT_DIR, REPO, SPLITS, bfs_to, load_layout

sys.path.insert(0, str(REPO))
from diffusion_policy.common.l2s_maze_verifier import MazeVerifier  # noqa: E402

# (model, arm) -> policy class the checkpoint must hold
ARM_POLICY = {
    ('st_gaussian', 'bon'): 'SearchPolicy',
    ('st_gaussian', 'fuzzy_bon'): 'SearchPolicy',
    ('st_gaussian', 'ours'): 'SearchPolicy',
    ('bc_unet', 'bon'): 'DiffusionUnetImagePolicy',
    ('bc_unet', 'fuzzy_bon'): 'DiffusionUnetImagePolicy',
    ('st_diffusion', 'ours'): 'DiffusionTransformerSearchPolicy',
}
MODELS = ('st_gaussian', 'bc_unet', 'st_diffusion')
ARMS = ('bon', 'fuzzy_bon', 'ours', 'expert')
SELECTS = ('argmax', 'last')
MAX_SAMPLE_ROWS = 2048   # cap on rows per i.i.d. sampling forward pass


def load_policy(ckpt, device):
    """(policy, cfg) from a training checkpoint; EMA weights if the run used EMA."""
    import dill
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    payload = torch.load(open(ckpt, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(pathlib.Path(ckpt).parent.parent))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.get('use_ema', False) else workspace.model
    policy.to(torch.device(device)).eval()
    return policy, cfg


def check_arm(policy, model, arm):
    want = ARM_POLICY.get((model, arm))
    assert want is not None, f'no arm {model}/{arm}; valid: {sorted(ARM_POLICY)}'
    name = type(policy).__name__
    assert name == want, f'{model}/{arm} needs {want}, got {name}'
    want_corrupt = arm == 'fuzzy_bon'
    assert bool(policy.corrupt_obs) == want_corrupt, \
        f'{model}/{arm} needs corrupt_obs={want_corrupt}, checkpoint has {policy.corrupt_obs}'


def tile(obs, n):
    return {k: v.repeat_interleave(n, dim=0) for k, v in obs.items()}


@torch.no_grad()
def sample_candidates(model, arm, policy, obs, n, verifier):
    """obs raw (B, To, ...) -> actions (B, n, H, 2) env units, values (B, n), in draw order."""
    B = obs['agent_pos'].shape[0]
    if arm == 'ours':
        actions, values = policy.predict_n_actions(obs, verifier, n)
        assert actions.shape[:2] == (B, n) and values.shape == (B, n)
        return actions, values
    flat = tile(obs, n)
    chunks = []
    for s in range(0, B * n, MAX_SAMPLE_ROWS):
        part = {k: v[s:s + MAX_SAMPLE_ROWS] for k, v in flat.items()}
        if model == 'st_gaussian':
            # token 0: no (action, value) context
            chunks.append(policy._predict_action(part)['action_pred'][:, 0])
        else:
            chunks.append(policy.predict_action(part)['action_pred'])
    actions = torch.cat(chunks, dim=0)
    values = verifier.get_value(flat, actions)
    return actions.reshape(B, n, *actions.shape[1:]), values.reshape(B, n)


def select_index(values, select):
    """(B, n) values -> (B,) index of the candidate to execute."""
    if select == 'argmax':
        return values.argmax(dim=1)
    return torch.full((values.shape[0],), values.shape[1] - 1, dtype=torch.long,
                      device=values.device)


def build_obs(histories, rows, To, device):
    """Last To observations of each row in `rows`, raw: image (B,To,3,H,W) in [0,1]."""
    hist = [histories[r][-To:] for r in rows]
    img = np.stack([np.stack([h['image'] for h in hs]) for hs in hist])
    return {
        'image': torch.from_numpy(img).to(device).permute(0, 1, 4, 2, 3).float() / 255.0,
        'agent_pos': torch.from_numpy(np.stack([np.stack([h['agent_pos'] for h in hs])
                                                for hs in hist])).to(device).float(),
        'goal_pos': torch.from_numpy(np.stack([np.stack([h['goal_pos'] for h in hs])
                                               for hs in hist])).to(device).float(),
    }


def finish(maze, episodes, envs, success, steps, chosen_values, n_replans):
    progress = np.empty(len(episodes))
    for k, (ep, env) in enumerate(zip(episodes, envs)):
        if success[k]:
            progress[k] = 1.0
            continue
        d = bfs_to(maze, tuple(ep['goal']))
        d_final = d[tuple(int(x) for x in np.rint(env.agent_xy))]
        progress[k] = float(np.clip(1.0 - d_final / max(ep['d0'], 1), 0.0, 1.0))
    return {
        'success': success.astype(float).tolist(), 'progress': progress.tolist(),
        'steps': steps.tolist(), 'n_replans': n_replans.tolist(),
        'mean_chosen_value': [float(np.mean(v)) if v else float('nan') for v in chosen_values],
        'final_pos': [env.agent_xy.tolist() for env in envs],
    }


def rollout(model, arm, select, policy, verifier, maze, episodes, n, device, To,
            n_action_steps):
    """Lockstep rollouts of all episodes with N = n candidates per replan."""
    envs = [make_env(maze) for _ in episodes]
    histories = []
    for env, ep in zip(envs, episodes):
        obs = env.reset(maze_index=0, start_ij=tuple(ep['start']), goal_ij=tuple(ep['goal']))
        histories.append([obs] * To)   # pad the first obs, as the dataset does
    max_steps = np.array([ep['max_steps'] for ep in episodes])
    success = np.zeros(len(episodes), dtype=bool)
    done = np.zeros(len(episodes), dtype=bool)
    steps = np.zeros(len(episodes), dtype=int)
    n_replans = np.zeros(len(episodes), dtype=int)
    chosen_values = [[] for _ in episodes]

    while not done.all():
        rows = np.flatnonzero(~done)
        obs = build_obs(histories, rows, To, device)
        actions, values = sample_candidates(model, arm, policy, obs, n, verifier)
        idx = select_index(values, select)
        chunk = actions[torch.arange(len(rows), device=actions.device), idx]
        execute = chunk[:, To - 1:To - 1 + n_action_steps].cpu().numpy()
        picked = values.gather(1, idx[:, None])[:, 0].cpu().numpy()
        for i, r in enumerate(rows):
            n_replans[r] += 1
            chosen_values[r].append(float(picked[i]))
            for a in execute[i]:
                obs_r, _, reached = envs[r].step(a)
                histories[r].append(obs_r)
                steps[r] += 1
                if reached:
                    success[r] = True
                if success[r] or steps[r] >= max_steps[r]:
                    done[r] = True
                    break
            histories[r] = histories[r][-To:]
    return finish(maze, episodes, envs, success, steps, chosen_values, n_replans)


def rollout_expert(maze, episodes):
    """Closed-loop waypoint expert under the same max_steps and metrics."""
    expert = CachedExpert(maze)
    envs, success, steps = [], np.zeros(len(episodes), bool), np.zeros(len(episodes), int)
    for k, ep in enumerate(episodes):
        ok, n_steps, _ = expert.rollout(tuple(ep['start']), tuple(ep['goal']), ep['max_steps'])
        success[k], steps[k] = ok, n_steps
        env = make_env(maze)
        env.reset(maze_index=0, start_ij=tuple(ep['start']), goal_ij=tuple(ep['goal']))
        env.agent_xy = expert.env.agent_xy.copy()
        envs.append(env)
    return finish(maze, episodes, envs, success, steps, [[] for _ in episodes],
                  np.zeros(len(episodes), int))


def summarize(res):
    out = {}
    for key in ('success', 'progress'):
        x = np.asarray(res[key])
        out[key] = float(x.mean())
        out[f'{key}_se'] = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=SPLITS, required=True)
    ap.add_argument('--model', choices=MODELS, default=None)
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--select', choices=SELECTS, default='argmax')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--ns', default='1,2,4,8,16,32,64')
    ap.add_argument('--n-episodes', type=int, default=None, help='first k episodes (default all)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--layout-dir', default=str(DEFAULT_LAYOUT_DIR))
    ap.add_argument('--out-root', default=str(REPO / 'data/l2s_maze_eval'))
    args = ap.parse_args()
    os.chdir(REPO)   # checkpoint configs hold repo-relative paths (dataset, maze_path)

    maze, layout = load_layout(args.layout_dir)
    manifest = json.loads((REPO / f'data/l2s_maze/{args.split}/eval_episodes.json').read_text())
    assert manifest['layout_sha'] == layout['sha']
    episodes = manifest['episodes'][:args.n_episodes]
    meta = {'split': args.split, 'model': args.model, 'arm': args.arm, 'select': args.select,
            'ckpt': args.ckpt, 'seed': args.seed, 'n_episodes': len(episodes),
            'layout_sha': layout['sha']}

    if args.arm == 'expert':
        out_dir = pathlib.Path(args.out_root) / args.split / 'expert'
        out_dir.mkdir(parents=True, exist_ok=True)
        res = rollout_expert(maze, episodes)
        record = {**meta, 'model': None, 'select': None, 'n': None, **summarize(res),
                  'episodes': res}
        (out_dir / 'expert.json').write_text(json.dumps(record))
        print(f"expert: success {record['success']:.3f}  progress {record['progress']:.3f}",
              flush=True)
        return

    assert args.model and args.ckpt, '--model and --ckpt are required for model arms'
    tag = f'{args.model}_{args.arm}'
    out_dir = pathlib.Path(args.out_root) / args.split / tag / args.select
    out_dir.mkdir(parents=True, exist_ok=True)
    policy, cfg = load_policy(args.ckpt, args.device)
    check_arm(policy, args.model, args.arm)
    assert cfg.maze_split == args.split, f'checkpoint trained on {cfg.maze_split}'
    To, n_action_steps = policy.n_obs_steps, policy.n_action_steps
    verifier = MazeVerifier(maze=maze, device=args.device, start_offset=To - 1)

    for n in (int(x) for x in args.ns.split(',')):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        t0 = time.time()
        res = rollout(args.model, args.arm, args.select, policy, verifier, maze, episodes, n,
                      args.device, To, n_action_steps)
        record = {**meta, 'n': n, 'train_cfg_name': cfg.name, 'seconds': time.time() - t0,
                  **summarize(res), 'episodes': res}
        (out_dir / f'N{n}.json').write_text(json.dumps(record))
        print(f"{args.split} {tag} {args.select} N={n}: "
              f"success {record['success']:.3f}±{record['success_se']:.3f}"
              f"  progress {record['progress']:.3f}±{record['progress_se']:.3f}"
              f"  ({record['seconds']:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
