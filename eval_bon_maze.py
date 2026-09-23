#!/usr/bin/env python3
"""Best-of-N evaluation for the procgen maze.

The maze counterpart of `eval_bon.py`. The CURVE MATH IS IMPORTED from it, not reforked --
`expected_best_of_n` / `compute_curves` are task-independent, and two copies would drift.
What is task-specific is only the rollout: `ProcgenMazeEnv` is pure numpy and fast, so
episodes run in-process, without gym wrappers or a vectorized pool.

WHICH EPISODES. The test episodes come from the run's own `split_file`, resolved through
the checkpoint's config, so an arm is never scored on an episode it trained on. Each test
episode contributes its (start, goal); the maze layout is fixed.

REWARD. The env's own reward is -1/0, which carries no partial credit and would make the
best-of-n curve a step function. Here a rollout is scored by how much of the corridor
distance it closed:

    reward = clip(1 - d_final / d_initial, 0, 1)      d = BFS steps to the goal

so reward == 1 exactly when the goal is reached (`SUCCESS_REWARD`), matching the bound
`eval_bon` assumes, and a policy that gets halfway scores 0.5 rather than 0.

ALL SPREAD COMES FROM THE POLICY: the env is deterministic and every sample of a given
episode starts from the same state, so `--n-samples` above 1 is only meaningful for a
policy that samples. A deterministic policy's draws are identical and its curve is flat.

Usage:
    python eval_bon_maze.py -c data/outputs/maze1_bc_unet/checkpoints/latest.ckpt \\
        -o data/outputs/maze1_bc_unet/bon --n-samples 16
"""
import sys

if __name__ == '__main__':
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import json
import pathlib

import click
import dill
import hydra
import numpy as np
import torch
import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / 'maze_data_scripts'))

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from eval_bon import compute_curves, plot_curves, SUCCESS_REWARD
from procgen_maze_env import ProcgenMazeEnv, bfs_distances, make_maze_split


def load_test_episodes(cfg, split='test'):
    """(start, goal) per episode of `split`, read from the run's own split manifest."""
    import zarr
    ds_cfg = cfg.task.dataset
    split_file = ds_cfg.get('split_file', None)
    if split_file is None:
        raise ValueError('this run has no split_file; there is no held-out set to score')
    manifest = json.loads(pathlib.Path(
        hydra.utils.to_absolute_path(str(split_file))).read_text())
    zarr_path = hydra.utils.to_absolute_path(manifest['zarr_path'])
    root = zarr.open(zarr_path, mode='r')
    ends = np.asarray(root['meta/episode_ends'])
    starts = np.concatenate([[0], ends[:-1]])
    agent = np.asarray(root['data/obs/agent_pos'])
    goals = np.asarray(root['data/obs/goal_pos'])
    meta = dict(root['meta'].attrs)
    maze = make_maze_split(meta['maze_pool_size'], seed=meta['maze_pool_seed'])[
        meta['single_maze']]
    eps = [(agent[starts[e]], goals[ends[e] - 1]) for e in manifest[split]]
    return eps, maze, manifest


def rollout_batch(policy, maze, jobs, device, n_obs_steps, max_steps, step_scale, seed):
    """Roll out a whole batch of (start, goal) jobs in lockstep.

    ONE policy call serves the whole batch. The env is pure numpy and costs nothing next
    to a diffusion sampler forward, so the batch dimension is what the eval time is made
    of: scoring one rollout at a time spends ~40s of GPU per rollout to step 23x23 cells.

    Returns (reward, success, steps) arrays, each (len(jobs),).
    """
    envs = [ProcgenMazeEnv([maze], step_scale=step_scale, seed=seed) for _ in jobs]
    fields, d0 = [], np.empty(len(jobs))
    history = []
    for i, (start, goal) in enumerate(jobs):
        field = bfs_distances(maze, np.rint(goal).astype(int))
        fields.append(field)
        d0[i] = field.get(tuple(np.rint(start).astype(int)), 1)
        obs = envs[i].reset(maze_index=0, start_ij=start, goal_ij=goal)
        history.append([obs] * n_obs_steps)

    policy.reset()
    # `done` is "this rollout has stopped", `success` is "it reached the goal". Keeping
    # them separate is what lets the loop terminate: a rollout that exhausts max_steps is
    # done but not successful, and one that succeeds stops incrementing its step count.
    done = np.zeros(len(jobs), dtype=bool)
    success = np.zeros(len(jobs), dtype=bool)
    steps = np.zeros(len(jobs), dtype=int)
    while not done.all():
        obs_dict = {
            k: torch.from_numpy(np.stack([
                np.stack([h[k] for h in hist[-n_obs_steps:]]) for hist in history])
            ).to(device=device, dtype=torch.float32)
            for k in ('agent_pos', 'goal_pos')}
        # (B, T, H, W, C) uint8 -> (B, T, C, H, W) float, the layout the encoder expects
        img = np.stack([np.stack([h['image'] for h in hist[-n_obs_steps:]])
                        for hist in history])
        obs_dict['image'] = torch.from_numpy(
            img.transpose(0, 1, 4, 2, 3)).to(device=device, dtype=torch.float32) / 255.0
        with torch.no_grad():
            actions = policy.predict_action(obs_dict)['action'].detach().cpu().numpy()

        for i in range(len(jobs)):
            if done[i]:
                # A finished rollout still needs an observation for the batch, so it
                # holds its last one; its env is never stepped again.
                history[i].append(history[i][-1])
                continue
            for a in actions[i]:
                _obs, _r, reached = envs[i].step(a)
                history[i].append(_obs)
                steps[i] += 1
                if reached:
                    success[i] = True
                if success[i] or steps[i] >= max_steps:
                    done[i] = True
                    break

    reward = np.empty(len(jobs))
    for i in range(len(jobs)):
        d1 = fields[i].get(tuple(np.rint(envs[i].agent_xy).astype(int)), d0[i])
        reward[i] = 1.0 if success[i] else float(
            np.clip(1.0 - d1 / max(d0[i], 1), 0.0, 1.0))
    return reward, success.astype(float), steps


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--n-samples', default=16, help='rollouts per test episode')
@click.option('--n-resets', default=None, type=int, help='test episodes (default: all)')
@click.option('--max-steps', default=600)
@click.option('--batch-size', default=50, help='rollouts stepped in lockstep per policy call')
@click.option('--seed', default=0)
@click.option('--split', type=click.Choice(['val', 'test']), default='test')
def main(checkpoint, output_dir, device, n_samples, n_resets, max_steps,
         batch_size, seed, split):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.get('use_ema', False):
        policy = workspace.ema_model
    policy.to(torch.device(device))
    policy.eval()

    eps, maze, manifest = load_test_episodes(cfg, split)
    if n_resets is not None:
        eps = eps[:n_resets]
    print(f'[INFO] scoring the {split} split: {len(eps)} episodes x {n_samples} samples, '
          f'split_file={cfg.task.dataset.get("split_file", None)}')

    step_scale = cfg.task.dataset.get('step_scale', 0.25)
    rewards = np.full((len(eps), n_samples), np.nan)
    successes = np.zeros_like(rewards)
    # one job per (episode, sample); the env is deterministic, so all spread is the policy
    pairs = [(r, s) for r in range(len(eps)) for s in range(n_samples)]
    for start_i in tqdm.tqdm(range(0, len(pairs), batch_size), desc='best-of-n'):
        chunk = pairs[start_i:start_i + batch_size]
        rew, suc, _ = rollout_batch(
            policy, maze, [eps[r] for r, _ in chunk], device, cfg.n_obs_steps,
            max_steps, step_scale, seed)
        for (r, s), rr, ss in zip(chunk, rew, suc):
            rewards[r, s] = rr
            successes[r, s] = ss

    c = compute_curves(rewards)
    out = {
        'checkpoint': str(checkpoint), 'split': split,
        'split_file': str(cfg.task.dataset.get('split_file', None)),
        'split_checksum': manifest.get('checksum'),
        'n_resets': len(eps), 'n_samples': n_samples,
        'mean_reward': float(c['mean_reward']),
        'mean_success': float(successes.mean()),
        'best_of_n_success': float((successes.max(axis=1)).mean()),
        'curves': {k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
                   for k, v in c.items()},
        'rewards': rewards.tolist(),
        'successes': successes.tolist(),
    }
    (pathlib.Path(output_dir) / 'bon.json').write_text(json.dumps(out, indent=2))
    try:
        plot_curves(c, rewards, str(pathlib.Path(output_dir) / 'bon.png'))
    except Exception as exc:                      # a failed plot must not lose the json
        print(f'[WARN] plot failed: {exc}')
    print(f"mean success (n=1): {out['mean_success']:.3f}   "
          f"best-of-{n_samples} success: {out['best_of_n_success']:.3f}   "
          f"mean reward: {out['mean_reward']:.3f}")
    print(f"wrote {output_dir}/bon.json")


if __name__ == '__main__':
    main()
