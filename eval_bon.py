"""Best-of-N evaluation for PushT.

For every held-out test reset state, roll the policy out ``--n-samples`` times (the
diffusion sampler is stochastic, the env is not) and report how the result improves
as you draw more samples:

  * best-of-n  : mean over resets of ``max(reward_1..reward_n)`` -- x is #samples
  * regret     : mean over resets of ``1 - mean(r_1..r_x)`` -- the gap to the best
                 achievable, since reward and success both max out at 1

Success is ``max_reward >= 1.0``, which by PushTEnv's reward
(``clip(coverage / success_threshold, 0, 1)``) means coverage >= 95%. Both reward and
success are therefore bounded above by ``MAX_RETURN = 1.0``.

Usage:
python eval_bon.py -c results/<run>/checkpoints/latest.ckpt -o results/<run>/bon
python eval_bon.py -c ... -o ... --n-samples 64 --n-envs 50
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import json
import math
import pathlib
import click
import dill
import hydra
import numpy as np
import torch
import tqdm

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.pusht_image_dataset import (
    get_split_masks, get_episode_init_states)
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.env.pusht.pusht_feedback import PushTFeedbackWrapper
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv

SUCCESS_REWARD = 1.0
# reward is clip(coverage / success_threshold, 0, 1) and success is 0/1, so the best
# attainable value of either metric is 1.0. Regret is measured against this.
MAX_RETURN = 1.0


def build_envs(n_envs, n_obs_steps, n_action_steps, max_steps, render_size=96):
    def env_fn():
        return MultiStepWrapper(
            PushTFeedbackWrapper(
                # legacy=False so reset_to_state round-trips a recorded state exactly
                PushTImageEnv(legacy=False, render_size=render_size)
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps
        )
    return AsyncVectorEnv([env_fn] * n_envs)


def run_chunk(env, policy, states, device):
    """Roll out one state per env; return each env's max reward."""
    def make_init_fn(state):
        state = np.asarray(state, dtype=np.float64)
        def _fn(env):
            # via .unwrapped: gym wrappers forward attribute reads but not writes
            env.unwrapped.reset_to_state = state
        return _fn

    env.call_each('run_dill_function',
        args_list=[(dill.dumps(make_init_fn(s)),) for s in states])

    obs = env.reset()
    policy.reset()
    done = False
    while not done:
        obs_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to(device=device))
        with torch.no_grad():
            action = policy.predict_action(obs_dict)['action'].detach().cpu().numpy()
        obs, reward, done, info = env.step(action)
        done = np.all(done)
    rewards = env.call('get_attr', 'reward')
    return np.array([np.max(r) for r in rewards])


def expected_best_of_n(values):
    """E[max of x draws] for x = 1..N, exactly, via order statistics.

    A single arbitrary draw order gives a curve whose shape is an artifact of that
    order. Instead average over all orderings: for values sorted ascending, drawing x
    of N without replacement has max == v_(i) with probability C(i-1, x-1) / C(N, x),
    so E[max of x] = sum_i v_(i) * C(i-1, x-1) / C(N, x).

    Args:
        values: (N,) sample outcomes for one reset.
    Returns:
        (N,) expected best-of-x, monotone non-decreasing; [0] == mean, [-1] == max.
    """
    v = np.sort(np.asarray(values, dtype=np.float64))
    N = len(v)
    out = np.empty(N)
    for x in range(1, N + 1):
        # weights over the sorted values; only i >= x can be the max of x draws
        i = np.arange(1, N + 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            w = np.array([math.comb(int(ii) - 1, x - 1) if ii >= x else 0
                          for ii in i], dtype=np.float64)
        w /= math.comb(N, x)
        out[x - 1] = float(np.dot(v, w))
    return out


def compute_curves(rewards):
    """rewards: (n_resets, n_samples) -> best-of-n and simple-regret curves.

    Both curves are order-independent: they are expectations over which x of the N
    samples you happened to draw, not a running statistic of one arbitrary order.
    """
    n_resets, n_samples = rewards.shape
    success = (rewards >= SUCCESS_REWARD).astype(np.float64)

    # per-reset expected best-of-x, then average over resets
    bon_r = np.stack([expected_best_of_n(row) for row in rewards])      # (R, N)
    bon_s = np.stack([expected_best_of_n(row) for row in success])      # (R, N)

    return {
        'n': np.arange(1, n_samples + 1),
        # best-of-n: y = E[best reward/success among x samples]
        'bon_reward': bon_r.mean(axis=0),
        'bon_success': bon_s.mean(axis=0),
        'bon_reward_se': bon_r.std(axis=0) / math.sqrt(n_resets),
        'bon_success_se': bon_s.std(axis=0) / math.sqrt(n_resets),
        # simple regret against the max possible (1.0): what you still lose after
        # taking the best of x samples. Decays to 0 once every reset has a success.
        'regret_reward': (MAX_RETURN - bon_r).mean(axis=0),
        'regret_success': (MAX_RETURN - bon_s).mean(axis=0),
        # single-sample reference (== bon at x=1)
        'mean_reward': bon_r[:, 0].mean(),
        'mean_success': bon_s[:, 0].mean(),
    }


def plot_curves(c, rewards, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    BLUE, YELLOW = '#2a78d6', '#eda100'
    INK, MUTED, GRID = '#0b0b0b', '#52514e', '#dedddb'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK,
                         'figure.facecolor': 'white', 'axes.grid': True, 'grid.color': GRID,
                         'grid.linewidth': .5, 'axes.axisbelow': True, 'legend.frameon': False})
    n = c['n']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.plot(n, c['bon_reward'], lw=2, color=BLUE, label='best-of-n reward')
    ax.fill_between(n, c['bon_reward'] - c['bon_reward_se'], c['bon_reward'] + c['bon_reward_se'],
                    color=BLUE, alpha=.18, lw=0)
    ax.axhline(c['mean_reward'], color=MUTED, ls='--', lw=1)
    ax.annotate('single-sample mean %.3f' % c['mean_reward'], (n[-1], c['mean_reward']),
                ha='right', va='bottom', color=MUTED, fontsize=8)
    ax.set_xlabel('n = independent full episodes (post-hoc oracle max)')
    ax.set_ylabel('best reward so far')
    ax.set_title('Oracle best-of-n over EPISODES (not deployable; cf. eval_search_pusht)'); ax.legend(loc='lower right', fontsize=8)

    ax = axes[1]
    ax.plot(n, c['bon_success'], lw=2, color=YELLOW, label='best-of-n success')
    ax.fill_between(n, c['bon_success'] - c['bon_success_se'], c['bon_success'] + c['bon_success_se'],
                    color=YELLOW, alpha=.18, lw=0)
    ax.axhline(c['mean_success'], color=MUTED, ls='--', lw=1)
    ax.annotate('single-sample mean %.3f' % c['mean_success'], (n[-1], c['mean_success']),
                ha='right', va='bottom', color=MUTED, fontsize=8)
    ax.set_xlabel('n = independent full episodes (post-hoc oracle max)')
    ax.set_ylabel('success rate so far (coverage >= 95%)')
    ax.set_ylim(-0.03, 1.03)
    ax.set_title('Oracle best-of-n over EPISODES (not deployable; cf. eval_search_pusht)'); ax.legend(loc='lower right', fontsize=8)

    ax = axes[2]
    ax.plot(n, c['regret_reward'], lw=2, color=BLUE, label='reward regret')
    ax.plot(n, c['regret_success'], lw=2, color=YELLOW, label='success regret')
    ax.set_xlabel('number of episodes (x)')
    ax.set_ylabel('regret: 1.0 - mean over first x')
    ax.set_ylim(bottom=0)
    ax.set_title('Regret vs max possible (1.0)'); ax.legend(loc='upper right', fontsize=8)

    # sample counts are integers
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle('PushT best-of-n over %d test resets x %d samples' % rewards.shape,
                 fontsize=12, y=1.0, color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--n-samples', default=64, help='rollouts per reset state')
@click.option('--n-resets', default=None, type=int, help='test resets to use (default: all)')
@click.option('--n-envs', default=50, help='parallel envs')
@click.option('--max-steps', default=300)
@click.option('--seed', default=0)
def main(checkpoint, output_dir, device, n_samples, n_resets, n_envs, max_steps, seed):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.to(torch.device(device))
    policy.eval()

    # the same held-out test resets the runner uses
    ds_cfg = cfg.task.dataset
    replay_buffer = ReplayBuffer.copy_from_path(
        ds_cfg.zarr_path, keys=['agent_pos', 'block_pos'])
    _, test_mask = get_split_masks(
        n_episodes=replay_buffer.n_episodes,
        n_test_episodes=ds_cfg.n_test_episodes,
        seed=ds_cfg.seed,
        n_train_episodes=ds_cfg.get('n_train_episodes', None))
    states = get_episode_init_states(replay_buffer, test_mask)
    episode_idxs = np.nonzero(test_mask)[0]
    if n_resets is not None:
        states = states[:n_resets]
        episode_idxs = episode_idxs[:n_resets]
    n_resets = len(states)

    # one job per (reset, sample); the env is deterministic, so all variation comes
    # from the diffusion sampler
    jobs = [(r, s) for r in range(n_resets) for s in range(n_samples)]
    print(f'{n_resets} resets x {n_samples} samples = {len(jobs)} rollouts '
          f'on {n_envs} envs ({math.ceil(len(jobs)/n_envs)} chunks)')

    env = build_envs(n_envs, cfg.n_obs_steps, cfg.n_action_steps, max_steps)
    rewards = np.full((n_resets, n_samples), np.nan)
    try:
        for start in tqdm.tqdm(range(0, len(jobs), n_envs), desc='best-of-n'):
            chunk = jobs[start:start + n_envs]
            chunk_states = [states[r] for r, _ in chunk]
            # pad the last chunk so every env has a state; padded results are dropped
            pad = n_envs - len(chunk_states)
            if pad > 0:
                chunk_states = chunk_states + [states[0]] * pad
            out = run_chunk(env, policy, chunk_states, torch.device(device))
            for (r, s), value in zip(chunk, out[:len(chunk)]):
                rewards[r, s] = value
    finally:
        env.close()

    assert not np.isnan(rewards).any(), 'some rollouts did not report a reward'
    curves = compute_curves(rewards)

    np.savez(os.path.join(output_dir, 'bon_rewards.npz'),
             rewards=rewards, episode_idxs=episode_idxs,
             **{k: v for k, v in curves.items()})
    plot_curves(curves, rewards, os.path.join(output_dir, 'bon_curves.png'))

    summary = {
        'checkpoint': checkpoint,
        'n_resets': int(n_resets),
        'n_samples': int(n_samples),
        'mean_reward_single_sample': float(curves['mean_reward']),
        'mean_success_single_sample': float(curves['mean_success']),
        'best_of_n_reward': float(curves['bon_reward'][-1]),
        'best_of_n_success': float(curves['bon_success'][-1]),
        'regret_reward_at_n': float(curves['regret_reward'][-1]),
        'regret_success_at_n': float(curves['regret_success'][-1]),
    }
    with open(os.path.join(output_dir, 'bon_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print('wrote', os.path.join(output_dir, 'bon_curves.png'))


if __name__ == '__main__':
    main()
