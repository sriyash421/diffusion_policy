"""Test-only best-of-N-search success curve for the offline PushT diffusion-search policy.

On the recreatable, seeded 50-episode **test** split, for each ``n in {2^0 .. 2^6}`` roll
the policy out where at every control step the search produces ``n`` candidate action
sequences and the **best-value** one (argmax verifier value) is executed:
  * n <= max_actions : ``predict_n_actions`` generates n sequential candidates (each
    conditioned on the previous ones) -- the same generation as n=max_actions, truncated.
  * n >  max_actions : rolling window -- context is the last ``max_actions`` candidates.
This is exactly ``policy.predict_action_best(obs, n_actions=n)``.

Success = episode max coverage >= threshold (max reward >= 1.0). We plot success rate vs n.

Single-checkpoint:
  python eval_search_pusht.py -c <run>/checkpoints/step_0010000.ckpt -o <out>
Watcher (evals each new step_*.ckpt as training writes it, logs curves to wandb):
  python eval_search_pusht.py --watch --run-dir <train output_dir>
"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import re
import json
import time
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
    get_split_masks_3way, get_episode_init_states)
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.env.pusht.pusht_feedback import PushTFeedbackWrapper
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv

SUCCESS_REWARD = 1.0
N_LIST = [int(2 ** k) for k in range(7)]  # 1, 2, 4, 8, 16, 32, 64
CKPT_RE = re.compile(r'step_(\d+)\.ckpt$')


def build_envs(n_envs, n_obs_steps, n_action_steps, max_steps, render_size=96):
    def env_fn():
        return MultiStepWrapper(
            PushTFeedbackWrapper(
                PushTImageEnv(legacy=False, render_size=render_size)
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps
        )
    return AsyncVectorEnv([env_fn] * n_envs)


def load_policy(checkpoint, device):
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.get('use_ema', False) and getattr(workspace, 'ema_model', None) is not None:
        policy = workspace.ema_model
    policy.to(torch.device(device))
    policy.eval()
    return policy, cfg


def get_test_states(cfg):
    """Recreatable seeded test-split reset states (identical to the dataset's test set)."""
    ds = cfg.task.dataset
    replay_buffer = ReplayBuffer.copy_from_path(
        ds.zarr_path, keys=['agent_pos', 'block_pos'])
    _, _, test_mask = get_split_masks_3way(
        n_episodes=replay_buffer.n_episodes,
        n_test_episodes=ds.n_test_episodes,
        n_val_episodes=ds.get('n_val_episodes', 0),
        seed=ds.seed,
        n_train_episodes=ds.get('n_train_episodes', None))
    return get_episode_init_states(replay_buffer, test_mask), np.nonzero(test_mask)[0]


def rollout_max_rewards(env, policy, states, device, n):
    """Roll one episode per env (reset to its state) using best-of-n search; return
    each env's max reward over the episode."""
    def make_init_fn(state):
        state = np.asarray(state, dtype=np.float64)
        def _fn(env):
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
            action = policy.predict_action_best(obs_dict, n_actions=n)['action']
        action = action.detach().cpu().numpy()
        obs, reward, done, info = env.step(action)
        done = np.all(done)
    rewards = env.call('get_attr', 'reward')
    return np.array([np.max(r) for r in rewards])


def eval_checkpoint(checkpoint, device, n_list=N_LIST, n_envs=50, max_steps=300):
    policy, cfg = load_policy(checkpoint, device)
    states, episode_idxs = get_test_states(cfg)
    n_resets = len(states)

    # Read the horizon from cfg.policy, NOT the top-level cfg: the policy is built with
    # n_action_steps + n_latency_steps, so with a nonzero latency the top-level values
    # would size the env's MultiStepWrapper differently from the chunk the policy emits.
    # The wrapper executes whatever it is handed, so that mismatch is silent -- a wrong
    # control cadence and a wrong success rate, with no exception.
    env = build_envs(n_envs, cfg.policy.n_obs_steps, cfg.policy.n_action_steps, max_steps)
    success_rate = {}
    per_n_rewards = {}
    try:
        for n in n_list:
            all_rewards = np.full(n_resets, np.nan)
            for start in tqdm.tqdm(range(0, n_resets, n_envs),
                    desc=f'n={n}', leave=False):
                chunk_states = list(states[start:start + n_envs])
                pad = n_envs - len(chunk_states)
                if pad > 0:
                    chunk_states = chunk_states + [states[0]] * pad
                t0 = time.perf_counter()
                out = rollout_max_rewards(env, policy, chunk_states, torch.device(device), n)
                dt = time.perf_counter() - t0
                all_rewards[start:start + n_envs - pad] = out[:n_envs - pad]
                tqdm.tqdm.write(f'  n={n} chunk {start}: {dt:.1f}s')
            assert not np.isnan(all_rewards).any()
            sr = float(np.mean(all_rewards >= SUCCESS_REWARD))
            success_rate[n] = sr
            per_n_rewards[n] = all_rewards
            print(f'n={n}: success_rate={sr:.3f}')
    finally:
        env.close()
        # The policy's verifier lazily forks its own pool of sim workers, which garbage
        # collection will not reap. A fresh policy is built per checkpoint, so without
        # this --watch mode leaks a full pool for every checkpoint it evaluates.
        close = getattr(policy, 'close', None)
        if close is not None:
            try:
                close()
            except Exception as e:
                print(f'warning: failed to close policy verifier pool: {e}')

    return {
        'n': list(n_list),
        'success_rate': [success_rate[n] for n in n_list],
        'episode_idxs': episode_idxs.tolist(),
        'per_n_rewards': {int(n): per_n_rewards[n].tolist() for n in n_list},
    }


def plot_curve(curve, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = np.array(curve['n'])
    sr = np.array(curve['success_rate'])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n, sr, 'o-', color='#2a78d6', lw=2)
    ax.set_xscale('log', base=2)
    ax.set_xticks(n)
    ax.set_xticklabels([str(int(x)) for x in n])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel('number of search samples (n)')
    ax.set_ylabel('success rate (coverage >= 95%)')
    ax.set_title('PushT best-of-n-search success rate (test split)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def save_outputs(curve, output_dir, checkpoint):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, 'success_curve.json'), 'w') as f:
        json.dump({'checkpoint': str(checkpoint), **curve}, f, indent=2)
    png = os.path.join(output_dir, 'success_curve.png')
    plot_curve(curve, png)
    return png


def step_from_ckpt(path):
    m = CKPT_RE.search(str(path))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Run-level result index. Training deliberately saves no "best" checkpoint (val loss
# does not identify the best task policy), so selection is done here, from real eval:
# every evaluated checkpoint is appended to success_curves.jsonl and best.json always
# names the current winner. A downstream process reads one file, not a directory tree.
# ---------------------------------------------------------------------------
CURVES_JSONL = 'success_curves.jsonl'
BEST_JSON = 'best.json'


def _curve_key(row):
    """Selection criterion: success rate at the largest n, tie-broken by the mean over
    all n (a checkpoint that is better across the whole curve beats a lucky top point)."""
    sr = row['success_rate']
    return (sr[-1], float(np.mean(sr)))


def append_curve_row(out_root, step, checkpoint, curve):
    """Append this checkpoint's result to the run-level jsonl and refresh best.json."""
    out_root = pathlib.Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    row = {
        'step': step,
        'checkpoint': str(checkpoint),
        'n': curve['n'],
        'success_rate': curve['success_rate'],
        'timestamp': time.time(),
    }
    with open(out_root.joinpath(CURVES_JSONL), 'a') as f:
        f.write(json.dumps(row) + '\n')

    rows = read_curve_rows(out_root)
    best = max(rows, key=_curve_key)
    with open(out_root.joinpath(BEST_JSON), 'w') as f:
        json.dump({
            'criterion': 'success_rate at max n, tie-broken by mean success_rate',
            'step': best['step'],
            'checkpoint': best['checkpoint'],
            'n': best['n'],
            'success_rate': best['success_rate'],
            'n_evaluated': len(rows),
        }, f, indent=2)
    return row


def read_curve_rows(out_root):
    """Every row previously written to success_curves.jsonl (skipping partial lines)."""
    path = pathlib.Path(out_root).joinpath(CURVES_JSONL)
    if not path.is_file():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # a torn final line from a preempted write; ignore it
                continue
    return rows


@click.command()
@click.option('-c', '--checkpoint', default=None, help='single checkpoint to eval')
@click.option('-o', '--output_dir', default=None, help='output dir for json/png')
@click.option('--watch', is_flag=True, help='poll run-dir/checkpoints and eval each new step_*.ckpt')
@click.option('--run-dir', default=None, help='training output dir (has checkpoints/) for --watch')
@click.option('-d', '--device', default='cuda:0')
@click.option('--n-envs', default=50)
@click.option('--max-steps', default=300)
@click.option('--poll-sec', default=60.0)
@click.option('--wandb/--no-wandb', 'use_wandb', default=True, help='log curves to wandb (watch mode)')
@click.option('--wandb-entity', default='l2sml')
@click.option('--wandb-project', default='pushT_diffusion_search')
def main(checkpoint, output_dir, watch, run_dir, device, n_envs, max_steps, poll_sec,
         use_wandb, wandb_entity, wandb_project):
    if not watch:
        assert checkpoint is not None, 'provide -c/--checkpoint or --watch'
        out = output_dir or os.path.join(os.path.dirname(checkpoint), '..', 'bon_search')
        curve = eval_checkpoint(checkpoint, device, n_envs=n_envs, max_steps=max_steps)
        png = save_outputs(curve, out, checkpoint)
        # keep one-off evals in the same run-level index the watcher maintains
        append_curve_row(out, step_from_ckpt(checkpoint), checkpoint, curve)
        print('wrote', png)
        return

    # watch mode
    assert run_dir is not None, '--watch requires --run-dir'
    ckpt_dir = pathlib.Path(run_dir).joinpath('checkpoints')
    out_root = pathlib.Path(output_dir or os.path.join(run_dir, 'bon_search'))
    run = None
    if use_wandb:
        import wandb
        run = wandb.init(entity=wandb_entity, project=wandb_project,
                         name=f'eval_{pathlib.Path(run_dir).name}',
                         job_type='eval', resume='allow')
    # Resume-safe: this watcher runs on the preemptible ckpt partition with --requeue,
    # so an in-memory-only `seen` would re-evaluate every checkpoint from scratch after
    # each preemption. Seed it from the results already on disk.
    seen = {row['step'] for row in read_curve_rows(out_root)}
    if seen:
        print(f'resuming: {len(seen)} checkpoint(s) already evaluated, skipping those')
    print(f'watching {ckpt_dir} for step_*.ckpt (poll {poll_sec}s)')
    while True:
        ckpts = sorted(ckpt_dir.glob('step_*.ckpt')) if ckpt_dir.is_dir() else []
        for ckpt in ckpts:
            step = step_from_ckpt(ckpt)
            if step is None or step in seen:
                continue
            seen.add(step)
            print(f'== evaluating {ckpt.name} (step {step}) ==')
            try:
                curve = eval_checkpoint(str(ckpt), device, n_envs=n_envs, max_steps=max_steps)
            except Exception as e:
                print(f'eval failed for {ckpt.name}: {e}')
                continue
            save_outputs(curve, str(out_root.joinpath(f'step_{step:07d}')), str(ckpt))
            append_curve_row(out_root, step, str(ckpt), curve)
            if run is not None:
                import wandb
                for nn, sr in zip(curve['n'], curve['success_rate']):
                    run.log({f'test/success_rate_n{nn}': sr}, step=step)
                table = wandb.Table(data=list(zip(curve['n'], curve['success_rate'])),
                                    columns=['n', 'success_rate'])
                run.log({'test/success_curve': wandb.plot.line(
                    table, 'n', 'success_rate',
                    title=f'best-of-n-search success (step {step})')}, step=step)
        time.sleep(poll_sec)


if __name__ == '__main__':
    main()
