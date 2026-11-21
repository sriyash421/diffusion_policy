"""Per-start eval of a context-rounds or DAgger DT run (train_pusht_dt_rounds / _dagger): the max
coverage of round (stage) n's scored attempt from every train / test start. The DAgger arm is
trained on its own context, so `own` is its protocol.

  replay: from each start pi_0 rolls out, then pi_1 conditioned on it, ...; pi_n's rollout is the
          one scored for round n (the context round n was trained on). One pass scores all rounds.
  own:    pi_n makes n attempts in its own context, then its (n+1)-th is scored.

Attempts run on eval_dt_pusht's harness (rollout_attempt). Writes
<run>/rounds_eval/<split>_<mode>.npz: rounds (R,), episode_idxs (E,), max_coverage / success /
n_steps (R, E).

    python eval_dt_rounds_pusht.py <run_dir> [--split train --split test] [--mode replay --mode own]
"""
import pathlib
import re

import click
import numpy as np
import torch

from diffusion_policy.env.pusht.pusht_feedback import build_parallel
from diffusion_policy.env.pusht.veritas.tracker import WAYPOINT_PLAN_DIRS, load_split_plans
from eval_dt_pusht import FEEDBACK_VALUE, load_eval_policy, rollout_attempt
from eval_search_pusht import get_split_states

KEYS = ('max_coverage', 'success', 'n_steps')


def round_ckpts(run_dir):
    """rounds/round_<n>.ckpt (rounds arm) or stages/pi_k<n>.ckpt (DAgger arm), in n order."""
    ckpts = list((run_dir / 'rounds').glob('round_*.ckpt')) + list((run_dir / 'stages').glob('pi_k*.ckpt'))
    ckpts = sorted(ckpts, key=lambda p: int(re.search(r'(\d+)\.ckpt$', p.name).group(1)))
    assert ckpts, f'no rounds/round_*.ckpt or stages/pi_k*.ckpt under {run_dir}'
    return ckpts


def eval_replay(ckpts, env, states, plans, device, max_steps, seed):
    ctx, out = [None] * len(states), []
    for n, ck in enumerate(ckpts):
        policy, _ = load_eval_policy(str(ck), device)
        torch.manual_seed(seed + n); np.random.seed(seed + n)
        policy.reset()
        policy.prefill(ctx, device)
        res = rollout_attempt(env, policy, states, plans, device, max_steps)
        ctx = [policy.append_context(c, r) for c, r in zip(ctx, policy.context_records())]
        out.append(res)
        print(f'replay round {n}: mean max coverage {np.mean(res["max_coverage"]):.3f}')
    return out


def eval_own(ckpts, env, states, plans, device, max_steps, seed):
    out = []
    for n, ck in enumerate(ckpts):
        policy, _ = load_eval_policy(str(ck), device)
        torch.manual_seed(seed + n); np.random.seed(seed + n)
        policy.reset()
        for _ in range(n + 1):
            res = rollout_attempt(env, policy, states, plans, device, max_steps)
        out.append(res)
        print(f'own round {n}: mean max coverage {np.mean(res["max_coverage"]):.3f}')
    return out


@click.command()
@click.argument('run_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--split', 'splits', multiple=True, default=('train', 'test'))
@click.option('--mode', 'modes', multiple=True, default=('replay', 'own'))
@click.option('--device', default='cuda:0')
@click.option('--max-steps', default=300)
@click.option('--seed', default=42)
@click.option('--max-episodes', default=None, type=int)
def main(run_dir, splits, modes, device, max_steps, seed, max_episodes):
    run_dir = pathlib.Path(run_dir).resolve()
    ckpts = round_ckpts(run_dir)
    _, cfg = load_eval_policy(str(ckpts[0]), device)
    out_dir = run_dir / 'rounds_eval'
    out_dir.mkdir(exist_ok=True)
    for split in splits:
        states, idxs = get_split_states(cfg, split, run_dir=run_dir)
        if max_episodes:
            states, idxs = states[:max_episodes], idxs[:max_episodes]
        plans = load_split_plans(WAYPOINT_PLAN_DIRS[FEEDBACK_VALUE], idxs, split, FEEDBACK_VALUE)
        env = build_parallel(n_envs=len(states), use_async=True, n_obs_steps=1, n_action_steps=1,
                             max_episode_steps=max_steps, legacy=False, tracker=True)
        try:
            for mode in modes:
                fn = {'replay': eval_replay, 'own': eval_own}[mode]
                res = fn(ckpts, env, list(states), list(plans), device, max_steps, seed)
                np.savez(out_dir / f'{split}_{mode}.npz', rounds=np.arange(len(res)),
                         episode_idxs=np.asarray(idxs),
                         **{k: np.stack([r[k] for r in res]) for k in KEYS})
                print(f'wrote {out_dir / f"{split}_{mode}.npz"}')
        finally:
            env.close()


if __name__ == '__main__':
    main()
