"""V0 pre-flight for `armTd`: what the candidate spreads ACTUALLY are, per control step.

`armTn` divides the two verifier terms by 13.6 and 52.1 px -- within-step candidate spreads
measured once, offline, over 8 candidates x 16 dataset states. `armTd` replaces those with
the spread measured at each control step. Whether that is worth doing is an empirical
question this script answers before any GPU time goes into a sweep:

  1. How far are the realized spreads from the two constants? That gap IS the premise.
  2. Is sd(d_T->goal) bimodal -- a spike at ~0 (no candidate touched the block) and a bulk
     well above the ARM_TD_EPS_PX floor? armTd amplifies whatever spread it finds, so a
     filled-in middle means it would inflate physically meaningless displacements into
     full-magnitude z-scores. That is the failure mode the floor exists to bound, and it is
     the one thing that could invalidate the whole idea.
  3. What fraction of steps fall below the floor (where armTd degenerates to "arm decides")?

  python scripts/armtd_preflight.py -c <ckpt> [-n 8] [--episodes 10] [--max-steps 300]
"""
if __name__ == "__main__":
    import sys, os, pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import json
import click
import dill
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.pusht.pusht_verifier import (
    ARM_TD_EPS_PX, T_GOAL_SPREAD, ARM_T_SPREAD)
from eval_search_pusht import (
    load_policy, build_envs, get_split_states, _episode_seed)


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float('nan')


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--out', default=None, help='write the raw per-step spreads here (.json)')
@click.option('-n', 'n_actions', default=8, type=int)
@click.option('--episodes', default=10, type=int)
@click.option('--max-steps', default=300, type=int)
@click.option('--device', default='cuda:0')
@click.option('--seed', default=None, type=int)
def main(checkpoint, out, n_actions, episodes, max_steps, device, seed):
    policy, cfg = load_policy(checkpoint, device)
    seed = cfg.training.seed if seed is None else seed
    # returns (states, idxs) -- unpack it; slicing the TUPLE silently hands the env
    # a list containing the whole state array plus the index array.
    states, _ = get_split_states(cfg, 'test')
    states = list(states)[:episodes]
    env = build_envs(len(states), cfg.policy.n_obs_steps, cfg.policy.n_action_steps, max_steps)
    try:
        seeder = getattr(policy, 'set_sample_seeds', None)
        if seeder is not None:
            seeder([_episode_seed(seed, n_actions, i) for i in range(len(states))])
        torch.manual_seed(seed); np.random.seed(seed)
        # the same reset-injection rollout_max_rewards uses; hand-rolling it got the
        # closure binding wrong and every worker died on the dill'd lambda.
        def make_init_fn(state):
            state = np.asarray(state, dtype=np.float64)
            def _fn(e):
                e.unwrapped.reset_to_state = state
            return _fn
        env.call_each('run_dill_function',
                      args_list=[(dill.dumps(make_init_fn(st)),) for st in states])
        obs = env.reset()
        policy.reset()
        sd_t, sd_a, done = [], [], np.zeros(len(states), dtype=bool)
        for step_i in range(max_steps):
            obs_d = dict_apply(obs, lambda x: torch.from_numpy(x).to(device=device))
            with torch.no_grad():
                res = policy.predict_n_actions(
                    obs_d, verifier=policy.verifier, n_actions=n_actions,
                    return_scores=True, return_terms=True)
            terms = res[-1]                                   # (B, n, 2) raw px
            assert terms is not None, 'policy returned no terms; is this a PushT policy?'
            s = terms.double().std(dim=1, unbiased=False).cpu().numpy()   # (B, 2)
            keep = ~done
            sd_t.extend(s[keep, 0].tolist()); sd_a.extend(s[keep, 1].tolist())
            best = res[2].argmax(dim=1)
            acts = res[0][torch.arange(res[0].shape[0]), best]
            To = cfg.policy.n_obs_steps
            Ta = cfg.policy.n_action_steps
            obs, _, dn, _ = env.step(acts[:, To - 1:To - 1 + Ta].cpu().numpy())
            done |= np.asarray(dn, dtype=bool)
            if done.all():
                break
        sd_t, sd_a = np.array(sd_t), np.array(sd_a)
    finally:
        env.close(); policy.close()

    print(f'\n{len(sd_t)} live control steps, n={n_actions}, {len(states)} episodes\n')
    print(f'{"term":>10} {"const":>7} {"median":>8} {"mean":>8} {"p10":>7} {"p90":>7} {"max":>8}')
    for name, arr, const in (('d_T->goal', sd_t, T_GOAL_SPREAD), ('d_arm->T', sd_a, ARM_T_SPREAD)):
        print(f'{name:>10} {const:>7.1f} {np.median(arr):>8.3f} {arr.mean():>8.3f} '
              f'{_pct(arr,10):>7.3f} {_pct(arr,90):>7.3f} {arr.max():>8.2f}')

    print(f'\nsd(d_T->goal) distribution -- the bimodality ARM_TD_EPS_PX={ARM_TD_EPS_PX} assumes:')
    edges = [0, 1e-6, 1e-4, ARM_TD_EPS_PX, 1e-2, 1e-1, 1.0, 10.0, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = int(((sd_t >= lo) & (sd_t < hi)).sum())
        bar = '#' * int(60 * k / max(len(sd_t), 1))
        print(f'  [{lo:>8.0e}, {hi:>8.0e}) {k:>6} {100*k/max(len(sd_t),1):>5.1f}% {bar}')
    below = float((sd_t < ARM_TD_EPS_PX).mean())
    middle = float(((sd_t >= ARM_TD_EPS_PX) & (sd_t < 0.1)).mean())
    print(f'\n  below the floor (armTd -> arm term decides): {100*below:.1f}%')
    print(f'  in [floor, 0.1) px -- the DANGER band, amplified but physically meaningless: '
          f'{100*middle:.1f}%')
    print('  -> bimodal (danger band small) means the floor is well placed; a filled-in '
          'middle means armTd would inflate noise and the floor should rise.')

    if out:
        json.dump({'sd_t_goal': sd_t.tolist(), 'sd_arm_t': sd_a.tolist(),
                   'n_actions': n_actions, 'checkpoint': checkpoint,
                   'eps_px': ARM_TD_EPS_PX}, open(out, 'w'))
        print(f'\nraw spreads -> {out}')


if __name__ == '__main__':
    main()
