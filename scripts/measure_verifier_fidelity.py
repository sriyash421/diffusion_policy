"""P3.1 — measure how wrong the verifier's simulated rollout is. READ-ONLY.

The verifier scores a candidate by resetting a fresh PushT sim to a 5-dim
``[agent_pos, block_pose]`` state and replaying the action chunk. Two things that state
cannot carry:

* **velocity** — the reset agent is always stationary, while the real agent arrives at the
  decision point carrying momentum;
* **the settle step** — ``PushTEnv._set_state`` runs one extra ``space.step`` to make the
  written pose take effect, which the real trajectory never experienced.

This script quantifies the resulting error before any training code changes, which is what
P3.2 is gated on. It changes nothing: it only replays recorded episodes.

At each sampled decision point *t* it compares three ways of reaching the same future:

  (c) TRUE     replay from the episode start to t, then continue -- exact velocity, the
               reference. The env has no RNG on the step path (the only RNG,
               ``pusht_env.py:98-102``, is bypassed whenever ``reset_to_state`` is set), so
               this is exact and repeatable.
  (a) TODAY    reset to state_t, replay -- zero velocity + one settle step.
  (b) WARMUP   reset to state_{t-1}, step a_{t-1} once, snap positions back to state_t
               keeping the velocity the warm-up produced, replay. This is the P3.2 proposal.

Reported per variant against (c): final block-pose position error in px, angle error in
degrees, and mean keypoint distance -- the last being the quantity the verifier's value is
actually built from, so it is the one that decides whether this matters.

    python scripts/measure_verifier_fidelity.py --n-episodes 20 --per-episode 6
"""
if __name__ == "__main__":
    import sys, os, pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import collections

import click
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.pusht_image_dataset import get_split_masks_3way
from diffusion_policy.env.pusht.pusht_env import PushTEnv
from diffusion_policy.env.pusht.feedback_util import compute_feedback_from_pose


def _state(env):
    """5-dim [agent_x, agent_y, block_x, block_y, block_angle] straight off the sim."""
    return np.array([*env.agent.position, *env.block.position, env.block.angle], dtype=np.float64)


def _keypoint_dist(pose):
    """Mean per-keypoint distance to the goal T -- the verifier's own value, up to sign."""
    disp = compute_feedback_from_pose(np.asarray(pose, dtype=np.float32)[None]).reshape(-1, 2)
    return float(np.linalg.norm(disp, axis=-1).mean())


def _replay(env, actions):
    for a in actions:
        env.step(np.asarray(a, dtype=np.float64))
    return _state(env)


def _fresh(legacy=False):
    e = PushTEnv(legacy=legacy)
    e.reset()
    return e


@click.command()
@click.option('--zarr', default='data/pusht_cchi_v7_replay.zarr')
@click.option('--n-episodes', default=20, help='test-split episodes to sample')
@click.option('--per-episode', default=6, help='decision points per episode')
@click.option('--horizon', default=15, help='steps replayed after t (P3.2 uses 15 of 16)')
@click.option('--seed', default=42)
def main(zarr, n_episodes, per_episode, horizon, seed):
    rb = ReplayBuffer.copy_from_path(zarr, keys=['state', 'action'])
    ends = np.asarray(rb.episode_ends[:])
    starts = np.concatenate([[0], ends[:-1]])
    _, _, test_mask = get_split_masks_3way(len(ends), 50, 30, seed=seed)
    eps = np.flatnonzero(test_mask)[:n_episodes]
    state_all, action_all = np.asarray(rb['state']), np.asarray(rb['action'])
    rng = np.random.default_rng(seed)

    err = collections.defaultdict(list)
    for ep in eps:
        s0, s1 = starts[ep], ends[ep]
        n = s1 - s0
        lo, hi = 2, n - horizon - 1
        if hi <= lo:
            continue
        for t_rel in rng.choice(np.arange(lo, hi), size=min(per_episode, hi - lo), replace=False):
            t = s0 + int(t_rel)
            acts = action_all[t:t + horizon]

            # (c) TRUE: replay the whole episode up to t, so velocity is exactly right
            e = _fresh(); e._set_state(state_all[s0])
            _replay(e, action_all[s0:t])
            true_pre = _state(e)
            true_post = _replay(e, acts)

            # (a) TODAY: what the verifier does -- reset to state_t, zero velocity
            e = _fresh(); e._set_state(state_all[t])
            today_post = _replay(e, acts)

            # (b) WARMUP: reset to t-1, step a_{t-1}, snap POSITIONS back to t (velocity
            #     survives because _set_state writes position/angle only), then replay
            e = _fresh(); e._set_state(state_all[t - 1])
            e.step(np.asarray(action_all[t - 1], dtype=np.float64))
            e._set_state(state_all[t])
            warm_post = _replay(e, acts)

            for tag, got in (('today', today_post), ('warmup', warm_post)):
                err[f'{tag}_pos'].append(float(np.linalg.norm(got[2:4] - true_post[2:4])))
                d = abs(got[4] - true_post[4]) % (2 * np.pi)
                err[f'{tag}_ang'].append(float(np.degrees(min(d, 2 * np.pi - d))))
                err[f'{tag}_kp'].append(abs(_keypoint_dist(got[2:5]) - _keypoint_dist(true_post[2:5])))
            err['agent_speed_at_t'].append(float(np.linalg.norm(true_pre[:2] - state_all[t - 1][:2])))

    n = len(err['today_pos'])
    print(f'\n{n} decision points from {len(eps)} test episodes, {horizon}-step replay\n')
    print(f'{"":<10}{"block pos err (px)":>34}{"angle err (deg)":>26}{"keypoint-dist err":>22}')
    print(f'{"variant":<10}{"mean":>10}{"median":>10}{"p95":>12}{"mean":>10}{"p95":>14}{"mean":>12}{"p95":>10}')
    for tag in ('today', 'warmup'):
        p, a, k = (np.array(err[f'{tag}_{x}']) for x in ('pos', 'ang', 'kp'))
        print(f'{tag:<10}{p.mean():>10.2f}{np.median(p):>10.2f}{np.percentile(p,95):>12.2f}'
              f'{a.mean():>10.2f}{np.percentile(a,95):>14.2f}{k.mean():>12.3f}{np.percentile(k,95):>10.3f}')
    sp = np.array(err['agent_speed_at_t'])
    print(f'\nagent displacement over the step before t (the momentum the reset discards): '
          f'mean {sp.mean():.2f} px, p95 {np.percentile(sp,95):.2f} px')
    imp = 1 - np.array(err['warmup_pos']).mean() / max(1e-9, np.array(err['today_pos']).mean())
    print(f'warm-up reduces mean block-position error by {100*imp:.0f}%')
    print('\nGATE: if `today` keypoint-dist error is small relative to the value spread '
          'between candidates (~5 units, measured), P3.2 is not worth ~2x verifier cost.')


if __name__ == '__main__':
    main()
