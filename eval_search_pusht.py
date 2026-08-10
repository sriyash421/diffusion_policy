"""Best-of-N-search success curve for the offline PushT diffusion-search policy.

For each ``n in {2^0 .. 2^6}`` the policy is rolled out so that at every control step the
search produces ``n`` candidate action sequences and the **best-value** one (argmax verifier
value) is executed -- exactly ``policy.predict_action_best(obs, n_actions=n)``:
  * n <= max_actions : ``predict_n_actions`` generates n sequential candidates, each
    conditioned on the previous ones. (Note these are freshly drawn per n; the n=1 candidate
    is NOT a prefix of the n=16 set.)
  * n >  max_actions : rolling window -- context is the last ``max_actions`` candidates.

Success = episode max coverage >= threshold (max reward >= 1.0), reported as a rate with a
95% Wilson interval, because the splits are small enough (50 test / 10 val) that sub-10pp
differences are not distinguishable from sampling noise.

BOTH splits are evaluated. **Selection is done on val; test is reported at the val-selected
step and never selected on** -- picking the max over many noisy test estimates and then
reporting that same maximum inflates the figure by roughly one to two standard errors.

Everything is seeded from ``cfg.training.seed`` (override with ``--seed``) and re-seeded
before each n, so a rerun reproduces and points on the curve are paired.

NOTE: this is in-the-loop search over CANDIDATES PER CONTROL STEP. ``eval_bon.py`` plots a
different quantity -- a post-hoc oracle max over n independent full episodes -- which is not
deployable. Do not put the two on the same axes.

Single-checkpoint:
  python eval_search_pusht.py -c <run>/checkpoints/step_0010000.ckpt
Watcher (evals each new step_*.ckpt as training writes it, logs curves to wandb):
  python eval_search_pusht.py --watch --run-dir <train output_dir> --idle-exit-sec 7200

"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import re
import json
import fcntl
import hashlib
import time
import math
import pathlib
import contextlib
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
    get_split_masks_3way, get_episode_init_states,
    load_split_manifest, masks_from_manifest)
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


def load_policy(checkpoint, device, num_inference_steps=None):
    """Rebuild the policy from a checkpoint payload.

    ``num_inference_steps`` overrides the value baked into the checkpoint's config. It is a
    pure sampling-time knob -- the trained weights are unchanged -- so sweeping it measures
    how much of a weak n=1 result is sampler error rather than a weak policy. The configs
    here use 8 DDIM steps (inherited from the search arms, where every candidate is
    sampled 15x per gradient step and 8 keeps training affordable); upstream PushT uses 100.
    """
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    if num_inference_steps is not None:
        cfg.policy.num_inference_steps = int(num_inference_steps)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.get('use_ema', False) and getattr(workspace, 'ema_model', None) is not None:
        policy = workspace.ema_model
    policy.to(torch.device(device))
    policy.eval()
    return policy, cfg


def get_split_states(cfg, split, run_dir=None):
    """Reset states for 'val' or 'test' -- the SAME split the checkpoint was trained under.

    `cfg` comes out of the checkpoint payload, so the split parameters are the training
    run's own. When the config names a split_file, that manifest is the source of truth;
    otherwise the seeded derivation is used, as before.

    If `run_dir` has a splits.json (written by the training run), the resolved val/test
    indices are cross-checked against it, so an eval can never silently score a checkpoint
    against a different partition than the one it was trained on.
    """
    assert split in ('val', 'test')
    ds = cfg.task.dataset
    replay_buffer = ReplayBuffer.copy_from_path(
        ds.zarr_path, keys=['agent_pos', 'block_pos'])
    split_file = ds.get('split_file', None)
    if split_file is not None:
        manifest = load_split_manifest(
            split_file,
            episode_ends=replay_buffer.episode_ends[:],
            expected_counts={
                'test': ds.get('n_test_episodes', None),
                'val': ds.get('n_val_episodes', None) or None,
                'train': ds.get('n_train_episodes', None),
            })
        _, val_mask, test_mask = masks_from_manifest(
            manifest, replay_buffer.n_episodes)
    else:
        _, val_mask, test_mask = get_split_masks_3way(
            n_episodes=replay_buffer.n_episodes,
            n_test_episodes=ds.n_test_episodes,
            n_val_episodes=ds.get('n_val_episodes', 0),
            seed=ds.seed,
            n_train_episodes=ds.get('n_train_episodes', None))
    mask = val_mask if split == 'val' else test_mask
    idxs = np.nonzero(mask)[0]

    if run_dir is not None:
        recorded = pathlib.Path(run_dir).joinpath('splits.json')
        if recorded.is_file():
            try:
                on_disk = json.loads(recorded.read_text()).get(split)
            except Exception:
                on_disk = None
            if on_disk is not None and sorted(int(i) for i in on_disk) != sorted(
                    int(i) for i in idxs):
                raise ValueError(
                    f'{recorded} records {len(on_disk)} {split} episodes but this config '
                    f'resolves to {len(idxs)}. The checkpoint was trained under a '
                    f'different partition than the one being evaluated; the resulting '
                    f'success rate would not be comparable with anything.')
    return get_episode_init_states(replay_buffer, mask), idxs


def get_test_states(cfg):
    """Back-compat alias; prefer get_split_states."""
    return get_split_states(cfg, 'test')


def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for a binomial rate.

    Reported alongside every success rate because the splits are small -- 50 test episodes
    give SE ~7pp and 10 val episodes ~16pp, so differences smaller than the interval are
    not distinguishable from sampling noise. Wilson rather than normal-approx because it
    stays inside [0,1] and behaves at p near 0 or 1, which is exactly where these land.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


# Discount for the discounted-return metric. Applied to the per-step reward sequence and
# normalized by (1-gamma) so the result stays in [0, 1] and is directly comparable to the
# other two reward metrics rather than scaling with episode length.
GAMMA = 0.99
# 'max' is primary: success thresholds it, and it is the only one stored by evals
# predating the other two. Order is fixed -- rollout_max_rewards returns this tuple.
REWARD_KINDS = ('max', 'final', 'discounted')


def _reduce_episode_rewards(rewards):
    """Three scalars per episode from its per-step reward sequence.

    PushT's per-step reward is clip(coverage/0.95, 0, 1), so all three live in [0, 1]:

      max         the best coverage the episode ever reached. This is what `success`
                  thresholds (>= 1.0) and what has always been stored.
      final       coverage at the LAST step. Differs from max exactly when the policy
                  reaches the goal and then leaves -- which max, by construction, cannot
                  see. A policy that parks the T correctly and one that clips through the
                  goal on its way past score the same max and very different final.
      discounted  (1-gamma) * sum_t gamma^t r_t. Rewards getting there SOONER; max and
                  final are both indifferent to when the coverage was achieved.
    """
    r = np.asarray(rewards, dtype=np.float64)
    if r.size == 0:
        return 0.0, 0.0, 0.0
    w = GAMMA ** np.arange(r.size)
    return float(r.max()), float(r[-1]), float((1.0 - GAMMA) * np.sum(w * r))


def rollout_max_rewards(env, policy, states, device, n):
    """Roll one episode per env (reset to its state) using best-of-n search.

    Returns (max, final, discounted) reward arrays, one entry per env. The name is kept
    because `max` remains the primary series -- success is defined on it.
    """
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
    red = np.array([_reduce_episode_rewards(r) for r in rewards])   # (n_envs, 3)
    return red[:, 0], red[:, 1], red[:, 2]


def _eval_split_at_n(env, policy, states, device, n, n_envs, seed, label):
    """Best-of-n success rate over one split at ONE n.

    Returns (rate, ci, rewards) where `rewards` is a dict of three per-episode arrays --
    'max', 'final', 'discounted' (see _reduce_episode_rewards). Success is defined on
    'max' only; the other two are reported alongside it and never selected on.
    """
    n_resets = len(states)
    # Re-seed to the SAME base before every n. Two reasons: the run is reproducible at
    # all (previously nothing was seeded, so the diffusion noise at
    # diffusion_transformer_search_policy.py's conditional_sample came from an
    # unseeded global RNG), and each n starts from an identical RNG state instead of
    # continuing one stream, so points on the curve are paired rather than each
    # carrying independent sampler noise. This is why interleaving the two splits per n
    # (rather than running all of val then all of test) changes no number: the seed is
    # reset at every (split, n), so neither split's stream can reach the other.
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = {k: np.full(n_resets, np.nan) for k in REWARD_KINDS}
    for start in tqdm.tqdm(range(0, n_resets, n_envs),
            desc=f'{label} n={n}', leave=False):
        chunk_states = list(states[start:start + n_envs])
        pad = n_envs - len(chunk_states)
        if pad > 0:
            chunk_states = chunk_states + [states[0]] * pad
        t0 = time.perf_counter()
        chunk = rollout_max_rewards(env, policy, chunk_states, torch.device(device), n)
        dt = time.perf_counter() - t0
        for kind, arr in zip(REWARD_KINDS, chunk):
            out[kind][start:start + n_envs - pad] = arr[:n_envs - pad]
        tqdm.tqdm.write(f'  {label} n={n} chunk {start}: {dt:.1f}s')
    assert not any(np.isnan(v).any() for v in out.values())
    all_rewards = out['max']
    k = int(np.sum(all_rewards >= SUCCESS_REWARD))
    sr = k / n_resets
    ci = wilson_interval(k, n_resets)
    print(f'{label} n={n}: success_rate={sr:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  '
          f'({k}/{n_resets})  mean reward max={out["max"].mean():.3f} '
          f'final={out["final"].mean():.3f} disc={out["discounted"].mean():.3f}')
    return float(sr), ci, out


def eval_checkpoint(checkpoint, device, n_list=N_LIST, n_envs=50, max_steps=300, seed=None,
                    run_dir=None, on_n_done=None, num_inference_steps=None,
                    selection=None, selection_temperature=1.0, skip_val=False):
    """Sweep n_list, evaluating val and test at each n.

    ``on_n_done(curve)`` is invoked after EVERY n with the curve accumulated so far, and is
    what makes the sweep interruption-tolerant: cost is linear in n and the levels sum to
    ~2*max_n, so at max_n=1024 a single sweep runs ~10h and previously wrote nothing until
    it returned. Six such jobs were killed at the walltime having produced zero output.
    With the callback, a job that dies at n=512 still leaves every n below it on disk, and
    the remaining levels can be run as separate jobs and merged (see --min-n).
    """
    policy, cfg = load_policy(checkpoint, device, num_inference_steps)
    # Selection is a pure READOUT rule -- which of the n scored candidates gets executed --
    # so it can be swapped on a trained checkpoint without retraining. That is what makes
    # 'same weights, argmax vs softmax' a controlled comparison rather than two runs.
    if selection is not None:
        policy.selection = selection
        policy.selection_temperature = float(selection_temperature)
    if seed is None:
        seed = int(cfg.training.get('seed', 42))
    # run_dir defaults to the checkpoint's own run (checkpoints/ sits directly under it), so
    # the splits.json cross-check applies to single-checkpoint evals too.
    if run_dir is None:
        run_dir = pathlib.Path(checkpoint).resolve().parent.parent
    test_states, test_idxs = get_split_states(cfg, 'test', run_dir=run_dir)
    # skip_val halves the cost: val's 10-or-30 episodes are padded out to n_envs=50, so the
    # two splits cost the SAME despite the episode counts (measured 505s vs 503s at n=32).
    # The tradeoff is that the run then has no val curve, so any checkpoint later picked
    # from it is picked on TEST -- and a test number read at a test-chosen step is not a
    # held-out estimate. Use --skip-val for REPORTING a checkpoint chosen some other way.
    if skip_val:
        val_states, val_idxs = [], np.zeros(0, dtype=int)
    else:
        val_states, val_idxs = get_split_states(cfg, 'val', run_dir=run_dir)

    # Read the horizon from cfg.policy, NOT the top-level cfg: the policy is built with
    # n_action_steps + n_latency_steps, so with a nonzero latency the top-level values
    # would size the env's MultiStepWrapper differently from the chunk the policy emits.
    # The wrapper executes whatever it is handed, so that mismatch is silent -- a wrong
    # control cadence and a wrong success rate, with no exception.
    env = build_envs(n_envs, cfg.policy.n_obs_steps, cfg.policy.n_action_steps, max_steps)
    val_sr, val_ci, test_sr, test_ci, test_rewards, val_rewards = {}, {}, {}, {}, {}, {}
    done = []
    # How the executed action is picked, and therefore what n COSTS. Under 'final_pass' the
    # policy draws one further sample, conditioned on the n scored candidates, and executes
    # that -- so a point at n is n+1 diffusion samples, not n. Recorded per curve because
    # otherwise a `subgoal-only` curve and a `value` curve read off the same x axis are
    # being compared at different compute, which flatters whichever one is doing more work.
    selection = getattr(policy, 'selection', 'argmax')
    n_extra = 1 if selection == 'final_pass' else 0
    temperature = getattr(policy, 'selection_temperature', None)

    def _curve():
        return {
            'n': list(done),
            'seed': seed,
            'selection': selection,
            # recorded so a softmax curve is self-describing: the number is meaningless
            # without knowing T, and T only means anything because the score is z-scored
            'selection_temperature': temperature if selection == 'softmax' else None,
            'n_generations': [int(n) + n_extra for n in done],
            # test split: REPORTED
            'success_rate': [test_sr[n] for n in done],
            'success_ci': [test_ci[n] for n in done],
            'n_episodes': len(test_states),
            'episode_idxs': test_idxs.tolist(),
            'per_n_rewards': {int(n): test_rewards[n]['max'].tolist() for n in done},
            # 'max' only, for backward compatibility: every curve ever written holds this
            # key and downstream readers index it directly. The other two kinds live in
            # per_n_rewards_by_kind and are absent from pre-2026-08-05 curves.
            'per_n_rewards_by_kind': {
                kind: {int(n): test_rewards[n][kind].tolist() for n in done}
                for kind in REWARD_KINDS},
            'gamma': GAMMA,
            # Mean of the same per-episode max rewards the binary rate thresholds, i.e.
            # clip(coverage/0.95, 0, 1) averaged over the test episodes. Success throws away
            # everything below the threshold, so two policies that both fail every episode
            # are indistinguishable by it while one is parking the block at 90% coverage and
            # the other never touches it. Reported next to success, never selected on (val
            # is). Derived, not measured: `per_n_rewards` already held it, so every curve
            # written before this field existed can be -- and was -- backfilled exactly.
            'mean_reward': [float(np.mean(test_rewards[n]['max'])) for n in done],
            'mean_reward_final': [float(np.mean(test_rewards[n]['final'])) for n in done],
            'mean_reward_discounted': [
                float(np.mean(test_rewards[n]['discounted'])) for n in done],
            # val split: SELECTED ON
            'val_success_rate': ([val_sr[n] for n in done] if val_states else None),
            'val_success_ci': ([val_ci[n] for n in done] if val_states else None),
            # Mean val reward, kept because binary success cannot always select. A BC
            # policy at n=1 scores 0% success at EVERY checkpoint, so the success-based
            # selector is a tie across the whole run and max() returns whichever row came
            # first. Mean reward separates the same checkpoints cleanly (0.12 -> 0.29).
            'val_mean_reward': ([float(np.mean(val_rewards[n]['max'])) for n in done]
                                if val_states else None),
            'val_mean_reward_final': ([float(np.mean(val_rewards[n]['final'])) for n in done]
                                      if val_states else None),
            'val_mean_reward_discounted': ([float(np.mean(val_rewards[n]['discounted']))
                                            for n in done] if val_states else None),
            'val_n_episodes': len(val_states),
            'val_episode_idxs': val_idxs.tolist(),
        }

    try:
        # Interleaved by n, not split-major: a partial sweep then holds BOTH splits at every
        # n it finished, so the val-based selector works on whatever survived. Split-major
        # would leave a killed job with a complete val curve and no test numbers at all.
        for n in n_list:
            # val drives checkpoint selection; test is reported only, never selected on.
            if val_states:
                val_sr[n], val_ci[n], val_rewards[n] = _eval_split_at_n(
                    env, policy, val_states, device, n, n_envs, seed, 'val')
            test_sr[n], test_ci[n], test_rewards[n] = _eval_split_at_n(
                env, policy, test_states, device, n, n_envs, seed, 'test')
            done.append(n)
            if on_n_done is not None:
                on_n_done(_curve())
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

    return _curve()


def plot_curve(curve, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = np.array(curve['n'])
    sr = np.array(curve['success_rate'])
    fig, ax = plt.subplots(figsize=(6, 4))
    ci = curve.get('success_ci')
    if ci:
        lo = np.array([c[0] for c in ci])
        hi = np.array([c[1] for c in ci])
        # the band is the point of the plot: with 50 test episodes the SE is ~7pp, so
        # curve movement smaller than the band is not distinguishable from noise
        ax.fill_between(n, lo, hi, color='#2a78d6', alpha=0.18, lw=0,
                        label='95% CI (Wilson)')
    ax.plot(n, sr, 'o-', color='#2a78d6', lw=2, label='test (reported)')
    vsr = curve.get('val_success_rate')
    if vsr:
        ax.plot(n, np.array(vsr), 's--', color='#d67a2a', lw=1.5, alpha=0.9,
                label='val (selected on)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(n)
    ax.set_xticklabels([str(int(x)) for x in n])
    ax.set_ylim(-0.03, 1.03)
    # explicit about WHICH best-of-n this is: eval_bon.py plots a different quantity
    # (max over n independent full episodes, a post-hoc oracle) on identically-labelled
    # axes, and the two must never be put on the same slide.
    ax.set_xlabel('n = candidates per control step (verifier-argmax executed)')
    ax.set_ylabel('success rate (coverage >= 95%)')
    n_ep = curve.get('n_episodes')
    ax.set_title('PushT in-the-loop best-of-n search'
                 + (f'  (test n={n_ep} episodes)' if n_ep else ''))
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# per-n lists that must stay index-aligned with curve['n'] whenever two curves are merged
_PER_N_LISTS = ('success_rate', 'success_ci',
                'mean_reward', 'mean_reward_final', 'mean_reward_discounted',
                'val_success_rate', 'val_success_ci',
                'val_mean_reward', 'val_mean_reward_final', 'val_mean_reward_discounted')
# fields that identify WHICH experiment a curve describes; merging across a mismatch here
# would splice together points measured on different episodes or a different sampler seed
_IDENTITY = ('seed', 'n_episodes', 'episode_idxs', 'val_n_episodes',
             'val_episode_idxs',
             # a softmax curve and an argmax curve at the same step are different
             # experiments; without these two a --selection override that landed in
             # the same directory would merge into the native curve n-by-n
             'selection', 'selection_temperature')


def merge_curves(old, new):
    """Union two curves over n. `new` wins where both have the same n.

    Returns `new` unchanged if the two disagree on seed or on either split's episode set --
    those points are not comparable and must not share a curve.
    """
    if not old or 'n' not in old:
        return new
    # Compare only fields BOTH sides carry: the jsonl row is a projection of the curve and
    # omits episode_idxs, so a plain != would read every row as a mismatch and silently
    # discard the n already on disk.
    clash = [k for k in _IDENTITY
             if old.get(k) is not None and new.get(k) is not None and old[k] != new[k]]
    if clash:
        print(f'warning: existing curve differs on {clash}; replacing it rather than '
              f'splicing points measured under different conditions')
        return new
    by_n = {}
    for src in (old, new):                      # new second, so it overwrites
        for i, n in enumerate(src['n']):
            by_n[int(n)] = {k: src[k][i] for k in _PER_N_LISTS if src.get(k) is not None}
    ns = sorted(by_n)
    out = dict(old)          # start from old so keys new has never heard of survive
    out.update(new)
    out['n'] = ns
    for k in _PER_N_LISTS:
        # rebuild from by_n, which already unions both sides, whenever EITHER carries the
        # key -- keying off `new` alone let an older writer silently delete a newer series
        if old.get(k) is not None or new.get(k) is not None:
            vals = [by_n[n].get(k) for n in ns]
            if any(v is not None for v in vals):
                out[k] = vals
    rewards = dict(old.get('per_n_rewards') or {})
    rewards.update(new.get('per_n_rewards') or {})
    out['per_n_rewards'] = {int(k): v for k, v in sorted(rewards.items(), key=lambda kv: int(kv[0]))}
    by_kind = {}
    for kind in REWARD_KINDS:
        merged = dict((old.get('per_n_rewards_by_kind') or {}).get(kind) or {})
        merged.update((new.get('per_n_rewards_by_kind') or {}).get(kind) or {})
        if merged:
            by_kind[kind] = {int(k): v for k, v in
                             sorted(merged.items(), key=lambda kv: int(kv[0]))}
    if by_kind:
        out['per_n_rewards_by_kind'] = by_kind
    return out


@contextlib.contextmanager
def _locked(path):
    """Exclusive lock around a read-modify-write of the shared result index.

    One job per n means several processes merge into the same success_curve.json and
    success_curves.jsonl concurrently; without this, two jobs that read before either
    writes would each drop the other's n.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.lock', 'w') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_outputs(curve, output_dir, checkpoint, plot=True):
    """Merge `curve` into any curve already in output_dir, then write json (+png)."""
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out.joinpath('success_curve.json')
    with _locked(dst):
        merged = merge_curves(_read_json(dst), curve)
        tmp = str(dst) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'checkpoint': str(checkpoint), **merged}, f, indent=2)
        os.replace(tmp, dst)
    png = os.path.join(output_dir, 'success_curve.png')
    if plot:
        plot_curve(merged, png)
    return png


def step_from_ckpt(path):
    m = CKPT_RE.search(str(path))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Run-level result index: every evaluated checkpoint is merged into
# success_curves.jsonl, one row per checkpoint.
#
# This file records measurements ONLY. It deliberately names no winner. Any rule for
# picking a checkpoint (which n to read it at, val vs test, success vs mean reward, how
# to break ties) is an analysis choice, and baking one into the eval writer made that
# choice look like a result -- downstream scripts and the generated tables then inherited
# it silently. Read the curves and choose explicitly.
# ---------------------------------------------------------------------------
CURVES_JSONL = 'success_curves.jsonl'


def _row_from_curve(step, checkpoint, curve):
    return {
        'step': step,
        'checkpoint': str(checkpoint),
        'n': curve['n'],
        'seed': curve.get('seed'),
        # how the executed action was picked, and the sample count n really cost -- absent
        # in rows written before selection modes existed, which all mean 'argmax'/n
        'selection': curve.get('selection', 'argmax'),
        'selection_temperature': curve.get('selection_temperature'),
        'n_generations': curve.get('n_generations'),
        'success_rate': curve['success_rate'],
        'success_ci': curve.get('success_ci'),
        'mean_reward': curve.get('mean_reward'),
        'mean_reward_final': curve.get('mean_reward_final'),
        'mean_reward_discounted': curve.get('mean_reward_discounted'),
        'n_episodes': curve.get('n_episodes'),
        'val_success_rate': curve.get('val_success_rate'),
        'val_success_ci': curve.get('val_success_ci'),
        'val_mean_reward': curve.get('val_mean_reward'),
        'val_mean_reward_final': curve.get('val_mean_reward_final'),
        'val_mean_reward_discounted': curve.get('val_mean_reward_discounted'),
        'val_n_episodes': curve.get('val_n_episodes'),
        'timestamp': time.time(),
    }


def append_curve_row(out_root, step, checkpoint, curve):
    """Merge this checkpoint's result into the run-level jsonl.

    Merge, not append: with one job per n a checkpoint is written several times, and plain
    appends left several partial rows per step -- which read back as distinct checkpoints
    and produced duplicate table rows downstream.
    """
    out_root = pathlib.Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl = out_root.joinpath(CURVES_JSONL)
    row = _row_from_curve(step, checkpoint, curve)

    with _locked(jsonl):
        rows = read_curve_rows(out_root)
        by_step = {r['step']: r for r in rows}
        if step in by_step:
            row = _row_from_curve(step, checkpoint, merge_curves(by_step[step], curve))
        by_step[step] = row
        rows = [by_step[s] for s in sorted(by_step)]
        tmp = str(jsonl) + '.tmp'
        with open(tmp, 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        os.replace(tmp, jsonl)

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
@click.option('--max-n', default=64, type=int,
              help='largest n in the best-of-n sweep; the sweep is the powers of two up to this. Cost is LINEAR in n per level and the levels sum to ~2*max_n, so 1024 is ~16x the wall time of the default 64.')
@click.option('--min-n', default=1, type=int,
              help='smallest n in the sweep. With --max-n this runs ONE slice, so a big '
                   'sweep can be split across jobs (e.g. --min-n 512 --max-n 512); results '
                   'merge into the same success_curve.json / success_curves.jsonl.')
@click.option('--max-steps', default=300)
@click.option('--poll-sec', default=60.0)
@click.option('--seed', default=None, type=int,
              help='eval seed; defaults to cfg.training.seed so runs are reproducible')
@click.option('--num-inference-steps', default=None, type=int,
              help='override the checkpoint\'s DDIM inference steps (sampling-time only; '
                   'the configs use 8, upstream PushT uses 100)')
@click.option('--idle-exit-sec', default=None, type=float,
              help='watch mode: exit after this many seconds with no new checkpoint '
                   '(default: run forever)')
# Default OFF: the curves are written to success_curves.jsonl / success_curve.json,
# which is what every downstream reader uses. The wandb copy added a run per watcher
# and nothing consumed it. Pass --wandb explicitly if you want it back.
@click.option('--wandb/--no-wandb', 'use_wandb', default=False, help='log curves to wandb (watch mode)')
@click.option('--wandb-entity', default='l2sml')
@click.option('--wandb-project', default='pushT_diffusion_search')
@click.option('--selection', default=None,
              type=click.Choice(['argmax', 'softmax', 'final_pass']),
              help='override the checkpoint\'s own selection rule. Results land in '
                   'bon_search_sel-<mode>/ so they never merge with the native-mode curve.')
@click.option('--selection-temperature', default=1.0, type=float,
              help='softmax temperature on the STANDARDIZED score (T->0 == argmax)')
@click.option('--skip-val', is_flag=True,
              help='evaluate the 50 test episodes only, skipping the val split')
def main(checkpoint, output_dir, watch, run_dir, device, n_envs, max_n, min_n, max_steps,
         poll_sec, seed, num_inference_steps, idle_exit_sec, use_wandb, wandb_entity,
         wandb_project, selection, selection_temperature, skip_val):
    # powers of two in [min_n, max_n]; identical to N_LIST at the defaults, so existing
    # curves stay comparable and success_curves.jsonl rows stay mergeable.
    n_list = [n for n in (int(2 ** k) for k in range(31)) if min_n <= n <= max_n]
    assert n_list, f'no powers of two in [{min_n}, {max_n}]'
    if not watch:
        assert checkpoint is not None, 'provide -c/--checkpoint or --watch'
        # resolve() not a '..'-relative join: with a symlinked checkpoints/ the old form
        # resolved through the link and landed beside the target instead of in the run dir.
        run_root = pathlib.Path(checkpoint).resolve().parent.parent
        # A selection override is a DIFFERENT experiment on the same weights, so it gets
        # its own directory. Writing it into bon_search/ would merge it with the native
        # curve at the same step -- silently averaging two selection rules into one row.
        sub = 'bon_search' if selection is None else f'bon_search_sel-{selection}'
        out = pathlib.Path(output_dir) if output_dir else run_root.joinpath(sub)
        step = step_from_ckpt(checkpoint)
        # SAME per-step subdir convention as watch mode. Previously this wrote a flat
        # bon_search/success_curve.{json,png}, so N concurrent single-ckpt jobs on one run
        # all clobbered each other and only the last finisher survived.
        step_dir = out.joinpath(f'step_{step:07d}') if step is not None else out

        def _persist(curve):
            """Checkpoint the sweep after every n, so a killed job keeps what it finished."""
            save_outputs(curve, str(step_dir), checkpoint)
            append_curve_row(out, step, checkpoint, curve)
            print(f'  [persisted n<={max(curve["n"])}]')

        curve = eval_checkpoint(checkpoint, device, n_list=n_list, n_envs=n_envs,
                                max_steps=max_steps, seed=seed, on_n_done=_persist,
                                num_inference_steps=num_inference_steps,
                                selection=selection,
                                selection_temperature=selection_temperature,
                                skip_val=skip_val)
        png = save_outputs(curve, str(step_dir), checkpoint)
        print('wrote', png)
        return

    # watch mode
    assert run_dir is not None, '--watch requires --run-dir'
    ckpt_dir = pathlib.Path(run_dir).joinpath('checkpoints')
    sub = 'bon_search' if selection is None else f'bon_search_sel-{selection}'
    out_root = pathlib.Path(output_dir or os.path.join(run_dir, sub))
    if not ckpt_dir.is_dir():
        # Fail loudly on a wrong --run-dir. Previously a bad path was indistinguishable
        # from "training has not started yet": the watcher polled an empty glob forever,
        # printing nothing, holding a GPU. eval_watchers.tsv accumulated exactly this.
        print(f'WARNING: {ckpt_dir} does not exist yet. If training has not started this '
              f'is expected; if the path is wrong this watcher will poll forever.')

    run = None
    if use_wandb:
        try:
            import wandb
            # Deterministic id derived from the run dir so a requeue RESUMES the same
            # wandb run. resume='allow' without an id minted a fresh run per preemption,
            # scattering one curve across many runs.
            wandb_id = 'eval-' + hashlib.md5(
                str(pathlib.Path(run_dir).resolve()).encode()).hexdigest()[:12]
            run = wandb.init(entity=wandb_entity, project=wandb_project,
                             name=f'eval_{pathlib.Path(run_dir).name}',
                             id=wandb_id, job_type='eval', resume='allow')
        except Exception as e:
            # never let a logging failure kill the watcher: it was unguarded, so a wandb
            # auth/network problem took the whole job down at startup
            print(f'warning: wandb disabled ({e})')
            run = None

    # Resume-safe: this watcher runs on the preemptible ckpt partition with --requeue,
    # so an in-memory-only `seen` would re-evaluate every checkpoint from scratch after
    # each preemption. Seed it from the results already on disk.
    # A row only counts as done if it covers every n this watcher is asked for. Previously
    # any row at all marked the step seen, so raising --max-n on a run that already had
    # n<=64 curves silently skipped every checkpoint and the watcher evaluated nothing.
    seen = {row['step'] for row in read_curve_rows(out_root)
            if set(n_list) <= set(row.get('n') or [])}
    if seen:
        print(f'resuming: {len(seen)} checkpoint(s) already evaluated at n<={max_n}, '
              f'skipping those')
    print(f'watching {ckpt_dir} for step_*.ckpt (poll {poll_sec}s)')
    failed = {}
    last_progress = time.time()
    while True:
        ckpts = sorted(ckpt_dir.glob('step_*.ckpt')) if ckpt_dir.is_dir() else []
        for ckpt in ckpts:
            step = step_from_ckpt(ckpt)
            if step is None or step in seen:
                continue
            print(f'== evaluating {ckpt.name} (step {step}) ==')
            step_dir = str(out_root.joinpath(f'step_{step:07d}'))

            def _persist(curve, _step=step, _ckpt=str(ckpt), _dir=step_dir):
                # png only at the end; plotting after every n is pure overhead here
                save_outputs(curve, _dir, _ckpt, plot=False)
                append_curve_row(out_root, _step, _ckpt, curve)

            try:
                curve = eval_checkpoint(str(ckpt), device, n_list=n_list, n_envs=n_envs,
                                        max_steps=max_steps, seed=seed, run_dir=run_dir,
                                        on_n_done=_persist,
                                        num_inference_steps=num_inference_steps,
                                        selection=selection,
                                        selection_temperature=selection_temperature,
                                        skip_val=skip_val)
            except Exception as e:
                # NOT marked seen: the checkpoint may simply have been half-written (the
                # trainer saves on a background thread), or this may be a transient CUDA
                # OOM. Marking before the try burned the step permanently.
                failed[step] = failed.get(step, 0) + 1
                print(f'eval failed for {ckpt.name} (attempt {failed[step]}): {e}')
                if failed[step] >= 3:
                    print(f'  giving up on step {step} after 3 attempts')
                    seen.add(step)
                continue
            seen.add(step)
            failed.pop(step, None)
            last_progress = time.time()
            save_outputs(curve, step_dir, str(ckpt))
            if run is not None:
                try:
                    import wandb
                    for nn, sr in zip(curve['n'], curve['success_rate']):
                        run.log({f'test/success_rate_n{nn}': sr}, step=step)
                    for nn, sr in zip(curve['n'], curve.get('val_success_rate') or []):
                        run.log({f'val/success_rate_n{nn}': sr}, step=step)
                    table = wandb.Table(data=list(zip(curve['n'], curve['success_rate'])),
                                        columns=['n', 'success_rate'])
                    run.log({'test/success_curve': wandb.plot.line(
                        table, 'n', 'success_rate',
                        title=f'in-the-loop best-of-n success (step {step})')}, step=step)
                except Exception as e:
                    print(f'warning: wandb log failed: {e}')
        if idle_exit_sec is not None and (time.time() - last_progress) > idle_exit_sec:
            # Terminate instead of holding a GPU forever after training finishes. The loop
            # had no exit condition at all, so watchers outlived their runs indefinitely
            # and monitor_pusht_search.sh kept resubmitting them.
            print(f'no new checkpoint for {idle_exit_sec:.0f}s; exiting')
            break
        time.sleep(poll_sec)


if __name__ == '__main__':
    main()
