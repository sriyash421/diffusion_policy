"""Replay buffers that carry the tau ladder alongside the ordinary SAC transition.

SB3 stores one `(reward, done)` pair per transition. The ladder needs `(reward, done, live)`
per rung, because each rung is a DIFFERENT MDP over the same trajectory -- it terminates at its
own threshold, and a transition past that threshold belongs to an episode that has already
ended and must be dropped from that rung's minibatch rather than trained on with `done=False`.

The env computes the triple (`ChunkPushTEnv.step`) and it rides in `infos`, which is the one
channel SB3 passes through `add()` untouched.
"""

import numpy as np
import torch as th
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer


class _TauMixin:
    """The extra arrays, and the one place they are filled and read."""

    def _init_tau(self, n_tau):
        self.n_tau = int(n_tau)
        shape = (self.buffer_size, self.n_envs, self.n_tau)
        self.tau_reward = np.zeros(shape, dtype=np.float32)
        self.tau_done = np.zeros(shape, dtype=np.float32)
        self.tau_live = np.ones(shape, dtype=np.float32)

    def _store_tau(self, pos, infos):
        for i, info in enumerate(infos):
            if "tau_reward" not in info:
                continue
            self.tau_reward[pos, i] = info["tau_reward"]
            self.tau_done[pos, i] = np.asarray(info["tau_done"], dtype=np.float32)
            self.tau_live[pos, i] = np.asarray(info["tau_live"], dtype=np.float32)

    def tau_batch(self, batch_inds, env_inds):
        """(reward, done, live) for one sampled minibatch, each (B, n_tau) on the right device."""
        to = lambda x: th.as_tensor(x[batch_inds, env_inds], device=self.device)  # noqa: E731
        return to(self.tau_reward), to(self.tau_done), to(self.tau_live)

    def _draw(self, batch_size):
        """The index pair SB3 draws internally and does not hand back.

        `_get_samples` picks `env_indices` itself with a fresh `np.random.randint`, so calling
        `sample()` and then `tau_batch()` would pair each transition with ANOTHER transition's
        rung labels -- silently, and the resulting Q would be regressed on shuffled targets.
        Drawing both here is the only way the two stay the same minibatch.
        """
        upper = self.buffer_size if self.full else self.pos
        return (np.random.randint(0, upper, size=batch_size),
                np.random.randint(0, self.n_envs, size=batch_size))


class ChunkReplayBuffer(_TauMixin, ReplayBuffer):
    """Flat-observation buffer (the keypoint arm)."""

    def __init__(self, *args, n_tau=4, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_tau(n_tau)

    def add(self, obs, next_obs, action, reward, done, infos):
        self._store_tau(self.pos, infos)
        super().add(obs, next_obs, action, reward, done, infos)

    def sample_chunk(self, batch_size, env=None):
        """(ReplayBufferSamples, (reward, done, live)) for ONE minibatch -- see `_draw`."""
        from stable_baselines3.common.type_aliases import ReplayBufferSamples

        b, e = self._draw(batch_size)
        nxt = self.observations[(b + 1) % self.buffer_size, e] if self.optimize_memory_usage \
            else self.next_observations[b, e]
        data = ReplayBufferSamples(
            self.to_torch(self._normalize_obs(self.observations[b, e], env)),
            self.to_torch(self.actions[b, e]),
            self.to_torch(self._normalize_obs(nxt, env)),
            self.to_torch((self.dones[b, e] * (1 - self.timeouts[b, e])).reshape(-1, 1)),
            self.to_torch(self._normalize_reward(self.rewards[b, e].reshape(-1, 1), env)))
        return data, self.tau_batch(b, e)


class ChunkDictReplayBuffer(_TauMixin, DictReplayBuffer):
    """Dict-observation buffer (the image arm).

    Note DictReplayBuffer asserts against `optimize_memory_usage`, so obs and next_obs are both
    stored in full: 54.0 KiB per transition at 96x96x3, measured. That is the binding constraint
    on `--buffer-size` for the image arm, not a tunable.
    """

    def __init__(self, *args, n_tau=4, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_tau(n_tau)

    def add(self, obs, next_obs, action, reward, done, infos):
        self._store_tau(self.pos, infos)
        super().add(obs, next_obs, action, reward, done, infos)

    def sample_chunk(self, batch_size, env=None):
        """(DictReplayBufferSamples, (reward, done, live)) for ONE minibatch -- see `_draw`."""
        from stable_baselines3.common.type_aliases import DictReplayBufferSamples

        b, e = self._draw(batch_size)
        obs = self._normalize_obs({k: v[b, e, :] for k, v in self.observations.items()}, env)
        nxt = self._normalize_obs({k: v[b, e, :] for k, v in self.next_observations.items()}, env)
        data = DictReplayBufferSamples(
            {k: self.to_torch(v) for k, v in obs.items()},
            self.to_torch(self.actions[b, e]),
            {k: self.to_torch(v) for k, v in nxt.items()},
            self.to_torch((self.dones[b, e] * (1 - self.timeouts[b, e])).reshape(-1, 1)),
            self.to_torch(self._normalize_reward(self.rewards[b, e].reshape(-1, 1), env)))
        return data, self.tau_batch(b, e)


def buffer_class_for(obs_type):
    return ChunkDictReplayBuffer if obs_type == "image" else ChunkReplayBuffer


def preload_demos(buffer, tr, verbose=True):
    """Push demo chunk transitions into a live replay buffer, through `add()`.

    Added in groups of `buffer.n_envs`, which is the shape SB3's buffers store -- and through
    `add()` rather than by writing the arrays directly, so demo transitions take exactly the
    path collected ones take. A direct write is how the two end up on different dtypes or
    different scales with nothing raising.
    """
    n_envs = buffer.n_envs
    n = (len(tr["action"]) // n_envs) * n_envs
    dict_obs = isinstance(tr["obs"], dict)
    sl = lambda x, a, b: {k: v[a:b] for k, v in x.items()} if dict_obs else x[a:b]  # noqa: E731
    for a in range(0, n, n_envs):
        b = a + n_envs
        info = [{"tau_reward": tr["reward"][j], "tau_done": tr["done"][j], "tau_live": tr["live"][j]}
                for j in range(a, b)]
        # The SCALAR reward is the TOP rung's -- what the SAC actor optimises. The lower rungs
        # exist for the verifier head alone and must never move the policy.
        buffer.add(sl(tr["obs"], a, b), sl(tr["next_obs"], a, b), tr["action"][a:b],
                   tr["reward"][a:b, -1].astype(np.float32),
                   tr["done"][a:b, -1].astype(np.float32), info)
    if verbose:
        print(f"[INFO] preloaded {n} demo chunk transitions "
              f"({buffer.pos * n_envs}/{buffer.buffer_size * n_envs} slots used)")
    return n
