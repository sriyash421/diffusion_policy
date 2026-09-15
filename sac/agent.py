"""The learner: SAC over action chunks, the verifier head, and the buffers that carry the ladder.

TWO THINGS ARE DIFFERENT from stock SAC, and both exist for the same reason: the Q is going to
be asked about actions a DIFFUSION POLICY proposes, not actions this actor proposes.

1. `_sample_action` draws from a MIXTURE, not from the actor. SAC is off-policy, so any
   behaviour policy is legal, and a buffer fed only by the actor yields a Q that is sharp
   exactly where it is not needed and flat where it is.

   The override is in `_sample_action` and NOT in an env wrapper, deliberately: SB3 stores the
   `buffer_action` this method returns, so a wrapper that swapped the action afterwards would
   write one action into the buffer and execute another. Nothing raises; the Q just regresses
   on labels that belong to different actions.

2. The verifier head is not SAC's critic. SAC learns the entropy-augmented `Q_soft^pi` -- the
   value of its own stochastic policy plus `-alpha log pi` at every future step. With a sparse
   reward whose entire magnitude is 1.0 and ~38 chunks of entropy accumulating against it, the
   soft value can be mostly entropy. Best-of-N does not evaluate a stochastic policy; it takes
   an argmax over n candidates and executes it, so the head's backup is that same operation.

   ISOLATION. The head is built AFTER the policy's own optimizers, carries its own, and takes
   detached features -- the construction `recurrent_ppo.corrupt_policy._QHeadMixin` uses and
   tests. It provably cannot move the policy, so the SAC run is unchanged by its presence.

THE LADDER. One head per rung over a shared trunk. Rung `tau` sees the terminate-at-tau MDP:
its own `(reward, done)` and a `live` mask dropping transitions from episodes that crossed tau
earlier. `Q_0.95` is therefore the SAME function it would be if trained alone -- the lower rungs
only shape the trunk, and they have signal the demonstrations already contain (658 and 226
terminals at 0.80 and 0.85, against ZERO at 0.95). SB3 stores one `(reward, done)` per
transition and has no slot for that, which is why the buffers below exist.
"""

import copy

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.utils import polyak_update
from torch import nn

from sac.config import WS
from sac.env import encode


# ==================================================================== the replay buffers


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


# ==================================================================== the verifier head


def _mlp(in_dim, net_arch, out_dim):
    layers, last = [], in_dim
    for hidden in net_arch:
        layers += [nn.Linear(last, hidden), nn.ReLU()]
        last = hidden
    return nn.Sequential(*layers, nn.Linear(last, out_dim))


class BONHead(nn.Module):
    """An ensemble of `n_critics` nets, each emitting one value per rung."""

    def __init__(self, feature_dim, action_dim, n_tau, n_critics=2, net_arch=(256, 256)):
        super().__init__()
        self.n_tau, self.n_critics = int(n_tau), int(n_critics)
        self.nets = nn.ModuleList([_mlp(feature_dim + action_dim, net_arch, n_tau)
                                   for _ in range(self.n_critics)])

    def forward(self, features, actions):
        """-> (n_critics, B, n_tau)."""
        x = th.cat([features, actions], dim=-1)
        return th.stack([net(x) for net in self.nets], dim=0)

    def mean(self, features, actions):
        """The point estimate used to RANK. `min` controls overestimation in a backup; for
        ranking the mean is the better estimator and the spread is the honest uncertainty."""
        return self.forward(features, actions).mean(0)

    def spread(self, features, actions):
        """Ensemble std -> (B, n_tau). When this exceeds the spread ACROSS candidates, the
        verifier is not resolving the question it is being asked and BON should fall back to
        n=1 rather than rank on noise."""
        return self.forward(features, actions).std(0)


def attach_bon_head(policy, n_tau, n_critics=2, net_arch=(256, 256), lr=3e-4):
    """Build the head and its optimizer on an already-constructed SAC policy.

    After `policy.__init__`, so `policy.optimizer`/`policy.critic.optimizer` were built over the
    parameters that existed then and cannot see these. That is what makes the isolation a fact
    about the construction order rather than a promise.
    """
    feature_dim = policy.critic.features_extractor.features_dim
    action_dim = policy.action_space.shape[0]
    head = BONHead(feature_dim, action_dim, n_tau, n_critics, net_arch).to(policy.device)
    policy.bon_head = head
    policy.bon_head_target = copy.deepcopy(head).requires_grad_(False)
    policy.bon_optimizer = th.optim.Adam(head.parameters(), lr=lr)
    return head


# ==================================================================== the agent


def agent_pos_from_obs(obs, obs_type):
    """Agent position in ARENA PIXELS, out of the observation the policy sees.

    Everything the policy sees is in [-1, 1] (PushTGymEnv's convention), so this inverts
    `_normalise`. The index is the one place each arm's layout is asserted.
    """
    if obs_type == "image":
        norm = np.asarray(obs["agent_pos"])
    else:
        norm = np.asarray(obs)[..., 18:20]      # [9 block kps (18), agent xy (2)] then the mask
    return (norm + 1.0) * (WS / 2)


class ChunkSAC(SAC):
    """SAC whose action is a chunk, plus the hard max-over-candidates verifier head."""

    def __init__(self, *args, obs_type="keypoint", tau_ladder=(0.95,), demo_chunks=None,
                 chunk=8,
                 mix=(0.40, 0.10, 0.25, 0.25), smooth_scale=12.0,
                 bon_net_arch=(256, 256), bon_n_critics=2,
                 bon_lr=3e-4, bon_candidates=8, **kwargs):
        # set BEFORE super().__init__(), which calls _setup_model() and therefore needs them
        self.obs_type = obs_type
        self.tau_ladder = tuple(float(t) for t in tau_ladder)
        self.chunk = int(chunk)
        self.mix = np.asarray(mix, dtype=np.float64) / np.sum(mix)
        self.bon_candidates = int(bon_candidates)
        self.smooth_scale = float(smooth_scale)
        self.bon_net_arch, self.bon_n_critics, self.bon_lr = tuple(bon_net_arch), int(bon_n_critics), float(bon_lr)
        # demo chunks as SHAPES (targets minus their own start agent position), so a chunk the
        # expert performed in one corner is a usable proposal anywhere. Under `absolute` the raw
        # encoding is a location rather than a shape, and injecting it would only ever teach Q
        # about the places the demos happened to visit.
        self.demo_offsets = None if demo_chunks is None else np.asarray(demo_chunks, np.float64)
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        """Attach the verifier head here, not in `__init__`.

        SB3's `load` constructs the class with `_init_setup_model=False`, then restores the
        saved attributes, then calls this -- so `__init__` has no `self.policy` on that path,
        and the head's width must be read from the RESTORED `tau_ladder` rather than from a
        value cached at construction. The head is a submodule of the policy, so its weights
        ride in `policy.state_dict()` and are restored by the normal parameter load.
        """
        super()._setup_model()
        attach_bon_head(self.policy, len(self.tau_ladder), self.bon_n_critics,
                        self.bon_net_arch, self.bon_lr)

    # ------------------------------------------------------------------ behaviour mixture
    def _mixture_actions(self, n_envs):
        """(n_envs, action_dim) in the env action range, drawn from the behaviour mixture."""
        actor, _ = self.predict(self._last_obs, deterministic=False)
        out = np.array(actor, dtype=np.float64)
        which = np.random.choice(len(self.mix), size=n_envs, p=self.mix)
        pos = agent_pos_from_obs(self._last_obs, self.obs_type)

        uniform = which == 1
        if uniform.any():
            out[uniform] = np.random.uniform(-1, 1, size=(int(uniform.sum()), out.shape[1]))

        demo = which == 2
        if demo.any() and self.demo_offsets is not None:
            pick = np.random.randint(0, len(self.demo_offsets), size=int(demo.sum()))
            targets = pos[demo][:, None, :] + self.demo_offsets[pick]      # re-anchored shape
            out[demo] = encode(targets, chunk=self.chunk)

        smooth = which == 3
        if smooth.any():
            out[smooth] = self._smooth_chunks(pos[smooth])
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def _smooth_chunks(self, pos):
        """A random walk at demo scale, anchored at the agent -- a plausible TRAJECTORY.

        Uniform over the 16-d cube is a uniform prior over target SEQUENCES, which is NOT a
        uniform prior over behaviours: it jumps the commanded target a mean 171 px per step
        against the demonstrations' 4 px median, so the agent slams across the table eight
        times per chunk and scatters the block. Measured at the tightest curriculum, uniform
        chunks solve 0/40 where demo-shaped chunks solve 3/40 -- so under a sparse reward the
        uniform arm does not merely explore inefficiently, it destroys the near-goal states the
        curriculum exists to create, and the top rung collects no terminals at all.

        The action SPACE stays absolute, because invertibility is what makes the Q a verifier.
        Only the exploration PRIOR over it changes. A small uniform arm is kept: Q does need to
        learn that the absurd chunks are bad.
        """
        step = np.random.normal(0.0, self.smooth_scale, size=(len(pos), self.chunk, 2))
        targets = np.clip(pos[:, None, :] + np.cumsum(step, axis=1), 0.0, WS)
        return encode(targets, chunk=self.chunk)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        from gymnasium import spaces

        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # NOT `action_space.sample()`, which SB3 uses by default: that is the uniform arm,
            # and `learning_starts` steps of it is `learning_starts` steps of scattering the
            # block. The actor's share is redistributed to the two structured arms, since an
            # untrained actor has nothing to contribute yet.
            warm = self.mix.copy()
            warm[2:] += warm[0] / 2.0
            warm[0] = 0.0
            saved, self.mix = self.mix, warm / warm.sum()
            try:
                unscaled = self._mixture_actions(n_envs)
            finally:
                self.mix = saved
        else:
            unscaled = self._mixture_actions(n_envs)
        if isinstance(self.action_space, spaces.Box):
            scaled = self.policy.scale_action(unscaled)
            if action_noise is not None:
                scaled = np.clip(scaled + action_noise(), -1, 1)
            return self.policy.unscale_action(scaled), scaled
        return unscaled, unscaled

    # ------------------------------------------------------------------ the verifier head
    def _features(self, obs):
        """The critic's features, DETACHED -- the head must not move the encoder either."""
        with th.no_grad():
            return self.policy.critic.extract_features(obs, self.policy.critic.features_extractor)

    def _candidate_actions(self, obs, m):
        """{ tanh(mu(s)) } u { m-1 actor samples } -> (m, B, action_dim)."""
        with th.no_grad():
            best = self.policy.actor(obs, deterministic=True)
            rest = [self.policy.actor(obs, deterministic=False) for _ in range(m - 1)]
        return th.stack([best] + rest, dim=0)

    @th.no_grad()
    def bon_target(self, next_obs, r_tau, d_tau):
        """`r + gamma * (1-d) * max over candidates` -- the operation BON itself performs.

        A separate method so it can be asserted directly. Ensemble-MEAN inside the max, because
        that is the estimator the deployment ranks on; the `min` that controls overestimation
        belongs to algorithms whose backup is over a single next action, not over a best-of-M.
        Taking the actor's expectation here instead would learn Q^pi, which is not what
        best-of-N evaluates.
        """
        nxt = self._features(next_obs)
        cand = self._candidate_actions(next_obs, self.bon_candidates)
        q_next = th.stack([self.policy.bon_head_target(nxt, cand[m]).mean(0)
                           for m in range(cand.shape[0])], dim=0)             # (M, B, n_tau)
        return r_tau + self.gamma * (1.0 - d_tau) * q_next.max(0).values

    def _train_bon_head(self, gradient_steps, batch_size):
        buf = self.replay_buffer
        if not hasattr(buf, "sample_chunk"):
            return
        losses, live_fracs = [], []
        for _ in range(gradient_steps):
            data, (r_tau, d_tau, live) = buf.sample_chunk(batch_size, env=self._vec_normalize_env)
            target = self.bon_target(data.next_observations, r_tau, d_tau)
            q = self.policy.bon_head(self._features(data.observations), data.actions)  # (E,B,T)
            # masked per RUNG: a transition past its rung's threshold is in an episode that has
            # already terminated for that rung, and training on it with done=False is a lie.
            mask = live.unsqueeze(0).expand_as(q)
            denom = mask.sum().clamp(min=1.0)
            loss = (((q - target.unsqueeze(0)) ** 2) * mask).sum() / denom
            self.policy.bon_optimizer.zero_grad()
            loss.backward()
            self.policy.bon_optimizer.step()
            losses.append(loss.item())
            live_fracs.append(live.mean(0).detach().cpu().numpy())
        polyak_update(self.policy.bon_head.parameters(),
                      self.policy.bon_head_target.parameters(), self.tau)
        self.logger.record("bon/head_loss", float(np.mean(losses)))
        for k, tau in enumerate(self.tau_ladder):
            self.logger.record(f"bon/live_frac_tau{tau:.2f}", float(np.mean(live_fracs, 0)[k]))

    def train(self, gradient_steps, batch_size=64):
        super().train(gradient_steps, batch_size)
        self._train_bon_head(gradient_steps, batch_size)

    # ------------------------------------------------------------------ the deliverable
    @th.no_grad()
    def q_values(self, obs, actions, rung=-1):
        """Q(s, chunk) for the ranking rung -> (B,). The verifier's whole interface."""
        obs_t, _ = self.policy.obs_to_tensor(obs)
        a = th.as_tensor(np.asarray(actions, dtype=np.float32), device=self.device)
        return self.policy.bon_head.mean(self._features(obs_t), a)[:, rung]
