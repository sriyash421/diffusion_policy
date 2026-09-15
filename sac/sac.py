"""SAC over action chunks, with a behaviour mixture and the BON verifier head.

TWO THINGS ARE DIFFERENT from stock SAC, and both exist for the same reason: the Q is going to
be asked about actions a DIFFUSION POLICY proposes, not actions this actor proposes.

1. `_sample_action` draws from a MIXTURE, not from the actor. SAC is off-policy, so any
   behaviour policy is legal, and a buffer fed only by the actor yields a Q that is sharp
   exactly where it is not needed and flat where it is. The mixture keeps uniform chunks and
   re-anchored demo chunks flowing past `learning_starts`, not just during it.

   The override is in `_sample_action` and NOT in an env wrapper, deliberately: SB3 stores the
   `buffer_action` this method returns, so a wrapper that swapped the action afterwards would
   write one action into the buffer and execute another. Nothing raises; the Q just regresses
   on labels that belong to different actions.

2. The verifier head is updated here, after `super().train()`, from its own minibatch. See
   policies.py for why it is not SAC's critic.
"""

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

from sac.chunk_codec import decode, encode
from sac.config import WS
from sac.policies import attach_bon_head


def agent_pos_from_obs(obs, obs_type):
    """Agent position in ARENA PIXELS, out of the observation the policy sees.

    Everything the policy sees is in [-1, 1] (PushTGymEnv's convention), so this inverts
    `_normalise`. The index is the one place each arm's layout is asserted.
    """
    if obs_type == "image":
        norm = np.asarray(obs["agent_pos"])
    elif obs_type == "keypoint":
        norm = np.asarray(obs)[..., 18:20]      # [9 block kps (18), agent xy (2)] then the mask
    else:
        norm = np.asarray(obs)[..., 0:2]        # state: [agent xy, block xy, cos, sin]
    return (norm + 1.0) * (WS / 2)


class ChunkSAC(SAC):
    """SAC whose action is a chunk, plus the hard max-over-candidates verifier head."""

    def __init__(self, *args, obs_type="keypoint", tau_ladder=(0.95,), demo_chunks=None,
                 chunk_action_mode="absolute", chunk_scale=64.0, chunk=8,
                 mix=(0.55, 0.20, 0.20, 0.05), bon_net_arch=(256, 256), bon_n_critics=2,
                 bon_lr=3e-4, bon_candidates=8, **kwargs):
        # set BEFORE super().__init__(), which calls _setup_model() and therefore needs them
        self.obs_type = obs_type
        self.tau_ladder = tuple(float(t) for t in tau_ladder)
        self.chunk_action_mode, self.chunk_scale, self.chunk = chunk_action_mode, float(chunk_scale), int(chunk)
        self.mix = np.asarray(mix, dtype=np.float64) / np.sum(mix)
        self.bon_candidates = int(bon_candidates)
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
            out[demo] = encode(targets, pos[demo], mode=self.chunk_action_mode,
                               scale=self.chunk_scale, chunk=self.chunk)

        wide = which == 3
        if wide.any():
            # the near-but-not-on-policy shell: the actor's own proposal, roughened
            out[wide] = out[wide] + np.random.normal(0, 0.3, size=out[wide].shape)
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        from gymnasium import spaces

        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            unscaled = np.array([self.action_space.sample() for _ in range(n_envs)])
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

    def _train_bon_head(self, gradient_steps, batch_size):
        buf = self.replay_buffer
        if not hasattr(buf, "sample_chunk"):
            return
        losses, live_fracs = [], []
        for _ in range(gradient_steps):
            data, (r_tau, d_tau, live) = buf.sample_chunk(batch_size, env=self._vec_normalize_env)
            with th.no_grad():
                nxt = self._features(data.next_observations)
                cand = self._candidate_actions(data.next_observations, self.bon_candidates)
                # HARD max over candidates -- the operation BON itself performs. Ensemble-mean
                # inside the max, because that is the estimator the deployment ranks on.
                q_next = th.stack([self.policy.bon_head_target(nxt, cand[m]).mean(0)
                                   for m in range(cand.shape[0])], dim=0)     # (M, B, n_tau)
                target = r_tau + self.gamma * (1.0 - d_tau) * q_next.max(0).values
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
