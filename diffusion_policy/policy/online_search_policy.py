"""Online search policy.

Learns ``pi(a* | h_t, c)`` where ``h_t`` is the expert observation at step ``t`` and
``c`` is a context built from on-policy rollouts of the current policy. Each rollout is
run in a live PushT env reset to the expert's start state; every rollout step's obs
carries a per-step ``feedback`` channel (goal-T vs achieved-T keypoint displacement),
so feedback is simply part of the obs -- there is no separate feedback token.

Sequence design: context tokens and expert tokens are both obs tokens (encoder ->
obs_projection); they are concatenated ``[context | expert]`` and run through a causal
GPT2, so each expert token attends to the whole context prefix plus earlier expert
steps. Supervision is a masked NLL of the expert actions over the full expert
trajectory. ``n_trajs=0`` degenerates to pure BC (no env, empty context).
"""
from typing import Dict, Any
import numpy as np
import tqdm
import torch
import torch.nn as nn

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.pusht.feedback_util import block_pose_from_feedback
from torch.distributions import Normal

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from transformers import GPT2Config, GPT2Model


class OnlineSearchPolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict[str, Any],
            obs_encoder: MultiImageObsEncoder,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
            hidden_dim: int = 512,
            hidden_depth: int = 4,
            n_head: int = 4,
            n_positions: int = 1024,
            n_trajs: int = 0,
            n_envs: int = 32,
            max_rollout_steps: int = 300,
            use_async_envs: bool = True,
            corrupt_obs: bool = False,
            mask_obs: bool = False,
            concat_obs: bool = False,
            **kwargs):
        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        assert horizon == 1 and n_action_steps == 1, \
            "Currently only support horizon=1 and n_action_steps=1"

        obs_feature_dim = obs_encoder.output_shape()[0]
        self.obs_encoder = obs_encoder
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.hidden_dim = hidden_dim
        self.normalizer = LinearNormalizer()

        # context / rollout config
        self.n_trajs = n_trajs
        self.n_envs = n_envs
        self.max_rollout_steps = max_rollout_steps
        self.use_async_envs = use_async_envs
        self._vec_env = None  # lazily built, only when n_trajs > 0
        self.kwargs = kwargs

        # legacy flags kept for config compatibility (default off)
        self.corrupt_obs = corrupt_obs
        self.mask_obs = mask_obs
        self.concat_obs = concat_obs

        # one token per obs frame (context step or expert step)
        self.obs_projection = nn.Linear(obs_feature_dim, hidden_dim)

        cfg = GPT2Config(
            n_positions=n_positions,
            n_embd=hidden_dim,
            n_layer=hidden_depth,
            n_head=n_head,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.transformer = GPT2Model(cfg)

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_limits = (-5.0, 2.0)

        self.obs_noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.001,
            beta_end=0.02,
            prediction_type="epsilon",
        )

    # ------------------------------------------------------------------ utils
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def corrupt_obs_features(self, obs_features):
        """Optionally noise context features to simulate corrupted context."""
        if not self.corrupt_obs:
            return obs_features
        noise = torch.randn_like(obs_features)
        bsz = obs_features.shape[0]
        timesteps = torch.randint(
            0, self.obs_noise_scheduler.config.num_train_timesteps,
            (bsz,), device=obs_features.device).long()
        return self.obs_noise_scheduler.add_noise(obs_features, noise, timesteps)

    def _encode_obs(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a (normalized) obs dict of shape (B, T, ...) into obs tokens (B, T, hidden)."""
        value = next(iter(nobs.values()))
        B, T = value.shape[:2]
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))  # (B*T, ...)
        feats = self.obs_encoder(this_nobs).reshape(B, T, -1)  # (B, T, obs_feature_dim)
        return self.obs_projection(feats)  # (B, T, hidden)

    def _get_vec_env(self):
        if self._vec_env is None:
            # lazy import so n_trajs=0 never needs gym / the env stack
            from diffusion_policy.env.pusht.pusht_feedback import build_parallel
            self._vec_env = build_parallel(
                n_envs=self.n_envs,
                use_async=self.use_async_envs,
                n_obs_steps=1,
                n_action_steps=1,
                max_episode_steps=self.max_rollout_steps,
                legacy=True)
        return self._vec_env

    def close(self):
        """Shut down the lazily-built context-rollout worker pool.

        The pool is a set of subprocesses, so it does not go away on garbage collection;
        the training workspace calls this during teardown (see
        TrainMLPImageWorkspace._close_worker_pools). Not done via __del__ --
        interpreter-shutdown ordering makes that unreliable for subprocess pools.
        """
        vec_env = self._vec_env
        self._vec_env = None
        if vec_env is not None:
            vec_env.close()

    # ------------------------------------------------------------------ core
    def forward(
            self,
            context_features: torch.Tensor,      # B, C, hidden (or None / C=0)
            expert_obs_features: torch.Tensor,   # B, E, hidden
            expert_actions: torch.Tensor = None,  # target only, unused as input
            context_mask: torch.Tensor = None,   # B, C
            expert_mask: torch.Tensor = None,    # B, E
        ) -> Normal:
        """Run causal GPT2 over [context | expert] and read out a Normal per expert step."""
        B, E = expert_obs_features.shape[:2]
        C = 0 if context_features is None else context_features.shape[1]

        if expert_mask is None:
            expert_mask = torch.ones(B, E, device=expert_obs_features.device)

        if C > 0:
            if self.corrupt_obs and self.training:
                context_features = self.corrupt_obs_features(context_features)
            if context_mask is None:
                context_mask = torch.ones(B, C, device=context_features.device)
            seq = torch.cat([context_features, expert_obs_features], dim=1)
            key_mask = torch.cat([context_mask, expert_mask], dim=1)
        else:
            seq = expert_obs_features
            key_mask = expert_mask

        h = self.transformer(
            inputs_embeds=seq,
            attention_mask=key_mask.to(seq.dtype),
        ).last_hidden_state  # B, C+E, hidden

        h_expert = h[:, C:, :]  # B, E, hidden -- the expert positions
        mean = self.mean_head(h_expert)
        log_std = self.log_std_head(h_expert).clamp(
            min=self.log_std_limits[0], max=self.log_std_limits[1])
        std = torch.exp(log_std)
        return Normal(mean, std)  # B, E, action_dim

    def predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor],
            feedback_dict: Dict[str, torch.Tensor] = None,
            done_mask: torch.Tensor = None,
        ) -> Dict[str, torch.Tensor]:
        """Context-free BC readout at the last valid obs step.

        Runner-compatible: accepts an optional ``attention_mask`` inside obs_dict. This is
        the same policy used to roll out context in ``generate_context``.
        """
        obs = dict(obs_dict)
        attention_mask = obs.pop('attention_mask', None)
        nobs = self.normalizer.normalize(obs)
        value = next(iter(nobs.values()))
        B, T = value.shape[:2]

        expert_tokens = self._encode_obs(nobs)  # B, T, hidden
        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=self.device)
        dist = self.forward(
            context_features=None,
            expert_obs_features=expert_tokens,
            expert_mask=attention_mask)
        naction = dist.rsample()  # B, T, action_dim
        action = self.normalizer['action'].unnormalize(naction)

        seq_lens = attention_mask.sum(dim=1).clamp(min=1)
        idx = (seq_lens - 1).long()
        arange = torch.arange(B, device=action.device)
        # (B, n_action_steps=1, action_dim) -- BaseImagePolicy contract (B, Ta, Da),
        # so MultiStepWrapper.step gets a leading action-step dim.
        return {
            'action': action[arange, idx].unsqueeze(1),        # B, 1, action_dim
            'action_pred': naction[arange, idx].unsqueeze(1),  # B, 1, action_dim (normalized)
        }

    @torch.no_grad()
    def generate_context_rollouts(self, init_states: torch.Tensor, n_trajs: int):
        """Per-trajectory list of n_trajs rollout token tensors (on CPU) for subsampling.

        Returns a list of length B, each a list of n_trajs (T_r, hidden) tensors.
        """
        B = init_states.shape[0]
        if n_trajs <= 0:
            return [[] for _ in range(B)]

        states_np = init_states.detach().cpu().numpy().astype(np.float64)
        out = [None] * B
        for start in tqdm.tqdm(range(0, B, self.n_envs),
                desc="generating rollouts", leave=False):
            idxs = list(range(start, min(start + self.n_envs, B)))
            chunk = self._incontext_rollout_chunk(states_np[idxs], n_trajs)
            for j, b in enumerate(idxs):
                out[b] = chunk[j]
        return out

    def pad_context_rows(self, rows):
        """List (len B) of (C_b, hidden) tensors -> (B, C, hidden), (B, C) bool on device."""
        B = len(rows)
        C = max((t.shape[0] for t in rows), default=0)
        context = torch.zeros(B, C, self.hidden_dim, device=self.device, dtype=self.dtype)
        mask = torch.zeros(B, C, device=self.device, dtype=torch.bool)
        for b, t in enumerate(rows):
            c = t.shape[0]
            if c:
                context[b, :c] = t.to(context)
                mask[b, :c] = True
        return context, mask

    @torch.no_grad()
    def generate_context(self, init_states: torch.Tensor, n_trajs: int):
        """Roll out the policy from each start state; concat all n_trajs rollouts per row."""
        rollouts = self.generate_context_rollouts(init_states, n_trajs)
        rows = [torch.cat(r, dim=0) if len(r) else torch.zeros(0, self.hidden_dim)
                for r in rollouts]
        return self.pad_context_rows(rows)

    def _incontext_rollout_chunk(self, states_chunk: np.ndarray, n_trajs: int):
        """Autoregressively generate n_trajs rollouts per env with the full transformer.

        The env is reset between rollouts, but the GPT2 KV cache is NOT: rollout k is
        produced while attending to all previous rollouts (feedback is in the obs) and
        its own partial trajectory. Returns list (len==len(states_chunk)) of n_trajs
        (T_r, hidden) token tensors.
        """
        from diffusion_policy.env.pusht.pusht_feedback import set_reset_states
        vec = self._get_vec_env()
        m = states_chunk.shape[0]
        if m < self.n_envs:
            pad = np.repeat(states_chunk[-1:], self.n_envs - m, axis=0)
            states_full = np.concatenate([states_chunk, pad], axis=0)
        else:
            states_full = states_chunk

        round_tokens = [[[] for _ in range(self.n_envs)] for _ in range(n_trajs)]
        past = None
        for k in range(n_trajs):
            set_reset_states(vec, list(states_full))
            obs = vec.reset()  # dict, each (n_envs, 1, ...)
            done = np.zeros(self.n_envs, dtype=bool)
            for _ in range(self.max_rollout_steps):
                obs_t = {key: torch.as_tensor(v, device=self.device, dtype=self.dtype)
                         for key, v in obs.items()}
                token = self._encode_obs(self.normalizer.normalize(obs_t))  # (n_envs, 1, hidden)
                gpt = self.transformer(
                    inputs_embeds=token, past_key_values=past, use_cache=True)
                past = gpt.past_key_values
                hidden = gpt.last_hidden_state[:, -1]  # (n_envs, hidden)
                log_std = self.log_std_head(hidden).clamp(
                    min=self.log_std_limits[0], max=self.log_std_limits[1])
                naction = Normal(self.mean_head(hidden), torch.exp(log_std)).rsample()
                action = self.normalizer['action'].unnormalize(naction)

                for i in range(self.n_envs):
                    round_tokens[k][i].append(token[i, -1])

                obs, _, step_done, _ = vec.step(
                    action.detach().cpu().numpy()[:, None, :])
                done |= np.asarray(step_done, dtype=bool)
                if done[:m].all():
                    break

        return [[torch.stack(round_tokens[k][i], dim=0).cpu() for k in range(n_trajs)]
                for i in range(m)]

    def compute_loss(self, batch, context_features=None, context_mask=None):
        obs = batch['obs']
        target = self.normalizer['action'].normalize(batch['action'])  # (B, E, action_dim)
        attention_mask = batch['attention_mask']            # (B, E)
        expert_mask = batch.get('expert_mask', None)        # (B, E, 1) or None

        if context_features is None:
            # (B, 5) env reset state. The block pose is reconstructed from feedback, which
            # is exact and is a declared obs key -- the dataset carries no block_pos.
            block_pose = torch.as_tensor(
                block_pose_from_feedback(
                    obs['feedback'][:, 0].detach().cpu().numpy()),
                dtype=obs['agent_pos'].dtype, device=obs['agent_pos'].device)
            init_states = torch.cat([obs['agent_pos'][:, 0], block_pose], dim=-1)  # (B, 5)
            context_features, context_mask = self.generate_context(init_states, self.n_trajs)

        nobs = self.normalizer.normalize(obs)               # image, agent_pos, feedback
        expert_tokens = self._encode_obs(nobs)              # (B, E, hidden)

        dist = self.forward(
            context_features=context_features,
            expert_obs_features=expert_tokens,
            expert_actions=target,
            context_mask=context_mask,
            expert_mask=attention_mask)

        loss_mask = attention_mask.to(target.dtype)         # (B, E)
        if expert_mask is not None:
            loss_mask = loss_mask * expert_mask[..., 0]
        log_prob = dist.log_prob(target).sum(dim=-1)        # (B, E)
        loss = -(log_prob * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        return loss
