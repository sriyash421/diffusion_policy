from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.diffusion_transformer_search_policy import (
    SearchTransformerForDiffusion,
)
from diffusion_policy.common.pytorch_util import dict_apply
from torch.distributions import Normal

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce
from transformers import GPT2Config, GPT2Model

from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

import numpy as np

# SeedSequence keys that separate the per-episode noise streams (see set_sample_seeds).
_SEED_KEY_CANDIDATE = 0
_SEED_KEY_OBS = 1


def _maybe_set_executed_chunk_window(verifier, n_obs_steps, n_action_steps):
    """Derive the chunk-sum verifier's scoring window from the policy.

    When the verifier opts into ``score_executed_chunk`` (and no explicit
    ``score_indices`` was given), set ``score_indices`` to the EXECUTED action
    window — horizon steps ``[n_obs_steps-1 : n_obs_steps-1+n_action_steps]`` —
    so the chunk-sum scoring always covers exactly the executed action steps and
    tracks ``n_action_steps`` instead of a hardcoded index list.
    """
    if not getattr(verifier, "score_executed_chunk", False):
        return
    if getattr(verifier, "score_indices", None) is not None:
        return
    start = int(n_obs_steps) - 1
    verifier.score_indices = list(range(start, start + int(n_action_steps)))


def _run_search(policy, obs_dict, n_actions, **predict_kwargs):
    """Run the policy's autoregressive verifier search.

    When ROBOMONKEY_KV_PREFIX=1 and the verifier supports it (in-process), the
    batch is split into image-chunks of ROBOMONKEY_KV_CHUNK; each chunk runs all
    `n_actions` rounds while the verifier reuses each image's cached prefix KV,
    then the cache is cleared so GPU memory stays bounded by the chunk size.
    Otherwise this is a transparent pass-through to ``policy.predict_action``.
    Bit-equivalent either way (the search is per-row independent).
    """
    import os as _os
    verifier = policy.verifier
    chunk = int(_os.environ.get("ROBOMONKEY_KV_CHUNK", "16"))
    enabled = (_os.environ.get("ROBOMONKEY_KV_PREFIX", "0") == "1"
               and getattr(verifier, "supports_kv_prefix", False) and chunk > 0)
    if not enabled:
        return policy.search_candidates(
            obs_dict, verifier=verifier, n_actions=n_actions, **predict_kwargs)

    B = None
    for _v in obs_dict.values():
        if torch.is_tensor(_v):
            B = _v.shape[0]
            break
    if B is None or B <= chunk:
        verifier.set_kv_prefix(True)
        try:
            verifier.clear_prefix_cache()
            return policy.search_candidates(
                obs_dict, verifier=verifier, n_actions=n_actions, **predict_kwargs)
        finally:
            verifier.set_kv_prefix(False)
            verifier.clear_prefix_cache()

    verifier.set_kv_prefix(True)
    acts, vals = [], []
    try:
        for s in range(0, B, chunk):
            e = min(s + chunk, B)
            sub_obs = {k: (v[s:e] if torch.is_tensor(v) else v)
                       for k, v in obs_dict.items()}
            sub_kw = {k: (v[s:e] if torch.is_tensor(v) else v)
                      for k, v in predict_kwargs.items()}
            verifier.clear_prefix_cache()
            a, vv = policy.search_candidates(
                sub_obs, verifier=verifier, n_actions=n_actions, **sub_kw)
            acts.append(a)
            vals.append(vv)
    finally:
        verifier.set_kv_prefix(False)
        verifier.clear_prefix_cache()
    return torch.cat(acts, dim=0), torch.cat(vals, dim=0)


class SearchPolicyRoboMonkey(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict[str, Any],
            obs_encoder,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
            verifier,
            hidden_dim: int = 512,
            hidden_depth: int = 4,
            corrupt_obs: bool = False,
            obs_noise_level: Optional[float] = None,
            mask_obs: bool = False,
            concat_obs: bool = False,
            pg_lambda: float = 0.0,
            **kwargs):
        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # Only low_dim keys go into the policy obs encoder. RGB / depth keys
        # may still be present in shape_meta (e.g. for an L2S verifier that
        # scores actions using the agentview image) — those are handed to the
        # verifier via obs_dict but kept out of the policy's state vector.
        self._state_keys = [
            k for k, v in shape_meta['obs'].items()
            if v.get('type', 'low_dim') == 'low_dim'
        ]
        obs_feature_dim = obs_encoder.output_shape()[0]
        self.obs_encoder = obs_encoder
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.normalizer = LinearNormalizer()
        self.verifier = verifier
        # Opt-in REINFORCE/advantage policy-gradient term (0.0 -> pure BC).
        self._pg_lambda = float(pg_lambda)
        _maybe_set_executed_chunk_window(verifier, n_obs_steps, n_action_steps)
        self.kwargs = kwargs
        self.mask_obs = mask_obs
        self.concat_obs = concat_obs
        if self.concat_obs:
            assert not mask_obs, "Cannot use both concat_obs and mask_obs"
            print("Using concat_obs: concatenating obs features with action-value features at each step")

        if self.mask_obs:
            self.first_action_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        if self.concat_obs:
            self.obs_projection = nn.Linear(obs_feature_dim * n_obs_steps, hidden_dim // 2)
            self.act_projection = nn.Linear(horizon * action_dim + 1, hidden_dim // 2)
            self.hidden_dim = hidden_dim
        else:
            self.obs_projection = nn.Linear(obs_feature_dim * n_obs_steps, hidden_dim)
            self.act_projection = nn.Linear(horizon * action_dim + 1, hidden_dim)
        self.max_actions = kwargs['max_actions']

        # Shared trunk
        cfg = GPT2Config(
            n_positions=1024,
            n_embd=hidden_dim,
            n_layer=hidden_depth,
            n_head=4,
            resid_pdrop=float(0.1),
            embd_pdrop=float(0.1),
            attn_pdrop=float(0.1),
        )
        self.transformer = GPT2Model(cfg)

        output_dim = horizon * action_dim
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.log_std_head = nn.Linear(hidden_dim, output_dim)

        self.log_std_limits = (-5.0, 2.0)

        self.corrupt_obs = corrupt_obs
        self.obs_noise_level = obs_noise_level
        self.obs_noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.001,
            beta_end=0.02,
            prediction_type="epsilon",
        )
        if corrupt_obs:
            level_str = "random" if obs_noise_level is None else f"fixed={obs_noise_level}"
            print(f"Corrupting obs with a separate noise scheduler (level={level_str})")

    def corrupt_obs_features(self, obs_features):
        # Tunable mode (obs_noise_level set): apply the configured level at both
        # train and eval. Random mode (None): only at train time.
        if not self.corrupt_obs:
            return obs_features
        if self.obs_noise_level is None and not self.training:
            return obs_features

        obs_noise = torch.randn_like(obs_features)
        bsz = obs_features.shape[0]
        max_t = self.obs_noise_scheduler.config.num_train_timesteps
        if self.obs_noise_level is None:
            timesteps = torch.randint(
                0, max_t, (bsz,), device=obs_features.device
            ).long()
        else:
            t = int(round(float(self.obs_noise_level) * (max_t - 1)))
            t = max(0, min(max_t - 1, t))
            timesteps = torch.full(
                (bsz,), t, device=obs_features.device, dtype=torch.long
            )
        return self.obs_noise_scheduler.add_noise(
            obs_features, obs_noise, timesteps)

    def forward(
            self,
            obs_features: torch.Tensor, # B, hidden_dim
            action_value_features: torch.Tensor = None, # B, max_actions, hidden_dim
            # attention_mask: torch.Tensor = None, # B, max_actions
        ) -> Normal:
        """
        Returns a factorized Normal over an action trajectory of shape
        (batch, horizon, action_dim).
        """

        B = obs_features.shape[0]
        if self.corrupt_obs:
            obs_features = self.corrupt_obs_features(obs_features)

        if self.mask_obs:
            h = self.first_action_mlp(obs_features) # B, hidden_dim
            h = h.unsqueeze(1) # B, 1, hidden_dim
            mean = self.mean_head(h).reshape(B, 1, self.horizon, self.action_dim)
            log_std = self.log_std_head(h).reshape(B, 1, self.horizon, self.action_dim)
            log_std = log_std.clamp(
                min=self.log_std_limits[0],
                max=self.log_std_limits[1]
            )
            std = torch.exp(log_std)
            first_action_mean = mean
            first_action_std = std

        if action_value_features is None:
             # B, 1, horizon, action_dim
            if self.mask_obs:
                return Normal(first_action_mean, first_action_std)
            if self.concat_obs:
                inputs = torch.cat([obs_features.unsqueeze(1), torch.zeros(B, 1, self.hidden_dim // 2, device=obs_features.device)], dim=-1) # B, 1, hidden_dim
            else:
                inputs = obs_features.unsqueeze(1) # B, 1, hidden_dim
            T = 1
        else:
            if self.mask_obs:
                inputs = action_value_features
                T = action_value_features.shape[1]
            else:
                if self.concat_obs:
                    first_token = torch.cat([obs_features.unsqueeze(1), torch.zeros(B, 1, self.hidden_dim // 2, device=obs_features.device)], dim=-1) # B, 1, hidden_dim
                    action_value_features = torch.cat([obs_features.unsqueeze(1).expand(-1, action_value_features.shape[1], -1), action_value_features], dim=-1) # B, max_actions, hidden_dim
                    inputs = torch.cat([first_token, action_value_features], dim=1) # B, max_actions+1, hidden_dim
                else:
                    inputs = torch.cat([obs_features.unsqueeze(1), action_value_features], dim=1)
                T = 1 + action_value_features.shape[1]

        attention_mask = torch.ones(B, T, device=obs_features.device)

        h = self.transformer(
            inputs_embeds=inputs,
            attention_mask=attention_mask
        ).last_hidden_state  # B, T+1, hidden_dim

        mean = self.mean_head(h).reshape(B, T, self.horizon, self.action_dim)
        log_std = self.log_std_head(h).reshape(B, T, self.horizon, self.action_dim)
        log_std = log_std.clamp(
            min=self.log_std_limits[0],
            max=self.log_std_limits[1]
        )
        std = torch.exp(log_std)
        if self.mask_obs:
            mean = torch.cat([first_action_mean, mean], dim=1)
            std = torch.cat([first_action_std, std], dim=1)
        return Normal(mean, std) # B, T, horizon, action_dim

    def _normalize_context_actions(self, actions):
        """Put context actions into the space the model works in.

        The search loop carries candidates in RAW action units because the verifier
        scores them there and the env executes them. But the model is trained against a
        NORMALIZED target, so handing it raw actions puts the two halves of the
        action+value context on wildly different scales. Normalizing here -- at the model
        boundary, not in the search loop -- keeps one tensor from serving both boundaries.

        Matches the convention already used for the policy-gradient branch, which keeps a
        normalized `na_pi` and unnormalizes a separate `a_pi_raw` copy for the verifier.
        """
        if actions is None:
            return None
        return self.normalizer['action'].normalize(actions)

    def predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor],
            actions: torch.Tensor = None,
            values: torch.Tensor = None,
        ) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        obs_value = next(iter(nobs.values()))
        B, To = obs_value.shape[:2]
        To = self.n_obs_steps

        # Encode obs: flatten all obs steps. Keep only the state keys — the
        # RGB image (if present) is for the verifier, not the policy.
        if isinstance(nobs, dict):
            this_nobs = dict_apply(
                {k: nobs[k] for k in self._state_keys},
                lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
        else:
            this_nobs = nobs[:, :To, ...].reshape(-1, *nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, To, -1)
        obs_features = nobs_features.reshape(B, -1)
        obs_features = self.obs_projection(obs_features)

        # Prepare action_value_features by concatenating actions and values, and projecting to hidden_dim
        if actions is None:
            action_value_features = None
        else:
            actions = self._normalize_context_actions(actions)
            actions = actions.reshape(B, actions.shape[1], -1).to(obs_features.device) # B, max_actions, horizon*action_dim
            values = values.to(obs_features.device) # B, max_actions
            action_value_input = torch.cat([actions, values.unsqueeze(-1)], dim=-1)
            action_value_features = self.act_projection(action_value_input)

        dist = self.forward(obs_features, action_value_features) # B, max_actions, horizon, action_dim
        naction_pred = dist.rsample() # B, max_actions, horizon, action_dim
        action_pred = self.normalizer['action'].unnormalize(naction_pred) # B, max_actions, horizon, action_dim

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, :, start:end] # B, max_actions, n_action_steps, action_dim
        return {
            'action': action,
            'action_pred': action_pred
        }

    def search_candidates(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions,
            return_scores: bool = False,
            return_subgoals: bool = False,
    ):
        actions = None
        values = None
        for _ in range(n_actions):
            new_action = self.predict_action(
                obs_dict,
                actions=actions,
                values=values
            )['action_pred'][:, -1]  # B, horizon, action_dim

            new_value = verifier.get_value(obs_dict, new_action)
            if actions is None:
                actions = new_action.unsqueeze(1)
                values = new_value.unsqueeze(1)
            else:
                actions = torch.cat([actions, new_action.unsqueeze(1)], dim=1)
                values = torch.cat([values, new_value.unsqueeze(1)], dim=1)
        # (actions, values[, scores][, subgoals]), as SearchProcedureMixin.search_candidates
        out = (actions, values) # B, n_actions, horizon, action_dim and B, n_actions
        if return_scores:
            out = out + (values,)
        if return_subgoals:
            out = out + (None,)
        return out

    @torch.inference_mode()
    def predict_n_actions(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions
    ):
        # return self.search_candidates(obs_dict, verifier, n_actions)
        if n_actions <= self.max_actions:
            actions, values = self.search_candidates(obs_dict, verifier, n_actions)
            actions = actions
            values = values
            return actions, values
        else:
            # If n_actions exceeds max_actions, we can call predict_action multiple times.
            actions, values = self.search_candidates(obs_dict, verifier, self.max_actions)

            all_actions = actions.clone()
            all_values = values.clone()

            action_history = actions[:, 1:]
            value_history = values[:, 1:]
            for i in range(self.max_actions, n_actions):
                new_action = self.predict_action(
                    obs_dict,
                    actions=action_history,
                    values=value_history
                )['action_pred'][:, -1]  # B, horizon, action_dim

                new_value = verifier.get_value(obs_dict, new_action)

                all_actions = torch.cat([all_actions, new_action.unsqueeze(1)], dim=1)
                all_values = torch.cat([all_values, new_value.unsqueeze(1)], dim=1)

                action_history = torch.cat([action_history[:, 1:], new_action.unsqueeze(1)], dim=1)
                value_history = torch.cat([value_history[:, 1:], new_value.unsqueeze(1)], dim=1)

            return all_actions, all_values  # B, n_actions, horizon, action_dim and B, n_actions


    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def generate_search_context(self, obs_dict, action=None):
        """The search context ``compute_loss`` conditions on: ``(actions, values)``.

        Same contract as ``SearchPolicyRoboMonkeyDiffusion.generate_search_context`` -- see
        there for why this is ``no_grad`` and not ``inference_mode`` (buffered inference
        tensors cannot legally be fed to an autograd-recorded op on a later step).
        """
        search_obs = dict(obs_dict)
        if action is not None:
            search_obs['action'] = action
        with torch.no_grad():
            return _run_search(self, search_obs, self.max_actions - 1)

    def compute_loss(self, batch, actions=None, values=None, return_aux=False):
        """NLL of the expert action under a best-of-n search context.

        ``actions``/``values`` optionally supply a PRE-GENERATED context (see
        ``generate_search_context``); ``None`` generates it here, which is the offline path.
        ``return_aux`` is accepted for interface parity with the diffusion search policies,
        but this head is Gaussian rather than a denoiser, so there is no
        ``(noisy_trajectory, timestep)`` to reproduce and no ``predict_epsilon`` to feed --
        the aux dict carries only the context.
        """
        nobs = self.normalizer.normalize(batch['obs'])
        target_actions = self.normalizer['action'].normalize(batch['action'])
        B, T, Da = target_actions.shape
        To = self.n_obs_steps
        assert T == self.horizon, \
            f"Expected action horizon {self.horizon}, got {T}"
        assert Da == self.action_dim, \
            f"Expected action dim {self.action_dim}, got {Da}"

        # Encode obs
        if isinstance(nobs, dict):
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
        else:
            this_nobs = nobs[:, :To, ...].reshape(-1, *nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, To, -1)
        obs_features = nobs_features.reshape(B, -1)
        obs_features = self.obs_projection(obs_features) # B, hidden_dim

        if actions is None:
            actions, values = self.generate_search_context(batch['obs'])
        raw_actions = actions

        actions = self._normalize_context_actions(actions)
        actions = actions.reshape(B, self.max_actions-1, -1) # B, max_actions, horizon*action_dim
        action_value_input = torch.cat([actions, values.unsqueeze(-1)], dim=-1) # B, max_actions, horizon*action_dim + 1
        action_value_features = self.act_projection(action_value_input) # B, max_actions, hidden_dim
        dist = self.forward(obs_features, action_value_features)  # B, max_actions, horizon, action_dim
        target_actions = target_actions.unsqueeze(1).expand(-1, self.max_actions, -1, -1) # B, max_actions, horizon, action_dim
        log_prob = dist.log_prob(target_actions).sum(dim=(-1, -2)) # B, max_actions
        loss = -log_prob.mean()

        # --- Opt-in advantage policy-gradient term (REINFORCE on policy samples) ---
        # grad = E[ grad_logpi(a_pi) * (Q(s, a_pi) - Q(s, a*)) ]; a* = expert action
        # is the baseline, Q is the FROZEN verifier (scalar coefficient, stop-grad —
        # no backprop through Q). Added to the BC loss as loss = loss_BC + lambda*loss_PG.
        if self._pg_lambda > 0.0:
            ma = self.max_actions
            a_pi_n = dist.sample()                                    # (B, ma, T, Da) normalized, fixed sample
            a_pi_raw = self.normalizer['action'].unnormalize(a_pi_n)  # raw action space for the verifier
            img_key = getattr(self.verifier, 'image_obs_key', 'agentview_image')
            img = batch['obs'][img_key]                              # (B, To, ...)
            img_rep = img.unsqueeze(1).expand(B, ma, *img.shape[1:]).reshape(B * ma, *img.shape[1:])
            with torch.no_grad():
                q_pi = self.verifier.get_value(
                    {img_key: img_rep}, a_pi_raw.reshape(B * ma, T, Da)
                ).reshape(B, ma).to(loss.device)
                q_star = self.verifier.get_value(
                    {img_key: img}, batch['action']
                ).reshape(B, 1).to(loss.device)
            adv = (q_pi - q_star).detach()                           # (B, ma) frozen advantage
            logp_pi = dist.log_prob(a_pi_n).sum(dim=(-1, -2))        # (B, ma)
            loss_pg = -(adv * logp_pi).mean()
            loss = loss + self._pg_lambda * loss_pg
        if not return_aux:
            return loss
        return loss, {'actions': raw_actions, 'values': values}


class SearchPolicyRoboMonkeyDiffusion(BaseImagePolicy):
    """RoboMonkey adaptation of :class:`DiffusionTransformerSearchPolicy`.

    Same encoder-decoder diffusion-transformer architecture as the maze
    version, with two differences:
      * The verifier is injected (Hydra-instantiated), not a hard-coded
        ``MazeVerifier``. Use any client from ``diffusion_policy.policy.verifiers``.
      * ``_state_keys`` filters the obs dict so RGB / depth keys are reserved
        for the verifier and never enter the policy's obs encoder.
    """

    # The eval harness seeds candidate + obs-corruption noise per episode via
    # set_sample_seeds (see _candidate_noise / corrupt_obs_features).
    supports_sample_seeds = True

    def __init__(
            self,
            shape_meta: dict[str, Any],
            obs_encoder,
            noise_scheduler: DDPMScheduler,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
            verifier,
            num_inference_steps=None,
            n_layer: int = 8,
            n_cond_layers: int = 0,
            n_head: int = 4,
            n_emb: int = 256,
            p_drop_emb: float = 0.0,
            p_drop_attn: float = 0.3,
            causal_attn: bool = True,
            corrupt_obs: bool = False,
            obs_noise_level: Optional[float] = None,
            pg_lambda: float = 0.0,
            pg_mode: str = 'reinforce',
            **kwargs,
        ):
        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        self._state_keys = [
            k for k, v in shape_meta['obs'].items()
            if v.get('type', 'low_dim') == 'low_dim'
        ]
        obs_feature_dim = obs_encoder.output_shape()[0]
        max_actions = kwargs['max_actions']

        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.verifier = verifier
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        _maybe_set_executed_chunk_window(verifier, n_obs_steps, n_action_steps)
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.max_actions = max_actions
        self.kwargs = kwargs
        self.step_kwargs = kwargs.get('scheduler_step_kwargs', dict())
        # Opt-in REINFORCE/advantage policy-gradient term (0.0 -> pure BC).
        # pg_mode selects how the frozen-verifier advantage enters the loss:
        #   'reinforce' -> REINFORCE on the policy sample a_pi (original behavior)
        #   'awbc'      -> advantage-weighted imitation of the expert a*
        self._pg_lambda = float(pg_lambda)
        self._pg_mode = str(pg_mode)

        # Same network as the maze / PushT search transformer. cond_encoder and
        # context_decay default to the pre-unification values so old checkpoints rebuild.
        self.model = SearchTransformerForDiffusion(
            action_dim=action_dim,
            obs_feature_dim=obs_feature_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            max_context_actions=max_actions - 1,
            n_layer=n_layer,
            n_cond_layers=n_cond_layers,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            cond_encoder=kwargs.get('cond_encoder', 'transformer'),
            context_decay=kwargs.get('context_decay', 1.0),
        )

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.corrupt_obs = corrupt_obs
        self.obs_noise_level = obs_noise_level
        self.obs_noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.001,
            beta_end=0.02,
            prediction_type="epsilon",
        )
        if corrupt_obs:
            level_str = "random" if obs_noise_level is None else f"fixed={obs_noise_level}"
            print(f"Corrupting obs with a separate noise scheduler (level={level_str})")
        # Eval-only switch: False feeds clean obs at eval even when the policy was trained
        # with a fixed obs_noise_level. Set by the eval harness (--obs-corruption clean).
        self.corrupt_obs_eval = True
        # Per-episode sampling seeds, set by the eval harness (set_sample_seeds). None ->
        # global RNG, which is what training uses.
        self._sample_seeds = None

    def set_sample_seeds(self, seeds):
        """Make eval noise a pure function of (episode seed, candidate index).

        ``seeds`` is one int per batch row (None restores the global RNG). Both the
        initial diffusion noise and the obs-corruption noise are then drawn from per-row
        CPU generators, so a row's samples do not depend on its batch position, on how
        many other envs are active, or on the device. DDIM (eta=0) draws nothing else.
        """
        self._sample_seeds = None if seeds is None else [int(s) for s in seeds]

    def _seeded_randn(self, shape, key, device, dtype):
        """(B, *shape) standard normal; row b is drawn from SeedSequence([seed_b, *key])."""
        rows = []
        for base in self._sample_seeds:
            g = torch.Generator()
            g.manual_seed(int(np.random.SeedSequence([base, *key]).generate_state(1)[0]))
            rows.append(torch.randn(tuple(shape), generator=g, dtype=torch.float32))
        return torch.stack(rows).to(device=device, dtype=dtype)

    def _check_seed_rows(self, batch_size):
        assert len(self._sample_seeds) == batch_size, (
            f'set_sample_seeds got {len(self._sample_seeds)} seeds, batch is {batch_size}')

    def _candidate_noise(self, index, batch_size, device, dtype):
        """Seeded initial noise for candidate ``index`` of this decision, or None."""
        if self._sample_seeds is None:
            return None
        self._check_seed_rows(batch_size)
        return self._seeded_randn(
            (self.horizon, self.action_dim), (_SEED_KEY_CANDIDATE, int(index)), device, dtype)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def encode_obs_cond(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        nobs = self.normalizer.normalize(obs_dict)
        obs_value = next(iter(nobs.values()))
        B = obs_value.shape[0]
        To = self.n_obs_steps
        if isinstance(nobs, dict):
            this_nobs = dict_apply(
                {k: nobs[k] for k in self._state_keys},
                lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]),
            )
        else:
            this_nobs = nobs[:, :To, ...].reshape(-1, *nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, To, -1)
        return self.corrupt_obs_features(nobs_features)

    def corrupt_obs_features(self, obs_features):
        # Tunable mode (obs_noise_level set): apply the configured level at
        # both train and eval — `obs_noise_level` in [0, 1] maps to a discrete
        # DDPM timestep. Random mode (None): random timestep at train time only,
        # eval is clean. corrupt_obs_eval=False also makes the tunable mode clean at eval.
        if not self.corrupt_obs:
            return obs_features
        if not self.training and (self.obs_noise_level is None or not self.corrupt_obs_eval):
            return obs_features

        bsz = obs_features.shape[0]
        if self._sample_seeds is not None and not self.training:
            self._check_seed_rows(bsz)
            obs_noise = self._seeded_randn(
                obs_features.shape[1:], (_SEED_KEY_OBS,),
                obs_features.device, obs_features.dtype)
        else:
            obs_noise = torch.randn_like(obs_features)
        max_t = self.obs_noise_scheduler.config.num_train_timesteps
        if self.obs_noise_level is None:
            timesteps = torch.randint(
                0, max_t, (bsz,), device=obs_features.device,
            ).long()
        else:
            t = int(round(float(self.obs_noise_level) * (max_t - 1)))
            t = max(0, min(max_t - 1, t))
            timesteps = torch.full(
                (bsz,), t, device=obs_features.device, dtype=torch.long,
            )
        return self.obs_noise_scheduler.add_noise(
            obs_features,
            obs_noise,
            timesteps,
        )

    def conditional_sample(
            self,
            obs_cond: torch.Tensor,
            actions: Optional[torch.Tensor] = None,
            values: Optional[torch.Tensor] = None,
            context_lengths: Optional[torch.Tensor] = None,
            generator=None,
            noise: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> torch.Tensor:
        scheduler = self.noise_scheduler
        # `noise`, when provided, is the initial trajectory (DDIM is deterministic
        # from here, so this fully determines the candidate). The search harness
        # uses it to seed candidate generation PER-ENV in tiled/batched rollouts so
        # each episode's noise is anchored to its own seed (not its batch position),
        # matching the single-env path. None -> draw from the (seeded) global RNG.
        if noise is not None:
            trajectory = noise.to(dtype=obs_cond.dtype, device=obs_cond.device)
        else:
            trajectory = torch.randn(
                size=(obs_cond.shape[0], self.horizon, self.action_dim),
                dtype=obs_cond.dtype,
                device=obs_cond.device,
                generator=generator,
            )

        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            model_output = self.model(
                trajectory,
                t,
                obs_cond=obs_cond,
                actions=actions,
                values=values,
                context_lengths=context_lengths,
            )
            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
                **kwargs,
            ).prev_sample
        return trajectory

    def _normalize_context_actions(self, actions):
        """Put context actions into the space the model works in.

        The search loop carries candidates in RAW action units because the verifier
        scores them there and the env executes them. But the model is trained against a
        NORMALIZED target, so handing it raw actions puts the two halves of the
        action+value context on wildly different scales. Normalizing here -- at the model
        boundary, not in the search loop -- keeps one tensor from serving both boundaries.

        Matches the convention already used for the policy-gradient branch, which keeps a
        normalized `na_pi` and unnormalizes a separate `a_pi_raw` copy for the verifier.
        """
        if actions is None:
            return None
        return self.normalizer['action'].normalize(actions)

    def predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor],
            actions: Optional[torch.Tensor] = None,
            values: Optional[torch.Tensor] = None,
            noise: Optional[torch.Tensor] = None,
            obs_cond: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
        """``obs_cond`` is an already-encoded (and, if enabled, corrupted) obs; the search
        loop encodes once per decision so every candidate sees the same observation."""
        assert 'past_action' not in obs_dict
        if obs_cond is None:
            obs_cond = self.encode_obs_cond(obs_dict)
        nsample = self.conditional_sample(
            obs_cond=obs_cond,
            actions=self._normalize_context_actions(actions),
            values=values,
            noise=noise,
            **self.step_kwargs,
        )
        action_pred = self.normalizer['action'].unnormalize(nsample)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {
            'action': action,
            'action_pred': action_pred,
        }

    def _init_candidate_noise(self, noise_seq, index, obs_cond):
        """Candidate ``index``'s initial noise: ``noise_seq`` if given, else the per-episode
        seeded stream, else None (global RNG)."""
        if noise_seq is not None:
            return noise_seq[index]
        return self._candidate_noise(index, obs_cond.shape[0], obs_cond.device, obs_cond.dtype)

    def search_candidates(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions,
            noise_seq: Optional[torch.Tensor] = None,
            return_scores: bool = False,
            return_subgoals: bool = False,
            obs_cond: Optional[torch.Tensor] = None,
        ):
        """``n_actions`` candidates, each conditioned on the previous ones.

        Returns ``(actions, values[, scores][, subgoals])``, the same tuple contract as the
        maze / PushT ``SearchProcedureMixin.search_candidates``: ``scores`` is the verifier
        value itself (the context here is scalar) and ``subgoals`` is always None.
        ``noise_seq``: optional (n_actions, B, horizon, action_dim) initial noise.
        """
        if obs_cond is None:
            obs_cond = self.encode_obs_cond(obs_dict)
        actions = None
        values = None
        for _i in range(n_actions):
            new_action = self.predict_action(
                obs_dict,
                actions=actions,
                values=values,
                noise=self._init_candidate_noise(noise_seq, _i, obs_cond),
                obs_cond=obs_cond,
            )['action_pred']
            new_value = verifier.get_value(obs_dict, new_action)
            if actions is None:
                actions = new_action.unsqueeze(1)
                values = new_value.unsqueeze(1)
            else:
                actions = torch.cat([actions, new_action.unsqueeze(1)], dim=1)
                values = torch.cat([values, new_value.unsqueeze(1)], dim=1)
        out = (actions, values)
        if return_scores:
            out = out + (values,)
        if return_subgoals:
            out = out + (None,)
        return out

    @torch.inference_mode()
    def predict_n_actions(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions,
            noise_seq: Optional[torch.Tensor] = None,
        ):
        # n_actions > max_actions slides a (max_actions - 1) context window. The obs is
        # encoded once for the whole decision, and candidate j's noise is keyed on j.
        obs_cond = self.encode_obs_cond(obs_dict)
        if n_actions <= self.max_actions:
            return self.search_candidates(
                obs_dict, verifier, n_actions, noise_seq=noise_seq, obs_cond=obs_cond)

        actions, values = self.search_candidates(
            obs_dict, verifier, self.max_actions,
            noise_seq=(noise_seq[:self.max_actions] if noise_seq is not None else None),
            obs_cond=obs_cond)
        all_actions = actions.clone()
        all_values = values.clone()

        action_history = actions[:, 1:]
        value_history = values[:, 1:]
        for _j in range(self.max_actions, n_actions):
            new_action = self.predict_action(
                obs_dict,
                actions=action_history,
                values=value_history,
                noise=self._init_candidate_noise(noise_seq, _j, obs_cond),
                obs_cond=obs_cond,
            )['action_pred']
            new_value = verifier.get_value(obs_dict, new_action)

            all_actions = torch.cat([all_actions, new_action.unsqueeze(1)], dim=1)
            all_values = torch.cat([all_values, new_value.unsqueeze(1)], dim=1)
            action_history = torch.cat([action_history[:, 1:], new_action.unsqueeze(1)], dim=1)
            value_history = torch.cat([value_history[:, 1:], new_value.unsqueeze(1)], dim=1)

        return all_actions, all_values

    def generate_search_context(self, obs_dict, action=None):
        """The search context ``compute_loss`` conditions on: ``(actions, values)``.

        Exactly the search the training loss runs, split out so a trainer can generate the
        context ONCE for a pool of windows and reuse it across several gradient steps (see
        TrainSearchOuterInnerRoboMonkeyWorkspace) instead of paying for ``max_actions - 1`` candidate
        samples plus their verifier calls on every update.

        ``action`` is the expert action, passed through so ground-truth verifiers (e.g.
        MSEVerifier) can read it; verifiers that ignore it are unaffected.

        Deliberately ``no_grad`` rather than ``inference_mode``: inference tensors carry a
        permanent restriction against taking part in autograd-recorded operations, which
        makes them unsafe to buffer and feed to the model on a LATER step. ``no_grad``
        yields ordinary tensors, so the returned context can be stored for as long as the
        caller likes. The search builds no graph either way.

        ``actions`` come back in RAW action units (the verifier scores them there); the
        normalization the model needs happens at the model boundary, see
        ``_normalize_context_actions``.
        """
        search_obs = dict(obs_dict)
        if action is not None:
            search_obs['action'] = action
        with torch.no_grad():
            return _run_search(self, search_obs, self.max_actions - 1)

    def predict_epsilon(self, batch, actions, values, noisy_trajectory, timesteps):
        """One denoiser forward on EXTERNALLY supplied noise, timesteps and context.

        Used to compare two snapshots of this policy at matched inputs. For a fixed
        ``(noisy_trajectory, timestep, obs, context)`` the two snapshots' reverse
        transition kernels are Gaussians whose variance comes from the scheduler and so is
        identical between them; they differ only in a mean that is affine in the predicted
        noise. Their KL is therefore proportional to ``||eps_a - eps_b||^2``, which makes
        epsilon-space MSE at matched inputs a per-denoising-step KL -- the tractable stand-in
        for the policy KL that PPO would monitor (a diffusion policy has no closed-form
        ``log p(a|s)`` to take a ratio of).

        Obs corruption is deliberately NOT applied: ``corrupt_obs_features`` draws fresh
        noise per call, so including it would book corruption noise as policy drift. The
        obs ARE re-encoded from ``batch`` rather than taken from the caller, because the
        obs encoder is part of what drifts and each snapshot must use its own.
        """
        with torch.no_grad():
            nobs = self.normalizer.normalize(batch['obs'])
            To = self.n_obs_steps
            B = next(iter(nobs.values())).shape[0]
            this_nobs = dict_apply(
                {k: nobs[k] for k in self._state_keys},
                lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]),
            )
            obs_cond = self.obs_encoder(this_nobs).reshape(B, To, -1)
            return self.model(
                noisy_trajectory,
                timesteps,
                obs_cond=obs_cond,
                actions=self._normalize_context_actions(actions),
                values=values,
            )

    def compute_loss(self, batch, actions=None, values=None, return_aux=False):
        """Denoising loss for the expert action, conditioned on a best-of-n search context.

        ``actions``/``values`` optionally supply a PRE-GENERATED search context, in the form
        ``generate_search_context`` returns it. ``None`` (the default) generates it here from
        the current weights -- the offline path, which pays the full search cost on every
        gradient step. A trainer that amortizes the search across several updates passes the
        buffered context back in instead.

        ``return_aux`` additionally returns the exact inputs the denoiser was called with,
        so a frozen snapshot of this policy can be re-run on identical inputs via
        ``predict_epsilon``; matched inputs are what make that comparison a KL.
        """
        target_actions = self.normalizer['action'].normalize(batch['action'])
        B, T, Da = target_actions.shape
        assert T == self.horizon, \
            f"Expected action horizon {self.horizon}, got {T}"
        assert Da == self.action_dim, \
            f"Expected action dim {self.action_dim}, got {Da}"

        obs_cond = self.encode_obs_cond(batch['obs'])

        if actions is None:
            actions, values = self.generate_search_context(
                batch['obs'], action=batch['action'])

        trajectory = target_actions.unsqueeze(1).expand(
            -1,
            self.max_actions,
            -1,
            -1,
        )

        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B, self.max_actions),
            device=trajectory.device,
        ).long()
        flat_trajectory = trajectory.reshape(
            B * self.max_actions,
            self.horizon,
            self.action_dim,
        )
        flat_noise = noise.reshape(
            B * self.max_actions,
            self.horizon,
            self.action_dim,
        )
        noisy_trajectory = self.noise_scheduler.add_noise(
            flat_trajectory,
            flat_noise,
            timesteps.reshape(-1),
        ).reshape(B, self.max_actions, self.horizon, self.action_dim)

        pred = self.model(
            noisy_trajectory,
            timesteps,
            obs_cond=obs_cond,
            actions=self._normalize_context_actions(actions),
            values=values,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss_b = reduce(
            F.mse_loss(pred, target, reduction='none'), 'b ... -> b', 'mean'
        )                                                              # (B,) per-sample BC loss
        loss = loss_b.mean()

        # --- Opt-in advantage policy-gradient term (frozen DQC verifier in the loss) ---
        # Both modes use the frozen verifier's Q as a stop-grad scalar (no backprop
        # through Q); a* = expert action, a_pi = the policy's OWN sample:
        #   'reinforce' (default, original): REINFORCE on a_pi with advantage
        #       Q(a_pi)-Q(a*), realized as advantage-weighted denoising loss on a_pi
        #       re-noised at an independent timestep t2. adv>0 -> lower a_pi's
        #       denoising loss -> raise pi(a_pi).
        #   'awbc': advantage-weighted imitation of the EXPERT,
        #       grad = E[ grad_logpi(a*) * (Q(a*)-Q(a_pi)) ], realized as the
        #       per-sample BC denoising loss on a* weighted by the clamped
        #       expert-over-policy advantage. The weight -> 0 once the policy matches
        #       a*'s Q; clamp>=0 so an over-optimistic critic cannot push pi AWAY
        #       from the expert.
        if self._pg_lambda > 0.0:
            img_key = getattr(self.verifier, 'image_obs_key', 'agentview_image')
            img = batch['obs'][img_key]
            with torch.no_grad():
                na_pi = self.conditional_sample(
                    obs_cond, actions=self._normalize_context_actions(actions),
                    values=values, **self.step_kwargs
                )                                                       # (B, T, Da) normalized
                a_pi_raw = self.normalizer['action'].unnormalize(na_pi)
                q_pi = self.verifier.get_value({img_key: img}, a_pi_raw)            # (B,)
                q_star = self.verifier.get_value({img_key: img}, batch['action'])  # (B,)
            if self._pg_mode == 'awbc':
                w = (q_star - q_pi).clamp(min=0.0).detach().to(loss.device)  # (B,) >= 0
                loss_pg = (w * loss_b).mean()
                loss = loss + self._pg_lambda * loss_pg
            else:
                adv = (q_pi - q_star).detach().to(loss.device)          # (B,) frozen advantage
                na_pi = na_pi.detach()
                noise2 = torch.randn_like(na_pi)
                t2 = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (B,), device=na_pi.device,
                ).long()
                noisy2 = self.noise_scheduler.add_noise(na_pi, noise2, t2)  # (B, T, Da)
                pred2 = self.model(
                    noisy2.unsqueeze(1), t2.unsqueeze(1),
                    obs_cond=obs_cond, actions=self._normalize_context_actions(actions),
                    values=values,
                )                                                           # (B, 1, T, Da)
                tgt2 = (noise2 if pred_type == 'epsilon' else na_pi).unsqueeze(1)
                mse2 = reduce(F.mse_loss(pred2, tgt2, reduction='none'), 'b ... -> b', 'mean')
                loss_pg = (adv * mse2).mean()
                loss = loss + self._pg_lambda * loss_pg
        if not return_aux:
            return loss
        # everything a frozen snapshot needs to reproduce this exact denoiser forward
        return loss, {
            'noisy_trajectory': noisy_trajectory,
            'timesteps': timesteps,
            'actions': actions,
            'values': values,
        }


class BCSearchWrapper(nn.Module):
    """Wraps a plain BC diffusion policy (e.g. DiffusionUnetImagePolicy) to
    support N-sample inference-time search via an external verifier.

    At eval time, ``predict_n_actions`` calls the base policy N times
    independently (each call draws a fresh diffusion sample), scores each
    candidate with the verifier, and returns the stacked results — the same
    interface as ``SearchPolicyRoboMonkeyDiffusion.predict_n_actions``.

    eval_search.py auto-detects BC checkpoints (no predict_n_actions) and
    wraps them automatically:
        wrapper = BCSearchWrapper(base_policy, verifier, max_actions=128)
    """

    def __init__(self, base_policy, verifier, max_actions: int = 128):
        super().__init__()
        self._base = base_policy
        self.verifier = verifier
        self.max_actions = max_actions
        # Proxy attributes accessed by eval_search._search_policy_predict
        self.normalizer = base_policy.normalizer
        self.n_obs_steps = base_policy.n_obs_steps
        self.n_action_steps = base_policy.n_action_steps

    @torch.no_grad()
    def predict_n_actions(
        self,
        obs_dict: Dict[str, Any],
        verifier,
        n_actions: int,
        **_kw,
    ):
        """Run BC policy n_actions times; verifier-score each sample.

        Returns:
            actions: (B, n_actions, horizon, action_dim)
            values:  (B, n_actions)
        """
        # Verifier scoring. The RoboMonkey verifier scores ONE 7-D action vs the
        # current image ("What action should the robot take ..."). To score a
        # whole chunk we evaluate EVERY executed action against the current frame
        # and aggregate into a chunk score, then BoN/softmax picks the best chunk.
        #
        # BC_VERIFIER_SCORE:
        #   "chunk" (default) -> score all n_action_steps executed actions
        #       (result['action'], horizon steps [start:end]) against the current
        #       frame; aggregate per BC_VERIFIER_AGG into one chunk value.
        #   "exec"  -> score a single executed step (index BC_VERIFIER_STEP).
        #   "full"  -> score the full-horizon tail step (action_pred[:, -1]); BC
        #       candidates collapse there, so this ties the verifier (parity only).
        # BC_VERIFIER_AGG (chunk mode): mean (default) | sum | min | last | first.
        import os as _os
        score_mode = _os.environ.get("BC_VERIFIER_SCORE", "chunk").lower()
        step = int(_os.environ.get("BC_VERIFIER_STEP", "0"))
        agg = _os.environ.get("BC_VERIFIER_AGG", "mean").lower()

        # 1) Draw the N candidate chunks.
        full_list, exec_list = [], []
        for _ in range(n_actions):
            result = self._base.predict_action(obs_dict)
            full_list.append(result.get("action_pred", result["action"]))  # (B,H,7)
            exec_list.append(result["action"])                              # (B,S,7)
        actions = torch.stack(full_list, dim=1)   # (B, N, H, 7) -> returned
        B, N = actions.shape[0], actions.shape[1]

        if score_mode == "full":
            vals = [verifier.get_value(obs_dict, full_list[i]) for i in range(N)]
            return actions, torch.stack(vals, dim=1)

        execs = torch.stack(exec_list, dim=1)      # (B, N, S, 7)
        S = execs.shape[2]

        if score_mode == "chunk_critic":
            # REAL DQC chunk critic: score each candidate's whole S-step executed
            # window in ONE verifier forward (the verifier flattens the S steps
            # into a 56-D action and calls the chunk_critic head). No per-step
            # aggregation. Needs S == chunk_horizon (8) and the verifier in
            # score_full_horizon mode (idxs=range(S)).
            assert B == 1, "BCSearchWrapper chunk scoring assumes rollout B=1"
            chunk_h = int(getattr(verifier, "chunk_horizon", S))
            img_key = getattr(verifier, "image_obs_key", None)
            img = obs_dict[img_key]
            batched_img = img.expand(N, *img.shape[1:])
            if _os.environ.get("BC_CHUNK_FROM_PRED", "0") == "1":
                # Score the first chunk_h PREDICTED actions as the chunk, decoupled
                # from rollout (mirrors the native search policy's score_full_horizon).
                # Lets rollout (n_action_steps) < chunk_h -- e.g. the H8 BC: predict 8,
                # score all 8, execute 4. NOTE: with n_obs_steps>1 (pad_before>0) the
                # predicted chunk is shifted ~1 step early vs the verifier's a_{t:t+8},
                # but the shift is identical across candidates so BoN ranking is sound.
                H = int(actions.shape[2])
                assert H >= chunk_h, (
                    f"BC_CHUNK_FROM_PRED needs predicted horizon >= chunk_h={chunk_h}, "
                    f"got H={H}.")
                chunk_acts = actions[0][:, :chunk_h, :]              # (N, chunk_h, 7)
                chunk_vals = verifier.get_value({img_key: batched_img}, chunk_acts)
            else:
                # REAL DQC chunk critic on the S-step executed window (S == chunk_h).
                assert S == chunk_h, (
                    f"chunk_critic needs S={chunk_h} executed steps, got S={S}; set "
                    f"N_ACTION_STEPS={chunk_h} or BC_CHUNK_FROM_PRED=1 (score predicted).")
                chunk_vals = verifier.get_value({img_key: batched_img}, execs[0])  # (N,)
            return actions, chunk_vals.unsqueeze(0)    # (B=1, N)

        if score_mode == "exec":
            sel = execs[:, :, step:step + 1, :]    # (B, N, 1, 7)
            steps_to_score = [step]
        else:  # "chunk": score every executed action
            sel = execs                            # (B, N, S, 7)
            steps_to_score = list(range(S))
        Ssel = sel.shape[2]

        # 2) Score all (candidate, step) pairs against the current frame in one
        #    batched verifier call. Rollout is B=1; replicate the single image to
        #    M = N*Ssel rows (identical image -> verifier image-feature cache hits).
        assert B == 1, "BCSearchWrapper chunk scoring assumes rollout batch B=1"
        img_key = getattr(verifier, "image_obs_key", None)
        img = obs_dict[img_key]                    # (1, To, ...) verifier reads [:, -1]
        M = N * Ssel
        batched_img = img.expand(M, *img.shape[1:])
        flat_actions = sel.reshape(M, 1, 7)        # (M, 1, 7); get_value reads [:, -1]
        flat_vals = verifier.get_value({img_key: batched_img}, flat_actions)  # (M,)
        per_step = flat_vals.reshape(N, Ssel)      # (N, S or 1)

        # 3) Aggregate per-step scores into one value per candidate chunk.
        if agg in ("discounted_sum", "discounted", "disc_sum"):
            # sum_t gamma^t * score_t over the executed window (t in exec order).
            # gamma from the verifier (DQC discount) or BC_VERIFIER_DISCOUNT env.
            gamma = float(getattr(verifier, "discount",
                                  _os.environ.get("BC_VERIFIER_DISCOUNT", 0.98)))
            Sn = per_step.shape[1]
            w = torch.tensor([gamma ** t for t in range(Sn)],
                             dtype=per_step.dtype, device=per_step.device)
            chunk_vals = (per_step * w).sum(dim=1)
        elif agg == "sum":
            chunk_vals = per_step.sum(dim=1)
        elif agg == "min":
            chunk_vals = per_step.min(dim=1).values
        elif agg == "last":
            chunk_vals = per_step[:, -1]
        elif agg == "first":
            chunk_vals = per_step[:, 0]
        else:  # mean
            chunk_vals = per_step.mean(dim=1)
        return actions, chunk_vals.unsqueeze(0)    # (B=1, N)

    def predict_action(self, obs_dict):
        return self._base.predict_action(obs_dict)

    def set_normalizer(self, normalizer):
        self._base.set_normalizer(normalizer)
        self.normalizer = normalizer

    def compute_loss(self, batch):
        return self._base.compute_loss(batch)


class SearchPolicyRoboMonkeyDiffusionNoiseCond(SearchPolicyRoboMonkeyDiffusion):
    """Noise-conditioned variant of SearchPolicyRoboMonkeyDiffusion.

    Mirrors tmrl's :class:`ContextSmoothedFlowPolicy`: at training time a
    per-sample ``tcont_context`` is drawn from ``U[0, 1]`` and used to
       (a) noise the encoded state via the DDPM scheduler at
           ``timestep = round(tcont_context * (num_train_timesteps-1))`` and
       (b) condition the denoiser by adding a sinusoidal embedding of the
           timestep onto every obs token before the diffusion transformer.

    At inference, the caller passes ``tcont_context`` explicitly (default
    ``0.0`` = clean state). A downstream RL head could learn to set it,
    exactly as tmrl's SAC actor outputs a timestep dimension alongside the
    action.
    """

    # Its own search_candidates / encode_obs_cond do not use the per-episode seed streams,
    # so the eval harness falls back to seeding the global RNG for it.
    supports_sample_seeds = False

    def __init__(
            self,
            *args,
            tcont_emb_dim: int = 128,
            **kwargs,
        ):
        # corrupt_obs is mandatory for this class — noise IS the conditioning.
        kwargs['corrupt_obs'] = True
        # obs_noise_level is unused here (tcont is always sampled / supplied),
        # null it out so the parent doesn't apply its own corruption.
        kwargs['obs_noise_level'] = None
        super().__init__(*args, **kwargs)

        max_t = self.obs_noise_scheduler.config.num_train_timesteps
        self.register_buffer(
            "_tcont_max_t", torch.tensor(max_t, dtype=torch.long),
            persistent=False,
        )

        hidden = max(64, tcont_emb_dim // 2)
        self.tcont_emb = nn.Sequential(
            SinusoidalPosEmb(tcont_emb_dim),
            nn.Linear(tcont_emb_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, self.obs_feature_dim),
        )
        print(
            "SearchPolicyRoboMonkeyDiffusionNoiseCond: "
            f"conditioning on obs-noise level (sinusoidal -> {self.obs_feature_dim}-dim "
            f"shift on obs tokens)."
        )

    # --- noise sampling + injection ---

    def _resolve_tcont(self, batch_size: int, device, tcont_context) -> torch.Tensor:
        """Return a (B,) float tensor in [0, 1]."""
        if tcont_context is None:
            if self.training:
                return torch.rand(batch_size, device=device)
            # Eval default: clean obs. Override to study sensitivity to noise.
            return torch.zeros(batch_size, device=device)
        if torch.is_tensor(tcont_context):
            t = tcont_context.to(device=device, dtype=torch.float)
            if t.ndim == 0:
                t = t.expand(batch_size)
            elif t.shape[0] == 1 and batch_size > 1:
                t = t.expand(batch_size)
            return t.clamp(0.0, 1.0)
        return torch.full(
            (batch_size,), float(tcont_context),
            device=device, dtype=torch.float,
        ).clamp(0.0, 1.0)

    def _apply_tunable_noise(
            self, obs_features: torch.Tensor, tcont: torch.Tensor,
        ) -> torch.Tensor:
        """obs_features: (B, To, D); tcont: (B,) float in [0, 1]."""
        max_t = int(self._tcont_max_t.item())
        timesteps = (tcont * max_t).long().clamp(max=max_t - 1)
        noise = torch.randn_like(obs_features)
        return self.obs_noise_scheduler.add_noise(obs_features, noise, timesteps)

    def encode_obs_cond(
            self,
            obs_dict: Dict[str, torch.Tensor],
            tcont_context: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        nobs = self.normalizer.normalize(obs_dict)
        obs_value = next(iter(nobs.values()))
        B = obs_value.shape[0]
        To = self.n_obs_steps
        if isinstance(nobs, dict):
            this_nobs = dict_apply(
                {k: nobs[k] for k in self._state_keys},
                lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]),
            )
        else:
            this_nobs = nobs[:, :To, ...].reshape(-1, *nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs).reshape(B, To, -1)

        tcont = self._resolve_tcont(B, nobs_features.device, tcont_context)
        noisy_features = self._apply_tunable_noise(nobs_features, tcont)

        # Inject noise-level conditioning: sinusoidal embedding broadcast over
        # the To obs tokens.  We embed the discrete timestep (not tcont) so the
        # representation aligns with how the DDPM scheduler indexes noise.
        max_t = int(self._tcont_max_t.item())
        t_discrete = (tcont * max_t).clamp(max=max_t - 1)
        emb = self.tcont_emb(t_discrete)  # (B, obs_feature_dim)
        return noisy_features + emb.unsqueeze(1)  # (B, To, obs_feature_dim)

    # --- threading tcont through the prediction stack ---

    def _normalize_context_actions(self, actions):
        """Put context actions into the space the model works in.

        The search loop carries candidates in RAW action units because the verifier
        scores them there and the env executes them. But the model is trained against a
        NORMALIZED target, so handing it raw actions puts the two halves of the
        action+value context on wildly different scales. Normalizing here -- at the model
        boundary, not in the search loop -- keeps one tensor from serving both boundaries.

        Matches the convention already used for the policy-gradient branch, which keeps a
        normalized `na_pi` and unnormalizes a separate `a_pi_raw` copy for the verifier.
        """
        if actions is None:
            return None
        return self.normalizer['action'].normalize(actions)

    def predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor],
            actions: Optional[torch.Tensor] = None,
            values: Optional[torch.Tensor] = None,
            tcont_context: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        obs_cond = self.encode_obs_cond(obs_dict, tcont_context=tcont_context)
        nsample = self.conditional_sample(
            obs_cond=obs_cond,
            actions=self._normalize_context_actions(actions),
            values=values,
            **self.step_kwargs,
        )
        action_pred = self.normalizer['action'].unnormalize(nsample)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {
            'action': action_pred[:, start:end],
            'action_pred': action_pred,
        }

    def search_candidates(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions,
            tcont_context: Optional[torch.Tensor] = None,
            return_scores: bool = False,
            return_subgoals: bool = False,
        ):
        actions = None
        values = None
        for _ in range(n_actions):
            new_action = self.predict_action(
                obs_dict,
                actions=actions,
                values=values,
                tcont_context=tcont_context,
            )['action_pred']
            new_value = verifier.get_value(obs_dict, new_action)
            if actions is None:
                actions = new_action.unsqueeze(1)
                values = new_value.unsqueeze(1)
            else:
                actions = torch.cat([actions, new_action.unsqueeze(1)], dim=1)
                values = torch.cat([values, new_value.unsqueeze(1)], dim=1)
        out = (actions, values)
        if return_scores:
            out = out + (values,)
        if return_subgoals:
            out = out + (None,)
        return out

    @torch.inference_mode()
    def predict_n_actions(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions,
            tcont_context: Optional[torch.Tensor] = None,
        ):
        if n_actions <= self.max_actions:
            return self.search_candidates(
                obs_dict, verifier, n_actions, tcont_context=tcont_context,
            )

        actions, values = self.search_candidates(
            obs_dict, verifier, self.max_actions, tcont_context=tcont_context,
        )
        all_actions = actions.clone()
        all_values = values.clone()
        action_history = actions[:, 1:]
        value_history = values[:, 1:]
        for _ in range(self.max_actions, n_actions):
            new_action = self.predict_action(
                obs_dict,
                actions=action_history,
                values=value_history,
                tcont_context=tcont_context,
            )['action_pred']
            new_value = verifier.get_value(obs_dict, new_action)
            all_actions = torch.cat([all_actions, new_action.unsqueeze(1)], dim=1)
            all_values = torch.cat([all_values, new_value.unsqueeze(1)], dim=1)
            action_history = torch.cat(
                [action_history[:, 1:], new_action.unsqueeze(1)], dim=1,
            )
            value_history = torch.cat(
                [value_history[:, 1:], new_value.unsqueeze(1)], dim=1,
            )
        return all_actions, all_values

    def generate_search_context(self, obs_dict, action=None):
        """Not supported for the noise-conditioned variant.

        This class draws a fresh ``tcont`` per batch and uses that SAME value both to noise
        the encoded state and to generate the search context, so the candidates and the
        observation they were produced from share one noise level (see ``compute_loss``).
        A buffered context breaks that coupling: the context would carry the ``tcont`` of
        the outer step while the obs it is paired with is re-encoded at a freshly drawn one.

        Raising is deliberate. Inheriting the parent's implementation would silently drop
        the ``tcont`` argument and train on mismatched pairs with no error. Buffering
        ``tcont`` alongside the context would fix it, but that is a change to what this
        variant measures and should be a deliberate decision, not a side effect.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support a buffered search context: its "
            "tcont noise level couples the context to the observation encoding, and "
            "buffering one without the other trains on mismatched pairs. Train this "
            "variant with TrainMLPImageWorkspace (context regenerated per step).")

    def compute_loss(self, batch, actions=None, values=None, return_aux=False):
        if actions is not None:
            raise NotImplementedError(
                f"{type(self).__name__} cannot consume a pre-generated search context; "
                "see generate_search_context.")
        target_actions = self.normalizer['action'].normalize(batch['action'])
        B, T, Da = target_actions.shape
        assert T == self.horizon
        assert Da == self.action_dim

        # Single tcont per sample, reused for the search-context rollout and
        # the diffusion loss so the candidate sequence is internally consistent.
        device = target_actions.device
        tcont = torch.rand(B, device=device)

        obs_cond = self.encode_obs_cond(batch['obs'], tcont_context=tcont)

        with torch.inference_mode():
            actions, values = _run_search(
                self, batch['obs'], self.max_actions - 1,
                tcont_context=tcont,
            )

        trajectory = target_actions.unsqueeze(1).expand(
            -1, self.max_actions, -1, -1,
        )
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B, self.max_actions),
            device=trajectory.device,
        ).long()
        flat_trajectory = trajectory.reshape(
            B * self.max_actions, self.horizon, self.action_dim,
        )
        flat_noise = noise.reshape(
            B * self.max_actions, self.horizon, self.action_dim,
        )
        noisy_trajectory = self.noise_scheduler.add_noise(
            flat_trajectory, flat_noise, timesteps.reshape(-1),
        ).reshape(B, self.max_actions, self.horizon, self.action_dim)

        pred = self.model(
            noisy_trajectory,
            timesteps,
            obs_cond=obs_cond,
            actions=self._normalize_context_actions(actions),
            values=values,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        return loss.mean()
