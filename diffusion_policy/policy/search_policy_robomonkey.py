from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.diffusion_transformer_search_policy import (
    DiffusionTransformerSearchPolicy,
)
from diffusion_policy.common.pytorch_util import dict_apply
from torch.distributions import Normal

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce
from transformers import GPT2Config, GPT2Model

from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb



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
        if self.obs_noise_level is not None and float(self.obs_noise_level) <= 0:
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


class SearchPolicyRoboMonkeyDiffusion(DiffusionTransformerSearchPolicy):
    """SIMPLER arm of the diffusion search transformer.

    Everything about the network, search loop, selection, seeding and loss is the maze
    class's. SIMPLER adds: an injected (Hydra-built) verifier, low_dim-only policy obs
    (rgb keys are reserved for the verifier), a fixed ``obs_noise_level`` corruption mode,
    the expert action forwarded to ground-truth verifiers, KV-prefix verifier chunking,
    and an optional advantage policy-gradient term (off by default; ablation only).
    """

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
            obs_noise_level: Optional[float] = None,
            pg_lambda: float = 0.0,
            pg_mode: str = 'reinforce',
            **kwargs,
        ):
        self._injected_verifier = verifier
        self._state_keys = [
            k for k, v in shape_meta['obs'].items()
            if v.get('type', 'low_dim') == 'low_dim'
        ]
        # Fixed corruption level in [0, 1] -> one DDPM timestep at train AND eval;
        # None -> the base's random-timestep, train-only corruption.
        self.obs_noise_level = obs_noise_level
        # pg_lambda > 0 adds a frozen-verifier advantage term; 'reinforce' re-noises the
        # policy's own sample, 'awbc' weights the expert loss. Unused by any config.
        self._pg_lambda = float(pg_lambda)
        self._pg_mode = str(pg_mode)
        # Pre-2026-09-30 checkpoints have no corrupt_obs_eval; their semantics were: a
        # fixed level is applied at eval, a random level is not.
        if kwargs.get('corrupt_obs') and kwargs.get('corrupt_obs_eval') is None:
            kwargs['corrupt_obs_eval'] = obs_noise_level is not None
        super().__init__(
            shape_meta=shape_meta, obs_encoder=obs_encoder,
            noise_scheduler=noise_scheduler, horizon=horizon,
            n_action_steps=n_action_steps, n_obs_steps=n_obs_steps, **kwargs)
        _maybe_set_executed_chunk_window(verifier, n_obs_steps, n_action_steps)

    def _build_verifier(self, **kwargs):
        return self._injected_verifier

    def _encode_obs_features(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return super()._encode_obs_features({k: obs_dict[k] for k in self._state_keys})

    def corrupt_obs_features(self, obs_features):
        if self.obs_noise_level is None:
            if self.corrupt_obs and (self.training or self.corrupt_obs_eval):
                obs_features = self._corrupt_at(obs_features, None)
            return obs_features
        if float(self.obs_noise_level) <= 0:
            return obs_features
        if not self.corrupt_obs or (not self.training and not self.corrupt_obs_eval):
            return obs_features
        max_t = self.obs_noise_scheduler.config.num_train_timesteps
        t = max(0, min(max_t - 1, int(round(float(self.obs_noise_level) * (max_t - 1)))))
        return self._corrupt_at(obs_features, t)

    def _corrupt_at(self, obs_features, t):
        """Noise drawn from the per-episode stream (`_init_noise`), like the candidates."""
        bsz = obs_features.shape[0]
        noise = self._init_noise(obs_features.shape, obs_features.dtype, obs_features.device)
        max_t = self.obs_noise_scheduler.config.num_train_timesteps
        if t is None:
            timesteps = torch.randint(0, max_t, (bsz,), device=obs_features.device)
        else:
            timesteps = torch.full((bsz,), t, device=obs_features.device, dtype=torch.long)
        return self.obs_noise_scheduler.add_noise(obs_features, noise, timesteps)

    def generate_search_context(self, obs_dict, obs_features=None, action=None):
        """``action`` (the expert chunk) is forwarded so ground-truth verifiers can read it."""
        search_obs = dict(obs_dict)
        if action is not None:
            search_obs['action'] = action
        with torch.no_grad():
            return _run_search(self, search_obs, self.max_actions - 1,
                               obs_features=obs_features)

    def compute_loss(self, batch, actions=None, values=None, return_aux=False,
                     slot_weighting=True, obs_features=None):
        if actions is None:
            actions, values = self.generate_search_context(
                batch['obs'], obs_features=obs_features, action=batch['action'])
        loss, aux = super().compute_loss(
            batch, actions, values, return_aux=True,
            slot_weighting=slot_weighting, obs_features=obs_features)
        if self._pg_lambda > 0.0:
            loss = loss + self._pg_lambda * self._pg_loss(batch, actions, values)
        return (loss, aux) if return_aux else loss

    def _pg_loss(self, batch, actions, values):
        """Frozen-verifier advantage term; Q is a stop-grad scalar in both modes."""
        img_key = getattr(self.verifier, 'image_obs_key', 'agentview_image')
        img = batch['obs'][img_key]
        obs_cond = self.corrupt_obs_features(self._encode_obs_features(batch['obs']))
        nctx = self._normalize_context_actions(actions)
        with torch.no_grad():
            na_pi = self.conditional_sample(obs_cond, actions=nctx, values=values,
                                            **self.step_kwargs)
            q_pi = self.verifier.get_value({img_key: img}, self.normalizer['action'].unnormalize(na_pi))
            q_star = self.verifier.get_value({img_key: img}, batch['action'])
        B = na_pi.shape[0]
        if self._pg_mode == 'awbc':
            target = self.normalizer['action'].normalize(batch['action'])
            noise = torch.randn_like(target)
            t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,),
                              device=target.device)
            pred = self.model(self.noise_scheduler.add_noise(target, noise, t).unsqueeze(1),
                              t.unsqueeze(1), obs_cond=obs_cond, actions=nctx, values=values)
            per_sample = reduce(F.mse_loss(pred[:, 0], noise, reduction='none'), 'b ... -> b', 'mean')
            w = (q_star - q_pi).clamp(min=0.0).to(per_sample)
            return (w * per_sample).mean()
        noise = torch.randn_like(na_pi)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,),
                          device=na_pi.device)
        pred = self.model(self.noise_scheduler.add_noise(na_pi, noise, t).unsqueeze(1),
                          t.unsqueeze(1), obs_cond=obs_cond, actions=nctx, values=values)
        per_sample = reduce(F.mse_loss(pred[:, 0], noise, reduction='none'), 'b ... -> b', 'mean')
        return ((q_pi - q_star).to(per_sample) * per_sample).mean()

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
        # Eval-only state noise (None = off): the fixed-level DDPM forward noise of
        # SearchPolicyRoboMonkeyDiffusion.corrupt_obs_features, on the normalized low-dim
        # obs the BC conditions on. One draw per replan, shared by all N candidates (a
        # noisy state sensor, not a source of candidate diversity); the verifier keeps
        # reading the clean obs.
        self.eval_obs_noise_level: Optional[float] = None
        self.obs_noise_scheduler = DDPMScheduler(
            num_train_timesteps=100, beta_start=0.001, beta_end=0.02,
            prediction_type="epsilon")

    OBS_NOISE_SEED_KEY = 2   # SeedSequence([replan_seed, 2]); 1 is eval_search's selection stream

    def noisy_policy_obs(self, obs_dict: Dict[str, Any],
                         obs_noise_seed: Optional[int] = None) -> Dict[str, Any]:
        """``obs_dict`` with the base policy's low-dim keys noised at
        ``eval_obs_noise_level``; every other key (images, verifier inputs) untouched.
        The noise comes from its own generator keyed on the replan seed, so the candidate
        draws stay on the same RNG stream as a clean run of the same seed."""
        level = self.eval_obs_noise_level
        if level is None or float(level) <= 0.0:
            return obs_dict
        import numpy as _np
        max_t = self.obs_noise_scheduler.config.num_train_timesteps
        t = max(0, min(max_t - 1, int(round(float(level) * (max_t - 1)))))
        abar = float(self.obs_noise_scheduler.alphas_cumprod[t])
        g = torch.Generator()
        if obs_noise_seed is not None:
            g.manual_seed(int(_np.random.SeedSequence(
                [int(obs_noise_seed), self.OBS_NOISE_SEED_KEY]).generate_state(1)[0]))
        else:
            g.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
        out = dict(obs_dict)
        for k in getattr(self._base, "obs_keys", ()):
            v = obs_dict.get(k)
            if not torch.is_tensor(v) or v.ndim > 3 or not v.is_floating_point():
                continue                     # images / non-state keys stay clean
            nv = self._base.normalizer[k].normalize(v)
            eps = torch.randn(nv.shape, generator=g, dtype=nv.dtype).to(nv.device)
            nv = abar ** 0.5 * nv + (1.0 - abar) ** 0.5 * eps
            out[k] = self._base.normalizer[k].unnormalize(nv)
        return out

    @torch.no_grad()
    def predict_n_actions(
        self,
        obs_dict: Dict[str, Any],
        verifier,
        n_actions: int,
        obs_noise_seed: Optional[int] = None,
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

        # 1) Draw the N candidate chunks. BC_BATCH_CANDIDATES=1 draws all N in ONE
        #    batched predict_action over N copies of the observation (same model, same
        #    sampler, independent noise per row) instead of N sequential calls; the
        #    candidates are a different, equally valid draw from the same distribution,
        #    still deterministic under the per-replan seed. Default off so earlier
        #    sweeps reproduce exactly.
        full_list, exec_list = [], []
        policy_obs = self.noisy_policy_obs(obs_dict, obs_noise_seed)   # verifier: obs_dict
        if _os.environ.get("BC_BATCH_CANDIDATES", "0") == "1":
            B0 = next(v for v in policy_obs.values() if torch.is_tensor(v)).shape[0]
            assert B0 == 1, "BC_BATCH_CANDIDATES assumes rollout batch B=1"
            # Optional fixed-size pool makes N sweeps exact prefixes at the same
            # state/seed, including batched CUDA RNG and numerical batch geometry.
            pool_size = int(_os.environ.get('BC_CANDIDATE_POOL', n_actions))
            if pool_size < n_actions:
                raise ValueError('BC_CANDIDATE_POOL must be >= n_actions')
            rep = {k: (v.expand(pool_size, *v.shape[1:]) if torch.is_tensor(v) else v)
                   for k, v in policy_obs.items()}
            result = self._base.predict_action(rep)
            full = result.get("action_pred", result["action"])               # (N,H,7)
            for i in range(n_actions):
                full_list.append(full[i:i + 1])
                exec_list.append(result["action"][i:i + 1])
        else:
            for _ in range(n_actions):
                result = self._base.predict_action(policy_obs)
                full_list.append(result.get("action_pred", result["action"]))  # (B,H,7)
                exec_list.append(result["action"])                              # (B,S,7)
        actions = torch.stack(full_list, dim=1)   # (B, N, H, 7) -> returned
        B, N = actions.shape[0], actions.shape[1]

        if score_mode == "full":
            if B == 1 and _os.environ.get("BC_BATCH_CANDIDATES", "0") == "1":
                # one verifier call over all N candidates (obs repeated per candidate)
                rep = {k: (v.expand(N, *v.shape[1:]) if torch.is_tensor(v) else v)
                       for k, v in obs_dict.items()}
                return actions, verifier.get_value(rep, actions[0]).unsqueeze(0)
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
