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
            mask_obs: bool = False,
            concat_obs: bool = False,
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
        self.obs_noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.001,
            beta_end=0.02,
            prediction_type="epsilon",
        )
        if corrupt_obs:
            print("Corrupting obs with a separate noise scheduler")

    def corrupt_obs_features(self, obs_features):
        if not self.corrupt_obs:
            return obs_features

        # Add noise to encoded obs features to simulate corrupted context.
        obs_noise = torch.randn_like(obs_features)
        bsz = obs_features.shape[0]
        timesteps = torch.randint(
            0, self.obs_noise_scheduler.config.num_train_timesteps,
            (bsz,), device=obs_features.device
        ).long()
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

    def _predict_action(
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

    def predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions
    ):
        actions = None
        values = None
        for _ in range(n_actions):
            new_action = self._predict_action(
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
        return actions, values # B, n_actions, horizon, action_dim and B, n_actions

    @torch.inference_mode()
    def predict_n_actions(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions
    ):
        # return self.predict_action(obs_dict, verifier, n_actions)
        if n_actions <= self.max_actions:
            actions, values = self.predict_action(obs_dict, verifier, n_actions)
            actions = actions
            values = values
            return actions, values
        else:
            # If n_actions exceeds max_actions, we can call predict_action multiple times.
            actions, values = self.predict_action(obs_dict, verifier, self.max_actions)

            all_actions = actions.clone()
            all_values = values.clone()

            action_history = actions[:, 1:]
            value_history = values[:, 1:]
            for i in range(self.max_actions, n_actions):
                new_action = self._predict_action(
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

    def compute_loss(self, batch):
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

        with torch.inference_mode():
            actions, values = self.predict_action(
                batch['obs'],
                verifier=self.verifier,
                n_actions=self.max_actions-1
            ) # B, max_actions, horizon*action_dim and B, max_actions

        actions = actions.reshape(B, self.max_actions-1, -1) # B, max_actions, horizon*action_dim
        action_value_input = torch.cat([actions, values.unsqueeze(-1)], dim=-1) # B, max_actions, horizon*action_dim + 1
        action_value_features = self.act_projection(action_value_input) # B, max_actions, hidden_dim
        dist = self.forward(obs_features, action_value_features)  # B, max_actions, horizon, action_dim
        target_actions = target_actions.unsqueeze(1).expand(-1, self.max_actions, -1, -1) # B, max_actions, horizon, action_dim
        log_prob = dist.log_prob(target_actions).sum(dim=(-1, -2)) # B, max_actions
        loss = -log_prob.mean()
        return loss


class SearchPolicyRoboMonkeyDiffusion(BaseImagePolicy):
    """RoboMonkey adaptation of :class:`DiffusionTransformerSearchPolicy`.

    Same encoder-decoder diffusion-transformer architecture as the maze
    version, with two differences:
      * The verifier is injected (Hydra-instantiated), not a hard-coded
        ``MazeVerifier``. Use any client from ``diffusion_policy.policy.verifiers``.
      * ``_state_keys`` filters the obs dict so RGB / depth keys are reserved
        for the verifier and never enter the policy's obs encoder.
    """

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
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.max_actions = max_actions
        self.kwargs = kwargs
        self.step_kwargs = kwargs.get('scheduler_step_kwargs', dict())

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
        )

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.corrupt_obs = corrupt_obs
        self.obs_noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.001,
            beta_end=0.02,
            prediction_type="epsilon",
        )
        if corrupt_obs:
            print("Corrupting obs with a separate noise scheduler")

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
        if not self.corrupt_obs:
            return obs_features

        obs_noise = torch.randn_like(obs_features)
        bsz = obs_features.shape[0]
        timesteps = torch.randint(
            0,
            self.obs_noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=obs_features.device,
        ).long()
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
            **kwargs,
        ) -> torch.Tensor:
        scheduler = self.noise_scheduler
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

    def _predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor],
            actions: Optional[torch.Tensor] = None,
            values: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        obs_cond = self.encode_obs_cond(obs_dict)
        nsample = self.conditional_sample(
            obs_cond=obs_cond,
            actions=actions,
            values=values,
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

    def predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions,
        ):
        actions = None
        values = None
        for _ in range(n_actions):
            new_action = self._predict_action(
                obs_dict,
                actions=actions,
                values=values,
            )['action_pred']
            new_value = verifier.get_value(obs_dict, new_action)
            if actions is None:
                actions = new_action.unsqueeze(1)
                values = new_value.unsqueeze(1)
            else:
                actions = torch.cat([actions, new_action.unsqueeze(1)], dim=1)
                values = torch.cat([values, new_value.unsqueeze(1)], dim=1)
        return actions, values

    @torch.inference_mode()
    def predict_n_actions(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions,
        ):
        if n_actions <= self.max_actions:
            return self.predict_action(obs_dict, verifier, n_actions)

        actions, values = self.predict_action(obs_dict, verifier, self.max_actions)
        all_actions = actions.clone()
        all_values = values.clone()

        action_history = actions[:, 1:]
        value_history = values[:, 1:]
        for _ in range(self.max_actions, n_actions):
            new_action = self._predict_action(
                obs_dict,
                actions=action_history,
                values=value_history,
            )['action_pred']
            new_value = verifier.get_value(obs_dict, new_action)

            all_actions = torch.cat([all_actions, new_action.unsqueeze(1)], dim=1)
            all_values = torch.cat([all_values, new_value.unsqueeze(1)], dim=1)
            action_history = torch.cat([action_history[:, 1:], new_action.unsqueeze(1)], dim=1)
            value_history = torch.cat([value_history[:, 1:], new_value.unsqueeze(1)], dim=1)

        return all_actions, all_values

    def compute_loss(self, batch):
        target_actions = self.normalizer['action'].normalize(batch['action'])
        B, T, Da = target_actions.shape
        assert T == self.horizon, \
            f"Expected action horizon {self.horizon}, got {T}"
        assert Da == self.action_dim, \
            f"Expected action dim {self.action_dim}, got {Da}"

        obs_cond = self.encode_obs_cond(batch['obs'])

        with torch.inference_mode():
            actions, values = self.predict_action(
                batch['obs'],
                verifier=self.verifier,
                n_actions=self.max_actions - 1,
            )

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
            actions=actions,
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
