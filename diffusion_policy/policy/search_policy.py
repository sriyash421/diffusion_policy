from typing import Dict, Any
import torch
import torch.nn as nn

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from torch.distributions import Normal

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from transformers import GPT2Config, GPT2Model

from l2s.verifier import MazeVerifier

class SearchPolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict[str, Any],
            obs_encoder: MultiImageObsEncoder,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
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
        obs_feature_dim = obs_encoder.output_shape()[0]
        self.obs_encoder = obs_encoder
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.normalizer = LinearNormalizer()
        self.verifier = MazeVerifier(
            maze_path=kwargs.get('maze_path', None),
            device=kwargs.get('device', 'cpu'),
            noise=kwargs.get('verifier_noise', 0.0),
        )
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

        # Encode obs: flatten all obs steps
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
