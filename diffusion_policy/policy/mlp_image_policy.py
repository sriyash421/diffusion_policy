from typing import Dict, Any
import torch
import torch.nn as nn

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from torch.distributions import Normal

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class MLPImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict[str, Any],
            obs_encoder: MultiImageObsEncoder,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
            hidden_dim: int = 512,
            hidden_depth: int = 4,
            corrupt_obs: bool = False,
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
        self.kwargs = kwargs
        
        # Input: all obs steps concatenated
        input_dim = obs_feature_dim * n_obs_steps
        
        # Shared trunk
        layers = []
        last_dim = input_dim
        for _ in range(hidden_depth):
            layers += [nn.Linear(last_dim, hidden_dim), nn.ReLU()]
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        
        output_dim = horizon * action_dim
        self.mean_head = nn.Linear(last_dim, output_dim)
        self.log_std_head = nn.Linear(last_dim, output_dim)
        
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
    
    def forward(self, obs_features: torch.Tensor) -> Normal:
        """
        Returns a factorized Normal over an action trajectory of shape
        (batch, horizon, action_dim).
        """
        if self.corrupt_obs:
            obs_features = self.corrupt_obs_features(obs_features)
            
        B = obs_features.shape[0]
        h = self.trunk(obs_features)
        mean = self.mean_head(h).reshape(B, self.horizon, self.action_dim)
        log_std = self.log_std_head(h).reshape(B, self.horizon, self.action_dim)
        log_std = log_std.clamp(
            min=self.log_std_limits[0],
            max=self.log_std_limits[1]
        )
        std = torch.exp(log_std)
        return Normal(mean, std)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
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
        mlp_input = nobs_features.reshape(B, -1)
        
        dist = self.forward(mlp_input)
        naction_pred = dist.rsample()
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {
            'action': action,
            'action_pred': action_pred
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B, T, Da = nactions.shape
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
        mlp_input = nobs_features.reshape(B, -1)
        
        dist = self.forward(mlp_input)
        log_prob = dist.log_prob(nactions).sum(dim=(-1, -2))
        loss = -log_prob.mean()
        
        return loss 
