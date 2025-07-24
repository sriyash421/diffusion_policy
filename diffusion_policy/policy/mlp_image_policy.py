from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.bet.utils import mlp

class MLPImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict[str, Any],
            obs_encoder: MultiImageObsEncoder,
            n_action_steps: int,
            n_obs_steps: int,
            hidden_dim: int = 512,
            hidden_depth: int = 2,
            **kwargs):
        assert n_action_steps == 1, "MLPImagePolicy only supports n_action_steps=1"
        
        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]
        self.obs_encoder = obs_encoder
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.normalizer = LinearNormalizer()
        self.kwargs = kwargs
        # Input: all obs steps concatenated
        input_dim = obs_feature_dim * n_obs_steps
        output_dim = action_dim * 2  # mean and log_std
        self.mlp = mlp(input_dim, hidden_dim, output_dim, hidden_depth)
        self.log_std_min = -20
        self.log_std_max = 2

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        To = self.n_obs_steps
        device = self.device
        dtype = self.dtype
        # Encode obs: flatten all obs steps
        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        else:
            this_nobs = nobs[:,:To,...].reshape(-1,*nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, To, -1)
        mlp_input = nobs_features.reshape(B, -1)
        # Predict mean and log_std
        mlp_out = self.mlp(mlp_input)
        mean, log_std = torch.chunk(mlp_out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        action_pred = self.normalizer['action'].unnormalize(mean)
        action = action_pred  # deterministic: mean
        return {
            'action': action,
            'action_pred': action_pred,
            'mean': mean,
            'log_std': log_std
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]
        To = self.n_obs_steps
        Ta = self.n_action_steps
        Da = self.action_dim
        # Encode obs
        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        else:
            this_nobs = nobs[:,:To,...].reshape(-1,*nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, To, -1)
        mlp_input = nobs_features.reshape(B, -1)
        # Target: only predict the next n_action_steps
        target = nactions[:, To-1:To+Ta-1]
        # Predict mean and log_std
        mlp_out = self.mlp(mlp_input)
        mean, log_std = torch.chunk(mlp_out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(target)
        loss = -log_prob.sum(dim=-1).mean()
        return loss 