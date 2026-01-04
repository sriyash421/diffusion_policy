from typing import Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from torch.distributions import Normal

from transformers import GPT2Config, GPT2Model

class TransformerImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict[str, Any],
            obs_encoder: MultiImageObsEncoder,
            n_action_steps: int,
            n_obs_steps: int,
            hidden_dim: int = 512,
            hidden_depth: int = 4,
            n_head: int = 8,
            dropout: float = 0.1,
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
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
        )
        # Shared trunk
        cfg = GPT2Config(
            n_positions=1024,
            n_embd=hidden_dim,
            n_layer=hidden_depth,
            n_head=n_head,
            resid_pdrop=float(dropout),
            embd_pdrop=float(dropout),
            attn_pdrop=float(dropout),
        )
        self.transformer = GPT2Model(cfg)
        
        # Separate heads for mean and log std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_limits = (-5.0, 2.0)

    def forward(self, obs_features: torch.Tensor, attention_mask: torch.Tensor = None) -> Normal:
        """
        Returns a Normal(mean, std) distribution over actions given observation features.
        """
        obs_features = self.input_proj(obs_features)
        h = self.transformer(inputs_embeds=obs_features, attention_mask=attention_mask).last_hidden_state
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(min=self.log_std_limits[0], max=self.log_std_limits[1])
        std = torch.exp(log_std)
        return Normal(mean, std) # B x T x Da

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        assert 'attention_mask' in obs_dict, "attention_mask is required for TransformerImagePolicy"
        attention_mask = obs_dict.pop('attention_mask')
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, T = value.shape[:2]
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype
        # Encode obs: flatten all obs steps
        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1,*x.shape[2:]))
        else:
            this_nobs = nobs.reshape(-1,*nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs).reshape(B, T, -1)
            
        # Get action distribution
        dist = self.forward(nobs_features, attention_mask=attention_mask)
        action_pred = dist.rsample()  # Sample from distribution
        action = self.normalizer['action'].unnormalize(action_pred)
        seq_lens = attention_mask.sum(dim=1)
        action_pred = action_pred[torch.arange(B), seq_lens - 1, :]  # Get last valid action prediction
        action = action[torch.arange(B), seq_lens - 1, :]  # Get last valid action
        return {
            'action': action,
            'action_pred': action_pred
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        expert_mask = batch['expert_mask']
        B = nactions.shape[0]
        T = nactions.shape[1]
        Da = self.action_dim

        # Encode obs
        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1,*x.shape[2:]))
        else:
            this_nobs = nobs.reshape(-1,*nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, T, -1)
        attention_mask = batch['attention_mask']
        target = nactions
        
        # Get action distribution and compute loss
        dist = self.forward(nobs_features, attention_mask=attention_mask) # B x T x Da
        loss_mask = expert_mask[...,0] * attention_mask  # B x T
        log_prob = dist.log_prob(target).sum(dim=-1) # B x T
        loss = -(log_prob * loss_mask).sum() / loss_mask.sum()
        
        return loss 