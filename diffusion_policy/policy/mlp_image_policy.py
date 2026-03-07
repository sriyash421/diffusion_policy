import copy
from typing import Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from torch.distributions import Normal


class EnsembleModel(nn.Module):
    def __init__(self, policy_fn, n_ensembles=2):
        super().__init__()
        self.n_ensembles = n_ensembles
        self.policies = nn.ModuleList([policy_fn() for _ in range(n_ensembles)])

    def forward(self, obs_features: torch.Tensor) -> Normal:
        # get the output from each policy and stack them to return an ensemble distribution
        outputs = [policy(obs_features) for policy in self.policies]
        return torch.stack(outputs, dim=0)  # shape: (n_ensembles, batch_size, action_dim)
        
def create_ensemble_fn(policy_fn, n_ensembles=1):
    # This is a helper function to create an ensemble of policies. It returns a new policy class that contains n_ensembles copies of the given policy_fn, and averages their outputs.
    if n_ensembles == 1:
        return policy_fn()
    else:
        return EnsembleModel(policy_fn, n_ensembles)

class MLPImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict[str, Any],
            obs_encoder: MultiImageObsEncoder,
            n_action_steps: int,
            n_obs_steps: int,
            hidden_dim: int = 512,
            hidden_depth: int = 4,
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
        
        # Shared trunk
        layers = []
        last_dim = input_dim
        for _ in range(hidden_depth):
            layers += [nn.Linear(last_dim, hidden_dim), nn.ReLU()]
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        
        # Separate heads for mean and log std
        self.mean_head = nn.Linear(last_dim, action_dim)
        self.log_std_head = nn.Linear(last_dim, action_dim)
        
        self.log_std_limits = (-5.0, 2.0)

    def forward(self, obs_features: torch.Tensor) -> Normal:
        """
        Returns a Normal(mean, std) distribution over actions given observation features.
        """
        h = self.trunk(obs_features)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(min=self.log_std_limits[0], max=self.log_std_limits[1])
        std = torch.exp(log_std)
        return Normal(mean, std)

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
        
        # Get action distribution
        dist = self.forward(mlp_input)
        action_pred = dist.rsample()  # Sample from distribution
        action = self.normalizer['action'].unnormalize(action_pred)
        return {
            'action': action,
            'action_pred': action_pred
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
        assert Ta == 1, "MLPImagePolicy only supports n_action_steps=1"
        target = nactions[:, To-1:To+Ta-1].squeeze()
        
        # Get action distribution and compute loss
        dist = self.forward(mlp_input)
        log_prob = dist.log_prob(target).sum(dim=-1)
        loss = -log_prob.mean()
        
        return loss 

class IQLMLPImagePolicy(MLPImagePolicy):
    def __init__(self,
            shape_meta: dict[str, Any],
            obs_encoder: MultiImageObsEncoder,
            n_action_steps: int,
            n_obs_steps: int,
            hidden_dim: int = 512,
            hidden_depth: int = 4,
            **kwargs):
        super().__init__(shape_meta, obs_encoder, n_action_steps, n_obs_steps, hidden_dim, hidden_depth, **kwargs)
        
        # define separate encoder and trunk for value function
        self.value_obs_encoder = copy.deepcopy(obs_encoder)
        value_layers = []
        last_dim = self.obs_feature_dim * n_obs_steps
        for _ in range(hidden_depth):
            value_layers += [nn.Linear(last_dim, hidden_dim), nn.ReLU()]
            last_dim = hidden_dim
        value_layers += [nn.Linear(last_dim, 1)]
        self.value_network = nn.Sequential(*value_layers)

        # define separate encoder and trunk for q function
        self.q_obs_encoder = copy.deepcopy(obs_encoder)
        def q_network_fn():
            q_layers = []
            last_dim = self.obs_feature_dim * n_obs_steps + self.action_dim
            for _ in range(hidden_depth):
                q_layers += [nn.Linear(last_dim, hidden_dim), nn.ReLU()]
                last_dim = hidden_dim
            q_layers += [nn.Linear(last_dim, 1)]
            return nn.Sequential(*q_layers)

        self.q_network = create_ensemble_fn(q_network_fn, n_ensembles=kwargs.get('n_q_ensembles', 2))
        self.target_q_network = create_ensemble_fn(q_network_fn, n_ensembles=kwargs.get('n_q_ensembles', 2))
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.requires_grad_(False)

        self.expectile = kwargs.get('expectile', None)
        self.discount = kwargs.get('discount', None)
        self.temperature = kwargs.get('temperature', 1.0)
        self.tau = kwargs.get('tau', 0.005)
        assert self.expectile is not None, "expectile must be provided for IQLMLPImagePolicy"
        assert self.discount is not None, "discount must be provided for IQLMLPImagePolicy"
    
    def l2_expectile_loss(self, diff, expectile):
        """Asymmetric L2 loss used in IQL for V-function training."""
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff ** 2)
    
    def get_q_values(self, obs_features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q_input = torch.cat([obs_features, actions], dim=-1)
        q_values = self.q_network(q_input).squeeze(-1)  # (N_ensembles, B)
        return q_values
    
    def get_v_values(self, obs_features: torch.Tensor) -> torch.Tensor:
        v_values = self.value_network(obs_features).squeeze(-1)  # (B,)
        return v_values

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        next_nobs = self.normalizer.normalize(batch['next_obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if len(nactions.shape) == 3 and nactions.shape[1] == 1:
            nactions = nactions[:, 0]
        assert len(nactions.shape) == 2, "actions should have shape (B, action_dim)"
        B = nactions.shape[0]

        # Encode observations for policy, value, and q networks.        
        pi_nobs_features = self.obs_encoder(nobs).reshape(B, -1)
        v_nobs_features = self.value_obs_encoder(nobs).reshape(B, -1)
        q_nobs_features = self.q_obs_encoder(nobs).reshape(B, -1)

        v_next_nobs_features = self.value_obs_encoder(next_nobs).reshape(B, -1)


        actions = nactions.reshape(B, -1)
        rewards = batch['reward'].reshape(B, -1)
        dones = batch['done'].reshape(B, -1)

        # ===== 1. Q-Network Loss (TD Error) =====
        # Q(s, a) for current state-action pairs
        q_input = torch.cat([q_nobs_features, actions], dim=-1)
        q_values = self.q_network(q_input).squeeze(-1)  # (N_ensembles, B)

        with torch.no_grad():
            # V(s') for next states
            next_v_values = self.value_network(v_next_nobs_features).squeeze(-1)  # (B,)
            # TD target: r + gamma * V(s') * (1 - done)
            td_target = rewards.squeeze(-1) + self.discount * next_v_values * (1 - dones.squeeze(-1)) # (B,)
        
        # Q loss: MSE between Q-values and TD target
        q_loss = F.mse_loss(q_values, td_target[None, :].expand_as(q_values))

        # ===== 2. Value Network Loss (Expectile Loss) =====
        # V(s) for current states
        v_values = self.value_network(v_nobs_features).squeeze(-1)  # (B,)
        with torch.no_grad():
            q_target = self.target_q_network(torch.cat([q_nobs_features, actions], dim=-1)).squeeze(-1)  # (N_ensembles, B)
            q_target = q_target.min(dim=0).values  # Take min over ensemble for conservative estimate (B,)
        v_loss = self.l2_expectile_loss(q_target - v_values, self.expectile).mean()

        # ===== 3. Policy Network Loss (Advantage-Weighted Regression) =====
        dist = self.forward(pi_nobs_features)
        log_prob = dist.log_prob(actions).sum(dim=-1)  # (B,)
        advantages = (q_target - v_values).detach()  # (B,)
        exp_advantages = torch.exp(advantages * self.temperature).clamp(max=100.0)  # (B,)
        pi_loss = -(exp_advantages * log_prob).mean()

        loss = q_loss + v_loss + pi_loss
        return loss

    def update_target_network(self):
        if isinstance(self.q_network, EnsembleModel):
            for target_q, q in zip(self.target_q_network.policies, self.q_network.policies):
                for target_param, param in zip(target_q.parameters(), q.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
