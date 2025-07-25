import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from typing import Dict
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator


def RK2(
    model=None,
    num_inference_steps=0,
    local_cond=None,
    global_cond=None,
    trajectory=None,
):
    for t in range(num_inference_steps):
        t_start = t / num_inference_steps
        t_end = (t + 1) / num_inference_steps
        delta_t = t_end - t_start

        t_start_tensor = torch.ones(trajectory.shape[0]).to(
            trajectory.device) * t_start
        t_mid_tensor = torch.ones(trajectory.shape[0]).to(
            trajectory.device) * (t_start + delta_t / 2)

        # First evaluation
        v0 = model(trajectory,
                   t_start_tensor,
                   local_cond=local_cond,
                   global_cond=global_cond)

        # Midpoint estimation
        x_mid = trajectory + (delta_t / 2) * v0
        v_mid = model(x_mid,
                      t_mid_tensor,
                      local_cond=local_cond,
                      global_cond=global_cond)

        # RK2 update
        trajectory = trajectory + delta_t * v_mid
    return trajectory


class CFMUnetImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            **kwargs):
        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False)
        self.inference_method = kwargs.get("inference_method", RK2)
        if num_inference_steps is None:
            num_inference_steps = getattr(noise_scheduler, "config", {}).get("num_train_timesteps", 20)
        self.num_inference_steps = num_inference_steps

    def conditional_sample(self, condition_data, condition_mask, local_cond=None, global_cond=None, generator=None, noise=None, **kwargs):
        model = self.model
        if noise is not None:
            trajectory = noise.clone()
        else:
            trajectory = torch.randn(size=condition_data.shape, dtype=condition_data.dtype, device=condition_data.device, generator=generator)
        trajectory = self.inference_method(
            model,
            self.num_inference_steps,
            local_cond=local_cond,
            global_cond=global_cond,
            trajectory=trajectory,
        )
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], noise=None) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        device = self.device
        dtype = self.dtype
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            noise=noise,
            **self.kwargs)
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps, noisy_trajectory, ut = self.noise_scheduler.sample_location_and_conditional_flow(
            noise, trajectory)
        loss_mask = ~condition_mask
        pred = self.model(noisy_trajectory, timesteps.reshape(-1), local_cond=local_cond, global_cond=global_cond.clone() if global_cond is not None else None)
        target = ut.clone()
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss 