from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
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
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]
        # create diffusion model
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

        # obs keys this policy actually consumes; anything else a runner happens to put
        # in the obs dict is dropped before normalize/encode (the normalizer is fit on
        # exactly these keys, so an extra one would KeyError).
        self.obs_keys = tuple(shape_meta['obs'].keys())
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        # Search-capable subclasses (PushTUNetSearchPolicy via SearchProcedureMixin) provide
        # _init_noise, which gives each EPISODE its own noise stream at eval so the sample
        # does not depend on batch position. Plain DiffusionUnetImagePolicy has no such
        # attribute and keeps the original global-RNG draw, bit-identical.
        _draw = getattr(self, '_init_noise', None)
        if _draw is None:
            trajectory = torch.randn(
                size=condition_data.shape,
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator)
        else:
            trajectory = _draw(tuple(condition_data.shape), condition_data.dtype,
                               condition_data.device, generator)
    
        # set step values -- only when they would change; see the search policy's copy.
        if getattr(scheduler, 'num_inference_steps', None) != self.num_inference_steps:
            scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def _select_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Keep only the obs keys declared in shape_meta."""
        missing = [k for k in self.obs_keys if k not in obs_dict]
        assert not missing, f'obs missing shape_meta keys: {missing}'
        return {k: obs_dict[k] for k in self.obs_keys}

    def _encode_images(self, this_nobs, batch_size):
        """(B*To, ...) normalized obs -> (B*To, D) features.

        A hook, not a helper: the two call sites below are identical and a subclass needs to
        change what happens between them. PushTUNetSearchPolicy overrides it to hand the
        encoder one crop offset per SAMPLE, so the observation window's frames share a crop
        the way the search transformer's already do. The default is exactly the direct call
        it replaces, so every other config using this policy is unchanged.
        """
        return self.obs_encoder(this_nobs)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                       global_cond=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key

        ``global_cond`` is an already-encoded conditioning vector, ``(B, To*D)``. It exists
        so best-of-n does not re-run the vision backbone once per candidate: every candidate
        conditions on the SAME observation, and with a frozen encoder the result is
        bit-identical, so n-1 of those forwards were pure waste. Only meaningful with
        ``obs_as_global_cond``; ``None`` encodes as before.

        Supplying it also skips the NORMALIZE, not just the encode. Under
        ``obs_as_global_cond`` the sole consumer of ``nobs`` is the encode that
        ``global_cond`` stands in for, so the normalizer used to run over the whole image
        tensor once per candidate for a result that was then dropped on the floor. ``B``
        comes off ``global_cond`` instead of off the obs.
        """
        assert 'past_action' not in obs_dict # not implemented yet
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # normalize input -- only when something below will actually read it
        nobs = None
        if global_cond is None:
            nobs = self.normalizer.normalize(self._select_obs(obs_dict))
            B = next(iter(nobs.values())).shape[0]
        else:
            # Asserted rather than ignored. The inpainting branch conditions on
            # PER-TIMESTEP obs features spliced into cond_data, which a flat (B, To*D)
            # vector cannot supply -- so it used to silently drop `global_cond` and
            # re-encode, i.e. exactly the per-candidate waste the argument exists to
            # remove, with nothing to say it had happened.
            assert self.obs_as_global_cond, (
                'global_cond is only meaningful under obs_as_global_cond; the inpainting '
                'branch needs per-timestep obs features and cannot consume it.')
            B = global_cond.shape[0]

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            if global_cond is None:
                this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                nobs_features = self._encode_images(this_nobs, B)
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self._encode_images(this_nobs, B)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(self._select_obs(batch['obs']))
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self._encode_images(this_nobs, batch_size)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self._encode_images(this_nobs, batch_size)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss
