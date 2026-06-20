from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.diffusion.mlp_denoiser import MLPDenoiser
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


class DiffusionMLPImagePolicy(DiffusionUnetImagePolicy):
    """BC diffusion policy with an MLP denoiser instead of the 1D-conv U-Net.

    Reuses all of ``DiffusionUnetImagePolicy`` (obs encoding, conditional_sample,
    predict_action, compute_loss, masking, set_normalizer) and only swaps the
    denoiser (``self.model``). Obs-only (no search conditioning); at eval time it
    is auto-wrapped by ``BCSearchWrapper`` like any other BC policy.
    """

    def __init__(
            self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            # MLP denoiser hyperparameters
            time_embed_dim=128,
            hidden_dim=1024,
            hidden_depth=4,
            **kwargs):
        assert obs_as_global_cond, \
            "DiffusionMLPImagePolicy requires obs_as_global_cond=True"
        super().__init__(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            obs_as_global_cond=obs_as_global_cond,
            **kwargs,
        )
        # Replace the U-Net denoiser built by the base class with an MLP denoiser.
        self.model = MLPDenoiser(
            input_dim=self.action_dim,
            horizon=self.horizon,
            global_cond_dim=self.obs_feature_dim * self.n_obs_steps,
            time_embed_dim=time_embed_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        )
