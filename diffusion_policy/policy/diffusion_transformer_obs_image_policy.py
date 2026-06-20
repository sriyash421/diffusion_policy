from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.diffusion.obs_cond_transformer import ObsCondTransformerForDiffusion
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


class DiffusionTransformerObsImagePolicy(DiffusionUnetImagePolicy):
    """BC diffusion policy with an obs-conditioned transformer encoder-decoder
    denoiser (the BC analogue of the transformer-denoiser search policy).

    Reuses all of ``DiffusionUnetImagePolicy`` and only swaps the denoiser
    (``self.model``) for ``ObsCondTransformerForDiffusion`` — the search
    transformer minus the action/value search-context tokens. Obs-only (no
    search conditioning); at eval time it is auto-wrapped by ``BCSearchWrapper``.
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
            # transformer denoiser hyperparameters
            n_layer=4,
            n_cond_layers=2,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.2,
            causal_attn=False,
            **kwargs):
        assert obs_as_global_cond, \
            "DiffusionTransformerObsImagePolicy requires obs_as_global_cond=True"
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
        # Replace the U-Net denoiser built by the base class with a transformer.
        self.model = ObsCondTransformerForDiffusion(
            action_dim=self.action_dim,
            obs_feature_dim=self.obs_feature_dim,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            n_layer=n_layer,
            n_cond_layers=n_cond_layers,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
        )
