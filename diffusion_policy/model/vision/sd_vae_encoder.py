"""The Stable Diffusion VAE encoder as an observation backbone.

NO VAE IS IMPLEMENTED HERE. Every module and weight comes from
`AutoencoderKL.from_pretrained`; this file is an adapter, and exists only because three
things about that class do not match what `MultiImageObsEncoder` asks of an `rgb_model`:

  * `vae.encode(x)` returns an `AutoencoderKLOutput`, not a tensor.
  * `vae.encoder(x)` IS a tensor, but `(B, 8, H/8, W/8)` -- four-dimensional, and eight
    channels (mean and logvar concatenated, before `quant_conv`). The contract is
    `(B, 3, H, W) -> (B, D)`, and `output_shape()[0]` would otherwise report 8.
  * `latent_dist.sample()` differs from call to call while `.mean` does not, so which of
    them is used is a decision rather than a default.

Input range needs no adaptation: LinearNormalizer already maps images to [-1, 1], which is
what the SD VAE expects. `imagenet_norm` must stay off, and `use_group_norm` is a no-op here
(the VAE has no BatchNorm2d for `replace_submodules` to find).
"""
import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class SDVAEEncoder(nn.Module):
    """(B, 3, H, W) in [-1,1] -> (B, 4 * H/8 * W/8), the flattened scaled latent mean.

    At the 72x72 crop that is a 4x9x9 latent, i.e. 324 dims, against the ResNet18's 512.
    Trainable by default, exactly as the ResNet it replaces is -- so swapping the two varies
    the architecture and nothing else.
    """

    def __init__(
            self,
            model_name: str = 'stabilityai/sd-vae-ft-mse',
            scaling_factor: float = 0.18215,
        ) -> None:
        super().__init__()
        vae = AutoencoderKL.from_pretrained(model_name)
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        # The decoder and post_quant_conv are dropped rather than kept unused: 49M of the
        # VAE's 83.7M parameters are never reached on this path, and holding them would put
        # them in the model, its EMA copy and both Adam moments of every checkpoint.
        #
        # SD's latent scale. Hardcoded because diffusers 0.11.1 does not carry
        # `scaling_factor` in the checkpoint config; measured on PushT frames it puts the
        # latent at mean 0.48 / std 1.00, so it is calibrated for this data too.
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The posterior MEAN, never a sample. Under the per-slot corruption ladder
        # (slot_obs_noise) the corruption is meant to be the one controlled noise source; a
        # sampled latent would add a second, uncontrolled one that redraws per call and
        # break the "one observation, progressively degraded" semantics _corrupt_scope
        # exists to guarantee.
        mean, _logvar = self.quant_conv(self.encoder(x)).chunk(2, dim=1)
        return (mean * self.scaling_factor).flatten(start_dim=1)
