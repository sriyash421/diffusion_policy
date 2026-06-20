from typing import Union
import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb


class MLPDenoiser(nn.Module):
    """Plain-MLP diffusion denoiser.

    Drop-in replacement for ``ConditionalUnet1D`` with the same call contract:

        forward(sample: (B, T, input_dim), timestep, local_cond=None,
                global_cond: (B, global_cond_dim)) -> (B, T, input_dim)

    The whole action trajectory is flattened to a single vector, concatenated
    with the diffusion-timestep embedding and the (obs) global conditioning, and
    pushed through a residual MLP that predicts the full flattened trajectory.
    Used by ``DiffusionMLPImagePolicy`` as a BC (obs-only) denoiser ablation
    against the U-Net / transformer denoisers.
    """

    def __init__(
            self,
            input_dim: int,
            horizon: int,
            global_cond_dim: int,
            time_embed_dim: int = 128,
            hidden_dim: int = 1024,
            hidden_depth: int = 4,
        ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.global_cond_dim = int(global_cond_dim or 0)

        # Diffusion-step encoder, mirroring ConditionalUnet1D.diffusion_step_encoder.
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        traj_dim = horizon * input_dim
        in_dim = traj_dim + time_embed_dim + self.global_cond_dim

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
            )
            for _ in range(hidden_depth)
        ])
        self.head = nn.Linear(hidden_dim, traj_dim)

    def forward(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            local_cond=None,
            global_cond=None,
            **kwargs,
        ) -> torch.Tensor:
        """
        sample: (B, T, input_dim)
        timestep: (B,) or scalar diffusion step
        global_cond: (B, global_cond_dim)
        output: (B, T, input_dim)
        """
        assert local_cond is None, "MLPDenoiser does not support local_cond"
        B, T, Da = sample.shape
        assert T == self.horizon and Da == self.input_dim

        # 1. time embedding (timestep-prep copied from ConditionalUnet1D.forward)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(B)
        time_emb = self.diffusion_step_encoder(timesteps)  # (B, time_embed_dim)

        # 2. assemble MLP input
        flat = sample.reshape(B, T * Da)
        parts = [flat, time_emb]
        if self.global_cond_dim > 0:
            assert global_cond is not None, "MLPDenoiser was built with global_cond_dim>0"
            parts.append(global_cond)
        x = torch.cat(parts, dim=-1)

        # 3. residual MLP trunk
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)
        out = self.head(x)  # (B, T*Da)
        return out.reshape(B, T, Da)
