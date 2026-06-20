from typing import Union
import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb


class ObsCondTransformerForDiffusion(nn.Module):
    """Obs-conditioned encoder-decoder diffusion denoiser.

    A trimmed copy of ``SearchTransformerForDiffusion`` with the action/value
    search-context tokens removed, so the encoder memory is built from the
    observation only. Drop-in replacement for ``ConditionalUnet1D`` with the
    same call contract:

        forward(sample: (B, T, action_dim), timestep, local_cond=None,
                global_cond: (B, n_obs_steps*obs_feature_dim)) -> (B, T, action_dim)

    Used by ``DiffusionTransformerObsImagePolicy`` as the BC (obs-only) analogue
    of the transformer-denoiser search policy, for an apples-to-apples n=1
    comparison.
    """

    def __init__(
            self,
            action_dim: int,
            obs_feature_dim: int,
            horizon: int,
            n_obs_steps: int,
            n_layer: int = 4,
            n_cond_layers: int = 2,
            n_head: int = 4,
            n_emb: int = 256,
            p_drop_emb: float = 0.0,
            p_drop_attn: float = 0.2,
            causal_attn: bool = False,
        ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_head = n_head
        self.n_cond_layers = n_cond_layers

        self.input_emb = nn.Linear(action_dim, n_emb)
        self.obs_emb = nn.Linear(obs_feature_dim, n_emb)
        self.time_emb = SinusoidalPosEmb(n_emb)

        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, n_obs_steps, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_cond_layers,
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_emb, 4 * n_emb),
                nn.Mish(),
                nn.Linear(4 * n_emb, n_emb),
            )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer,
        )
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, action_dim)

        if causal_attn:
            mask = torch.triu(torch.ones(horizon, horizon), diagonal=1).bool()
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name in ['in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            for name in ['in_proj_bias', 'bias_k', 'bias_v']:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, ObsCondTransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    def forward(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            local_cond=None,
            global_cond=None,
            **kwargs,
        ) -> torch.Tensor:
        """
        sample: (B, T, action_dim)
        timestep: (B,) or scalar diffusion step
        global_cond: (B, n_obs_steps * obs_feature_dim)
        output: (B, T, action_dim)
        """
        assert local_cond is None, "ObsCondTransformerForDiffusion does not support local_cond"
        B, T, Da = sample.shape
        assert T == self.horizon and Da == self.action_dim
        device = sample.device

        # 1. time embedding (timestep-prep copied from ConditionalUnet1D.forward)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)
        timesteps = timesteps.expand(B)
        time_emb = self.time_emb(timesteps).unsqueeze(1)  # (B, 1, n_emb)

        # 2. encoder memory from obs only
        assert global_cond is not None, "ObsCondTransformerForDiffusion requires global_cond"
        obs_cond = global_cond.reshape(B, self.n_obs_steps, self.obs_feature_dim)
        memory = self.obs_emb(obs_cond)  # (B, n_obs_steps, n_emb)
        memory = self.drop(memory + self.cond_pos_emb)
        memory = self.encoder(memory)

        # 3. decode noisy action trajectory conditioned on memory
        x = self.input_emb(sample)  # (B, T, n_emb)
        x = self.drop(x + self.pos_emb + time_emb)
        x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, action_dim)
