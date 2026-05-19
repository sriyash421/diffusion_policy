from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from l2s.verifier import MazeVerifier


class SearchTransformerForDiffusion(nn.Module):
    """Diffusion transformer conditioned on obs and previous search candidates."""

    def __init__(
            self,
            action_dim: int,
            obs_feature_dim: int,
            horizon: int,
            n_obs_steps: int,
            max_context_actions: int,
            n_layer: int = 8,
            n_cond_layers: int = 0,
            n_head: int = 4,
            n_emb: int = 256,
            p_drop_emb: float = 0.0,
            p_drop_attn: float = 0.3,
            causal_attn: bool = True,
        ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.max_context_actions = max_context_actions
        self.max_candidates = max_context_actions + 1
        self.n_head = n_head
        self.n_cond_layers = n_cond_layers

        self.input_emb = nn.Linear(action_dim, n_emb)
        self.obs_emb = nn.Linear(obs_feature_dim, n_emb)
        self.action_value_emb = nn.Linear(horizon * action_dim + 1, n_emb)
        self.time_emb = SinusoidalPosEmb(n_emb)

        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.cond_pos_emb = nn.Parameter(
            torch.zeros(1, n_obs_steps + max_context_actions, n_emb)
        )
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
        elif isinstance(module, SearchTransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    def _normalize_timesteps(
            self,
            timestep: Union[torch.Tensor, float, int],
            batch_size: int,
            n_candidates: int,
            device: torch.device,
        ) -> torch.Tensor:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
        else:
            timesteps = timesteps.to(device)

        if len(timesteps.shape) == 0:
            timesteps = timesteps[None]
        if timesteps.shape == (batch_size * n_candidates,):
            timesteps = timesteps.reshape(batch_size, n_candidates)
        elif timesteps.shape == (batch_size,):
            timesteps = timesteps[:, None].expand(-1, n_candidates)
        elif timesteps.shape == (1,):
            timesteps = timesteps.expand(batch_size, n_candidates)
        elif timesteps.shape != (batch_size, n_candidates):
            raise ValueError(
                f"Expected timestep shape scalar, ({batch_size},), "
                f"({batch_size * n_candidates},), or "
                f"({batch_size}, {n_candidates}); got {tuple(timesteps.shape)}"
            )
        return timesteps

    def _build_tgt_mask(
            self,
            n_candidates: int,
            device: torch.device,
        ) -> torch.Tensor:
        L = n_candidates * self.horizon
        candidate_ids = torch.arange(n_candidates, device=device).repeat_interleave(self.horizon)
        step_ids = torch.arange(self.horizon, device=device).repeat(n_candidates)
        invalid = candidate_ids[:, None] != candidate_ids[None, :]
        if self.mask is not None:
            invalid = invalid | (
                (candidate_ids[:, None] == candidate_ids[None, :])
                & (step_ids[:, None] < step_ids[None, :])
            )
        return invalid

    def _build_encoder_mask(self, device: torch.device) -> torch.Tensor:
        S = self.n_obs_steps + self.max_context_actions
        obs_ids = torch.arange(self.n_obs_steps, device=device)
        action_ids = torch.arange(self.max_context_actions, device=device)
        invalid = torch.ones(S, S, dtype=torch.bool, device=device)

        invalid[obs_ids[:, None], obs_ids[None, :]] = False

        query_actions = self.n_obs_steps + action_ids
        invalid[query_actions[:, None], obs_ids[None, :]] = False
        causal_actions = action_ids[:, None] >= action_ids[None, :]
        invalid[
            query_actions[:, None],
            self.n_obs_steps + action_ids[None, :],
        ] = ~causal_actions
        return invalid

    def _build_memory_masks(
            self,
            batch_size: int,
            n_candidates: int,
            n_context_actions: int,
            context_lengths: Optional[torch.Tensor],
            device: torch.device,
        ):
        S = self.cond_pos_emb.shape[1]
        action_offset = self.n_obs_steps

        if context_lengths is None:
            if n_candidates == 1:
                context_lengths = torch.full(
                    (batch_size, 1),
                    n_context_actions,
                    dtype=torch.long,
                    device=device,
                )
            else:
                context_lengths = torch.arange(
                    n_candidates,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0).expand(batch_size, -1)
        else:
            context_lengths = context_lengths.to(device=device, dtype=torch.long)
            if context_lengths.shape == (batch_size,):
                context_lengths = context_lengths[:, None].expand(-1, n_candidates)
            elif context_lengths.shape != (batch_size, n_candidates):
                raise ValueError(
                    f"Expected context_lengths shape ({batch_size},) or "
                    f"({batch_size}, {n_candidates}); got {tuple(context_lengths.shape)}"
                )

        context_lengths = context_lengths.clamp(min=0, max=n_context_actions)
        invalid = torch.ones(
            batch_size,
            n_candidates,
            S,
            dtype=torch.bool,
            device=device,
        )
        invalid[:, :, :action_offset] = False

        action_positions = torch.arange(self.max_context_actions, device=device)
        valid_actions = action_positions[None, None, :] < context_lengths[:, :, None]
        invalid[:, :, action_offset:] = ~valid_actions

        memory_mask = invalid.repeat_interleave(self.horizon, dim=1)
        memory_mask = memory_mask.unsqueeze(1).expand(
            batch_size,
            self.n_head,
            n_candidates * self.horizon,
            S,
        ).reshape(batch_size * self.n_head, n_candidates * self.horizon, S)

        key_padding = torch.ones(batch_size, S, dtype=torch.bool, device=device)
        key_padding[:, :action_offset] = False
        if n_context_actions > 0:
            key_padding[:, action_offset:action_offset + n_context_actions] = False
        return memory_mask, key_padding

    def forward(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            obs_cond: torch.Tensor,
            actions: Optional[torch.Tensor] = None,
            values: Optional[torch.Tensor] = None,
            context_lengths: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """
        sample: (B, horizon, action_dim), noisy action trajectory to denoise.
        obs_cond: (B, n_obs_steps, obs_feature_dim), encoded observation context.
        actions: optional (B, K, horizon, action_dim), previous generated candidates.
        values: optional (B, K), verifier values for previous candidates.
        context_lengths: optional (B,), number of previous candidates visible per row.
        """
        squeeze_candidate_dim = False
        if sample.ndim == 3:
            sample = sample.unsqueeze(1)
            squeeze_candidate_dim = True
        elif sample.ndim != 4:
            raise ValueError(
                "Expected sample shape (B, horizon, action_dim) or "
                "(B, K, horizon, action_dim)"
            )

        B, K_decode, H, Da = sample.shape
        assert H == self.horizon
        assert Da == self.action_dim
        assert K_decode <= self.max_candidates, \
            f"Got {K_decode} candidates, max is {self.max_candidates}"
        device = sample.device
        timesteps = self._normalize_timesteps(timestep, B, K_decode, device)

        if actions is None:
            K_context = 0
            action_value_tokens = torch.zeros(
                B,
                self.max_context_actions,
                self.cond_pos_emb.shape[-1],
                dtype=sample.dtype,
                device=device,
            )
        else:
            K_context = actions.shape[1]
            assert K_context <= self.max_context_actions, \
                f"Got {K_context} context actions, max is {self.max_context_actions}"
            assert values is not None
            action_value_input = torch.cat([
                actions.reshape(B, K_context, -1).to(device=device, dtype=sample.dtype),
                values.to(device=device, dtype=sample.dtype).unsqueeze(-1),
            ], dim=-1)
            action_value_tokens = self.action_value_emb(action_value_input)
            if K_context < self.max_context_actions:
                pad = torch.zeros(
                    B,
                    self.max_context_actions - K_context,
                    action_value_tokens.shape[-1],
                    dtype=action_value_tokens.dtype,
                    device=device,
                )
                action_value_tokens = torch.cat([action_value_tokens, pad], dim=1)

        memory = torch.cat([
            self.obs_emb(obs_cond),
            action_value_tokens,
        ], dim=1)
        memory = self.drop(memory + self.cond_pos_emb[:, :memory.shape[1], :])
        memory_mask, memory_key_padding_mask = self._build_memory_masks(
            batch_size=B,
            n_candidates=K_decode,
            n_context_actions=K_context,
            context_lengths=context_lengths,
            device=device,
        )
        if isinstance(self.encoder, nn.TransformerEncoder):
            memory = self.encoder(
                memory,
                mask=self._build_encoder_mask(device),
                src_key_padding_mask=memory_key_padding_mask,
            )
        else:
            memory = self.encoder(memory)

        x = self.input_emb(sample)
        pos_emb = self.pos_emb[:, None, :self.horizon, :]
        time_emb = self.time_emb(timesteps.reshape(-1)).reshape(B, K_decode, 1, -1)
        x = self.drop(x + pos_emb + time_emb)
        x = x.reshape(B, K_decode * self.horizon, -1)
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self._build_tgt_mask(K_decode, device),
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        x = self.ln_f(x)
        x = self.head(x).reshape(B, K_decode, self.horizon, self.action_dim)
        if squeeze_candidate_dim:
            x = x[:, 0]
        return x


class DiffusionTransformerSearchPolicy(BaseImagePolicy):
    def __init__(
            self,
            shape_meta: dict[str, Any],
            obs_encoder: MultiImageObsEncoder,
            noise_scheduler: DDPMScheduler,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
            num_inference_steps=None,
            n_layer: int = 8,
            n_cond_layers: int = 0,
            n_head: int = 4,
            n_emb: int = 256,
            p_drop_emb: float = 0.0,
            p_drop_attn: float = 0.3,
            causal_attn: bool = True,
            corrupt_obs: bool = False,
            **kwargs,
        ):
        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]
        max_actions = kwargs['max_actions']

        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.verifier = MazeVerifier(
            maze_path=kwargs.get('maze_path', None),
            device=kwargs.get('device', 'cpu'),
            noise=kwargs.get('verifier_noise', 0.0),
        )
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.max_actions = max_actions
        self.kwargs = kwargs
        self.step_kwargs = kwargs.get('scheduler_step_kwargs', dict())

        self.model = SearchTransformerForDiffusion(
            action_dim=action_dim,
            obs_feature_dim=obs_feature_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            max_context_actions=max_actions - 1,
            n_layer=n_layer,
            n_cond_layers=n_cond_layers,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
        )

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.corrupt_obs = corrupt_obs
        self.obs_noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.001,
            beta_end=0.02,
            prediction_type="epsilon",
        )
        if corrupt_obs:
            print("Corrupting obs with a separate noise scheduler")

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def encode_obs_cond(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        nobs = self.normalizer.normalize(obs_dict)
        obs_value = next(iter(nobs.values()))
        B = obs_value.shape[0]
        To = self.n_obs_steps
        if isinstance(nobs, dict):
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]),
            )
        else:
            this_nobs = nobs[:, :To, ...].reshape(-1, *nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, To, -1)
        return self.corrupt_obs_features(nobs_features)

    def corrupt_obs_features(self, obs_features):
        if not self.corrupt_obs:
            return obs_features

        obs_noise = torch.randn_like(obs_features)
        bsz = obs_features.shape[0]
        timesteps = torch.randint(
            0,
            self.obs_noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=obs_features.device,
        ).long()
        return self.obs_noise_scheduler.add_noise(
            obs_features,
            obs_noise,
            timesteps,
        )

    def conditional_sample(
            self,
            obs_cond: torch.Tensor,
            actions: Optional[torch.Tensor] = None,
            values: Optional[torch.Tensor] = None,
            context_lengths: Optional[torch.Tensor] = None,
            generator=None,
            **kwargs,
        ) -> torch.Tensor:
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=(obs_cond.shape[0], self.horizon, self.action_dim),
            dtype=obs_cond.dtype,
            device=obs_cond.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            model_output = self.model(
                trajectory,
                t,
                obs_cond=obs_cond,
                actions=actions,
                values=values,
                context_lengths=context_lengths,
            )
            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
                **kwargs,
            ).prev_sample
        return trajectory

    def _predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor],
            actions: Optional[torch.Tensor] = None,
            values: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        obs_cond = self.encode_obs_cond(obs_dict)
        nsample = self.conditional_sample(
            obs_cond=obs_cond,
            actions=actions,
            values=values,
            **self.step_kwargs,
        )
        action_pred = self.normalizer['action'].unnormalize(nsample)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {
            'action': action,
            'action_pred': action_pred,
        }

    def predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions,
        ):
        actions = None
        values = None
        for _ in range(n_actions):
            new_action = self._predict_action(
                obs_dict,
                actions=actions,
                values=values,
            )['action_pred']
            new_value = verifier.get_value(obs_dict, new_action)
            if actions is None:
                actions = new_action.unsqueeze(1)
                values = new_value.unsqueeze(1)
            else:
                actions = torch.cat([actions, new_action.unsqueeze(1)], dim=1)
                values = torch.cat([values, new_value.unsqueeze(1)], dim=1)
        return actions, values

    @torch.inference_mode()
    def predict_n_actions(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions,
        ):
        if n_actions <= self.max_actions:
            return self.predict_action(obs_dict, verifier, n_actions)

        actions, values = self.predict_action(obs_dict, verifier, self.max_actions)
        all_actions = actions.clone()
        all_values = values.clone()

        action_history = actions[:, 1:]
        value_history = values[:, 1:]
        for _ in range(self.max_actions, n_actions):
            new_action = self._predict_action(
                obs_dict,
                actions=action_history,
                values=value_history,
            )['action_pred']
            new_value = verifier.get_value(obs_dict, new_action)

            all_actions = torch.cat([all_actions, new_action.unsqueeze(1)], dim=1)
            all_values = torch.cat([all_values, new_value.unsqueeze(1)], dim=1)
            action_history = torch.cat([action_history[:, 1:], new_action.unsqueeze(1)], dim=1)
            value_history = torch.cat([value_history[:, 1:], new_value.unsqueeze(1)], dim=1)

        return all_actions, all_values

    def compute_loss(self, batch):
        target_actions = self.normalizer['action'].normalize(batch['action'])
        B, T, Da = target_actions.shape
        assert T == self.horizon, \
            f"Expected action horizon {self.horizon}, got {T}"
        assert Da == self.action_dim, \
            f"Expected action dim {self.action_dim}, got {Da}"

        obs_cond = self.encode_obs_cond(batch['obs'])

        with torch.inference_mode():
            actions, values = self.predict_action(
                batch['obs'],
                verifier=self.verifier,
                n_actions=self.max_actions - 1,
            )

        trajectory = target_actions.unsqueeze(1).expand(
            -1,
            self.max_actions,
            -1,
            -1,
        )

        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B, self.max_actions),
            device=trajectory.device,
        ).long()
        flat_trajectory = trajectory.reshape(
            B * self.max_actions,
            self.horizon,
            self.action_dim,
        )
        flat_noise = noise.reshape(
            B * self.max_actions,
            self.horizon,
            self.action_dim,
        )
        noisy_trajectory = self.noise_scheduler.add_noise(
            flat_trajectory,
            flat_noise,
            timesteps.reshape(-1),
        ).reshape(B, self.max_actions, self.horizon, self.action_dim)

        pred = self.model(
            noisy_trajectory,
            timesteps,
            obs_cond=obs_cond,
            actions=actions,
            values=values,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        return loss.mean()
