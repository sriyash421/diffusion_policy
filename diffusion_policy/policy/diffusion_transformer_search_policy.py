from typing import Any, Dict, Optional, Union
import contextlib
import math

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
# NOTE: `l2s.verifier.MazeVerifier` is an external, maze-only dependency that is not
# installed in every environment (e.g. the PushT setup). It is imported lazily inside
# __init__ (only when actually building the maze verifier) so that importing this module
# -- and subclassing it for other tasks -- never crashes when `l2s` is absent.


# How predict_action_best picks the action it returns. See
# DiffusionTransformerSearchPolicy.__init__ for what each one means.
SELECTION_MODES = ('argmax', 'softmax', 'final_pass', 'index')


def _stack_subgoals(per_candidate):
    """[{k: (B, ...)}] (one dict per candidate) -> {k: (B, n, ...)}; None if empty."""
    if not per_candidate:
        return None
    return {k: torch.stack([d[k] for d in per_candidate], dim=1)
            for k in per_candidate[0]}


def _cat_subgoals(per_block):
    """[{k: (B, n_i, ...)}] -> {k: (B, sum n_i, ...)}; None if empty."""
    if not per_block:
        return None
    return {k: torch.cat([d[k] for d in per_block], dim=1) for k in per_block[0]}


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
            context_dim: int = 1,
            context_decay: float = 1.0,
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
        # width of the per-candidate feedback appended to each context action token.
        # 1 == the verifier scalar (default); wider when the feedback is a rollout state
        # (see PushTDiffusionSearchPolicy's `search_context` modes).
        self.context_dim = context_dim
        # Recency discount on the search context. For a candidate generated against m context
        # entries, the LATEST entry counts at weight 1, the one before it at `context_decay`,
        # the one before that at context_decay^2 -- entry j gets context_decay^(m-1-j).
        # 1.0 == off.
        #
        # It depends only on distance-from-latest, never on the entry's absolute index, on
        # max_context_actions, or on how many candidates the search was asked for. That is the
        # point: the profile is identical in every loop at every search width, unlike a weight
        # keyed to absolute slot index (see DiffusionTransformerSearchPolicy._slot_weights,
        # whose deployed-slot weight varies with n).
        assert 0.0 < context_decay <= 1.0, \
            f'context_decay must be in (0, 1], got {context_decay}'
        self.context_decay = context_decay

        self.input_emb = nn.Linear(action_dim, n_emb)
        self.obs_emb = nn.Linear(obs_feature_dim, n_emb)
        self.action_value_emb = nn.Linear(horizon * action_dim + context_dim, n_emb)
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

        if self.context_decay >= 1.0:
            memory_mask = invalid
        else:
            # Recency discount, applied as an ADDITIVE ATTENTION BIAS rather than by scaling
            # the context tokens. It has to be: candidate c sees context_lengths[b,c] entries,
            # so "the latest entry" differs per candidate, making the weight a function of the
            # (candidate, entry) PAIR. The memory tensor (B, S, n_emb) is shared by all
            # n_candidates*horizon decoder queries, so a per-token scale cannot express that
            # without materializing n_candidates copies of the memory. This mask is already
            # indexed by exactly that pair.
            #
            # Adding (m-1-j)*log(decay) to a logit multiplies the post-softmax attention
            # weight on entry j by decay^(m-1-j) -- the same arithmetic as scaling the entry,
            # applied where the entry is actually consumed.
            dtype = self.cond_pos_emb.dtype
            dist = (context_lengths[:, :, None] - 1 - action_positions[None, None, :])
            bias = torch.zeros(batch_size, n_candidates, S, dtype=dtype, device=device)
            bias[:, :, action_offset:] = dist.clamp(min=0).to(dtype) * math.log(
                self.context_decay)
            # finfo.min rather than -inf: a fully-masked row would produce NaN under some
            # attention kernels. No row is fully masked here (obs stay visible), so this is
            # defensive, but it costs nothing.
            memory_mask = bias.masked_fill(invalid, torch.finfo(dtype).min)

        memory_mask = memory_mask.repeat_interleave(self.horizon, dim=1)
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
        values: optional (B, K) or (B, K, context_dim), the feedback for the previous
            candidates -- the verifier scalar, and/or the state their rollout reached.
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

        # `actions is None` is "no context supplied"; an EMPTY context tensor is the same
        # thing and must take the same branch. It is not a corner case -- the BC baseline
        # runs at max_actions=1, so max_context_actions is 0 and search_candidates hands
        # back a (B, 0, T, Da) tensor on every call after the first. Falling through to the
        # else branch reshapes 0 elements into (B, 0, -1), where -1 is ambiguous, and
        # raises. Before this guard, BC could be trained and rolled out at n=1 but could
        # not be evaluated at n>1 at all.
        if actions is None or actions.shape[1] == 0 or self.max_context_actions == 0:
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
            values = values.to(device=device, dtype=sample.dtype)
            if values.ndim == 2:
                # (B, K) scalar feedback -> (B, K, 1)
                values = values.unsqueeze(-1)
            assert values.shape[-1] == self.context_dim, \
                f"Got context feedback dim {values.shape[-1]}, expected {self.context_dim}"
            action_value_input = torch.cat([
                actions.reshape(B, K_context, -1).to(device=device, dtype=sample.dtype),
                values,
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
    """Search policy: the standard `predict_action` contract plus a best-of-n readout.

    Public interface, in order of increasing specialization:
      * `predict_action(obs_dict)`   -- BaseImagePolicy contract, one action chunk.
      * `predict_action_best(obs_dict, n_actions)` -- same output format, but the
        argmax-verifier-value candidate out of n. This is what the env runners call.
      * `search_candidates(...)` / `predict_n_actions(...)` -- the raw candidate
        generators, returning (actions, values[, scores][, subgoals]).
    """

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
            crop_shape=None,
            random_crop: bool = True,
            **kwargs,
        ):
        super().__init__()
        self._validate_kwargs(kwargs)
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]
        max_actions = kwargs['max_actions']

        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        # subclasses (e.g. PushTDiffusionSearchPolicy) override _build_verifier to swap
        # in a task-specific verifier without pulling in the maze-only `l2s` package.
        self.verifier = self._build_verifier(**kwargs)
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.max_actions = max_actions
        self.kwargs = kwargs
        self.step_kwargs = kwargs.get('scheduler_step_kwargs', dict())

        # How the candidate that actually gets EXECUTED is picked. Orthogonal to
        # search_context, which only says what each candidate reports back.
        #   'argmax'     -- the verifier scalar ranks the candidates and the best one is
        #                   executed. Every arm before this one.
        #   'final_pass' -- after the n candidates are generated and scored, ONE MORE
        #                   sample is drawn conditioned on all of them, and that sample is
        #                   executed as-is. It is never simulated, and no argmax is taken,
        #                   so the verifier scalar takes no part in selection at all -- it
        #                   only ever reaches the model through the search context. This is
        #                   the only mode in which the deployed action is the model's own
        #                   synthesis rather than an oracle pick.
        #   'softmax'    -- the same verifier scalar ranks the candidates, but the executed
        #                   one is SAMPLED from softmax(z / T) rather than taken greedily,
        #                   where z is the score standardized across the n candidates. So
        #                   the verifier still selects, just stochastically: it is the same
        #                   information used with less commitment, which is the axis
        #                   'final_pass' cannot isolate (that one also changes WHO
        #                   synthesizes the action).
        self.selection = kwargs.get('selection', 'argmax') or 'argmax'
        assert self.selection in SELECTION_MODES, \
            f"selection must be one of {SELECTION_MODES}, got {self.selection!r}"
        # Softmax temperature, on the STANDARDIZED score. Standardizing first is what makes
        # one temperature mean the same thing everywhere: the raw score is -mean keypoint
        # distance in pixels, whose spread differs by arm, by checkpoint and with n, so a
        # fixed T on raw scores would be near-greedy for one arm and near-uniform for
        # another. After z-scoring, T is in units of "standard deviations of candidate
        # quality" and the two limits are exact: T->0 reproduces 'argmax' candidate for
        # candidate, T->inf is a uniform pick among the n.
        self.selection_temperature = float(
            kwargs.get('selection_temperature', 1.0) or 1.0)
        assert self.selection_temperature > 0, 'selection_temperature must be > 0'
        # Restrict the pool that 'argmax'/'softmax' rank over to the LAST W candidates.
        # None == all of them.
        #
        # This is a selection knob only: all n candidates are still generated and still
        # scored, so it costs nothing extra and every candidate's score is still available
        # to a caller that wants to log it. It exists because candidate order is meaningful
        # here -- candidate k is conditioned on candidates 0..k-1, so the trailing ones are
        # the deeply-conditioned ones. "argmax over the last 8 of 16" asks whether the
        # oracle pick is better when it may only choose among well-conditioned candidates,
        # which is a different question from "argmax over all 16".
        window = kwargs.get('selection_window', None)
        self.selection_window = None if window is None else int(window)
        assert self.selection_window is None or self.selection_window > 0, \
            f'selection_window must be positive or None, got {self.selection_window!r}'
        # For selection == 'index': WHICH candidate to execute, as a plain index into the n
        # generated candidates. Negative counts from the end, so -1 is the last (most
        # conditioned) candidate and -8 the eighth from last. The verifier score takes no
        # part in selection at all in this mode -- it is the control that says how much of
        # any best-of-n gain is the ranking rather than the conditioning depth.
        self.selection_index = int(kwargs.get('selection_index', -1) or -1)
        # Geometric decay of the per-candidate-slot loss weight, counting BACK from the
        # last slot: w_k ∝ decay^(K-1-k), so the full-context slot carries the most weight
        # and each earlier slot a factor `decay` less (see _slot_weights / _compute_loss).
        # 1.0 == uniform == the unweighted objective, bit-identical.
        self.slot_weight_decay = float(kwargs.get('slot_weight_decay', 1.0) or 1.0)
        assert 0.0 < self.slot_weight_decay <= 1.0, \
            f'slot_weight_decay must be in (0, 1], got {self.slot_weight_decay}'

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
            context_dim=self._context_dim(obs_feature_dim, **kwargs),
            context_decay=float(kwargs.get('context_decay', 1.0) or 1.0),
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

        # Image crop. Owned by the policy rather than by the obs encoder's CropRandomizer,
        # because only the policy knows WHICH IMAGES BELONG TO THE SAME SAMPLE: the offset
        # drawn for the observation must also be applied to every subgoal image that sample
        # generates, or the model is asked to compare two views of the same scene that are
        # translated relative to each other at train time and aligned at eval time.
        # CropRandomizer cannot be handed an offset -- it draws internally and sits inside
        # an nn.Sequential, which cannot forward extra arguments.
        self.crop_shape = None if crop_shape is None else tuple(crop_shape)
        self.random_crop = random_crop
        # uncropped image size, from shape_meta -- the offsets must be drawn against the
        # size the encoder actually receives
        self._crop_input_hw = (96, 96)
        for attr in shape_meta['obs'].values():
            if attr.get('type', 'low_dim') == 'rgb':
                self._crop_input_hw = tuple(attr['shape'][1:])
                break
        # Offsets come from a dedicated generator seeded from (seed, global_step), so the
        # crop is a pure function of those two: identical across restarts and machines, with
        # no RNG state to checkpoint, and no longer interleaved with the diffusion-noise
        # stream on the global RNG (which is what made it irreproducible after a resume).
        self._crop_generator = torch.Generator(device='cpu')
        self._crop_seed = 0
        self._crop_step = 0
        self._crop_offsets = None
        # `_crop_scope` is reentrant; only the outermost entry resets the offsets, so a
        # search entry point can guarantee a shared offset without disturbing a caller
        # that already opened one. See its docstring.
        self._crop_depth = 0

    # ---------------------------------------------------------------- config validation

    # Every key this class (or a subclass) legitimately reads out of **kwargs. Anything
    # else is a typo, and a typo'd config key used to be silently ignored -- which is how an
    # ablation arm ends up secretly identical to its sibling.
    _KNOWN_KWARGS = frozenset({
        'max_actions', 'search_context', 'selection', 'slot_weight_decay', 'context_decay',
        'selection_temperature', 'selection_window', 'selection_index',
        'scheduler_step_kwargs',
        # PushT verifier
        'verifier_n_envs', 'verifier_legacy', 'verifier_use_async', 'verifier_steps',
        'render_size',
        # maze verifier
        'maze_path', 'device', 'verifier_noise',
    })

    @classmethod
    def _validate_kwargs(cls, kwargs):
        unknown = sorted(set(kwargs) - cls._KNOWN_KWARGS)
        if unknown:
            raise TypeError(
                f'{cls.__name__} got unknown config key(s) {unknown}. These are transported '
                f'through **kwargs and would otherwise be silently ignored, leaving the '
                f'policy on its defaults. Known keys: {sorted(cls._KNOWN_KWARGS)}')

    # ------------------------------------------------------------------------- cropping

    def set_crop_step(self, seed: int, step: int):
        """Fix the (seed, step) the next training crop offsets are derived from.

        Called once per optimizer step by the workspace. Eval never needs it: outside
        train mode the crop is deterministic (center), so no offsets are drawn.
        """
        self._crop_seed = int(seed)
        self._crop_step = int(step)

    def _draw_crop_offsets(self, batch_size: int, height: int, width: int):
        """(B, 2) top-left crop offsets, one per SAMPLE.

        Valid range is [0, H-CH-1] x [0, W-CW-1], matching crop_image_from_indices'
        assertions and sample_random_image_crops' own bound.
        """
        ch, cw = self.crop_shape
        if not (self.training and self.random_crop):
            # center crop -- exactly CropRandomizer.forward_in's eval behaviour
            centre = torch.tensor([(height - ch) // 2, (width - cw) // 2],
                                  dtype=torch.long)
            return centre.unsqueeze(0).expand(batch_size, 2).clone()
        # Deterministic in (seed, step): two calls at the same optimizer step produce the
        # SAME crop, which is what lets the observation's offset be reused for the subgoals
        # generated from it, and what makes a resumed run reproduce an uninterrupted one.
        self._crop_generator.manual_seed(
            (self._crop_seed * 1_000_003 + self._crop_step) % (2 ** 31 - 1))
        dy = torch.randint(0, height - ch, (batch_size,), generator=self._crop_generator)
        dx = torch.randint(0, width - cw, (batch_size,), generator=self._crop_generator)
        return torch.stack([dy, dx], dim=-1)

    def _crop_offsets_for(self, batch_size: int, repeat: int = 1):
        """Per-image crop offsets for a flattened (B*repeat, C, H, W) batch, or None.

        One offset is drawn per SAMPLE and cached for the whole forward pass, then repeated
        across that sample's `repeat` images (the obs window's timesteps). Every later call
        inside the same `_crop_scope` -- notably each candidate's subgoal -- reuses the very
        same per-sample offsets, which is the point: the observation and the subgoals
        predicted from it stay spatially registered, exactly as they are at eval where both
        are center-cropped.
        """
        if self.crop_shape is None:
            return None
        offsets = self._crop_offsets
        if offsets is None or offsets.shape[0] != batch_size:
            offsets = self._draw_crop_offsets(batch_size, *self._crop_input_hw)
            self._crop_offsets = offsets
        if repeat > 1:
            offsets = offsets.repeat_interleave(repeat, dim=0)
        return offsets

    @contextlib.contextmanager
    def _crop_scope(self):
        """Hold one set of crop offsets for the duration of one forward pass.

        The obs encode and every subgoal encode inside the scope reuse the same per-sample
        offsets; the outermost exit clears them so they never leak into the next batch,
        whose batch size may differ.

        REENTRANT, and that is what makes the guarantee hold for every caller. The scope
        used to be opened only by `compute_loss` and `predict_action_best`, so the fix was
        a property of those two paths rather than of the search itself -- and
        `generate_search_context` / `search_candidates` are also called DIRECTLY, by
        TrainSearchOuterInnerWorkspace (which regenerates the context every outer pass) and
        by the diagnostic scripts. Those calls encoded the observation and each candidate's
        subgoal under independent per-image crops, i.e. exactly the split-crop defect the
        policy-level offset was introduced to remove (AUDIT.md P1-1), for any mode whose
        context contains a subgoal image.

        Every search entry point now opens a scope of its own. Nesting is a no-op: only
        depth 0 resets, so `compute_loss` -> `generate_search_context` -> `search_candidates`
        still shares the single offset set drawn at the top, and the offline trainer's
        behaviour is bit-for-bit unchanged.
        """
        outermost = self._crop_depth == 0
        if outermost:
            self._crop_offsets = None
        self._crop_depth += 1
        try:
            yield
        finally:
            self._crop_depth -= 1
            if self._crop_depth == 0:
                self._crop_offsets = None

    def _build_verifier(self, **kwargs):
        """Build the maze verifier. Imported lazily so `l2s` is only required here.

        Subclasses override this to inject a different verifier (see
        PushTDiffusionSearchPolicy) without importing the maze-only package.
        """
        from l2s.verifier import MazeVerifier
        return MazeVerifier(
            maze_path=kwargs.get('maze_path', None),
            device=kwargs.get('device', 'cpu'),
            noise=kwargs.get('verifier_noise', 0.0),
        )

    def _context_dim(self, obs_feature_dim: int, **kwargs) -> int:
        """Width of the per-candidate feedback in the search context.

        1 == the verifier scalar. Subclasses override to feed a richer signal (see
        PushTDiffusionSearchPolicy, which can feed the rollout state or the *encoded*
        subgoal observation instead of / along with the scalar -- hence obs_feature_dim).
        Called from __init__ to size the transformer's context embedding, so it must
        depend only on the config kwargs and the encoder width.
        """
        return 1

    def _score_candidates(self, verifier, obs_dict, action, want_subgoals: bool = False):
        """Evaluate one batch of candidates.

        Returns ``(context, score, subgoal)``:
          * ``context`` (B,) or (B, context_dim) -- the feedback fed back into the search
            context so the next candidate is conditioned on it.
          * ``score`` (B,) -- the scalar used to *rank* candidates (argmax at eval time).
          * ``subgoal`` -- dict of per-candidate debug tensors for logging, or None. Only
            populated when ``want_subgoals``; verifiers without a renderable outcome
            (e.g. the maze one) always return None.
        By default context and score are both the verifier value. Subclasses override to
        widen the context while keeping the scalar ranking signal.
        """
        value = verifier.get_value(obs_dict, action)
        return value, value, None

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _normalize_context_actions(
            self, actions: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Put context actions into the space the model works in.

        The search loop carries candidates in RAW action units because the verifier
        simulates them (``_verifier_inputs`` resets a sim from pixel coords), and the env
        runner executes them. But the model denoises a NORMALIZED trajectory against a
        normalized target, so feeding it raw actions puts the two halves of
        ``action_value_emb``'s input on scales orders of magnitude apart.

        Normalizing here -- at the model boundary, not in the search loop -- keeps one
        tensor from having to serve both boundaries. This mirrors the invariant
        OnlineSearchPolicy maintains: actions exist only in normalized space anywhere the
        model touches them, and are unnormalized strictly at the env/verifier boundary.
        """
        if actions is None:
            return None
        return self.normalizer['action'].normalize(actions)

    def _encode_obs(self, nobs) -> torch.Tensor:
        """Encode an ALREADY-NORMALIZED obs dict (B, T, ...) -> (B, T, obs_feature_dim).

        Same contract as ``OnlineSearchPolicy._encode_obs`` so both search paths share one
        encode structure: normalize at the caller, encode here. That lets the observation
        conditioning and the subgoal context (see PushTDiffusionSearchPolicy) go through
        the same path instead of each re-implementing normalize+encode.

        Online's trailing ``obs_projection`` (obs_feature_dim -> hidden) has its offline
        counterpart inside the transformer as ``SearchTransformerForDiffusion.obs_emb``,
        so there is deliberately no projection here.

        The crop offsets are chosen HERE and handed to the encoder, rather than left to
        CropRandomizer's own per-image sampling: one offset per sample covers the whole obs
        window and, via the surrounding `_crop_scope`, every subgoal image predicted from
        it. Per-image sampling gave each of those its own offset at train time while eval
        center-cropped them all -- the registration mismatch this path exists to avoid.
        """
        if isinstance(nobs, dict):
            value = next(iter(nobs.values()))
            B, T = value.shape[:2]
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        else:
            B, T = nobs.shape[:2]
            this_nobs = nobs.reshape(-1, *nobs.shape[2:])
        offsets = self._crop_offsets_for(B, repeat=T)
        if offsets is None:
            return self.obs_encoder(this_nobs).reshape(B, T, -1)
        return self.obs_encoder(this_nobs, crop_offsets=offsets).reshape(B, T, -1)

    def _encode_obs_features(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Normalize + encode the obs window -> (B, n_obs_steps, obs_feature_dim).

        The expensive, *deterministic* half of encode_obs_cond (for image obs this is the
        ResNet forward). Split out so the search loop can encode once and reuse across
        candidates -- they all condition on the same observation, so re-encoding per
        candidate ran the encoder max_actions times per step for identical inputs.
        """
        nobs = self.normalizer.normalize(obs_dict)
        To = self.n_obs_steps
        if isinstance(nobs, dict):
            nobs = dict_apply(nobs, lambda x: x[:, :To, ...])
        else:
            nobs = nobs[:, :To, ...]
        return self._encode_obs(nobs)

    def encode_obs_cond(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.corrupt_obs_features(self._encode_obs_features(obs_dict))

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

    def predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor],
            actions: Optional[torch.Tensor] = None,
            values: Optional[torch.Tensor] = None,
            obs_features: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
        """Standard ``BaseImagePolicy`` readout: a single (non-searched) action chunk.

        Returns ``{'action': (B, n_action_steps, Da), 'action_pred': (B, horizon, Da)}``,
        so every generic consumer (eval.py, the plain env runners, the workspace's
        sampling block) works against this policy exactly as against any other. The
        best-of-n search readout is ``predict_action_best``; the raw candidate generator
        is ``search_candidates``.

        ``actions``/``values`` optionally supply the search context this chunk is
        conditioned on (used by the search loop; ``None`` for a plain BC readout).
        ``actions`` arrive in RAW action units -- the search loop keeps them raw because the
        verifier simulates them -- and are normalized here, see _normalize_context_actions.
        ``obs_features``: optional pre-computed _encode_obs_features output, reused across
        the candidates of one search loop. Corruption is still drawn per call -- caching
        the corrupted features instead would make every candidate share one noise sample.
        """
        assert 'past_action' not in obs_dict
        if obs_features is None:
            obs_features = self._encode_obs_features(obs_dict)
        obs_cond = self.corrupt_obs_features(obs_features)
        nsample = self.conditional_sample(
            obs_cond=obs_cond,
            actions=self._normalize_context_actions(actions),
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

    def search_candidates(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions,
            return_scores: bool = False,
            obs_features: Optional[torch.Tensor] = None,
            return_subgoals: bool = False,
        ):
        """Generate n_actions candidates, each conditioned on the previous ones.

        Returns ``(actions, values)``, with ``scores`` appended when ``return_scores``
        and ``subgoals`` appended last when ``return_subgoals`` -- i.e. the tuple grows
        left-to-right: ``(actions, values[, scores][, subgoals])``.

        ``values`` is the search *context* feedback -- (B, n) for the scalar verifier
        value, (B, n, context_dim) for a wider context -- while ``scores`` is always the
        (B, n) scalar used to rank candidates. ``subgoals`` is a dict of stacked
        (B, n, ...) debug tensors (or None if the verifier has none); it is for logging
        only and is never fed back into the search.

        ``obs_features`` optionally supplies an already-encoded obs (see
        _encode_obs_features); otherwise it is encoded once here and shared by every
        candidate, since they all condition on the same observation.

        The crop scope is opened HERE rather than only in the callers, so the observation
        and every candidate's subgoal share one crop offset no matter who called -- this is
        the entry point TrainSearchOuterInnerWorkspace and the diagnostic scripts reach
        directly. It is reentrant, so the offline trainer's nested call is a no-op.
        """
        with self._crop_scope():
            if obs_features is None:
                obs_features = self._encode_obs_features(obs_dict)
            actions = None
            values = None
            scores = None
            subgoals = list()
            for _ in range(n_actions):
                new_action = self.predict_action(
                    obs_dict,
                    actions=actions,
                    values=values,
                    obs_features=obs_features,
                )['action_pred']
                new_value, new_score, new_subgoal = self._score_candidates(
                    verifier, obs_dict, new_action, want_subgoals=return_subgoals)
                if actions is None:
                    actions = new_action.unsqueeze(1)
                    values = new_value.unsqueeze(1)
                    scores = new_score.unsqueeze(1)
                else:
                    actions = torch.cat([actions, new_action.unsqueeze(1)], dim=1)
                    values = torch.cat([values, new_value.unsqueeze(1)], dim=1)
                    scores = torch.cat([scores, new_score.unsqueeze(1)], dim=1)
                if new_subgoal is not None:
                    subgoals.append(new_subgoal)

        out = (actions, values)
        if return_scores:
            out = out + (scores,)
        if return_subgoals:
            out = out + (_stack_subgoals(subgoals),)
        return out

    @torch.inference_mode()
    def predict_n_actions(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions,
            return_scores: bool = False,
            return_subgoals: bool = False,
            obs_features: Optional[torch.Tensor] = None,
        ):
        """Search with a rolling context window; see search_candidates for the return shape.

        ``obs_features`` optionally supplies an already-encoded obs so a caller that needs
        the features for something else too (``predict_action_best`` in 'final_pass' mode
        draws one extra sample from them) pays for a single encoder pass rather than two.

        Opens its own (reentrant) crop scope. The rolling-window branch below encodes
        subgoals in a loop of its own rather than inside `search_candidates`, so wrapping
        only that method would leave every candidate past `max_actions` on an independent
        crop -- the exact split-crop defect, reappearing only at n > K.
        """
        with self._crop_scope():
            # encode once for the whole search, however many candidates it runs
            if obs_features is None:
                obs_features = self._encode_obs_features(obs_dict)
            if n_actions <= self.max_actions:
                return self.search_candidates(
                    obs_dict, verifier, n_actions, return_scores=return_scores,
                    obs_features=obs_features, return_subgoals=return_subgoals)

            # scores are always needed internally (the caller may only want values), but
            # subgoals are only rendered when actually asked for.
            head = self.search_candidates(
                obs_dict, verifier, self.max_actions, return_scores=True,
                obs_features=obs_features, return_subgoals=return_subgoals)
            actions, values, scores = head[0], head[1], head[2]
            subgoals = head[3] if return_subgoals else None
            all_actions = actions.clone()
            all_values = values.clone()
            all_scores = scores.clone()
            all_subgoals = [subgoals] if subgoals is not None else []

            action_history = actions[:, 1:]
            value_history = values[:, 1:]
            for _ in range(self.max_actions, n_actions):
                new_action = self.predict_action(
                    obs_dict,
                    actions=action_history,
                    values=value_history,
                    obs_features=obs_features,
                )['action_pred']
                new_value, new_score, new_subgoal = self._score_candidates(
                    verifier, obs_dict, new_action, want_subgoals=return_subgoals)

                all_actions = torch.cat([all_actions, new_action.unsqueeze(1)], dim=1)
                all_values = torch.cat([all_values, new_value.unsqueeze(1)], dim=1)
                all_scores = torch.cat([all_scores, new_score.unsqueeze(1)], dim=1)
                action_history = torch.cat(
                    [action_history[:, 1:], new_action.unsqueeze(1)], dim=1)
                value_history = torch.cat(
                    [value_history[:, 1:], new_value.unsqueeze(1)], dim=1)
                if new_subgoal is not None:
                    # already (B, 1, ...) from the inner stack vs (B, ...) from the loop
                    all_subgoals.append({k: v.unsqueeze(1) for k, v in new_subgoal.items()})

        out = (all_actions, all_values)
        if return_scores:
            out = out + (all_scores,)
        if return_subgoals:
            out = out + (_cat_subgoals(all_subgoals),)
        return out

    @torch.inference_mode()
    def predict_action_best(
            self,
            obs_dict: Dict[str, torch.Tensor],
            n_actions: Optional[int] = None,
        ) -> Dict[str, torch.Tensor]:
        """Search readout, in the standard ``predict_action`` output format.

        Generates ``n_actions`` candidates via the sliding-window search
        (``predict_n_actions``), scores each with this policy's verifier, and returns one
        action chunk as ``{'action': (B, n_action_steps, Da), 'action_pred': (B, horizon,
        Da), 'scores': (B, n)}`` -- the shape the env runners and MultiStepWrapper expect.
        (Tensors only: the runners push this dict straight through ``dict_apply``, so a
        string mode tag here would crash them. Read the mode off ``policy.selection``.)

        WHICH chunk depends on ``self.selection``:
          * 'argmax'     -- the argmax-verifier-value candidate. Best-of-n over an oracle.
          * 'softmax'    -- sampled from softmax(z / T) over the standardized scores.
          * 'index'      -- a FIXED slot (``self.selection_index``, negative counts from the
            end), ignoring the scores entirely. The control for the two above: it isolates
            how much of a best-of-n gain comes from the ranking rather than from candidate
            k simply being conditioned on candidates 0..k-1.
          * 'final_pass' -- one MORE sample, conditioned on all n scored candidates, is
            drawn and returned. It is not simulated and not compared to anything, so the
            verifier scalar never touches selection; it reaches the model only as search
            context. ``scores`` still describes the n context candidates (so the caller can
            log the spread), but no longer describes the returned action.

        ``self.selection_window`` narrows the pool 'argmax'/'softmax' rank over to the last
        W candidates. It does not change generation: all n are still sampled and scored, so
        ``scores`` is always the full (B, n) and ``pick`` is always a full-candidate index.

        Cost at width n: 'argmax' is n samples + n sims, 'final_pass' is n+1 samples + n
        sims. Consumers that compare arms should compare at equal SAMPLES, not equal n --
        see the ``n_generations`` field eval_search_pusht.py records.
        """
        n = n_actions if n_actions is not None else self.max_actions
        # `scores` is the scalar verifier value in every search_context mode; `values` may
        # be a wider context (e.g. a subgoal state), which is not rankable.
        # The crop scope spans the whole search so the obs and every candidate's subgoal
        # share one offset, exactly as in training. (In eval mode that offset is the
        # deterministic center crop, so this is belt-and-braces rather than load-bearing.)
        with self._crop_scope():
            # encoded once and shared by the search AND (in 'final_pass') the extra sample
            obs_features = self._encode_obs_features(obs_dict)
            actions, values, scores = self.predict_n_actions(
                obs_dict, verifier=self.verifier, n_actions=n,
                return_scores=True, obs_features=obs_features)  # (B,n,H,Da), ctx, (B,n)

            if self.selection == 'index':
                # Fixed slot, no ranking. Resolved against the candidates ACTUALLY
                # generated (n, which may be below max_actions), and clamped so a request
                # for the 8th-from-last at n=4 takes candidate 0 rather than wrapping round
                # to a silently different slot.
                B = actions.shape[0]
                idx = self.selection_index
                if idx < 0:
                    idx = n + idx
                idx = max(0, min(n - 1, idx))
                pick = torch.full((B,), idx, dtype=torch.long, device=actions.device)
                action_pred = actions[:, idx]                           # (B, H, Da)
            elif self.selection in ('argmax', 'softmax'):
                B = actions.shape[0]
                arange = torch.arange(B, device=actions.device)
                # Rank over the last `selection_window` candidates only, when set. `lo` is
                # the offset of that window in the full candidate axis, so `pick` is
                # returned in FULL-candidate coordinates -- callers logging it alongside
                # `scores` (which is always all n) would otherwise be off by `lo`.
                lo = 0 if self.selection_window is None \
                    else max(0, n - self.selection_window)
                pool = scores[:, lo:]                                   # (B, n-lo)
                if self.selection == 'argmax':
                    pick = pool.argmax(dim=1) + lo                      # (B,)
                else:
                    # z-score across the pooled candidates, then sample. unbiased=False so a
                    # single candidate (n=1) gives std 0 rather than NaN; the eps then
                    # leaves z==0, i.e. a uniform draw over one option -- the same action
                    # argmax would have taken, so the n=1 column is comparable by
                    # construction rather than by luck. Standardizing over the POOL, not
                    # over all n, keeps T in units of "sd of the candidates being chosen
                    # among" in both the windowed and unwindowed cases.
                    mu = pool.mean(dim=1, keepdim=True)
                    sd = pool.std(dim=1, unbiased=False, keepdim=True)
                    z = (pool - mu) / (sd + 1e-6)
                    logits = z / self.selection_temperature
                    pick = torch.distributions.Categorical(logits=logits).sample() + lo
                action_pred = actions[arange, pick]                     # (B, H, Da)
            else:
                # Condition on the last max_actions-1 candidates: that is the widest
                # context the model was ever trained at (the staircase memory mask tops out
                # at max_context_actions), so a longer one would index past cond_pos_emb.
                # `values`, not `scores` -- in the subgoal modes the context is the encoded
                # subgoal observation and the bare scalar would be the wrong width.
                keep = self.max_actions - 1
                action_pred = self.predict_action(
                    obs_dict,
                    actions=actions[:, -keep:],
                    values=values[:, -keep:],
                    obs_features=obs_features,
                )['action_pred']                                        # (B, H, Da)
                # The executed action is a FURTHER sample, not one of the n candidates, so
                # there is no candidate index to report. -1 marks "not a candidate" rather
                # than pointing at an arbitrary slot the caller might plot as chosen.
                pick = torch.full((actions.shape[0],), -1,
                                  dtype=torch.long, device=actions.device)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {
            'action': action_pred[:, start:end],            # (B, n_action_steps, Da)
            'action_pred': action_pred,
            'scores': scores,                               # (B, n) -- ALWAYS all n
            # Which candidate was executed, in full-candidate coordinates; -1 under
            # 'final_pass'. Returned so a caller can record the selection actually made
            # instead of re-deriving it (a re-derivation cannot reproduce a 'softmax' draw
            # at all, and would silently disagree with what was executed).
            'pick': pick,                                   # (B,)
        }

    def generate_search_context(self, obs_dict, obs_features=None):
        """The search context ``compute_loss`` conditions on: ``(actions, values)``.

        Exactly the ``search_candidates`` call the training loss makes, split out so a
        trainer can generate the context ONCE for a pool of windows and reuse it across
        several gradient steps (see TrainSearchOuterInnerWorkspace) instead of paying for
        ``max_actions - 1`` candidate samples plus their verifier rollouts on every update.

        Deliberately ``no_grad`` rather than ``inference_mode``: inference tensors carry a
        permanent restriction against taking part in autograd-recorded operations, which
        makes them unsafe to buffer and feed to the model on a later step. ``no_grad``
        yields ordinary tensors, so the returned context can be stored for as long as the
        caller likes. The search is a pure generator either way -- no graph is built.

        ``actions`` come back in RAW action units (the verifier simulates them); the
        normalization the model needs happens at the model boundary, see
        ``_normalize_context_actions``.
        """
        with torch.no_grad():
            return self.search_candidates(
                obs_dict,
                verifier=self.verifier,
                n_actions=self.max_actions - 1,
                obs_features=obs_features,
            )

    def predict_epsilon(self, batch, actions, values, noisy_trajectory, timesteps):
        """One denoiser forward on EXTERNALLY supplied noise, timesteps and context.

        Used to compare two snapshots of this policy at matched inputs. For a fixed
        ``(noisy_trajectory, timestep, obs, context)`` the two snapshots' reverse
        transition kernels are Gaussians whose variance comes from the scheduler and so is
        identical between them; they differ only in a mean that is affine in the predicted
        noise. Their KL is therefore exactly proportional to ``||eps_a - eps_b||^2``, which
        makes epsilon-space MSE at matched inputs a per-denoising-step KL -- the tractable
        stand-in for the policy KL that PPO would monitor (a diffusion policy has no
        closed-form ``log p(a|s)`` to take a ratio of).

        Obs corruption is deliberately NOT applied: ``corrupt_obs_features`` draws fresh
        noise per call, so including it would book corruption noise as policy drift. The
        obs ARE re-encoded here rather than taken from the caller, because the vision
        backbone is part of what drifts and each snapshot must use its own.
        """
        with torch.no_grad():
            return self.model(
                noisy_trajectory,
                timesteps,
                obs_cond=self._encode_obs_features(batch['obs']),
                actions=self._normalize_context_actions(actions),
                values=values,
            )

    def _slot_weights(self, device, dtype) -> Optional[torch.Tensor]:
        """Per-candidate-slot loss weights, or None for the uniform (unweighted) path.

        WHAT THE SLOTS ARE. One forward decodes all K = max_actions candidate slots, and
        every one of them is trained on the SAME target -- the expert action. What differs
        is how much context each is allowed to see: the staircase memory mask lets slot k
        attend to exactly the first k context candidates. So the model is fitting a family
        of conditionals at once, from "no context" (slot 0, the BC case) up to "K-1 scored
        candidates" (slot K-1).

        WHY WEIGHT THEM. Under ``selection: final_pass`` the executed action is a sample
        drawn with a FULL context, so slot K-1 is the deployment condition and the earlier
        slots exist to manufacture the context it reads. Averaging flat spends 1 - 1/K of
        the gradient on conditionals that arm never executes.

        THE SHAPE. Geometric decay counting back from the last slot,
        ``w_k ∝ decay^(K-1-k)``: weight rises monotonically with context length, each step
        back costing a constant factor. Monotone rather than last-slot-only because the
        earlier slots are not disposable -- they generate the candidates slot K-1 reads, so
        starving them degrades its input. Geometric rather than linear because the thing
        being traded off (how far a slot's conditioning is from deployment) is multiplicative
        in the number of missing context entries, not additive.

        THE NORMALIZATION. Weights are scaled to mean 1, i.e. they sum to K exactly as the
        uniform weights do. Three things downstream read the loss MAGNITUDE --
        ``gradient_clip_norm`` (a uniformly larger loss clips more often), the effective
        step size under a shared ``lr``, and ``val_loss``, which the topk selector monitors
        and the reports compare across arms. Without this, changing the decay would silently
        change all three and the ablation would differ in scale as well as in mechanism.

        Note the model has ONE set of weights shared by every slot -- this reweights which
        conditioning regime those shared weights are tuned for, it does not give the final
        slot parameters of its own.
        """
        if self.slot_weight_decay == 1.0 or self.max_actions == 1:
            return None
        k = torch.arange(self.max_actions, device=device, dtype=dtype)
        w = self.slot_weight_decay ** (self.max_actions - 1 - k)
        return w / w.mean()

    def compute_loss(self, batch, actions=None, values=None, return_aux=False):
        """Denoising loss for the expert action, conditioned on a best-of-n search context.

        ``actions``/``values`` optionally supply a PRE-GENERATED search context, in the
        form ``generate_search_context`` returns it. ``None`` (the default) generates it
        here from the current weights -- the offline path, which pays the full search cost
        on every gradient step. A trainer that amortizes the search across several updates
        passes the buffered context back in instead.

        ``return_aux`` additionally returns the exact inputs the denoiser was called with,
        so a frozen snapshot of this policy can be re-run on identical inputs via
        ``predict_epsilon``; matched inputs are what make that comparison a KL.
        """
        with self._crop_scope():
            return self._compute_loss(batch, actions, values, return_aux)

    def _compute_loss(self, batch, actions=None, values=None, return_aux=False):
        target_actions = self.normalizer['action'].normalize(batch['action'])
        B, T, Da = target_actions.shape
        assert T == self.horizon, \
            f"Expected action horizon {self.horizon}, got {T}"
        assert Da == self.action_dim, \
            f"Expected action dim {self.action_dim}, got {Da}"

        # encode the obs ONCE: the same features back the grad-tracked training forward
        # and every candidate of the context search below (which only reads them, under
        # no_grad, so no graph is built for the search).
        obs_features = self._encode_obs_features(batch['obs'])
        obs_cond = self.corrupt_obs_features(obs_features)

        if actions is None:
            actions, values = self.generate_search_context(
                batch['obs'], obs_features=obs_features)

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
            # search_candidates returns raw-unit actions (the verifier simulates them);
            # the model works in normalized space, same as noisy_trajectory/target.
            actions=self._normalize_context_actions(actions),
            values=values,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')      # (B, K, H, Da)
        weights = self._slot_weights(loss.device, loss.dtype)
        if weights is None:
            loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
        else:
            per_slot = reduce(loss, 'b k ... -> b k', 'mean')  # (B, K)
            loss = (per_slot * weights).mean()
        if not return_aux:
            return loss
        # everything a frozen snapshot needs to reproduce this exact forward
        return loss, {
            'noisy_trajectory': noisy_trajectory,
            'timesteps': timesteps,
            'actions': actions,
            'values': values,
        }
