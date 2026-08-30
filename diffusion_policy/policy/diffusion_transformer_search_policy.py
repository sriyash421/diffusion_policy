from typing import Any, Dict, Optional, Union
import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.selection_util import SELECTION_MODES, select_candidate
from diffusion_policy.model.common.gpt2_trunk import build_gpt2_trunk
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.common.obs_corruption import ObsCorruptionMixin
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.crop_scope import CropScopeMixin
from diffusion_policy.policy.search_procedure import SearchProcedureMixin
# NOTE: `l2s.verifier.MazeVerifier` is an external, maze-only dependency that is not
# installed in every environment (e.g. the PushT setup). It is imported lazily inside
# __init__ (only when actually building the maze verifier) so that importing this module
# -- and subclassing it for other tasks -- never crashes when `l2s` is absent.
#
# `transformers` (for cond_encoder='gpt2') is lazy for the SAME reason. It is declared in
# conda_environment.yaml, but was NOT until this option was added, so any env built before
# then still lacks it -- and a top-level import would take down every arm on such an env,
# including the ones that never ask for this trunk. See _build_cond_encoder.


# SELECTION_MODES is re-exported from common.selection_util so `from
# diffusion_transformer_search_policy import SELECTION_MODES` keeps working.

# What encodes the conditioning memory (the obs tokens plus the search-context tokens)
# before the decoder cross-attends into it. See SearchTransformerForDiffusion.__init__.
COND_ENCODERS = ('transformer', 'gpt2', 'mlp')


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
            cond_encoder: str = 'transformer',
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

        # WHAT ENCODES THE CONDITIONING MEMORY. Only the memory -- the decoder that actually
        # denoises is untouched by this, as is every cross-attention mask.
        #
        #   'transformer' -- nn.TransformerEncoder under `_build_encoder_mask`, which is NOT
        #                    fully causal: the obs tokens attend to each other in BOTH
        #                    directions, while a context token attends to the obs plus the
        #                    context tokens at or before it.
        #   'gpt2'        -- a causal GPT2Model over the same flat token stream, exactly as
        #                    SearchPolicy and OnlineSearchPolicy do it: NO mask is built
        #                    here at all, GPT-2's own triangular mask supplies the
        #                    causality and the only thing passed in is a PADDING mask.
        #                    Relative to 'transformer' precisely one edge disappears -- obs
        #                    step 0 can no longer see obs step 1; every other connection is
        #                    already what causal masking gives.
        #   'mlp'         -- the token-wise Mish MLP, i.e. no cross-token mixing at all.
        #                    Selected by n_cond_layers == 0 regardless of this setting,
        #                    which is the documented way to reclaim the trunk's parameters
        #                    (25.20M at train_pusht_diffusion_search's n_emb 1024, for a 2-token memory).
        assert cond_encoder in COND_ENCODERS, \
            f"cond_encoder must be one of {COND_ENCODERS}, got {cond_encoder!r}"
        # n_cond_layers == 0 means "no encoder at all", and it outranks the trunk choice:
        # a zero-layer GPT-2 is not a thing, and the MLP fallback is the documented way to
        # ask for one. Normalizing here keeps every later branch a plain string compare.
        self.cond_encoder = 'mlp' if n_cond_layers == 0 else cond_encoder

        self.input_emb = nn.Linear(action_dim, n_emb)
        self.obs_emb = nn.Linear(obs_feature_dim, n_emb)
        self.action_value_emb = nn.Linear(horizon * action_dim + context_dim, n_emb)
        self.time_emb = SinusoidalPosEmb(n_emb)

        # Length of the conditioning memory, and the dtype/width its masks are built at.
        # Read directly instead of off `cond_pos_emb.shape`, because that Parameter does not
        # exist in 'gpt2' mode (GPT-2 carries its own `wpe`).
        self._cond_len = n_obs_steps + max_context_actions
        self.n_emb = n_emb

        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        # 'gpt2' adds GPT-2's own learned positional embeddings inside the trunk, so a
        # `cond_pos_emb` here would be a second, redundant positional signal on the same
        # tokens. Omitted rather than zeroed so it cannot be trained into a shadow of wpe.
        self.cond_pos_emb = None if self.cond_encoder == 'gpt2' else nn.Parameter(
            torch.zeros(1, self._cond_len, n_emb)
        )
        self.drop = nn.Dropout(p_drop_emb)

        if self.cond_encoder == 'gpt2':
            # Built AFTER self.apply(self._init_weights) below -- see _build_cond_encoder.
            self.encoder = None
        elif n_cond_layers > 0:
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

        # AFTER _init_weights, deliberately. GPT2Model's own post_init scales every `c_proj`
        # weight by 1/sqrt(2*n_layer) -- the residual-growth correction the architecture is
        # tuned around -- and running our flat std=0.02 pass over it would silently undo
        # that. (Today it would not even get the chance: GPT-2 is built from `Conv1D`, not
        # `nn.Linear`, so the Linear branch below misses it while `nn.Embedding` falls
        # through to the `else` and raises. Ordering makes the guarantee independent of
        # which HF internals happen to match our isinstance chain.)
        if self.cond_encoder == 'gpt2':
            self.encoder = self._build_cond_encoder(
                n_emb=n_emb, n_head=n_head, n_cond_layers=n_cond_layers,
                p_drop_emb=p_drop_emb, p_drop_attn=p_drop_attn)

    def _build_cond_encoder(self, n_emb, n_head, n_cond_layers,
                            p_drop_emb, p_drop_attn):
        """A causal GPT-2 over the conditioning memory, as in the Gaussian search policies.

        Sized to the memory (`_cond_len`), not to a default context. See build_gpt2_trunk
        for why the import is lazy and why vocab_size=1 is load-bearing.
        """
        return build_gpt2_trunk(
            n_emb=n_emb, n_head=n_head, n_layer=n_cond_layers,
            n_positions=self._cond_len,
            p_drop_emb=p_drop_emb, p_drop_attn=p_drop_attn)

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
            # None in 'gpt2' mode, where GPT-2's own `wpe` carries position instead.
            if module.cond_pos_emb is not None:
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
            fold_slots: bool = False,
        ):
        S = self._cond_len
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
            # pos_emb, not cond_pos_emb: same dtype, and it exists in every cond_encoder mode.
            dtype = self.pos_emb.dtype
            dist = (context_lengths[:, :, None] - 1 - action_positions[None, None, :])
            bias = torch.zeros(batch_size, n_candidates, S, dtype=dtype, device=device)
            bias[:, :, action_offset:] = dist.clamp(min=0).to(dtype) * math.log(
                self.context_decay)
            # finfo.min rather than -inf: a fully-masked row would produce NaN under some
            # attention kernels. No row is fully masked here (obs stay visible), so this is
            # defensive, but it costs nothing.
            memory_mask = bias.masked_fill(invalid, torch.finfo(dtype).min)

        if fold_slots:
            # One decoder row per (batch, slot) instead of one per batch: (B*K, H) queries
            # against a per-slot memory. Head-minor, matching the b-major/head-minor layout
            # torch expects for `(N*num_heads, L, S)` -- see the unfolded branch below.
            memory_mask = memory_mask.reshape(batch_size * n_candidates, 1, 1, S).expand(
                batch_size * n_candidates, self.n_head, self.horizon, S,
            ).reshape(batch_size * n_candidates * self.n_head, self.horizon, S)
        else:
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
        if fold_slots:
            key_padding = key_padding.repeat_interleave(n_candidates, dim=0)
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
        obs_cond: (B, n_obs_steps, obs_feature_dim) shared by every slot, or
            (B, K, n_obs_steps, obs_feature_dim) for one view per slot (see the corruption
            ladder). The 4-D form folds K into the batch dimension.
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
        # thing and must take the same branch. It is not a corner case -- the ST k=1 arm
        # runs at max_actions=1, so max_context_actions is 0 and search_candidates hands
        # back a (B, 0, T, Da) tensor on every call after the first. Falling through to the
        # else branch reshapes 0 elements into (B, 0, -1), where -1 is ambiguous, and
        # raises. Before this guard, ST k=1 could be trained and rolled out at n=1 but could
        # not be evaluated at n>1 at all.
        if actions is None or actions.shape[1] == 0 or self.max_context_actions == 0:
            K_context = 0
            action_value_tokens = torch.zeros(
                B,
                self.max_context_actions,
                self.n_emb,
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

        # PER-SLOT CONDITIONING. A 3-D obs_cond is one observation shared by every slot --
        # the original path, left untouched. A 4-D (B, K, To, D) obs_cond carries one view
        # per slot (the corruption ladder), which the shared (B, S, n_emb) memory cannot
        # express, so K folds into the batch dimension: (B*K, S, n_emb) memory against
        # (B*K, H) decoder queries. That fold is EXACTLY equivalent to the joint decode,
        # because _build_tgt_mask already forbids every cross-slot edge -- the slots never
        # attended to one another, so splitting them into separate rows changes nothing but
        # which memory each one reads.
        fold_slots = obs_cond.ndim == 4
        if fold_slots:
            assert obs_cond.shape[1] == K_decode, (
                f'obs_cond has {obs_cond.shape[1]} slots but {K_decode} are being decoded')
            obs_tokens = self.obs_emb(obs_cond)                     # (B, K, To, E)
            action_value_tokens = action_value_tokens.unsqueeze(1).expand(
                B, K_decode, *action_value_tokens.shape[1:])        # (B, K, MCA, E)
            memory = torch.cat([obs_tokens, action_value_tokens], dim=2)
            memory = memory.reshape(B * K_decode, memory.shape[2], memory.shape[3])
        else:
            memory = torch.cat([
                self.obs_emb(obs_cond),
                action_value_tokens,
            ], dim=1)
        if self.cond_pos_emb is not None:
            memory = self.drop(memory + self.cond_pos_emb[:, :memory.shape[1], :])
        memory_mask, memory_key_padding_mask = self._build_memory_masks(
            batch_size=B,
            n_candidates=K_decode,
            n_context_actions=K_context,
            context_lengths=context_lengths,
            device=device,
            fold_slots=fold_slots,
        )
        if self.cond_encoder == 'gpt2':
            # Exactly OnlineSearchPolicy's call: one flat stream, causality from GPT-2's own
            # triangular mask, and the ONLY mask handed in is a padding mask.
            # `_build_encoder_mask` is deliberately not consulted on this path.
            #
            # Polarity flips at this boundary and silently means the opposite if missed:
            # torch's `src_key_padding_mask` is True == IGNORE, HF's `attention_mask` is
            # 1 == ATTEND.
            #
            # Positional embedding and embedding dropout both happen inside GPT2Model
            # (`wpe`, `embd_pdrop`), which is why neither cond_pos_emb nor self.drop was
            # applied above -- doing either here would double them.
            memory = self.encoder(
                inputs_embeds=memory,
                attention_mask=(~memory_key_padding_mask).to(memory.dtype),
            ).last_hidden_state
        elif self.cond_encoder == 'transformer':
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
        if fold_slots:
            # One row per (batch, slot); the tgt mask collapses to the plain causal one
            # because each row now holds a single slot's horizon.
            x = x.reshape(B * K_decode, self.horizon, -1)
            tgt_mask = self._build_tgt_mask(1, device)
        else:
            x = x.reshape(B, K_decode * self.horizon, -1)
            tgt_mask = self._build_tgt_mask(K_decode, device)
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        x = self.ln_f(x)
        x = self.head(x).reshape(B, K_decode, self.horizon, self.action_dim)
        if squeeze_candidate_dim:
            x = x[:, 0]
        return x


class DiffusionTransformerSearchPolicy(ObsCorruptionMixin, CropScopeMixin, SearchProcedureMixin, BaseImagePolicy):
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
            cond_encoder: str = 'transformer',
            corrupt_obs: bool = False,
            crop_shape=None,
            random_crop: bool = True,
            obs_noise_scheduler: DDPMScheduler = None,
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
        # `search_kwargs`, not `kwargs`: DiffusionUnetImagePolicy already owns `kwargs` for
        # its scheduler.step() keyword arguments, and PushTUNetSearchPolicy inherits both.
        # One unambiguous name means that class needs no aliases and no overrides.
        self.search_kwargs = kwargs
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
        self._init_selection(**kwargs)
        # How much gradient each candidate slot's loss term gets (_slot_weights) and which
        # norm that term uses (_slot_norm_alphas). Both default OFF -- uniform weights, MSE
        # everywhere -- which is bit-identical to the objective that predates them.
        self._init_slot_weighting(kwargs.get('slot_weight_decay', False),
                                  kwargs.get('slot_weights', None))
        self._init_slot_loss_norm(kwargs.get('slot_loss_norm', None))

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
            cond_encoder=cond_encoder,
            context_dim=self._context_dim(obs_feature_dim, **kwargs),
            context_decay=float(kwargs.get('context_decay', 1.0) or 1.0),
        )

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self._init_corruption(corrupt_obs, kwargs.get('corrupt_obs_eval'))
        # THE OBS CORRUPTION SCHEDULE. Separate from `noise_scheduler`, which denoises the
        # ACTION -- two independent processes, as in TMRL.
        #
        # It is what decides how corrupted the ladder's noisiest slot can be, and that is not
        # a free parameter of the shapes: sqrt(alpha_bar) at t = T-1 is the floor, and no
        # shape, cap or timestep can go below it. The legacy default below (T=100, beta
        # 0.001->0.02 linear) bottoms out at sqrt(alpha_bar) = 0.589, i.e. it still retains
        # 59% of the signal at maximum corruption -- so slot 0 could only ever be blurred,
        # never uninformative. It is kept as the DEFAULT so the maze and procgen arms that
        # share this class are bit-identical; the PushT config overrides it.
        #
        # The PushT arms pass TMRL's VLA schedule instead (T=1000, beta 1e-4 -> 0.02 linear,
        # tmrl_openpi/src/tmrl_openpi/models/cspi0.py:79-114), whose floor is
        # sqrt(alpha_bar) = 0.0064 -- 0.6% signal, i.e. essentially the marginal. That is the
        # continuum the method rests on, and the 10x finer grid also stops 16 slots from
        # rounding onto each other at the clean end.
        self.obs_noise_scheduler = obs_noise_scheduler or DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.001,
            beta_end=0.02,
            prediction_type="epsilon",
        )
        if corrupt_obs:
            print("Corrupting obs with a separate noise scheduler")

        # The per-slot corruption ladder. AFTER _init_corruption (it refuses to coexist with
        # corrupt_obs) and AFTER obs_noise_scheduler (linear_signal inverts its alpha_bars).
        self._init_slot_obs_noise(kwargs.get('slot_obs_noise', None))
        # Exactly one of these is non-None when the ladder is on: `slot_obs_t` for the fixed
        # shapes, `slot_obs_shape` for random_base, whose absolute levels do not exist until a
        # base is drawn and the shape is rescaled into [0, base]. The other stays a plain
        # `None` attribute, and it must be assigned in the ELSE branch rather than up front:
        # register_buffer refuses a name that already exists as an attribute.
        #
        # Buffers are registered ONLY when the ladder is on, so an arm that does not use it
        # has a byte-identical state_dict and every checkpoint that predates this still loads
        # under the default strict=True (see BaseWorkspace.load_payload).
        slot_obs_t = self._slot_obs_timesteps()
        slot_obs_shape = self._slot_obs_shape()
        if slot_obs_t is not None:
            self.register_buffer('slot_obs_t', slot_obs_t)
        else:
            self.slot_obs_t = None
        if slot_obs_shape is not None:
            self.register_buffer('slot_obs_shape', slot_obs_shape)
        else:
            self.slot_obs_shape = None
        if slot_obs_t is not None or slot_obs_shape is not None:
            # Per-dimension feature scale the corruption is measured against, so sqrt(abar)
            # is an SNR rather than an absolute magnitude (the corruption-calibration finding). A running estimate
            # rather than a per-batch one: rollout batches are small and B=1 would give a
            # degenerate std, and the corruption must not depend on batch shape -- the same
            # property set_sample_seeds guarantees for the sampling noise.
            self.register_buffer('obs_feature_std', torch.ones(obs_feature_dim))
            self.register_buffer('obs_feature_std_inited',
                                 torch.zeros((), dtype=torch.bool))
        # One corruption sample per DECISION, not per candidate (the per-candidate-redraw finding): the agent has
        # one observation, not max_actions of them. `_corrupt_t_base` is pinned the same way
        # and for the same reason -- under random_base the whole decision must sit at ONE
        # base level, or the slots stop being one observation seen at graded levels. Plain
        # attributes, like _sample_seeds: they hold no learnable state and a checkpoint
        # should not pin them.
        self._corrupt_eps = None
        self._corrupt_t_base = None
        self._corrupt_depth = 0

        self._init_crop(shape_meta, crop_shape, random_crop)

    # ---------------------------------------------------------------- config validation

    # Every key this class (or a subclass) legitimately reads out of **kwargs. Anything
    # else is a typo, and a typo'd config key used to be silently ignored -- which is how an
    # ablation arm ends up secretly identical to its sibling.
    _KNOWN_KWARGS = frozenset({
        'max_actions', 'search_context', 'selection', 'selection_temperature',
        'slot_weight_decay', 'slot_weights', 'slot_loss_norm', 'slot_obs_noise',
        'context_decay',
        'corrupt_obs_eval',
        'scheduler_step_kwargs',
        # PushT verifier
        'verifier_n_envs', 'verifier_legacy', 'verifier_use_async', 'verifier_steps',
        'verifier_value', 'render_size',
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

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

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

        SLICED TO To BEFORE NORMALIZING, not after. LinearNormalizer is elementwise
        (`_normalize` is `x * scale + offset` with per-key broadcast params), so the two
        orders give bit-identical output -- but normalizing first paid for the whole
        `horizon` window, of which the dataset yields 16 steps and this reads 2. It also
        makes the method safe against a dataset that emits only the observed frames (see
        PushTImageDataset.obs_image_steps): the discarded steps are never touched.
        """
        To = self.n_obs_steps
        if isinstance(obs_dict, dict):
            obs_dict = dict_apply(obs_dict, lambda x: x[:, :To, ...])
        else:
            obs_dict = obs_dict[:, :To, ...]
        return self._encode_obs(self.normalizer.normalize(obs_dict))

    # ------------------------------------------------- per-slot observation corruption

    @contextlib.contextmanager
    def _corrupt_scope(self):
        """Pin ONE corruption noise sample for the span of a decision. Reentrant.

        Mirrors ``_crop_scope``, and for the same reason: within a single decision every
        candidate must be looking at the SAME observation. The ladder grades how hard each
        slot has to look at it -- only the LEVEL varies by slot, never the sample.
        """
        self._corrupt_depth += 1
        try:
            yield
        finally:
            self._corrupt_depth -= 1
            if self._corrupt_depth == 0:
                self._corrupt_eps = None
                self._corrupt_t_base = None

    def _decision_noise(self, obs_features: torch.Tensor) -> torch.Tensor:
        """The corruption sample for this decision, drawn once and reused inside a scope."""
        eps = self._corrupt_eps
        if eps is None or eps.shape != obs_features.shape or eps.device != obs_features.device:
            # The feature-scale estimate is refreshed HERE, on the first corruption of a
            # decision, not on every call: the EMA moves on each update, so updating per
            # candidate would corrupt slot 0 and slot K-1 at slightly different scales and
            # the ladder would no longer be one observation seen at graded levels.
            self._update_obs_feature_std(obs_features)
            eps = torch.randn_like(obs_features)
            if self._corrupt_depth > 0:
                self._corrupt_eps = eps
        return eps

    @torch.no_grad()
    def _update_obs_feature_std(self, obs_features: torch.Tensor) -> None:
        """Track the per-dimension feature scale. Training only; seeded by the first batch."""
        if not self.training:
            return
        std = obs_features.reshape(-1, obs_features.shape[-1]).std(dim=0).clamp_min(1e-6)
        if not bool(self.obs_feature_std_inited):
            self.obs_feature_std.copy_(std)
            self.obs_feature_std_inited.fill_(True)
        else:
            self.obs_feature_std.mul_(0.99).add_(std, alpha=0.01)

    def corrupt_obs_features_slotwise(self, obs_features, slot=None):
        """Apply the per-slot ladder to encoded obs features.

        ``slot=None`` returns every slot at once, ``(B, To, D) -> (B, K, To, D)``, which is
        what the training forward needs. An int returns that one slot's view, ``(B, To, D)``,
        which is what a single ``predict_action`` call needs. Indices at or past K-1 clamp to
        the last slot: past ``max_actions`` the search runs a rolling window that pins every
        further generation at the widest context, so they belong at the ladder's clean end.

        Off (no ladder buffer registered) this is the identity, so an arm without the ladder
        is unaffected. The train/eval gate is ``corrupt_obs_eval``, exactly as for
        ``corrupt_obs_features`` -- with the caveat that evaluating clean means the slot ->
        level mapping the model trained under does not hold at rollout.
        """
        if self.slot_obs_t is None and self.slot_obs_shape is None:
            return obs_features
        if not self.training and not self.corrupt_obs_eval:
            return obs_features

        eps = self._decision_noise(obs_features) * self.obs_feature_std
        t = self._decision_slot_timesteps(obs_features)          # (B, K)
        abars = self.obs_noise_scheduler.alphas_cumprod.to(obs_features.device)
        if slot is None:
            ab = abars[t].to(obs_features.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,K,1,1)
            x = obs_features.unsqueeze(1)                                     # (B,1,To,D)
            return ab.sqrt() * x + (1.0 - ab).sqrt() * eps.unsqueeze(1)
        idx = min(int(slot), t.shape[1] - 1)
        ab = abars[t[:, idx]].to(obs_features.dtype).view(-1, 1, 1)           # (B,1,1)
        return ab.sqrt() * obs_features + (1.0 - ab).sqrt() * eps

    def _decision_slot_timesteps(self, obs_features: torch.Tensor) -> torch.Tensor:
        """The ``(B, K)`` timestep ladder in force for this decision.

        Under the fixed shapes every row is the `slot_obs_t` buffer -- the ladder does not
        depend on the sample. Under `random_base` row b is the `slot_obs_shape` profile
        rescaled into `[0, t_base[b]]`: slot 0 sits at a level drawn per sample and the
        schedule runs from there down to clean at slot K-1. Per sample, not per batch,
        because that is how the flat `corrupt_obs_features` has always drawn its timestep,
        and it keeps the level uncorrelated across a training batch.

        `t_base` is pinned by `_corrupt_scope` for the span of a decision, exactly as the
        noise sample is. Redrawing per candidate would put slot 0 and slot K-1 at unrelated
        bases, and the ladder would stop being one observation seen at graded levels.
        """
        B = obs_features.shape[0]
        if self.slot_obs_t is not None:
            return self.slot_obs_t.to(obs_features.device).unsqueeze(0).expand(B, -1)
        base = self._corrupt_t_base
        # Same staleness test as _decision_noise: a scope re-entered at a different batch
        # size or on another device must redraw rather than broadcast the wrong rows.
        if base is None or base.shape[0] != B or base.device != obs_features.device:
            lo, hi = self.slot_obs_base_range()
            base = torch.randint(lo, hi + 1, (B,), device=obs_features.device)
            if self._corrupt_depth > 0:
                self._corrupt_t_base = base
        return self.rescale_slot_timesteps(
            self.slot_obs_shape.to(obs_features.device), base)

    def encode_obs_cond(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.corrupt_obs_features(self._encode_obs_features(obs_dict))

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
        # _init_noise, not torch.randn: under eval it gives each EPISODE its own stream so
        # the sample does not depend on batch position (see set_sample_seeds). Falls back to
        # torch.randn with this generator when no seeds are set, which is the training path.
        trajectory = self._init_noise(
            (obs_cond.shape[0], self.horizon, self.action_dim),
            obs_cond.dtype, obs_cond.device, generator)

        # Re-derived only when it would differ. The timestep grid is a pure function of
        # num_inference_steps, but this runs once per CANDIDATE per decision -- 16 x ~38
        # times per episode at k=16 -- rebuilding the same small tensor each time. The
        # guard (rather than a one-shot init) keeps an eval harness that overrides
        # num_inference_steps on a loaded policy correct.
        if getattr(scheduler, 'num_inference_steps', None) != self.num_inference_steps:
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
            slot: Optional[int] = None,
        ) -> Dict[str, torch.Tensor]:
        """Standard ``BaseImagePolicy`` readout: a single (non-searched) action chunk.

        Returns ``{'action': (B, n_action_steps, Da), 'action_pred': (B, horizon, Da)}``,
        so every generic consumer (eval.py, the plain env runners, the workspace's
        sampling block) works against this policy exactly as against any other. The
        best-of-n search readout is ``predict_action_best``; the raw candidate generator
        is ``search_candidates``.

        ``actions``/``values`` optionally supply the search context this chunk is
        conditioned on (used by the search loop; ``None`` for a plain width-1 readout).
        ``actions`` arrive in RAW action units -- the search loop keeps them raw because the
        verifier simulates them -- and are normalized here, see _normalize_context_actions.
        ``obs_features``: optional pre-computed _encode_obs_features output, reused across
        the candidates of one search loop.

        ``slot``: which candidate slot this generation is, for the per-slot corruption
        ladder. ``None`` means "not laddered" and falls back to the flat ``corrupt_obs``
        path. Under the ladder the noise SAMPLE is shared across the decision (see
        ``_corrupt_scope``) and only the level varies by slot; under the flat path it is
        still drawn per call, because caching it there would make every candidate share one
        sample without any of the ladder's structure to justify it.
        """
        assert 'past_action' not in obs_dict
        if obs_features is None:
            obs_features = self._encode_obs_features(obs_dict)
        if slot is None:
            obs_cond = self.corrupt_obs_features(obs_features)
        else:
            obs_cond = self.corrupt_obs_features_slotwise(obs_features, slot)
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

    def predict_epsilon(self, batch, actions, values, noisy_trajectory, timesteps,
                        obs_features=None):
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
        noise per call, so including it would book corruption noise as policy drift. Pass
        the PRE-corruption features for the same reason.

        ``obs_features`` is the encode from the forward being analysed, handed in so BOTH
        snapshots share it. That is exact because the obs backbone is frozen -- the two
        snapshots hold the identical encoder, so re-encoding would produce the identical
        tensor twice. (It used to re-encode, on the grounds that "the vision backbone is
        part of what drifts and each snapshot must use its own". That stopped being true
        when the encoder was frozen.) Sharing also makes the comparison matched to the
        forward it is about: re-encoding happens under `eval()`, i.e. at the CENTRE crop,
        while the loss ran on a random training crop. ``None`` still re-encodes, for callers
        with no features to hand.
        """
        with torch.no_grad():
            if obs_features is None:
                obs_features = self._encode_obs_features(batch['obs'])
            return self.model(
                noisy_trajectory,
                timesteps,
                obs_cond=obs_features,
                actions=self._normalize_context_actions(actions),
                values=values,
            )

    def compute_loss(self, batch, actions=None, values=None, return_aux=False,
                     slot_weighting=True, obs_features=None):
        """Denoising loss for the expert action, conditioned on a best-of-n search context.

        ``actions``/``values`` optionally supply a PRE-GENERATED search context, in the
        form ``generate_search_context`` returns it. ``None`` (the default) generates it
        here from the current weights -- the offline path, which pays the full search cost
        on every gradient step. A trainer that amortizes the search across several updates
        passes the buffered context back in instead.

        ``obs_features`` optionally supplies the encoded observation, ``(B, To, D)``, for the
        same reason and from the same buffer: with a frozen backbone the encode is a pure
        function of (image, crop offset), so re-running it on every inner update is waste.
        Passing it also PINS THE CROP to whatever the buffer was filled with, which is what
        makes a buffered subgoal image and the observation it was predicted from share one
        crop.

        ``return_aux`` additionally returns the exact inputs the denoiser was called with,
        so a frozen snapshot of this policy can be re-run on identical inputs via
        ``predict_epsilon``; matched inputs are what make that comparison a KL.

        ``slot_weighting=False`` computes the CANONICAL objective whatever the config says
        -- uniform slot weights AND plain MSE. The validation loop passes it (see
        slot_weights.val) so val_loss remains comparable across arms and across a curriculum
        ramp instead of moving with the weighting.
        """
        with self._crop_scope(), self._corrupt_scope():
            return self._compute_loss(batch, actions, values, return_aux, slot_weighting,
                                      obs_features)

    def _compute_loss(self, batch, actions=None, values=None, return_aux=False,
                      slot_weighting=True, obs_features=None):
        target_actions = self.normalizer['action'].normalize(batch['action'])
        B, T, Da = target_actions.shape
        assert T == self.horizon, \
            f"Expected action horizon {self.horizon}, got {T}"
        assert Da == self.action_dim, \
            f"Expected action dim {self.action_dim}, got {Da}"

        # encode the obs ONCE: the same features back the grad-tracked training forward
        # and every candidate of the context search below (which only reads them, under
        # no_grad, so no graph is built for the search).
        #
        # `obs_features` may instead be supplied by a trainer that encoded this pool of
        # windows already (TrainSearchOuterInnerWorkspace). THAT IS ONLY SOUND BECAUSE THE
        # OBS BACKBONE IS FROZEN: with a trainable encoder, reusing features computed on an
        # earlier step would both feed the model stale activations and silently cut the
        # encoder out of the gradient. The caller asserts the freeze; this is the note that
        # says why it has to.
        if obs_features is None:
            obs_features = self._encode_obs_features(batch['obs'])
        # (B, To, D) flat, or (B, K, To, D) under the ladder -- one conditioning view per
        # candidate slot, same observation and same noise sample, graded level.
        if self.slot_obs_t is None:
            obs_cond = self.corrupt_obs_features(obs_features)
        else:
            obs_cond = self.corrupt_obs_features_slotwise(obs_features)

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

        # slot_weighting=False forces the canonical objective (uniform weights, plain MSE)
        # regardless of config. The validation loop uses it so val_loss stays a fixed
        # yardstick across arms and across a curriculum ramp -- see slot_weights.val.
        loss = self._slot_norm_loss(pred, target, slot_weighting)   # (B, K, H, Da)
        weights = self._slot_weights(loss.device, loss.dtype) if slot_weighting else None
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
            # PRE-corruption, and pre-slot-expansion: predict_epsilon omits the corruption
            # on purpose (it redraws per call, which would book corruption noise as drift).
            'obs_features': obs_features,
        }
