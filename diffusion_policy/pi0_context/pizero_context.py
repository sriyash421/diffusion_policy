"""open-pi-zero with RICL-style verifier feedback in the action expert's context.

Token layout (base pi0 is ``[img/text 276 | proprio 1 | action 4]``):

    [ img/text 276 | proprio 1 | block_1 (5) ... block_{Kc} (5) | action 4 ]      Kc = 4 -> 301

Each context block is one earlier candidate chunk and its verifier value, written in the
model's own input vocabulary (RICL technique 1): the 4 chunk actions go through the expert's
``action_encoder`` at flow time t=1 (a clean action), the value through ``feedback_encoder``
(Linear 1 -> hidden) plus a learned feedback type embedding; a learned slot embedding marks
the block index. Blocks are ordered best value first.

Mask (RICL technique 2): proprio and image/text are unchanged; a block attends to image/text,
proprio, earlier blocks and itself (causal across blocks, bidirectional within); the action
tokens attend to everything valid. Context columns in the action rows carry a learned scalar
bias (init -8) so the fresh model reproduces the base checkpoint, and with no context the
base is reproduced exactly (its inference path is taken unchanged). Missing blocks are masked
with ``finfo.min`` for every row. The loss is the base flow-matching loss on the query chunk
only (technique 4). Context tokens are routed through the action-expert mixture (tied, like
proprio in the official training) and cached per decision like proprio, so the 10 Euler steps
cost the same as the base.

Position ids: image/text, proprio (1) and action (2..5) unchanged; context 6..25.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn

VERITAS_ROOT = os.environ.get("VERITAS_ROOT", "/gscratch/robotics/harine/veritas")
if VERITAS_ROOT not in sys.path:
    sys.path.insert(0, VERITAS_ROOT)

from src.model.kv_cache import KVCache  # noqa: E402
from src.model.vla.pizero import PiZero  # noqa: E402

CONTEXT_SKIP_NAMES = ("vlm", "proprio", "context")   # their final-layer outputs are unused


class PiZeroContext(PiZero):
    def __init__(self, cfg, use_ddp: bool = False):
        super().__init__(cfg, use_ddp)
        self.num_context_blocks = int(cfg.get("num_context_blocks", 4))
        self.context_block_len = self.num_action_tokens + 1
        self.num_context_tokens = self.num_context_blocks * self.context_block_len
        self.total_num_tokens_ctx = self.total_num_tokens + self.num_context_tokens
        W = self.action_hidden_size
        self.feedback_encoder = nn.Linear(1, W)
        self.feedback_type_emb = nn.Parameter(torch.zeros(W))
        self.ctx_slot_emb = nn.Embedding(self.num_context_blocks, W)
        self.ctx_attn_bias = nn.Parameter(torch.tensor(float(cfg.get("context_attn_bias_init", -8.0))))
        nn.init.normal_(self.feedback_type_emb, std=0.02)
        nn.init.normal_(self.ctx_slot_emb.weight, std=0.02)
        self.tie_context_weights()

    # ------------------------------------------------------------------ setup
    def tie_context_weights(self):
        """Context (and proprio, as in the official training) run through the action expert.
        Call again after loading a checkpoint."""
        mixtures = self.joint_model.mixtures
        mixtures["proprio"] = mixtures["action"]
        mixtures["context"] = mixtures["action"]
        self.joint_model.cache_names = ["vlm", "proprio", "context"]

    def context_parameters(self):
        """Everything trainable: expert LoRA (if built with mixture.action.use_lora=True) and
        the context encoder/bias. The checkpoint weights stay frozen."""
        ps = [p for n, p in self.joint_model.mixtures["action"].named_parameters() if "lora_" in n]
        ps += [self.feedback_type_emb, self.ctx_attn_bias, *self.feedback_encoder.parameters(),
               *self.ctx_slot_emb.parameters()]
        return ps

    def freeze_for_context_training(self):
        self.freeze_all_weights()
        for p in self.context_parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------ embeddings
    def embed_context(self, ctx_actions: torch.Tensor, ctx_values: torch.Tensor) -> torch.Tensor:
        """ctx_actions (B, Kc, H, A) model-space chunks, ctx_values (B, Kc) normalised verifier
        values -> (B, Kc*(H+1), W). Blocks beyond a sample's n_ctx are masked later, so their
        content is irrelevant (pass zeros)."""
        B, Kc, H, A = ctx_actions.shape
        t1 = torch.ones(B * Kc, device=ctx_actions.device, dtype=ctx_actions.dtype)
        time_cond = self.time_embedding(t1)
        if self.action_expert_adaptive_mode:
            a = self.action_encoder(ctx_actions.reshape(B * Kc, H, A))
        else:
            a = self.action_encoder(ctx_actions.reshape(B * Kc, H, A), time_cond)
        a = a.view(B, Kc, H, -1)
        f = self.feedback_encoder(ctx_values.to(ctx_actions.dtype)[..., None]) + self.feedback_type_emb
        blocks = torch.cat([a, f[:, :, None, :]], dim=2)                     # (B, Kc, H+1, W)
        blocks = blocks + self.ctx_slot_emb.weight[None, :, None, :]
        return blocks.reshape(B, Kc * (H + 1), -1)

    # ------------------------------------------------------------------ masks
    def build_context_mask_and_position_ids(self, attention_mask: torch.Tensor, n_ctx: torch.Tensor,
                                            dtype: torch.dtype):
        """Full (B, 1, T, T) mask over [img/text | proprio | context | action] with the learned
        bias on the action rows' context columns, plus position ids per mixture."""
        bsz = attention_mask.size(0)
        dev = attention_mask.device
        P = self.max_image_text_tokens
        C0 = P + self.num_proprio_tokens
        A0 = C0 + self.num_context_tokens
        T = self.total_num_tokens_ctx
        L = self.context_block_len
        neg = torch.finfo(dtype).min
        mask = torch.full((bsz, T, T), neg, dtype=dtype, device=dev)
        cnts = attention_mask.sum(dim=1)
        for idx, cnt in enumerate(cnts.tolist()):
            mask[idx, :cnt, :cnt] = 0            # image/text among itself
            mask[idx, P:, :cnt] = 0              # proprio, context, action -> image/text
        mask[:, P:C0, P:C0] = 0                  # proprio -> itself
        for j in range(self.num_context_blocks):  # block j -> proprio, blocks <= j
            r = slice(C0 + j * L, C0 + (j + 1) * L)
            mask[:, r, P:C0] = 0
            mask[:, r, C0:C0 + (j + 1) * L] = 0
        mask[:, A0:, P:C0] = 0                   # action -> proprio, context (biased), itself
        mask[:, A0:, C0:A0] = 0
        mask[:, A0:, A0:] = 0
        bias_map = torch.zeros_like(mask)
        bias_map[:, A0:, C0:A0] = 1
        mask = mask + bias_map * self.ctx_attn_bias.to(dtype)
        n_ctx = torch.as_tensor(n_ctx, device=dev).view(bsz, 1)
        valid_cols = (torch.arange(self.num_context_blocks, device=dev)[None, :] < n_ctx)
        valid_cols = valid_cols.repeat_interleave(L, dim=1)                    # (B, Kc*L)
        mask[:, :, C0:A0] = torch.where(valid_cols[:, None, :], mask[:, :, C0:A0], torch.full_like(mask[:, :, C0:A0], neg))
        mask = mask.unsqueeze(1)
        vlm_pos = torch.arange(1, P + 1, device=dev).repeat(bsz, 1)
        prop_pos = torch.arange(1, self.num_proprio_tokens + 1, device=dev).repeat(bsz, 1)
        act_pos = torch.arange(self.num_proprio_tokens + 1,
                               self.num_proprio_tokens + self.num_action_tokens + 1, device=dev).repeat(bsz, 1)
        ctx_pos = torch.arange(self.num_proprio_tokens + self.num_action_tokens + 1,
                               self.num_proprio_tokens + self.num_action_tokens + 1 + self.num_context_tokens,
                               device=dev).repeat(bsz, 1)
        return mask, vlm_pos, prop_pos, ctx_pos, act_pos

    def split_context_mask(self, mask: torch.Tensor):
        """(prefix [.., 277, 277], context rows [.., 20, 297], action rows [.., 4, 301])."""
        P = self.max_image_text_tokens
        C0 = P + self.num_proprio_tokens
        A0 = C0 + self.num_context_tokens
        return mask[..., :C0, :C0], mask[..., C0:A0, :A0], mask[..., A0:, :]

    # ------------------------------------------------------------------ inference
    def infer_action_with_context(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        proprios: torch.FloatTensor,
        ctx_actions: Optional[torch.Tensor] = None,
        ctx_values: Optional[torch.Tensor] = None,
        n_ctx: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        prefix_caches: Optional[Dict[str, KVCache]] = None,
    ) -> torch.FloatTensor:
        """One chunk per sample. With n_ctx all zero (or no context given) this is exactly the
        base ``infer_action``; otherwise the context KV is cached once beside VLM and proprio.
        ``prefix_caches`` (from :meth:`prefix_caches`) skips the image/text/proprio pass: the
        sequential modes reuse it for every candidate of a decision (the context path is then
        taken even with n_ctx = 0, its masked columns leave the base output unchanged)."""
        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)
        if n_ctx is None:
            n_ctx = torch.zeros(bsz, dtype=torch.long, device=device)
        n_ctx = torch.as_tensor(n_ctx, device=device)
        use_ctx = bool((n_ctx > 0).any())
        if ctx_actions is None:
            ctx_actions = torch.zeros(bsz, self.num_context_blocks, self.horizon_steps, self.action_dim,
                                      device=device, dtype=dtype)
            ctx_values = torch.zeros(bsz, self.num_context_blocks, device=device, dtype=dtype)
        if not use_ctx and prefix_caches is None:
            # the base path with only vlm/proprio caches (the official infer_action would also
            # build the context cache from cache_names and then try to read it)
            base_mask, vlm_pos, prop_pos, act_pos = self.build_causal_mask_and_position_ids(
                attention_mask.cpu(), dtype)
            itp, am = self.split_full_mask_into_submasks(base_mask)
            return self._infer_action_noise(input_ids, pixel_values, itp.to(device), am.to(device),
                                            vlm_pos.to(device), prop_pos.to(device), act_pos.to(device),
                                            proprios, noise)

        mask, vlm_pos, prop_pos, ctx_pos, act_pos = self.build_context_mask_and_position_ids(
            attention_mask, n_ctx, dtype)
        prefix_mask, ctx_mask, action_mask = self.split_context_mask(mask)
        if prefix_caches is None:
            kv_caches = self._prefix_caches(input_ids, pixel_values, proprios, prefix_mask, vlm_pos, prop_pos)
        else:
            kv_caches = dict(prefix_caches)   # the shared vlm/proprio caches are only read below
        kv_caches["context"] = KVCache()     # after proprio: KV column order vlm, proprio, context
        ctx_embeds = self.embed_context(ctx_actions.to(dtype), ctx_values.to(dtype))
        self.joint_model(
            attention_mask=ctx_mask,
            position_ids_all={"context": ctx_pos},
            embeds_all={"context": ctx_embeds},
            kv_caches=kv_caches, cache_mode="append_non_active",
            final_layer_post_attn_skip_names=CONTEXT_SKIP_NAMES)
        return self._euler(kv_caches, action_mask, act_pos, bsz, device, dtype, noise)

    def _prefix_caches(self, input_ids, pixel_values, proprios, prefix_mask, vlm_pos, prop_pos):
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)
        proprio_embeds = self.proprio_encoder(proprios)
        kv_caches = {"vlm": KVCache(), "proprio": KVCache()}
        _, kv_caches = self.joint_model(
            attention_mask=prefix_mask,
            position_ids_all={"vlm": vlm_pos, "proprio": prop_pos},
            embeds_all={"vlm": inputs_embeds, "proprio": proprio_embeds},
            kv_caches=kv_caches, return_caches=True)
        return kv_caches

    def prefix_caches(self, input_ids, pixel_values, attention_mask, proprios) -> Dict[str, KVCache]:
        """VLM + proprio KV of a decision, computed once and passed to every
        :meth:`infer_action_with_context` call of that decision (the prefix rows never see the
        context, so the caches are the same for every n_ctx)."""
        dtype = pixel_values.dtype
        n0 = torch.zeros(pixel_values.size(0), dtype=torch.long, device=pixel_values.device)
        mask, vlm_pos, prop_pos, _, _ = self.build_context_mask_and_position_ids(attention_mask, n0, dtype)
        prefix_mask = self.split_context_mask(mask)[0]
        return self._prefix_caches(input_ids, pixel_values, proprios, prefix_mask, vlm_pos, prop_pos)

    def _euler(self, kv_caches, action_mask, act_pos, bsz, device, dtype, noise=None):
        action = torch.randn((bsz, self.horizon_steps, self.action_dim), device=device,
                             dtype=dtype) if noise is None else noise.clone()
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            time_cond = self.time_embedding(t)
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            action_embeds = self.joint_model(
                attention_mask=action_mask, position_ids_all={"action": act_pos},
                embeds_all={"action": action_embeds}, time_cond=time_cond,
                kv_caches=kv_caches, cache_mode="append_non_active")["action"]
            action += delta_t * self.action_decoder(action_embeds)
            t += delta_t
        if self.final_action_clip_value is not None:
            action = torch.clamp(action, -self.final_action_clip_value, self.final_action_clip_value)
        return action

    def _infer_action_noise(self, input_ids, pixel_values, image_text_proprio_mask, action_mask,
                            vlm_position_ids, proprio_position_ids, action_position_ids, proprios,
                            noise=None):
        """Base inference path; ``noise`` fixes the initial sample (paired comparisons)."""
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)
        proprio_embeds = self.proprio_encoder(proprios)
        kv_caches = {"vlm": KVCache(), "proprio": KVCache()}
        _, kv_caches = self.joint_model(
            attention_mask=image_text_proprio_mask,
            position_ids_all={"vlm": vlm_position_ids, "proprio": proprio_position_ids},
            embeds_all={"vlm": inputs_embeds, "proprio": proprio_embeds},
            kv_caches=kv_caches, return_caches=True)
        return self._euler(kv_caches, action_mask, action_position_ids, pixel_values.size(0),
                           pixel_values.device, pixel_values.dtype, noise)

    # ------------------------------------------------------------------ training
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        proprios: torch.FloatTensor,
        actions: torch.FloatTensor,
        t: torch.FloatTensor,
        ctx_actions: torch.Tensor,
        ctx_values: torch.Tensor,
        n_ctx: torch.Tensor,
        reduce: bool = True,
    ) -> torch.FloatTensor:
        """Flow-matching loss on the query chunk with n_ctx[b] context blocks visible
        (scalar, or per-sample (B,) with reduce=False)."""
        dtype = pixel_values.dtype
        x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)
        mask, vlm_pos, prop_pos, ctx_pos, act_pos = self.build_context_mask_and_position_ids(
            attention_mask, n_ctx, dtype)
        # the official embedding index-puts image features into the token embeddings, which
        # needs one dtype: run it in the model's dtype even under bf16 autocast
        with torch.autocast("cuda", enabled=False):
            inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)
        proprio_embeds = self.proprio_encoder(proprios)
        time_cond = self.time_embedding(t)
        if self.action_expert_adaptive_mode:
            action_embeds = self.action_encoder(psi_t)
        else:
            action_embeds = self.action_encoder(psi_t, time_cond)
        ctx_embeds = self.embed_context(ctx_actions.to(dtype), ctx_values.to(dtype))
        action_embeds = self.joint_model(
            attention_mask=mask,
            position_ids_all={"vlm": vlm_pos, "proprio": prop_pos, "context": ctx_pos, "action": act_pos},
            embeds_all={"vlm": inputs_embeds, "proprio": proprio_embeds, "context": ctx_embeds,
                        "action": action_embeds},
            time_cond=time_cond, kv_caches={},
            final_layer_post_attn_skip_names=CONTEXT_SKIP_NAMES)["action"]
        v_psi = self.action_decoder(action_embeds)
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        per_sample = ((v_psi - d_psi) ** 2).mean(dim=(1, 2))
        return per_sample.mean() if reduce else per_sample


def build_pizero_context(cfg, checkpoint_path: str, lora: bool = True, device="cuda",
                         dtype=torch.float32):
    """PiZeroContext from the official eval config + checkpoint. ``lora=True`` builds the
    action expert with LoRA layers (cfg.lora_r / cfg.lora_dropout); the base weights load
    strictly, only the LoRA and context parameters are new."""
    from omegaconf import open_dict
    from diffusion_policy.pi0_context.official import checkpoint_state_dict
    with open_dict(cfg):
        # cfg is resolved: joint.config.mixture is a copy of mixture, set both
        cfg.mixture.action.use_lora = bool(lora)
        cfg.joint.config.mixture.action.use_lora = bool(lora)
    model = PiZeroContext(cfg, use_ddp=False)
    sd = checkpoint_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    new_ok = ("feedback_encoder", "feedback_type_emb", "ctx_slot_emb", "ctx_attn_bias")
    bad = [k for k in missing if not (k.startswith(new_ok) or "lora_" in k
                                      or k.startswith("joint_model.mixtures.context."))]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {bad[:5]} unexpected {unexpected[:5]}")
    model.tie_context_weights()
    model.freeze_for_context_training()
    return model.to(dtype).to(device)
