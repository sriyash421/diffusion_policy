#!/usr/bin/env python
"""GPU checks for PiZeroContext against the official PiZeroInference (same checkpoint).

1. n_ctx = 0 takes the base path: output equals the base model bit for bit (same noise).
2. n_ctx = 0 forced through the context path (masked columns): allclose to the base.
3. n_ctx > 0 at init (bias -8): runs, close to the base but not identical.
4. Training forward for n_ctx in 0..Kc: finite loss, gradients only on LoRA + context params.
5. Mask: every row attends to >= 1 column; context columns of missing blocks are masked.
Frames: real replay frames if --replay-dir has any, else random uint8 images.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

DP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, DP_ROOT)
from diffusion_policy.pi0_context.official import (compose_eval_cfg, frame_inputs, load_pizero,  # noqa: E402
                                                   make_adapter)
from diffusion_policy.pi0_context.pizero_context import PiZeroContext, build_pizero_context  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", default="/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2/eggplant/replay")
    ap.add_argument("--checkpoint", default="models/bridge_uniform_step19296_2024-12-26_22-31_42.pt")
    ap.add_argument("--n-frames", type=int, default=3)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cfg = compose_eval_cfg()
    adapter = make_adapter(cfg)
    files = sorted(glob.glob(os.path.join(args.replay_dir, "ep*.npz")))
    if files:
        rp = np.load(files[0])
        idx = np.linspace(0, len(rp["image"]) - 1, args.n_frames).astype(int)
        images, proprio = rp["image"][idx], rp["proprio"][idx]
        print(f"frames from {files[0]} rows {idx.tolist()}")
    else:
        rng = np.random.default_rng(0)
        images = rng.integers(0, 255, (args.n_frames, 224, 224, 3), dtype=np.uint8)
        proprio = rng.normal(size=(args.n_frames, 7)).astype(np.float32)
        print("no replay frames yet: random images")
    inp = frame_inputs(adapter, images, proprio, "put eggplant into yellow basket")
    B = len(images)

    base, _ = load_pizero(cfg, args.checkpoint)
    ctx_model = build_pizero_context(cfg, args.checkpoint, lora=True)
    ctx_model.eval()
    n_train = sum(p.numel() for p in ctx_model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.2f}M of {sum(p.numel() for p in ctx_model.parameters()) / 1e9:.2f}B")

    noise = torch.randn(B, 4, 7, device="cuda")
    Kc = ctx_model.num_context_blocks
    ctx_actions = torch.rand(B, Kc, 4, 7, device="cuda") * 2 - 1
    ctx_values = torch.randn(B, Kc, device="cuda")

    # 1. base path
    from diffusion_policy.pi0_context.official import model_masks
    masks = model_masks(base, inp["attention_mask"])
    # PiZeroInference draws its noise inside infer_action: seed, run it, re-seed and draw the
    # same noise for the context model's base path.
    with torch.inference_mode():
        torch.manual_seed(123)
        out_ref = base.infer_action(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                                    proprios=inp["proprios"], **masks)
        torch.manual_seed(123)
        seeded_noise = torch.randn((B, 4, 7), device="cuda")
        out0 = ctx_model.infer_action_with_context(inp["input_ids"], inp["pixel_values"], inp["attention_mask"],
                                                   inp["proprios"], noise=seeded_noise)
    d1 = (out0 - out_ref).abs().max().item()
    print(f"[1] n_ctx=0 base path vs PiZeroInference: max|diff| = {d1:.3e}  {'OK' if d1 == 0 else 'NOT bit-identical'}")

    # 2. n_ctx = 0 through the context path
    with torch.inference_mode():
        mask, vlm_pos, prop_pos, ctx_pos, act_pos = ctx_model.build_context_mask_and_position_ids(
            inp["attention_mask"], torch.zeros(B, dtype=torch.long, device="cuda"), torch.float32)
        prefix_mask, ctx_mask, action_mask = ctx_model.split_context_mask(mask)
        from src.model.kv_cache import KVCache
        emb = ctx_model._forward_siglip_and_text_embedding(inp["input_ids"], inp["pixel_values"])
        kv = {"vlm": KVCache(), "proprio": KVCache()}
        _, kv = ctx_model.joint_model(attention_mask=prefix_mask, position_ids_all={"vlm": vlm_pos, "proprio": prop_pos},
                                      embeds_all={"vlm": emb, "proprio": ctx_model.proprio_encoder(inp["proprios"])},
                                      kv_caches=kv, return_caches=True)
        kv["context"] = KVCache()
        ctx_model.joint_model(attention_mask=ctx_mask, position_ids_all={"context": ctx_pos},
                              embeds_all={"context": ctx_model.embed_context(ctx_actions, ctx_values)},
                              kv_caches=kv, cache_mode="append_non_active",
                              final_layer_post_attn_skip_names=("vlm", "proprio", "context"))
        out0_ctx = ctx_model._euler(kv, action_mask, act_pos, B, torch.device("cuda"), torch.float32, seeded_noise)
    d2 = (out0_ctx - out_ref).abs().max().item()
    print(f"[2] n_ctx=0 via masked context path: max|diff| = {d2:.3e}  {'OK' if d2 < 1e-4 else 'FAIL'}")

    # 3. n_ctx > 0 at init
    with torch.inference_mode():
        n_ctx = torch.tensor([Kc] + [1] * (B - 1), device="cuda")[:B]
        out_k = ctx_model.infer_action_with_context(inp["input_ids"], inp["pixel_values"], inp["attention_mask"],
                                                    inp["proprios"], ctx_actions, ctx_values, n_ctx, noise=seeded_noise)
    d3 = (out_k - out_ref).abs().max().item()
    print(f"[3] n_ctx={n_ctx.tolist()} at init (bias {ctx_model.ctx_attn_bias.item():.1f}): max|diff| vs base = {d3:.3e} "
          f"{'OK' if 0 < d3 < 5e-2 else 'CHECK'}")

    # 4. training forward + grads
    ctx_model.train()
    actions = torch.rand(B, 4, 7, device="cuda") * 2 - 1
    t = torch.rand(B, device="cuda")
    for k in range(Kc + 1):
        n_ctx = torch.full((B,), k, device="cuda")
        loss = ctx_model(inp["input_ids"], inp["pixel_values"], inp["attention_mask"], inp["proprios"],
                         actions, t, ctx_actions, ctx_values, n_ctx)
        ctx_model.zero_grad(set_to_none=True)
        loss.backward()
        grads = [n for n, p in ctx_model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
        lora = sum("lora_" in n for n in grads)
        extra = [n for n in grads if "lora_" not in n]
        print(f"[4] n_ctx={k}: loss {loss.item():.4f}, grads on {lora} lora tensors + {extra}")
    ctx_model.eval()

    # 5. mask sanity
    mask, *_ = ctx_model.build_context_mask_and_position_ids(
        inp["attention_mask"], torch.tensor([2] * B, device="cuda"), torch.float32)
    m = mask[:, 0]
    neg = torch.finfo(torch.float32).min
    # image/text pad rows are fully masked in the base model too; check the rest
    cnt = int(inp["attention_mask"].sum(1).min())
    rows_ok = bool(((m[:, :cnt] > neg / 2).sum(-1) >= 1).all()) and \
        bool(((m[:, ctx_model.max_image_text_tokens:] > neg / 2).sum(-1) >= 1).all())
    C0 = ctx_model.max_image_text_tokens + ctx_model.num_proprio_tokens
    L = ctx_model.context_block_len
    blocked = bool((m[:, :, C0 + 2 * L:C0 + Kc * L] <= neg / 2).all())
    visible = bool((m[:, -1, C0:C0 + 2 * L] > neg / 2).all())
    print(f"[5] mask: rows have >=1 visible column: {rows_ok}; blocks >= n_ctx masked: {blocked}; "
          f"valid blocks visible to action: {visible}")
    # 6. shared prefix caches (sequential modes) reproduce the per-call path
    with torch.inference_mode():
        prefix = ctx_model.prefix_caches(inp["input_ids"], inp["pixel_values"], inp["attention_mask"], inp["proprios"])
        n2 = torch.full((B,), 2, device="cuda")
        out_a = ctx_model.infer_action_with_context(inp["input_ids"], inp["pixel_values"], inp["attention_mask"],
                                                    inp["proprios"], ctx_actions, ctx_values, n2, noise=seeded_noise)
        out_b = ctx_model.infer_action_with_context(inp["input_ids"], inp["pixel_values"], inp["attention_mask"],
                                                    inp["proprios"], ctx_actions, ctx_values, n2, noise=seeded_noise,
                                                    prefix_caches=prefix)
        out_c = ctx_model.infer_action_with_context(inp["input_ids"], inp["pixel_values"], inp["attention_mask"],
                                                    inp["proprios"], noise=seeded_noise, prefix_caches=prefix)
        out_d = ctx_model.infer_action_with_context(inp["input_ids"], inp["pixel_values"], inp["attention_mask"],
                                                    inp["proprios"], ctx_actions, ctx_values, n2, noise=seeded_noise,
                                                    prefix_caches=prefix)
    d6a = (out_a - out_b).abs().max().item(); d6c = (out_c - out_ref).abs().max().item(); d6d = (out_b - out_d).abs().max().item()
    print(f"[6] prefix caches: n_ctx=2 vs per-call {d6a:.3e} {'OK' if d6a < 1e-4 else 'FAIL'}; "
          f"n_ctx=0 vs base {d6c:.3e} {'OK' if d6c < 1e-4 else 'FAIL'}; reuse unchanged {d6d:.3e} {'OK' if d6d == 0 else 'FAIL'}")
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
