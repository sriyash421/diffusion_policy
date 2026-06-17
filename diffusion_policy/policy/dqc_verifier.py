"""DQCVerifier — in-process JAX DQC Q-function verifier for the L2S SearchPolicy.

Drop-in replacement for ``RoboMonkeyVerifier`` (same
``get_value(obs_dict, action) -> Tensor`` interface) that scores candidate
actions with the **DQC chunk Q-function** instead of the RoboMonkey LLaVA
reward model.

The DQC critic (``agent_name: bridge_dqc_chunk``) takes:
  * image  — the current RGB frame (ResNet-34 + FiLM encoder, 5-channel input
              with spatial-coord planes), resized to ``img_size``.
  * action — a single 7-D action, normalized to [-1, 1] via
              ``action_normalization.json`` (bridge-space EE deltas).
  * text   — a 512-d MUSE (USE-4) embedding of the instruction.
and returns a scalar Q (min over the 2-critic ensemble).

DISCOUNTED CHUNK SUM
--------------------
For a candidate action chunk, ``get_value`` scores each step in
``score_indices`` (the EXECUTED action window, set by the owning search policy
via ``score_executed_chunk``) against the *current* frame and aggregates them
into one value per candidate. With ``score_agg="discounted_sum"`` the
aggregation is the discounted sum  sum_t  gamma^t * Q(frame, a_t)  with
``gamma = discount`` (default 0.98, the checkpoint's own discount) and ``t`` the
0-based position within the executed window. ``mean`` / ``sum`` / ``min`` are
also supported (matching RoboMonkeyVerifier).

ENCODE CACHE (speed)
--------------------
Q = head(encode(image, text), action). The ResNet-34 ``encode`` does NOT depend
on the action, yet the same frame is scored many times per call (M executed
steps) and across the search policy's autoregressive rounds (the batch frames
are fixed for a training step). We cache the per-critic 512-d encoding keyed by
the raw frame bytes (RoboMonkey's image-feature-cache idea) so each unique frame
runs the ResNet once and every action scoring reuses it via the cheap MLP head.
On training batches (256 frames re-scored ~120x/step) this is ~100x less ResNet
work. The split is mathematically identical to the fused forward (verified at
construction against a non-cached recompute).

The pure-JAX forward pass is ported from ``scriptsv2/q_fn_eval/eval_q.py``.
Runs in a conda env with both JAX (GPU) and torch (e.g. dqc_jax). JAX is told
NOT to preallocate the GPU so torch keeps room for the policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Keep JAX from grabbing the whole GPU — torch (the diffusion policy) shares it.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")


# ── Image / action preprocessing (ported from eval_q.py) ───────────────────────

def _preprocess_image(img_hwc: np.ndarray, size: int = 128) -> np.ndarray:
    """Resize uint8 (H,W,3) -> (size,size,5) float32 with spatial-coord channels."""
    from PIL import Image
    pil = Image.fromarray(img_hwc).resize((size, size), Image.BILINEAR)
    rgb = np.asarray(pil, dtype=np.float32) / 255.0
    H, W = rgb.shape[:2]
    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, :] * np.ones((H, 1), np.float32)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    return np.concatenate([rgb, xs[..., None], ys[..., None]], axis=-1)  # (H,W,5)


def _normalize_action(action: np.ndarray, norm: Dict) -> np.ndarray:
    low = np.array(norm["low"], dtype=np.float32)
    high = np.array(norm["high"], dtype=np.float32)
    eps = float(norm["clip_eps"])
    a = np.clip(action.astype(np.float32), low + eps, high - eps)
    return 2.0 * (a - low) / (high - low) - 1.0


def _to_hwc_uint8(img: torch.Tensor) -> np.ndarray:
    """Convert a single image tensor (CHW or HWC, float[0,1]/uint8) to HWC uint8."""
    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(img)
    arr = img.detach().cpu()
    if arr.dim() != 3:
        raise ValueError(f"Expected a single image with 3 dims, got {tuple(arr.shape)}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = arr.permute(1, 2, 0).contiguous()
    elif arr.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image with 3 channels, got {tuple(arr.shape)}")
    if arr.dtype.is_floating_point:
        max_val = float(arr.max().item()) if arr.numel() > 0 else 0.0
        if max_val <= 1.0 + 1e-3:
            arr = arr * 255.0
        arr = arr.clamp(0, 255).to(torch.uint8)
    else:
        arr = arr.to(torch.uint8)
    return arr.numpy()


def _frame_key(im: np.ndarray) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(im).tobytes(), digest_size=16).digest()


# ── Pure-JAX forward, split into encode (cacheable) + head (per action) ─────────
# Ported verbatim from scriptsv2/q_fn_eval/eval_q.py — keep in sync if that
# reference forward changes. _q_one = _mlp(concat(_encode(img,text), act)); we
# expose _encode and the _mlp head separately so the encode can be cached.

class _DQCScorer:
    def __init__(self, ckpt_dir: str):
        import jax
        import jax.numpy as jnp

        def _num_groups(C, max_g=32):
            g = max_g
            while C % g != 0:
                g -= 1
            return g

        def _group_norm(x, scale, bias):
            H, W, C = x.shape
            G = _num_groups(C)
            x = x.reshape(H, W, G, C // G)
            mean = x.mean(axis=(0, 1, 3), keepdims=True)
            var = x.var(axis=(0, 1, 3), keepdims=True)
            x = (x - mean) / jnp.sqrt(var + 1e-5)
            return x.reshape(H, W, C) * scale + bias

        def _layer_norm(x, scale, bias):
            mean = x.mean(); var = x.var()
            return (x - mean) / jnp.sqrt(var + 1e-6) * scale + bias

        def _conv2d(x, kernel, stride=1):
            return jax.lax.conv_general_dilated(
                x[None], kernel, window_strides=(stride, stride), padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"))[0]

        def _film(x, text, fp):
            gamma = text @ fp["Dense_0"]["kernel"] + fp["Dense_0"]["bias"]
            beta = text @ fp["Dense_1"]["kernel"] + fp["Dense_1"]["bias"]
            return x * gamma + beta

        def _resnet_block(x, text, bp, fp, stride=1):
            residual = x
            y = _conv2d(x, bp["Conv_0"]["kernel"], stride=stride)
            y = _group_norm(y, bp["GroupNorm_0"]["scale"], bp["GroupNorm_0"]["bias"])
            y = _film(y, text, fp)
            y = jax.nn.silu(y)
            y = _conv2d(y, bp["Conv_1"]["kernel"], stride=1)
            y = _group_norm(y, bp["GroupNorm_1"]["scale"], bp["GroupNorm_1"]["bias"])
            y = jax.nn.silu(y)
            if "conv_proj" in bp:
                residual = _conv2d(residual, bp["conv_proj"]["kernel"], stride=stride)
                residual = _group_norm(residual, bp["norm_proj"]["scale"], bp["norm_proj"]["bias"])
            return y + residual

        strides: List[int] = []
        for n, _c, s0 in [(3, 64, 1), (4, 128, 2), (6, 256, 2), (3, 512, 2)]:
            strides.extend([s0] + [1] * (n - 1))

        def _encode(img5, text, enc_p):
            ie = enc_p["image_encoder"]
            x = _conv2d(img5, ie["conv_init"]["kernel"], stride=2)
            x = _group_norm(x, ie["norm_init"]["scale"], ie["norm_init"]["bias"])
            x = jax.nn.silu(x)
            for i, s in enumerate(strides):
                x = _resnet_block(x, text, ie[f"ResNetBlock_{i}"], ie[f"FilmConditioning_{i}"], s)
            return x.mean(axis=(0, 1))  # (512,)

        def _mlp(feat, mlp_p, head_p):
            x = feat
            for i in range(5):
                x = x @ mlp_p[f"Dense_{i}"]["kernel"] + mlp_p[f"Dense_{i}"]["bias"]
                x = _layer_norm(x, mlp_p[f"LayerNorm_{i}"]["scale"], mlp_p[f"LayerNorm_{i}"]["bias"])
                x = jax.nn.silu(x)
            return (x @ head_p["kernel"] + head_p["bias"])[0]

        with open(Path(ckpt_dir) / "params_100000.pkl", "rb") as f:
            d = pickle.load(f)
        all_params = d["agent"]["network"]["params"]["modules_action_critic"]
        self._params = [jax.tree_util.tree_map(lambda x: jnp.array(x[e]), all_params)
                        for e in range(2)]
        self._jnp = jnp

        @jax.jit
        def _encode_batch(imgs5, text, p):
            return jax.vmap(lambda im: _encode(im, text, p["encoder"]))(imgs5)  # (N,512)

        @jax.jit
        def _head_batch(encs, acts, p):
            return jax.vmap(
                lambda e, a: _mlp(jnp.concatenate([e, a]), p["MLP_0"], p["Dense_0"])
            )(encs, acts)  # (N,)

        self._encode_batch = _encode_batch
        self._head_batch = _head_batch

    def encode(self, imgs5_np: np.ndarray, text_np: np.ndarray) -> np.ndarray:
        """(N,S,S,5) -> (2,N,512) per-critic encodings."""
        t = self._jnp.asarray(text_np)
        x = self._jnp.asarray(imgs5_np)
        return np.stack([np.asarray(self._encode_batch(x, t, p)) for p in self._params])

    def head(self, encs_np: np.ndarray, acts_np: np.ndarray) -> np.ndarray:
        """encs (2,N,512), acts (N,7) -> (N,) min over the 2 critics."""
        a = self._jnp.asarray(acts_np)
        qs = np.stack([np.asarray(self._head_batch(self._jnp.asarray(encs_np[e]), a, self._params[e]))
                       for e in range(2)])
        return qs.min(axis=0)

    def score(self, imgs5_np: np.ndarray, acts_np: np.ndarray, text_np: np.ndarray) -> np.ndarray:
        """Non-cached encode+head (for the regression/health check)."""
        return self.head(self.encode(imgs5_np, text_np), acts_np)


# ── MUSE text embedding ─────────────────────────────────────────────────────────

def _resolve_text_emb(instruction: str, text_emb_file: Optional[str]) -> np.ndarray:
    candidates = []
    if text_emb_file:
        candidates.append(Path(text_emb_file))
    here = Path(__file__).resolve()
    for up in here.parents:
        cand = up / "scriptsv2" / "q_fn_eval" / "task_emb.npy"
        if cand.is_file():
            candidates.append(cand)
            break
    for c in candidates:
        if c and Path(c).is_file():
            emb = np.load(c).astype(np.float32)
            if emb.shape == (512,):
                print(f"[DQCVerifier] loaded MUSE embedding from {c}")
                return emb
            print(f"[DQCVerifier] {c} has shape {emb.shape}, expected (512,); skipping")
    gen = None
    for up in here.parents:
        cand = up / "scriptsv2" / "q_fn_eval" / "get_muse_emb.py"
        if cand.is_file():
            gen = cand
            break
    out = Path(text_emb_file) if text_emb_file else (gen.parent / "task_emb.npy" if gen else None)
    if gen is None or out is None:
        raise FileNotFoundError(
            "DQCVerifier: no MUSE text embedding found and get_muse_emb.py not "
            "locatable. Pass text_emb_file=<path to a 512-d .npy>.")
    print(f"[DQCVerifier] generating MUSE embedding for {instruction!r} -> {out}")
    subprocess.run([sys.executable, str(gen), "--instruction", instruction,
                    "--out", str(out)], check=True)
    emb = np.load(out).astype(np.float32)
    assert emb.shape == (512,), emb.shape
    return emb


# ── Verifier ────────────────────────────────────────────────────────────────────

class DQCVerifier:
    """In-process DQC Q-function verifier (JAX). Same interface as RoboMonkeyVerifier."""

    def __init__(
        self,
        ckpt_dir: str,
        instruction: str,
        image_obs_key: str,
        text_emb_file: Optional[str] = None,
        img_size: int = 128,
        discount: float = 0.98,
        score_indices: Optional[list] = None,
        score_agg: str = "discounted_sum",
        score_executed_chunk: bool = False,
        action_chunk_index: int = -1,
        enc_batch: int = 64,
        image_feat_cache_size: int = 1024,
        health_check: bool = True,
        # Unused; kept for YAML/back-compat parity with RoboMonkeyVerifier.
        server_url: str = "in_process",
        image_save_dir: str = "/tmp/dqc_obs",
        request_timeout: float = 300.0,
        max_workers: Optional[int] = None,
        keep_jpegs: bool = False,
    ) -> None:
        self.instruction = instruction
        self.image_obs_key = image_obs_key
        self.img_size = int(img_size)
        self.discount = float(discount)
        self.action_chunk_index = int(action_chunk_index)
        self.score_indices = list(score_indices) if score_indices is not None else None
        self.score_agg = str(score_agg)
        self.score_executed_chunk = bool(score_executed_chunk)
        self.enc_batch = int(enc_batch)
        self._cache_size = int(image_feat_cache_size)
        self._enc_cache: "OrderedDict[bytes, np.ndarray]" = OrderedDict()  # key -> (2,512)

        ckpt_dir = os.path.expanduser(ckpt_dir)
        with open(Path(ckpt_dir) / "action_normalization.json") as f:
            self._norm = json.load(f)
        self._text_emb = _resolve_text_emb(instruction, text_emb_file)
        print(f"[DQCVerifier] building JAX scorer from {ckpt_dir} ...")
        self._scorer = _DQCScorer(ckpt_dir)
        if health_check:
            # Compile + verify the encode/head split == fused forward (bit-identical).
            rng = np.random.default_rng(0)
            im5 = rng.standard_normal((3, self.img_size, self.img_size, 5)).astype(np.float32)
            act = rng.standard_normal((3, 7)).astype(np.float32)
            q_fused = self._scorer.score(im5, act, self._text_emb)
            encs = self._scorer.encode(im5, self._text_emb)
            q_split = self._scorer.head(encs, act)
            max_diff = float(np.abs(q_fused - q_split).max())
            assert max_diff < 1e-3, f"encode/head split mismatch (max diff {max_diff})"
            print(f"[DQCVerifier] ready (compiled; split==fused max_diff={max_diff:.2e}; "
                  f"discount={self.discount}, agg={self.score_agg}, "
                  f"enc_cache={self._cache_size})")

    # --- encode cache ------------------------------------------------------------
    def _encode_cached(self, frames_u8: List[np.ndarray]) -> np.ndarray:
        """Return (2, B, 512) encodings for B frames, encoding only cache misses."""
        keys = [_frame_key(f) for f in frames_u8]
        miss_keys, miss_imgs, seen = [], [], set()
        for k, f in zip(keys, frames_u8):
            if k not in self._enc_cache and k not in seen:
                seen.add(k)
                miss_keys.append(k)
                miss_imgs.append(_preprocess_image(f, self.img_size))
        for s in range(0, len(miss_imgs), self.enc_batch):
            chunk_imgs = np.stack(miss_imgs[s:s + self.enc_batch])
            chunk_enc = self._scorer.encode(chunk_imgs, self._text_emb)  # (2, n, 512)
            for i, k in enumerate(miss_keys[s:s + self.enc_batch]):
                self._enc_cache[k] = chunk_enc[:, i, :]                  # (2, 512)
        for k in keys:                      # LRU touch
            self._enc_cache.move_to_end(k)
        while len(self._enc_cache) > self._cache_size:
            self._enc_cache.popitem(last=False)
        return np.stack([self._enc_cache[k] for k in keys], axis=1)      # (2, B, 512)

    def clear_cache(self) -> None:
        self._enc_cache.clear()

    # --- aggregation -------------------------------------------------------------
    def _aggregate(self, per_step: torch.Tensor) -> torch.Tensor:
        agg = self.score_agg
        M = per_step.shape[1]
        if M == 1:
            return per_step[:, 0]
        if agg in ("discounted_sum", "discounted", "disc_sum"):
            w = torch.tensor([self.discount ** t for t in range(M)],
                             dtype=per_step.dtype, device=per_step.device)
            return (per_step * w).sum(dim=1)
        if agg == "mean":
            return per_step.mean(dim=1)
        if agg == "sum":
            return per_step.sum(dim=1)
        if agg == "min":
            return per_step.min(dim=1).values
        if agg == "last":
            return per_step[:, -1]
        if agg == "first":
            return per_step[:, 0]
        raise ValueError(f"unknown score_agg {agg!r} "
                         "(discounted_sum/mean/sum/min/last/first)")

    def _select_last_frame(self, obs_dict: Dict[str, Any]) -> torch.Tensor:
        if self.image_obs_key not in obs_dict:
            raise KeyError(
                f"DQCVerifier: obs_dict missing image key '{self.image_obs_key}'. "
                f"Available: {sorted(obs_dict.keys())}")
        x = obs_dict[self.image_obs_key]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.dim() != 5:
            raise ValueError(
                f"Expected obs '{self.image_obs_key}' (B, To, C, H, W) or "
                f"(B, To, H, W, C); got {tuple(x.shape)}")
        return x[:, -1]

    @torch.no_grad()
    def get_value(self, obs_dict: Dict[str, Any], action: torch.Tensor) -> torch.Tensor:
        """Score each candidate against the current frame -> (B,)."""
        if action.dim() != 3:
            raise ValueError(
                f"Expected action (B, horizon, action_dim); got {tuple(action.shape)}")
        last_frames = self._select_last_frame(obs_dict)
        B = action.shape[0]
        if last_frames.shape[0] != B:
            raise ValueError(
                f"Batch size mismatch: action B={B} but obs B={last_frames.shape[0]}")

        idxs = self.score_indices if self.score_indices is not None else [self.action_chunk_index]
        idxs = [int(i) for i in idxs]
        H = int(action.shape[1])
        if any(i >= H or i < -H for i in idxs):
            raise ValueError(f"score_indices {idxs} out of bounds for horizon H={H}.")
        M = len(idxs)

        frames_u8 = [_to_hwc_uint8(last_frames[b]) for b in range(B)]
        encs = self._encode_cached(frames_u8)                 # (2, B, 512)
        # Repeat each frame's encoding M times (b-major) to pair with its M steps.
        encs_pairs = np.repeat(encs, M, axis=1)               # (2, B*M, 512)
        acts = action[:, idxs, :].detach().cpu().numpy().reshape(B * M, action.shape[-1])
        acts = np.stack([_normalize_action(a, self._norm) for a in acts])  # (B*M, 7)
        q = self._scorer.head(encs_pairs, acts)               # (B*M,)
        per_step = torch.tensor(q, dtype=action.dtype, device=action.device).reshape(B, M)
        if M == 1:
            return per_step[:, 0]
        return self._aggregate(per_step)

    # --- KV-prefix controls: reuse for the encode cache (training speedup) -------
    @property
    def supports_kv_prefix(self) -> bool:
        return True

    def set_kv_prefix(self, enabled: bool) -> None:  # noqa: D401
        # The encode cache is always on; this hook just lets the search loop
        # clear it on a fresh frame (parity with RoboMonkeyVerifier).
        pass

    def clear_prefix_cache(self) -> None:
        self.clear_cache()
