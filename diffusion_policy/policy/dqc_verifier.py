"""DQCVerifier — in-process REAL DQC verifier for the L2S SearchPolicy.

Drop-in replacement for ``RoboMonkeyVerifier`` (same
``get_value(obs_dict, action) -> Tensor`` interface) that scores candidate
actions with the **real** ``BridgeDQCChunkAgent`` (flax) loaded from the DQC
checkpoint — NOT the earlier hand-rolled pure-JAX forward (``eval_q.py``), which
had several encoder bugs vs the trained model:

  * image norm  /127.5-1  (not /255)
  * GroupNorm   groups=4  (not 32)
  * a stem max-pool        (was missing)
  * FiLM applied AFTER each ResNet block as ``x*(1+gamma)+beta`` (not inside)
  * MLP head activation GELU (not SiLU); Dense->act->LayerNorm order
  * text embedding from multilingual USE-3 (not USE-4)

Using the real agent + its real text processor removes all of that guesswork —
this is the same scorer the offline comparison (``eval_dqc_critics.py``) uses,
so training and the verifier-vs-verifier eval are numerically identical.

CRITICS
-------
The DQC checkpoint holds two heads (same ResNet-34 + FiLM encoder):
  * ``critic_type="chunk"`` (DEFAULT) -> ``score_chunks``: scores the WHOLE
      backup_horizon-step chunk in one pass -> one Q(frame, a_{t:t+H}) per
      candidate. The BC policy predicts ``horizon`` (16) actions; the executed
      window (first ``n_action_steps``=8, set by the search policy via
      ``score_executed_chunk``) IS that chunk. NO inference-time discounting —
      the discount is baked into the chunk critic's training target.
  * ``critic_type="action"`` -> ``forward_action_critic``: per-step single-action
      Q over the executed window, aggregated (``score_agg``).

Chunks are assembled exactly as training did: each sub-action normalized to
[-1,1] via the Bridge action stats (``action_normalization.json``), flattened
time-major to (H*7,). (No NaN pad here — the BC policy always emits a full
16-action horizon, so all H executed steps are real.)

Runs in the ``dqc`` conda env (flax 0.12 / jax 0.10 + torch). JAX is told NOT to
preallocate the GPU so torch (the diffusion policy) keeps room. The MUSE text
embedding is precomputed (USE-3) and loaded from ``text_emb_file`` so TensorFlow
is NOT imported into the training process.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Keep JAX from grabbing the whole GPU — torch (the diffusion policy) shares it.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")

_REPO = Path(__file__).resolve().parents[3]  # /home/harine/RoboMonkey
_DQC_PKG = _REPO / "dqc"


# ── action normalization (Bridge EE deltas) ─────────────────────────────────────

def _normalize_action(action: np.ndarray, norm: Dict) -> np.ndarray:
    low = np.array(norm["low"], dtype=np.float32)
    high = np.array(norm["high"], dtype=np.float32)
    eps = float(norm.get("clip_eps", 1e-5))
    a = 2.0 * (action.astype(np.float32) - low) / (high - low) - 1.0
    return np.clip(a, -1.0 + eps, 1.0 - eps)


def _to_hwc_uint8(img: torch.Tensor) -> np.ndarray:
    """Single image (CHW or HWC, float[0,1]/uint8) -> HWC uint8. The DQC encoder
    normalizes /127.5-1 internally, so we must hand it [0,255] uint8 RGB."""
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


# ── text embedding ──────────────────────────────────────────────────────────────

def _resolve_text_emb(instruction: str, text_emb_file: Optional[str]) -> np.ndarray:
    """Load a precomputed 512-d MUSE (USE-3) embedding; only fall back to the live
    text processor (imports TensorFlow!) if no valid file is given."""
    candidates = []
    if text_emb_file:
        candidates.append(Path(text_emb_file))
    candidates.append(_REPO / "scriptsv2" / "q_fn_eval" / "task_emb_use3.npy")
    for c in candidates:
        if c and Path(c).is_file():
            emb = np.load(c).astype(np.float32)
            if emb.shape == (512,):
                print(f"[DQCVerifier] loaded MUSE (USE-3) embedding from {c}")
                return emb
            print(f"[DQCVerifier] {c} has shape {emb.shape}, expected (512,); skipping")
    print("[DQCVerifier] no precomputed USE-3 embedding found; encoding live "
          "(this imports TensorFlow into the training process) ...")
    sys.path.insert(0, str(_DQC_PKG))
    from utils.text_processing import make_text_processor
    tp = make_text_processor("muse_embedding")
    emb = np.asarray(tp.encode([instruction]), np.float32)[0]
    assert emb.shape == (512,), emb.shape
    return emb


# ── Verifier ────────────────────────────────────────────────────────────────────

class DQCVerifier:
    """In-process REAL DQC verifier. Same interface as RoboMonkeyVerifier."""

    def __init__(
        self,
        ckpt_dir: str,
        instruction: str,
        image_obs_key: str,
        text_emb_file: Optional[str] = None,
        critic_type: str = "chunk",
        score_indices: Optional[list] = None,
        score_agg: str = "discounted_sum",
        score_executed_chunk: bool = False,
        score_full_horizon: bool = False,
        action_chunk_index: int = -1,
        chunk_horizon: Optional[int] = None,
        discount: float = 0.98,
        q_agg: str = "min",
        image_hw: tuple = (224, 224),
        health_check: bool = True,
        # Unused; kept for YAML/back-compat parity with RoboMonkeyVerifier.
        img_size: int = 128,
        enc_batch: int = 64,
        image_feat_cache_size: int = 1024,
        server_url: str = "in_process",
        image_save_dir: str = "/tmp/dqc_obs",
        request_timeout: float = 300.0,
        max_workers: Optional[int] = None,
        keep_jpegs: bool = False,
    ) -> None:
        self.instruction = instruction
        self.image_obs_key = image_obs_key
        self.critic_type = str(critic_type).lower()
        if self.critic_type not in ("action", "chunk"):
            raise ValueError(f"critic_type must be 'action' or 'chunk', got {critic_type!r}")
        self.score_indices = list(score_indices) if score_indices is not None else None
        self.score_agg = str(score_agg)
        self.score_executed_chunk = bool(score_executed_chunk)
        # Score the FULL predicted horizon as the chunk (all H actions), rather
        # than the executed sub-window. Use when the policy predicts exactly
        # chunk_horizon actions from the current frame (e.g. horizon=8, n_obs=1).
        self.score_full_horizon = bool(score_full_horizon)
        self.action_chunk_index = int(action_chunk_index)
        self.discount = float(discount)
        self.q_agg = str(q_agg)
        self.score_batch = int(enc_batch)  # sub-batch B to bound JAX GPU memory
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self._chunk_horizon_cfg = int(chunk_horizon) if chunk_horizon is not None else None
        self._cache_size = int(image_feat_cache_size)
        self._enc_cache: "OrderedDict[bytes, np.ndarray]" = OrderedDict()  # frame -> (num_qs,512)

        ckpt_dir = os.path.expanduser(ckpt_dir)
        with open(Path(ckpt_dir) / "action_normalization.json") as f:
            self._norm = json.load(f)
        self._action_dim = len(self._norm["low"])
        self._text_emb = _resolve_text_emb(instruction, text_emb_file)

        print(f"[DQCVerifier] building REAL BridgeDQCChunkAgent from {ckpt_dir} "
              f"(critic_type={self.critic_type}) ...")
        self._agent, self.chunk_horizon = self._make_agent(ckpt_dir, self.image_hw)
        if self.critic_type == "chunk" and self._chunk_horizon_cfg is not None \
                and self._chunk_horizon_cfg != self.chunk_horizon:
            raise ValueError(
                f"chunk_horizon={self._chunk_horizon_cfg} but checkpoint chunk "
                f"critic expects backup_horizon={self.chunk_horizon}")

        # Split the active critic into encode (cacheable per frame) + head (per
        # action), reusing the REAL flax modules + checkpoint params. Within a
        # gradient step the search loop scores ~max_actions candidates against the
        # SAME frames, so we encode each unique frame once and reuse the 512-d
        # feature for every candidate's cheap MLP head.
        self._build_split()

        if health_check:
            # Verify the cached encode+head split == the fused agent forward.
            H = self.chunk_horizon
            rng = np.random.default_rng(0)
            imgs = rng.integers(0, 256, (3, self.image_hw[0], self.image_hw[1], 3), np.uint8)
            lang = np.repeat(self._text_emb[None], 3, axis=0)
            encs = self._encode_cached(list(imgs))                 # (num_qs,3,512)
            if self.critic_type == "chunk":
                act = rng.uniform(-1, 1, (3, H * self._action_dim)).astype(np.float32)
                q_fused = np.asarray(self._agent.score_chunks(
                    {"image": imgs}, {"language": lang}, act, q_agg=self.q_agg))
            else:
                act = rng.uniform(-1, 1, (3, self._action_dim)).astype(np.float32)
                qa = np.asarray(self._agent.forward_action_critic(
                    {"image": imgs}, {"language": lang}, act, train=False))
                q_fused = qa.min(0) if self.q_agg == "min" else qa.mean(0)
            q_split = self._head_min(encs, act)
            max_diff = float(np.abs(q_fused - q_split).max())
            # The encode/head split materializes the 512-d feature between two
            # XLA graphs, so it differs from the fused forward only by float32
            # reassociation noise (~1e-2 abs on Q~30, corr 1.0). A real logic bug
            # (e.g. the old hand-port) differed by tens. 0.1 cleanly separates them.
            assert np.isfinite(q_split).all() and max_diff < 0.1, \
                f"encode/head split mismatch (max diff {max_diff})"
            self._enc_cache.clear()
            print(f"[DQCVerifier] ready (real agent, cached encode; "
                  f"critic={self.critic_type}, chunk_horizon={self.chunk_horizon}, "
                  f"q_agg={self.q_agg}, split==fused max_diff={max_diff:.2e}, "
                  f"image_hw={self.image_hw}, enc_cache={self._cache_size})")

    # --- build the real flax agent (mirrors eval_dqc_critics.make_agent) ----------
    def _make_agent(self, ckpt_dir: str, image_hw):
        sys.path.insert(0, str(_DQC_PKG))
        from ml_collections import ConfigDict
        from agents.bridge_dqc_chunk import BridgeDQCChunkAgent
        from utils.flax_utils import restore_agent

        cfg = ConfigDict(json.load(open(Path(ckpt_dir) / "flags.json"))["agent"])
        H = int(cfg["backup_horizon"]); A = self._action_dim; Hh, Ww = image_hw
        ex = {
            "observations": {"image": np.zeros((1, Hh, Ww, 3), np.uint8)},
            "next_observations": {"image": np.zeros((1, Hh, Ww, 3), np.uint8)},
            "high_value_next_observations": {"image": np.zeros((1, Hh, Ww, 3), np.uint8)},
            "goals": {"language": np.zeros((1, 512), np.float32)},
            "high_value_goals": {"language": np.zeros((1, 512), np.float32)},
            "actions": np.zeros((1, A), np.float32),
            "high_value_action_chunks": np.zeros((1, H * A), np.float32),
            "rewards": np.zeros((1,), np.float32), "masks": np.ones((1,), np.float32),
            "high_value_rewards": np.zeros((1,), np.float32),
            "high_value_masks": np.ones((1,), np.float32),
            "high_value_backup_horizon": np.full((1,), H, np.float32),
            "valids": np.ones((1, H), np.float32), "mc_returns": np.zeros((1,), np.float32),
        }
        agent = BridgeDQCChunkAgent.create(0, ex, cfg)
        agent = restore_agent(agent, ckpt_dir, 100000)
        self._cfg = cfg
        return agent, H

    # --- encode/head split of the real critic (bit-exact, cacheable) -------------
    def _build_split(self) -> None:
        import jax
        import jax.numpy as jnp
        import flax.linen as nn
        sys.path.insert(0, str(_DQC_PKG))
        from agents.bridge_calql_chunk import BridgeFilmResNet34, LanguageImageEncoder
        from utils.networks import MLP

        cfg = self._cfg
        module = ("modules_chunk_critic" if self.critic_type == "chunk"
                  else "modules_action_critic")
        hidden = tuple(cfg["critic_hidden_dims"] if self.critic_type == "chunk"
                       else cfg["action_critic_hidden_dims"])
        params = self._agent.network.params[module]
        self._num_qs = int(jax.tree_util.tree_leaves(params["Dense_0"])[0].shape[0])

        enc_mod = LanguageImageEncoder(BridgeFilmResNet34(
            pooling_method=cfg["encoder_kwargs"]["pooling_method"],
            add_spatial_coordinates=cfg["encoder_kwargs"]["add_spatial_coordinates"]))
        mlp_mod = MLP(hidden, activate_final=True, layer_norm=cfg["layer_norm"])
        dense_mod = nn.Dense(1)

        def member(tree, e):
            return jax.tree_util.tree_map(lambda x: x[e], tree)
        self._enc_p = [member(params["encoder"], e) for e in range(self._num_qs)]
        self._mlp_p = [member(params["MLP_0"], e) for e in range(self._num_qs)]
        self._dns_p = [member(params["Dense_0"], e) for e in range(self._num_qs)]
        self._jnp = jnp

        @jax.jit
        def _encode_e(imgs, lang, p):                  # imgs uint8 (N,H,W,3), lang (N,512)
            return enc_mod.apply({"params": p}, ({"image": imgs}, {"language": lang}),
                                 train=False)          # (N,512)

        @jax.jit
        def _head_e(enc, act, pm, pd):                 # enc (N,512), act (N,Adim)
            x = jnp.concatenate([enc, act], axis=-1)
            h = mlp_mod.apply({"params": pm}, x)
            return jnp.squeeze(dense_mod.apply({"params": pd}, h), -1)  # (N,)

        self._encode_e = _encode_e
        self._head_e = _head_e

    def _encode_cached(self, frames_u8: List[np.ndarray]) -> np.ndarray:
        """Return (num_qs, B, 512) encodings for B frames; encode only cache misses
        (sub-batched to bound GPU memory). Text is fixed, so keying on the frame
        bytes is sufficient."""
        keys = [hashlib.blake2b(np.ascontiguousarray(f).tobytes(), digest_size=16).digest()
                for f in frames_u8]
        miss_keys, miss_imgs, seen = [], [], set()
        for k, f in zip(keys, frames_u8):
            if k not in self._enc_cache and k not in seen:
                seen.add(k); miss_keys.append(k); miss_imgs.append(f)
        for s in range(0, len(miss_imgs), self.score_batch):
            batch = np.stack(miss_imgs[s:s + self.score_batch])             # (n,H,W,3)
            lang = np.repeat(self._text_emb[None], batch.shape[0], axis=0)  # (n,512)
            encs = np.stack([np.asarray(self._encode_e(self._jnp.asarray(batch),
                            self._jnp.asarray(lang), self._enc_p[e]))
                            for e in range(self._num_qs)])                  # (num_qs,n,512)
            for i, k in enumerate(miss_keys[s:s + self.score_batch]):
                self._enc_cache[k] = encs[:, i, :]                         # (num_qs,512)
        for k in keys:
            self._enc_cache.move_to_end(k)
        while len(self._enc_cache) > self._cache_size:
            self._enc_cache.popitem(last=False)
        return np.stack([self._enc_cache[k] for k in keys], axis=1)         # (num_qs,B,512)

    def _head_min(self, encs: np.ndarray, act: np.ndarray) -> np.ndarray:
        """encs (num_qs,B,512), act (B,Adim) -> (B,) aggregated over the ensemble."""
        a = self._jnp.asarray(act)
        qs = np.stack([np.asarray(self._head_e(self._jnp.asarray(encs[e]), a,
                      self._mlp_p[e], self._dns_p[e])) for e in range(self._num_qs)])
        return qs.min(0) if self.q_agg == "min" else qs.mean(0)

    def clear_cache(self) -> None:
        self._enc_cache.clear()

    # --- obs / aggregation helpers -----------------------------------------------
    def _select_last_frames_uint8(self, obs_dict: Dict[str, Any], B: int) -> np.ndarray:
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
        last = x[:, -1]  # (B, C, H, W) or (B, H, W, C)
        if last.shape[0] != B:
            raise ValueError(f"Batch mismatch: action B={B} but obs B={last.shape[0]}")
        return np.stack([_to_hwc_uint8(last[b]) for b in range(B)])  # (B,H,W,3) uint8

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
        raise ValueError(f"unknown score_agg {agg!r}")

    @torch.no_grad()
    def get_value(self, obs_dict: Dict[str, Any], action: torch.Tensor) -> torch.Tensor:
        """Score each candidate against the current frame -> (B,)."""
        if action.dim() != 3:
            raise ValueError(
                f"Expected action (B, horizon, action_dim); got {tuple(action.shape)}")
        B, H, A = int(action.shape[0]), int(action.shape[1]), int(action.shape[-1])
        images = self._select_last_frames_uint8(obs_dict, B)        # (B,Hh,Ww,3) uint8
        encs = self._encode_cached(list(images))                    # (num_qs,B,512) cached

        if self.score_full_horizon:
            idxs = list(range(H))                       # all predicted actions = the chunk
        elif self.score_indices is not None:
            idxs = self.score_indices
        else:
            idxs = [self.action_chunk_index]
        idxs = [int(i) for i in idxs]
        if any(i >= H or i < -H for i in idxs):
            raise ValueError(f"score_indices {idxs} out of bounds for horizon H={H}.")
        M = len(idxs)

        # ── Chunk critic: the executed window IS the chunk -> ONE head call.
        if self.critic_type == "chunk":
            if M != self.chunk_horizon:
                raise ValueError(
                    f"chunk critic expects {self.chunk_horizon} actions per chunk "
                    f"(backup_horizon) but the scoring window has {M} indices {idxs}. "
                    f"Set the executed window (n_action_steps) to {self.chunk_horizon}.")
            sub = action[:, idxs, :].detach().cpu().numpy().reshape(B * M, A)
            sub = np.stack([_normalize_action(a, self._norm) for a in sub])
            chunk = sub.reshape(B, M * A).astype(np.float32)        # time-major (B, H*7)
            # Match training chunk assembly: zero-fill past-episode pad in
            # NORMALIZED space (the BC policy emits no NaN, so this is a no-op
            # there; it only matters when scoring NaN-padded eval chunks).
            chunk = np.where(np.isnan(chunk), 0.0, chunk).astype(np.float32)
            q = self._head_min(encs, chunk)                         # (B,)
            return torch.tensor(np.asarray(q), dtype=action.dtype, device=action.device)

        # ── Action critic: per-step single-action Q over the window, aggregated.
        per_step_cols = []
        for i in idxs:
            a_i = action[:, i, :].detach().cpu().numpy()
            a_i = np.stack([_normalize_action(a, self._norm) for a in a_i]).astype(np.float32)
            qa = self._head_min(encs, a_i)                          # (B,)
            per_step_cols.append(torch.tensor(np.asarray(qa), dtype=action.dtype,
                                              device=action.device))
        per_step = torch.stack(per_step_cols, dim=1)                # (B, M)
        if M == 1:
            return per_step[:, 0]
        return self._aggregate(per_step)

    # --- KV-prefix controls: no-ops (kept for SearchPolicy parity) ----------------
    @property
    def supports_kv_prefix(self) -> bool:
        return False

    def set_kv_prefix(self, enabled: bool) -> None:
        pass

    def clear_prefix_cache(self) -> None:
        pass
