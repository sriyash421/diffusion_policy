"""In-memory training frames for PiZeroContext from the step-2 precompute outputs.

One frame = replay image (what pi0 sees at eval) + raw proprio with gripper openness +
expert chunk target (zarr actions t..t+3, last action repeated past the episode end,
model-space normalised) + the frozen base's candidate chunks with their verifier values
(generation order). Frames whose replay drifted more than ``max_tcp_err_mm`` from the zarr
are dropped. The whole train split is ~3k frames, ~0.5 GB as uint8, so it lives in RAM.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Sequence

import numpy as np
import torch
import zarr

from diffusion_policy.pi0_context.official import clip_value, normalize_bridge_actions


class ContextFrames:
    def __init__(self, split_file: str, replay_dir: str, candidates_dir: str, adapter,
                 splits: Sequence[str] = ("train",), horizon: int = 4, max_tcp_err_mm: float = 5.0,
                 n_ctx_blocks: int = 4):
        man = json.load(open(split_file))
        from diffusion_policy.pi0_context.rollout_dataset import INSTRUCTIONS
        self.instruction = man.get("instruction") or INSTRUCTIONS[man["env_task"]]
        eps = sorted(set(i for s in splits for i in man[s]))
        root = zarr.open(man["zarr_path"], "r")
        ends = root["meta/episode_ends"][:]
        starts = np.concatenate([[0], ends[:-1]])
        actions_all = root["data/actions"][:]
        imgs, props, targets, ctx_a, ctx_v, ep_ids, ts = [], [], [], [], [], [], []
        n_drop = 0
        for ep in eps:
            rp = np.load(os.path.join(replay_dir, f"ep{ep:04d}.npz"))
            cd = np.load(os.path.join(candidates_dir, f"ep{ep:04d}.npz"))
            s, e = int(starts[ep]), int(ends[ep])
            T = e - s
            assert len(rp["image"]) == T == len(cd["ctx_values"]), (ep, T)
            acts = actions_all[s:e]
            idx = np.minimum(np.arange(T)[:, None] + np.arange(horizon)[None, :], T - 1)
            chunk = normalize_bridge_actions(adapter, acts[idx])                     # (T, H, 7)
            ok = rp["tcp_err"] * 1e3 <= max_tcp_err_mm
            n_drop += int((~ok).sum())
            # generation order (search-transformer convention: slot k sees the first k)
            ca = cd["ctx_actions_norm"]
            cv = clip_value(cd["ctx_values"])
            imgs.append(rp["image"][ok]); props.append(rp["proprio"][ok]); targets.append(chunk[ok])
            ctx_a.append(ca[ok, :n_ctx_blocks]); ctx_v.append(cv[ok, :n_ctx_blocks])
            ep_ids.append(np.full(int(ok.sum()), ep)); ts.append(np.arange(T)[ok])
        self.image = np.concatenate(imgs)
        self.proprio = np.concatenate(props).astype(np.float32)
        self.target = np.concatenate(targets).astype(np.float32)
        self.ctx_actions = np.concatenate(ctx_a).astype(np.float32)
        self.ctx_values = np.concatenate(ctx_v).astype(np.float32)
        self.episode = np.concatenate(ep_ids)
        self.t = np.concatenate(ts)
        self.n_ctx_blocks = n_ctx_blocks
        self.n_dropped = n_drop
        self.episodes = eps

    def __len__(self):
        return len(self.image)

    def batch(self, idx: np.ndarray, n_ctx: np.ndarray) -> dict:
        """numpy batch; n_ctx (B,) how many best-first blocks each sample sees."""
        return dict(image=self.image[idx], proprio=self.proprio[idx], actions=self.target[idx],
                    ctx_actions=self.ctx_actions[idx], ctx_values=self.ctx_values[idx],
                    n_ctx=n_ctx.astype(np.int64))

    def sample_n_ctx(self, rng: np.random.Generator, size: int, mode: str) -> np.ndarray:
        if mode == "none":
            return np.zeros(size, dtype=np.int64)
        if mode == "full":
            return np.full(size, self.n_ctx_blocks, dtype=np.int64)
        if mode == "uniform":
            return rng.integers(0, self.n_ctx_blocks + 1, size=size)
        raise ValueError(mode)


class MultiTaskFrames:
    """Several ContextFrames (one per task) pooled, with ``task_id`` indexing ``instructions``."""

    def __init__(self, parts: Dict[str, ContextFrames]):
        self.tasks = sorted(parts)
        self.instructions = [parts[t].instruction for t in self.tasks]
        cat = lambda k: np.concatenate([getattr(parts[t], k) for t in self.tasks])
        self.image, self.proprio, self.target = cat("image"), cat("proprio"), cat("target")
        self.ctx_actions, self.ctx_values = cat("ctx_actions"), cat("ctx_values")
        self.task_id = np.concatenate([np.full(len(parts[t]), i) for i, t in enumerate(self.tasks)])
        self.n_ctx_blocks = min(parts[t].n_ctx_blocks for t in self.tasks)
        self.n_dropped = sum(parts[t].n_dropped for t in self.tasks)

    def __len__(self):
        return len(self.image)

    def batch(self, idx, n_ctx):
        return dict(image=self.image[idx], proprio=self.proprio[idx], actions=self.target[idx],
                    ctx_actions=self.ctx_actions[idx], ctx_values=self.ctx_values[idx],
                    n_ctx=n_ctx.astype(np.int64), task_id=self.task_id[idx])

    sample_n_ctx = ContextFrames.sample_n_ctx

    def summary(self):
        per = {t: int((self.task_id == i).sum()) for i, t in enumerate(self.tasks)}
        return f"{len(self)} frames ({self.n_dropped} dropped by tcp_err); per task {per}"


class BatchEncoder:
    """Frame batch -> model inputs. Instructions are tokenised once (one or several tasks;
    a batch's ``task_id`` picks the row); pixel values use the official processor's image path."""

    def __init__(self, adapter, instruction, device="cuda"):
        from src.model.vla.processing import process_images, IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
        self._process_images = process_images
        self._mean, self._std = IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
        texts = [instruction] if isinstance(instruction, str) else list(instruction)
        tok = adapter.processor(text=texts, images=torch.zeros(len(texts), 3, 224, 224, dtype=torch.uint8))
        self.input_ids = tok["input_ids"].to(device)            # (n_tasks, 276)
        self.attention_mask = tok["attention_mask"].to(device)
        stats = adapter.dataset_statistics["proprio"]
        self.p01 = np.array(stats["p01"], dtype=np.float32)
        self.p99 = np.array(stats["p99"], dtype=np.float32)
        self.adapter = adapter
        self.device = device

    def __call__(self, b: dict) -> dict:
        B = len(b["image"])
        images = torch.as_tensor(b["image"]).permute(0, 3, 1, 2)
        pixel_values = self._process_images(images, rescale_factor=1 / 255.0, image_mean=self._mean,
                                            image_std=self._std).to(self.device, torch.float32)
        proprio = self.adapter.normalize_bound(b["proprio"], self.p01, self.p99, clip_min=-1, clip_max=1)
        d = self.device
        tid = torch.as_tensor(b.get("task_id", np.zeros(B, dtype=np.int64)), device=d, dtype=torch.long)
        return dict(
            input_ids=self.input_ids[tid].contiguous(),
            attention_mask=self.attention_mask[tid].contiguous(),
            pixel_values=pixel_values,
            proprios=torch.as_tensor(proprio, dtype=torch.float32, device=d)[:, None],
            actions=torch.as_tensor(b["actions"], device=d),
            ctx_actions=torch.as_tensor(b["ctx_actions"], device=d),
            ctx_values=torch.as_tensor(b["ctx_values"], device=d),
            n_ctx=torch.as_tensor(b["n_ctx"], device=d),
        )
