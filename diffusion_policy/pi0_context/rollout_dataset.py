"""Training frames from collected pi0+verifier rollouts (ContextEvalAgent ``+collect_decisions``).

One frame = one decision: the frame pi0 saw, raw proprio, the task instruction, the N candidates
sampled there with their verifier values (generation order) and the official argmax's pick.
Targets (``target``):
  argmax   the chosen chunk; context (``context``):
             others   the other N-1 candidates in generation order (arm A)
             prefix   the first N-1 candidates in generation order, the chosen one included if
                      it is among them: what the sliding eval shows the model (chunk i sees the
                      previous ones, good ones too), target = best of the whole set (arm S)
  expert   not available here (demo-based ``ContextFrames`` covers it)
Episodes of several tasks are pooled; ``success_only`` keeps decisions of successful attempts.
Train/val split by seed: the last ``n_val_seeds`` collection seeds of each task are val.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Sequence

import numpy as np

from diffusion_policy.pi0_context.official import clip_value

INSTRUCTIONS = {   # official veritas README table
    "widowx_put_eggplant_in_basket": "put eggplant into yellow basket",
    "widowx_stack_cube": "stack the green cube on top of the yellow cube",
    "widowx_carrot_on_plate": "put carrot on plate",
    "widowx_spoon_on_towel": "put the spoon on the towel",
}


class RolloutFrames:
    def __init__(self, collect_dirs: Dict[str, str], split: str = "train", n_val_seeds: int = 30,
                 target: str = "argmax", success_only: bool = False, n_ctx_blocks: int = 4,
                 attempts: Sequence[int] = (1, 2, 3), context: str = "others"):
        assert target == "argmax", target
        assert context in ("others", "prefix"), context
        imgs, props, tgts, ctx_a, ctx_v, task_ids, seeds, atts = [], [], [], [], [], [], [], []
        self.tasks = sorted(collect_dirs)
        self.task_index = {t: i for i, t in enumerate(self.tasks)}
        self.n_episodes = 0
        for task in self.tasks:
            files = sorted(glob.glob(os.path.join(collect_dirs[task], "ep*_a*.npz")))
            all_seeds = sorted(set(int(os.path.basename(f)[2:8]) for f in files))
            val_seeds = set(all_seeds[-n_val_seeds:])
            for f in files:
                seed = int(os.path.basename(f)[2:8]); att = int(os.path.basename(f).split("_a")[1].split(".")[0])
                if (seed in val_seeds) != (split == "val") or att not in attempts:
                    continue
                d = np.load(f)
                if success_only and not bool(d["success"]):
                    continue
                n = int(d["n_decisions"])
                if n == 0:
                    continue
                cand, val, best = d["cand_norm"], clip_value(d["values"]), d["best_idx"].astype(int)   # (n,N,4,7),(n,N),(n,)
                N = cand.shape[1]
                if context == "others":
                    keep = np.array([[j for j in range(N) if j != b] for b in best])        # (n,N-1)
                else:
                    keep = np.tile(np.arange(N - 1), (n, 1))                                 # (n,N-1)
                others = np.take_along_axis(cand, keep[:, :, None, None], axis=1)[:, :n_ctx_blocks]
                others_v = np.take_along_axis(val, keep, axis=1)[:, :n_ctx_blocks]
                imgs.append(d["image"]); props.append(d["proprio"])
                tgts.append(cand[np.arange(n), best]); ctx_a.append(others); ctx_v.append(others_v)
                task_ids.append(np.full(n, self.task_index[task])); seeds.append(np.full(n, seed)); atts.append(np.full(n, att))
                self.n_episodes += 1
        self.image = np.concatenate(imgs)
        self.proprio = np.concatenate(props).astype(np.float32)
        self.target = np.concatenate(tgts).astype(np.float32)
        self.ctx_actions = np.concatenate(ctx_a).astype(np.float32)
        self.ctx_values = np.concatenate(ctx_v).astype(np.float32)
        self.task_id = np.concatenate(task_ids)
        self.seed = np.concatenate(seeds)
        self.attempt = np.concatenate(atts)
        self.n_ctx_blocks = min(n_ctx_blocks, self.ctx_actions.shape[1])
        self.instructions = [INSTRUCTIONS[t] for t in self.tasks]

    def __len__(self):
        return len(self.image)

    def batch(self, idx: np.ndarray, n_ctx: np.ndarray) -> dict:
        return dict(image=self.image[idx], proprio=self.proprio[idx], actions=self.target[idx],
                    ctx_actions=self.ctx_actions[idx], ctx_values=self.ctx_values[idx],
                    n_ctx=n_ctx.astype(np.int64), task_id=self.task_id[idx])

    def sample_n_ctx(self, rng: np.random.Generator, size: int, mode: str) -> np.ndarray:
        if mode == "none":
            return np.zeros(size, dtype=np.int64)
        if mode == "full":
            return np.full(size, self.n_ctx_blocks, dtype=np.int64)
        if mode == "uniform":
            return rng.integers(0, self.n_ctx_blocks + 1, size=size)
        raise ValueError(mode)

    def summary(self) -> str:
        per_task = {t: int((self.task_id == i).sum()) for t, i in self.task_index.items()}
        return (f"{len(self)} decisions from {self.n_episodes} attempts; per task {per_task}; "
                f"value mean {self.ctx_values.mean():.3f} std {self.ctx_values.std():.3f}")
