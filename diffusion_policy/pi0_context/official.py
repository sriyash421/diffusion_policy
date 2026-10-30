"""Thin access to the official open-pi-zero pieces: config, checkpoint, Bridge adapter, and the
conversion of recorded frames / Bridge-format actions into the model's input space.

Every normalisation here is the adapter's own (``normalize_bound`` with the Bridge p01/p99
statistics), so a chunk normalised by :func:`normalize_bridge_actions` and passed through
``adapter.postprocess`` returns the env action the official eval would execute.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Sequence

import numpy as np
import torch

VERITAS_ROOT = os.environ.get("VERITAS_ROOT", "/gscratch/robotics/harine/veritas")
if VERITAS_ROOT not in sys.path:
    sys.path.insert(0, VERITAS_ROOT)

DEFAULT_CKPT = "models/bridge_uniform_step19296_2024-12-26_22-31_42.pt"
GRIPPER_OPEN_THRESHOLD = 0.5   # Bridge gripper label: 0 = close, 1 = open (zarr stores 0.996)
VALUE_MIN = -10.0   # feedback = -cost/tol_px, clipped: beyond 10 tolerances (or unprojectable, 1e4) is "far"


def clip_value(v):
    return np.maximum(np.asarray(v, dtype=np.float32), VALUE_MIN)


def compose_eval_cfg(task: str = "widowx_put_eggplant_in_basket", overrides: Sequence[str] = ()):
    """``config/eval/bridge.yaml`` composed like ``scripts/run.py`` does, resolved."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    os.environ.setdefault("VLA_LOG_DIR", "/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2/logs")
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(VERITAS_ROOT, "transformers_cache"))
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
    with initialize_config_dir(config_dir=os.path.join(VERITAS_ROOT, "config", "eval"),
                               version_base=None):
        cfg = compose(config_name="bridge.yaml", overrides=[f"env.task={task}", *overrides])
    OmegaConf.resolve(cfg)
    with open_dict(cfg):
        cfg.gpu_id = 0
        cfg.multi_gpu = False
        # relative to the checkout in the official scripts (they cd there); absolute here
        for key in ("checkpoint_path",):
            if cfg.get(key) and not os.path.isabs(cfg[key]):
                cfg[key] = os.path.join(VERITAS_ROOT, cfg[key])
        if not os.path.isabs(cfg.env.adapter.dataset_statistics_path):
            cfg.env.adapter.dataset_statistics_path = os.path.join(
                VERITAS_ROOT, cfg.env.adapter.dataset_statistics_path)
    return cfg


def checkpoint_state_dict(path: str) -> dict:
    """Model weights of an official checkpoint (``_orig_mod.`` prefixes stripped)."""
    if not os.path.isabs(path):
        path = os.path.join(VERITAS_ROOT, path)
    data = torch.load(path, weights_only=True, map_location="cpu")
    return {k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}


def load_pizero(cfg, checkpoint_path: str = DEFAULT_CKPT, cls=None, strict: bool = True,
                device="cuda", dtype=torch.float32):
    """``PiZeroInference`` (default) or a subclass, with the checkpoint loaded, frozen, in eval
    mode. ``strict=False`` is for subclasses that add parameters."""
    from src.model.vla.pizero import PiZeroInference
    cls = cls or PiZeroInference
    model = cls(cfg, use_ddp=False)
    missing, unexpected = model.load_state_dict(checkpoint_state_dict(checkpoint_path), strict=strict)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:5]}")
    model.freeze_all_weights()
    return model.to(dtype).to(device).eval(), missing


def make_adapter(cfg):
    import hydra
    return hydra.utils.instantiate(cfg.env.adapter)


def frame_inputs(adapter, images_u8: np.ndarray, proprio_raw: np.ndarray, instruction: str,
                 device="cuda", dtype=torch.float32) -> dict:
    """Recorded frames -> the dict ``adapter.preprocess`` would build at eval.

    images_u8 (B, 224, 224, 3) uint8 already LANCZOS4-resized; proprio_raw (B, 7) as
    ``preprocess_proprio`` returns it (xyz, rpy, gripper openness)."""
    images = torch.as_tensor(np.ascontiguousarray(images_u8), dtype=torch.uint8).permute(0, 3, 1, 2)
    out = adapter.processor(text=[instruction] * len(images), images=images)
    stats = adapter.dataset_statistics["proprio"]
    proprio = adapter.normalize_bound(np.asarray(proprio_raw, dtype=np.float32),
                                      np.array(stats["p01"]), np.array(stats["p99"]),
                                      clip_min=-1, clip_max=1)
    return dict(
        input_ids=out["input_ids"].to(device),
        attention_mask=out["attention_mask"].to(device),
        pixel_values=out["pixel_values"].to(device, dtype),
        proprios=torch.as_tensor(proprio, dtype=dtype, device=device)[:, None],   # (B, 1, 7)
    )


def normalize_bridge_actions(adapter, actions: np.ndarray) -> np.ndarray:
    """Bridge-format actions (..., 7): xyz + rpy deltas and gripper in [0, 1] -> model space
    (dims 0-5 bound-normalised to [-1, 1]; gripper binarised to {0, 1}, raw as in training)."""
    a = np.asarray(actions, dtype=np.float32)
    stats = adapter.dataset_statistics["action"]
    body = adapter.normalize_bound(a[..., :6], np.array(stats["p01"])[:-1],
                                   np.array(stats["p99"])[:-1], clip_min=-1, clip_max=1)
    grip = (a[..., 6:7] > GRIPPER_OPEN_THRESHOLD).astype(np.float32)
    return np.concatenate([body, grip], axis=-1).astype(np.float32)


def model_masks(model, attention_mask: torch.Tensor, dtype=torch.float32, split: bool = True):
    """Masks and position ids for a batch, on the attention mask's device."""
    causal_mask, vlm_pos, prop_pos, act_pos = model.build_causal_mask_and_position_ids(
        attention_mask.cpu(), dtype)
    dev = attention_mask.device
    ids = dict(vlm_position_ids=vlm_pos.to(dev), proprio_position_ids=prop_pos.to(dev),
               action_position_ids=act_pos.to(dev))
    if split:
        itp, am = model.split_full_mask_into_submasks(causal_mask)
        return dict(image_text_proprio_mask=itp.to(dev), action_mask=am.to(dev), **ids)
    return dict(causal_mask=causal_mask.to(dev), **ids)
