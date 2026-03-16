"""Dataset for loading UWLab .pt trajectory files (image + low-dim obs)."""

from typing import Dict, List, Optional
import copy
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from threadpoolctl import threadpool_limits

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.common.normalize_util import get_image_range_normalizer


def _find_obs_value(traj_data: dict, key: str):
    """Search for an observation key across all UWLab trajectory obs dicts."""
    for group in ("obs_proprio", "obs_assets", "obs_images", "obs_other_state"):
        container = traj_data.get(group, {})
        if isinstance(container, dict) and key in container:
            v = container[key]
            return v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    if key == "rendered_images":
        v = traj_data.get("rendered_images")
        if v is not None:
            return v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    return None


def _to_hwc_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalise an image array to (T, H, W, C) uint8."""
    if arr.ndim == 3:
        arr = arr[..., np.newaxis]
    if arr.ndim == 4:
        if arr.shape[1] in (1, 3, 4) and arr.shape[1] < arr.shape[2]:
            arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


class UwlabTrajectoryImageDataset(BaseImageDataset):
    """Loads UWLab `.pt` trajectory data for image-conditioned diffusion policy training.

    Each trajectory file is expected to contain:
        - ``actions``          (T, Da)  torch.Tensor
        - ``obs_proprio``      dict[str, (T, d)]
        - ``obs_assets``       dict[str, (T, d)]
        - ``obs_images``       dict[str, (T, C, H, W) | (T, H, W, C)]
        - ``obs_other_state``  dict[str, (T, d)]
        - ``rendered_images``  (T, H, W, C) np.ndarray   (optional)

    ``shape_meta`` determines which keys are loaded and whether they are ``rgb``
    or ``low_dim``.
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_dir: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: Optional[int] = None,
        n_latency_steps: int = 0,
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: Optional[int] = None,
    ):
        dataset_dir = Path(dataset_dir)
        assert dataset_dir.is_dir(), f"Dataset directory not found: {dataset_dir}"

        rgb_keys: List[str] = []
        lowdim_keys: List[str] = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            if attr.get("type", "low_dim") == "rgb":
                rgb_keys.append(key)
            else:
                lowdim_keys.append(key)

        manifest_path = dataset_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        replay_buffer = ReplayBuffer.create_empty_numpy()
        traj_files = manifest.get("files", [])
        for traj_info in tqdm(traj_files, desc="Loading UWLab image trajectories"):
            traj_path = dataset_dir / traj_info["file"]
            traj_data = torch.load(traj_path, map_location="cpu")

            actions = traj_data["actions"]
            if isinstance(actions, torch.Tensor):
                actions = actions.numpy()

            episode: Dict[str, np.ndarray] = {
                "action": actions.astype(np.float32),
            }

            for key in lowdim_keys:
                value = _find_obs_value(traj_data, key)
                if value is None:
                    raise KeyError(
                        f"Low-dim obs key '{key}' not found in {traj_path}. "
                        f"Available proprio={list(traj_data.get('obs_proprio', {}).keys())}, "
                        f"assets={list(traj_data.get('obs_assets', {}).keys())}, "
                        f"other={list(traj_data.get('obs_other_state', {}).keys())}"
                    )
                episode[key] = value.astype(np.float32)

            for key in rgb_keys:
                value = _find_obs_value(traj_data, key)
                if value is None:
                    raise KeyError(
                        f"Image obs key '{key}' not found in {traj_path}. "
                        f"Available images={list(traj_data.get('obs_images', {}).keys())}"
                    )
                episode[key] = _to_hwc_uint8(value)

            replay_buffer.add_episode(episode)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        key_first_k: dict = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"]
        )
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key]
            )
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)

        obs_dict: dict = {}
        for key in self.rgb_keys:
            obs_dict[key] = (
                np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        action = data["action"].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }
        return torch_data
