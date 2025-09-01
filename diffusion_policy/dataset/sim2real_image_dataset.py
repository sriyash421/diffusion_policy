from typing import Dict
import torch
import numpy as np
import copy
import zarr
import os
import gc
import shutil
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)
from diffusion_policy.dataset.base_dataset import BaseImageDataset


class CustomWeightedRandomSampler(torch.utils.data.WeightedRandomSampler):
    """
    WeightedRandomSampler except allows for more than 2^24 samples
    copied from https://github.com/pytorch/pytorch/issues/2576
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        weights_np = self.weights.numpy()
        weights_sum = torch.sum(self.weights).numpy()
        rand_tensor = np.random.choice(
            range(0, len(self.weights)),
            size=self.num_samples,
            p=weights_np / weights_sum,
            replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


class Sim2RealImageDataset(BaseImageDataset):
    def __init__(
        self,
        dataset_path: str,
        shape_meta: dict,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        n_latency_steps=0,
        seed=42,
        val_ratio=0.0,
        use_cache: bool = False,
        use_disk: bool = True,
    ):
        super().__init__()
        assert os.path.isdir(dataset_path)

        # Load data and create replay buffer
        self.replay_buffer = self._create_replay_buffer_from_zarr(
            dataset_path, shape_meta=shape_meta, use_cache=use_cache,
            use_disk=use_disk)

        # Parse keys
        self.rgb_keys = [
            k for k, v in shape_meta['obs'].items()
            if v.get('type', 'low_dim') == 'rgb']
        self.lowdim_keys = [
            k for k, v in shape_meta['obs'].items()
            if v.get('type', 'low_dim') == 'low_dim']
        if shape_meta.get('auxiliary_obs', None) is not None:
            self.lowdim_keys.extend([
                k for k, v in shape_meta['auxiliary_obs'].items()
            ])

        # Create key_first_k for performance optimization
        key_first_k = dict()
        if n_obs_steps is not None:
            for key in self.rgb_keys + self.lowdim_keys:
                key_first_k[key] = n_obs_steps

        # Split train/val
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        # Create sampler
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)

        # Store parameters
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.val_ratio = val_ratio
        self.seed = seed
        self.shape_meta = shape_meta
        self.val_mask = val_mask
        self.train_mask = train_mask

    def _create_replay_buffer_from_zarr(
            self, dataset_path, shape_meta, use_cache=False, use_disk=True):
        """Create replay buffer from zarr data with caching and memory options."""
        replay_buffer = None

        if use_cache:
            # Create fingerprint for caching
            shape_meta_json = json.dumps(
                OmegaConf.to_container(shape_meta), sort_keys=True)
            dataset_stat = os.stat(dataset_path)
            cache_fingerprint = {
                'shape_meta': shape_meta_json,
                'dataset_path': dataset_path,
                'dataset_mtime': dataset_stat.st_mtime,
                'dataset_size': dataset_stat.st_size,
                'use_disk': use_disk
            }
            cache_fingerprint_json = json.dumps(
                cache_fingerprint, sort_keys=True)
            cache_hash = hashlib.md5(
                cache_fingerprint_json.encode('utf-8')).hexdigest()

            cache_zarr_path = os.path.join(
                dataset_path, cache_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'

            print(f'Acquiring lock on cache: {cache_zarr_path}')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # Cache does not exist, create it
                    try:
                        print('Cache does not exist. Creating!')
                        # Always load to memory for caching
                        replay_buffer = self._load_zarr_data(
                            dataset_path, use_disk=False)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)

                        # Clear memory after cache creation
                        del replay_buffer
                        gc.collect()
                        replay_buffer = None
                    except Exception as e:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from disk.')

                if replay_buffer is None:
                    # Load from cache
                    if use_disk:
                        # Load cache but keep it on disk (memory mapped)
                        zip_store = zarr.ZipStore(cache_zarr_path, mode='r')
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    else:
                        # Load cache into memory
                        with zarr.ZipStore(
                                cache_zarr_path, mode='r') as zip_store:
                            replay_buffer = ReplayBuffer.copy_from_store(
                                src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded cached data!')
        else:
            # No caching, load directly
            replay_buffer = self._load_zarr_data(
                dataset_path, use_disk=use_disk)

        return replay_buffer

    def _load_zarr_data(self, dataset_path, use_disk=True):
        """Load zarr data with disk/memory options"""
        z = zarr.open(dataset_path, mode='r')
        obs_group = z['data']['obs']
        action_arr = z['data']['actions']
        episode_ends = z['meta']['episode_ends']

        # Create replay buffer
        replay_buffer = ReplayBuffer.create_empty_numpy()

        # Add observations
        for key in obs_group.keys():
            if use_disk:
                # Keep data on disk (memory mapped)
                replay_buffer.root['data'][key] = obs_group[key]
            else:
                # Load into memory
                replay_buffer.root['data'][key] = obs_group[key][:]

        # Always load to memory as it's small
        replay_buffer.root['data']['action'] = action_arr[:]
        replay_buffer.root['meta']['episode_ends'] = episode_ends[:]

        return replay_buffer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask)
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        # action
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        # obs
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])
        # don't normalize rgb, obs_encoder has image_net norm
        for key in self.rgb_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_identity()
        # for key in self.rgb_keys:
        #     normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = (np.moveaxis(data[key][T_slice], -1, 1)
                             .astype(np.float32) / 255.)
            # T,C,H,W
            # save ram
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            # save ram
            del data[key]

        action = data['action'].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action),
        }
        return torch_data


class Sim2RealImageMultiDataset(BaseImageDataset):
    """
    Multi-dataset wrapper that combines multiple Sim2RealImageDataset instances
    with weighted sampling.
    """

    def __init__(
        self,
        dataset_config: list,
        shape_meta: dict,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        n_latency_steps=0,
        seed=42,
        val_ratio=0.0,
        use_cache: bool = True,
        use_disk: bool = True,
    ):
        super().__init__()
        self.dataset_config = dataset_config
        self.shape_meta = shape_meta

        # Initialize datasets
        self.datasets = []
        self.dataset_lengths = []
        self.sampling_ratios = []

        common_kwargs = {
            'shape_meta': shape_meta,
            'horizon': horizon,
            'pad_before': pad_before,
            'pad_after': pad_after,
            'n_obs_steps': n_obs_steps,
            'n_latency_steps': n_latency_steps,
            'seed': seed,
            'val_ratio': val_ratio,
            'use_cache': use_cache,
            'use_disk': use_disk
        }

        for config in dataset_config:
            dataset_kwargs = {**common_kwargs}
            dataset_kwargs['dataset_path'] = config['dataset_path']

            dataset = Sim2RealImageDataset(**dataset_kwargs)

            self.datasets.append(dataset)
            self.dataset_lengths.append(len(dataset))
            self.sampling_ratios.append(config['sampling_ratio'])

        # Calculate cumulative indices for mapping global index to
        # (dataset_idx, local_idx)
        self.cumulative_lengths = np.cumsum([0] + self.dataset_lengths)
        self.total_length = int(self.cumulative_lengths[-1])

        # Calculate sampling weights
        self.weights = self._calculate_weights()

        # Create weighted sampler
        self.weighted_sampler = CustomWeightedRandomSampler(
            weights=self.weights, num_samples=self.total_length,
            replacement=True)

    def _calculate_weights(self) -> torch.Tensor:
        """Calculate per-sample weights based on dataset sizes and sampling ratios."""
        weights = []

        # Normalize sampling ratios
        total_ratio = sum(self.sampling_ratios)
        normalized_ratios = [r / total_ratio for r in self.sampling_ratios]

        for dataset_idx, (dataset_len, ratio) in enumerate(
                zip(self.dataset_lengths, normalized_ratios)):
            # Weight per sample = ratio / dataset_length
            sample_weight = ratio / dataset_len
            weights.extend([sample_weight] * dataset_len)

        return torch.tensor(weights, dtype=torch.float32)

    def _global_to_local_index(self, global_idx: int) -> tuple:
        """Convert global index to (dataset_idx, local_idx)."""
        dataset_idx = np.searchsorted(
            self.cumulative_lengths[1:], global_idx, side='right')
        local_idx = global_idx - self.cumulative_lengths[dataset_idx]
        return dataset_idx, local_idx

    def get_validation_dataset(self):
        """Create validation version of this multi-dataset wrapper."""
        val_wrapper = copy.copy(self)
        val_wrapper.datasets = [
            dataset.get_validation_dataset() for dataset in self.datasets]

        # Recalculate lengths and weights for validation sets
        val_wrapper.dataset_lengths = [
            len(dataset) for dataset in val_wrapper.datasets]
        val_wrapper.cumulative_lengths = np.cumsum(
            [0] + val_wrapper.dataset_lengths)
        val_wrapper.total_length = int(val_wrapper.cumulative_lengths[-1])
        val_wrapper.weights = val_wrapper._calculate_weights()

        # Create new weighted sampler for validation
        val_wrapper.weighted_sampler = CustomWeightedRandomSampler(
            weights=val_wrapper.weights,
            num_samples=val_wrapper.total_length,
            replacement=True
        )

        return val_wrapper

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """Combine normalizers from all datasets."""
        # Get normalizers from all datasets
        normalizers = [
            dataset.get_normalizer(**kwargs) for dataset in self.datasets]

        if len(normalizers) == 1:
            return normalizers[0]

        # Combine normalizers using weighted averaging
        combined_normalizer = LinearNormalizer()

        # Get all keys from the first normalizer
        first_normalizer = normalizers[0]

        for key in first_normalizer.params_dict.keys():
            # Collect statistics from all normalizers for this key
            means = []
            stds = []
            weights = []

            for i, normalizer in enumerate(normalizers):
                field_normalizer = normalizer[key]
                # Access the parameters directly from params_dict
                params = field_normalizer.params_dict
                means.append(params['offset'])
                stds.append(1.0 / params['scale'])

                # Weight by dataset size * sampling ratio
                dataset_weight = (
                    self.dataset_lengths[i] * self.sampling_ratios[i])
                weights.append(dataset_weight)

            # Convert to tensors for computation - will error if lists are empty
            means = torch.stack(means)
            stds = torch.stack(stds)
            weights = torch.tensor(weights, dtype=torch.float32)

            # Normalize weights
            weights = weights / weights.sum()

            # Compute weighted mean and std
            combined_mean = (means * weights.unsqueeze(-1)).sum(dim=0)
            combined_std = (stds * weights.unsqueeze(-1)).sum(dim=0)

            # Create combined normalizer for this key
            # Use the first normalizer's input_stats as a template
            first_stats = normalizers[0][key].params_dict['input_stats']
            combined_normalizer[key] = SingleFieldLinearNormalizer.create_manual(
                scale=1.0 / combined_std,
                offset=combined_mean,
                input_stats_dict=dict(first_stats)
            )

        return combined_normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Concatenate actions from all datasets."""
        all_actions = []
        for dataset in self.datasets:
            all_actions.append(dataset.get_all_actions())
        return torch.cat(all_actions, dim=0)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # For direct indexing, use the provided index
        dataset_idx, local_idx = self._global_to_local_index(idx)
        return self.datasets[dataset_idx][local_idx]

    def sample_weighted(self) -> Dict[str, torch.Tensor]:
        """Sample using the weighted sampler."""
        # Get a weighted sample index
        sampled_indices = list(self.weighted_sampler)
        global_idx = sampled_indices[0]  # Take first sample

        # Convert to local index and get sample
        dataset_idx, local_idx = self._global_to_local_index(global_idx)
        return self.datasets[dataset_idx][local_idx]
