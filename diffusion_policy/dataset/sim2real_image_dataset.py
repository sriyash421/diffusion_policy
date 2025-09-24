from typing import Dict
import json
import hashlib
import shutil
from filelock import FileLock
from omegaconf import OmegaConf
import torch
import numpy as np
import copy
import zarr
import os
from threadpoolctl import threadpool_limits
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


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
    ):
        super().__init__()
        assert os.path.isdir(dataset_path)
        #debug printing stuff
        print("shape_meta: ", shape_meta)
        # Load data and create replay buffer
        self.replay_buffer = self._create_replay_buffer_from_zarr(dataset_path, use_cache=use_cache)

        # replay_buffer = None
        # if use_cache:
        #     # fingerprint shape_meta
        #     shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
        #     shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
        #     cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
        #     cache_lock_path = cache_zarr_path + '.lock'
        #     print('Acquiring lock on cache.')
        #     with FileLock(cache_lock_path):
        #         if not os.path.exists(cache_zarr_path):
        #             # cache does not exists
        #             try:
        #                 print('Cache does not exist. Creating!')
        #                 replay_buffer = _get_replay_buffer( #doesn't exist in this file, so I've taken it from real_pusht_image_dataset.py
        #                     dataset_path=dataset_path,
        #                     shape_meta=shape_meta,
        #                     store=zarr.MemoryStore()
        #                 )
        #                 print('Saving cache to disk.')
        #                 with zarr.ZipStore(cache_zarr_path) as zip_store:
        #                     replay_buffer.save_to_store(
        #                         store=zip_store
        #                     )
        #             except Exception as e:
        #                 shutil.rmtree(cache_zarr_path)
        #                 raise e
        #         else:
        #             print('Loading cached ReplayBuffer from Disk.')
        #             with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
        #                 replay_buffer = ReplayBuffer.copy_from_store(
        #                     src_store=zip_store, store=zarr.MemoryStore())
        #             print('Loaded!')
        # else:
        #     replay_buffer = _get_replay_buffer(
        #         dataset_path=dataset_path,
        #         shape_meta=shape_meta,
        #         store=zarr.MemoryStore()
        #     )

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

    def _create_replay_buffer_from_zarr(self, dataset_path, use_cache=False):
        """Create replay buffer from zarr data"""
        z = zarr.open(dataset_path, mode='r')
        # before creating obs_group, create an obs with last arm action and last gripper action
        # z[data][obs] = [last_arm_action, last_gripper_action] each with shape of 6
        #THE PROBLEM IS HERE
        # if (z['data']['obs'] is None):
        #     #midify the zarr so that it has all the obs
        #     call the read-data_conversion.py function real_data_to_replay_buffer
        obs_group = z['data']['obs']      #obs does not exist in zarr data
        action_arr = z['data']['actions'] #change to action not actions
        reward_arr = z['data']['rewards'] #rewards does not exist in zarr data
        episode_ends = z['meta']['episode_ends'] #good, it exists 

        # Create replay buffer
        replay_buffer = ReplayBuffer.create_empty_numpy()

        # Add observations
        for key in obs_group.keys():
            if use_cache:
                replay_buffer.root['data'][key] = obs_group[key][:]
            else:
                replay_buffer.root['data'][key] = obs_group[key]

        # Add actions
        if use_cache:
            replay_buffer.root['data']['action'] = action_arr[:]
        else:
            replay_buffer.root['data']['action'] = action_arr

        # Add rewards
        if use_cache:
            replay_buffer.root['data']['reward'] = reward_arr[:]
        else:
            replay_buffer.root['data']['reward'] = reward_arr

        # Add episode metadata
        if use_cache:
            replay_buffer.root['meta']['episode_ends'] = episode_ends[:]
        else:
            replay_buffer.root['meta']['episode_ends'] = episode_ends

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
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer['action'])
        # obs
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer[key])
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
        reward = data['reward'].astype(np.float32)

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action),
            'reward': torch.from_numpy(reward)
        }
        return torch_data
    
    #take from real_pusht_image_dataset.py
    # def _get_replay_buffer(dataset_path, shape_meta, store):
    #     # parse shape from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer
    #     rgb_keys = list()
    #     lowdim_keys = list()
    #     out_resolutions = dict()
    #     lowdim_shapes = dict()
    #     obs_shape_meta = shape_meta['obs']
    #     for key, attr in obs_shape_meta.items():
    #         type = attr.get('type', 'low_dim')
    #         shape = tuple(attr.get('shape'))
    #         if type == 'rgb':
    #             rgb_keys.append(key)
    #             c,h,w = shape
    #             out_resolutions[key] = (w,h)
    #         elif type == 'low_dim':
    #             lowdim_keys.append(key)
    #             lowdim_shapes[key] = tuple(shape)
    #             if 'pose' in key:
    #                 assert tuple(shape) in [(2,),(6,)]
        
    #     action_shape = tuple(shape_meta['action']['shape'])
    #     # assert action_shape in [(2,),(6,)]

    #     # load data
    #     cv2.setNumThreads(1)
    #     with threadpool_limits(1):
    #         replay_buffer = real_data_to_replay_buffer(
    #             dataset_path=dataset_path,
    #             out_store=store,
    #             out_resolutions=out_resolutions,
    #             lowdim_keys=lowdim_keys + ['action'],
    #             image_keys=rgb_keys
    #         )

    #     # transform lowdim dimensions
    #     if action_shape == (2,):
    #         # 2D action space, only controls X and Y
    #         zarr_arr = replay_buffer['action']
    #         zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])
        
    #     for key, shape in lowdim_shapes.items():
    #         if 'pose' in key and shape == (2,):
    #             # only take X and Y
    #             zarr_arr = replay_buffer[key]
    #             zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])

    #     return replay_buffer