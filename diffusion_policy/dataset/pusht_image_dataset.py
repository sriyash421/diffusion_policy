from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.env.pusht.feedback_util import compute_feedback_from_pose


def get_split_masks(n_episodes, n_test_episodes, seed, n_train_episodes=None):
    """Hold out n_test_episodes at random; every remaining episode is the train set.

    There are only two splits: train and test. Passing n_train_episodes shrinks the
    train set instead of using the whole remainder.
    """
    assert n_test_episodes < n_episodes, (
        f'{n_test_episodes} test >= {n_episodes} episodes')
    rng = np.random.default_rng(seed=seed)
    perm = rng.permutation(n_episodes)
    test_idxs = perm[:n_test_episodes]
    train_idxs = perm[n_test_episodes:]
    if n_train_episodes is not None:
        assert n_train_episodes <= len(train_idxs), (
            f'{n_train_episodes} > {len(train_idxs)} non-test episodes')
        train_idxs = train_idxs[:n_train_episodes]
    train_mask = np.zeros(n_episodes, dtype=bool)
    test_mask = np.zeros(n_episodes, dtype=bool)
    train_mask[train_idxs] = True
    test_mask[test_idxs] = True
    return train_mask, test_mask


def episode_frame_mask(episode_ends, episode_mask):
    """Expand a per-episode boolean mask to a per-frame boolean mask."""
    frame_mask = np.zeros(episode_ends[-1], dtype=bool)
    starts = np.concatenate([[0], episode_ends[:-1]])
    for keep, s, e in zip(episode_mask, starts, episode_ends):
        if keep:
            frame_mask[s:e] = True
    return frame_mask


def get_episode_init_states(replay_buffer, episode_mask):
    """Initial (agent_pos, block_pos) of each selected episode, as env reset states.

    Returns (n_selected, 5) laid out as the PushT env's state:
    [agent_x, agent_y, block_x, block_y, block_angle].
    """
    ends = np.asarray(replay_buffer.episode_ends[:])
    starts = np.concatenate([[0], ends[:-1]])[np.asarray(episode_mask)]
    return np.concatenate([
        np.asarray(replay_buffer['agent_pos'])[starts],
        np.asarray(replay_buffer['block_pos'])[starts]
    ], axis=-1).astype(np.float64)


class PushTImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            n_test_episodes=50,
            n_train_episodes=None,
            split='train',
            max_train_episodes=None,
            train_ratio=None,
            return_sequences=False
            ):

        super().__init__()
        assert split in ('train', 'test')
        assert train_ratio is None or 0 < train_ratio <= 1, (
            f'train_ratio must be in (0, 1], got {train_ratio}')
        if return_sequences:
            assert pad_before == 0 and pad_after == 0 and horizon >= 100

        # agent_pos (2d) and block_pos (3d) are separate arrays in the zarr.
        # block_pos is always loaded because env resets need the full state, but it
        # is deliberately kept out of the policy obs (image + agent_pos only).
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'agent_pos', 'block_pos', 'action'])

        # two disjoint splits covering every episode: train and test
        train_mask, test_mask = get_split_masks(
            n_episodes=self.replay_buffer.n_episodes,
            n_test_episodes=n_test_episodes,
            seed=seed,
            n_train_episodes=n_train_episodes)
        self.train_pool = train_mask
        self.test_pool = test_mask

        # train_ratio keeps that fraction of the train episodes (0.2 -> 20%).
        # The test split is never subsampled, so evals stay comparable across ratios.
        max_n = max_train_episodes
        if train_ratio is not None:
            ratio_n = max(1, int(round(train_ratio * train_mask.sum())))
            max_n = ratio_n if max_n is None else min(max_n, ratio_n)
        # the episodes actually trained on; also the only data the normalizer sees,
        # so a reduced train_ratio really does mean less data used.
        self.train_used = downsample_mask(mask=train_mask, max_n=max_n, seed=seed)

        episode_mask = self.train_used if split == 'train' else test_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=episode_mask,
            return_sequences=return_sequences)
        self.episode_mask = episode_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.return_sequences = return_sequences

    def get_validation_dataset(self):
        """Validation during training runs on the held-out test episodes."""
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.test_pool,
            return_sequences=self.return_sequences
            )
        val_set.episode_mask = self.test_pool
        return val_set

    def get_test_reset_states(self):
        """Env reset states for the 50 held-out test episodes."""
        return get_episode_init_states(self.replay_buffer, self.test_pool)

    def get_normalizer(self, mode='limits', **kwargs):
        # fit on the train episodes actually used (after train_ratio), never on the
        # held-out test episodes
        frames = episode_frame_mask(
            self.replay_buffer.episode_ends[:], self.train_used)
        data = {
            'action': self.replay_buffer['action'][frames],
            'agent_pos': self.replay_buffer['agent_pos'][frames],
            # feedback is a goal-relative transform of block_pos (a valid policy input);
            # block_pos itself is reset-only and never normalized.
            'feedback': compute_feedback_from_pose(
                self.replay_buffer['block_pos'][frames]),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['agent_pos'].astype(np.float32)
        block_pos = sample['block_pos'].astype(np.float32)  # T, 3
        image = np.moveaxis(sample['img'],-1,1)/255
        feedback = compute_feedback_from_pose(block_pos)  # T, 16

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
                'feedback': feedback, # T, 16 (goal-relative, a policy input)
                # block_pos rides in obs only as a reset carrier: it is popped in the
                # policy before encode/normalize and kept out of shape_meta.
                'block_pos': block_pos, # T, 3
            },
            'action': sample['action'].astype(np.float32), # T, 2
            # all-ones so the shared get_collate_fn (which requires expert_mask) works;
            # every expert step is a valid supervision target.
            'expert_mask': np.ones((agent_pos.shape[0], 1), dtype=np.float32),
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('~/Projects/gym-pusht/data/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)
