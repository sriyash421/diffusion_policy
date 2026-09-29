"""PushT keypoint observations on the SAME split manifest the image arms train on.

WHY THIS EXISTS AS A SIBLING RATHER THAN A FLAG. The repo already has two PushT datasets and
they answer different questions:

  PushTImageDataset    manifest-driven, 3-way (train/val/test), records what it resolved into
                       splits.json, supports whole-episode sequences -- but it loads the
                       2.84 GB `img` array and its obs contract is image + the verifier's two
                       keys, asserted as image-only at policy construction.
  PushTLowdimDataset   emits exactly the 20-d keypoint observation this arm wants -- but it
                       partitions with `val_ratio`/`downsample_mask`, has NO test split, no
                       reset states, no `get_split_indices` and no sequence mode. Using it
                       would put the LSTM arm on different episodes from every other arm,
                       which is precisely what this arm must not do.

So this file is the first dataset's machinery with the second's observation. The partition
helpers are IMPORTED from pusht_image_dataset rather than re-derived: one definition of "which
episodes are train" is the invariant that the manifest exists to enforce, and a second
implementation agreeing by convention is the bug class the manifest replaced.

THE OBSERVATION. `{keypoint: (T, 9, 2), agent_pos: (T, 2)}`, the same composition as
PushTLowdimDataset (zarr `keypoint` + `agent_pos`), flattened to 20 by FlattenObsEncoder. Raw
512-px arena units, which is what PushTKeypointsEnv emits, so a normalizer fitted here is
valid at rollout.

NO VISIBILITY MASK. recurrent_ppo's keypoint arm is 40-d: the 20 values, then their visibility
as {-1,+1}, because occlusion is an axis it studies. The demonstrations are fully visible, so
on this arm those 20 dims would be a hardcoded +1 -- twenty dead inputs asserting a fact that
is never false. This arm trains and evaluates clean in every observation mode, so the mask has
nothing to say and is not emitted.
"""
from typing import Dict

import copy
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
# ONE definition of the partition, shared with the image arms. Importing rather than copying
# is what makes "did these two arms train on the same episodes?" a checksum comparison.
from diffusion_policy.dataset.pusht_image_dataset import (
    SPLIT_NAMES,
    episode_ends_checksum,
    episode_frame_mask,
    get_episode_init_states,
    load_split_manifest,
    masks_from_manifest,
    split_checksum,
)

# What the policy encodes, via shape_meta. Unlike the image arm there is nothing wider here:
# this arm has no verifier, so no key rides along for one.
POLICY_OBS_KEYS = ('keypoint', 'agent_pos')
# What LinearNormalizer fits. All raw arena pixels, so 'limits' mode maps each to [-1, 1].
NORMALIZER_KEYS = ('action', 'keypoint', 'agent_pos')


class PushTKeypointDataset(BaseImageDataset):
    """Subclasses BaseImageDataset because TrainMLPImageWorkspace asserts that type.

    The name is nominal -- the base class declares a dict-of-keys obs, which is what this
    emits; there is no image anywhere in this class.
    """

    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            n_test_episodes=50,
            n_val_episodes=0,
            n_train_episodes=None,
            split='train',
            return_sequences=False,
            split_file=None,
            ):
        super().__init__()
        assert split in ('train', 'val', 'test')
        # REQUIRED, exactly as PushTImageDataset requires it. The alternative -- deriving the
        # partition from `seed` -- is a second independent answer to "which episodes are
        # train?" and is what this whole arm must not have, since its only claim to
        # comparability is training on the episodes the UNet BC arm trained on.
        if split_file is None:
            raise ValueError(
                'PushTKeypointDataset requires split_file: the split manifest is the only '
                'source of truth for the partition, and this arm exists to be compared '
                'against arms that read the same one. Generate one with '
                'scripts/dump_pusht_splits.py.')
        if return_sequences:
            assert pad_before == 0 and pad_after == 0 and horizon >= 100

        # NOT the image dataset's key list: `img` is 2.84 GB and this arm never reads a pixel.
        # block_pos is loaded only so get_episode_init_states can seed env resets.
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['keypoint', 'agent_pos', 'block_pos', 'action'])

        self.split_file = split_file
        self._manifest = load_split_manifest(
            split_file,
            episode_ends=self.replay_buffer.episode_ends[:],
            expected_counts={
                'test': n_test_episodes,
                'val': n_val_episodes if n_val_episodes else None,
                'train': n_train_episodes,
            })
        train_mask, val_mask, test_mask = masks_from_manifest(
            self._manifest, self.replay_buffer.n_episodes)
        self.train_pool = train_mask
        self.val_pool = val_mask
        self.test_pool = test_mask
        # No budget on top: the manifest already names the exact train episodes, so a smaller
        # budget is a different file rather than a runtime subsample of this one.
        self.train_used = train_mask

        self.zarr_path = zarr_path
        self.seed = seed

        episode_mask = {
            'train': self.train_used,
            'val': self.val_pool,
            'test': self.test_pool,
        }[split]

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
        # Read as a plain attribute by TrainMLPImageWorkspace to choose the collate fn, with
        # no getattr default, so it must exist.
        self.return_sequences = return_sequences

    def _split_copy(self, episode_mask):
        """A shallow copy whose sampler is restricted to episode_mask."""
        split_set = copy.copy(self)
        split_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=episode_mask,
            return_sequences=self.return_sequences)
        split_set.episode_mask = episode_mask
        return split_set

    def get_validation_dataset(self):
        return self._split_copy(self.val_pool)

    def get_test_dataset(self):
        return self._split_copy(self.test_pool)

    def get_test_reset_states(self):
        return get_episode_init_states(self.replay_buffer, self.test_pool)

    def get_val_reset_states(self):
        return get_episode_init_states(self.replay_buffer, self.val_pool)

    def get_split_indices(self):
        """The partition this dataset resolved, in manifest form, for <run_dir>/splits.json.

        Written by BaseWorkspace.write_splits, which reads 'checksum' and the three index
        lists and raises on a resume whose partition changed. Reports the whole partition,
        not the split this object samples -- a validation copy returns the same three lists.
        """
        episode_ends = np.asarray(self.replay_buffer.episode_ends[:])
        masks = {'train': self.train_used, 'val': self.val_pool, 'test': self.test_pool}
        out = {
            'generated_by': type(self).__name__ + '.get_split_indices',
            'zarr_path': str(self.zarr_path),
            'split_file': str(self.split_file),
            'n_episodes': int(len(episode_ends)),
            'episode_ends_checksum': episode_ends_checksum(episode_ends),
        }
        out['derivation'] = self._manifest.get('derivation')
        for name in SPLIT_NAMES:
            out[name] = [int(i) for i in np.nonzero(masks[name])[0]]
            out[f'{name}_frames'] = int(
                episode_frame_mask(episode_ends, masks[name]).sum())
        out['checksum'] = split_checksum(out['train'], out['val'], out['test'])
        return out

    def get_video_episode_idxs(self, split, n=10):
        mask = {
            'train': self.train_used,
            'val': self.val_pool,
            'test': self.test_pool,
        }[split]
        return np.nonzero(mask)[0][:n]

    def get_normalizer(self, mode='limits', **kwargs):
        """Fit on the train episodes only -- never on val or test."""
        frames = episode_frame_mask(
            self.replay_buffer.episode_ends[:], self.train_used)
        source = {
            'action': lambda: self.replay_buffer['action'][frames],
            # last_n_dims=1 normalizes per final-axis component, so the (N, 9, 2) keypoint
            # array is fitted as two x/y ranges shared by all nine points -- which is what
            # keeps the nine points on one spatial scale instead of eighteen private ones.
            'keypoint': lambda: self.replay_buffer['keypoint'][frames],
            'agent_pos': lambda: self.replay_buffer['agent_pos'][frames],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data={k: source[k]() for k in NORMALIZER_KEYS},
                       last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        return {
            'obs': {
                'keypoint': sample['keypoint'].astype(np.float32),      # T, 9, 2
                'agent_pos': sample['agent_pos'].astype(np.float32),    # T, 2
            },
            'action': sample['action'].astype(np.float32),              # T, 2
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        return dict_apply(self._sample_to_data(sample), torch.from_numpy)
