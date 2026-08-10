from typing import Dict
import hashlib
import json
import pathlib
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


def get_split_masks_3way(n_episodes, n_test_episodes, n_val_episodes, seed,
                         n_train_episodes=None):
    """Seeded, recreatable 3-way split into train / val / test.

    The permutation is taken with the SAME ``seed`` as ``get_split_masks`` and the test
    episodes are drawn FIRST (``perm[:n_test]``), so the held-out test set is byte-for-byte
    identical to the 2-way split -- eval_bon / the runner stay consistent. Val is the next
    ``n_val`` episodes; the remainder is the train pool (optionally shrunk to
    ``n_train_episodes``).
    """
    assert n_test_episodes + n_val_episodes < n_episodes, (
        f'{n_test_episodes} test + {n_val_episodes} val >= {n_episodes} episodes')
    rng = np.random.default_rng(seed=seed)
    perm = rng.permutation(n_episodes)
    test_idxs = perm[:n_test_episodes]
    val_idxs = perm[n_test_episodes:n_test_episodes + n_val_episodes]
    train_idxs = perm[n_test_episodes + n_val_episodes:]
    if n_train_episodes is not None:
        assert n_train_episodes <= len(train_idxs), (
            f'{n_train_episodes} > {len(train_idxs)} non-test/val episodes')
        train_idxs = train_idxs[:n_train_episodes]
    train_mask = np.zeros(n_episodes, dtype=bool)
    val_mask = np.zeros(n_episodes, dtype=bool)
    test_mask = np.zeros(n_episodes, dtype=bool)
    train_mask[train_idxs] = True
    val_mask[val_idxs] = True
    test_mask[test_idxs] = True
    return train_mask, val_mask, test_mask


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


# ---------------------------------------------------------------------------
# Split manifest.
#
# The three splits used to be DERIVED at runtime, independently, in three places (this
# dataset, PushTSearchImageRunner, eval_search_pusht) from five keys: seed,
# n_test_episodes, n_val_episodes, n_train_episodes, train_ratio. Nothing recorded which
# episodes a checkpoint had actually been trained on, so changing any one of those keys
# silently repartitioned the data -- which is exactly what happened when n_val_episodes
# went 10 -> 30 and the training budget fell 29 -> 25 episodes unnoticed.
#
# A manifest fixes that: the partition is generated ONCE by scripts/dump_pusht_splits.py,
# committed, and read by everything. Derivation-from-seed remains as the generator and as
# the `split_file: null` fallback, but it is no longer the source of truth.
# ---------------------------------------------------------------------------

SPLIT_NAMES = ('train', 'val', 'test')


def episode_ends_checksum(episode_ends) -> str:
    """Fingerprint of the dataset an episode index refers to.

    Episode index 7 only means something relative to a particular zarr, and
    README_pusht.md documents a heredoc that MUTATES the zarr in place to add
    agent_pos/block_pos -- so "same indices, different frames" is a real failure mode.
    Hashing episode_ends pins the episode boundaries the indices were drawn against.
    """
    arr = np.ascontiguousarray(np.asarray(episode_ends, dtype=np.int64))
    return hashlib.md5(arr.tobytes()).hexdigest()


def split_checksum(train, val, test) -> str:
    """Fingerprint of the partition itself, order-independent."""
    payload = json.dumps(
        {name: sorted(int(i) for i in idxs)
         for name, idxs in zip(SPLIT_NAMES, (train, val, test))},
        sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def build_split_manifest(episode_ends, zarr_path, seed, n_test_episodes,
                         n_val_episodes, n_train_episodes=None):
    """Derive the 3-way split from the seed and describe it fully.

    This is the ONLY place the manifest is produced; scripts/dump_pusht_splits.py calls it
    and writes the result. `n_train_episodes` slices a PREFIX of the permuted train pool,
    so budgets are nested (a 25-episode set is a strict subset of a 100-episode one) and
    the selection does not depend on the pool size the way downsample_mask's rng.choice
    does.

    Takes ``episode_ends`` rather than a ReplayBuffer so the manifest can be generated
    without loading the ~2.8 GB image array.
    """
    episode_ends = np.asarray(episode_ends)
    n_episodes = len(episode_ends)
    train_mask, val_mask, test_mask = get_split_masks_3way(
        n_episodes=n_episodes,
        n_test_episodes=n_test_episodes,
        n_val_episodes=n_val_episodes,
        seed=seed,
        n_train_episodes=n_train_episodes)
    masks = dict(zip(SPLIT_NAMES, (train_mask, val_mask, test_mask)))
    manifest = {
        'generated_by': 'scripts/dump_pusht_splits.py',
        'zarr_path': str(zarr_path),
        'n_episodes': int(n_episodes),
        'episode_ends_checksum': episode_ends_checksum(episode_ends),
        'derivation': {
            'method': 'get_split_masks_3way; train = perm[n_test+n_val:][:n_train]',
            'seed': int(seed),
            'n_test_episodes': int(n_test_episodes),
            'n_val_episodes': int(n_val_episodes),
            'n_train_episodes': (None if n_train_episodes is None
                                 else int(n_train_episodes)),
        },
    }
    for name in SPLIT_NAMES:
        manifest[name] = [int(i) for i in np.nonzero(masks[name])[0]]
        manifest[f'{name}_frames'] = int(
            episode_frame_mask(episode_ends, masks[name]).sum())
    manifest['checksum'] = split_checksum(
        manifest['train'], manifest['val'], manifest['test'])
    return manifest


def load_split_manifest(split_file, episode_ends=None, expected_counts=None):
    """Read a manifest and validate it hard. Never merges -- disagreement is an error.

    Args:
        split_file: path to the manifest. Resolved against the original working directory
            (hydra may have chdir'd), same as checkpoint.pretrained_ckpt_path.
        episode_ends: if given, the manifest's n_episodes and episode_ends_checksum must
            match this dataset -- otherwise the indices refer to a different zarr.
        expected_counts: optional {'train'|'val'|'test': n} from the config. Each present
            entry must equal the corresponding list length. A config that disagrees with
            the manifest is a bug, not a preference, so this raises rather than choosing.
    """
    import hydra.utils
    path = pathlib.Path(hydra.utils.to_absolute_path(str(split_file)))
    if not path.is_file():
        raise FileNotFoundError(f'split_file not found: {path}')
    manifest = json.loads(path.read_text())

    for name in SPLIT_NAMES:
        if name not in manifest:
            raise ValueError(f'{path}: manifest is missing the "{name}" split')
    idxs = {name: [int(i) for i in manifest[name]] for name in SPLIT_NAMES}

    # disjoint, no duplicates
    seen = set()
    for name in SPLIT_NAMES:
        dup = [i for i in idxs[name] if i in seen]
        if dup:
            raise ValueError(
                f'{path}: episode(s) {dup[:5]} appear in "{name}" and another split')
        seen.update(idxs[name])

    stored = manifest.get('checksum')
    actual = split_checksum(idxs['train'], idxs['val'], idxs['test'])
    if stored is not None and stored != actual:
        raise ValueError(
            f'{path}: checksum {stored} does not match its own index lists ({actual}). '
            f'The file was hand-edited; regenerate with scripts/dump_pusht_splits.py.')

    if episode_ends is not None:
        episode_ends = np.asarray(episode_ends)
        n_episodes = len(episode_ends)
        if int(manifest.get('n_episodes', n_episodes)) != n_episodes:
            raise ValueError(
                f'{path}: manifest was built for {manifest["n_episodes"]} episodes but '
                f'this zarr has {n_episodes}.')
        out_of_range = [i for i in seen if not (0 <= i < n_episodes)]
        if out_of_range:
            raise ValueError(
                f'{path}: episode index(es) {sorted(out_of_range)[:5]} out of range for '
                f'{n_episodes} episodes.')
        stored_ck = manifest.get('episode_ends_checksum')
        actual_ck = episode_ends_checksum(episode_ends)
        if stored_ck is not None and stored_ck != actual_ck:
            raise ValueError(
                f'{path}: episode_ends_checksum {stored_ck} != {actual_ck} for the zarr on '
                f'disk. The episode boundaries changed, so these indices no longer refer '
                f'to the same frames. Regenerate the manifest, and expect every previously '
                f'reported number to be measured on different data.')

    for name, expected in (expected_counts or {}).items():
        if expected is None:
            continue
        if len(idxs[name]) != int(expected):
            raise ValueError(
                f'{path}: config asks for {expected} {name} episodes but the manifest has '
                f'{len(idxs[name])}. The manifest is the source of truth -- either fix the '
                f'config or regenerate with scripts/dump_pusht_splits.py.')
    return manifest


def masks_from_manifest(manifest, n_episodes):
    """(train, val, test) boolean masks from a validated manifest."""
    out = list()
    for name in SPLIT_NAMES:
        mask = np.zeros(n_episodes, dtype=bool)
        mask[np.asarray(manifest[name], dtype=int)] = True
        out.append(mask)
    return tuple(out)


class PushTImageDataset(BaseImageDataset):
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
            max_train_episodes=None,
            train_ratio=None,
            return_sequences=False,
            split_file=None
            ):

        super().__init__()
        assert split in ('train', 'val', 'test')
        assert train_ratio is None or 0 < train_ratio <= 1, (
            f'train_ratio must be in (0, 1], got {train_ratio}')
        if return_sequences:
            assert pad_before == 0 and pad_after == 0 and horizon >= 100

        # agent_pos (2d) and block_pos (3d) are separate arrays in the zarr. block_pos is
        # loaded because get_episode_init_states needs the full state to seed env resets,
        # and because `feedback` is derived from it -- but it is never returned in a
        # sample's obs dict (see _sample_to_data).
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'agent_pos', 'block_pos', 'action'])

        self.split_file = split_file
        self._manifest = None
        if split_file is not None:
            # The manifest is the SOURCE OF TRUTH: it names the exact episodes, and the
            # count keys below are validated against it rather than generating anything.
            # This is what stops a change to n_val_episodes (or to train_ratio, or to the
            # seed) from silently repartitioning the data underneath a running experiment.
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
            # train_ratio / max_train_episodes would re-subsample what the manifest already
            # pinned, so the budget would stop being a property of the file. Refuse rather
            # than silently applying one on top of the other.
            if train_ratio is not None or max_train_episodes is not None:
                raise ValueError(
                    'train_ratio / max_train_episodes cannot be combined with split_file: '
                    'the manifest already names the exact train episodes. Set them to null '
                    'and regenerate the manifest if you want a different budget.')
            self.train_used = train_mask
        else:
            # Legacy: derive from the seed. Kept so the 2-way configs and any dataset
            # without a manifest behave exactly as before.
            #
            # With n_val_episodes>0 this is a seeded, recreatable 3-way split
            # (train/val/test); the test set is identical to the legacy 2-way split for the
            # same seed. With n_val_episodes==0 val falls back to the test set (legacy).
            if n_val_episodes > 0:
                train_mask, val_mask, test_mask = get_split_masks_3way(
                    n_episodes=self.replay_buffer.n_episodes,
                    n_test_episodes=n_test_episodes,
                    n_val_episodes=n_val_episodes,
                    seed=seed,
                    n_train_episodes=n_train_episodes)
            else:
                train_mask, test_mask = get_split_masks(
                    n_episodes=self.replay_buffer.n_episodes,
                    n_test_episodes=n_test_episodes,
                    seed=seed,
                    n_train_episodes=n_train_episodes)
                val_mask = test_mask
            self.train_pool = train_mask
            self.val_pool = val_mask
            self.test_pool = test_mask

            # train_ratio keeps that fraction of the train episodes (0.2 -> 20%). NOTE this
            # is a fraction of whatever is LEFT after test and val are taken, so changing
            # either split size changes the training budget -- the drift a manifest exists
            # to prevent. The test/val splits are never subsampled.
            max_n = max_train_episodes
            if train_ratio is not None:
                ratio_n = max(1, int(round(train_ratio * train_mask.sum())))
                max_n = ratio_n if max_n is None else min(max_n, ratio_n)
            # the episodes actually trained on; also the only data the normalizer sees,
            # so a reduced train_ratio really does mean less data used.
            self.train_used = downsample_mask(mask=train_mask, max_n=max_n, seed=seed)

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
        self.return_sequences = return_sequences

    def _split_copy(self, episode_mask):
        """A shallow copy of this dataset whose sampler is restricted to episode_mask."""
        split_set = copy.copy(self)
        split_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=episode_mask,
            return_sequences=self.return_sequences
            )
        split_set.episode_mask = episode_mask
        return split_set

    def get_validation_dataset(self):
        """Validation during training runs on the held-out val episodes.

        With a 3-way split val is its own held-out set; with the legacy 2-way split
        val_pool == test_pool (validation runs on the test episodes, as before).
        """
        return self._split_copy(self.val_pool)

    def get_test_dataset(self):
        """A dataset restricted to the held-out test episodes."""
        return self._split_copy(self.test_pool)

    def get_test_reset_states(self):
        """Env reset states for the held-out test episodes."""
        return get_episode_init_states(self.replay_buffer, self.test_pool)

    def get_val_reset_states(self):
        """Env reset states for the held-out val episodes."""
        return get_episode_init_states(self.replay_buffer, self.val_pool)

    def get_split_indices(self):
        """The partition this dataset actually resolved, in manifest form.

        Written to <run_dir>/splits.json by BaseWorkspace.write_splits so a run directory
        records which episodes its checkpoints were trained on -- whether the split came
        from a manifest or from the seed. ``train`` is ``train_used`` (post-budget), i.e.
        the episodes really trained on, not the pool they were drawn from.
        """
        episode_ends = np.asarray(self.replay_buffer.episode_ends[:])
        masks = {'train': self.train_used, 'val': self.val_pool, 'test': self.test_pool}
        out = {
            'generated_by': type(self).__name__ + '.get_split_indices',
            'zarr_path': str(self.zarr_path),
            'split_file': None if self.split_file is None else str(self.split_file),
            'n_episodes': int(len(episode_ends)),
            'episode_ends_checksum': episode_ends_checksum(episode_ends),
        }
        if self._manifest is not None:
            out['derivation'] = self._manifest.get('derivation')
        else:
            out['derivation'] = {
                'method': 'derived at runtime from the seed (no split_file)',
                'seed': int(self.seed),
            }
        for name in SPLIT_NAMES:
            out[name] = [int(i) for i in np.nonzero(masks[name])[0]]
            out[f'{name}_frames'] = int(
                episode_frame_mask(episode_ends, masks[name]).sum())
        out['checksum'] = split_checksum(out['train'], out['val'], out['test'])
        return out

    def get_video_episode_idxs(self, split, n=10):
        """First ``n`` episode indices of a split (seeded/stable) for demo videos."""
        mask = {
            'train': self.train_used,
            'val': self.val_pool,
            'test': self.test_pool,
        }[split]
        return np.nonzero(mask)[0][:n]

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

        # obs is exactly shape_meta['obs'] -- no extra keys. block_pos is NOT emitted:
        # feedback is an exact, invertible function of it (see
        # pusht_verifier.block_pose_from_feedback), so anything needing the block pose
        # reconstructs it from feedback. That keeps the obs dict normalizer-complete (an
        # unnormalized extra key KeyErrors in every policy that does not filter obs) and
        # makes the verifier's train-time and eval-time resets bit-identical.
        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
                'feedback': feedback, # T, 16 (goal-relative, a policy input)
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
