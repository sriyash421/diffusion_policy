from typing import Dict
import hashlib
import json
import pathlib
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
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


# ---------------------------------------------------------------------------------------
# THE OBS CONTRACT. Which keys exist, and what each is for. These were previously implicit:
# _sample_to_data and get_normalizer each hardcoded their own list, independent of
# shape_meta and of each other, so the three could disagree forever without anything
# noticing. PushTSearchMixin asserts shape_meta.obs against POLICY_OBS_KEYS at policy
# construction, which is what makes "the observation is image-only" a checked invariant
# rather than a property of the current default.
#
# The sample dict is deliberately WIDER than shape_meta: the verifier reads its two keys off
# the raw obs dict, never through the encoder.
POLICY_OBS_KEYS = ('image',)
# Emitted for PushTVerifier.rollout, which resets a pymunk sim to [agent_pos, feedback], and
# for PushTSearchMixin._normalize_value. Never encoded, never in shape_meta.
VERIFIER_OBS_KEYS = ('agent_pos', 'feedback')
# What LinearNormalizer fits. 'image' is added separately with a fixed [0,1] -> [-1,1] map.
# 'feedback' is fitted ONLY so _normalize_value has a scale to rescale the verifier's
# context scalar by -- it is not a policy input and is not normalized on the sample path.
NORMALIZER_KEYS = ('action', 'agent_pos', 'feedback')


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
            return_sequences=False,
            split_file=None,
            obs_image_steps=None
            ):
        """``obs_image_steps``: how many image frames of each window to actually load.

        ``None`` loads the whole ``horizon`` (the original behaviour). Set it to
        ``n_obs_steps`` and only the frames the policy conditions on are read: every
        consumer of the ``image`` key slices ``[:, :n_obs_steps]`` before encoding
        (``_encode_obs_features`` on the search transformer and the Gaussian arm,
        ``_encode_images`` on the UNet arm under ``obs_as_global_cond``), so at
        horizon 16 / n_obs_steps 2 this reads, copies and transfers 8x less image data
        for a bit-identical result.
        """
        super().__init__()
        assert split in ('train', 'val', 'test')
        # REQUIRED since 2026-08-29. The alternative -- deriving the partition from `seed` at
        # runtime -- was a second, independent way to answer "which episodes are train?", and
        # its n_val_episodes==0 path silently ran validation on the TEST set. One code path to
        # the split means BC UNet, ST k=1, ST k=16 and every ladder arm provably share it.
        # `max_train_episodes` / `train_ratio` went with it: they existed only to subsample
        # what that branch produced, and the manifest already names the exact train episodes.
        if split_file is None:
            raise ValueError(
                'PushTImageDataset requires split_file: the split manifest is the only '
                'source of truth for the partition. Generate one with '
                'scripts/dump_pusht_splits.py.')
        if return_sequences:
            assert pad_before == 0 and pad_after == 0 and horizon >= 100

        # agent_pos (2d) and block_pos (3d) are separate arrays in the zarr. block_pos is
        # loaded because get_episode_init_states needs the full state to seed env resets,
        # and because `feedback` is derived from it -- but it is never returned in a
        # sample's obs dict (see _sample_to_data).
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'agent_pos', 'block_pos', 'action'])

        # FEEDBACK PRECOMPUTED ONCE, for all frames. It is an exact per-frame function of
        # block_pos (compute_feedback_from_pose broadcasts over leading dims), so computing
        # it here and slicing is identical to slicing and computing -- but the per-sample
        # version re-ran the keypoint trig for every window on every epoch. 25650 x 16
        # float32 is 1.6 MB. Registering it as a replay-buffer key lets the sampler slice
        # it like any other, including the episode-boundary padding.
        self.replay_buffer.data['feedback'] = compute_feedback_from_pose(
            self.replay_buffer['block_pos'])

        self.split_file = split_file
        # The manifest is the SOURCE OF TRUTH: it names the exact episodes, and the count
        # keys are validated against it rather than generating anything. This is what stops a
        # change to n_val_episodes, or to the seed, from silently repartitioning the data
        # underneath a running experiment.
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
        # No budget is applied on top: the manifest already names the exact train episodes,
        # so train_used IS train_pool. (n_demos selects WHICH manifest, so a smaller budget
        # is a different file, not a runtime subsample of this one.)
        self.train_used = train_mask

        self.zarr_path = zarr_path
        self.seed = seed

        episode_mask = {
            'train': self.train_used,
            'val': self.val_pool,
            'test': self.test_pool,
        }[split]

        # Sampler keys are named rather than left to default to every replay-buffer key:
        # block_pos is in the buffer for get_episode_init_states and for the feedback
        # precompute, but nothing reads it per sample any more (feedback replaced it), so
        # naming the keys keeps it out of the per-window slice.
        self.obs_image_steps = None if obs_image_steps is None else int(obs_image_steps)
        self._sampler_keys = ['img', 'agent_pos', 'feedback', 'action']
        self._key_first_k = ({} if self.obs_image_steps is None
                             else {'img': self.obs_image_steps})
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=self._sampler_keys,
            key_first_k=self._key_first_k,
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
            keys=self._sampler_keys,
            key_first_k=self._key_first_k,
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
        records which episodes its checkpoints were trained on. Since the manifest is the
        only source of the partition, this is a faithful copy of it plus the frame counts
        and a checksum -- which is what makes "did these two arms train on the same data?"
        a byte comparison rather than an argument.

        Note this reports the whole partition, not the split THIS object samples: a
        validation copy from get_validation_dataset() returns the same three lists, by
        design.
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
        """First ``n`` episode indices of a split (seeded/stable) for demo videos."""
        mask = {
            'train': self.train_used,
            'val': self.val_pool,
            'test': self.test_pool,
        }[split]
        return np.nonzero(mask)[0][:n]

    def get_normalizer(self, mode='limits', **kwargs):
        """Fit on the train episodes only -- never on val or test.

        Keys come from NORMALIZER_KEYS, which is a superset of what the policy encodes:
        'feedback' is fitted although it is not a policy input, because
        PushTSearchMixin._normalize_value reads its scale to rescale the verifier's context
        scalar to O(1). 'agent_pos' is fitted for the same reason the dataset still emits it
        -- the verifier path -- and costs one 2-d min/max.
        """
        frames = episode_frame_mask(
            self.replay_buffer.episode_ends[:], self.train_used)
        source = {
            'action': lambda: self.replay_buffer['action'][frames],
            'agent_pos': lambda: self.replay_buffer['agent_pos'][frames],
            # goal-relative transform of block_pos; block_pos itself is never normalized.
            # Read off the precompute rather than recomputed: compute_feedback_from_pose is
            # per-frame, so slicing it gives the same values as computing on the slice, and
            # one source keeps the fitted scale and the sampled key provably in step.
            'feedback': lambda: self.replay_buffer['feedback'][frames],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data={k: source[k]() for k in NORMALIZER_KEYS},
                       last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # No .astype(np.float32) on agent_pos / feedback: the zarr is already <f4 and the
        # precompute emits <f4, so those calls were full copies that changed nothing.
        agent_pos = sample['agent_pos']
        feedback = sample['feedback']  # T, 16 -- precomputed, see __init__
        # Truncated to the frames the policy actually encodes. Under obs_image_steps the
        # sampler only READ that many (the rest is key_first_k's NaN fill), so the slice
        # is what keeps the fill out of the batch entirely rather than normalizing and
        # then discarding it.
        img = sample['img']
        if self.obs_image_steps is not None:
            img = img[:self.obs_image_steps]
        image = np.moveaxis(img, -1, 1) / 255

        # POLICY_OBS_KEYS + VERIFIER_OBS_KEYS. The obs dict is WIDER than shape_meta on
        # purpose: agent_pos and feedback are here for PushTVerifier, which resets a pymunk
        # sim from them off the raw dict, and are not in shape_meta so the encoder never
        # reads them. block_pos is NOT emitted -- feedback is an exact, invertible function
        # of it (pusht_verifier.block_pose_from_feedback), so anything needing the block
        # pose reconstructs it, and the verifier's train-time and eval-time resets stay
        # bit-identical.
        data = {
            'obs': {
                'image': image, # To, 3, 96, 96 (T when obs_image_steps is None)
                'agent_pos': agent_pos, # T, 2   verifier only
                'feedback': feedback, # T, 16   verifier only (goal-relative)
            },
            'action': sample['action'].astype(np.float32), # T, 2
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
