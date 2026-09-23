from typing import Dict
import hashlib
import json
import pathlib
import torch
import numpy as np
import copy
from omegaconf import OmegaConf
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


def _resolve_goal_mask_noise(cfg):
    """Validate the goal-only corruption block. None (off) or {t, margin_px, exclude_block}.

    Validated rather than trusted: an unknown key here would be silently ignored, and
    `margin_px` is in IMAGE pixels (96px observation), not the 512px arena -- a value
    meant for one read as the other is a 5.3x error in masked area, not a typo that fails.

    `exclude_block` (default False) drops the block's own pixels from the mask per frame, so
    the corruption stays on the goal region instead of destroying the block once it arrives
    there. Optional and defaulted so the original arm's config still resolves unchanged.
    """
    if cfg is None:
        return None
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = dict(cfg)
    unknown = set(cfg) - {'t', 'margin_px', 'exclude_block'}
    if unknown:
        raise ValueError(f'goal_mask_noise got unknown key(s) {sorted(unknown)}; expected '
                         f'only t, margin_px and exclude_block (margin is in IMAGE px).')
    if 't' not in cfg or 'margin_px' not in cfg:
        raise ValueError('goal_mask_noise needs both `t` and `margin_px`')
    t = int(cfg['t'])
    if not (0 <= t < 1000):
        raise ValueError(f'goal_mask_noise.t must be in [0, 1000), got {t}')
    m = float(cfg['margin_px'])
    if m < 0:
        raise ValueError(f'goal_mask_noise.margin_px must be >= 0, got {m}')
    return {'t': t, 'margin_px': m, 'exclude_block': bool(cfg.get('exclude_block', False))}


def decision_frame(buffer_start_idx, sample_start_idx, n_obs_steps):
    """Absolute frame a sampler window's DECISION step sits on -- its last observed frame.

    A SequenceSampler row is (buffer_start, buffer_end, sample_start, sample_end); sample
    index k maps to buffer_start + (k - sample_start). The decision step is sample index
    n_obs_steps-1, and the max() clamps a window padded at the episode START, where
    sample_sequence repeats the first real frame backwards -- a repeated frame is not one the
    demonstrator visited.

    Shared by the transition-manifest builders (scripts/moving_transitions_util.py) and by the
    dataset filter that consumes their output, so "which frame is this window about" has one
    answer. The row -> frame map is one-to-one within an episode.
    """
    return max(buffer_start_idx + (n_obs_steps - 1 - sample_start_idx), buffer_start_idx)


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


# ---------------------------------------------------------------------------
# Transition manifest.
#
# The split manifest says which EPISODES a run sees. This says which TRANSITIONS inside them
# it sees: the windows where the demonstrator's own T moves over the executed window, which
# is where the `t_goal` verifier has any signal at all (it scores by where the T ends up, so
# on a decision that moves nothing every candidate ties).
#
# It is a committed artifact for the same reason the split manifest is: derived once by
# scripts/make_moving_transitions.py, checksummed, and read -- never recomputed at runtime
# from a predicate that could quietly change underneath a half-trained run.
#
# IT STORES ABSOLUTE DECISION FRAMES, NOT WINDOW INDICES. A window index depends on `horizon`
# and the pad settings, so it would silently mean something else under a different dataloader
# config; a decision frame is a position in the zarr and does not.
# ---------------------------------------------------------------------------


def transition_checksum(splits) -> str:
    """Fingerprint of a transition partition. `splits` is {split name: [frame, ...]}.

    The `kind` discriminator is load-bearing: without it this payload is byte-identical to
    `split_checksum`'s, so an episode manifest and a transition manifest holding the same
    integers would fingerprint the same -- and a transition manifest would validate as a
    split manifest. They index different things (frames, not episodes) and must not share a
    hash space.
    """
    payload = json.dumps({'kind': 'transitions',
                          **{k: sorted(int(i) for i in v) for k, v in splits.items()}},
                         sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def load_transition_manifest(transition_file, episode_ends, split_manifest,
                             n_action_steps=None):
    """Read a transition manifest and validate it hard. Never merges -- disagreement raises.

    The load-bearing check is `source_split_checksum`: it pins this file to the exact episode
    partition it was derived from, so a transition list can never be paired with a split it
    does not describe. Without it the two manifests would be independently valid and jointly
    meaningless.
    """
    import hydra.utils
    path = pathlib.Path(hydra.utils.to_absolute_path(str(transition_file)))
    if not path.is_file():
        raise FileNotFoundError(f'transition_file not found: {path}')
    manifest = json.loads(path.read_text())

    for key in ('predicate', 'checksum', 'source_split_checksum'):
        if key not in manifest:
            raise ValueError(f'{path}: transition manifest is missing "{key}"')

    idxs = {name: [int(i) for i in manifest.get(name, [])] for name in SPLIT_NAMES}
    stored, actual = manifest['checksum'], transition_checksum(idxs)
    if stored != actual:
        raise ValueError(
            f'{path}: checksum {stored} does not match its own index lists ({actual}). The '
            f'file was hand-edited; regenerate with scripts/make_moving_transitions.py.')

    expected = split_manifest.get('checksum')
    if expected is not None and manifest['source_split_checksum'] != expected:
        raise ValueError(
            f'{path}: was derived from split checksum '
            f'{manifest["source_split_checksum"]} but this run resolved {expected}. These '
            f'two manifests describe different episode partitions, so the transition lists '
            f'do not refer to the episodes being trained on.')

    ck = manifest.get('episode_ends_checksum')
    if ck is not None and episode_ends is not None:
        actual_ck = episode_ends_checksum(np.asarray(episode_ends))
        if ck != actual_ck:
            raise ValueError(
                f'{path}: episode_ends_checksum {ck} != {actual_ck} for the zarr on disk. '
                f'The episode boundaries changed, so these frame indices no longer point at '
                f'the same transitions.')

    got = manifest['predicate'].get('n_action_steps')
    if n_action_steps is not None and got != int(n_action_steps):
        raise ValueError(
            f'{path}: built for n_action_steps={got} but this config uses {n_action_steps}. '
            f'The executed window is what the predicate reads, so the two disagree about '
            f'which transitions move the T.')

    n_frames = int(np.asarray(episode_ends)[-1]) if episode_ends is not None else None
    if n_frames is not None:
        bad = [i for v in idxs.values() for i in v if not (0 <= i < n_frames)]
        if bad:
            raise ValueError(
                f'{path}: frame index(es) {sorted(bad)[:5]} out of range for {n_frames} '
                f'frames.')
    return manifest


def check_transition_filter_labels(cfg):
    """Refuse to start when `transition_file` and `split_suffix` disagree.

    Same failure the gm_suffix / son_suffix guards exist for, and the costliest version of
    it: run_name has no transition component of its own, so a filtered run at n_demos=176
    with an empty split_suffix resolves to the SAME hydra.run.dir as an unfiltered one, and
    `training.resume: True` continues it -- training on a different set of transitions than
    the checkpoints in that directory were built from, with nothing in the name to show it.

    No-op for configs that declare no `split_suffix` (the non-PushT ones).
    """
    if not (OmegaConf.is_config(cfg) and 'split_suffix' in cfg):
        return
    tf = cfg.get('transition_file', None)
    suffix = str(cfg.get('split_suffix', '') or '')
    tag = cfg.get('transition_tag', None)
    if tf is not None and not suffix:
        raise ValueError(
            f'transition_file is set ({tf}) but split_suffix is empty. The run name would '
            f'be identical to the unfiltered arm at the same n_demos, so this run would '
            f'resume INTO it. Pass e.g. split_suffix=_split-mv.')
    if tf is None and '-mv' in suffix:
        raise ValueError(
            f'split_suffix={suffix!r} claims a moving-transition filter but transition_file '
            f'is null, so this run would train on every window under a name saying it did '
            f'not. Set transition_file, or rename the suffix.')
    if tf is not None and not str(tag or '').strip():
        raise ValueError('transition_tag must be a non-empty wandb label when '
                         'transition_file is set')


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
            goal_mask_noise=None,
            transition_file=None
            ):

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

        # ---- transition filter -------------------------------------------------------
        # Which WINDOWS inside the kept episodes this run sees. Unlike goal_mask_noise this
        # is not a training-time ablation but part of the dataset's definition, so it applies
        # to every split and _split_copy carries it through rather than clearing it.
        self.transition_file = transition_file
        self._transitions = None
        self._transition_frames = None
        if transition_file is not None:
            self._transitions = load_transition_manifest(
                transition_file,
                episode_ends=self.replay_buffer.episode_ends[:],
                split_manifest=self._manifest,
                n_action_steps=pad_after + 1)
            pred = self._transitions['predicate']
            # The manifest's frame list is only the right one for the geometry it was built
            # under: a different horizon enumerates different windows, and a different
            # n_obs_steps puts the decision step on a different frame.
            if int(pred.get('horizon', horizon)) != int(horizon):
                raise ValueError(
                    f'{transition_file}: built at horizon {pred["horizon"]}, this config '
                    f'uses {horizon}.')
            if int(pred.get('n_obs_steps', pad_before + 1)) != int(pad_before + 1):
                raise ValueError(
                    f'{transition_file}: built at n_obs_steps {pred["n_obs_steps"]}, this '
                    f'config uses {pad_before + 1}.')
            self._transition_frames = {
                name: np.asarray(self._transitions.get(name, []), dtype=np.int64)
                for name in SPLIT_NAMES}
            counts = ', '.join(f'{n} {self._transitions[f"{n}_kept"]}/'
                               f'{self._transitions[f"{n}_total"]}' for n in SPLIT_NAMES)
            print(f'PushTImageDataset: transition filter ON '
                  f'({pathlib.Path(transition_file).name}) {counts}')

        episode_mask = {
            'train': self.train_used,
            'val': self.val_pool,
            'test': self.test_pool,
        }[split]

        # Assigned BEFORE the sampler, because _index_filter below reads it.
        self.n_obs_steps = pad_before + 1
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=episode_mask,
            return_sequences=return_sequences,
            index_filter=self._index_filter(split))
        self.episode_mask = episode_mask
        self.split = split
        # PART-3 GOAL-ONLY CORRUPTION. Image-space, so it cannot reuse slot_obs_noise (which
        # noises the ENCODED vector and has no spatial structure). The goal pose is a
        # constant, so the mask is built once here; only the noise is redrawn per sample.
        #
        # TRAIN SPLIT ONLY: `_split_copy` clears the flag, so get_validation_dataset() and
        # get_test_dataset() are clean, and the env runner never touches this class at all.
        self.goal_mask_noise = _resolve_goal_mask_noise(goal_mask_noise)
        self._goal_mask = None
        self._goal_mask_np = None
        self._goal_mask_on = self.goal_mask_noise is not None and split == 'train'
        if self._goal_mask_on:
            from diffusion_policy.env.pusht.goal_mask import goal_mask as _gm, sqrt_alpha_bar
            self._goal_mask_np = _gm(self.goal_mask_noise['margin_px'])
            self._goal_mask = torch.from_numpy(self._goal_mask_np)
            xb = self.goal_mask_noise['exclude_block']
            print(f'PushTImageDataset: goal-only obs corruption ON for split={split} -- '
                  f"t={self.goal_mask_noise['t']} "
                  f"(sqrt(alpha_bar)={sqrt_alpha_bar(self.goal_mask_noise['t']):.4f}), "
                  f"margin={self.goal_mask_noise['margin_px']}px, "
                  f'{self._goal_mask.float().mean().item()*100:.1f}% of the frame'
                  + (', block pixels EXCLUDED per frame' if xb else ''))
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.return_sequences = return_sequences

    def _index_filter(self, split):
        """The window filter for one split, or None when no transition manifest is loaded.

        Maps each sampler row to its decision frame and keeps the row iff that frame is in
        the split's list. `decision_frame` is shared with the manifest builders, so the two
        sides cannot disagree about which frame a window is about.
        """
        if self._transition_frames is None:
            return None
        frames = self._transition_frames[split]
        # self.n_obs_steps, not self.pad_before: __init__ calls this while BUILDING the
        # sampler, and does not assign self.pad_before until after that.
        n_obs = self.n_obs_steps

        def keep(indices):
            i = np.maximum(indices[:, 0] + (n_obs - 1) - indices[:, 2], indices[:, 0])
            return np.isin(i, frames)

        return keep

    def _split_copy(self, episode_mask, split):
        """A shallow copy of this dataset whose sampler is restricted to episode_mask."""
        split_set = copy.copy(self)
        split_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=episode_mask,
            return_sequences=self.return_sequences,
            index_filter=self._index_filter(split)
            )
        split_set.episode_mask = episode_mask
        split_set.split = split
        # val/test are evaluated CLEAN -- the goal corruption is a training-time ablation.
        # copy.copy is shallow, so without this the flag would be inherited.
        #
        # The TRANSITION filter is deliberately NOT cleared here: it is part of what this
        # dataset IS, not a corruption applied only while training, and the held-out set is
        # meant to be the held-out transitions. Hence the split-specific filter above.
        split_set._goal_mask_on = False
        return split_set

    def get_validation_dataset(self):
        """Validation during training runs on the held-out val episodes.

        With a 3-way split val is its own held-out set; with the legacy 2-way split
        val_pool == test_pool (validation runs on the test episodes, as before).
        """
        return self._split_copy(self.val_pool, 'val')

    def get_test_dataset(self):
        """A dataset restricted to the held-out test episodes."""
        return self._split_copy(self.test_pool, 'test')

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
        # Which TRANSITIONS, not just which episodes. Recorded separately, and compared
        # separately by BaseWorkspace.write_splits: `checksum` above is over episode lists
        # only, so two runs differing solely in the filter would otherwise be
        # indistinguishable to the resume guard. Absent means "no filter", which is what
        # every run predating this key was.
        if self._transitions is not None:
            t = self._transitions
            out['transition_filter'] = {
                'transition_file': str(self.transition_file),
                'checksum': t['checksum'],
                'source_split_checksum': t['source_split_checksum'],
                'predicate': t['predicate'],
                **{f'{n}_{k}': t[f'{n}_{k}'] for n in SPLIT_NAMES for k in ('total', 'kept')},
            }
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
            'feedback': lambda: compute_feedback_from_pose(
                self.replay_buffer['block_pos'][frames]),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data={k: source[k]() for k in NORMALIZER_KEYS},
                       last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['agent_pos'].astype(np.float32)
        block_pos = sample['block_pos'].astype(np.float32)  # T, 3
        image = np.moveaxis(sample['img'],-1,1)/255
        feedback = compute_feedback_from_pose(block_pos)  # T, 16

        # POLICY_OBS_KEYS + VERIFIER_OBS_KEYS. The obs dict is WIDER than shape_meta on
        # purpose: agent_pos and feedback are here for PushTVerifier, which resets a pymunk
        # sim from them off the raw dict, and are not in shape_meta so the encoder never
        # reads them. block_pos is NOT emitted -- feedback is an exact, invertible function
        # of it (pusht_verifier.block_pose_from_feedback), so anything needing the block
        # pose reconstructs it, and the verifier's train-time and eval-time resets stay
        # bit-identical.
        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
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
        if self._goal_mask_on:
            from diffusion_policy.env.pusht.goal_mask import apply_goal_mask_noise
            img = torch_data['obs']['image']
            mask = self._goal_mask
            if self.goal_mask_noise['exclude_block']:
                # Per FRAME, not per sample: the block moves between the To observed steps,
                # so one mask for the window would corrupt the block at whichever step it
                # was not computed for. (To, 1, H, W) broadcasts against (To, 3, H, W).
                from diffusion_policy.env.pusht.goal_mask import goal_mask_excluding_block
                poses = np.asarray(sample['block_pos'], dtype=np.float32)[:img.shape[0]]
                mask = torch.from_numpy(np.stack(
                    [goal_mask_excluding_block(self._goal_mask_np, p) for p in poses]
                )).unsqueeze(1)
            torch_data['obs']['image'] = apply_goal_mask_noise(
                img, mask, self.goal_mask_noise['t'], torch.randn_like(img))
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('~/Projects/gym-pusht/data/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)
