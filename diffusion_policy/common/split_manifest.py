"""Episode split manifests: the JSON files under ``diffusion_policy/config/splits`` that
name a dataset's train / val / test episodes outright (PushT, maze and the SimplerEnv
strict datasets alike). Env-independent, so dataset classes for every task can import it
without pulling in any env registration.
"""
import hashlib
import json
import pathlib

import numpy as np


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
