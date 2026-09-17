"""The split machinery: the one invariant everything else is measured against.

An episode index only means something relative to a partition and a zarr. If the partition
drifts, nothing raises -- the numbers simply stop being about what their label says, and two
arms stop being comparable while still printing the same column headers. That is why this file
exists, and why almost every test here is about a MISMATCH being refused rather than about a
split being correct.

Everything runs on synthetic `episode_ends` and `tmp_path`, so the suite needs no zarr. The one
test that does is marked `slow`.
"""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from diffusion_policy.dataset.pusht_image_dataset import (  # noqa: E402
    build_split_manifest, episode_ends_checksum, get_split_masks, get_split_masks_3way,
    load_split_manifest, masks_from_manifest, split_checksum)

N_EPISODES = 206                      # the real dataset's count, so the budgets below are real
ENDS = np.cumsum(np.full(N_EPISODES, 120, dtype=np.int64))
SPLITS_DIR = pathlib.Path(__file__).resolve().parents[1] / 'diffusion_policy/config/splits'
# the five manifests that are NOT seed-derived; see test_geometric_manifests_are_not_seed_derived
GEOMETRIC = ('pusht_blockborder_train30_core50.json', 'pusht_blockborder_train60_core50.json',
             'pusht_blockborder_train100_core50.json',
             'pusht_blockquad_bottomleft_train137.json',
             'pusht_blockquad_topright_train173.json')


def _manifest(n_train=100, n_val=30, n_test=50, seed=42, ends=ENDS):
    return build_split_manifest(ends, 'data/fake.zarr', seed=seed, n_test_episodes=n_test,
                                n_val_episodes=n_val, n_train_episodes=n_train)


def _write(tmp_path, manifest, name='m.json'):
    p = tmp_path / name
    p.write_text(json.dumps(manifest))
    return str(p)


# ------------------------------------------------------------------ derivation is a function
def test_the_derivation_does_not_touch_the_global_rng():
    """Two builds must agree even if something reseeds numpy between them.

    `default_rng(seed)` is a private stream; `np.random.seed` is the global one. A refactor to
    the global RNG would still look deterministic in a single-test process and would drift the
    moment anything else drew a number first.
    """
    a = _manifest()
    np.random.seed(0)
    np.random.random(1000)
    b = _manifest()
    assert a == b


def test_train_budgets_are_nested():
    """25 < 30 < 100 must be strict PREFIXES, not independent draws.

    This is what lets a 30-demo run be compared against a 100-demo one as a data-budget
    ablation: the smaller arm trained on a subset of the larger arm's episodes, so the only
    thing that varied is how many. `rng.choice` per budget would break it silently.
    """
    sets = {n: set(_manifest(n_train=n)['train']) for n in (25, 30, 100, 126)}
    for small, big in ((25, 30), (30, 100), (100, 126)):
        assert sets[small] < sets[big], f'{small} is not a strict subset of {big}'


def test_val_and_test_do_not_move_when_the_train_budget_changes():
    """Only `train` may shrink; the held-out sets are the report and must be fixed."""
    base = _manifest(n_train=100)
    for n in (25, 30, 126):
        other = _manifest(n_train=n)
        assert other['val'] == base['val']
        assert other['test'] == base['test']


def test_the_three_splits_are_disjoint_and_sized_as_declared():
    m = _manifest(n_train=100, n_val=30, n_test=50)
    train, val, test = (set(m[k]) for k in ('train', 'val', 'test'))
    assert not (train & val) and not (train & test) and not (val & test)
    assert (len(train), len(val), len(test)) == (100, 30, 50)
    assert m['derivation']['seed'] == 42


def test_the_3way_test_set_is_the_2way_test_set():
    """The docstring's central claim, which eval paths rely on: test is drawn FIRST.

    It is what made eval_bon.py's seed derivation agree with the manifest on the seed-42 family
    for months, and therefore what hid the bug until a geometric manifest was evaluated.
    """
    _, test2 = get_split_masks(N_EPISODES, n_test_episodes=50, seed=42)
    _, _, test3 = get_split_masks_3way(N_EPISODES, n_test_episodes=50, n_val_episodes=30,
                                       seed=42)
    assert np.array_equal(test2, test3)


# ------------------------------------------------------------------ the two checksums
def test_split_checksum_is_order_independent_but_content_sensitive():
    m = _manifest()
    shuffled = list(reversed(m['train']))
    assert split_checksum(shuffled, m['val'], m['test']) == m['checksum']
    moved = [m['train'][0] + 1000] + m['train'][1:]
    assert split_checksum(moved, m['val'], m['test']) != m['checksum']


def test_episode_ends_checksum_tracks_boundaries_not_dtype():
    """Same boundaries in int32 and int64 are the same dataset; a shifted one is not."""
    assert episode_ends_checksum(ENDS.astype(np.int32)) == episode_ends_checksum(ENDS)
    shifted = ENDS.copy()
    shifted[3] += 1
    assert episode_ends_checksum(shifted) != episode_ends_checksum(ENDS)


# ------------------------------------------------------------------ every refusal branch
def test_a_missing_file_is_not_silently_skipped(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split_manifest(str(tmp_path / 'nope.json'))


def test_a_missing_split_key_raises(tmp_path):
    m = _manifest()
    del m['val']
    with pytest.raises(ValueError, match='missing the "val" split'):
        load_split_manifest(_write(tmp_path, m))


def test_an_episode_in_two_splits_raises(tmp_path):
    m = _manifest()
    m['val'] = [m['train'][0]] + m['val'][1:]
    with pytest.raises(ValueError, match='another split'):
        load_split_manifest(_write(tmp_path, m))


def test_a_hand_edited_index_list_raises(tmp_path):
    """The checksum exists to catch an edit that leaves the file still well-formed."""
    m = _manifest()
    m['train'] = m['train'][:-1]
    with pytest.raises(ValueError, match='does not match its own index lists'):
        load_split_manifest(_write(tmp_path, m))


def test_a_manifest_for_a_different_episode_count_raises(tmp_path):
    m = _manifest()
    with pytest.raises(ValueError, match='was built for'):
        load_split_manifest(_write(tmp_path, m), episode_ends=ENDS[:100])


def test_a_changed_zarr_raises(tmp_path):
    """The failure this guards is 'same indices, different frames'."""
    m = _manifest()
    shifted = ENDS.copy()
    shifted[0] += 5
    with pytest.raises(ValueError, match='episode_ends_checksum'):
        load_split_manifest(_write(tmp_path, m), episode_ends=shifted)


def test_a_config_that_disagrees_with_the_manifest_raises(tmp_path):
    """The manifest is the source of truth; a config asking for another size is a bug."""
    m = _manifest(n_train=100)
    with pytest.raises(ValueError, match='config asks for 30 train'):
        load_split_manifest(_write(tmp_path, m), expected_counts={'train': 30})


def test_an_out_of_range_index_raises(tmp_path):
    m = _manifest()
    m['train'] = m['train'][:-1] + [N_EPISODES + 5]
    m['checksum'] = split_checksum(m['train'], m['val'], m['test'])
    with pytest.raises(ValueError, match='out of range'):
        load_split_manifest(_write(tmp_path, m), episode_ends=ENDS)


# ------------------------------------------------------------------ round trip
def test_masks_from_manifest_reproduces_the_derivation(tmp_path):
    m = _manifest(n_train=100, n_val=30, n_test=50)
    loaded = load_split_manifest(_write(tmp_path, m), episode_ends=ENDS)
    got = masks_from_manifest(loaded, N_EPISODES)
    expected = get_split_masks_3way(N_EPISODES, n_test_episodes=50, n_val_episodes=30,
                                    seed=42, n_train_episodes=100)
    for g, e in zip(got, expected):
        assert np.array_equal(g, e)


# ------------------------------------------------------------------ the committed manifests
@pytest.mark.parametrize('name', sorted(p.name for p in SPLITS_DIR.glob('*.json')))
def test_every_committed_manifest_is_self_consistent(name):
    """Loads clean, and its own checksum and disjointness hold."""
    load_split_manifest(str(SPLITS_DIR / name))


@pytest.mark.parametrize('name', GEOMETRIC)
def test_geometric_manifests_are_not_seed_derived(name):
    """THE REGRESSION TEST for the eval_bon train-leak.

    eval_bon.py and bon_video.py used to re-derive the test split from (seed, n_test_episodes)
    and ignore `split_file`. On the seed-42 family that agrees exactly, which is why it went
    unnoticed. On these five it does not: the seed-derived set overlapped the checkpoint's own
    TRAINING episodes by 7 to 42 of 50. If anyone reintroduces a seed derivation on an eval
    path, this asserts that doing so cannot be equivalent.
    """
    manifest = load_split_manifest(str(SPLITS_DIR / name))
    n_episodes = manifest['n_episodes']
    _, seeded_test = get_split_masks(n_episodes, n_test_episodes=50, seed=42)
    seeded = set(np.nonzero(seeded_test)[0].tolist())
    actual = set(manifest['test'])
    assert actual != seeded, f'{name}: a seed derivation would be indistinguishable here'
    # and the overlap is with TRAIN, which is what makes it a leak rather than a relabelling
    assert seeded & set(manifest['train']), f'{name}: expected seed-derived test to hit train'


def test_the_seed42_alias_is_byte_identical_to_train100():
    """pusht_seed42.json duplicates pusht_seed42_train100.json. Documented, not drift."""
    a = json.loads((SPLITS_DIR / 'pusht_seed42.json').read_text())
    b = json.loads((SPLITS_DIR / 'pusht_seed42_train100.json').read_text())
    for key in ('train', 'val', 'test', 'checksum', 'n_episodes'):
        assert a[key] == b[key], f'{key} diverged between the alias and its target'


@pytest.mark.slow
def test_every_committed_manifest_matches_the_real_zarr():
    """What scripts/dump_pusht_splits.py --check-zarr does, but run automatically.

    Without this the episode_ends_checksum guard is only ever exercised against synthetic
    boundaries, so a manifest built for a re-downloaded zarr would pass the rest of this file.
    """
    import zarr

    from recurrent_ppo.config import DEMO_ZARR

    if not pathlib.Path(DEMO_ZARR).exists():
        pytest.skip(f'{DEMO_ZARR} not present')
    ends = np.asarray(zarr.open(str(DEMO_ZARR), 'r')['meta/episode_ends'])
    for path in sorted(SPLITS_DIR.glob('*.json')):
        load_split_manifest(str(path), episode_ends=ends)


# ------------------------------------------------- the RL arms resolve the SAME episodes
@pytest.mark.slow
def test_the_rl_path_resolves_the_same_episodes_as_the_offline_arms():
    """`states_from_manifest` must agree with the manifest, index for index and state for state.

    This is what makes an RL number comparable with an ST number at all. The RL arms cannot call
    `eval_search_pusht.get_split_states` -- that reads a hydra `cfg.task.dataset` they do not
    have -- so they resolve episodes by naming the manifest directly. Two resolvers for one
    question is exactly how the eval_bon train-leak happened, so this pins them together: the
    episode indices must BE the manifest's own list, and each state must be the zarr's recorded
    first frame of that episode.
    """
    import zarr

    from diffusion_policy.common.replay_buffer import ReplayBuffer
    from recurrent_ppo.config import DEMO_ZARR
    from recurrent_ppo.eval_episodes import states_from_manifest

    if not pathlib.Path(DEMO_ZARR).exists():
        pytest.skip(f"{DEMO_ZARR} not present")

    split_file = str(SPLITS_DIR / 'pusht_seed42_train30.json')
    states, idxs = states_from_manifest(split_file, 'test', zarr_path=DEMO_ZARR)

    manifest = json.loads(pathlib.Path(split_file).read_text())
    assert idxs.tolist() == sorted(manifest['test']), "episode indices are not the manifest's"
    assert len(states) == 50 and states.shape[1] == 5

    # each state is [agent_x, agent_y, block_x, block_y, angle] at that episode's FIRST frame
    rb = ReplayBuffer.copy_from_path(DEMO_ZARR, keys=['agent_pos', 'block_pos'])
    ends = np.asarray(rb.episode_ends[:])
    starts = np.concatenate([[0], ends[:-1]])
    for row, ep in zip(states, idxs):
        f = starts[ep]
        expected = np.concatenate([np.asarray(rb['agent_pos'])[f],
                                   np.asarray(rb['block_pos'])[f]])
        assert np.allclose(row, expected), f"episode {ep}: not its recorded first frame"


@pytest.mark.slow
def test_a_geometric_manifest_gives_the_rl_arms_a_different_set():
    """Sanity that the manifest actually selects: blq and seed-42 must not resolve alike.

    If `states_from_manifest` ignored its argument and fell back to a seeded derivation -- the
    eval_bon failure, in a new place -- this is what would catch it.
    """
    from recurrent_ppo.config import DEMO_ZARR
    from recurrent_ppo.eval_episodes import states_from_manifest

    if not pathlib.Path(DEMO_ZARR).exists():
        pytest.skip(f"{DEMO_ZARR} not present")

    _, a = states_from_manifest(str(SPLITS_DIR / 'pusht_seed42_train30.json'), 'test')
    _, b = states_from_manifest(str(SPLITS_DIR / 'pusht_blockquad_bottomleft_train137.json'),
                                'test')
    assert set(a) != set(b), "the manifest argument is not being honoured"
