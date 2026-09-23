"""The transition filter: which windows inside the split's episodes a run actually sees.

The filter's whole job is to be the SAME predicate the analysis stratifies by, applied to the
dataset. So the load-bearing test here is not "does it select something" but "does it select
exactly what expert_moved selects" -- test_filter_agrees_with_expert_moved. Everything else
guards the plumbing around that.
"""
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_policy.common.replay_buffer import ReplayBuffer          # noqa: E402
from diffusion_policy.common.sampler import SequenceSampler             # noqa: E402
from diffusion_policy.dataset.pusht_image_dataset import (               # noqa: E402
    check_transition_filter_labels, decision_frame, load_transition_manifest,
    split_checksum, transition_checksum)

SPLITS = ROOT / 'diffusion_policy/config/splits'
SPLIT_FILE = SPLITS / 'pusht_seed42_train176.json'
TRANS_FILE = SPLITS / 'transitions/pusht_seed42_train176_moving_ta8.json'
ZARR = ROOT / 'data/pusht_cchi_v7_replay.zarr'
HORIZON, N_OBS, N_ACT = 16, 2, 8

requires_zarr = pytest.mark.skipif(not ZARR.exists(), reason='PushT zarr not present')


# --------------------------------------------------------------------------- decision_frame

def test_decision_frame_is_the_last_observed_frame():
    # unpadded window starting at frame 100: obs steps are 100, 101; the decision is 101
    assert decision_frame(100, 0, N_OBS) == 101


def test_decision_frame_clamps_a_window_padded_at_the_episode_start():
    # sample_start_idx=1 means sample index 0 is an edge-repeat of frame 100, not a visited
    # frame; the decision step (sample index 1) IS frame 100.
    assert decision_frame(100, 1, N_OBS) == 100


def test_decision_frame_is_one_to_one_within_an_episode():
    rows = _rows(0, 60)
    frames = [decision_frame(int(bs), int(ss), N_OBS) for bs, _, ss, _ in rows]
    assert len(set(frames)) == len(frames)


# ----------------------------------------------------------------- the predicate on synthetic data

def _rows(start, end):
    from diffusion_policy.common.sampler import create_indices
    ends = np.array([end], dtype=np.int64)
    assert start == 0, 'helper builds a single episode starting at 0'
    return create_indices(ends, sequence_length=HORIZON,
                          episode_mask=np.array([True]),
                          pad_before=N_OBS - 1, pad_after=N_ACT - 1, debug=False)


def _index_filter(frames):
    f = np.asarray(frames, dtype=np.int64)

    def keep(ix):
        i = np.maximum(ix[:, 0] + (N_OBS - 1) - ix[:, 2], ix[:, 0])
        return np.isin(i, f)
    return keep


def _synthetic_block_pos(n, moving_from, moving_to):
    """Block still everywhere except a ramp over [moving_from, moving_to)."""
    bp = np.zeros((n, 3), dtype=np.float32)
    bp[:, 0] = 10.0
    for t in range(n):
        if t >= moving_to:
            bp[t, 0] = 10.0 + (moving_to - moving_from)
        elif t >= moving_from:
            bp[t, 0] = 10.0 + (t - moving_from)
    return bp


def test_filter_agrees_with_expert_moved():
    """The dataset filter and the analysis stratum must select the same windows.

    If these ever diverge the arms are measured on transitions they were not trained on,
    which is the one failure this whole mechanism exists to prevent.
    """
    from scripts.astar_uniform_walk import expert_moved
    n = 60
    bp = _synthetic_block_pos(n, 20, 40)
    rows = _rows(0, n)
    direct, viafilter = [], []
    for bs, be, ss, _ in rows:
        i = decision_frame(int(bs), int(ss), N_OBS)
        hi = min(i + N_ACT + 1, int(be))
        direct.append(hi - i >= 2 and bool(expert_moved([bp[i:hi]], N_ACT)[0]))
    kept = [decision_frame(int(bs), int(ss), N_OBS)
            for (bs, be, ss, _), m in zip(rows, direct) if m]
    viafilter = list(_index_filter(kept)(rows))
    assert viafilter == direct
    assert any(direct) and not all(direct), 'synthetic case must exercise both outcomes'


def test_padding_at_the_episode_end_does_not_read_as_moving():
    """A tail window is edge-repeat padded; a repeated pose never moves.

    The executed window must be CLIPPED at buffer_end rather than padded, or the last
    windows would be classified on an artefact of padding instead of on the demonstration.
    """
    from scripts.astar_uniform_walk import expert_moved
    n = 60
    bp = _synthetic_block_pos(n, 20, 40)          # still from frame 40 on
    rows = _rows(0, n)
    tail = [r for r in rows if int(r[1]) == n][-1]
    bs, be, ss, _ = (int(x) for x in tail)
    i = decision_frame(bs, ss, N_OBS)
    assert be - i < N_ACT + 1, 'this window must actually be truncated for the test to bite'
    assert not bool(expert_moved([bp[i:be]], N_ACT)[0])


# ------------------------------------------------------------------ SequenceSampler integration

def _tiny_buffer(n_eps=3, ep_len=60):
    rb = ReplayBuffer.create_empty_numpy()
    for _ in range(n_eps):
        rb.add_episode({'action': np.zeros((ep_len, 2), dtype=np.float32),
                        'block_pos': np.zeros((ep_len, 3), dtype=np.float32)})
    return rb


def test_index_filter_is_optional_and_defaults_to_every_window():
    rb = _tiny_buffer()
    plain = SequenceSampler(rb, HORIZON, pad_before=N_OBS - 1, pad_after=N_ACT - 1)
    assert plain.index_filter is None and len(plain) == len(plain.indices)


def test_index_filter_selects_exactly_the_named_frames():
    rb = _tiny_buffer()
    full = SequenceSampler(rb, HORIZON, pad_before=N_OBS - 1, pad_after=N_ACT - 1)
    frames = [decision_frame(int(bs), int(ss), N_OBS) for bs, _, ss, _ in full.indices][::3]
    filt = SequenceSampler(rb, HORIZON, pad_before=N_OBS - 1, pad_after=N_ACT - 1,
                           index_filter=_index_filter(frames))
    assert len(filt) == len(frames)
    got = [decision_frame(int(bs), int(ss), N_OBS) for bs, _, ss, _ in filt.indices]
    assert sorted(got) == sorted(frames)


def test_index_filter_returning_the_wrong_shape_raises():
    rb = _tiny_buffer()
    with pytest.raises(AssertionError):
        SequenceSampler(rb, HORIZON, pad_before=N_OBS - 1, pad_after=N_ACT - 1,
                        index_filter=lambda ix: np.ones(len(ix) + 1, dtype=bool))


# --------------------------------------------------------------------- manifest validation

def _manifest(**over):
    m = json.loads(TRANS_FILE.read_text())
    m.update(over)
    return m


def _write(tmp_path, m):
    p = tmp_path / 'transitions.json'
    p.write_text(json.dumps(m))
    return p


@pytest.fixture
def split_manifest():
    return json.loads(SPLIT_FILE.read_text())


def test_manifest_loads(split_manifest):
    m = load_transition_manifest(TRANS_FILE, None, split_manifest, n_action_steps=N_ACT)
    assert m['train_kept'] < m['train_total'] and m['test_kept'] < m['test_total']


def test_missing_file_raises(split_manifest, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_transition_manifest(tmp_path / 'nope.json', None, split_manifest)


@pytest.mark.parametrize('key', ['predicate', 'checksum', 'source_split_checksum'])
def test_missing_required_key_raises(split_manifest, tmp_path, key):
    m = _manifest()
    del m[key]
    with pytest.raises(ValueError, match='missing'):
        load_transition_manifest(_write(tmp_path, m), None, split_manifest)


def test_hand_edited_index_list_raises(split_manifest, tmp_path):
    m = _manifest()
    m['test'] = m['test'][:-1]                      # drop one frame, leave the checksum
    with pytest.raises(ValueError, match='does not match its own index lists'):
        load_transition_manifest(_write(tmp_path, m), None, split_manifest)


def test_wrong_source_split_raises(split_manifest, tmp_path):
    m = _manifest(source_split_checksum='0' * 32)
    with pytest.raises(ValueError, match='different episode partitions'):
        load_transition_manifest(_write(tmp_path, m), None, split_manifest)


def test_wrong_n_action_steps_raises(split_manifest, tmp_path):
    with pytest.raises(ValueError, match='n_action_steps'):
        load_transition_manifest(TRANS_FILE, None, split_manifest, n_action_steps=N_ACT + 1)


def test_changed_zarr_raises(split_manifest, tmp_path):
    m = _manifest(episode_ends_checksum='0' * 32)
    with pytest.raises(ValueError, match='episode_ends_checksum'):
        load_transition_manifest(_write(tmp_path, m), np.array([10, 20, 30]), split_manifest)


def test_out_of_range_frame_raises(split_manifest, tmp_path):
    m = _manifest()
    m['test'] = sorted(m['test'] + [10 ** 9])
    m['checksum'] = transition_checksum({k: m.get(k, []) for k in ('train', 'val', 'test')})
    m['episode_ends_checksum'] = None
    with pytest.raises(ValueError, match='out of range'):
        load_transition_manifest(_write(tmp_path, m), np.array([100, 200]), split_manifest)


def test_checksum_is_order_independent():
    a = transition_checksum({'train': [3, 1, 2], 'val': [], 'test': [9]})
    b = transition_checksum({'train': [1, 2, 3], 'val': [], 'test': [9]})
    assert a == b


# ------------------------------------------------------------------------- the label guard

@pytest.mark.parametrize('tf,suffix,ok', [
    (str(TRANS_FILE), '_split-mv', True),      # matched
    (None, '', True),                          # every run predating the filter
    (str(TRANS_FILE), '', False),              # would resume INTO the unfiltered run
    (None, '_split-mv', False),                # claims a filter it does not have
])
def test_label_guard(tf, suffix, ok):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({'split_suffix': suffix, 'transition_file': tf,
                            'transition_tag': 't_goal_moving'})
    if ok:
        check_transition_filter_labels(cfg)
    else:
        with pytest.raises(ValueError):
            check_transition_filter_labels(cfg)


def test_label_guard_is_a_noop_without_split_suffix():
    from omegaconf import OmegaConf
    check_transition_filter_labels(OmegaConf.create({'name': 'robomimic'}))


# -------------------------------------------------------------------------- the resume guard

class _FakeDataset:
    def __init__(self, indices):
        self._indices = indices

    def get_split_indices(self):
        return self._indices


def _splits(filter_checksum=None):
    out = {'train': [0, 1], 'val': [], 'test': [2]}
    out['checksum'] = split_checksum(out['train'], out['val'], out['test'])
    if filter_checksum is not None:
        out['transition_filter'] = {'checksum': filter_checksum}
    return out


def _write_splits(tmp_path, dataset):
    """Drive the real write_splits. `output_dir` is a read-only property on BaseWorkspace,
    so the stub sets the `_output_dir` the property reads instead of shadowing it."""
    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    ws = object.__new__(BaseWorkspace)
    ws._output_dir = str(tmp_path)
    return BaseWorkspace.write_splits(ws, dataset)


def test_resume_guard_allows_an_unchanged_filter(tmp_path):
    _write_splits(tmp_path, _FakeDataset(_splits('abc')))
    _write_splits(tmp_path, _FakeDataset(_splits('abc')))


def test_resume_guard_allows_filterless_runs(tmp_path):
    """Every splits.json written before this key existed must still resume."""
    _write_splits(tmp_path, _FakeDataset(_splits(None)))
    _write_splits(tmp_path, _FakeDataset(_splits(None)))


def test_resume_guard_refuses_a_changed_filter(tmp_path):
    """The episode lists are identical here -- only the transitions moved."""
    _write_splits(tmp_path, _FakeDataset(_splits('abc')))
    with pytest.raises(RuntimeError, match='transition filter'):
        _write_splits(tmp_path, _FakeDataset(_splits('def')))


def test_resume_guard_refuses_adding_a_filter_to_an_unfiltered_run(tmp_path):
    _write_splits(tmp_path, _FakeDataset(_splits(None)))
    with pytest.raises(RuntimeError, match='transition filter'):
        _write_splits(tmp_path, _FakeDataset(_splits('abc')))


# ------------------------------------------------------------------------------ on real data

@requires_zarr
@pytest.mark.slow
def test_committed_manifests_agree_with_the_zarr():
    from scripts.moving_transitions_util import moving_frames_for
    import zarr
    split = json.loads(SPLIT_FILE.read_text())
    trans = json.loads(TRANS_FILE.read_text())
    root = zarr.open(str(ZARR), 'r')
    ends = np.asarray(root['meta/episode_ends'])
    bp = np.asarray(root['data/block_pos'])
    for name in ('train', 'test'):
        kept, total = moving_frames_for(bp, ends, split[name])
        assert kept == trans[name], f'{name}: committed frames differ from the zarr'
        assert total == trans[f'{name}_total']


@requires_zarr
@pytest.mark.slow
def test_the_committed_ratio_is_100_to_20():
    trans = json.loads(TRANS_FILE.read_text())
    ratio = 100 * trans['test_kept'] / trans['train_kept']
    assert abs(ratio - 20.0) < 0.5, f'ratio is 100:{ratio:.2f}'


@requires_zarr
@pytest.mark.slow
def test_filtered_sampler_length_matches_the_manifest():
    split = json.loads(SPLIT_FILE.read_text())
    trans = json.loads(TRANS_FILE.read_text())
    from diffusion_policy.dataset.pusht_image_dataset import masks_from_manifest
    rb = ReplayBuffer.copy_from_path(str(ZARR), keys=['block_pos', 'action'])
    masks = dict(zip(('train', 'val', 'test'),
                     masks_from_manifest(split, rb.n_episodes)))
    for name in ('train', 'test'):
        s = SequenceSampler(rb, HORIZON, pad_before=N_OBS - 1, pad_after=N_ACT - 1,
                            episode_mask=masks[name],
                            index_filter=_index_filter(trans[name]))
        assert len(s) == trans[f'{name}_kept']


@requires_zarr
@pytest.mark.slow
def test_dataset_constructs_and_filters_every_split():
    """The FULL construction path, which is where an ordering bug in __init__ hides.

    `_index_filter` is called while the sampler is being built, before __init__ assigns
    `self.pad_before` -- so reading pad_before in there is an AttributeError that only the
    real constructor reaches. Testing the filter arithmetic in isolation does not catch it.

    Also pins the asymmetry with goal_mask_noise: the transition filter SURVIVES _split_copy
    (the held-out set is meant to be the held-out transitions) while the goal corruption does
    not (it is a training-time ablation).
    """
    from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
    trans = json.loads(TRANS_FILE.read_text())
    ds = PushTImageDataset(
        zarr_path=str(ZARR), horizon=HORIZON, pad_before=N_OBS - 1, pad_after=N_ACT - 1,
        split_file=str(SPLIT_FILE), transition_file=str(TRANS_FILE),
        n_test_episodes=30, n_val_episodes=0, n_train_episodes=176, split='train')
    assert len(ds) == trans['train_kept']
    assert len(ds.get_test_dataset()) == trans['test_kept']

    test_ds = ds.get_test_dataset()
    assert test_ds._transition_frames is not None, '_split_copy dropped the filter'
    assert test_ds._goal_mask_on is False

    rec = ds.get_split_indices()['transition_filter']
    assert rec['checksum'] == trans['checksum']
    assert rec['train_kept'] == trans['train_kept']

    sample = ds[0]
    assert sample['obs']['image'].shape[0] == HORIZON or sample['action'].shape[0] == HORIZON


@requires_zarr
@pytest.mark.slow
def test_dataset_without_a_transition_file_is_unchanged():
    """The filter is opt-in: every arm predating it must see every window, as before."""
    from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
    trans = json.loads(TRANS_FILE.read_text())
    ds = PushTImageDataset(
        zarr_path=str(ZARR), horizon=HORIZON, pad_before=N_OBS - 1, pad_after=N_ACT - 1,
        split_file=str(SPLIT_FILE), n_test_episodes=30, n_val_episodes=0,
        n_train_episodes=176, split='train')
    assert len(ds) == trans['train_total']
    assert ds.get_split_indices().get('transition_filter') is None
