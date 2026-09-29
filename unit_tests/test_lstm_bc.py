"""Regression tests for the LSTM BC arm.

    pytest unit_tests/test_lstm_bc.py
    pytest unit_tests/test_lstm_bc.py -m 'not slow'      # skip the two that load the zarr

WHAT THESE ARE FOR. A recurrent policy whose recurrence is broken -- state dropped between
calls, re-zeroed every call, or a trunk that ignores the LSTM's output -- still produces
correctly shaped actions and a loss that falls. Nothing in a training log distinguishes it from
a working one. So group A does not test shapes: it tests that the output is a FUNCTION OF
HISTORY, which is the only thing that makes this arm different from an MLP. Group B then pins
that the history seen at rollout is the same history seen in training, which is where an
off-by-one hides.
"""
import os
import pathlib
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_policy.common.pytorch_util import dict_apply                       # noqa: E402
from diffusion_policy.model.common.normalizer import LinearNormalizer             # noqa: E402
from diffusion_policy.model.vision.multi_image_obs_encoder import (              # noqa: E402
    FlattenObsEncoder)
from diffusion_policy.policy.lstm_bc_policy import LSTMBCPolicy                   # noqa: E402

ZARR = 'data/pusht_cchi_v7_replay.zarr'
SPLIT = 'diffusion_policy/config/splits/pusht_seed42_train30.json'
# the keypoint arm's shape_meta, so no ResNet is built and every test below runs on CPU in ms
SHAPE_META = {
    'obs': {
        'keypoint': {'shape': [9, 2], 'type': 'low_dim'},
        'agent_pos': {'shape': [2], 'type': 'low_dim'},
    },
    'action': {'shape': [2]},
}
HORIZON = 16
N_ACTION_STEPS = 8
N_OBS_STEPS = 8


def _normalizer(seed=0):
    g = torch.Generator().manual_seed(seed)
    data = {
        'keypoint': torch.rand((64, 9, 2), generator=g) * 512.0,
        'agent_pos': torch.rand((64, 2), generator=g) * 512.0,
        'action': torch.rand((64, 2), generator=g) * 512.0,
    }
    norm = LinearNormalizer()
    norm.fit(data=data, last_n_dims=1, mode='limits')
    return norm


def _policy(horizon=HORIZON, n_action_steps=N_ACTION_STEPS, n_obs_steps=N_OBS_STEPS, seed=0):
    torch.manual_seed(seed)
    policy = LSTMBCPolicy(
        shape_meta=SHAPE_META,
        obs_encoder=FlattenObsEncoder(SHAPE_META),
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps)
    policy.set_normalizer(_normalizer())
    policy.eval()          # deterministic: no dropout, and the keypoint arm never crops
    return policy


def _obs_seq(T, B=1, seed=0):
    """A (B, T, ...) observation sequence in raw arena pixels, as the runner supplies."""
    g = torch.Generator().manual_seed(seed)
    return {
        'keypoint': torch.rand((B, T, 9, 2), generator=g) * 512.0,
        'agent_pos': torch.rand((B, T, 2), generator=g) * 512.0,
    }


def _windows(obs_seq, n_obs_steps=N_OBS_STEPS, n_action_steps=N_ACTION_STEPS):
    """The observation windows MultiStepWrapper would hand the policy over one episode.

    Mirrors `stack_last_n_obs`: the FIRST window is n_obs_steps copies of frame 0, because
    reset() seeds the deque with one observation and it pads up by repeating. Every later
    window is the n_action_steps frames just stepped.
    """
    T = next(iter(obs_seq.values())).shape[1]
    yield dict_apply(obs_seq, lambda x: x[:, :1].expand(-1, n_obs_steps, *x.shape[2:]))
    t = 1
    while t < T:
        end = min(t + n_action_steps, T)
        yield dict_apply(obs_seq, lambda x, a=t, b=end: x[:, a:b])
        t = end


# ============================================================ A. the recurrence is real
def test_output_depends_on_history_not_just_current_frame():
    """THE test. Same final frame, different history -> different action.

    If this fails the LSTM's state is not reaching the head and the arm is a memoryless MLP
    wearing an nn.LSTM, which every other signal in the run would report as success.
    """
    policy = _policy()
    final = _obs_seq(1, seed=99)

    def chunk_after(prefix_seed):
        policy.reset()
        for window in _windows(_obs_seq(17, seed=prefix_seed)):
            policy.predict_action(dict_apply(window, lambda x: x.clone()))
        return policy.predict_action(
            dict_apply(final, lambda x: x.expand(-1, N_ACTION_STEPS, *x.shape[2:]))
        )['action_pred']

    a, b = chunk_after(1), chunk_after(2)
    assert not torch.allclose(a, b, atol=1e-6), \
        'two different histories gave the same action for the same current frame: the ' \
        'hidden state is not influencing the output'


def test_repeated_identical_observation_changes_the_output():
    """The state is carried between calls, not rebuilt from the current frame each time."""
    policy = _policy()
    window = dict_apply(_obs_seq(1, seed=7),
                        lambda x: x.expand(-1, N_OBS_STEPS, *x.shape[2:]).clone())
    policy.reset()
    first = policy.predict_action(dict_apply(window, lambda x: x.clone()))['action_pred']
    second = policy.predict_action(dict_apply(window, lambda x: x.clone()))['action_pred']
    assert not torch.allclose(first, second, atol=1e-6), \
        'feeding the same frame twice gave the same chunk: (h, c) is being discarded'


def test_reset_restores_the_initial_state():
    """reset() is the episode boundary the env runner relies on, once per chunk of envs."""
    policy = _policy()
    window = dict_apply(_obs_seq(1, seed=7),
                        lambda x: x.expand(-1, N_OBS_STEPS, *x.shape[2:]).clone())
    policy.reset()
    first = policy.predict_action(dict_apply(window, lambda x: x.clone()))['action_pred']
    for _ in range(5):
        policy.predict_action(dict_apply(window, lambda x: x.clone()))
    policy.reset()
    again = policy.predict_action(dict_apply(window, lambda x: x.clone()))['action_pred']
    assert torch.allclose(first, again, atol=1e-6), \
        'after reset() the policy did not reproduce its first action: state survived the reset'


def test_hidden_state_is_per_batch_element():
    """No state leaks between envs.

    The runner drives 80 episodes as one batch. If state leaked across the batch dimension
    every rollout would be corrupted at once, and the only symptom would be a bad success rate.
    """
    policy = _policy()
    a, b = _obs_seq(17, seed=11), _obs_seq(17, seed=22)
    together = {k: torch.cat([a[k], b[k]], dim=0) for k in a}

    def drive(seq):
        policy.reset()
        out = None
        for window in _windows(seq):
            out = policy.predict_action(dict_apply(window, lambda x: x.clone()))['action_pred']
        return out

    batched = drive(together)
    alone_a, alone_b = drive(a), drive(b)
    # atol is in arena pixels on an O(512) output after 17 LSTM steps; state leaking across
    # the batch changes the action by whole pixels, so this is nowhere near the failure scale.
    assert torch.allclose(batched[0:1], alone_a, atol=1e-3), 'element 0 differs when batched'
    assert torch.allclose(batched[1:2], alone_b, atol=1e-3), 'element 1 differs when batched'


# ============================================================ B. train/eval correspondence
class _CountingLSTM(nn.Module):
    """Records the sequence length handed to the wrapped LSTM on every call."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.seq_lens = []

    def forward(self, x, state=None):
        self.seq_lens.append(int(x.shape[1]))
        return self.inner(x, state)


def test_first_call_after_reset_ingests_one_frame():
    """MultiStepWrapper pads the first window to n_obs_steps copies of frame 0.

    Ingesting all of them would tick the LSTM 8 times on one frame while training reaches t=0
    having consumed it once -- the hidden-state trajectories would diverge from step zero.
    """
    policy = _policy()
    counting = _CountingLSTM(policy.lstm)
    policy.lstm = counting

    policy.reset()
    for window in _windows(_obs_seq(1 + 3 * N_ACTION_STEPS, seed=5)):
        policy.predict_action(dict_apply(window, lambda x: x.clone()))

    assert counting.seq_lens[0] == 1, \
        f'first call ingested {counting.seq_lens[0]} frames, expected 1 (the padded copies)'
    assert all(n == N_ACTION_STEPS for n in counting.seq_lens[1:]), \
        f'later calls ingested {counting.seq_lens[1:]}, expected all {N_ACTION_STEPS}'


def test_lstm_advances_once_per_env_step():
    """Over a 300-step episode at 8 actions/call the LSTM must advance exactly 300 times."""
    policy = _policy()
    counting = _CountingLSTM(policy.lstm)
    policy.lstm = counting

    n_steps = 1 + N_ACTION_STEPS * 37          # 297; the window generator ends on a boundary
    policy.reset()
    for window in _windows(_obs_seq(n_steps, seed=6)):
        policy.predict_action(dict_apply(window, lambda x: x.clone()))

    assert sum(counting.seq_lens) == n_steps, \
        f'LSTM advanced {sum(counting.seq_lens)} times over {n_steps} env steps'


def test_rollout_state_matches_teacher_forced_state():
    """THE STRONGEST TEST: the two shapes of time agree step for step.

    Training runs the LSTM over a whole episode in one pass; rollout replays it in chunks of
    n_action_steps carrying (h, c). If the frame accounting is off by one -- anywhere -- the
    chunk the rollout emits at step t is not the chunk training supervised at t, and NOTHING
    else reports it: the loss still falls and the rollout still runs, just on a policy that was
    optimised for a different input.
    """
    policy = _policy()
    T = 1 + N_ACTION_STEPS * 3                 # 25: windows end exactly at t = 0, 8, 16, 24
    seq = _obs_seq(T, seed=3)

    # teacher forced: one pass over the whole episode, as compute_loss does
    with torch.no_grad():
        nobs = policy.normalizer.normalize(seq)
        features = policy._encode(nobs, 1, T)
        latent, _ = policy.lstm(features)
        forced_mean = policy._chunk_dist(policy.trunk(latent), 1, T).mean   # (1, T, 16, 2)

    # rollout: chunked, carrying state
    policy.reset()
    rolled = []
    with torch.no_grad():
        for window in _windows(seq):
            rolled.append(policy.predict_action(
                dict_apply(window, lambda x: x.clone()))['action_pred'])

    steps = [0] + list(range(N_ACTION_STEPS, T, N_ACTION_STEPS))
    assert len(steps) == len(rolled), (steps, len(rolled))
    for t, got in zip(steps, rolled):
        # compared in NORMALIZED space, where the signal is O(1): unnormalized actions are
        # O(512) arena pixels, so float32 round-off there is already ~5e-5 and any tolerance
        # tight enough to catch a real off-by-one would be flaky instead.
        got_n = policy.normalizer['action'].normalize(got)
        assert torch.allclose(got_n, forced_mean[:, t], atol=1e-5), (
            f'rollout chunk at env step {t} does not match the training forward at sequence '
            f'index {t}: max|diff| = {(got_n - forced_mean[:, t]).abs().max().item():.3e}')


def test_executed_action_is_the_head_of_the_chunk():
    """No control lag: the chunk is rooted at the CURRENT step, so execution starts at 0.

    The window policies in this repo slice from `n_obs_steps-1`, which is right for them and
    would put a silent 7-step lag in this controller.
    """
    policy = _policy()
    policy.reset()
    out = policy.predict_action(
        dict_apply(_obs_seq(1, seed=4), lambda x: x.expand(-1, N_OBS_STEPS, *x.shape[2:])))
    assert out['action_pred'].shape[1] == HORIZON
    assert out['action'].shape[1] == N_ACTION_STEPS
    assert torch.equal(out['action'], out['action_pred'][:, :N_ACTION_STEPS]), \
        'executed actions are not the first n_action_steps of the chunk'


def test_n_action_steps_must_fit_the_obs_deque():
    """MultiStepWrapper's deque is maxlen=n_obs_steps+1; beyond that it drops a frame."""
    with pytest.raises(AssertionError, match='n_action_steps'):
        _policy(n_action_steps=10, n_obs_steps=8)
    _policy(n_action_steps=9, n_obs_steps=8)       # the boundary is allowed


# ============================================================ C. loss correctness
def test_chunk_target_alignment():
    """The chunk supervised at t is actions[t : t+horizon], and its tail is masked off."""
    policy = _policy(horizon=4)
    T = 10
    actions = torch.arange(T, dtype=torch.float32).reshape(1, T, 1).repeat(1, 1, 2)
    mask = torch.ones((1, T))
    target, chunk_mask = policy._chunk_targets(actions, mask)

    assert target.shape == (1, T, 4, 2) and chunk_mask.shape == (1, T, 4)
    for t in range(T):
        for j in range(4):
            if t + j < T:
                assert chunk_mask[0, t, j] == 1.0, (t, j)
                assert target[0, t, j, 0] == float(t + j), (t, j)
            else:
                assert chunk_mask[0, t, j] == 0.0, f'({t},{j}) runs past the episode end'


def test_padding_does_not_contribute_to_the_loss():
    """A padded batch gives the same loss as the episodes weighted individually.

    collate_fn pads actions with 0.0, which in normalized action space is the MIDDLE OF THE
    ARENA. Supervising those would teach the policy to drive to the centre during the last
    `horizon-1` steps of every episode -- wrong, and invisible except as a mediocre success
    rate. This also proves padding cannot leak backwards through the LSTM.
    """
    policy = _policy(horizon=4)
    Ta, Tb = 12, 20
    a, b = _obs_seq(Ta, seed=31), _obs_seq(Tb, seed=32)
    g = torch.Generator().manual_seed(33)
    act_a = torch.rand((1, Ta, 2), generator=g) * 512.0
    act_b = torch.rand((1, Tb, 2), generator=g) * 512.0

    def loss_and_count(seq, act, T):
        batch = {'obs': seq, 'action': act,
                 'attention_mask': torch.ones((1, T), dtype=torch.bool)}
        with torch.no_grad():
            loss = policy.compute_loss(batch)
            nact = policy.normalizer['action'].normalize(act)
            _, cm = policy._chunk_targets(nact, torch.ones((1, T)))
        return float(loss), float(cm.sum())

    la, na = loss_and_count(a, act_a, Ta)
    lb, nb = loss_and_count(b, act_b, Tb)

    pad = lambda x, n: torch.cat([x, torch.zeros_like(x[:, :1]).repeat(
        1, n, *([1] * (x.dim() - 2)))], dim=1)
    batch = {
        'obs': {k: torch.cat([pad(a[k], Tb - Ta), b[k]], dim=0) for k in a},
        'action': torch.cat([pad(act_a, Tb - Ta), act_b], dim=0),
        'attention_mask': torch.cat([
            torch.arange(Tb)[None] < Ta, torch.ones((1, Tb), dtype=torch.bool)], dim=0),
    }
    with torch.no_grad():
        combined = float(policy.compute_loss(batch))

    expected = (la * na + lb * nb) / (na + nb)
    assert combined == pytest.approx(expected, rel=1e-5), (
        f'padded batch loss {combined:.6f} != weighted individual losses {expected:.6f}; '
        f'the padding is being supervised or is leaking through the LSTM')


@pytest.mark.slow
def test_overfits_a_single_episode():
    """The whole path -- encode, LSTM, chunk, masked NLL -- can actually learn.

    Also an end-to-end no-lag check: after overfitting, the chunk the rollout emits at t=0 must
    be the demonstration's own actions[0:8].
    """
    torch.manual_seed(0)
    policy = _policy(horizon=4)
    T = 24
    seq = _obs_seq(T, seed=41)
    # a learnable target: the action is a fixed linear function of the current agent position
    actions = seq['agent_pos'] * 0.5 + 60.0
    batch = {'obs': seq, 'action': actions,
             'attention_mask': torch.ones((1, T), dtype=torch.bool)}

    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    first = float(policy.compute_loss(batch))
    for _ in range(400):
        opt.zero_grad()
        loss = policy.compute_loss(batch)
        loss.backward()
        opt.step()
    last = float(loss)
    assert last < first - 1.0, f'loss barely moved: {first:.3f} -> {last:.3f}'

    policy.eval()
    policy.reset()
    with torch.no_grad():
        out = policy.predict_action(dict_apply(
            dict_apply(seq, lambda x: x[:, :1]),
            lambda x: x.expand(-1, N_OBS_STEPS, *x.shape[2:])))
    err = (out['action'][:, 0] - actions[:, 0]).abs().max().item()
    assert err < 15.0, f'overfitted policy is {err:.1f}px off its own first action'


# ============================================================ D. architecture
def test_architecture_matches_recurrent_ppo():
    """Pinned against recurrent_ppo's own DEFAULTS, not against literals, so they cannot drift.

    The objective of this arm is "the LSTM as used in recurrent PPO"; if that stops being true
    the comparison silently becomes a different one.
    """
    from recurrent_ppo.config import DEFAULTS

    policy = _policy()
    assert policy.lstm.hidden_size == DEFAULTS['lstm_hidden_size']
    assert policy.lstm.num_layers == DEFAULTS['n_lstm_layers']
    assert policy.lstm.batch_first is True

    widths = [int(x) for x in DEFAULTS['net_arch'].split(',') if x]
    linears = [m for m in policy.trunk if isinstance(m, nn.Linear)]
    acts = [m for m in policy.trunk if not isinstance(m, nn.Linear)]
    assert [m.out_features for m in linears] == widths
    assert linears[0].in_features == DEFAULTS['lstm_hidden_size']
    assert all(isinstance(a, nn.Tanh) for a in acts), 'SB3 defaults to Tanh, never overridden'

    # state-independent log_std, exactly SB3's DiagGaussianDistribution
    assert isinstance(policy.log_std, nn.Parameter)
    assert tuple(policy.log_std.shape) == (2,)
    assert torch.allclose(policy.log_std.detach(),
                          torch.full((2,), float(DEFAULTS['log_std_init'])))

    # BC has no use for PPO's critic machinery, and its presence would train dead weights
    for absent in ('value_net', 'q_net', 'critic', 'lstm_critic', 'log_std_head'):
        assert not hasattr(policy, absent), f'{absent} has no role in behaviour cloning'


# ============================================================ E. observation plumbing
def test_obs_to_dict_slicing_is_row_major_and_drops_the_mask():
    """PushTKeypointsEnv's 40-d Box -> the {keypoint, agent_pos} dict the dataset emits.

    Bit-exact and physics-free: the value half is `kps.flatten()` then agent_pos, so the first
    18 entries reshape row-major to (9, 2). The second half is the visibility mask and is
    dropped -- this arm never occludes, so it is a constant.
    """
    from diffusion_policy.env_runner.pusht_search_keypoints_runner import (
        PushTSearchKeypointsRunner as R)

    value = np.arange(20, dtype=np.float32)
    obs = np.concatenate([value, np.full(20, -7.0, dtype=np.float32)])[None, None]
    out = R._obs_to_dict(obs)
    assert out['keypoint'].shape == (1, 1, 9, 2) and out['agent_pos'].shape == (1, 1, 2)
    np.testing.assert_array_equal(out['keypoint'][0, 0], value[:18].reshape(9, 2))
    np.testing.assert_array_equal(out['agent_pos'][0, 0], value[18:20])
    assert not np.any(out['keypoint'] == -7.0), 'the mask half leaked into the observation'


# ============================================================ F. same data, same episodes
requires_zarr = pytest.mark.skipif(
    not os.path.isdir(ZARR), reason=f'{ZARR} not present')


@pytest.mark.slow
@requires_zarr
def test_keypoint_dataset_split_matches_the_image_arms():
    """One partition, byte-identical, so a keypoint number sits beside a UNet BC number.

    Compared against the MANIFEST's own committed checksum rather than against a constructed
    PushTImageDataset. That is the same claim: `load_split_manifest` recomputes the checksum
    from the index lists and RAISES if it disagrees, and the image dataset resolves its
    partition through that very call on this very file -- so matching the manifest is matching
    the image arm, transitively and by construction. Asking PushTImageDataset directly would
    copy the zarr's 2.84 GB `img` array into RAM to read three lists of ints.

    Also pins one sample per episode and an attention_mask whose row sums are the true
    episode lengths.
    """
    import json

    from torch.utils.data import DataLoader

    from diffusion_policy.common.sampler import get_collate_fn
    from diffusion_policy.dataset.pusht_keypoint_dataset import PushTKeypointDataset

    kp = PushTKeypointDataset(
        zarr_path=ZARR, horizon=300, pad_before=0, pad_after=0, return_sequences=True,
        n_test_episodes=50, n_val_episodes=30, n_train_episodes=30, split='train',
        split_file=SPLIT)
    assert (len(kp), len(kp.get_validation_dataset()), len(kp.get_test_dataset())) == (30, 30, 50)

    manifest = json.loads(pathlib.Path(SPLIT).read_text())
    resolved = kp.get_split_indices()
    assert resolved['checksum'] == manifest['checksum'], \
        'the keypoint arm resolved a different partition from the manifest the image arms read'
    for name in ('train', 'val', 'test'):
        assert resolved[name] == [int(i) for i in manifest[name]], name
    assert resolved['episode_ends_checksum'] == manifest['episode_ends_checksum'], \
        'the manifest was built for a different zarr than the one on disk'

    batch = next(iter(DataLoader(kp, batch_size=4, collate_fn=get_collate_fn(), shuffle=False)))
    T = batch['action'].shape[1]
    assert batch['obs']['keypoint'].shape[:2] == (4, T)
    assert batch['attention_mask'].shape == (4, T)
    lengths = [len(kp.sampler.sample_sequence(i)['action']) for i in range(4)]
    assert batch['attention_mask'].sum(1).tolist() == lengths


@pytest.mark.slow
@requires_zarr
def test_env_keypoint_obs_reproduces_the_zarr():
    """The env's keypoints are the zarr's keypoints, up to PushTEnv's own reset settle step.

    THIS IS THE CHECK THAT CAN INVALIDATE EVERY KEYPOINT NUMBER. The observation is built from
    `PymunkKeypointManager`'s local map; the zarr's `keypoint` array was built from that same
    map at collection time. If they ever diverge, the policy trains on one quantity and is
    evaluated on another, and the success rate means nothing. A drifted map would show up here
    as an error of tens of pixels.

    WHY THIS IS NOT BIT-EXACT, and why that is fine. `PushTEnv._set_state` ends with
    `self.space.step(1/sim_hz)` ("Run physics to take effect"), so resetting onto a recorded
    state advances the simulation one 10ms tick. For 184 of the 206 episodes the block is at
    rest and nothing moves (error exactly 0.0); for the ~22 whose recorded first frame has it
    in contact it settles by up to ~2px. The IMAGE arms reset through the very same
    `_set_state`, so their rendered first frame carries the identical offset -- this is a
    property of the shared eval protocol, not of this arm, and it does not affect
    comparability. The bound is what is asserted; a real map drift cannot hide under it.
    """
    import zarr

    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
    from diffusion_policy.env_runner.pusht_search_keypoints_runner import (
        PushTSearchKeypointsRunner as R)

    z = zarr.open(ZARR, 'r')
    ends = z['meta']['episode_ends'][:]
    starts = np.concatenate([[0], ends[:-1]])
    env = PushTKeypointsEnv(legacy=False, keypoint_visible_rate=1.0, agent_keypoints=False,
                            **PushTKeypointsEnv.genenerate_keypoint_manager_params())

    errors = []
    for ep in range(0, len(ends), 7):          # every 7th episode: 30 of them, ~2s
        i = int(starts[ep])
        env.reset_to_state = np.asarray(z['data']['state'][i], dtype=np.float64)
        obs = env.reset()
        assert obs.shape == (40,)
        assert np.all(obs[20:] == 1.0), 'the visibility mask is not pinned to fully visible'
        got = R._obs_to_dict(obs[None, None])
        np.testing.assert_allclose(
            got['agent_pos'][0, 0], z['data']['agent_pos'][i], atol=1e-4,
            err_msg='agent_pos is set directly by _set_state and must be exact')
        errors.append(np.abs(
            got['keypoint'][0, 0] - np.asarray(z['data']['keypoint'][i])).max())

    errors = np.asarray(errors)
    assert errors.max() < 3.0, (
        f'keypoint observation is up to {errors.max():.2f}px from the zarr, beyond what one '
        f'reset settle step explains -- the local keypoint map has probably drifted from the '
        f'one that generated the data')
    assert (errors == 0.0).mean() > 0.7, (
        f'only {(errors == 0.0).mean():.0%} of episodes reproduced exactly (expected ~90%); '
        f'the settle step should be a no-op wherever the block starts at rest')


# ============================================================ G. best-of-n eval, both arms
# eval_bon.py, like train.py / eval.py / eval_search_pusht.py, reopens sys.stdout and
# sys.stderr AT MODULE LEVEL to force line buffering in SLURM logs. Under pytest those file
# descriptors belong to the capture machinery, and the replacement objects close them when
# they are garbage collected -- which surfaces much later as `OSError: [Errno 9] Bad file
# descriptor` during teardown, attributed to whatever test happened to be running. Importing
# through this helper keeps the originals and pins the replacements alive so the fds survive.
# The convention itself is left alone: five other entry points share it and nothing else
# imports them.
_EVAL_BON_KEEPALIVE = []


def _import_eval_bon():
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        import eval_bon
    finally:
        _EVAL_BON_KEEPALIVE.extend([sys.stdout, sys.stderr])
        sys.stdout, sys.stderr = saved_out, saved_err
    return eval_bon


def test_eval_bon_selects_the_env_from_shape_meta():
    """`eval_bon.py` must pick its env from what the POLICY consumes, not from a config name.

    The two arms differ in the env they need and in whether the observation is already a dict,
    and the failure mode of getting it wrong is not a crash at the boundary: an image env
    handed to a keypoint policy fails later, inside the normalizer, where it reads as a
    checkpoint problem. So the discriminator is pinned here.
    """
    import ast
    src = pathlib.Path(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'eval_bon.py')).read_text()
    tree = ast.parse(src)
    fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert 'build_keypoint_envs' in fns, 'eval_bon.py lost its keypoint branch'
    assert "'keypoint' in cfg.shape_meta.obs" in src, (
        'the arm must be chosen from shape_meta, which is what the policy actually reads')


def test_eval_bon_keypoint_obs_feeds_the_policy_unchanged():
    """The conversion eval_bon applies is the SAME one training was scored under.

    Not a re-implementation of the slicing (test_obs_to_dict_slicing_is_row_major already pins
    that): the point here is that eval_bon routes the env's flat Box through the runner's own
    `_obs_to_dict` and that the result drops straight into `predict_action`. Two independent
    splittings of a 40-d vector agreeing today is exactly the thing that silently stops being
    true.
    """
    eval_bon = _import_eval_bon()
    from diffusion_policy.env_runner.pusht_search_keypoints_runner import (
        PushTSearchKeypointsRunner as R)

    assert eval_bon.PushTSearchKeypointsRunner is R, (
        'eval_bon must reuse the runner\'s conversion, not carry a private copy')

    policy = _policy()
    policy.reset()
    # one env, n_action_steps frames, as MultiStepWrapper returns them after a step
    raw = np.concatenate([
        np.random.RandomState(0).rand(N_ACTION_STEPS, 20).astype(np.float32) * 512.0,
        np.ones((N_ACTION_STEPS, 20), dtype=np.float32)], axis=-1)[None]
    obs = R._obs_to_dict(raw)
    assert set(obs) == {'keypoint', 'agent_pos'}, 'eval_bon would hand the policy extra keys'
    with torch.no_grad():
        action = policy.predict_action(
            dict_apply(obs, lambda x: torch.from_numpy(x)))['action']
    assert action.shape == (1, N_ACTION_STEPS, 2)


def test_eval_bon_writes_no_curve_at_n1():
    """At n=1 every 'curve' is one point, so the figure must be suppressed.

    A deterministic policy's best-of-n equals its single sample by construction. Emitting
    `bon_curves.png` anyway would put a plot titled best-of-n next to a number that measured
    no spread at all, which is the kind of artifact that outlives the caveat explaining it.
    """
    eval_bon = _import_eval_bon()

    c = eval_bon.compute_curves(np.array([[0.4], [1.0], [0.0]]))
    assert len(c['n']) == 1
    assert c['bon_success'][-1] == c['mean_success'], (
        'best-of-1 must equal the single-sample mean')
    assert abs(c['mean_success'] - 1 / 3) < 1e-12
    src = pathlib.Path(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'eval_bon.py')).read_text()
    assert 'if n_samples > 1:\n        plot_curves(' in src, (
        'plot_curves must be guarded so n=1 writes no best-of-n figure')
