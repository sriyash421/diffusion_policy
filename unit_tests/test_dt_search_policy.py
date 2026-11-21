"""CPU tests for the PushT full-trajectory DT (PushTDTSearchPolicy).

    pytest unit_tests/test_dt_search_policy.py            (robodiff env: pytest + transformers)

Stubs only (FlattenObsEncoder on a low_dim shape_meta, no zarr), except the two `slow` tests
that roll out 2 sync PushT envs. What a training log cannot show: the stream mechanics, that the
KV-cached step equals the full-stream forward, and that the rolled-out context reaches autograd.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_policy.model.common.normalizer import LinearNormalizer             # noqa: E402
from diffusion_policy.model.vision.multi_image_obs_encoder import FlattenObsEncoder  # noqa: E402
from diffusion_policy.policy.pusht_dt_search_policy import PushTDTSearchPolicy    # noqa: E402

SHAPE_META = {
    'obs': {'agent_pos': {'shape': [2], 'type': 'low_dim'},
            'feedback': {'shape': [16], 'type': 'low_dim'}},
    'action': {'shape': [2]},
}


def _normalizer():
    g = torch.Generator().manual_seed(0)
    norm = LinearNormalizer()
    norm.fit(data={'agent_pos': torch.rand((64, 2), generator=g) * 512,
                   'feedback': torch.rand((64, 16), generator=g) * 100 - 50,
                   'action': torch.rand((64, 2), generator=g) * 512},
             last_n_dims=1, mode='limits')
    return norm


def _policy(max_actions=1, backbone='gpt2', **kw):
    torch.manual_seed(0)
    p = PushTDTSearchPolicy(SHAPE_META, FlattenObsEncoder(SHAPE_META), horizon=1,
                            n_action_steps=1, n_obs_steps=1, max_actions=max_actions,
                            backbone=backbone, gpt2_n_positions=64,
                            n_emb=32, n_layer=1, n_head=2, p_drop=0.0, **kw)
    p.set_normalizer(_normalizer())
    p.eval()
    return p


def _obs(B, T=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {'agent_pos': torch.rand((B, T, 2), generator=g) * 512,
            'feedback': torch.rand((B, T, 16), generator=g) * 100 - 50}


def _step(p, B, fb, done, seed=None):
    o = _obs(B, seed=int(torch.randint(0, 10 ** 6, (1,))) if seed is None else seed)
    o['fb'] = torch.as_tensor(fb, dtype=torch.float32)
    o['done'] = torch.as_tensor(done, dtype=torch.bool)
    return p.predict_step(o)


@pytest.mark.parametrize('backbone', ['gpt2', 'te'])
def test_predict_step_masks_ended_rows_and_settles_boundary(backbone):
    p = _policy(max_actions=8, backbone=backbone)
    p.reset(); p.begin_attempt()
    a, active = _step(p, 3, [-1.0, -1.0, -1.0], [0, 0, 0])
    assert a.shape == (3, 2) and active.all()
    # row 1 is done by the env: it goes inactive, gets a zero action and its final feedback
    a, active = _step(p, 3, [-0.5, -0.2, -0.9], [0, 1, 0])
    assert active.tolist() == [True, False, True]
    assert torch.all(a[1] == 0)
    assert torch.allclose(p._st['last_fb'][1], torch.tensor([-0.2, -0.2]))
    assert p.attempt_stop_reason[1] == 2
    assert p._st['len'].tolist() == [2, 1, 2]      # the stream only grew for the active rows
    p.max_attempt_steps = 2                         # end the rest via the guard
    a, active = _step(p, 3, [-0.4, -0.2, -0.8], [0, 1, 0])
    assert not active.any() and p.attempt_stop_reason.tolist() == [3, 2, 3]
    # next attempt: boundary token carries d=1 and the final feedback on both channels
    p.begin_attempt()
    _step(p, 3, [-1.0, -1.0, -1.0], [0, 0, 0])
    S = p._st
    first = S['att_starts'][-1]
    assert S['prev_d'][torch.arange(3), first].flatten().tolist() == [1.0, 1.0, 1.0]
    assert torch.allclose(S['fb'][1, first[1]], torch.tensor([-0.2, -0.2]))


def test_gpt2_cached_step_matches_full_forward():
    """The action mean read off the KV cache equals the full-stream forward at that position,
    including after rows went inactive (masked dummy columns, explicit position ids)."""
    p = _policy(max_actions=8, backbone='gpt2')
    p.reset(); p.begin_attempt()
    dones = [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 1]]
    with torch.inference_mode():       # the stream was built under predict_step's inference_mode
        for i, d in enumerate(dones):
            _step(p, 3, [-0.5, -0.4, -0.3], d, seed=i)
            S = p._st
            h_full = p._hidden(S)[torch.arange(3), (S['len'] - 1).clamp_min(0)]
            assert torch.allclose(h_full, S['h_last'], atol=1e-4), f'step {i}'
        p.begin_attempt()                                 # a boundary token after the cache grew
        _step(p, 3, [-0.5, -0.4, -0.3], [0, 0, 0], seed=9)
        S = p._st
        h_full = p._hidden(S)[torch.arange(3), S['len'] - 1]
        assert torch.allclose(h_full, S['h_last'], atol=1e-4)


def test_window_slides_and_bon_resets():
    p = _policy(max_actions=2, backbone='gpt2')
    p.reset()
    for _ in range(3):
        p.begin_attempt()
        _step(p, 2, [-1.0, -1.0], [0, 0])
        _step(p, 2, [-1.0, -1.0], [1, 1])
    # at most max_actions-1 = 1 earlier attempt is kept besides the current one, and the
    # rebuilt cache agrees with the compact stream
    assert len(p._st['att_starts']) == 2
    # the rebuilt cache has exactly one live column per stream token (dummy columns are masked)
    assert (p._st['valid'].sum(1) == p._st['len']).all()
    p1 = _policy(max_actions=1)
    p1.reset(); p1.begin_attempt(); _step(p1, 2, [-1.0, -1.0], [0, 0])
    p1.begin_attempt()
    assert p1._st is None   # best-of-N: every attempt starts an empty stream


@pytest.mark.parametrize('backbone', ['gpt2', 'te'])
def test_k1_loss_ignores_padding(backbone, monkeypatch):
    p = _policy(max_actions=1, backbone=backbone)
    p.train()
    monkeypatch.setattr(p, '_expert_feedback', lambda obs, mask: -0.3 * torch.ones(mask.shape))
    B, T = 2, 6
    batch = {'obs': _obs(B, T), 'action': torch.rand(B, T, 2) * 512,
             'attention_mask': torch.tensor([[1] * 6, [1] * 4 + [0] * 2], dtype=torch.bool)}
    loss = p.compute_loss(batch)
    assert torch.isfinite(loss) and loss.ndim == 0
    batch2 = dict(batch); batch2['action'] = batch['action'].clone(); batch2['action'][1, 4:] = 7.0
    assert torch.allclose(loss, p.compute_loss(batch2))


def test_feedback_bounds():
    from diffusion_policy.env.pusht.pusht_step_feedback import D_T_MAX, PIN_MAX, step_feedback
    assert abs(D_T_MAX - 512 * np.sqrt(2)) < 1e-6
    assert abs(PIN_MAX - 96 * np.sqrt(2) * 1.75) < 1e-6
    fb = step_feedback(np.zeros((2, 2)), np.zeros((2, 16)), [None, None])
    assert fb.tolist() == [0.0, 0.0]            # T on the goal, no plan -> 0


@pytest.mark.slow
@pytest.mark.parametrize('backbone', ['gpt2', 'te'])
def test_rollout_candidates_in_live_env_and_k2_loss(backbone, monkeypatch):
    """Two sequential attempts in 2 sync PushT envs feed a k=3 loss (K=2 context attempts)."""
    p = _policy(max_actions=3, backbone=backbone, max_attempt_steps=4,
                rollout_n_envs=2, rollout_use_async=False)
    init = np.array([[200.0, 200.0, 300.0, 300.0, 0.5], [100.0, 400.0, 256.0, 256.0, 1.0]])
    cands = p.rollout_candidates(init, [None, None], 2)
    assert cands['values'].shape == (2, 2) and cands['n_in_stream'] == 2
    assert (cands['n_steps'] == 4).all()        # no stop head: every attempt runs to the guard
    assert len(cands['snaps']) == 2 and cands['stream']['feat'].shape[0] == 2
    p.train()
    monkeypatch.setattr(p, '_expert_feedback', lambda obs, mask: -0.3 * torch.ones(mask.shape))
    monkeypatch.setattr(p, '_init_states_and_plans', lambda obs: (init, [None, None]))
    B, T = 2, 5
    batch = {'obs': _obs(B, T), 'action': torch.rand(B, T, 2) * 512,
             'attention_mask': torch.tensor([[1] * 5, [1] * 3 + [0] * 2], dtype=torch.bool)}
    loss = p.compute_loss(batch)
    assert torch.isfinite(loss) and p.training
    loss.backward()            # the rolled-out context must be usable by autograd
    assert p.mean_head.weight.grad is not None
    p.close()


def test_coverage_terminal_feedback_on_boundary_token():
    p = _policy(max_actions=8, backbone='gpt2', terminal_feedback='coverage')
    p.reset(); p.begin_attempt()
    o = _obs(2); o['fb'] = torch.tensor([-0.5, -0.5]); o['done'] = torch.tensor([False, False])
    o['coverage'] = torch.tensor([0.3, 0.4])
    p.predict_step(o)
    o = _obs(2, seed=1); o['fb'] = torch.tensor([-0.2, -0.3]); o['done'] = torch.tensor([True, True])
    o['coverage'] = torch.tensor([0.96, 0.61])
    _, active = p.predict_step(o)
    assert not active.any()
    # channel 0 = verifier value at the final state, channel 1 = the env's max coverage
    assert torch.allclose(p._st['last_fb'], torch.tensor([[-0.2, 0.96], [-0.3, 0.61]]))
    p.begin_attempt()
    o = _obs(2, seed=2); o['fb'] = torch.tensor([-0.5, -0.5]); o['done'] = torch.tensor([False, False])
    o['coverage'] = torch.tensor([0.0, 0.0])
    p.predict_step(o)
    first = p._st['att_starts'][-1]
    assert torch.allclose(p._st['fb'][torch.arange(2), first], torch.tensor([[-0.2, 0.96], [-0.3, 0.61]]))
    # the value-mode policy repeats the verifier value and never needs `coverage`
    q = _policy(max_actions=8, backbone='gpt2')
    q.reset(); q.begin_attempt()
    o = _obs(2); o['fb'] = torch.tensor([-0.5, -0.5]); o['done'] = torch.tensor([False, False])
    q.predict_step(o)
    o = _obs(2, seed=1); o['fb'] = torch.tensor([-0.2, -0.3]); o['done'] = torch.tensor([True, True])
    q.predict_step(o)
    assert torch.allclose(q._st['last_fb'], torch.tensor([[-0.2, -0.2], [-0.3, -0.3]]))


def _cov_obs(B, fb, done, cov, seed):
    o = _obs(B, seed=seed)
    o.update(fb=torch.as_tensor(fb, dtype=torch.float32), done=torch.as_tensor(done, dtype=torch.bool),
             coverage=torch.as_tensor(cov, dtype=torch.float32))
    return o


def _state_attempt(p, lens):
    """One attempt in which row b ends (env done) after lens[b] actions."""
    B = len(lens)
    p.begin_attempt()
    for t in range(max(lens) + 1):
        p.predict_step(_cov_obs(B, [-0.1 * (t + 1)] * B, [t >= n for n in lens], [0.1 * t] * B, seed=t))


def test_state_context_handoff_matches_live_stream():
    """Context records prefilled into ANOTHER policy give the same next token as the live stream
    that made them (state tokens, boundary feedback), and the cached step equals the full forward."""
    p = _policy(max_actions=8, backbone='gpt2', terminal_feedback='coverage', context_obs='state')
    q = _policy(max_actions=8, backbone='gpt2', terminal_feedback='coverage', context_obs='state')
    q.load_state_dict(p.state_dict())
    with torch.inference_mode():
        p.reset()
        _state_attempt(p, [3, 2])
        recs = p.context_records()
        assert [r['att_lens'] for r in recs] == [[3], [2]]          # one token per action
        p.begin_attempt()
        assert p._st['ctx'][:, :3].bool().all()                    # the finished attempt is context now
        q.reset(); q.prefill(recs); q.begin_attempt()
        o = _cov_obs(2, [-0.5, -0.5], [False, False], [0.0, 0.0], seed=99)
        p.predict_step(dict(o)); q.predict_step(dict(o))
        assert torch.allclose(p._st['h_last'], q._st['h_last'], atol=1e-4)
        S = q._st
        h_full = q._hidden(S)[torch.arange(2), S['len'] - 1]
        assert torch.allclose(h_full, S['h_last'], atol=1e-4)
        # the boundary token carries [verifier value at the end, max coverage]
        assert torch.allclose(S['fb'][torch.arange(2), S['att_starts'][-1]],
                              torch.stack([r['last_fb'][-1] for r in recs]))
    both = p.append_context(recs[0], recs[0])
    assert both['att_lens'] == [3, 3] and both['tok']['state'].shape == (6, 6)


def test_state_context_loss(monkeypatch):
    p = _policy(max_actions=8, backbone='gpt2', terminal_feedback='coverage', context_obs='state')
    with torch.inference_mode():
        p.reset()
        _state_attempt(p, [3, 1])
        recs = p.context_records()
    recs = [{**r, 'tok': {k: v.clone() for k, v in r['tok'].items()},
             'last_a': r['last_a'].clone(), 'last_fb': r['last_fb'].clone()} for r in recs]
    monkeypatch.setattr(p, '_expert_feedback', lambda obs, mask: -0.3 * torch.ones(mask.shape))
    p.train()
    B, T = 2, 5
    batch = {'obs': _obs(B, T), 'action': torch.rand(B, T, 2) * 512,
             'attention_mask': torch.tensor([[1] * 5, [1] * 3 + [0] * 2], dtype=torch.bool)}
    p.eval()       # no dropout: compare the two n=0 paths exactly
    loss0 = p.compute_loss(batch, context=[None, None])
    p.max_actions = 1
    assert torch.allclose(loss0, p.compute_loss(batch))
    p.max_actions = 8
    loss = p.compute_loss(batch, context=recs)
    assert torch.isfinite(loss) and not torch.allclose(loss, loss0)
    loss.backward()
    assert p.state_proj.weight.grad is not None and p.state_proj.weight.grad.abs().sum() > 0


def test_image_context_obs_keeps_state_dict():
    assert not any(k.startswith('state_proj') for k in _policy(max_actions=8).state_dict())


def test_mixed_width_context_loss(monkeypatch):
    """A batch whose rows hold 0, 1 and 2 context attempts trains; a batch of only empty contexts
    equals the no-context loss; truncate_context keeps the per-attempt boundary values."""
    p = _policy(max_actions=8, backbone='gpt2', terminal_feedback='coverage', context_obs='state')
    with torch.inference_mode():
        p.reset()
        _state_attempt(p, [3, 2, 2])
        r1 = p.context_records()
        p.begin_attempt()
        _state_attempt(p, [1, 2, 3])
        r2 = p.context_records()
    clone = lambda r: {**r, 'tok': {k: v.clone() for k, v in r['tok'].items()},
                       'last_a': r['last_a'].clone(), 'last_fb': r['last_fb'].clone()}
    ctx = [clone(p.append_context(a, b)) for a, b in zip(r1, r2)]
    assert ctx[0]['att_lens'] == [3, 1] and ctx[0]['last_fb'].shape == (2, 2)
    one = p.truncate_context(ctx[1], 1)
    assert one['att_lens'] == [2] and torch.equal(one['last_fb'], ctx[1]['last_fb'][:1])
    assert p.truncate_context(ctx[2], 0) is None
    monkeypatch.setattr(p, '_expert_feedback', lambda obs, mask: -0.3 * torch.ones(mask.shape))
    B, T = 3, 4
    batch = {'obs': _obs(B, T), 'action': torch.rand(B, T, 2) * 512,
             'attention_mask': torch.ones(B, T, dtype=torch.bool)}
    p.eval()
    empty = p.compute_loss(batch, context=[None, None, None])
    p.max_actions = 1
    assert torch.allclose(empty, p.compute_loss(batch))
    p.max_actions = 8
    mixed = p.compute_loss(batch, context=[None, one, ctx[2]])
    assert torch.isfinite(mixed) and not torch.allclose(mixed, empty)
    mixed.backward()
    assert p.state_proj.weight.grad.abs().sum() > 0
    with pytest.raises(AssertionError):
        p.prefill([None, one, ctx[2]])


def test_feature_noise_loss(monkeypatch):
    """feat_noise_t noises only the demo's features: t = 0 stays near clean, a large t changes the
    loss, per-sample timesteps backprop, and the feature-std EMA is set."""
    p = _policy(max_actions=8, backbone='gpt2', terminal_feedback='coverage', context_obs='state')
    monkeypatch.setattr(p, '_expert_feedback', lambda obs, mask: -0.3 * torch.ones(mask.shape))
    B, T = 3, 4
    batch = {'obs': _obs(B, T), 'action': torch.rand(B, T, 2) * 512,
             'attention_mask': torch.ones(B, T, dtype=torch.bool)}
    p.eval()
    ctx = [None] * B
    clean = p.compute_loss(batch, context=ctx)
    torch.manual_seed(0)
    near = p.compute_loss(batch, context=ctx, feat_noise_t=torch.zeros(B, dtype=torch.long))
    assert torch.allclose(clean, near, atol=1e-2)
    noisy = p.compute_loss(batch, context=ctx, feat_noise_t=torch.tensor([600, 400, 200]))
    assert torch.isfinite(noisy) and not torch.allclose(noisy, clean)
    noisy.backward()
    assert p._feat_std.shape == (p.obs_feature_dim,)
