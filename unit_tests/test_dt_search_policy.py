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
