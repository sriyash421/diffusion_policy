"""The learned-Q verifier: the registry, the causal prefix z-score, and what must NOT move.

    pytest unit_tests/test_q_context.py

Two claims are load-bearing and both are written down here.

1. NOTHING ELSE MOVED. The whole learned-Q path sits behind `is_q_value`, so no simulated
   arm reaches any of it: `_score_candidates` returns before the Q branch, the normalizer
   buffers are never registered, and `sync_value_norm_from` is a no-op. `search_procedure.py`
   is untouched.

2. THE NORMALIZER USES STALE STATISTICS AND IS RANK-NEUTRAL. It normalizes with the running
   mean/var and updates after, never with the current batch's own statistics -- that is the
   BatchNorm trap, and it would leave the feature exactly 0 at deployment (batch of one) after
   looking fine for the whole of training. And the transform stays monotone in the raw Q, so
   the model is conditioned on the ordering argmax actually applies.

3. THE EMA COPY CARRIES THE STATISTICS. `EMAModel.step` averages parameters and never touches
   buffers, and evaluation scores the EMA weights -- so without `sync_value_norm_from` the arm
   would evaluate under statistics it never trained with, silently.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_policy.env.pusht.pusht_verifier import (        # noqa: E402
    PushTVerifier, Q_VERIFIERS, VALUE_FNS, VERIFIER_VALUES,
    check_verifier_value, is_q_value, q_verifier_spec)
from diffusion_policy.policy.pusht_search_mixin import PushTSearchMixin   # noqa: E402


# ---------------------------------------------------------------- the registry

def test_q_names_are_verifier_values_but_not_value_fns():
    """A learned value is selectable but is NOT a function of the reached state."""
    for name in Q_VERIFIERS:
        assert name in VERIFIER_VALUES
        assert name not in VALUE_FNS
        assert is_q_value(name)
    for name in VALUE_FNS:
        assert not is_q_value(name)


def test_registry_entries_resolve():
    """The name IS the binding, so a missing file is a broken run identity, not a fallback."""
    for name in Q_VERIFIERS:
        checkpoint, rung = q_verifier_spec(name)
        assert os.path.exists(checkpoint), f'{name} -> {checkpoint}'
        # PushTQVerifier reads `obs` and `tau_ladder` out of this and would otherwise build
        # against sac.config.DEFAULTS -- the wrong observation width for a non-default arm.
        assert os.path.exists(os.path.join(os.path.dirname(checkpoint), 'params', 'args.yaml'))
        assert isinstance(rung, int)


def test_unknown_q_name_raises():
    with pytest.raises(ValueError, match='not a learned value'):
        q_verifier_spec('q_sac_nonexistent')


def test_sim_verifier_refuses_a_learned_value():
    """The sim cannot score one, and says so at assignment rather than KeyErroring in rollout."""
    v = PushTVerifier(value_fn='t_goal')
    with pytest.raises(ValueError, match='LEARNED value'):
        v.value_fn = 'q_sac_all'
    assert v.value_fn == 't_goal'        # and the failed assignment changed nothing


def test_config_check_accepts_a_q_tag_and_rejects_a_mismatch():
    ok = {'verifier_tag': 'q_sac_all', 'policy': {'verifier_value': 'q_sac_all'}}
    check_verifier_value(ok)             # resolves the registry; raises if the file is gone

    with pytest.raises(ValueError, match='policy.verifier_value'):
        check_verifier_value({'verifier_tag': 'q_sac_all',
                              'policy': {'verifier_value': 't_goal'}})
    with pytest.raises(ValueError, match='not a known verifier value'):
        check_verifier_value({'verifier_tag': 'q_sac_nope',
                              'policy': {'verifier_value': 'q_sac_nope'}})


# ------------------------------------------------- the normalizer, on a bare stub

class _Stub(PushTSearchMixin):
    """Just enough policy to exercise the value normalizer -- no model, no weights."""

    def __init__(self, value='q_sac_all', training=True):
        self.search_kwargs = {'verifier_value': value, 'search_context': 'value'}
        self._value_norm_training = training
        self.q_norm_mean = torch.zeros(())
        self.q_norm_var = torch.ones(())
        self.q_norm_inited = torch.zeros((), dtype=torch.bool)

    # nn.Module.register_buffer is unavailable on a bare object; the three buffers above
    # stand in for it and `copy_`/`mul_`/`fill_` work identically on plain tensors.


def test_first_batch_seeds_the_statistics():
    """Seeded, not blended from (0, 1) -- the same idiom as obs_feature_std_inited."""
    s = _Stub()
    q = torch.tensor([0.70, 0.80, 0.90])
    s._normalize_q(q)
    assert bool(s.q_norm_inited)
    assert torch.allclose(s.q_norm_mean, q.mean())
    assert torch.allclose(s.q_norm_var, q.var(unbiased=False))


def test_normalizes_with_stale_statistics_then_updates():
    """THE BATCHNORM TRAP. The first batch must NOT be normalized by its own mean.

    If it were, a constant batch would map to exactly 0 -- and at deployment every batch is
    one episode, so every batch is effectively constant and the feature is always 0.
    """
    s = _Stub()
    q = torch.full((4,), 0.77)
    out = s._normalize_q(q)                   # stats are still (0, 1) at this point
    assert torch.allclose(out, q), 'the first batch must use the PRIOR statistics'
    assert not torch.allclose(out, torch.zeros_like(out))
    # ...and only now do the statistics reflect it
    assert torch.allclose(s.q_norm_mean, torch.tensor(0.77))


def test_batch_of_one_is_not_zero():
    """The deployment case, stated directly: B=1 must still carry signal."""
    s = _Stub()
    s._normalize_q(torch.tensor([0.5, 0.9]))           # seed the statistics
    s._value_norm_training = False                      # eval: frozen
    out = s._normalize_q(torch.tensor([0.9]))
    assert out.abs().item() > 1e-3, 'a single-sample batch collapsed to ~0'


def test_decay_is_applied_after_seeding():
    s = _Stub()
    s._normalize_q(torch.tensor([1.0, 1.0]))            # seeds mean 1, var 0
    s._normalize_q(torch.tensor([0.0, 0.0]))            # blends toward mean 0
    d = PushTSearchMixin.Q_NORM_DECAY
    assert torch.allclose(s.q_norm_mean, torch.tensor(d), atol=1e-6)


def test_frozen_when_not_training():
    """Off by default, so an eval script cannot drift the statistics with its own order."""
    s = _Stub(training=False)
    before = (s.q_norm_mean.clone(), s.q_norm_var.clone(), bool(s.q_norm_inited))
    s._normalize_q(torch.tensor([0.1, 0.9, 0.5]))
    assert torch.equal(s.q_norm_mean, before[0])
    assert torch.equal(s.q_norm_var, before[1])
    assert bool(s.q_norm_inited) == before[2]


def test_defaults_to_frozen():
    """A policy nobody told is not training. set_value_norm_training is the only way in."""
    s = _Stub()
    del s._value_norm_training
    s._value_norm_training = False                      # what __init__ sets
    s._normalize_q(torch.tensor([0.5]))
    assert not bool(s.q_norm_inited)
    s.set_value_norm_training(True)
    s._normalize_q(torch.tensor([0.5]))
    assert bool(s.q_norm_inited)


def test_clamped_to_five_sigma():
    """An early-training spike must not reach action_value_emb at full magnitude."""
    s = _Stub()
    s.q_norm_mean = torch.tensor(0.77)
    s.q_norm_var = torch.tensor(1e-6)                   # a poor early variance estimate
    s.q_norm_inited = torch.ones((), dtype=torch.bool)
    out = s._normalize_q(torch.tensor([0.77 + 1.0, 0.77 - 1.0]))
    assert torch.allclose(out, torch.tensor([5.0, -5.0]))


def test_ordering_within_a_decision_is_preserved():
    """The model and the selector must agree on which candidate is better.

    The transform is affine with a positive scale SHARED by every candidate of a decision,
    so it is monotone in the raw Q. Ranking is separately guaranteed by search_candidates
    keeping `scores` raw, but if these two disagreed the context would teach an ordering
    argmax does not apply. Run at the MEASURED scale (mean 0.77, within-decision spread
    0.0011), where saturation is not in play.
    """
    rng = np.random.default_rng(0)
    s = _Stub()
    s.q_norm_mean = torch.tensor(0.77)
    s.q_norm_var = torch.tensor(0.0011 ** 2)
    s.q_norm_inited = torch.ones((), dtype=torch.bool)
    s._value_norm_training = False                      # hold the statistics fixed
    for _ in range(200):
        q = torch.as_tensor(rng.normal(0.77, 0.0011, size=8), dtype=torch.float32)
        out = s._normalize_q(q)
        assert np.array_equal(np.argsort(q.numpy()), np.argsort(out.numpy()))


def test_one_normalizer_serves_every_slot():
    """Per-slot statistics would manufacture a slot-index signal out of nothing.

    One instance is called once per candidate, so the same (mean, var) divides every slot of
    a decision -- which is what makes a slot's context comparable to its neighbours'.
    """
    s = _Stub()
    s.q_norm_mean = torch.tensor(0.77)
    s.q_norm_var = torch.tensor(0.0011 ** 2)
    s.q_norm_inited = torch.ones((), dtype=torch.bool)
    s._value_norm_training = False
    same = torch.tensor([0.775])
    assert torch.equal(s._normalize_q(same), s._normalize_q(same)), \
        'the same Q at two slots must give the same context'


def test_sync_copies_rather_than_averages():
    """EMAModel never touches buffers, and eval scores the EMA copy."""
    src, dst = _Stub(), _Stub()
    src.q_norm_mean = torch.tensor(0.77)
    src.q_norm_var = torch.tensor(0.0011 ** 2)
    src.q_norm_inited = torch.ones((), dtype=torch.bool)
    dst.sync_value_norm_from(src)
    assert torch.equal(dst.q_norm_mean, src.q_norm_mean)
    assert torch.equal(dst.q_norm_var, src.q_norm_var)
    assert bool(dst.q_norm_inited)


def test_sync_is_a_noop_without_the_buffers():
    """Every simulated-value arm goes through the same workspace line."""
    class _Bare:
        pass
    _Stub().sync_value_norm_from(_Bare())        # must not raise
    _Stub(value='t_goal').sync_value_norm_from(_Bare())


def test_normalize_value_refuses_a_learned_value():
    """There is no reached state to mirror, and silently mirroring the CURRENT one would
    hand every candidate the same number -- a constant context, invisible in the loss."""
    with pytest.raises(ValueError, match='cannot mirror'):
        _Stub()._normalize_value(torch.zeros(4, PushTVerifier.STATE_DIM))


def test_subgoal_context_is_refused_for_a_learned_value():
    """A Q reaches nothing, so there is no observation for a subgoal context to encode."""
    s = _Stub()
    s.search_kwargs['search_context'] = 'subgoal'
    with pytest.raises(ValueError, match='no reached observation'):
        s._init_learned_value()
    s.search_kwargs['search_context'] = 'value'
    s._init_learned_value()                    # the supported one


def test_both_hosts_construct_under_every_value():
    """THE INIT-ORDER REGRESSION, written down.

    The two search hosts set `search_kwargs` at different points: the transformer arms set it
    inside their own __init__, PushTUNetSearchPolicy sets it AFTER super().__init__() returns.
    A mixin __init__ that reads it unconditionally therefore raises AttributeError on every
    UNet BC load -- which is every existing t_goal BC checkpoint, not just a learned-Q one.
    Caught in a best-of-N sweep, not by a test; hence this test.
    """
    import inspect

    from diffusion_policy.policy.pusht_unet_search_policy import PushTUNetSearchPolicy

    src = inspect.getsource(PushTSearchMixin.__init__)
    assert "hasattr(self, 'search_kwargs')" in src, \
        'the mixin must tolerate a host that has not set search_kwargs yet'
    unet = inspect.getsource(PushTUNetSearchPolicy.__init__)
    assert '_init_learned_value()' in unet, \
        'the UNet host must run the setup itself once search_kwargs exists'
    assert unet.index('self.search_kwargs = search_kwargs') < unet.index('_init_learned_value()'), \
        'the UNet host must set search_kwargs BEFORE running the setup'


def test_init_learned_value_is_idempotent():
    """Both hosts may reach it; registering the buffers twice must not raise or reset them."""
    s = _Stub()
    s.q_norm_mean = torch.tensor(0.77)
    s._init_learned_value()
    assert torch.equal(s.q_norm_mean, torch.tensor(0.77)), 'a second call reset the statistics'


def test_simulated_arms_register_no_value_norm_buffers():
    """No existing checkpoint's state_dict gains keys it does not have."""
    from diffusion_policy.policy.pusht_unet_search_policy import PushTUNetSearchPolicy
    import inspect
    src = inspect.getsource(PushTSearchMixin._init_learned_value)
    assert src.index('is_q_value') < src.index('register_buffer'), \
        'the buffers must stay behind the learned-value branch'
    assert PushTUNetSearchPolicy is not None


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
