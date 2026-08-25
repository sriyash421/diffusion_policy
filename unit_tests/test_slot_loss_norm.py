"""Exercise the per-slot loss norm (`slot_loss_norm`) and print the per-candidate weights.

    python unit_tests/test_slot_loss_norm.py     # prints the table, then asserts
    pytest unit_tests/test_slot_loss_norm.py

Runs on a stub carrying nothing but `max_actions`, so it needs no model, dataset, or GPU.
The blend under test is `_slot_norm_loss`, the exact call `_compute_loss` makes:

    loss_k = (1 - alpha_k) * (pred - target)**2 + alpha_k * |pred - target|
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_policy.policy.search_procedure import SearchProcedureMixin  # noqa: E402

CPU, F32 = torch.device('cpu'), torch.float32
K = 16


class _Stub(SearchProcedureMixin):
    """The only state _slot_norm_alphas / _slot_weights read."""

    def __init__(self, max_actions=K, slot_loss_norm=None, slot_weights=None):
        self.max_actions = max_actions
        self._init_slot_loss_norm(slot_loss_norm)
        self._init_slot_weighting(False, slot_weights)


def _alphas(mode, max_actions=K):
    return _Stub(max_actions, {'mode': mode})._slot_norm_alphas(CPU, F32)


def _blend(stub, err, slot_weighting=True):
    """The production elementwise loss, fed pred = err and target = 0."""
    return stub._slot_norm_loss(err, torch.zeros_like(err), slot_weighting)


def test_l2_is_the_untouched_path():
    assert _alphas('l2') is None, 'l2 must return None so the MSE path stays bit-identical'
    assert _alphas('l2tol1', max_actions=1) is None, 'K=1 has nothing to interpolate'


def test_l1_is_all_ones():
    assert torch.equal(_alphas('l1'), torch.ones(K))


def test_l2tol1_endpoints_and_monotonicity():
    a = _alphas('l2tol1')
    assert a.shape == (K,), a.shape
    assert a[0].item() == 0.0, 'slot 0 must be pure L2'
    assert a[-1].item() == 1.0, 'slot K-1 must be pure L1'
    assert torch.all(a[1:] > a[:-1]), a
    assert torch.allclose(a, torch.arange(K, dtype=F32) / (K - 1))


def test_blend_matches_the_two_pure_norms_at_the_ends():
    torch.manual_seed(0)
    err = torch.randn(4, K, 8, 2)
    stub = _Stub(slot_loss_norm={'mode': 'l2tol1'})
    per_slot = _blend(stub, err).mean(dim=(0, 2, 3))
    # slot_weighting=False is the canonical objective: plain MSE, whatever the config says
    assert torch.equal(_blend(stub, err, slot_weighting=False), err.pow(2))
    assert torch.allclose(per_slot[0], err[:, 0].pow(2).mean(), atol=1e-6)
    assert torch.allclose(per_slot[-1], err[:, -1].abs().mean(), atol=1e-6)
    # and every slot in between is the exact convex combination
    a = _alphas('l2tol1')
    for k in range(K):
        want = (1 - a[k]) * err[:, k].pow(2).mean() + a[k] * err[:, k].abs().mean()
        assert torch.allclose(per_slot[k], want, atol=1e-6), k


def test_composes_with_slot_weights():
    """The norm picks the shape of each term; slot_weights then scales the terms."""
    torch.manual_seed(0)
    err = torch.randn(4, K, 8, 2)
    stub = _Stub(slot_loss_norm={'mode': 'l2tol1'},
                 slot_weights={'mode': 'geometric', 'decay': 0.9})
    w = stub._slot_weights(CPU, F32)
    per_slot = _blend(stub, err).mean(dim=(2, 3))                            # (B, K)
    got = (per_slot * w).mean()

    B = err.shape[0]
    ref = 0.0
    for k in range(K):
        a_k = k / (K - 1)
        term = (1 - a_k) * err[:, k].pow(2) + a_k * err[:, k].abs()
        ref = ref + w[k].item() * term.mean(dim=(1, 2)).sum().item()
    assert abs(got.item() - ref / (B * K)) < 1e-6, (got.item(), ref / (B * K))
    assert not torch.allclose(got, per_slot.mean()), \
        'the weighting must actually change the scalar'


def test_typos_and_bad_modes_raise():
    """A silently ignored key would train plain L2 while the config claims otherwise."""
    try:
        _Stub(slot_loss_norm={'mdoe': 'l2tol1'})
    except TypeError:
        pass
    else:
        raise AssertionError('an unknown slot_loss_norm key must raise')
    try:
        _Stub(slot_loss_norm={'mode': 'huber'})
    except ValueError:
        pass
    else:
        raise AssertionError('an unknown slot_loss_norm mode must raise')


def print_table(max_actions=K):
    stub = _Stub(max_actions, {'mode': 'l2tol1'})
    a = stub._slot_norm_alphas(CPU, F32)
    torch.manual_seed(0)
    err = torch.randn(64, max_actions, 8, 2)
    per_slot = _blend(stub, err).mean(dim=(0, 2, 3))
    l2 = err.pow(2).mean(dim=(0, 2, 3))
    l1 = err.abs().mean(dim=(0, 2, 3))
    print(f'\nslot_loss_norm: l2tol1, K={max_actions}   '
          f'(loss_k = (1-a)*e^2 + a*|e|, on e ~ N(0,1))\n')
    print(f"{'slot':>4} {'a (L1)':>8} {'1-a (L2)':>9} {'pure L2':>9} {'pure L1':>9} "
          f"{'blended':>9}")
    for k in range(max_actions):
        print(f'{k:>4} {a[k]:>8.4f} {1 - a[k]:>9.4f} {l2[k]:>9.4f} {l1[k]:>9.4f} '
              f'{per_slot[k]:>9.4f}')
    print(f"\nmean alpha {a.mean():.4f} -- NOT renormalized to 1: these pick a norm, they "
          f"do not\nsplit a fixed loss budget the way slot_weights does.")


if __name__ == '__main__':
    print_table()
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f'[ok] {t.__name__}')
    print(f'\n{len(tests)} tests passed')
