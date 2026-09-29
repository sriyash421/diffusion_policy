"""Exercise the per-slot observation-corruption ladder (`slot_obs_noise`) and print it.

    python unit_tests/test_slot_obs_noise.py     # prints the three ladders, then asserts
    pytest unit_tests/test_slot_obs_noise.py

Runs on a stub carrying nothing but `max_actions` and an obs noise scheduler, so it needs no
model, dataset, or GPU. Under test is `_slot_obs_timesteps`, the exact call the policy makes
to build its ladder:

    slot 0 is the MOST corrupted (highest t), slot K-1 the least, because slot k conditions
    on the first k scored context candidates.

`mode: random_base` has no fixed ladder to print: it borrows a shape and rescales it into
`[0, t_base]` for a `t_base` drawn per sample, so slot 0's corruption is random and the
schedule runs from it down to clean. What is asserted there is that every draw is that same
shape under a positive scaling.
"""
import os
import sys

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_policy.policy.search_procedure import SearchProcedureMixin  # noqa: E402

CPU = torch.device('cpu')
K = 16


class _Stub(SearchProcedureMixin):
    """The only state _slot_obs_timesteps / _init_slot_obs_noise read."""

    def __init__(self, slot_obs_noise=None, max_actions=K, corrupt_obs=False):
        self.max_actions = max_actions
        self.corrupt_obs = corrupt_obs
        # Same scheduler the ST policy builds for obs corruption.
        self.obs_noise_scheduler = DDPMScheduler(
            num_train_timesteps=100, beta_start=0.001, beta_end=0.02,
            prediction_type='epsilon')
        self._init_slot_obs_noise(slot_obs_noise)

    def sig(self, t=None):
        """sqrt(alpha_bar) at each slot's timestep -- the factor the obs is scaled by."""
        t = self._slot_obs_timesteps() if t is None else t
        return self.obs_noise_scheduler.alphas_cumprod[t].sqrt()


SHAPES = {
    'linear_t': {'mode': 'linear_t'},
    'geometric d=0.7': {'mode': 'geometric', 'decay': 0.7},
    'linear_signal': {'mode': 'linear_signal'},
}


def _print_table():
    stubs = {n: _Stub(c) for n, c in SHAPES.items()}
    ts = {n: s._slot_obs_timesteps() for n, s in stubs.items()}
    sg = {n: s.sig() for n, s in stubs.items()}
    print(f'\nPer-slot ladder, K={K}, T=100, betas 0.001->0.02\n')
    print('slot | ' + ' | '.join(f'{n:>17}' for n in SHAPES))
    print('     | ' + ' | '.join('    t   sqrt(abar)' for _ in SHAPES))
    for k in range(K):
        cells = ' | '.join(f'  {int(ts[n][k]):3d}   {sg[n][k]:.3f}   ' for n in SHAPES)
        print(f'  {k:2d} | {cells}')
    print()
    for n in SHAPES:
        step = (sg[n][1:] - sg[n][:-1]).abs()
        dup = int(step.lt(5e-3).sum())
        print(f'{n:>17}: step in sqrt(abar) min {step.min():.3f} max {step.max():.3f} | '
              f'{dup}/{K - 1} adjacent pairs within 0.005')


def test_ladders_are_ordered():
    """Slot 0 the noisiest, monotone non-increasing to slot K-1. True of every shape."""
    for name, cfg in SHAPES.items():
        t = _Stub(cfg)._slot_obs_timesteps()
        assert t.shape == (K,), f'{name}: got {tuple(t.shape)}'
        assert int(t[0]) == 99, f'{name}: slot 0 must sit at the noisiest timestep'
        assert bool((t[1:] <= t[:-1]).all()), f'{name}: not monotone: {t.tolist()}'


def test_linear_signal_is_evenly_spaced():
    """The point of linear_signal: equal steps in RETAINED SIGNAL, not in the index."""
    s = _Stub(SHAPES['linear_signal'])
    step = s.sig()[1:] - s.sig()[:-1]
    assert bool((step > 0).all()), 'signal must strictly increase toward the clean end'
    assert float(step.max() - step.min()) < 0.01, \
        f'steps not equal to tolerance: {step.tolist()}'
    assert int(s._slot_obs_timesteps()[-1]) == 0, 'slot K-1 must be clean'


def test_linear_t_is_uneven_in_signal():
    """The contrast that motivated linear_signal: even in t is NOT even in corruption."""
    step = _Stub(SHAPES['linear_t']).sig().diff()
    assert step.max() / step.min() > 3.0, \
        'linear_t should spend its steps unevenly; if not, the two shapes are redundant'


def test_geometric_collapses_its_clean_end():
    """Documented weakness, asserted so it cannot regress into a silent surprise."""
    sg = _Stub(SHAPES['geometric d=0.7']).sig()
    assert int(sg.diff().abs().lt(5e-3).sum()) >= 5, \
        'geometric at decay 0.7 is expected to leave many slots indistinguishable'


def test_off_and_degenerate_widths_return_none():
    assert _Stub(None)._slot_obs_timesteps() is None                    # unset
    assert _Stub({'mode': 'uniform'})._slot_obs_timesteps() is None     # explicit off
    # width 1 has a single slot, so there is no ladder -- same rule _slot_weights uses.
    assert _Stub(SHAPES['linear_signal'], max_actions=1)._slot_obs_timesteps() is None


def test_explicit_list():
    prof = list(range(99, 99 - K, -1))
    t = _Stub({'mode': 'list', 'timesteps': prof})._slot_obs_timesteps()
    assert t.tolist() == prof
    try:
        _Stub({'mode': 'list', 'timesteps': [1, 2]})._slot_obs_timesteps()
    except ValueError as e:
        assert 'must name every slot' in str(e)
    else:
        raise AssertionError('a short explicit profile must raise')


def test_reversed_list_is_flagged(capsys=None):
    """`list` is the only mode that can point the ladder the wrong way; it must say so."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _Stub({'mode': 'list', 'timesteps': list(range(K))})   # ASCENDING == reversed
    assert 'not non-increasing' in buf.getvalue(), buf.getvalue()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _Stub({'mode': 'list', 'timesteps': list(range(99, 99 - K, -1))})
    assert 'not non-increasing' not in buf.getvalue()


def test_config_validation():
    """A typo'd key must raise, not silently train an uncorrupted run."""
    for cfg, exc, frag in (
        ({'mode': 'linear_signl'}, ValueError, 'must be one of'),
        ({'mod': 'linear_signal'}, TypeError, 'unknown key'),
        ({'mode': 'geometric'}, ValueError, 'needs a `decay`'),
        ({'mode': 'geometric', 'decay': 1.0}, ValueError, 'use mode: uniform'),
    ):
        try:
            _Stub(cfg)
        except exc as e:
            assert frag in str(e), f'{cfg}: wrong message {e}'
        else:
            raise AssertionError(f'{cfg} should have raised {exc.__name__}')


def test_refuses_to_stack_with_corrupt_obs():
    """Both surfaces noise obs_cond; enabling both would corrupt it twice."""
    try:
        _Stub(SHAPES['linear_signal'], corrupt_obs=True)
    except ValueError as e:
        assert 'noises it twice' in str(e)
    else:
        raise AssertionError('corrupt_obs + slot_obs_noise must raise')


# ------------------------------------------------------------------ mode: random_base
#
# Slot 0's level is drawn per sample and the borrowed shape is rescaled into [0, base], so
# there is no fixed ladder to assert. What has to hold is that every draw is the same shape
# under a positive scaling: monotone, clean at slot K-1, and exactly the fixed ladder when
# the base is drawn at the top of its range.

RANDOM_BASE = {'mode': 'random_base', 'shape': 'linear_signal'}
T = 100


def test_random_base_has_no_fixed_ladder():
    """It registers a shape profile, not timesteps: the levels do not exist until a draw."""
    s = _Stub(RANDOM_BASE)
    assert s._slot_obs_timesteps() is None, 'random_base must not pin a fixed ladder'
    sh = s._slot_obs_shape()
    assert sh is not None and sh.shape == (K,)
    assert int(sh[0]) == T - 1 and int(sh[-1]) == 0


def test_random_base_borrows_its_shape_exactly():
    """A random_base arm and its fixed counterpart must not be able to drift apart."""
    for shape, cfg in (('linear_t', {}), ('linear_signal', {}),
                       ('geometric', {'decay': 0.7})):
        fixed = _Stub({'mode': shape, **cfg})._slot_obs_timesteps()
        sh = _Stub({'mode': 'random_base', 'shape': shape, **cfg})._slot_obs_shape()
        assert torch.equal(fixed, sh), f'{shape}: random_base shape differs from the fixed ladder'


def test_random_base_at_max_base_is_the_fixed_ladder():
    """base = T-1 is the identity rescale, so the arm degenerates to its shape there."""
    s = _Stub(RANDOM_BASE)
    lo, hi = s.slot_obs_base_range()
    assert (lo, hi) == (0, T - 1)
    t = s.rescale_slot_timesteps(s._slot_obs_shape(), hi)
    assert torch.equal(t, _Stub(SHAPES['linear_signal'])._slot_obs_timesteps())


def test_random_base_every_draw_is_ordered_and_ends_clean():
    s = _Stub(RANDOM_BASE)
    sh = s._slot_obs_shape()
    for base in range(0, T):
        t = s.rescale_slot_timesteps(sh, base)
        assert bool((t[1:] <= t[:-1]).all()), f'base {base}: not monotone: {t.tolist()}'
        assert int(t[0]) == base, f'base {base}: slot 0 must sit AT the drawn level'
        assert int(t[-1]) == 0, f'base {base}: slot K-1 must stay clean'


def test_random_base_extent_scales_with_the_draw():
    """The point of the mode: a lower base is a SHORTER ladder, not a shifted one."""
    s = _Stub(RANDOM_BASE)
    sh = s._slot_obs_shape()
    spans = [int(s.rescale_slot_timesteps(sh, b).max()) for b in (20, 50, 99)]
    assert spans == [20, 50, 99], spans
    # and the mid-slot level is monotone in the base
    mids = [int(s.rescale_slot_timesteps(sh, b)[K // 2]) for b in (20, 50, 99)]
    assert mids[0] < mids[1] < mids[2], mids


def test_random_base_is_batched():
    """A (B,) base gives a (B, K) ladder, one row per sample."""
    s = _Stub(RANDOM_BASE)
    base = torch.tensor([10, 55, 99])
    t = s.rescale_slot_timesteps(s._slot_obs_shape(), base)
    assert t.shape == (3, K)
    for row, b in zip(t, base.tolist()):
        assert int(row[0]) == b and int(row[-1]) == 0


def test_random_base_config_validation():
    for cfg, exc, frag in (
        ({'mode': 'random_base', 'shape': 'list'}, ValueError, 'shape must be one of'),
        ({'mode': 'random_base', 'shape': 'geometric'}, ValueError, 'needs a `decay`'),
        ({'mode': 'random_base', 'shape': 'linear_t', 'decay': 0.7},
         ValueError, 'only means something under shape: geometric'),
        ({'mode': 'random_base', 'base_range': [50, 10]}, ValueError, 'base_range must be'),
        # shape/base_range are meaningless on the fixed ladders and must not be ignored
        ({'mode': 'linear_signal', 'shape': 'linear_t'}, ValueError, 'only mean something'),
        ({'mode': 'linear_signal', 'base_range': [0, 9]}, ValueError, 'only mean something'),
    ):
        try:
            _Stub(cfg)
        except exc as e:
            assert frag in str(e), f'{cfg}: wrong message {e}'
        else:
            raise AssertionError(f'{cfg} should have raised {exc.__name__}')


def test_random_base_off_at_width_one():
    assert _Stub(RANDOM_BASE, max_actions=1)._slot_obs_shape() is None


# ---------------------------------------------------- the schedule sets the noise FLOOR
#
# The shapes decide how the range is distributed across slots; the SCHEDULE decides how
# corrupted the noisiest slot can be at all. These pin that, because it is the thing that
# silently caps the whole method: on the legacy T=100 schedule no shape, cap or timestep can
# take slot 0 below 59% retained signal.

def _stub_on(T, b0, b1, cfg=None):
    s = _Stub.__new__(_Stub)
    s.max_actions = K
    s.corrupt_obs = False
    s.obs_noise_scheduler = DDPMScheduler(
        num_train_timesteps=T, beta_start=b0, beta_end=b1,
        beta_schedule='linear', prediction_type='epsilon')
    s._init_slot_obs_noise(cfg or SHAPES['linear_signal'])
    return s


def test_legacy_schedule_cannot_make_slot0_noise():
    """The default T=100 schedule retains ~59% of the signal at its noisiest timestep."""
    s = _stub_on(100, 0.001, 0.02)
    floor = float(s.obs_noise_scheduler.alphas_cumprod[-1].sqrt())
    assert 0.58 < floor < 0.60, floor
    assert float(s.sig()[0]) == pytest_approx(floor), 'slot 0 should sit AT the floor'


def test_tmrl_vla_schedule_drives_slot0_to_the_marginal():
    """TMRL's VLA schedule (T=1000, beta 1e-4->0.02) reaches ~0.6% signal -- pure noise."""
    s = _stub_on(1000, 1e-4, 0.02)
    floor = float(s.obs_noise_scheduler.alphas_cumprod[-1].sqrt())
    assert floor < 0.01, f'expected the marginal, got sqrt(abar)={floor}'
    sig = s.sig()
    assert float(sig[0]) == pytest_approx(floor), 'slot 0 should sit AT the floor'
    assert float(sig[-1]) > 0.99, 'slot K-1 should still be clean'


def test_linear_signal_still_grades_evenly_on_the_long_schedule():
    """The property linear_signal exists for must survive the 10x longer schedule."""
    step = _stub_on(1000, 1e-4, 0.02).sig().diff()
    assert bool((step > 0).all())
    assert float(step.max() / step.min()) < 1.1, \
        f'steps not even on T=1000: {step.tolist()}'


def test_linear_t_is_badly_uneven_on_the_long_schedule():
    """Documented weakness, asserted so the linear_t arm is read for what it is."""
    step = _stub_on(1000, 1e-4, 0.02, {'mode': 'linear_t'}).sig().diff()
    assert float(step.max() / step.min()) > 10.0


# ------------------------------------------------------------------------ max_t
#
# The ceiling on slot 0, in timesteps of the configured scheduler. It is what turns
# "linear_t with slot 0 at 400" into an arm of its own rather than a different shape.

def test_max_t_compresses_the_shape():
    full = _stub_on(1000, 1e-4, 0.02, {'mode': 'linear_t'})._slot_obs_timesteps()
    capped = _stub_on(1000, 1e-4, 0.02,
                      {'mode': 'linear_t', 'max_t': 400})._slot_obs_timesteps()
    assert int(full[0]) == 999, full[0]
    assert int(capped[0]) == 400, 'slot 0 must land exactly on max_t'
    assert int(capped[-1]) == 0, 'slot K-1 must stay clean'
    assert bool((capped[1:] <= capped[:-1]).all()), 'still monotone'


def test_max_t_matches_random_base_at_the_same_level():
    """The two must not drift: both go through rescale_slot_timesteps."""
    s = _stub_on(1000, 1e-4, 0.02, {'mode': 'linear_t', 'max_t': 400})
    rb = _stub_on(1000, 1e-4, 0.02, {'mode': 'random_base', 'shape': 'linear_t'})
    assert torch.equal(s._slot_obs_timesteps(),
                       rb.rescale_slot_timesteps(rb._slot_obs_shape(), 400))


def test_max_t_validation():
    for cfg, frag in (
        ({'mode': 'random_base', 'shape': 'linear_t', 'max_t': 400}, 'random_base sets'),
        ({'mode': 'uniform', 'max_t': 400}, 'nothing with mode: uniform'),
        ({'mode': 'linear_t', 'max_t': 0}, 'must be > 0'),
    ):
        try:
            _stub_on(1000, 1e-4, 0.02, cfg)
        except ValueError as e:
            assert frag in str(e), f'{cfg}: wrong message {e}'
        else:
            raise AssertionError(f'{cfg} should have raised')
    # a ceiling at or above the schedule length is a schedule mismatch, not a clamp
    try:
        _stub_on(100, 0.001, 0.02, {'mode': 'linear_t', 'max_t': 400})._slot_obs_timesteps()
    except ValueError as e:
        assert 'num_train_timesteps' in str(e), e
    else:
        raise AssertionError('max_t >= T should raise')


# ------------------------------------------------- the dispatch every mode has to pass
#
# The ladder is reached through TWO buffers: the fixed shapes register `slot_obs_t`,
# `random_base` registers `slot_obs_shape` and leaves `slot_obs_t` None by construction. A
# dispatch that tests `slot_obs_t` alone therefore sends random_base down the FLAT path,
# where corrupt_obs is False (the two are mutually exclusive) and the corruption is the
# identity -- the arm trains entirely clean while the startup print announces its ladder.
# That was live in _compute_loss. `slot_ladder_on` is the single predicate both sites read.


def _ladder_on(cfg):
    """`slot_ladder_on` evaluated on the real policy, without building one."""
    from diffusion_policy.policy.diffusion_transformer_search_policy import (
        DiffusionTransformerSearchPolicy)
    stub = _Stub(cfg)
    p = object.__new__(DiffusionTransformerSearchPolicy)
    p.slot_obs_t = stub._slot_obs_timesteps()
    p.slot_obs_shape = stub._slot_obs_shape()
    return p.slot_ladder_on


def test_every_non_uniform_mode_is_ladder_on():
    """random_base included -- it is the one whose fixed-ladder buffer is always None."""
    for cfg in (*SHAPES.values(), RANDOM_BASE,
                {'mode': 'random_base', 'shape': 'linear_t'},
                {'mode': 'random_base', 'shape': 'geometric', 'decay': 0.7},
                {'mode': 'linear_signal', 'max_t': 40},
                {'mode': 'list', 'timesteps': list(range(K - 1, -1, -1))}):
        assert _ladder_on(cfg), f'{cfg} must take the slotwise path, not the flat one'


def test_uniform_and_width_one_are_ladder_off():
    """Off must stay off: no buffers, so the state_dict of an unladdered arm is unchanged."""
    assert not _ladder_on({'mode': 'uniform'})
    assert not _ladder_on(None)


def test_random_base_registers_only_the_shape_buffer():
    """The asymmetry the bug turned on, pinned so a refactor cannot quietly reverse it."""
    s = _Stub(RANDOM_BASE)
    assert s._slot_obs_timesteps() is None and s._slot_obs_shape() is not None
    f = _Stub(SHAPES['linear_signal'])
    assert f._slot_obs_timesteps() is not None and f._slot_obs_shape() is None


def pytest_approx(x, tol=1e-4):
    class _A:
        def __eq__(self, other): return abs(other - x) < tol
    return _A()


if __name__ == '__main__':
    _print_table()
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'  ok  {name}')
    print('\nall assertions passed')
