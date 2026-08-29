"""Exercise the per-slot observation-corruption ladder (`slot_obs_noise`) and print it.

    python unit_tests/test_slot_obs_noise.py     # prints the three ladders, then asserts
    pytest unit_tests/test_slot_obs_noise.py

Runs on a stub carrying nothing but `max_actions` and an obs noise scheduler, so it needs no
model, dataset, or GPU. Under test is `_slot_obs_timesteps`, the exact call the policy makes
to build its ladder:

    slot 0 is the MOST corrupted (highest t), slot K-1 the least, because slot k conditions
    on the first k scored context candidates.
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


if __name__ == '__main__':
    _print_table()
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'  ok  {name}')
    print('\nall assertions passed')
