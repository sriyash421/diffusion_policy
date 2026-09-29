"""Exercise the slot-weight profiles and the curriculum, and print every vector.

    python unit_tests/test_slot_weight_profiles.py     # prints the tables, then asserts
    pytest unit_tests/test_slot_weight_profiles.py

CPU-only, on a stub carrying nothing but `max_actions` -- no model, dataset or GPU.

The invariant everything here turns on: every profile is renormalized to mean exactly 1, so
switching profiles, mirroring one, or interpolating between two never moves the loss SCALE
that gradient_clip_norm, the effective step size and val_loss all read.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_policy.policy.search_procedure import SearchProcedureMixin  # noqa: E402

CPU, F32 = torch.device('cpu'), torch.float32
K = 16
R = 100.0

# THE ROUND-2 CURRICULUM: 30k steps on uniform weights, then linear r=100 for the
# remaining 70k. interp='step' so each profile is actually TRAINED ON for its whole stretch
# rather than touched for an instant on the way past.
CURRICULUM = {'mode': 'curriculum', 'interp': 'step', 'waypoints': [
    {'step': 0, 'mode': 'uniform'},
    {'step': 30000, 'mode': 'linear', 'ratio': R},
]}

# A three-shape curriculum, kept as a machinery test: it is the only thing exercising
# `tent`, `reverse` and a >2-waypoint schedule, none of which the round-2 runs use.
WAYPOINTS = [
    {'step': 0, 'mode': 'linear', 'ratio': R, 'reverse': True},
    {'step': 20000, 'mode': 'tent', 'ratio': R},
    {'step': 40000, 'mode': 'linear', 'ratio': R},
]
THREE_PHASE = {'mode': 'curriculum', 'waypoints': WAYPOINTS, 'interp': 'step'}


class _Stub(SearchProcedureMixin):
    def __init__(self, sw, max_actions=K):
        self.max_actions = max_actions
        self._init_slot_weighting(False, sw)


def _w(sw, step=None):
    return _Stub(sw)._slot_weights(CPU, F32, step=step)


def _mean1(v):
    return abs(v.mean().item() - 1.0) < 1e-5


def test_linear_ratio_100():
    v = _w({'mode': 'linear', 'ratio': R})
    assert _mean1(v), v.mean()
    assert abs(v[0].item() - 0.0198) < 1e-3, v[0]
    assert abs(v[-1].item() - 1.9802) < 1e-3, v[-1]
    assert abs((v[-1] / v[0]).item() - R) < 0.5


def test_geometric_decay_735():
    v = _w({'mode': 'geometric', 'decay': 0.735})
    assert _mean1(v), v.mean()
    assert abs(v[0].item() - 0.0422) < 1e-3, v[0]
    assert abs(v[-1].item() - 4.271) < 1e-2, v[-1]
    # the shape claim: it stays under 1 until slot 10, then spikes
    assert bool((v[:10] < 1.0).all()) and bool((v[11:] > 1.0).all()), v


def test_tent_is_symmetric_and_peaks_in_the_middle():
    v = _w({'mode': 'tent', 'ratio': R})
    assert _mean1(v), v.mean()
    assert torch.allclose(v, torch.flip(v, dims=(0,)), atol=1e-6), 'tent must be symmetric'
    assert int(v.argmax()) in (K // 2 - 1, K // 2), v.argmax()
    assert v[0].item() < v[K // 2].item()


def test_reverse_mirrors_and_keeps_mean_1():
    fwd = _w({'mode': 'linear', 'ratio': R})
    rev = _w({'mode': 'linear', 'ratio': R, 'reverse': True})
    assert torch.allclose(rev, torch.flip(fwd, dims=(0,)), atol=1e-6)
    assert _mean1(rev), rev.mean()
    assert rev[0].item() > rev[-1].item(), 'reversed linear must be slot-0 heavy'


def test_round2_curriculum_durations():
    """30k uniform, then linear r=100 to 100k -- each HELD, not passed through."""
    flat = torch.ones(K)
    lin = _w({'mode': 'linear', 'ratio': R})
    for step, want, name in (
            (0, flat, 'phase 1 start'), (15000, flat, 'phase 1 mid'),
            (29999, flat, 'phase 1 last step'),
            (30000, lin, 'phase 2 STARTS at its own waypoint step'),
            (65000, lin, 'phase 2 mid'), (100000, lin, 'phase 2 end'),
            (250000, lin, 'past the end, held')):
        assert torch.allclose(_w(CURRICULUM, step=step), want, atol=1e-6), name


def test_three_phase_durations():
    """The tent/reverse machinery: 20k slot-0 heavy, 20k tent, then slot-15 heavy."""
    mirror = _w({'mode': 'linear', 'ratio': R, 'reverse': True})
    tent = _w({'mode': 'tent', 'ratio': R})
    lin = _w({'mode': 'linear', 'ratio': R})
    for step, want, name in (
            (0, mirror, 'phase 1 start'), (19999, mirror, 'phase 1 last step'),
            (20000, tent, 'phase 2 STARTS at its own waypoint step'),
            (39999, tent, 'phase 2 last step'),
            (40000, lin, 'phase 3 start'), (100000, lin, 'phase 3 end')):
        assert torch.allclose(_w(THREE_PHASE, step=step), want, atol=1e-6), name


def test_curriculum_is_mean_1_at_every_step():
    """The whole point: the loss scale must not move while the profile does."""
    for sw in (CURRICULUM, THREE_PHASE,
               {'mode': 'curriculum', 'waypoints': WAYPOINTS}):
        for step in range(0, 110001, 2500):
            v = _w(sw, step=step)
            assert _mean1(v), (sw['interp'] if 'interp' in sw else 'linear',
                               step, v.mean().item())


def test_curriculum_interpolates_between_waypoints():
    """interp='linear' is still available and still interpolates."""
    sw = {'mode': 'curriculum', 'waypoints': WAYPOINTS}
    a = _w(sw, step=20000)
    b = _w(sw, step=40000)
    mid = _w(sw, step=30000)
    assert torch.allclose(mid, 0.5 * (a + b), atol=1e-5), 'midpoint must be the average'
    assert not torch.allclose(mid, a, atol=1e-3)


def test_curriculum_mass_moves_slot0_to_middle_to_last():
    peaks = [int(_w(THREE_PHASE, step=s).argmax()) for s in (0, 20000, 40000)]
    assert peaks[0] == 0, peaks
    assert peaks[1] in (K // 2 - 1, K // 2), peaks
    assert peaks[2] == K - 1, peaks


def test_bad_curricula_raise():
    for sw, why in (
            ({'mode': 'curriculum'}, 'no waypoints'),
            ({'mode': 'curriculum', 'waypoints': [dict(WAYPOINTS[0])]}, 'only one waypoint'),
            ({'mode': 'curriculum', 'waypoints': list(reversed(WAYPOINTS))},
             'steps not increasing'),
            ({'mode': 'curriculum', 'waypoints': WAYPOINTS,
              'schedule': {'shape': 'linear', 'start_step': 0, 'end_step': 10}},
             'schedule AND waypoints'),
            ({'mode': 'linear', 'ratio': R, 'waypoints': WAYPOINTS}, 'waypoints without curriculum'),
            ({'mode': 'linear', 'ratio': R, 'interp': 'step'}, 'interp without curriculum'),
            ({'mode': 'curriculum', 'waypoints': WAYPOINTS, 'interp': 'cosine'}, 'bad interp'),
            ({'mode': 'curriculum', 'waypoints': [{'step': 0, 'mode': 'tent'},
                                                  dict(WAYPOINTS[2])]}, 'tent with no ratio'),
            ({'mode': 'curriculum', 'waypoints': [{'step': 0, 'mode': 'linear', 'rato': 4},
                                                  dict(WAYPOINTS[2])]}, "typo'd waypoint key"),
            ({'mode': 'tent'}, 'tent with no ratio at top level'),
    ):
        try:
            _w(sw)
        except (ValueError, TypeError, AssertionError):
            pass
        else:
            raise AssertionError(f'must raise: {why}')


def test_existing_profiles_are_unchanged():
    """The round-1 arms must still resolve to exactly what they trained under."""
    v = _w({'mode': 'linear', 'ratio': 4.857})
    assert abs(v[0].item() - 0.3415) < 1e-3 and abs(v[-1].item() - 1.6585) < 1e-3, v
    g = _w({'mode': 'geometric', 'decay': 0.9})
    assert abs(g[0].item() - 0.4044) < 1e-3 and abs(g[-1].item() - 1.9639) < 1e-3, g
    assert _Stub({'mode': 'uniform'})._slot_weights(CPU, F32) is None


def print_tables():
    def row(name, v):
        print(f'{name:<26} ' + ' '.join(f'{x:5.3f}' for x in v.tolist())
              + f'   mean {v.mean():.4f}')
    print(f'\nProfiles at K={K}\n')
    row('linear r=100', _w({'mode': 'linear', 'ratio': R}))
    row('geometric d=0.735', _w({'mode': 'geometric', 'decay': 0.735}))
    row('tent r=100', _w({'mode': 'tent', 'ratio': R}))
    row('linear r=100 reversed', _w({'mode': 'linear', 'ratio': R, 'reverse': True}))
    print(f'\nRound-2 curriculum (30k uniform, then linear r=100), by training step\n')
    for st in (0, 15000, 29999, 30000, 60000, 100000):
        row(f'step {st:,}', _w(CURRICULUM, step=st))


if __name__ == '__main__':
    print_tables()
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    print()
    for t in tests:
        t()
        print(f'[ok] {t.__name__}')
    print(f'\n{len(tests)} tests passed')
