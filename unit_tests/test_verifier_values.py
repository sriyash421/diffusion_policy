"""Pin what `d_t_goal` is: `t_goal` divided by its spread, and therefore rank-identical.

    python unit_tests/test_verifier_values.py
    pytest unit_tests/test_verifier_values.py

The whole design rests on the monotonicity claim -- it is why a checkpoint's `t_goal`
success curve IS its `d_t_goal` curve, and why the re-measured reference arms are a
consistency check rather than a new number. These tests are that claim, written down.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_policy.env.pusht.pusht_verifier import (        # noqa: E402
    T_GOAL_SPREAD, VALUE_FNS, VERIFIER_VALUES, base_value_fn, value_d_t_goal, value_t_goal)

RNG = np.random.default_rng(0)
N, N_KP = 64, 8


def _candidates(n=N):
    """(agent_pos, feedback) for n candidates, in the layout the verifier is handed."""
    return RNG.uniform(0, 512, (n, 2)), RNG.uniform(-200, 200, (n, 2 * N_KP))


def test_registered():
    assert 'd_t_goal' in VALUE_FNS and 'd_t_goal' in VERIFIER_VALUES
    # not cross-candidate, so it backs itself and IS a legal verifier_tag for training
    assert base_value_fn('d_t_goal') == 'd_t_goal'


def test_is_t_goal_over_the_spread():
    a, f = _candidates()
    assert np.allclose(value_d_t_goal(a, f), value_t_goal(a, f) / T_GOAL_SPREAD)


def test_ranking_is_identical_to_t_goal():
    """The claim the plan rests on: same argmax, same full ordering, on every draw."""
    for _ in range(200):
        a, f = _candidates()
        t, d = value_t_goal(a, f), value_d_t_goal(a, f)
        assert np.array_equal(np.argsort(t), np.argsort(d))
        assert int(np.argmax(t)) == int(np.argmax(d))


def test_still_non_positive_and_zero_only_at_the_goal():
    """t_goal's defining property survives the rescale -- a positive divisor cannot flip it."""
    a, f = _candidates()
    assert np.all(value_d_t_goal(a, f) <= 0)
    at_goal = np.zeros((1, 2 * N_KP))                 # feedback 0 == T exactly on the goal
    assert value_d_t_goal(np.zeros((1, 2)), at_goal)[0] == 0.0


def test_ignores_agent_pos_like_t_goal():
    """Inherited flatness: until the arm touches the block, candidates are indistinguishable."""
    _, f = _candidates()
    a1, a2 = RNG.uniform(0, 512, (N, 2)), RNG.uniform(0, 512, (N, 2))
    assert np.array_equal(value_d_t_goal(a1, f), value_d_t_goal(a2, f))


def test_magnitude_is_the_point():
    """It is NOT a no-op: the recorded scalar changes units, which is what d_t_goal buys."""
    a, f = _candidates()
    t, d = value_t_goal(a, f), value_d_t_goal(a, f)
    assert np.abs(d).max() < np.abs(t).max()
    assert np.allclose(np.abs(t).mean() / np.abs(d).mean(), T_GOAL_SPREAD)


def test_normalize_value_shares_the_t_goal_branch():
    """The context copy must be byte-identical to t_goal's, or the arms are incomparable."""
    import inspect

    from diffusion_policy.policy.pusht_search_mixin import PushTSearchMixin
    src = inspect.getsource(PushTSearchMixin._normalize_value)
    assert "if base in ('t_goal', 'd_t_goal'):" in src, \
        "d_t_goal must share t_goal's context branch; see value_d_t_goal"
    # and the guard that forced us to add it deliberately is still there
    assert "assert base == 'armT'" in src


if __name__ == '__main__':
    a, f = _candidates(8)
    print(f'\n{"cand":>4} {"t_goal (px)":>13} {"d_t_goal":>10}   ratio')
    for i, (t, d) in enumerate(zip(value_t_goal(a, f), value_d_t_goal(a, f))):
        print(f'{i:>4} {t:>13.3f} {d:>10.4f}   {t / d:.3f}')
    print(f'\nT_GOAL_SPREAD = {T_GOAL_SPREAD}; argmax t_goal = '
          f'{int(np.argmax(value_t_goal(a, f)))}, argmax d_t_goal = '
          f'{int(np.argmax(value_d_t_goal(a, f)))}\n')
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f'[ok] {t.__name__}')
    print(f'\n{len(tests)} tests passed')
