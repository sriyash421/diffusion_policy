"""Exercise the `index` selection rule -- execute a fixed candidate, scores ignored.

    python unit_tests/test_selection_index.py     # prints which candidate each rule picks
    pytest unit_tests/test_selection_index.py

`selection_index` is 1-BASED, the way "the 8th candidate" is said out loud; negatives count
from the end. CPU-only, no policy needed -- select_candidate is a pure function.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_policy.common.selection_util import (      # noqa: E402
    SELECTION_MODES, select_candidate)

B, N, H, DA = 3, 16, 4, 2


def _fixture():
    """actions[b, k] is the constant k, so the picked candidate is readable off the value."""
    actions = torch.arange(N, dtype=torch.float32).view(1, N, 1, 1).expand(B, N, H, DA)
    scores = torch.zeros(B, N)
    scores[:, 3] = 1.0                      # argmax would take candidate 3 (0-based)
    return actions.contiguous(), scores


def _picked(mode, index=None):
    a, s = _fixture()
    return select_candidate(a, s, mode, index=index)[:, 0, 0]


def test_index_is_one_based():
    assert torch.equal(_picked('index', 8), torch.full((B,), 7.0)), \
        'the 8th candidate is 0-based index 7'
    assert torch.equal(_picked('index', 1), torch.zeros(B))


def test_negative_counts_from_the_end():
    assert torch.equal(_picked('index', -1), torch.full((B,), float(N - 1)))
    assert torch.equal(_picked('index', -8), torch.full((B,), float(N - 8)))


def test_scores_are_ignored():
    """The whole point: no verifier involvement, unlike argmax."""
    assert torch.equal(_picked('argmax'), torch.full((B,), 3.0))   # scores DO drive argmax
    assert torch.equal(_picked('index', 8), torch.full((B,), 7.0))


def test_out_of_range_raises_rather_than_picking_another_slot():
    a, s = _fixture()
    a, s = a[:, :4], s[:, :4]              # n = 4
    for bad in (8, -8, 0):
        try:
            select_candidate(a, s, 'index', index=bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'index {bad} at n=4 must raise, not silently pick')


def test_index_needs_an_index():
    a, s = _fixture()
    try:
        select_candidate(a, s, 'index')
    except AssertionError:
        pass
    else:
        raise AssertionError('selection index with no selection_index must raise')


def test_mode_is_registered():
    assert 'index' in SELECTION_MODES


def print_table():
    print(f'\nselect_candidate at n={N}; actions[b,k] == k, scores peak at candidate 3\n')
    print(f"{'rule':<22} {'executes candidate (0-based)':>30}")
    for label, mode, idx in (('argmax', 'argmax', None),
                             ('index 1 (1st)', 'index', 1),
                             ('index 8 (8th)', 'index', 8),
                             ('index -1 (last)', 'index', -1),
                             ('index -8 (8th from end)', 'index', -8)):
        print(f'{label:<22} {int(_picked(mode, idx)[0].item()):>30}')


if __name__ == '__main__':
    print_table()
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f'[ok] {t.__name__}')
    print(f'\n{len(tests)} tests passed')
