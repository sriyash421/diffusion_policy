"""Validate the per-step selection traces written by `eval_search_pusht --criteria-sweep`.

The load-bearing check is the LAST one: for `argmax-all`, the recorded `chosen_idx` must
equal `scores.argmax(-1)` at every step. That is the cheapest proof that the trace and the
action actually executed correspond -- if the selection code and the trace-recording code
ever disagree, every plot built off these files is wrong and nothing else would say so.

It has to be an identity check on a criterion whose pick is DETERMINISTIC, which is why
argmax is used rather than softmax; the softmax criteria are checked only for staying inside
the pool they are allowed to draw from.

Usage: python scripts/check_criteria_traces.py <.../criteria_search/step_XXXXXXX>
"""
import pathlib
import sys

import numpy as np

EXPECT_WINDOW = {          # criterion -> lowest candidate index it may ever pick
    'cand-last': None,     # checked by exact index instead
    'cand-8th-from-last': None,
    'argmax-all': 0,
    'argmax-last8': 8,
    'softmax-all': 0,
    'softmax-last8': 8,
}
PAD = -2


def main(step_dir):
    step_dir = pathlib.Path(step_dir)
    tdir = step_dir.joinpath('traces')
    if not tdir.is_dir():
        print(f'no traces/ under {step_dir}')
        return 1
    fail = []
    for label in EXPECT_WINDOW:
        p = tdir.joinpath(f'{label}.npz')
        if not p.is_file():
            print(f'{label:<20} MISSING')
            fail.append(label)
            continue
        z = np.load(p)
        scores, pick = z['scores'], z['chosen_idx']
        valid, K = z['valid_len'], int(z['n_candidates'])
        n_ep, T = pick.shape
        # mask out the padded tail: every env is stepped until ALL are done, so a short
        # episode's tail is padding and would otherwise fail every check below
        m = np.arange(T)[None, :] < valid[:, None]

        ok_shape = scores.shape == (n_ep, T, K)
        ok_pad = bool((pick[~m] == PAD).all()) if (~m).any() else True
        ok_range = bool(((pick[m] >= 0) & (pick[m] < K)).all())
        ok_scores = bool(np.isfinite(scores[m]).all())

        lo = EXPECT_WINDOW[label]
        ok_window = True if lo is None else bool((pick[m] >= lo).all())
        if label == 'cand-last':
            ok_window = bool((pick[m] == K - 1).all())
        elif label == 'cand-8th-from-last':
            ok_window = bool((pick[m] == K - 8).all())

        # the identity: argmax criteria must reproduce exactly
        ok_identity = True
        if label == 'argmax-all':
            ok_identity = bool((pick[m] == scores.argmax(-1)[m]).all())
        elif label == 'argmax-last8':
            ok_identity = bool((pick[m] == (scores[..., K - 8:].argmax(-1) + K - 8)[m]).all())

        good = all([ok_shape, ok_pad, ok_range, ok_scores, ok_window, ok_identity])
        print(f'{label:<20} eps={n_ep} T={T} K={K} '
              f'shape={"ok" if ok_shape else "BAD"} pad={"ok" if ok_pad else "BAD"} '
              f'range={"ok" if ok_range else "BAD"} finite={"ok" if ok_scores else "BAD"} '
              f'window={"ok" if ok_window else "BAD"} '
              f'identity={"ok" if ok_identity else "BAD"}  '
              f'{"PASS" if good else "FAIL"}')
        if not good:
            fail.append(label)

    # the verifier value is -(T-to-goal + arm-to-T distance), so it must be <= 0 and
    # higher = better;
    # a sign flip anywhere upstream would show up here and silently invert every plot
    z = np.load(tdir.joinpath('argmax-all.npz'))
    s = z['scores'][np.isfinite(z['scores'])]
    print(f'\nscore range [{s.min():.3f}, {s.max():.3f}] '
          f'(verifier value is NEGATIVE distance: <= 0, higher is better)')
    if s.max() > 0:
        print('  WARNING: positive scores present -- check the sign convention before plotting')

    print('\nRESULT:', 'PASS' if not fail else f'FAIL {fail}')
    return 0 if not fail else 1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
