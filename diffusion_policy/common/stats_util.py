"""Small statistics shared by the eval path and the read-only reporting scripts.

Deliberately stdlib-only. `eval_search_pusht` costs ~88 s and ~440 MB to import because it
pulls torch, hydra and the sim stack; a script that only reads success_curve.json files and
prints a table should not pay that, and on the shared login cgroup the memory alone is a
hazard. Keeping the function here rather than duplicating it is what stops the reporting
scripts and the eval from drifting to two different intervals.
"""
import math


def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for a binomial rate.

    Reported alongside every success rate because the splits are small -- 50 test episodes
    give SE ~7pp and 10 val episodes ~16pp, so differences smaller than the interval are
    not distinguishable from sampling noise. Wilson rather than normal-approx because it
    stays inside [0,1] and behaves at p near 0 or 1, which is exactly where these land.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))
