#!/usr/bin/env python
"""Compare pi0 eval arms from their episodes.jsonl.

    python scripts/simpler/veritas_pi0/summarize.py <campaign_dir or arm dirs...>

A campaign dir is scanned one level deep for episodes.jsonl. Prints success rate with a Wilson
95% CI, mean steps and, for the paper protocol, the per-variation success rates side by side.
"""
import glob
import json
import math
import os
import sys


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(args):
    files = []
    for a in args:
        if os.path.isfile(a):
            files.append(a)
        elif os.path.isfile(os.path.join(a, 'episodes.jsonl')):
            files.append(os.path.join(a, 'episodes.jsonl'))
        else:
            files += sorted(glob.glob(os.path.join(a, '*', 'episodes.jsonl')))
    if not files:
        sys.exit('no episodes.jsonl found')
    arms = {}
    print(f"{'arm':<16} {'n':>4} {'succ':>5} {'SR':>7} {'95% CI':>15} {'steps':>6}")
    for fp in files:
        rows = load(fp)
        name = os.path.basename(os.path.dirname(fp))
        n = len(rows)
        k = sum(r['success'] for r in rows)
        lo, hi = wilson_ci(k, n)
        steps = sum(r['steps'] for r in rows) / n if n else float('nan')
        print(f"{name:<16} {n:>4} {k:>5} {k / n if n else float('nan'):>7.3f} "
              f"[{lo:.3f}, {hi:.3f}] {steps:>6.1f}")
        arms[name] = rows
    per_var = {}
    for name, rows in arms.items():
        for r in rows:
            if r.get('variation') is None:
                continue
            per_var.setdefault(r['variation'], {}).setdefault(name, []).append(r['success'])
    if per_var:
        names = list(arms)
        print('\nper-variation success rate')
        print(f"{'var':>4} " + ' '.join(f'{n:>12}' for n in names))
        for v in sorted(per_var):
            cells = []
            for n in names:
                s = per_var[v].get(n)
                cells.append(f'{sum(s) / len(s):>8.2f}({len(s):>2})' if s else f"{'-':>12}")
            print(f'{v:>4} ' + ' '.join(cells))


if __name__ == '__main__':
    main(sys.argv[1:])
