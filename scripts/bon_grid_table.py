"""Render the checkpoint x selection x n grid from the sweep log.

Rows are (checkpoint, selection); columns are n. Reads whatever has completed so far,
so it is safe to run against an in-progress sweep.

    python scripts/bon_grid_table.py [--metric reward|success] [--split test|val]
"""
import argparse, re, sys

NS = [1, 2, 4, 8, 16]


def parse(path):
    txt = open(path).read()
    secs = re.split(r'=== \S+ (\w+) (step_\d+) (\w+) ===', txt)
    out = {}
    for i in range(1, len(secs), 4):
        arm, step, sel, body = secs[i], secs[i+1], secs[i+2], secs[i+3]
        for m in re.finditer(
                r'(val|test) n=(\d+): success_rate=([\d.]+).*?mean reward max=([\d.]+)', body):
            out[(arm, step, sel, int(m.group(2)), m.group(1))] = (
                float(m.group(3)), float(m.group(4)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='logs/bon_grid_30demo.log')
    ap.add_argument('--metric', choices=['reward', 'success'], default='reward')
    ap.add_argument('--split', choices=['test', 'val'], default='test')
    a = ap.parse_args()
    idx = 1 if a.metric == 'reward' else 0
    d = parse(a.log)
    if not d:
        sys.exit('no completed cells yet')
    for arm in ('search', 'bc'):
        steps = sorted({s for (ar, s, _, _, _) in d if ar == arm})
        if not steps:
            continue
        print(f'\n### {arm.upper()}  --  {a.split} {a.metric}\n')
        print('| checkpoint | selection | ' + ' | '.join(f'n={n}' for n in NS) + ' |')
        print('|---|---|' + '---|' * len(NS))
        for st in steps:
            for sel in ('argmax', 'softmax', 'final_pass'):
                cells = []
                for n in NS:
                    v = d.get((arm, st, sel, n, a.split))
                    cells.append(f'{v[idx]:.3f}' if v else '-')
                if set(cells) == {'-'}:
                    continue
                print(f'| {int(st.split("_")[1]):,} | {sel} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
