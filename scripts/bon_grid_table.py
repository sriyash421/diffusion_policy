"""Render the policy x selection x n grid from the sweep log.

Rows are (policy, checkpoint, selection); columns are n. Checkpoint groups the selection
rules so the three are read side by side -- in particular their n=1 cells, which must agree
exactly. Reads whatever has completed so far, so it is safe against an in-progress sweep.

    python scripts/bon_grid_table.py [--metric reward|success] [--split test|val]
"""
import argparse, re, sys

NS = [1, 2, 4, 8, 16, 32, 64]
ARMS = (('search', 'ST-diffusion-k16'), ('bc', 'BC'))
SELECTIONS = ('argmax', 'softmax', 'final_pass')


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

    print(f'### {a.split.upper()}  --  {a.metric}\n')
    print('| policy | checkpoint | selection | ' + ' | '.join(f'n={n}' for n in NS) + ' |')
    print('|---|---|---|' + '---|' * len(NS))
    # Repeated policy/checkpoint labels are blanked, so each checkpoint reads as one block
    # of three selection rules rather than the same two words on every row.
    prev = (None, None)
    for arm, label in ARMS:
        for st in sorted({s for (ar, s, _, _, _) in d if ar == arm}):
            ckpt = f'{int(st.split("_")[1]):,}'
            for sel in SELECTIONS:
                cells = [(lambda v: f'{v[idx]:.3f}' if v else '-')(
                    d.get((arm, st, sel, n, a.split))) for n in NS]
                if set(cells) == {'-'}:
                    continue
                pol_c = label if prev[0] != label else ''
                ck_c = ckpt if prev != (label, ckpt) else ''
                prev = (label, ckpt)
                print(f'| {pol_c} | {ck_c} | {sel} | ' + ' | '.join(cells) + ' |')

if __name__ == '__main__':
    main()
