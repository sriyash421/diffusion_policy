"""Print the success-rate curve of each BC expert arm, one row per checkpoint.

    python scripts/bc_expert_curves.py

The two arms are scored by different scripts and therefore write different files -- the
image arm's watcher is eval_search_pusht.py (bon_search/success_curves.jsonl, a best-of-n
curve pinned here to n=1) and the keypoint arm's is eval_bc_keypoint_pusht.py
(bc_eval/success_rates.jsonl). This reads both and puts them on one axis so a checkpoint can
be chosen by reading down the column.

NOTHING HERE NOMINATES A CHECKPOINT. It prints every measurement; the step is picked by hand
and named explicitly, as everything else in this repo does. The 176-demo manifest has NO val
split, so selecting on this column means selecting and reporting on the same 30 episodes --
the peak is optimistic by roughly a standard error, which is why the interval is printed
beside every point rather than only the rate.
"""
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get(
    'DP_OUTPUT_ROOT', '/gscratch/robotics/harine/diffusion_policy_outputs')) / 'pusht_search'

ARMS = {
    'image': ROOT / 'pusht_image_search_imgonly' / 'unet_bc'
             / 'unetbc_ver-t_goal_enc-resnet18_demos-176_seed-42'
             / 'bon_search' / 'success_curves.jsonl',
    'keypoint': ROOT / 'pusht_keypoint_manifest' / 'unet_bc'
                / 'unetbckp_demos-176_seed-42' / 'bc_eval' / 'success_rates.jsonl',
}


def _rows(path):
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = r.get('step')
        if step is None:
            continue
        if 'splits' in r:                       # eval_bc_keypoint_pusht.py
            t = r['splits'].get('test')
            if not t:
                continue
            out[step] = (t['success_rate'], tuple(t['ci95']),
                         t['n_success'], t['n_episodes'], t.get('mean_score'))
        elif 'success_rate' in r:               # eval_search_pusht.py, list parallel to `n`
            ns = r.get('n') or []
            if 1 not in ns:
                continue
            i = ns.index(1)
            rate = r['success_rate'][i]
            ci = tuple(r['success_ci'][i])
            n_ep = r.get('n_episodes') or 0
            out[step] = (rate, ci, int(round(rate * n_ep)), n_ep,
                         (r.get('mean_reward') or [None])[i])
    return out


def main():
    curves = {a: _rows(p) for a, p in ARMS.items()}
    steps = sorted({s for c in curves.values() for s in c})
    if not steps:
        print('no evaluated checkpoints yet')
        for a, p in ARMS.items():
            print(f'  {a:9s} {p} {"(exists)" if p.is_file() else "(not written yet)"}')
        return 1

    print(f'{"step":>8}  | {"IMAGE success (95% CI)":<30} | {"KEYPOINT success (95% CI)":<30}')
    print('-' * 8 + '--+-' + '-' * 30 + '-+-' + '-' * 30)
    for s in steps:
        cells = []
        for arm in ('image', 'keypoint'):
            v = curves[arm].get(s)
            if v is None:
                cells.append(f'{"-":<30}')
            else:
                rate, ci, k, n, _ = v
                cells.append(f'{rate:.3f} ({k}/{n})  [{ci[0]:.3f}, {ci[1]:.3f}]'.ljust(30))
        print(f'{s:>8}  | {cells[0]} | {cells[1]}')

    print()
    for arm in ('image', 'keypoint'):
        c = curves[arm]
        if not c:
            print(f'{arm:9s}: no rows yet')
            continue
        best = max(c, key=lambda s: c[s][0])
        rate, ci, k, n, _ = c[best]
        print(f'{arm:9s}: {len(c)} checkpoint(s) scored; highest so far {rate:.3f} '
              f'({k}/{n}) at step {best}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]')
    print('\nThe highest point is NOT a recommendation: it is selected on the same 30 '
          'episodes it is\nreported on, so it is optimistic. Overlapping intervals mean the '
          'steps are not distinguishable.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
