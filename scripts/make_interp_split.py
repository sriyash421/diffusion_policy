"""Interpolation test set from the BC failures on a random-train manifest.

The parent manifest (pusht_rand50_train50_test156: 50 random training starts, the other 156 held
out) trains one U-Net BC full300, sampled under 3 eval seeds (scripts/slurm/eval_bc_fulltraj.sbatch).
A held-out start is a BC FAILURE when candidate 1 fails under at least `--min-fail` of the seeds:
not a success (coverage < 0.95), or with `--fail-cov t`, coverage < t. The child manifest keeps the
parent's train and takes as test up to `--n-fail` failures (default `--n-test`; a seeded random pick if
there are more), filled up to `--n-test` with seeded random other held-out starts. Val is empty.

Also writes <name>.bcfail.json beside the manifest: the BC-failure episodes and, per child test
start, each seed's candidate-1 coverage and success, for the plots to split by group.

    python scripts/make_interp_split.py --bc-run <unet run> --step 50000 [--fail-cov 0.3 --n-fail 25 --name ...]
"""
import json
import pathlib
import sys

import click
import numpy as np
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from make_geometric_splits import SPLITS, finish

SEED_DIRS = {0: 'dt_eval', 1: 'dt_eval_seed1', 2: 'dt_eval_seed2'}


def first_candidate(bc_run, step, sub):
    z = np.load(pathlib.Path(bc_run) / 'bon_search' / sub / f'step_{step:07d}' / 'test_candidates.npz',
                allow_pickle=True)
    return {int(e): (float(c), bool(s))
            for e, c, s in zip(z['episode_idxs'], z['max_coverage'][:, 0], z['success'][:, 0])}


@click.command()
@click.option('--bc-run', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--step', required=True, type=int)
@click.option('--parent', default='pusht_rand50_train50_test156', show_default=True)
@click.option('--name', default='pusht_rand50_train50_bcfail50', show_default=True)
@click.option('--n-test', default=50, show_default=True)
@click.option('--min-fail', default=2, show_default=True, help='seeds (of 3) whose candidate 1 must fail')
@click.option('--fail-cov', default=None, type=float, help='failure = coverage below this (default: not a success)')
@click.option('--n-fail', default=None, type=int, help='failures to take into test (default: up to --n-test)')
@click.option('--seed', default=42, show_default=True)
def main(bc_run, step, parent, name, n_test, min_fail, fail_cov, n_fail, seed):
    par = json.loads((SPLITS / f'{parent}.json').read_text())
    held = sorted(par['test'])
    per_seed = {s: first_candidate(bc_run, step, d) for s, d in SEED_DIRS.items()}
    for s, r in per_seed.items():
        assert sorted(r) == held, f'seed {s}: BC eval episodes differ from {parent} test'

    failed = (lambda cov, ok: cov < fail_cov) if fail_cov is not None else (lambda cov, ok: not ok)
    rule = f'coverage < {fail_cov}' if fail_cov is not None else 'not a success (coverage < 0.95)'
    n_fail_seeds = {e: sum(failed(*per_seed[s][e]) for s in per_seed) for e in held}
    fails = [e for e in held if n_fail_seeds[e] >= min_fail]
    others = [e for e in held if n_fail_seeds[e] < min_fail]
    rng = np.random.default_rng(seed)
    pick_fail = sorted(rng.permutation(fails)[:min(n_fail or n_test, n_test)].tolist()) if fails else []
    pick_other = sorted(rng.permutation(others)[:n_test - len(pick_fail)].tolist())
    assert len(pick_fail) + len(pick_other) == n_test, f'only {len(pick_fail) + len(pick_other)} starts for {n_test}'
    test = pick_fail + pick_other

    ends =np.asarray(zarr.open(str(ROOT / par['zarr_path']), 'r')['meta']['episode_ends'][:])
    method = (f'test = BC-failure starts of {parent} (U-Net BC full300 {pathlib.Path(bc_run).name} '
              f'step {step}: candidate 1 {rule} under >= {min_fail} of eval seeds {sorted(SEED_DIRS)}), '
              f'a seeded random {len(pick_fail)} of them, filled to {n_test} with seeded random other held-out '
              f'starts; train = the parent\'s; val empty')
    extra = dict(parent=parent, bc_run=str(bc_run), bc_step=step, fail_rule=rule, min_fail=min_fail, seed=seed,
                 n_bc_fail_held_out=len(fails), n_bc_fail_test=len(pick_fail))
    finish(name, method, extra, par['train'], [], test, ends, par['zarr_path'],
           generated_by='scripts/make_interp_split.py')

    side = dict(manifest=f'{name}.json', bc_run=str(bc_run), bc_step=step, fail_rule=rule, min_fail=min_fail,
                eval_seeds=sorted(SEED_DIRS), bc_fail_held_out=fails, bc_fail_test=pick_fail,
                per_start={str(e): dict(bc_fail=e in pick_fail, n_seeds_failed=n_fail_seeds[e],
                                        cov=[round(per_seed[s][e][0], 4) for s in sorted(per_seed)],
                                        success=[per_seed[s][e][1] for s in sorted(per_seed)])
                           for e in test})
    (SPLITS / f'{name}.bcfail.json').write_text(json.dumps(side, indent=1) + '\n')
    print(f'BC failures: {len(fails)} of {len(held)} held-out starts '
          f'(rule: {rule}; by seeds failed: {np.bincount(list(n_fail_seeds.values()), minlength=4).tolist()} for 0..3); '
          f'test = {len(pick_fail)} failures + {len(pick_other)} others')


if __name__ == '__main__':
    main()
