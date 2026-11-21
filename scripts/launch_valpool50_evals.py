"""Held-out-train evals for the far-50 runs: every checkpoint the test sweeps scored, re-scored
on the val of its quadrant's `*_train50_far_valpool50` manifest (50 training-region episodes
the run never trained on; same train and test as the run's own manifest).

Phase `select`: what checkpoint selection needs -- ST argmax N = 1..64 at every step (clean,
plus corrupted obs for the slot-noise arms), BC argmax N = 1 (its rule), and the 6
full-trajectory blq50far seed-42 arms (eval_dt_pusht, 8 attempts) -- plus the full curves at
the test-best step (BC argmax N = 1..64, ST final_pass), known before any of it runs. Phase `final`: at each
run's held-out-train-best step, ST final_pass and BC argmax N = 1..64. Phase `train`: the 50
training demos themselves at both chosen steps. Phase `cand`: the
report's candidate metrics (scripts/candidate_rmse_pusht.py), blq50far ST arms, test windows at
the test-best step and held-out-train windows at the held-out-train-best step. Phase `rollout`:
the closed-loop side of those metrics, same arms and steps -- argmax at N = 4/8/16/64 with every
candidate's verifier value stored (eval_search_pusht --store-scores), plus videos of episodes
0-2 at those N side by side for seed 42.

Cells run in priority order -- (blq50far, 42), (tlq50far, 42), (brq50far, 42), then seeds
43, 44 -- as ONE throttled array, whose tasks start in index order. Re-running skips cells
whose results are on disk.

    python scripts/launch_valpool50_evals.py [--phase select|final|train|cand|rollout]   # dry run
    python scripts/launch_valpool50_evals.py --phase select --submit
"""
import argparse
import collections
import json
import pathlib
import subprocess

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path('/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search')
IMG = OUT / 'pusht_image_search_imgonly'
AUDIT = OUT / 'pusht_valpool50_20261029'
SPLITS = ROOT / 'diffusion_policy/config/splits'
QUADS = {'blq50far': 'bottomleft', 'tlq50far': 'topleft', 'brq50far': 'bottomright'}
SEEDS = (42, 43, 44)
NS = [1, 2, 4, 8, 16, 32, 64]
WP = 'wp_v3_la_tgn'
# label, run-dir template, noisy (slot-noise arm: corrupted-obs readout too)
CHUNKED = [
    ('BC', 'unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-50_split-{q}_seed-{s}', False),
    ('ST k1', 'outer_inner/value_k1_ver-wp_v3_la_tgn_ts-term_enc-resnet18_demos-50_split-{q}_seed-{s}_fresh_context_v3', False),
    ('ST k16 uniform', 'outer_inner/value_k16_ver-wp_v3_la_tgn_ts-term_enc-resnet18_demos-50_split-{q}_seed-{s}_fresh_context_v3', False),
    ('ST k16 flat600', 'outer_inner/value_k16_ver-wp_v3_la_tgn_ts-term_son-flat600_enc-resnet18_demos-50_split-{q}_seed-{s}_fresh_context_v3', True),
    ('ST k16 ramp200to600', 'outer_inner/value_k16_ver-wp_v3_la_tgn_ts-term_son-ramp200to600_enc-resnet18_demos-50_split-{q}_seed-{s}_fresh_context_v3', True),
]
FULL = [
    ('BC U-Net open-loop 300', IMG / 'unet_bc/unetbc-full300_ver-t_goal_enc-resnet18_demos-50_split-blq50far_seed-42'),
    ('Online search (GPT-2)', OUT / 'pusht_image_online/online_search/online_ntraj-8_enc-resnet18_demos-50_split-blq50far_seed-42'),
    ('DT k=1 · image', IMG / 'dt_search/dt-gpt2_k1_fb-step_ver-wp_v3_la_tgn_enc-resnet18_demos-50_split-blq50far_seed-42'),
    ('DT k=1 · image+pos+fb', IMG / 'dt_search/dt-gpt2-posfb_k1_fb-step_ver-wp_v3_la_tgn_enc-resnet18_demos-50_split-blq50far_seed-42'),
    ('DT k=8 · verifier end-feedback', IMG / 'dt_search/dt-gpt2_k8_fb-step_ver-wp_v3_la_tgn_enc-resnet18_demos-50_split-blq50far_seed-42'),
    ('DT k=8 · coverage end-feedback', IMG / 'dt_search/dt-gpt2_k8_fb-step_tf-cov_ver-wp_v3_la_tgn_enc-resnet18_demos-50_split-blq50far_seed-42'),
]


def split_file(q):
    return str(SPLITS / f'pusht_blockquad_{QUADS[q]}_train50_far_valpool50.json')


def readout(arm, selection, noisy, split=None, q=None):
    """Result directory name, exactly as eval_search_pusht._bon_subdir derives it."""
    sub = f'bon_search_sel-{selection}' + (f'_ver-{WP}' if arm == 'BC' else '')
    sub += f'_obs-{"corrupt" if noisy else "clean"}_terminal-success'
    if split:   # train: the run's own manifest (equal to the valpool50 one), so no stem
        sub += f'_split-{split}' + (f'-{pathlib.Path(split_file(q)).stem}' if split == 'val' else '')
    return sub


def curves(run, sub):
    """{step: success_curve.json} under run/sub."""
    out = {}
    for f in sorted((run / sub).glob('step_*/success_curve.json')):
        out[int(f.parent.name[5:])] = json.loads(f.read_text())
    return out


def best_step(curves_by_step, arm):
    """BC: highest N = 1 success; ST: highest success at any N. Ties -> latest step."""
    def key(step):
        c = curves_by_step[step]
        sr = dict(zip(c['n'], c['success_rate']))
        return ((sr.get(1, -1) if arm == 'BC' else max(sr.values())), step)
    return max(curves_by_step, key=key) if curves_by_step else None


def val_curves(run, arm, noisy, q):
    """Held-out-train argmax curves, or None until every test-evaluated step has one with the
    N its selection rule reads -- selecting on a partial sweep would pick from a subset."""
    val = curves(run, readout(arm, 'argmax', noisy, 'val', q))
    need = {1} if arm == 'BC' else set(NS)
    done = {st for st, c in val.items() if need <= set(c['n'])}
    return val if set(curves(run, readout(arm, 'argmax', noisy))) <= done else None


def cand_dir(arm, seed, split, q='blq50far', kind='cand'):
    return AUDIT / kind / q / arm.replace(' ', '_') / f's{seed}' / split


ROLLOUT_NS = [4, 8, 16, 64]


def runs():
    """(group, q, seed, arm, run dir, noisy) in priority order."""
    for s in SEEDS:
        for q in QUADS:
            for arm, tmpl, noisy in CHUNKED:
                yield (q, s), q, s, arm, IMG / tmpl.format(q=q, s=s), noisy


def cell(run, sub, step, n, selection, noisy, q, arm, split='val'):
    entry = dict(step=f'step_{step:07d}', checkpoint=str(run / 'checkpoints' / f'step_{step:07d}.ckpt'),
                 n=n, n_envs=50, seed=42, selection=selection, terminal_scoring=True,
                 verifier_value=WP if arm == 'BC' else None, split=split,
                 split_file=split_file(q) if split == 'val' else None)
    return dict(run=str(run), output=str(run / sub), corrupt_obs_eval=noisy, entries=[entry])


def pending(c):
    e = c['entries'][0]
    if c.get('evaluator') == 'video':
        return not list(pathlib.Path(c['output']).glob('ep02_tiled_*.mp4'))
    if c.get('evaluator') == 'cand':
        f = pathlib.Path(c['output']) / 'candidates.npz'
        return not f.exists() or str(np.load(f)['checkpoint']) != e['checkpoint']
    f = pathlib.Path(c['output']) / e['step'] / ('metrics.csv' if c.get('evaluator') == 'dt' else 'success_curve.json')
    if not f.exists():
        return True
    return c.get('evaluator') != 'dt' and not set(e['n']) <= set(json.loads(f.read_text())['n'])


def select_cells():
    cells = []
    for group, q, s, arm, run, noisy in runs():
        for obs in ((False, True) if noisy else (False,)):
            test = curves(run, readout(arm, 'argmax', obs))
            at_test_best = best_step(test, arm)   # its held-out-train curve is read in full
            for step in sorted(test):
                cells.append((group, cell(run, readout(arm, 'argmax', obs, 'val', q), step,
                                          NS if arm != 'BC' or step == at_test_best else [1], 'argmax', obs, q, arm)))
            if arm != 'BC' and at_test_best is not None:
                cells.append((group, cell(run, readout(arm, 'final_pass', obs, 'val', q), at_test_best, NS,
                                          'final_pass', obs, q, arm)))
        if (group, arm) == (('blq50far', 42), CHUNKED[-1][0]):
            for _, frun in FULL:
                for d in sorted((frun / 'bon_search/dt_eval').glob('*/')):
                    ck = frun / 'checkpoints' / (d.name + '.ckpt' if d.name != 'latest' else 'latest.ckpt')
                    entry = dict(step=d.name, checkpoint=str(ck), split='val', split_file=split_file('blq50far'),
                                 n=8, n_envs=50, seed=0)
                    cells.append((group, dict(evaluator='dt', run=str(frun), entries=[entry],
                                              output=str(frun / 'bon_search' / f'dt_eval_{pathlib.Path(split_file("blq50far")).stem}'))))
    return cells


def final_cells():
    cells = []
    for group, q, s, arm, run, noisy in runs():
        for obs in ((False, True) if noisy else (False,)):
            val = val_curves(run, arm, obs, q)
            if val is None:
                continue
            step = best_step(val, arm)
            if arm == 'BC':
                cells.append((group, cell(run, readout(arm, 'argmax', obs, 'val', q), step, NS, 'argmax', obs, q, arm)))
            else:
                cells.append((group, cell(run, readout(arm, 'final_pass', obs, 'val', q), step, NS, 'final_pass', obs, q, arm)))
    return cells


def train_cells():
    """The 50 training demos themselves, at the test-best and held-out-train-best steps: argmax
    N = 1..64 (+ final_pass for ST); full-trajectory arms at the report's two choices."""
    from wandb_fulltraj_report import full_selection
    cells = []
    for group, q, s, arm, run, noisy in runs():
        for obs in ((False, True) if noisy else (False,)):
            val = val_curves(run, arm, obs, q)
            steps = {best_step(curves(run, readout(arm, 'argmax', obs)), arm), best_step(val or {}, arm)} - {None}
            for step in sorted(steps):
                for sel in (('argmax',) if arm == 'BC' else ('argmax', 'final_pass')):
                    cells.append((group, cell(run, readout(arm, sel, obs, 'train', q), step, NS, sel, obs, q, arm,
                                              split='train')))
        if (group, arm) == (('blq50far', 42), CHUNKED[-1][0]):
            for (label, frun), d in zip(FULL, full_selection().values()):
                for step in sorted({d['step_test'], d['step_val']} - {None}):
                    name = 'latest' if step < 0 else f'step_{step:07d}'
                    entry = dict(step=name, checkpoint=str(frun / 'checkpoints' / f'{name}.ckpt'), split='train',
                                 split_file=split_file('blq50far'), n=8, n_envs=50, seed=0)
                    cells.append((group, dict(evaluator='dt', run=str(frun), entries=[entry],
                                              output=str(frun / 'bon_search' / 'dt_eval_train'))))
    return cells


def cand_cells():
    cells = []
    for group, q, s, arm, run, noisy in runs():
        if q != 'blq50far' or arm == 'BC':
            continue
        for split, step in cand_steps(run, arm, q):
            entry = dict(step=f'step_{step:07d}', checkpoint=str(run / 'checkpoints' / f'step_{step:07d}.ckpt'),
                         split=split, split_file=split_file(q) if split == 'val' else None)
            cells.append((group, dict(evaluator='cand', run=str(run), output=str(cand_dir(arm, s, split)),
                                      entries=[entry])))
    return cells


def cand_steps(run, arm, q):
    """[(split, step)]: test at the test-best step, held-out train at the held-out-train-best one."""
    out = []
    for split, cs in (('test', curves(run, readout(arm, 'argmax', False))), ('val', val_curves(run, arm, False, q))):
        step = best_step(cs, arm) if cs else None
        if step is not None:
            out.append((split, step))
    return out


def rollout_cells():
    cells = []
    for group, q, s, arm, run, noisy in runs():
        if q != 'blq50far' or arm == 'BC':
            continue
        for split, step in cand_steps(run, arm, q):
            ckpt = str(run / 'checkpoints' / f'step_{step:07d}.ckpt')
            sf = split_file(q) if split == 'val' else None
            entry = dict(step=f'step_{step:07d}', checkpoint=ckpt, n=ROLLOUT_NS, n_envs=50, seed=42,
                         selection='argmax', terminal_scoring=True, store_scores=True, split=split, split_file=sf)
            cells.append((group, dict(run=str(run), output=str(cand_dir(arm, s, split, kind='rollout')),
                                      corrupt_obs_eval=False, entries=[entry])))
            if s == 42:
                entry = dict(step=f'step_{step:07d}', checkpoint=ckpt, n=ROLLOUT_NS, n_seeds=3, seed=42,
                             split=split, split_file=sf, label=arm.replace(' ', '_'))
                cells.append((group, dict(evaluator='video', run=str(run), entries=[entry],
                                          output=str(cand_dir(arm, s, split, kind='video')))))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['select', 'final', 'train', 'cand', 'rollout'], default='select')
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--throttle', type=int, default=16)
    a = ap.parse_args()
    cells = [(g, c) for g, c in dict(select=select_cells, final=final_cells, train=train_cells, cand=cand_cells, rollout=rollout_cells)[a.phase]() if pending(c)]
    for g, n in collections.Counter(g for g, _ in cells).items():
        print(f'{g[0]} seed {g[1]}: {n} cells')
    print(f'{len(cells)} pending cells, phase {a.phase}')
    missing = [c['entries'][0]['checkpoint'] for _, c in cells if not pathlib.Path(c['entries'][0]['checkpoint']).exists()]
    assert not missing, f'missing checkpoints: {missing[:5]}'
    if not a.submit or not cells:
        return
    AUDIT.mkdir(parents=True, exist_ok=True)
    name = f'{a.phase}_manifest.json'
    (AUDIT / name).write_text(json.dumps([c for _, c in cells], indent=1))
    job = subprocess.check_output(
        ['sbatch', '--parsable', f'--array=0-{len(cells) - 1}%{a.throttle}',
         f'--export=NONE,MANIFEST={name}', f'--job-name=vp50_{a.phase}',
         '--output=/gscratch/robotics/harine/slurm_logs/vp50-%A_%a.out',
         'scripts/slurm/redo_noisy_evals.sbatch', str(AUDIT)], text=True, cwd=ROOT).strip()
    with open(AUDIT / 'jobs.jsonl', 'a') as f:
        f.write(json.dumps(dict(phase=a.phase, job=job, manifest=name, cells=len(cells))) + '\n')
    print('submitted', job)


if __name__ == '__main__':
    main()
