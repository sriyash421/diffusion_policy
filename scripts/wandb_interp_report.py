"""Build / refresh the PushT interpolation W&B report (l2sml/pusht_reports): random 50 / 75 training starts, U-Net BC
full300, and the in-context ST (train_pusht_dt_asteroid: clean, flat / ramp feature noise, no-context control) on
25 BC-failure + 25 other held-out starts. Reads the
figures under docs/reports/figures/interp_rand50 (plot_split_maps --interp, plot_st_candidates --bcfail,
plot_st_positions; draw them in a Slurm job first) and the BC / ST candidates; the report keeps its URL
(OUT/report_url.txt).

    python scripts/wandb_interp_report.py
"""
import json
import pathlib

import numpy as np

ENTITY, PROJECT = 'l2sml', 'pusht_reports'
TITLE = 'PushT · interpolation: does in-context ST fix the held-out starts where BC fails?'
REPO = pathlib.Path(__file__).resolve().parents[1]
FIG = REPO / 'docs/reports/figures/interp_rand50'
SPLITS = REPO / 'diffusion_policy/config/splits'
ROOT = pathlib.Path('/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search')
OUT = ROOT / '_interp_report'
RUNS = ROOT / 'pusht_image_search_imgonly'
DEMOS = (50, 75)
BC_STEP = 50000
# per-arm figures: plot_st_candidates --arm order clean, flat 400, ramp 600-200, control (k+1), control (attempt 1)
ARMS = ('clean', 'flat 400', 'ramp 600-200', 'no-context control')
SIMPLE = ('overview', 'candidates', 'flat', 'ramp', 'startmaps')
PER_RUN = ('bcfail_context', 'full_context', *(f'{k}_{i}' for k in ('curves', 'perf', 'bon') for i in range(4)),
           'startmap', 'fail_stk2', 'fail_stk7')


def bc_run(d):
    return RUNS / 'unet_bc' / f'unetbc-full300_ver-t_goal_enc-resnet18_demos-{d}_split-rand{d}-50k_seed-42'


def st_run(d):
    return (RUNS / 'dt_asteroid' / f'dt-gpt2_asteroid8x30_s30_b128_cps1_on-none_ctx-state_fb-step_tf-cov_ver-wp_v3_la_tgn'
            f'_enc-resnet18_demos-{d}_split-rand{d}cov30_seed-42')


def manifest(d):
    return f'pusht_rand{d}_train{d}_bccov30_25x25'


def bc_table():
    rows = ['| BC (50k steps) | train starts: coverage / success | held-out starts: coverage / success '
            '| held-out starts solved by any of 8 samples | BC failures among held-out (coverage < 0.3) |',
            '|---|---|---|---|---|']
    for d in DEMOS:
        sub = bc_run(d) / 'bon_search'
        tr = np.load(sub / 'dt_eval_train' / f'step_{BC_STEP:07d}' / 'train_candidates.npz')
        te = [np.load(sub / s / f'step_{BC_STEP:07d}' / 'test_candidates.npz')
              for s in ('dt_eval', 'dt_eval_seed1', 'dt_eval_seed2')]
        side = json.loads((SPLITS / f'{manifest(d)}.bcfail.json').read_text())
        n_held = len(te[0]['episode_idxs'])
        rows.append(f'| {d} demos | {tr["max_coverage"][:, 0].mean():.2f} / {tr["success"][:, 0].mean():.2f} '
                    f'| {np.mean([z["max_coverage"][:, 0].mean() for z in te]):.2f} / '
                    f'{np.mean([z["success"][:, 0].mean() for z in te]):.3f} '
                    f'| {int(te[0]["success"].any(1).sum())} of {n_held} '
                    f'| {len(side["bc_fail_held_out"])} of {n_held} |')
    return '\n'.join(rows)


def st_table():
    """Per arm: candidate k+1 (control attempt 1: candidate 1) coverage on all / BC-failure / other test starts."""
    rows = ['| | arm | stages | BC candidate 1 | best STk (by all starts) | last STk | last STk gain over BC '
            '| last STk better than BC on | test starts solved: last candidate (any of 8) |',
            '|---|---|---|---|---|---|---|---|---|']
    f3 = lambda v: ' / '.join(f'{x:.2f}' for x in v)
    for d in DEMOS:
        f = FIG / f'st{d}' / 'bcfail_context.json'
        if not f.is_file():
            continue
        for arm, groups in json.loads(f.read_text()).items():
            gs = list(groups.values())                      # all, BC-failure, other
            cov = [{int(k): v for k, v in g['st_coverage_by_k'].items()} for g in gs]
            kl, kb = gs[0]['last_stage'], max(cov[0], key=cov[0].get)
            rows.append(f'| {d} demos | {arm} | {kl + 1} of 8 | {f3(g["bc_coverage_c1"] for g in gs)} '
                        f'| STk{kb} {f3(c[kb] for c in cov)} | STk{kl} {f3(c[kl] for c in cov)} '
                        f'| {" / ".join(format(g["coverage_gain_by_k"][str(kl)], "+.2f") for g in gs)} '
                        f'| {gs[0]["frac_starts_improved_by_k"][str(kl)]:.0%} '
                        f'| {gs[0]["n_solved_by_last_candidate"]} ({gs[0]["n_solved_in_any_of_8_attempts_of_last_stage"]}) |')
    return '\n'.join(rows)


GROUPS = ('train starts', 'all test starts', 'BC-failure test starts')


def cand_line(d):
    """Verifier-picked coverage of each drawn model at the largest n they share (plot_interp_simple candidates)."""
    f = FIG / f'simple{d}' / 'summary.json'
    c = {k: v for k, v in json.loads(f.read_text()).get('candidates', {}).items() if v} if f.is_file() else {}
    if not c:
        return ''
    n = min(v[GROUPS[0]]['n'][-1] for v in c.values())
    at = lambda v, g: v[g]['verifier'][v[g]['n'].index(n)]
    return (f'  At n = {n}, verifier pick (train / all test / BC-failure test): '
            + '; '.join(f'{k} ' + ' / '.join(f'{at(v, g):.2f}' for g in GROUPS) for k, v in c.items()) + '.')


def pick_table(d):
    """Per model: the stage picked by train coverage vs the one picked by all-test coverage (plot_interp_simple)."""
    f = FIG / f'simple{d}' / 'summary.json'
    if not f.is_file():
        return ''
    j = json.loads(f.read_text())
    bc = ' / '.join(f'{v:.2f}' for v in j.pop('bc').values())
    j.pop('candidates', None)
    rows = ['| model | stages evaluated | train-picked: train / all test / BC-fail '
            '| test-picked: train / all test / BC-fail | test pick higher on all test by |',
            '|---|---|---|---|---|', f'| BC (UNet), candidate 1 | | {bc} | | |']
    for label, v in j.items():
        if v is None:
            rows.append(f'| {label} | 0 (training) | | | |')
            continue
        t, q = v['train_picked'], v['test_picked']
        a, b = v['stages'][str(t)], v['stages'][str(q)]
        gap = list(b.values())[1] - list(a.values())[1]
        rows.append(f'| {label} (attempt {v["scored"]}) | {len(v["stages"])} of 8 '
                    f'| STk{t}: {" / ".join(f"{x:.2f}" for x in a.values())} '
                    f'| STk{q}: {" / ".join(f"{x:.2f}" for x in b.values())} '
                    f'| {"**" if gap >= .03 else ""}{gap:+.2f}{"**" if gap >= .03 else ""} |')
    return '\n'.join(rows)


def unbiased_bc():
    """BC seed-0 samples 2..8 were not used to pick the BC-failure starts: an unselected baseline per group."""
    out = []
    for d in DEMOS:
        side = json.loads((SPLITS / f'{manifest(d)}.bcfail.json').read_text())
        z = np.load(bc_run(d) / 'bon_search' / 'dt_eval' / f'step_{BC_STEP:07d}' / 'test_candidates.npz')
        row = {int(e): i for i, e in enumerate(z['episode_idxs'])}
        fail = set(side['bc_fail_test'])
        test = [int(e) for e in side['per_start']]
        parts = []
        for name, eps in (('BC-failure', [e for e in test if e in fail]), ('other', [e for e in test if e not in fail])):
            parts.append(f'{name} {z["max_coverage"][[row[e] for e in eps], 1:].mean():.2f}')
        out.append(f'{d} demos: ' + ', '.join(parts))
    return '; '.join(out)


def main():
    figs = {'interp/split_maps': FIG / 'split_maps.png'}
    for d in DEMOS:
        figs.update({f'interp/simple{d}_{n}': FIG / f'simple{d}' / f'{n}.png' for n in SIMPLE})
        figs.update({f'interp/st{d}_{n}': FIG / f'st{d}' / f'{n}.png' for n in PER_RUN})
    figs = {k: p for k, p in figs.items() if p.is_file()}

    import wandb
    import wandb_workspaces.reports.v2 as wr
    OUT.mkdir(parents=True, exist_ok=True)
    rid_file = OUT / 'run_id.txt'
    rid = rid_file.read_text().strip() if rid_file.exists() else wandb.util.generate_id()
    rid_file.write_text(rid)
    run = wandb.init(entity=ENTITY, project=PROJECT, id=rid, resume='allow', dir=str(OUT),
                     name='pusht_interp_report_figures', settings=wandb.Settings(start_method='fork'))
    run.log({k: wandb.Image(str(p)) for k, p in figs.items()})
    run.finish()

    md = wr.MarkdownBlock

    def grid(key, h):
        if key not in figs:
            return md('*figure missing.*')
        rs = wr.Runset(entity=ENTITY, project=PROJECT, name='figures')
        return wr.PanelGrid(runsets=[rs], panels=[wr.MediaBrowser(media_keys=[key], num_columns=1,
                                                                  layout=wr.Layout(x=0, y=0, w=24, h=h))])

    blocks = [md('## Simple plots\n'
                 '- Panels: train starts / all 50 test starts / 25 BC-failure test starts; y = max T coverage, mean '
                 'over the starts.\n'
                 '- **Context length (ST only):** x = candidates n; STk with k = n − 1 of its own attempts in '
                 'context, scored on candidate n. Star = the stage picked by train-start coverage; the table flags '
                 '(bold) where picking by test coverage would score ≥ 0.03 higher on all test starts. BC = its first '
                 'sample, dashed.\n'
                 '- **n candidates:** BC, clean STk7 and the no-context control trained with / without per-step '
                 'feedback for 7,200 updates (as long as STk7) each draw n = 1 … 32 candidates. BC and the controls '
                 'draw without context; STk7 keeps its own last 7 attempts in context. Solid = the verifier picks '
                 'one; dashed = the best coverage among the n. A model whose 32-candidate eval has not landed shows n ≤ 8 from '
                 'its existing 8-candidate eval.')]
    for d in DEMOS:
        q = f'interp/simple{d}_'
        blocks += [
            md(f'### {d} demos\n' + pick_table(d)),
            md('- **Context length:** clean, flat 400 and ramp 600-200 ST'), grid(f'{q}overview', 9),
            md('- **n candidates:** BC vs STk7 vs STk0 with / without action feedback (STk0 trained 7,200 updates, as long as STk7).' + cand_line(d)),
            grid(f'{q}candidates', 9),
            md('- **Flat noise (t = 400):** clean vs noised ST against context length'), grid(f'{q}flat', 9),
            md('- **Ramp noise (t = 600 → 200):** clean vs noised ST against context length'), grid(f'{q}ramp', 9),
            md('- **Per start:** each model at its train-picked stage, coverage at the T start position'),
            grid(f'{q}startmaps', 18)]
    blocks += [
        md('## Details\n### Question\n'
           '- The held-out-quadrant runs (blq50far) test extrapolation. Here the test starts lie among the training '
           'starts (interpolation).\n'
           '- Does the in-context ST policy, given veritas + T-to-goal feedback (`wp_v3_la_tgn`) on its own earlier '
           'attempts, raise coverage on the held-out starts where BC fails?\n\n'
           '### Setup\n'
           '- **Splits:** 50 or 75 random training starts (seed 42); the other 156 / 131 of the 206 demos are '
           'held out. Nested: the 50 training starts are inside the 75, the 131 held-out inside the 156.\n'
           '- **BC:** one U-Net full300 (open-loop 300-step plan), 50k steps, image + goal encoder ResNet-18. Every '
           'held-out start sampled under 3 eval seeds (only the diffusion noise differs).\n'
           '- **BC failure:** BC\'s first sample has coverage < 0.3 under at least 2 of the 3 seeds. (BC solves '
           'almost no held-out start, so a success-based rule would mark them all.)\n'
           '- **Test set for ST:** 25 BC-failure starts + 25 other held-out starts (seeded random picks).\n'
           '- **ST:** in-context GPT-2 policy (`train_pusht_dt_asteroid`), 8 collect-and-train cycles × 30 batches '
           'of 128 × 30 gradient steps, seed 42. STk has its own k earlier attempts in context and is scored on '
           'attempt k + 1.\n'
           '- **ST arms:** *clean*; *flat 400*: training-only Gaussian noise on the demo\'s ResNet features at DDPM '
           'timestep 400 (ᾱ ≈ 0.19); *ramp 600-200*: timestep 600 for items with no context, falling linearly to 200 '
           'at 7 context attempts; *no-context control*: the demos only (no collected attempts) and one AdamW over '
           'all 8 × 900 steps, so STk = the same number of updates as ST stage k. Every arm, the control included, '
           'gets the per-step feedback inside the current attempt; only the earlier attempts are missing from the '
           'control. Its fair score is attempt 1; attempt k + 1 shows it its own (never trained on) earlier attempts.\n'
           '- **Metric:** max T coverage of the executed attempt (success = coverage ≥ 0.95). Coverage gain = ST '
           'attempt k + 1 − BC first sample (mean over its 3 seeds), per start, then averaged.\n'
           '- One BC training seed and one ST seed per demo count.'),
        md('### 1. Train and test positions\n- pink diamond = BC-failure test start; orange × = other test start'),
        grid('interp/split_maps', 10),
        md('### 2. BC\n' + bc_table() + '\n\n- 25 more demos barely help BC on the held-out starts.'),
        md('### 3. ST vs BC on the test starts\n'
           '- cells: all / BC-failure / other test starts; ST scored on attempt k + 1, the control on attempt 1 '
           '(and on k + 1).\n\n'
           + st_table() + '\n\n'
           '- **Feature noise helps.** Flat 400 is the best arm at both demo counts (STk7 0.42 / 0.46 on all test '
           'starts vs clean 0.32 / 0.42) and has the highest coverage on the other test starts (STk6 0.57 / 0.60 vs '
           'BC 0.55 / 0.47). Ramp peaks early (k3–k5: 0.45 / 0.47) and drops back.\n'
           '- **Flat 400 solves the first test starts** (50 demos: 1 at STk6; 75 demos: 3 at STk7); clean and ramp '
           'solve none with their scored attempt.\n'
           '- **Extra training explains much of it.** After the full 7,200 updates on the demos alone, the control\'s '
           'attempt 1 reaches 0.36 / 0.35 on all test starts (clean STk7: 0.32 / 0.42) and 0.28 / 0.33 on '
           'BC-failure starts (0.26 / 0.41). Without per-step feedback: 0.34 / 0.35 all test, 0.28 / 0.33 BC-failure.\n'
           '- **With 32 candidates and the verifier, STk0 ≈ STk7.** At 50 demos STk0 with action feedback scores '
           '0.50 on all test starts and 0.48 on BC-failure starts, against STk7 0.51 / 0.49; without feedback '
           '0.48 / 0.43. At 75 demos STk7 leads on all test starts (0.54 vs 0.47 with, 0.49 without feedback) '
           'and all three tie on BC-failure starts (0.50 / 0.50 / 0.48). All of them beat BC on test starts (0.45 / 0.44 all, 0.26 / 0.35 BC-failure) and trail it '
           'on train starts (BC 0.84 / 0.79 vs 0.45–0.52).\n'
           '- On BC-failure starts every ST arm peaks at 0.36–0.46 (0.26–0.41 at k7), against 0.09 / 0.13 for BC; '
           'the control at 7,200 updates reaches 0.28 / 0.33 there. The BC gap is not a '
           'selection artefact: BC\'s seed-0 samples 2–8, unused in picking the '
           f'groups, average {unbiased_bc()}.')]
    for d in DEMOS:
        p = f'interp/st{d}_'
        blocks += [
            md(f'### {3 + DEMOS.index(d) + 1}. ST, {d} demos\n'
               '- columns: all 50 test starts, the 25 BC-failure starts, the 25 others; rows: coverage gain over BC, '
               'coverage, success; x = stage k (attempt k + 1 scored; control also on attempt 1); one line per arm'),
            grid(f'{p}bcfail_context', 14),
            md('- STk at n = k + 1 only (its trained context length); BC at the same n; train and test starts'),
            grid(f'{p}full_context', 10),
            md('- during training: x = gradient steps within cycle k; one line per STk (attempt k + 1); one figure '
               'per arm: ' + ', '.join(ARMS))]
        blocks += [grid(f'{p}curves_{i}', 11) for i in range(4)]
        blocks += [md('- perf@n: candidate n alone, then best of n: the verifier picks one of the first n '
                      'candidates; one pair per arm: ' + ', '.join(ARMS))]
        blocks += [grid(f'{p}{kind}_{i}', 11) for i in range(4) for kind in ('perf', 'bon')]
        blocks += [
            md('- clean arm: coverage per start at the T start position; × = BC-failure test start'),
            grid(f'{p}startmap', 16),
            md('- clean arm: the test start STk2 and STk7 handle worst: all 8 attempts in a row, replayed from the saved actions'),
            grid(f'{p}fail_stk2', 11), grid(f'{p}fail_stk7', 11)]

    rep_file = OUT / 'report_url.txt'
    report = None
    if rep_file.exists():
        try:
            report = wr.Report.from_url(rep_file.read_text().strip().replace('?', ''))   # '?' in the title breaks the id parse
        except Exception as e:   # noqa: BLE001 -- a deleted report just means "make a new one"
            print('could not load existing report:', e)
    if report is None:
        report = wr.Report(entity=ENTITY, project=PROJECT, title=TITLE, blocks=blocks)
    else:
        report.title = TITLE
        report.blocks = blocks
    report.description = ('Random 50/75 training starts; in-context ST vs U-Net BC on 25 BC-failure + 25 other '
                          'held-out starts, coverage gain over BC.')
    report.save()
    rep_file.write_text(report.url)
    print('report:', report.url.replace('https://wandb.ai/', 'https://forge.coreweave.com/wandb/'))


if __name__ == '__main__':
    main()
