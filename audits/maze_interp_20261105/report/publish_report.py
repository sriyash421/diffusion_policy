"""W&B report for the interpolation split (maze_interp_20261105). Creates the report on the first run (report_id.txt), updates it
after; figures that do not exist yet are skipped, so re-run when evals finish.
  C=/gscratch/robotics/harine/codex_audits/maze_report_consolidated_20261008; TMPDIR=<writable dir> WANDB_DISABLE_SERVICE=true \
    PYTHONPATH=$C/wandb013_shim <vae_pushT_l2s python> publish_report.py
"""
import inspect, json, os, textwrap
from pathlib import Path

R = Path(__file__).resolve().parent; HERE = R.parent
os.environ.pop('WANDB_API_KEY', None); os.environ['WANDB_DIR'] = str(R); os.environ['WANDB_CACHE_DIR'] = str(R / 'wandb_cache')
import wandb
from wandb.sdk.internal import internal_api  # W&B 0.13.3 vs current servers: optional server-settings type may be null
src = textwrap.dedent(inspect.getsource(internal_api.Api.server_settings_introspection))
old = 'res.get("ServerSettingsType", {}).get("fields", [{}])'
if old in src:
    ns = dict(vars(internal_api)); exec(compile(src.replace(old, '(res.get("ServerSettingsType") or {}).get("fields", [])'), 'wandb_compat', 'exec'), ns)
    internal_api.Api.server_settings_introspection = ns['server_settings_introspection']
wandb.require('report-editing:v0')
import wandb.apis.reports as wr

RID, TITLE = 'maze-interp30', 'Maze · interpolation: 30 random train goals, does ST with feedback fix the pairs BC fails on? (11×11)'
FB = [('int_s30_bfs', 'BFS'), ('int_s30_bfsdir', 'BFS + dx, dy'), ('int_s30_dir', 'dx, dy'),
      ('int_s30_nofb', 'BFS, zeroed for the attempts in context (ablation)')]
images = {'split/maze': R / 'maze_split.png', 'bc/scan': R / 'bc_scan_goals.png', 'fb/compare': R / 'feedback_compare.png',
          'train/curves': R / 'train_curves.png', 'abl/nofb': R / 'ablation_nofb.png', 'noise/compare': R / 'noise_compare.png',
          'k0long/goalmap_compare': R / 'goalmap_k0long_compare.png', 'allpairs/compare': R / 'allpairs.png',
          'allpairs/goalmap': R / 'allpairs_goalmap.png'}
NOISE_RUNS = [('int_s30_on400', 'BFS + dx, dy, obs noise flat t=400'), ('int_s30_onramp', 'BFS + dx, dy, obs noise ramp t=600→200'),
              ('int_k0long', 'BFS + dx, dy, round 0 only, 7200 steps')]
VAR_RUNS = [('int_s30_bfsdir_ol', 'BFS + dx, dy, open-loop observations (start cell)'),
            ('int_s30_relabel', 'BFS + dx, dy, DAgger: the BFS expert labels the policy\'s own attempts')]
HG_RUNS = [('int_s30_hg_demo', 'hidden goal, ASTEROID (repo): the expert\'s path after the policy\'s own attempts'),
           ('int_s30_hg_relabel', 'hidden goal, DAgger: the BFS expert labels the policy\'s own attempt')]
images.update({'hidden/compare': R / 'hidden_compare.png', 'hidden/allpairs': R / 'allpairs_hidden.png',
               'hidden/goalmap': R / 'allpairs_hidden_goalmap.png'})
images.update({'variants/compare': R / 'variants_compare.png', 'variants/allpairs': R / 'allpairs_variants.png',
               'variants/goalmap': R / 'allpairs_variants_goalmap.png'})
images.update({'cont/demos': R / 'cont_demos.png', 'cont/n16': R / 'n16_cont.png',
               'cont/goalmap': R / 'cont_goalmap.png'})
images.update({'allpairs/n16': R / 'n16.png'})
SIMPLE = [(k, p) for k in ('grid', 'cont', 'online') for p in ('stk', 'steps', 'fb')] + [('grid', 'k7fb'), ('noise', 'flat'), ('noise', 'ramp')]
for k, p in SIMPLE:
    images[f'simple/{k}_{p}'] = R / f'simple_{k}_{p}.png'; images[f'simple/{k}_{p}_goals'] = R / f'simple_{k}_{p}_goals.png'
for t, _ in FB + NOISE_RUNS + VAR_RUNS + HG_RUNS:
    images[f'{t}/perfbon'] = R / f'perf_bon_{t}.png'; images[f'{t}/goalmap'] = R / f'goalmap_{t}.png'
images = {k: v for k, v in images.items() if v.exists()}; assert 'split/maze' in images
run = wandb.init(entity='l2sml', project='maze_reports', id=RID, resume='allow', dir=str(R), name=TITLE, settings=wandb.Settings(start_method='fork'))
run.log({k: wandb.Image(str(p)) for k, p in images.items()}); run.finish()


def grid(key, h, title):
    rs = wr.RunSet(entity='l2sml', project='maze_reports', name='Interpolation'); rs.spec['id'] = RID
    rs.spec['selections'] = {'root': 0, 'bounds': [], 'tree': [RID]}; rs.spec['pinnedRunIds'] = [RID]
    p = wr.MediaBrowser(num_columns=1, media_keys=key)
    p.spec['config'] = {'chartTitle': title, 'columnCount': 1, 'mediaKeys': [key], 'mode': 'gallery', 'gallerySettings': {'axis': 'run'}}
    p.spec['layout'] = {'x': 0, 'y': 0, 'w': 24, 'h': h}
    return wr.PanelGrid(runsets=[rs], panels=[p])


md = wr.MarkdownBlock


def fig(key, h, title):
    return grid(key, h, title) if key in images else md(f'*{title}: running, added when it finishes.*')


sc = json.loads((HERE / 'split_checks.json').read_text()); te = json.loads((HERE / 'test_episodes.json').read_text())
sel = json.loads((HERE / 'bc_selection.json').read_text()); eps = te['episodes']
n_fail = sum(e['bc_fail'] for e in eps); goals_fail = len({tuple(e['goal']) for e in eps if e['bc_fail']})
fc = json.loads((R / 'feedback_compare.json').read_text()) if (R / 'feedback_compare.json').exists() else {}
ab = json.loads((R / 'ablation_nofb.json').read_text()) if (R / 'ablation_nofb.json').exists() else {}


def abl_summary():
    if 'fail' not in ab:
        return md('*Ablation results: added when the run finishes.*')
    lines = ['- BC-fail pairs (STk0 → STk7 on attempt k+1; STk7 attempt 1 → 8; STk7 best of 8):']
    for t, name in (('int_s30_bfs', 'BFS'), ('int_s30_nofb', 'BFS, context feedback zeroed')):
        f = ab['fail'][t]
        lines.append(f"  - {name}: {f['stk_diag'][0]:.2f} → {f['stk_diag'][-1]:.2f}; {f['stk7_perf'][0]:.2f} → {f['stk7_perf'][-1]:.2f}; "
                     f"{f['stk7_bon'][-1]:.2f}" + (f"; control STk7 {ab['ctrl'][t]['stk_diag'][-1]:.2f}" if t in ab.get('ctrl', {}) else ''))
    lines += ['- **Reading:** the two runs track each other within run-to-run noise. At k = 0 nothing differs between them (no attempts in '
              'context yet), and they still land at 0.31 vs 0.24, so gaps of ~0.1 here are noise from a single training seed. Without the feedback, '
              'success still rises over rounds and ST still beats BC on the BC-fail pairs; within STk7 neither run improves much from attempt 1 to 8. '
              'So the gain on this split comes from the closed-loop policy and training on its own rollouts, not from reading feedback in context.']
    return md('\n'.join(lines))


nz = json.loads((R / 'noise_compare.json').read_text()) if (R / 'noise_compare.json').exists() else {}


def noise_summary():
    if not nz:
        return md('*Noise results: added when the evals finish.*')
    lines = ['- STk on attempt k+1 over rounds; last saved STk attempt 1 → 8 (8 seeds; source file in brackets while evals are still running):']
    for k, title in (('train', 'Train starts'), ('test', 'Test pairs (all BC failures)'), ('ctrl', 'Control pairs')):
        if k not in nz:
            continue
        lines.append(f'  - **{title}**' + (f" (BC {nz[k]['bc']:.2f})" if 'bc' in nz[k] else ''))
        for t, name in [('int_s30_bfsdir', 'no noise')] + [(t + s, n + e) for t, n in NOISE_RUNS[:2] for s, e in (('', ', clean eval'), ('_noisy', ', noisy eval'))]:
            if t in nz[k]:
                v = nz[k][t]; d = v['stk_diag']
                lines.append(f"    - {name}: STk0 {d[0]:.2f} → STk{v['rounds'] - 1} {d[-1]:.2f}; attempt 1 → 8 {v['last_perf'][0]:.2f} → "
                             f"{v['last_perf'][-1]:.2f}" + ('' if v['source'] in ('bon_seeds8.npz', 'bon_seeds8_noisyobs.npz', 'bon_seeds8_control.npz') else f" [{v['source']}]"))
        if 'int_k0long' in nz[k]:
            v = nz[k]['int_k0long']; lines.append(f"    - round 0 only, 7200 steps: {v['first']:.2f}; attempt 1 → 8 {v['perf'][0]:.2f} → {v['perf'][-1]:.2f}")
    return md('\n'.join(lines))


def matched_section(fname='matched_table.json'):
    """Matched steps (7200), no-feedback STk0 and closed-loop BC: table from matched_table.json (plots.matched_table); the same layout
    for variants_table.json (open-loop obs / DAgger relabel / BC on the DAgger data) and hidden_table.json."""
    if not (R / fname).exists():
        return md('*table: running, added when it finishes.*')
    mt = json.loads((R / fname).read_text()); n = mt['n_pairs']
    v = lambda x: '…' if x is None else f'{x:.2f}'
    rows = ['| policy | train starts | 50 test pairs (BC-fail) | 50 control pairs | all held-out pairs | ... BC fails | ... BC solves |',
            '|---|---|---|---|---|---|---|']
    for r in mt['rows']:
        rows.append(f"| {r['name']} | " + ' | '.join(v(r.get(c)) for c in ('train', 'test', 'control', 'all', 'all_bcfail', 'all_bcok')) + ' |')
    return md(f"- 8 seeds (BC scan: 3); STk on its attempt k+1. All held-out pairs: every (start, held-out goal) pair, {n['all']} pairs over "
              f"41 goals; BC fails = ≤ 1 of 3 scan seeds ({n['all_bcfail']} pairs), BC solves = ≥ 2 of 3 ({n['all_bcok']}). "
              + ("… = eval still running" if any(r.get(c) is None for r in mt['rows'] for c in ('train', 'test', 'control', 'all')) else '')
              + "\n\n" + '\n'.join(rows))


def summary():
    if 'fail' not in fc:
        return md('*Results: added when the evals finish.*')
    lines = ['### Summary (BC-fail pairs; STk on its attempt k+1, 8 seeds)',
             f"- BC (fresh seeds): {fc['fail']['bc']['first']:.2f} on the BC-fail pairs"
             + (f", {fc['ctrl']['bc']['first']:.2f} on the control pairs" if 'ctrl' in fc else '')]
    for t, name in FB:
        if t in fc['fail']:
            f, o = fc['fail'][t], fc['train'][t]
            lines.append(f"- {name}: STk0 {f['stk_diag'][0]:.2f} → STk7 {f['stk_diag'][-1]:.2f}; STk7 attempt 1 → 8 "
                         f"{f['stk7_perf'][0]:.2f} → {f['stk7_perf'][-1]:.2f}; train starts STk7 {o['stk_diag'][-1]:.2f}"
                         + (f"; control STk7 {fc['ctrl'][t]['stk_diag'][-1]:.2f}" if t in fc.get('ctrl', {}) else ''))
    return md('\n'.join(lines))


blocks = [
    md("### Setup\n"
       "- Same policy, feedback and training as the held-out-quadrant runs with 30 gradient steps per batch "
       "(8 collect-and-train cycles; each 30 batches of 128 × 30 gradient steps; STk = weights after cycle k, trained with k own attempts in context)\n"
       "- Only the goal split changes: **interpolation** instead of a held-out quadrant\n"
       "- Feedback variants: **BFS** (step −BFS; end −BFS, −BFS, dx, dy) · **BFS + dx, dy** (step −BFS, dx, dy; end the same) · "
       "**dx, dy** (step and end dx, dy only); BFS = maze distance to the goal / 32, (dx, dy) = goal − cell / 10"),
    md(f"### 1. Split\n- {sc['train_goal_cells']} train goals sampled at random (seed 0) from the 71 free cells; the other "
       f"{sc['test_goal_cells']} are held out\n- {sc['episodes']} A\\* demos to train goals ({sc['train_episodes']} train / "
       f"{sc['validation_episodes']} validation episodes by pair)"),
    fig('split/maze', 16, 'Train and held-out goals'),
    md(f"### 2. BC and the test set\n- BC = UNet on the demos, checkpoint {sel['step']} picked by success on unseen pairs of "
       f"train goals (no test goals used)\n- BC run on every (start, held-out goal) pair with 3 seeds\n"
       f"- Test set = 50 pairs, BC failures first: {n_fail} pairs where BC solved ≤ 1 of 3 ({goals_fail} distinct goals), "
       f"then {50 - n_fail} others\n- Control: 50 random held-out-goal pairs that BC solved on 3 of 3 scan seeds (eval only)\n"
       f"- The BC lines below use fresh seeds, not the scan's"),
    fig('bc/scan', 16, 'BC success per held-out goal (all starts × 3 seeds)'),
    summary(),
    md("### 3. Feedback comparison\n- rows: test pairs (all are BC failures), control pairs (50 held-out-goal pairs BC solved on 3/3 seeds), train starts\n"
       "- left: STk on its attempt k+1; middle: STk7 perf@n; right: STk7 best of n with the verifier"),
    fig('fb/compare', 22, 'BFS vs BFS + dx, dy vs dx, dy'),
    md("- test success during training (all 50 test pairs, every 30 steps)"), fig('train/curves', 10, 'test success during training'),
    md("### Ablation: is the feedback in context what helps?\n"
       "- Same run as BFS (same schedule, seed, data collection), but the feedback of every earlier attempt in context is zeroed: "
       "its per-step −BFS and the end-of-attempt [−BFS, −BFS, dx, dy]. Its cells and moves stay in context, and the current attempt keeps "
       "its own per-step −BFS.\n"
       "- If it matches BFS, the rise over rounds comes from training on its own rollouts, not from reading feedback"),
    abl_summary(), fig('abl/nofb', 22, 'BFS vs BFS with context feedback zeroed'),
    md("### Observation noise: do the candidates improve from attempt 1 to 8 when the current observation is noisy?\n"
       "- Same run as BFS + dx, dy, plus DDPM noise on the encoded observation of the attempt being predicted, in training only: "
       "x' = √ᾱₜ·x + √(1−ᾱₜ)·ε·std (T = 1000, β 1e-4 → 0.02, std = running per-feature std). Attempts in context stay clean.\n"
       "- **flat**: t = 400 for every slot. **ramp**: t = 600 at slot 0 (no context) down to 200 at slot 7, like a denoising schedule\n"
       "- How much survives (train demos; cell decoded from the noisy one-hot image / from the noisy (x, y)):\n\n"
       "| t | √ᾱ | agent cell from image | goal cell from (x, y) | (x, y) error, cells |\n|---|---|---|---|---|\n"
       "| 200 | .81 | 1.00 | .12 | 2.6 |\n| 300 | .63 | .98 | .07 | 4.2 |\n| 400 | .44 | .77 | .05 | 5.8 |\n"
       "| 500 | .28 | .38 | .04 | 7.3 |\n| 600 | .16 | .15 | .03 | 8.4 |\n\n"
       "- Held-out goals never light a goal pixel in training (feature std ≈ 0), so a held-out goal is visible only through goal (x, y)\n"
       "- Two evals: **clean** (the normal eval) and **noisy** (each attempt gets its slot's training noise; context stays clean)\n"
       "- **Reference, matched steps:** round 0 only (A\\* demos, empty context) trained for 240 × 30 = 7200 steps, the total STk7 gets. "
       "STk0 in the runs above has only 900"),
    noise_summary(),
    md("- **Reading:**\n"
       "  - Noise makes the candidates improve from attempt 1 to 8, which the runs without noise never do; the ramp most clearly.\n"
       "  - That is mostly attempt 1 getting worse, not later attempts getting better: the best single attempt of the noise runs is ~0.5 on the "
       "test pairs, below no-noise STk7 attempt 1 (0.66) and far below the matched-steps STk0 (0.79). The same holds on train starts and control pairs.\n"
       "  - Best of 8 with the verifier: the ramp's noisy eval reaches 0.80 on the test pairs (no noise 0.75, matched-steps STk0 0.82), because its "
       "8 attempts differ more; the verifier is the true maze distance, so this measures diversity, not the policy alone.\n"
       "  - Noisy eval: context attempts keep their clean observations, including goal (x, y), so a jump at attempt 2 can come from reading "
       "the goal out of the context, not from using the feedback.\n"
       "  - Clean ramp eval: slot 0 is trained only at t = 600, so it never learns to use a clean observation; lighter-noise later slots "
       "would rise even without the context.\n"
       "  - Matched steps: STk0 trained 7200 steps beats every STk7, so the rise over rounds in the runs above is mostly more training on the demos."),
    fig('noise/compare', 22, 'observation noise: flat t=400 and ramp t=600→200 vs no noise, and STk0 trained 7200 steps'),
    md("#### Matched steps, per goal: STk0 trained 900 vs 7200 steps vs STk7\n"
       "- Same feedback (BFS + dx, dy); STk on attempt k+1, 8 seeds; the test pairs use 22 distinct held-out goals (x = column, y = row from the top)\n"
       "- STk0 at 7200 steps solves 15 of the 22 goals on every pair and seed, plus (0, 3) and (2, 6) at 0.96 and (0, 0) at 0.79 (6 pairs); "
       "STk7 scores 0.00 on (0, 0) and on (10, 2), which STk0 at 7200 steps solves\n"
       "- 10 of the 50 pairs fail on all 8 seeds, mostly at (6, 6) (4 pairs, STk7 also 0), (8, 4) (2 pairs, STk7 0.19), "
       "(0, 4) (0.33, STk7 0.46) and (0, 9) (1 pair, STk7 also 0)"),
    fig('k0long/goalmap_compare', 12, 'goal maps: BC, STk0 900 steps, STk0 7200 steps, STk7'),
    md("### Matched steps, no feedback, and closed-loop BC (all held-out goals)\n"
       "- **Matched steps:** every ST run here gets 7200 updates, the total STk7 gets: STk0 = round 0 only (240 × 30 steps); "
       "STk1 = 2 rounds × 3600; STk2 = 3 rounds × 2400. BFS + dx, dy feedback unless noted\n"
       "- **No feedback:** STk0, 7200 steps, with every feedback channel zero (`--fb none`): its tokens carry only the current cell's "
       "observation, the previous move and the step index, like BC plus closed-loop execution\n"
       "- **Closed-loop BC:** BC everywhere else in this report predicts a 64-step chunk from the start state and runs it open loop. Here it "
       "re-plans from the current cell every step and executes only the first action (64 env steps):\n"
       "  - the same 64-step BC (checkpoint 7000), re-planned every step;\n"
       "  - a new BC UNet with horizon 4 (same demos, UNet and 10k updates; checkpoint picked by closed-loop train_heldout success, all "
       "checkpoints from 3000 on score 1.0, so 10000)"),
    matched_section(),
    md("- **Reading:**\n"
       "  - Closed-loop execution does not explain BC's failures: re-planned every step, BC solves 0.45–0.46 of all held-out pairs "
       "(open loop 0.43) and 0.03–0.06 of the 50 test pairs, while STk0 at 7200 steps solves 0.82 and 0.79.\n"
       "  - The per-step feedback carries most of ST's advantage: without it STk0 drops from 0.79 to 0.38 on the test pairs and from 0.72 to "
       "0.43 on all BC-fail pairs. It still beats closed-loop BC (0.43 vs 0.14 on BC-fail pairs), so training on its own rollouts with "
       "stop prediction adds something too.\n"
       "  - At matched steps, context adds nothing: STk1 and STk2 score at or below STk0 on every set; on all held-out pairs they are "
       "0.79–0.80 vs 0.82.\n"
       "  - The same holds for the 8-round run on all held-out pairs: STk0 (900 steps) 0.68 → STk7 0.76, below STk0 trained for 7200 "
       "steps (0.82; BC-fail pairs 0.50 → 0.66 vs 0.72), so its rise over rounds is training length.\n"
       "  - Re-planning costs a little where BC already works (control 0.97 → 0.86–0.90).\n"
       "  - The BC-fail / BC-solves columns are picked with the scan, so the open-loop BC is near 0 / 1 there by construction; other rows "
       "are fresh seeds"),
    fig('allpairs/compare', 14, 'all held-out pairs: STk0 / STk1 / STk2 at 7200 steps, STk0 without feedback, BC open loop and re-planned'),
    md("- success per held-out goal cell, averaged over all starts (dashed = held-out goal; hatched = BC solves it < 50% in the scan)"),
    fig('allpairs/goalmap', 26, 'goal maps over all held-out pairs: BC, BC horizon 4 re-planned, STk0, STk0 no feedback, STk1, STk2'),
    md("#### n = 1..16 attempts: closed-loop BC vs ST at 7200 updates\n"
       "- Train starts (642 pairs), all held-out pairs BC fails (≤ 1 of 3 scan seeds, 1631) and all 2870 held-out pairs; 8 seeds per pair\n"
       "- **perf@n** (top): success of attempt n alone. ST makes 16 attempts in a row: STk0 runs have no context (window 0), so their attempts "
       "are iid; STk7 sees the last min(7, n − 1) attempts (it was trained with up to 7). BC: the n-th of 16 iid closed-loop rollouts\n"
       "- **best of n** (bottom): the verifier (−BFS of the final cell) picks one of the first n attempts and it is executed\n"
       "- Checkpoints: BC 64-step = step 7000 and BC horizon 4 = step 10000 (picked by train_heldout); ST = the weights at the end of the "
       "last round, no selection\n"
       "- **Reading** (BC-fail pairs):\n"
       "  - perf@n is flat for everything but STk7: STk0 0.72, STk0 without feedback 0.43, closed-loop BC 0.11–0.14.\n"
       "  - STk7 on attempt 1 (no context) matches STk0 at 7200 steps (0.71 vs 0.72); its own attempts in context lower attempt n to "
       "~0.67, but make the attempts more diverse: best of 16 is 0.83 vs 0.80 for STk0 (all held-out pairs 0.89 vs 0.86).\n"
       "  - Without feedback, or for BC, repeated attempts end in nearly the same place: best of 16 adds 1–3 points (0.45; BC 0.14–0.15)."),
    fig('allpairs/n16', 22, 'n = 1..16: perf@n and best of n, closed-loop BC, STk0 7200 with / without feedback, STk7'),
]
VARIANTS_READING = json.loads((R / 'variants_reading.json').read_text()) if (R / 'variants_reading.json').exists() else None
blocks += [
    md("### Open-loop observations and DAgger relabelling (BFS + dx, dy feedback, 8 rounds × 900 steps)\n"
       "- **A\\* demo target** (the run above): the loss is the start's A\\* demo after the policy's own earlier attempts\n"
       "- **Open-loop observations:** the same, but every token of an attempt carries the observation (image, agent position) of the attempt's "
       "START cell. The previous move and the per-step feedback (−BFS, dx, dy of the current cell) still follow the rollout, so the current "
       "cell is still recoverable from goal − 10 · (dx, dy)\n"
       "- **DAgger relabel** (asked for as \"similar to ASTEROID\"; it differs from github.com/WEIRDLabUW/asteroid, where the EXPERT plays "
       "the supervised last attempt, i.e. the A\\* demo target above is the repo's scheme; here the POLICY plays it): round 0 = rollouts of the BFS expert from the 642 training starts (the "
       "same BFS as the demo generator; it reproduces all 642 demos). Round n = the current policy makes n + 1 attempts per training start; "
       "the BFS expert labels the move (and stop) at every cell of the last attempt, the first n are context and not supervised. Data "
       "aggregated over rounds, weights carried over, fresh optimizer per round. Only training goals are ever labelled\n"
       "- **BC on the DAgger data:** the BC UNet (same as above, 10k updates) retrained on the BFS expert's path from every state the "
       "relabel run's policy visited, all rounds (round 0: the expert rollouts). Horizon 64 (open loop, and re-planned every step) and "
       "horizon 4 (re-planned); checkpoints picked by train_heldout success in the same rollout mode"),
    matched_section('variants_table.json'),
    md('\n'.join(VARIANTS_READING) if VARIANTS_READING else '*reading: added when all evals finish.*'),
    fig('variants/compare', 26, 'A* demo target vs open-loop obs vs DAgger relabel: STk, perf@n, best of n'),
    fig('variants/allpairs', 14, 'all held-out pairs: the three 8-round runs, BC on demos and on the DAgger data'),
    fig('variants/goalmap', 26, 'goal maps over all held-out pairs: BC (demos / DAgger data), STk7 of each run'),
]
HIDDEN_READING = json.loads((R / 'hidden_reading.json').read_text()) if (R / 'hidden_reading.json').exists() else None
blocks += [
    md("### Hidden goal: ASTEROID (repo) vs DAgger (8 rounds × 900 steps)\n"
       "- **Hidden goal** (ASTEROID's partially observed setting): the policy never sees the goal; its image has the goal plane cleared "
       "and goal_pos = 0. The env, the success test, the BFS expert and the best-of-n verifier still use the true goal\n"
       "- **Feedback = sparse reward:** no per-step feedback; the first token of the next attempt carries [r, r, 0, 0], r = 1 if the "
       "previous attempt ended within 1 of the goal (BFS / dx, dy feedback would reveal the goal). The goal is learned only from the context\n"
       "- **ASTEROID (repo, github.com/WEIRDLabUW/asteroid):** round n = the current policy makes n attempts per training start, then the "
       "EXPERT plays the next attempt (its A\\* path from the start); loss on that attempt only\n"
       "- **DAgger:** round n = the POLICY plays attempt n + 1 too, and the BFS expert labels every cell it visits (move + stop)\n"
       "- Both: round 0 = expert demos with no context; data aggregated over rounds; weights carried over, fresh optimizer per round; "
       "same 30 train goals and test sets. Goal-visible runs (BFS + dx, dy feedback) shown for reference; BC sees the goal. All-pairs "
       "evals of the hidden-goal runs: 3 seeds (others 8)\n"
       "- Success = the attempt passes within 1 of the goal at any step, so covering the maze scores without knowing the goal"),
    matched_section('hidden_table.json'),
    md('\n'.join(HIDDEN_READING) if HIDDEN_READING else '*reading: added when all evals finish.*'),
    fig('hidden/compare', 26, 'hidden goal vs goal visible: STk, perf@n, best of n (rows: BC-fail test, control, train)'),
    fig('hidden/allpairs', 14, 'all held-out pairs: hidden-goal runs (attempt k+1 and attempt 1) vs goal-visible runs'),
    fig('hidden/goalmap', 26, 'goal maps over all held-out pairs: BC (goal visible), hidden-goal STk0 and STk7'),
]
for t, name in HG_RUNS:
    blocks += [md(f"#### {name}\n- perf@n (left) and best of n (right), 8 seeds; rows: train, test pairs (BC failures), control pairs"),
               fig(f'{t}/perfbon', 26, f'perf@n and best of n, {name}')]


def cont_table():
    """Continuous-action results from cont_table.json (plots.cont_section; the n = 1..16 evals, 8 seeds): BC, STk0 with / without
    feedback (7200 steps), STk7; per set attempt 1 → attempt 16 and best of 16."""
    if not (R / 'cont_table.json').exists():
        return md('*table: running, added when it finishes.*')
    ct = json.loads((R / 'cont_table.json').read_text()); n = ct['n_pairs']
    cell = lambda sp, key: '…' if key not in ct[sp] else f"{ct[sp][key]['first']:.2f} → {ct[sp][key]['attempt16']:.2f} · **{ct[sp][key]['best16']:.2f}**"
    rows = ['| policy (all 7200 training steps) | train starts | held-out pairs BC fails | all held-out pairs |', '|---|---|---|---|']
    for key, name in (('bc', 'BC horizon 4, closed loop'), ('c_k0long_r0', 'STk0, BFS + dx, dy feedback'),
                      ('c_k0long_nofb_r0', 'STk0, no feedback'), ('c_s30_bfsdir_r7', 'STk7 (8 rounds × 900 steps)')):
        rows.append(f'| {name} | ' + ' | '.join(cell(sp, key) for sp in ('train', 'fail', 'all')) + ' |')
    return md(f"- Each cell: attempt 1 → attempt 16 · **best of 16** (the −BFS verifier picks one of the 16), 8 seeds. STk0 has no context, "
              f"so its 16 attempts are independent; STk7's attempt n sees its last min(7, n − 1) attempts; BC = independent rollouts. Held-out = "
              f"every (start, held-out goal) pair, {n['all']} pairs; BC fails = the continuous closed-loop BC solves ≤ 1/3 of 8 seeds "
              f"({n['fail']} pairs; the BC row uses 128 fresh rollouts)\n\n" + '\n'.join(rows))


blocks += [
    md("## Continuous actions\n"
       "- **Environment:** the same maze and split, but the agent is a point with a real-valued position (cell (x, y) = the square "
       "[x − ½, x + ½] × [y − ½, y + ½]). An action is a real-valued (dx, dy), clipped to length ≤ 1 cell and executed in 8 sub-steps; a "
       "sub-step into a wall slides along it (x-only or y-only) or is dropped, and the agent cannot squeeze diagonally between two walls that "
       "meet at a corner. Success = within 1 of the goal centre\n"
       "- **Demos (smoothed A\\*):** the same 800 (start, goal) episodes and train / validation pairs. The start is jittered by U(±0.3) "
       "inside its cell; the A\\* cell path is string-pulled (straight segments that keep 0.1 clear of every wall) and each segment is split "
       "into ⌈length⌉ equal steps. 12.1 steps per demo on average (grid A\\*: 11.6); mean action length 0.96, 23% of steps not axis-aligned. "
       "Every demo replays exactly through the environment\n"
       "- **Observations and feedback:** the image marks the agent's rounded cell, agent (x, y) is real-valued; feedback = BFS + dx, dy "
       "(maze distance of the rounded cell, continuous goal − position)\n"
       "- **BC:** UNet, horizon 4, re-planned from the current state every step (closed loop), same 10k updates; checkpoint by train_heldout success\n"
       "- **ST:** the same policy and schedule as the grid runs, actions not snapped: an 8-round run (900 steps per round) and STk0 trained "
       "7200 steps with and without feedback\n"
       "- **Eval:** train starts and all 2870 held-out pairs, 8 seeds × 16 attempts in a row (attempt n sees up to its last 7 attempts). "
       "The 50-pair test set of the grid sections is picked by the grid BC, so it is not used here"),
    fig('cont/demos', 12, 'continuous: smoothed-A* demos, action lengths and directions'),
    cont_table(),
    md("- **Reading:**\n"
       "  - BC still fails to interpolate with continuous actions: 0.40 on all held-out pairs (grid closed-loop BC 0.45); on its failure "
       "pairs 16 rollouts plus the verifier reach only 0.10.\n"
       "  - The per-step feedback carries ST's advantage: STk0 solves 0.56 of BC's failure pairs with BFS + dx, dy and 0.29 without "
       "(grid: 0.72 vs 0.43 on the grid BC's failures).\n"
       "  - Context does not help: STk7 starts slightly above STk0 (0.59 vs 0.56 on BC's failure pairs, 0.72 vs 0.69 on all) and drops to "
       "STk0's level by attempt 6, so later attempts do not improve on earlier ones.\n"
       "  - Best of 16 with the verifier: STk7 0.89 / 0.93, STk0 0.86 / 0.91, no feedback 0.50 / 0.69, BC 0.10 / 0.46 (BC-fail / all). The "
       "verifier is the true maze distance, so this measures how varied the attempts are, not the policy alone.\n"
       "  - Continuous ST is below grid ST (all held-out pairs, STk0 7200 steps: 0.69 vs 0.82)"),
    fig('cont/n16', 26, 'continuous, n = 1..16: perf@n (top) and best of n (bottom); BC, STk0 with / without feedback, STk7'),
    md("- success per held-out goal cell, averaged over all starts (dashed = held-out goal; hatched = the continuous BC solves it < 50%)"),
    fig('cont/goalmap', 14, 'continuous: goal maps over all held-out pairs'),
]

for t, name in VAR_RUNS:
    blocks += [md(f"#### {name}\n- perf@n (left) and best of n (right), 8 seeds; rows: train, test pairs (BC failures), control pairs"),
               fig(f'{t}/perfbon', 26, f'perf@n and best of n, {name}')]
for t, name in NOISE_RUNS:
    blocks += [md(f"#### {name}\n- perf@n (left) and best of n (right), 8 seeds; rows: train, test pairs (BC failures), control pairs"),
               fig(f'{t}/perfbon', 26, f'perf@n and best of n, {name}')]
for i, (t, name) in enumerate(FB):
    blocks += [md(f"### {4 + i}. Feedback {name}\n- perf@n (left) and best of n (right), 8 seeds; rows: train, test pairs (BC failures), control pairs"),
               fig(f'{t}/perfbon', 26, f'perf@n and best of n, feedback {name}'),
               md("- success per goal cell; hatched = goal of a BC-fail test pair"), fig(f'{t}/goalmap', 26, f'goal map, feedback {name}')]



def simple_blocks():
    """Simple-plots section at the top (simple.py -> simple_*.png, simple.json)."""
    sj = json.loads((R / 'simple.json').read_text()) if (R / 'simple.json').exists() else {}
    def cell(v):
        return f"{v['perf1']:.2f} → {v['perf16']:.2f} · **{v['best16']:.2f}**" if v else ''
    def row(d, lab):
        return f"| {lab} | " + " | ".join(cell(d.get(s)) for s in ('train', 'all', 'fail')) + " |"
    def table(key, part):
        d = sj.get(key, {}).get(part) if key != 'noise' else sj.get('noise', {}).get(part)
        if not d:
            return ''
        return ("\nEach cell: candidate 1 → candidate 16 · **best of 16**\n\n"
                "| policy | train starts | all held-out pairs | held-out pairs BC fails |\n|---|---|---|---|\n" +
                "\n".join(row(v, lab) for lab, v in d.items()) + "\n")
    out = [md("## Simple plots\n"
              "- Every ST policy is at its **best checkpoint** of the round, picked by success on 100 unseen pairs of train goals "
              "(checkpoints every 60 steps; held-out goals never used); STk is scored on attempt k+1\n"
              "- Every BC is **closed loop** (horizon 4, re-planned every step, like ST), checkpoint picked by the same rule\n"
              "- x axis = **number of candidates n** (1..16). Top: perf@n, success of candidate n alone. Bottom: best of n, the verifier "
              "(−BFS of the final cell) picks one of the first n. ST candidates = 16 attempts in a row (attempt n sees its last min(7, n − 1) "
              "attempts; STk0 has no context, so its 16 are independent); BC candidates = 16 independent rollouts\n"
              "- Columns: train starts (642) · all held-out pairs (2870 = every start × 41 held-out goals) · the held-out pairs closed-loop "
              "BC fails (≤ 1/3 of 8 seeds; the BC curves use 128 fresh rollouts)\n"
              "- Goal maps: success per goal cell (STk on attempt k+1, BC on one rollout), red = fail, green = success; dashed = held-out goal")]
    names = {'grid': 'Discrete grid maze', 'cont': 'Continuous maze', 'online': 'Online expert (DAgger: the BFS expert labels the policy\'s own attempts)'}
    for k, name in names.items():
        n = sj.get(k, {}).get('n_fail')
        out += [md(f"### {name}" + (f"\n- held-out pairs closed-loop BC fails: {n} of 2870" if n else '')),
                md("#### STk0..7, each at its best checkpoint\n" + table(k, 'stk')),
                fig(f'simple/{k}_stk', 26, f'{name}: STk0..7 at their best checkpoints, perf@n and best of n'),
                fig(f'simple/{k}_stk_goals', 26, f'{name}: goal maps, BC and STk0..7'),
                md("#### Same number of training steps: STk0 (one round, 7200 steps) vs STk7 (8 rounds × 900) vs BC\n" + table(k, 'steps')),
                fig(f'simple/{k}_steps', 26, f'{name}: same training steps'), fig(f'simple/{k}_steps_goals', 14, f'{name}: goal maps, same training steps'),
                md("#### Feedback: STk0 (7200 steps) with vs without feedback vs BC\n" + table(k, 'fb')),
                fig(f'simple/{k}_fb', 26, f'{name}: feedback'), fig(f'simple/{k}_fb_goals', 14, f'{name}: goal maps, feedback')]
        if k == 'grid':
            out += [md("#### STk7 vs STk0 with and without feedback\n- STk7: 8 rounds × 900 steps with feedback, candidate n = attempt n "
                       "(sees its last min(7, n − 1) attempts); STk0: one round, 7200 steps, no context, with and without feedback\n" + table(k, 'k7fb')),
                    fig(f'simple/{k}_k7fb', 26, f'{name}: STk7 vs STk0 with / without feedback'),
                    fig(f'simple/{k}_k7fb_goals', 14, f'{name}: goal maps, STk7 vs STk0 with / without feedback')]
    out += [md("### Observation noise (discrete grid maze)\n- ST trained with noise on its observation features (DDPM t = 400 flat, or "
               "ramped 600 → 200 over context slots 0..7), evaluated clean and with its training noise; BC evaluated clean and with the same "
               "noise on its observation features")]
    for nk, lab in (('flat', 'flat noise t = 400'), ('ramp', 'ramped noise t = 600 → 200')):
        out += [md(f"#### {lab}\n" + table('noise', nk)), fig(f'simple/noise_{nk}', 26, f'{lab}: clean vs noised evaluation'), fig(f'simple/noise_{nk}_goals', 14, f'{lab}: goal maps')]
    return out + [md("---\n## Detailed results")]


blocks = simple_blocks() + blocks
api = wandb.Api(timeout=120); idf = R / 'report_id.txt'
report = (api.load_report('l2sml/maze_reports/reports/' + idf.read_text().strip()) if idf.exists()
          else wr.Report(project='maze_reports', entity='l2sml', title=TITLE, blocks=[md('Building…')]).save())
if not idf.exists(): idf.write_text(report.id)
report.title = TITLE; report.blocks = blocks; report.save()
saved = api.load_report('l2sml/maze_reports/reports/' + idf.read_text().strip()); assert len(saved.blocks) == len(blocks)
url = saved.url.replace('https://wandb.ai/', 'https://forge.coreweave.com/wandb/').replace('wandb//', 'wandb/')
(R / 'report_url.txt').write_text(url); print('Published', url, len(blocks), 'blocks', flush=True)
