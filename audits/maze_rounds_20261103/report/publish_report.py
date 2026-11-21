"""W&B report for the in-context rollout study (maze_rounds_20261103). Creates the report on the first run (report_id.txt),
updates it after; figures that do not exist yet are skipped, so re-run when evals finish.
  C=../../maze_report_consolidated_20261008; TMPDIR=<writable dir> WANDB_DISABLE_SERVICE=true PYTHONPATH=$C/wandb013_shim <vae_pushT_l2s python> publish_report.py
  (without WANDB_DISABLE_SERVICE the wandb service can hang forever on the login node)
"""
import inspect, json, os, textwrap
from pathlib import Path

R = Path(__file__).resolve().parent; HERE = R.parent; RUNS = HERE / 'runs'
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

RID, TITLE = 'maze-incontext-rounds', 'Maze · training the full-trajectory ST policy with its own rollouts in context (11×11, bottom-right held out)'
images = {'split/maze': R / 'maze_split.png', 'curves/s20': R / 'curves_ast_s20.png', 'curves/s1': R / 'curves_ast_s1.png',
          'compare': R / 'compare.png', 'goalmap/s20': R / 'goalmap_ast_s20.png', 'fail/stk2': R / 'fail_ast_s20_ep3_STk2.png', 'fail/stk7': R / 'fail_ast_s20_ep3_STk7.png',
          'curves/s30': R / 'curves_ast_s30.png', 'curves/nofb': R / 'curves_ast_s20_nofb.png',
          'goalmap/main': R / 'goalmap_ast_main.png', 'curves/on400': R / 'curves_ast_s20_on400.png', 'curves/onramp': R / 'curves_ast_s20_onramp.png', 'fbablate': R / 'fb_ablate_ast_s20_r7.png', 'bon/curves': R / 'bon_curves_ast_s30.png', 'bon/line': R / 'bon_line_ast_s30.png', 'goalmap/s30': R / 'goalmap_ast_s30.png', 'bon/main_curves': R / 'bon_curves_ast_main.png', 'bon/main_line': R / 'bon_line_ast_main.png', 'perf/curves': R / 'perf_curves_ast_s30.png',
          'curves/main': R / 'curves_ast_main_curves.png', 'perf/main_curves': R / 'perf_curves_ast_main.png', 'perfbon/s30': R / 'perf_bon_ast_s30.png', 'perfbon/main': R / 'perf_bon_ast_main.png', 'bfs/attempt': R / 'bfs_per_attempt.png', 'perfbon/on400': R / 'perf_bon_ast_s20_on400.png', 'perfbon/onramp': R / 'perf_bon_ast_s20_onramp.png', 'perfbon/on400_nz': R / 'perf_bon_ast_s20_on400_noisyobs.png', 'perfbon/onramp_nz': R / 'perf_bon_ast_s20_onramp_noisyobs.png', 'n3/ast_s30_bfsdir_on400/gm': R / 'goalmap_ast_s30_bfsdir_on400.png', 'n3/ast_s30_bfsdir_on400/gmnz': R / 'goalmap_ast_s30_bfsdir_on400_noisyobs.png', 'n3/ast_s30_bfsdir_onramp/gm': R / 'goalmap_ast_s30_bfsdir_onramp.png', 'n3/ast_s30_bfsdir_onramp/gmnz': R / 'goalmap_ast_s30_bfsdir_onramp_noisyobs.png', 'n3/ast_main_on400/gm': R / 'goalmap_ast_main_on400.png', 'n3/ast_main_on400/gmnz': R / 'goalmap_ast_main_on400_noisyobs.png', 'n3/ast_main_onramp/gm': R / 'goalmap_ast_main_onramp.png', 'n3/ast_main_onramp/gmnz': R / 'goalmap_ast_main_onramp_noisyobs.png', 'n3/ast_s30_bfsdir_on400/curves': R / 'curves_ast_s30_bfsdir_on400.png', 'n3/ast_s30_bfsdir_on400/pb': R / 'perf_bon_ast_s30_bfsdir_on400.png', 'n3/ast_s30_bfsdir_on400/pbnz': R / 'perf_bon_ast_s30_bfsdir_on400_noisyobs.png', 'n3/ast_s30_bfsdir_onramp/curves': R / 'curves_ast_s30_bfsdir_onramp.png', 'n3/ast_s30_bfsdir_onramp/pb': R / 'perf_bon_ast_s30_bfsdir_onramp.png', 'n3/ast_s30_bfsdir_onramp/pbnz': R / 'perf_bon_ast_s30_bfsdir_onramp_noisyobs.png', 'n3/ast_main_on400/curves': R / 'curves_ast_main_on400.png', 'n3/ast_main_on400/pb': R / 'perf_bon_ast_main_on400.png', 'n3/ast_main_on400/pbnz': R / 'perf_bon_ast_main_on400_noisyobs.png', 'n3/ast_main_onramp/curves': R / 'curves_ast_main_onramp.png', 'n3/ast_main_onramp/pb': R / 'perf_bon_ast_main_onramp.png', 'n3/ast_main_onramp/pbnz': R / 'perf_bon_ast_main_onramp_noisyobs.png',
          'failgrid/unsolved': R / 'failgrid_unsolved_ast_main.png', 'failgrid/main_vs_s30': R / 'failgrid_main_vs_s30.png'}
images = {k: v for k, v in images.items() if v.exists()}; assert 'split/maze' in images
run = wandb.init(entity='l2sml', project='maze_reports', id=RID, resume='allow', dir=str(R), name=TITLE, settings=wandb.Settings(start_method='fork'))
run.log({k: wandb.Image(str(p)) for k, p in images.items()}); run.finish()


def grid(key, h, title):
    rs = wr.RunSet(entity='l2sml', project='maze_reports', name='In-context rollouts'); rs.spec['id'] = RID
    rs.spec['selections'] = {'root': 0, 'bounds': [], 'tree': [RID]}; rs.spec['pinnedRunIds'] = [RID]
    p = wr.MediaBrowser(num_columns=1, media_keys=key)
    p.spec['config'] = {'chartTitle': title, 'columnCount': 1, 'mediaKeys': [key], 'mode': 'gallery', 'gallerySettings': {'axis': 'run'}}
    p.spec['layout'] = {'x': 0, 'y': 0, 'w': 24, 'h': h}
    return wr.PanelGrid(runsets=[rs], panels=[p])


def fig(key, h, title):
    return grid(key, h, title) if key in images else md(f'*{title}: eval running, added when it finishes.*')


md = wr.MarkdownBlock
blocks = [
    md("### Setup\n"
       "- **Input:** 3 × 11 × 11 maze image (walls, goal, agent) + agent cell + goal cell; current step only\n"
       "- **Encoder:** flatten + concatenate (367) → linear → 256-d token, + embeddings of the previous move, previous feedback, "
       "stop flag, step and stream position → 4-layer causal transformer → next move (snapped to a cardinal move) + stop flag\n"
       "- **Context:** the policy's own k earlier attempts from the same start (k = 0 … 7): every step's observation, move and "
       "feedback, before the current attempt in one token stream\n"
       "- **Feedback:** every step [−BFS(cell after the move)/32, 0, 0, 0]; first token of the next attempt "
       "[−BFS(final cell)/32, −BFS(final cell)/32, dx/10, dy/10]; BFS = maze distance to the goal, (dx, dy) = goal − final cell"),
    md("### 1. Train and test positions\n- test goals (bottom-right quadrant) held out"),
    grid('split/maze', 16, 'Train and test starts and goals'),
    md("### 2. 30 gradient steps over each batch\n"
       "- collect-and-train cycle k = 0 … 7: the current weights make k attempts in a row from each training start → (those k attempts as context, A\\* demo) are added to the dataset → train → STk\n"
       "- each cycle: 30 batches of 128 × 30 gradient steps each\n"
       "- x = gradient steps within cycle k; eval every 30 steps; train = 637 episodes, test = 100"),
    fig('curves/s30', 11, '30 gradient steps per batch'),
    md("- success per goal cell; STk on attempt k+1; 8 seeds; dashed box = test goals"),
    fig('goalmap/s30', 22, 'Success per goal cell, 30 gradient steps per batch'),
    md("- perf@n (left): success of attempt n alone, no selection; STk: n attempts in a row, attempt n with n − 1 earlier attempts in context; BC: its n-th iid sample\n"
       "- best of n (right): verifier = −BFS(final cell) picks one of the first n, that candidate is executed\n"
       "- 637 train / 100 test episodes × 8 seeds"),
    fig('perfbon/s30', 21, 'perf@n and best of n: STk0 … STk7 and BC'),
    md("- STk at n = k + 1 only (its trained context length); BC at the same n"),
    fig('bon/line', 11, 'Best of n, STk at its trained context length'),
    md("### 3. Noised observation\n"
       "- training only: the predicted attempt's encoded observation x → √ᾱ_t·x + √(1−ᾱ_t)·ε·std (DDPM, T = 1000, linear β); attempts in context clean\n"
       "- flat: t = 400 at every context length; ramp: t = 600 (no context) → 200 (7 attempts in context)\n"
       "- eval: clean observations, and observations noised as in training (attempts in context clean)"),
    md("#### 3.1 30 gradient steps per batch, dx, dy feedback\n"
       "- cycles and training as section 2\n"
       "- feedback: every step [−BFS(cell after the move)/32, 0, dx, dy]; first token of the next attempt [−BFS(final cell)/32, −BFS(final cell)/32, dx, dy]; (dx, dy) = (goal − cell)/10"),
    md("- flat t = 400"), fig('n3/ast_s30_bfsdir_on400/curves', 11, 'flat t = 400: training curves'),
    md("- success per goal cell; STk on attempt k+1; 8 seeds; dashed box = test goals; first clean eval, then observations noised as in training"),
    fig('n3/ast_s30_bfsdir_on400/gm', 22, 'flat t = 400: success per goal cell, clean eval'), fig('n3/ast_s30_bfsdir_on400/gmnz', 22, 'flat t = 400: success per goal cell, noised eval'),
    md("- perf@n (left) and best of n (right), clean observations, 8 seeds"), fig('n3/ast_s30_bfsdir_on400/pb', 21, 'flat t = 400: perf@n and best of n, clean eval'),
    md("- perf@n (left) and best of n (right), observations noised as in training, 8 seeds"), fig('n3/ast_s30_bfsdir_on400/pbnz', 21, 'flat t = 400: perf@n and best of n, noised eval'),
    md("- ramp t = 600 → 200"), fig('n3/ast_s30_bfsdir_onramp/curves', 11, 'ramp t = 600 → 200: training curves'),
    md("- success per goal cell; STk on attempt k+1; 8 seeds; dashed box = test goals; first clean eval, then observations noised as in training"),
    fig('n3/ast_s30_bfsdir_onramp/gm', 22, 'ramp t = 600 → 200: success per goal cell, clean eval'), fig('n3/ast_s30_bfsdir_onramp/gmnz', 22, 'ramp t = 600 → 200: success per goal cell, noised eval'),
    md("- perf@n (left) and best of n (right), clean observations, 8 seeds"), fig('n3/ast_s30_bfsdir_onramp/pb', 21, 'ramp t = 600 → 200: perf@n and best of n, clean eval'),
    md("- perf@n (left) and best of n (right), observations noised as in training, 8 seeds"), fig('n3/ast_s30_bfsdir_onramp/pbnz', 21, 'ramp t = 600 → 200: perf@n and best of n, noised eval'),
    md("#### 3.2 10k updates per collect-and-train cycle, BFS feedback\n"
       "- cycles and training as section 6; feedback as in Setup\n"
       "- eval every 500 updates; train = 637 episodes × 8 seeds, test = 100 × 8"),
    md("- flat t = 400"), fig('n3/ast_main_on400/curves', 11, 'flat t = 400: training curves'),
    md("- success per goal cell; STk on attempt k+1; 8 seeds; dashed box = test goals; first clean eval, then observations noised as in training"),
    fig('n3/ast_main_on400/gm', 22, 'flat t = 400: success per goal cell, clean eval'), fig('n3/ast_main_on400/gmnz', 22, 'flat t = 400: success per goal cell, noised eval'),
    md("- perf@n (left) and best of n (right), clean observations, 8 seeds"), fig('n3/ast_main_on400/pb', 21, 'flat t = 400: perf@n and best of n, clean eval'),
    md("- perf@n (left) and best of n (right), observations noised as in training, 8 seeds"), fig('n3/ast_main_on400/pbnz', 21, 'flat t = 400: perf@n and best of n, noised eval'),
    md("- ramp t = 600 → 200"), fig('n3/ast_main_onramp/curves', 11, 'ramp t = 600 → 200: training curves'),
    md("- success per goal cell; STk on attempt k+1; 8 seeds; dashed box = test goals; first clean eval, then observations noised as in training"),
    fig('n3/ast_main_onramp/gm', 22, 'ramp t = 600 → 200: success per goal cell, clean eval'), fig('n3/ast_main_onramp/gmnz', 22, 'ramp t = 600 → 200: success per goal cell, noised eval'),
    md("- perf@n (left) and best of n (right), clean observations, 8 seeds"), fig('n3/ast_main_onramp/pb', 21, 'ramp t = 600 → 200: perf@n and best of n, clean eval'),
    md("- perf@n (left) and best of n (right), observations noised as in training, 8 seeds"), fig('n3/ast_main_onramp/pbnz', 21, 'ramp t = 600 → 200: perf@n and best of n, noised eval'),
    md("### 4. 20 gradient steps over each batch\n- each cycle: 30 batches of 128 × 20 gradient steps each"),
    fig('curves/s20', 11, '20 gradient steps per batch'),
    md("- success per goal cell; STk on attempt k+1; 8 seeds"),
    fig('goalmap/s20', 22, 'Success per goal cell, 20 gradient steps per batch'),
    md("- test episode 3 (goal at the box's bottom-left cell), 0 of 8 seeds for both STk2 and STk7\n"
       "- every attempt in a row from the start, attempt j with j − 1 earlier attempts in context; dashed = shortest path"),
    fig('fail/stk2', 8, 'STk2: 3 attempts in a row on test episode 3'),
    fig('fail/stk7', 15, 'STk7: 8 attempts in a row on test episode 3'),
    md("### 5. Context feedback: STk7 with the feedback vs zeroed\n"
       "- same STk7 weights (20 steps); black = feedback as trained; blue = earlier attempts' feedback set to 0, cells and moves kept\n"
       "- top: attempt n; bottom: best of the first n with the verifier; 8 seeds"),
    fig('fbablate', 21, 'STk7: feedback in context vs zeroed'),
    md("- STk0 … STk7 trained with the context feedback zeroed"),
    fig('curves/nofb', 11, 'context without feedback'),
    md("### 6. 10k updates per collect-and-train cycle\n"
       "- same cycles, policy, context and feedback as section 2; only the training in each cycle differs\n\n"
       "| | 30 steps per batch (section 2) | 10k updates per cycle |\n|---|---|---|\n"
       "| gradient updates per cycle | 900 (30 batches × 30 steps) | 10,000 |\n"
       "| batch | 128 items, each batch reused for 30 steps | a fresh random batch of 64 items every update |\n"
       "| learning rate | constant 1e-4 | 500-update warmup, then cosine decay from 1e-4 |\n"
       "| total updates by STk7 | 7,200 | 80,000 |\n\n"
       "- x = updates within cycle k; eval every 500 updates; train = 637 episodes, test = 100\n"
       "- curves from a rerun with the same settings and evals during training (the first run logged them only for cycles 6–7)"),
    fig('curves/main', 11, '10k updates per cycle'),
    md("- success per goal cell; STk on attempt k+1; 8 seeds; dashed box = test goals"),
    fig('goalmap/main', 22, 'Success per goal cell, 10k updates per cycle'),
    md("- perf@n (left) and best of n (right), as in section 2"),
    fig('perfbon/main', 21, 'perf@n and best of n: STk0 … STk7 and BC, 10k updates per cycle'),
    md("- mean BFS of the final cell per attempt, test starts; left 30 steps per batch, right 10k updates per cycle\n"
       "- bottom: STk7 only, goal cells grouped by its attempt-8 success (≥ 50%, between 0 and 50%, never)"),
    fig('bfs/attempt', 21, 'Final-cell BFS per attempt: 30 steps per batch vs 10k updates per cycle'),
    md("- STk at n = k + 1 only (its trained context length); BC at the same n"),
    fig('bon/main_line', 11, 'Best of n, STk at its trained context length, 10k updates per cycle'),
    md("- test goals never solved by STk2 or STk7 with 10k updates (0 of 8 seeds); one episode per goal cell\n"
       "- one chain per panel: every attempt in a row from the start, overlaid, colour = attempt number; dashed = shortest path"),
    fig('failgrid/unsolved', 40, 'Never-solved test goals, 10k updates per cycle: STk2 and STk7 attempts'),
    md("- test episodes STk7 solves with 10k updates but not with 30 steps per batch; one episode per goal cell\n"
       "- left: 30 steps per batch STk7; right: 10k updates STk7; 8 attempts in a row each"),
    fig('failgrid/main_vs_s30', 32, 'STk7: 30 steps per batch vs 10k updates per cycle'),
    md("### 7. Training observation noise, 20 gradient steps per batch\n"
       "- predicted attempt's encoded observation: x → √ᾱ_t·x + √(1−ᾱ_t)·ε·std (DDPM, T = 1000, linear β)\n"
       "- context, rollouts and evals clean"),
    md("- flat t = 400"), fig('curves/on400', 11, 'training obs noise, flat t = 400'),
    md("- perf@n (left) and best of n (right), clean observations, 8 seeds"), fig('perfbon/on400', 21, 'perf@n and best of n, obs noise flat t = 400'),
    md("- perf@n (left) and best of n (right), observations noised as in training, 8 seeds"), fig('perfbon/on400_nz', 21, 'perf@n and best of n, obs noise flat t = 400, noised eval'),
    md("- ramp t = 600 (no context) → 200 (7 attempts in context)"), fig('curves/onramp', 11, 'training obs noise, ramp 600 → 200'),
    md("- perf@n (left) and best of n (right), clean observations, 8 seeds"), fig('perfbon/onramp', 21, 'perf@n and best of n, obs noise ramp 600 → 200'),
    md("- perf@n (left) and best of n (right), observations noised as in training, 8 seeds"), fig('perfbon/onramp_nz', 21, 'perf@n and best of n, obs noise ramp 600 → 200, noised eval'),
    md("### 8. 1 vs 20 gradient steps per batch\n- point = STk at the end of cycle k; x = own rollouts collected before STk is trained"),
    fig('compare', 11, 'BC vs 1 vs 20 gradient steps per batch'),
    md("- 1 gradient step per batch"), fig('curves/s1', 11, '1 gradient step per batch'),
]

api = wandb.Api(timeout=120); idf = R / 'report_id.txt'
report = (api.load_report('l2sml/maze_reports/reports/' + idf.read_text().strip()) if idf.exists()
          else wr.Report(project='maze_reports', entity='l2sml', title=TITLE, blocks=[md('Building…')]).save())
if not idf.exists(): idf.write_text(report.id)
report.title = TITLE; report.blocks = blocks; report.save()
saved = api.load_report('l2sml/maze_reports/reports/' + idf.read_text().strip()); assert len(saved.blocks) == len(blocks)
url = saved.url.replace('https://wandb.ai/', 'https://forge.coreweave.com/wandb/').replace('wandb//', 'wandb/')
(R / 'report_url.txt').write_text(url); print('Published', url, len(blocks), 'blocks', flush=True)
