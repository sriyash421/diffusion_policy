"""W&B report for the DAgger runs with a hidden goal (7x7 window / DDPM-noised image): plots_simple.py + plots_positions.py figures.
Creates the report on the first run (report/dag_report_id.txt), updates it after. Missing figures (runs still training) are skipped.
  WANDB_DISABLE_SERVICE=true TMPDIR=<dir> PYTHONPATH=<maze_report_consolidated_20261008>/wandb013_shim python publish_dag_report.py   (login node)
"""
import inspect, os, textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent; R = HERE / 'report'; PIPE = HERE.parent / 'maze_report_consolidated_20261008'
os.environ.pop('WANDB_API_KEY', None); os.environ['WANDB_DIR'] = str(HERE); os.environ['WANDB_CACHE_DIR'] = str(HERE / 'wandb_cache')
import wandb
os.environ['PYTHONPATH'] = str(PIPE / 'wandb_compat') + os.pathsep + os.environ.get('PYTHONPATH', '')
from wandb.sdk.internal import internal_api  # W&B 0.13.3 vs current servers: optional server-settings type may be null
src = textwrap.dedent(inspect.getsource(internal_api.Api.server_settings_introspection))
old = 'res.get("ServerSettingsType", {}).get("fields", [{}])'
if old in src:
    ns = dict(vars(internal_api)); exec(compile(src.replace(old, '(res.get("ServerSettingsType") or {}).get("fields", [])'), 'wandb_compat', 'exec'), ns)
    internal_api.Api.server_settings_introspection = ns['server_settings_introspection']
wandb.require('report-editing:v0')
import wandb.apis.reports as wr

RID, TITLE = 'maze-dag-hidden-goal', 'Maze · DAgger with a hidden goal: does success rise with more attempts in context?'
SETS = [('interp30c', 'One maze · random split'), ('11brc', 'One maze · bottom-right quadrant held out'), ('pg', 'New mazes')]
OBS = [('window', '7×7 window'), ('ddpm400', 'DDPM t=400')]
want = {'overview/train': R / 'dag_overview_train.png', 'overview/heldout': R / 'dag_overview_heldout.png',
        'mse/heldout': R / 'dag_mse_overview_heldout.png', 'mse/train': R / 'dag_mse_overview_train.png', 'mse/end': R / 'dag_end.png',
        'ns/train': R / 'dag_overview_train_ns.png', 'ns/heldout': R / 'dag_overview_heldout_ns.png',
        'ns/mse_heldout': R / 'dag_mse_overview_heldout_ns.png', 'ns/end': R / 'dag_end_ns.png'}
for s, _ in SETS:
    want.update({f'{s}/pos': R / f'pos_{s}.png', f'{s}/succmap': R / f'succmap_{s}.png'})
    want.update({f'{s}/{o}': R / f'dag_{s}_{o}.png' for o, _ in OBS})
    want[f'{s}/sched'] = R / f'dag_sched_{s}.png'
images = {k: p for k, p in want.items() if p.exists()}
print('skipped (not rendered yet):', sorted(set(want) - set(images)))

run = wandb.init(entity='l2sml', project='maze_reports', id=RID, resume='allow', dir=str(HERE), name=TITLE, settings=wandb.Settings(start_method='fork'))
run.log({k: wandb.Image(str(p)) for k, p in images.items()}); run.finish()


def grid(keys, h, titles=None):
    keys = [k for k in ([keys] if isinstance(keys, str) else keys) if k in images]
    if not keys:
        return md('*(still training)*')
    rs = wr.RunSet(entity='l2sml', project='maze_reports', name='DAgger hidden goal'); rs.spec['id'] = RID
    rs.spec['selections'] = {'root': 0, 'bounds': [], 'tree': [RID]}; rs.spec['pinnedRunIds'] = [RID]
    panels, w = [], 24 // len(keys)
    for i, key in enumerate(keys):
        p = wr.MediaBrowser(num_columns=1, media_keys=key)
        p.spec['config'] = {'chartTitle': (titles or {}).get(key, key), 'columnCount': 1, 'mediaKeys': [key], 'mode': 'gallery',
                            'gallerySettings': {'axis': 'run'}}
        p.spec['layout'] = {'x': i * w, 'y': 0, 'w': w, 'h': h}; panels.append(p)
    return wr.PanelGrid(runsets=[rs], panels=panels)


md = wr.MarkdownBlock
blocks = [
    md("**Setup.** Continuous maze, goal hidden from the policy (7×7 window around the agent, or the whole image DDPM-noised). "
       "ST transformer trained with DAgger: the student makes attempts, the expert labels every point it visited. "
       "**x** = earlier attempts in context, **y** = success of the next attempt, final-round policy.  \n"
       "**Arms:** context + reward feedback · context + BFS + goal offset (leaks the goal: positive control) · context, no feedback · "
       "trained without context · trained without context on the clean image (goal visible)."),
    md("**Takeaway.** Only the leak arm improves with more attempts in context. Reward feedback and no feedback do not beat training "
       "without context. With the goal visible (grey dotted), every train failure is the stop head firing one step short of the "
       "goal: with the stop head off it solves every train start. On the held-out quadrant it stays low either way (wrong moves). "
       "Panels marked *running* fill in as runs finish."),
    md('# Overview'),
    grid(['overview/train', 'overview/heldout'], 22, {'overview/train': 'Train starts', 'overview/heldout': 'Held out'}),
    md('# Action error to the expert'),
    grid(['mse/heldout', 'mse/train'], 22, {'mse/heldout': 'Held out', 'mse/train': 'Train starts'}),
    grid('mse/end', 16, {'mse/end': 'How attempts end'}),
    md('# Stop head off (retrained: attempts end at the goal or the step budget)'),
    grid(['ns/train', 'ns/heldout'], 22, {'ns/train': 'Train starts', 'ns/heldout': 'Held out'}),
    grid(['ns/mse_heldout', 'ns/end'], 16, {'ns/mse_heldout': 'Action error, held out', 'ns/end': 'How attempts end'}),
]
for s, name in SETS:
    blocks += [md(f'# {name}'), grid([f'{s}/{o}' for o, _ in OBS], 12, {f'{s}/{o}': on for o, on in OBS}),
               *([grid(f'{s}/sched', 18, {f'{s}/sched': 'DAgger rounds, updates, context length'})]
                 if s != 'pg' else []),
               md('Train vs held-out positions, and success of the last attempt per goal cell (green = success, red = fail).'),
               grid([f'{s}/pos', f'{s}/succmap'], 16, {f'{s}/pos': 'Positions', f'{s}/succmap': 'Success by goal position'})]

api = wandb.Api(timeout=120); idf = R / 'dag_report_id.txt'
report = (api.load_report('l2sml/maze_reports/reports/' + idf.read_text().strip()) if idf.exists()
          else wr.Report(project='maze_reports', entity='l2sml', title=TITLE, blocks=[md('Building…')]).save())
if not idf.exists():
    idf.write_text(report.id)
report.title = TITLE; report.blocks = blocks; report.save()
saved = api.load_report('l2sml/maze_reports/reports/' + idf.read_text().strip()); assert len(saved.blocks) == len(blocks)
url = saved.url.replace('https://wandb.ai/', 'https://forge.coreweave.com/wandb/').replace('wandb//', 'wandb/')
(R / 'dag_report_url.txt').write_text(url); print('Published', url, len(blocks), 'blocks', flush=True)
