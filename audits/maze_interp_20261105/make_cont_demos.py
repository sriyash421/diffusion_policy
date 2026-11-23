"""Continuous (smoothed A*) demos. --setting interp30 (default) -> interp30c; --setting 11br -> 11brc (2026-11-22, the held-out
bottom-right quadrant; output datasets/11brc/, entry 11brc). Below, interp30 / interp30c stand for <setting> / <setting>c.
For MAZE_SETTING=interp30c: the same 800 (start cell, goal) episodes as datasets/interp30, in the same
order (so the pair train/val split is identical), with a jittered real-valued start and string-pulled paths (cmaze.py). Writes
datasets/interp30c/data.zarr (same keys as maze.make_dataset), the interp30c entry of maze_quadrants_delta_20261019/datasets.json
(backup datasets.before_cont.json), and datasets/interp30c/checks.json. CPU only.
"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
import torch
import zarr

I = Path(__file__).resolve().parent; D = I.parent / 'maze_quadrants_delta_20261019'
sys.path[:0] = [str(I), str(D), '/gscratch/robotics/harine/L2S']
import cmaze
from setup_data import maze
from diffusion_policy.common.replay_buffer import ReplayBuffer

ap = argparse.ArgumentParser(); ap.add_argument('--setting', default='interp30', choices=['interp30', '11br']); SRC = ap.parse_args().setting
NAME = SRC + 'c'
spec = json.loads((D / 'datasets.json').read_text())[SRC]
src = zarr.open(spec['dataset'], 'r'); S = src['meta/start_xy'][:]; G = src['meta/goal_xy'][:]
env = maze.MazeEnv(spec['layouts'], seed=0, split='train', layout_index=0, goal_split=spec['goal_split']); env.reset()
nav = env.maze.astype(bool); free1 = torch.as_tensor(nav)[None]
assert np.array_equal(nav, np.asarray(src['data/obs/image'][0, :, :, 0]) == 0)
rng = np.random.default_rng(0); buf = ReplayBuffer.create_empty_numpy(); lens, steps, grid_steps, worst = [], [], [], 0.
for s, g in zip(S, G):
    start = s.astype(np.float64) + rng.uniform(-cmaze.JITTER, cmaze.JITTER, 2)
    cells = maze.shortest_path(env, s, g); grid_steps.append(len(cells) - 1)
    pts = cmaze.resample(cmaze.smooth_path(free1, start, cells)).astype(np.float32)
    act = np.diff(pts, axis=0)
    assert np.linalg.norm(act, axis=-1).max() <= 1 + 1e-5 and np.allclose(pts[-1], g)
    p = torch.as_tensor(pts[:1]); rep = [p]                       # replay through the env step: must follow the plan exactly
    for a in act:
        p = cmaze.cont_step_torch(free1, p, torch.as_tensor(a[None])); rep.append(p)
    worst = max(worst, float((torch.cat(rep).numpy() - pts).__abs__().max()))
    T = len(act); steps.append(T); lens.append(float(np.linalg.norm(act, axis=-1).sum()))
    img = np.stack([maze.build_image(env.maze, np.rint(q).astype(np.int32), g) for q in pts[:-1]])
    rew = np.zeros(T, np.float32); rew[-1] = 1; done = np.zeros(T, bool); done[-1] = True
    buf.add_episode({'obs/image': img, 'obs/agent_pos': pts[:-1], 'obs/goal_pos': np.repeat(g[None].astype(np.float32), T, 0),
                     'actions': act.astype(np.float32), 'reward': rew, 'done': done, 'expert_mask': np.ones((T, 1), np.float32)})
assert worst < 1e-3, worst
out = I / f'datasets/{NAME}/data.zarr'; shutil.rmtree(out, ignore_errors=True); out.parent.mkdir(parents=True, exist_ok=True)
buf.save_to_path(str(out)); meta = zarr.open(str(out), 'a')['meta']
meta.array('start_xy', S.astype(np.int32), overwrite=True); meta.array('goal_xy', G.astype(np.int32), overwrite=True)
meta.attrs.update(dict(src['meta'].attrs)); meta.attrs.update(continuous=True, jitter=cmaze.JITTER, max_step=cmaze.MAX_STEP)
an = np.linalg.norm(np.asarray(buf['actions']), axis=-1)
checks = dict(episodes=len(S), steps_mean=float(np.mean(steps)), steps_max=int(max(steps)), grid_steps_mean=float(np.mean(grid_steps)),
              grid_steps_max=int(max(grid_steps)), path_length_mean=float(np.mean(lens)), action_norm=dict(min=float(an.min()), mean=float(an.mean()),
              max=float(an.max())), diagonal_fraction=float(((np.abs(np.asarray(buf['actions'])) > 1e-4).all(-1)).mean()), replay_max_err=worst)
(out.parent / 'checks.json').write_text(json.dumps(checks, indent=1)); print(json.dumps(checks))
dj = D / 'datasets.json'
if not (D / 'datasets.before_cont.json').exists():
    shutil.copy(dj, D / 'datasets.before_cont.json')
allspec = json.loads(dj.read_text())
allspec[NAME] = dict(spec, dataset=str(out), expert_max=int(max(steps)), continuous=True, actions='continuous', jitter=cmaze.JITTER,
                     max_step=cmaze.MAX_STEP, demo_clearance=cmaze.CLEAR)
dj.write_text(json.dumps(allspec, indent=1)); print('datasets.json', NAME, allspec[NAME])
