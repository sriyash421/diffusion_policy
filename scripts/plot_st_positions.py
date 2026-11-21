"""Per-start figures for one in-context arm (train_pusht_dt_asteroid), from its stage candidates
(<run>/bon_search/stage_eval/pi_k<k>/{train,test}_candidates.npz) and the U-Net BC's.

  startmap.png       BC + STk0..7: every train / test start at its T start position, colour = max coverage
                     of the scored candidate (STk: candidate k+1, k own attempts in context; BC: candidate 1).
  fail_stk<k>.png    the test start STk2 and STk7 (--fail-stk) handle worst (lowest summed coverage of their scored
                     candidate): STk's 8 attempts in a row, replayed from the saved actions -- agent path,
                     T at the start (dashed), T at its best-coverage step (solid), goal T (dotted).

    python scripts/plot_st_positions.py <run> --bc-run <unet run> --bc-step 50000 --out-prefix <dir>/x_
"""
import json
import pathlib
import sys

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from make_geometric_splits import QUADRANT_SIDES, WS
from plot_st_candidates import bc_candidates, manifest_test

ZARR = ROOT / 'data/pusht_cchi_v7_replay.zarr'
FAIL_STK = (2, 7)
N = 8


def init_states():
    """(n_episodes, 5) [agent xy, T xy, T angle] at each demo's first frame."""
    import zarr
    root = zarr.open(str(ZARR), 'r')
    ends = np.asarray(root['meta']['episode_ends'][:])
    starts = np.concatenate([[0], ends[:-1]])
    return np.concatenate([np.asarray(root['data']['agent_pos'][:])[starts],
                           np.asarray(root['data']['block_pos'][:])[starts]], -1).astype(np.float64)


def load(run, k, split):
    f = pathlib.Path(run) / 'bon_search' / 'stage_eval' / f'pi_k{k}' / f'{split}_candidates.npz'
    return np.load(f, allow_pickle=True) if f.is_file() else None


def held_out_box(ax, held_out):
    if held_out == 'none':
        return
    hx, hy = QUADRANT_SIDES[held_out]
    ax.add_patch(Rectangle((WS / 2 if hx else 0, WS / 2 if hy else 0), WS / 2, WS / 2, fill=False,
                           ls='--', lw=1.5, edgecolor='black'))


def plot_startmap(label, run, bc_run, bc_step, xy, path, held_out, keep, bcfail):
    panels = [('BC (UNet), candidate 1', {s: bc_candidates(bc_run, bc_step, s, keep if s == 'test' else None)
                                          for s in ('train', 'test')}, 0)]
    panels += [(f'STk{k} (context = {k}), candidate {k + 1}', {s: load(run, k, s) for s in ('train', 'test')}, k)
               for k in range(N)]
    fig, axs = plt.subplots(3, 3, figsize=(12, 11.5), constrained_layout=True)
    sc = None
    for ax, (title, z, j) in zip(axs.flat, panels):
        held_out_box(ax, held_out)
        means = {}
        for s, mk in (('train', 'o'), ('test', 's')):
            if z[s] is None:
                continue
            cov = z[s]['max_coverage'][:, j]; e = z[s]['episode_idxs']
            sc = ax.scatter(xy[e, 0], xy[e, 1], c=cov, cmap='cividis', vmin=0, vmax=1, marker=mk, s=60,
                            edgecolor='black', lw=.6, zorder=3)
            means[s] = cov.mean()
            if s == 'test' and bcfail:
                f = np.isin(e, bcfail)
                for c, lw in (('black', 3), ('white', 1.4)):
                    ax.scatter(xy[e[f], 0], xy[e[f], 1], marker='x', s=30, c=c, lw=lw, zorder=4)
        ax.set_xlim(0, WS); ax.set_ylim(WS, 0); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        sub = '   ·   '.join(f'{s} {v:.2f}' for s, v in means.items()) if means else 'not evaluated yet'
        ax.set_title(f'{title}\n{sub}', fontsize=9)
    if sc is not None:
        fig.colorbar(sc, ax=axs, shrink=.6, label='max T coverage of the scored candidate')
    keys = ['circle = train start', 'square = test start']
    keys += ['x = BC-failure start'] if bcfail else []
    keys += ['dashed box = held-out quadrant'] if held_out != 'none' else []
    fig.suptitle(f'{label}: coverage per start at the T start position ({", ".join(keys)}; 1 seed)', fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches='tight'); plt.close(fig)
    print('wrote', path)


def replay(state, actions):
    """Agent path, T vertices at the start and at the best-coverage step, the goal T's, max coverage."""
    from diffusion_policy.env.pusht.pusht_env import PushTEnv
    env = PushTEnv(legacy=False)
    env.reset_to_state = state
    env.reset()

    def tee(body):
        return [np.array([body.local_to_world(v) for v in s.get_vertices()]) for s in env.block.shapes]

    goal = tee(env._get_goal_pose_body(env.goal_pose))
    start, best, best_cov = tee(env.block), tee(env.block), -1.
    path = [np.array(env.agent.position)]
    for a in actions:
        env.step(np.asarray(a, dtype=np.float64))
        path.append(np.array(env.agent.position))
        if env.max_coverage > best_cov:
            best_cov, best = env.max_coverage, tee(env.block)
    return np.array(path), start, best, goal, env.max_coverage


def plot_fail(label, run, k, ep_row, states, path):
    z = load(run, k, 'test')
    e = int(z['episode_idxs'][ep_row])
    fig, axs = plt.subplots(2, 4, figsize=(17, 9))
    for j, ax in enumerate(axs.flat):
        p, start, best, goal, cov = replay(states[e], list(z['actions'][ep_row, j]))
        for poly in goal:
            ax.add_patch(Polygon(poly, closed=True, fill=False, ls=':', lw=1.4, edgecolor='#009E73'))
        for poly in start:
            ax.add_patch(Polygon(poly, closed=True, fill=False, ls='--', lw=1.2, edgecolor='#555555'))
        for poly in best:
            ax.add_patch(Polygon(poly, closed=True, facecolor='#E69F00', alpha=.5, edgecolor='#E69F00', lw=1.4))
        ax.plot(p[:, 0], p[:, 1], color='#0072B2', lw=1, alpha=.8)
        ax.plot(*p[0], 'o', color='#0072B2', ms=8, mec='black')
        ax.plot(*p[-1], 'X', color='#D55E00', ms=9, mec='black')
        rec = z['max_coverage'][ep_row, j]
        tag = ' (scored)' if j == k else ''
        chk = '' if abs(cov - rec) < 1e-3 else f' [replay {cov:.2f}]'
        ax.set_title(f'attempt {j + 1} ({j} in context){tag}: coverage {rec:.2f}{chk}, {len(p) - 1} steps',
                     fontsize=9, fontweight='bold' if tag else 'normal')
        ax.set_xlim(0, WS); ax.set_ylim(WS, 0); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{label}: STk{k} on test start {ep_row} (episode {e}), 8 attempts in a row\n'
                 'blue line = agent path (circle start, X end); dashed grey = T at start; orange = T at its '
                 'best-coverage step; dotted green = goal T', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print('wrote', path)


@click.command()
@click.argument('run')
@click.option('--label', default='')
@click.option('--bc-run', required=True)
@click.option('--bc-step', required=True, type=int)
@click.option('--out-prefix', required=True)
@click.option('--fail-stk', multiple=True, type=int, default=FAIL_STK)
@click.option('--held-out', default='bottomleft', type=click.Choice([*QUADRANT_SIDES, 'none']),
              help='quadrant to outline (blq50far: bottomleft); none for a random split')
@click.option('--test-manifest', default=None, help="the run's manifest: BC's test rows are cut to its test starts")
@click.option('--bcfail', default=None, help='<manifest>.bcfail.json: marks the BC-failure test starts')
def main(run, label, bc_run, bc_step, out_prefix, fail_stk, held_out, test_manifest, bcfail):
    if not any(load(run, k, 'test') is not None for k in range(N)):
        return
    states = init_states()
    fails = json.loads(pathlib.Path(bcfail).read_text())['bc_fail_test'] if bcfail else None
    plot_startmap(label, run, bc_run, bc_step, states[:, 2:4], f'{out_prefix}startmap.png', held_out,
                  manifest_test(test_manifest), fails)
    zs = [load(run, k, 'test') for k in fail_stk]
    if any(z is None for z in zs):
        return
    worst = int(np.argmin(sum(z['max_coverage'][:, k] for z, k in zip(zs, fail_stk))))
    for k in fail_stk:
        plot_fail(label, run, k, worst, states, f'{out_prefix}fail_stk{k}.png')


if __name__ == '__main__':
    main()
