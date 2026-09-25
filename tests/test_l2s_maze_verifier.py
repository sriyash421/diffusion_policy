"""MazeVerifier must reproduce ProcgenMazeEnv.step exactly.

    python -m pytest tests/test_l2s_maze_verifier.py -q
"""
import pathlib
import sys

import numpy as np
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/l2s_maze'))

from diffusion_policy.common.l2s_maze_verifier import MazeVerifier  # noqa: E402
from expert_utils import make_env  # noqa: E402
from maze_layout import build_layout, free_cells  # noqa: E402

N_CHUNKS = 5000
T = 15


@pytest.fixture(scope='module')
def maze():
    return build_layout(seed=0)


def env_rollout(env, start, actions):
    env.reset(maze_index=0, start_ij=start, goal_ij=start)
    env.agent_xy = np.asarray(start, dtype=np.float64)  # continuous start
    for a in actions:
        env.step(a)
    return env.agent_xy.copy()


def random_chunks(rng, maze, n):
    """Continuous starts inside free cells; three action regimes, incl. wall-hugging."""
    cells = np.array(free_cells(maze), dtype=np.float64)
    starts = cells[rng.integers(len(cells), size=n)] + rng.uniform(-0.45, 0.45, (n, 2))
    actions = rng.uniform(-1.5, 1.5, (n, T, 2))                  # exercises the clip
    third = n // 3
    # constant direction: drives into walls and slides along them
    actions[third:2 * third] = rng.uniform(-1.5, 1.5, (third, 1, 2))
    # near-axis moves: one component tiny, so single-axis sliding dominates
    k = slice(2 * third, n)
    actions[k, :, rng.integers(2)] *= 0.05
    return starts, actions


def test_rollout_matches_env(maze):
    rng = np.random.default_rng(0)
    starts, actions = random_chunks(rng, maze, N_CHUNKS)
    env = make_env(maze)
    expected = np.stack([env_rollout(env, s, a) for s, a in zip(starts, actions)])
    got = MazeVerifier(maze=maze).rollout(
        torch.from_numpy(starts), torch.from_numpy(actions))[:, -1].numpy()
    np.testing.assert_array_equal(got, expected)
    # sanity: the test actually hits walls (some chunks end short of free motion)
    free_motion = starts + 0.25 * np.clip(actions, -1, 1).sum(1)
    assert (np.abs(free_motion - expected).max(1) > 1e-6).mean() > 0.3


def test_float32_inputs_close(maze):
    rng = np.random.default_rng(1)
    starts, actions = random_chunks(rng, maze, N_CHUNKS)
    v = MazeVerifier(maze=maze)
    f64 = v.rollout(torch.from_numpy(starts), torch.from_numpy(actions))[:, -1]
    f32 = v.rollout(torch.from_numpy(starts).float(), torch.from_numpy(actions).float())[:, -1]
    close = (f64 - f32).abs().max(1).values < 1e-4
    assert close.float().mean() >= 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_cpu_cuda_agree(maze):
    rng = np.random.default_rng(2)
    starts, actions = random_chunks(rng, maze, N_CHUNKS)
    v = MazeVerifier(maze=maze)
    cpu = v.rollout(torch.from_numpy(starts), torch.from_numpy(actions))
    gpu = v.rollout(torch.from_numpy(starts).cuda(), torch.from_numpy(actions).cuda())
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=0, atol=0)


def test_get_value(maze):
    rng = np.random.default_rng(3)
    B, To, H = 64, 2, 16
    starts, actions = random_chunks(rng, maze, B)
    chunk = np.concatenate([rng.uniform(-1, 1, (B, 1, 2)), actions], axis=1)  # (B, 16, 2)
    goals = np.array(free_cells(maze), dtype=np.float64)[rng.integers(127, size=B)]
    obs = {
        # frame t-1 is garbage: only agent_pos[:, -1] may be used
        'agent_pos': torch.from_numpy(np.stack([starts + 99, starts], 1)).float(),
        'goal_pos': torch.from_numpy(np.stack([goals, goals], 1)).float(),
    }
    v = MazeVerifier(maze=maze, start_offset=To - 1)
    value = v.get_value(obs, torch.from_numpy(chunk).float())
    assert value.shape == (B,) and value.dtype == torch.float32
    final = v.rollout(obs['agent_pos'][:, -1], torch.from_numpy(chunk[:, 1:]).float())[:, -1]
    expected = -torch.linalg.norm(final - torch.from_numpy(goals), dim=-1).float()
    torch.testing.assert_close(value, expected)
    # (B, N, H, 2) input scores the first candidate
    torch.testing.assert_close(v.get_value(obs, torch.from_numpy(chunk).float()[:, None]), value)


def test_noise(maze):
    B = 4096
    obs = {'agent_pos': torch.full((B, 2, 2), 5.0), 'goal_pos': torch.full((B, 2, 2), 9.0)}
    act = torch.zeros(B, 16, 2)
    clean = MazeVerifier(maze=maze).get_value(obs, act)
    noisy = MazeVerifier(maze=maze, noise=0.1).get_value(obs, act)
    ratio = noisy / clean
    assert abs(ratio.mean().item() - 1) < 0.01 and abs(ratio.std().item() - 0.1) < 0.01


ZARR = REPO / 'data/l2s_maze/as_is/train/expert.zarr'


@pytest.mark.skipif(not ZARR.exists(), reason='no collected data')
def test_matches_recorded_data():
    """Dataset windows: rolling actions[t .. t+14] from agent_pos[t] lands on agent_pos[t+15]."""
    import zarr
    maze = np.load(REPO / 'data/l2s_maze/layout/maze.npy')
    root = zarr.open(str(ZARR), mode='r')
    ends = np.asarray(root['meta/episode_ends'])
    starts_idx = np.concatenate([[0], ends[:-1]])
    pos = np.asarray(root['data/obs/agent_pos'], dtype=np.float64)
    act = np.asarray(root['data/actions'], dtype=np.float64)
    rng = np.random.default_rng(4)
    idx = []
    for s, e in zip(starts_idx, ends):
        if e - s > T + 1:
            idx.extend(rng.integers(s, e - T, size=4))
    idx = np.asarray(idx)
    windows = act[idx[:, None] + np.arange(T)]
    got = MazeVerifier(maze=maze).rollout(torch.from_numpy(pos[idx]),
                                          torch.from_numpy(windows))[:, -1].numpy()
    # recorded positions are float32
    np.testing.assert_allclose(got, pos[idx + T], atol=1e-5)
