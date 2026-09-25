"""Fixed L2S maze layout and goal splits.

A 17x17 procgen-style maze centred in a 25x25 wall canvas (procgen-hard world_dim=25).
`maze[i, j] == 1` is wall. Positions are grid (row, col).

Splits constrain GOALS only; starts are uniform over free cells.
  as_is       train and test goals: all free cells
  inner_outer train: Chebyshev dist from CENTER > r; test: <= r
  quadrant    train: TL, TR, BL; test: BR. Goals on the centre row/col are unused.
"""
import hashlib
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'maze_data_scripts'))

from procgen_maze_env import bfs_distances, generate_maze  # noqa: E402
from procgen_maze_env import free_cells as _free_cells  # noqa: E402

MAZE_SIZE = 17
CANVAS_SIZE = 25
PAD = (CANVAS_SIZE - MAZE_SIZE) // 2
CENTER = (CANVAS_SIZE // 2, CANVAS_SIZE // 2)
SPLITS = ('as_is', 'inner_outer', 'quadrant')
TRAIN_QUADRANTS = ('TL', 'TR', 'BL')
TEST_QUADRANTS = ('BR',)
DEFAULT_LAYOUT_DIR = REPO / 'data/l2s_maze/layout'


def build_layout(seed=0):
    """(25, 25) uint8, 1 = wall: the carved 17x17 maze inside a wall pad."""
    maze = generate_maze(np.random.default_rng(seed), grid_size=MAZE_SIZE)
    canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    canvas[PAD:PAD + MAZE_SIZE, PAD:PAD + MAZE_SIZE] = maze
    return canvas


def free_cells(maze):
    """Free (row, col) cells as plain-int tuples (JSON-serialisable)."""
    return [(int(i), int(j)) for i, j in _free_cells(maze)]


def layout_sha(maze):
    return hashlib.sha256(np.ascontiguousarray(maze, dtype=np.uint8).tobytes()).hexdigest()


def chebyshev(cell):
    return max(abs(cell[0] - CENTER[0]), abs(cell[1] - CENTER[1]))


def inner_radius(maze, target_frac=0.25):
    """Radius whose inner fraction of free cells is closest to `target_frac` (ties: smaller)."""
    d = np.array([chebyshev(c) for c in free_cells(maze)])
    radii = np.arange(d.max() + 1)
    fracs = np.array([(d <= r).mean() for r in radii])
    return int(radii[np.argmin(np.abs(fracs - target_frac))])


def quadrant(cell):
    """'TL' | 'TR' | 'BL' | 'BR', or None on the centre row/col."""
    di, dj = cell[0] - CENTER[0], cell[1] - CENTER[1]
    if di == 0 or dj == 0:
        return None
    return ('T' if di < 0 else 'B') + ('L' if dj < 0 else 'R')


def goal_pool(maze, split, role, r_inner=None):
    """Free cells eligible as goals for `role` in {'train', 'test'} under `split`."""
    assert split in SPLITS, split
    assert role in ('train', 'test'), role
    cells = free_cells(maze)
    if split == 'as_is':
        return cells
    if split == 'inner_outer':
        assert r_inner is not None
        inner = role == 'test'
        return [c for c in cells if (chebyshev(c) <= r_inner) == inner]
    quads = TRAIN_QUADRANTS if role == 'train' else TEST_QUADRANTS
    return [c for c in cells if quadrant(c) in quads]


def sample_pair(rng, free, goals):
    """(start, goal): goal uniform over `goals`, start uniform over `free` minus the goal."""
    goal = goals[int(rng.integers(len(goals)))]
    while True:
        start = free[int(rng.integers(len(free)))]
        if start != goal:
            return start, goal


def load_layout(layout_dir=DEFAULT_LAYOUT_DIR):
    """(maze, meta) as written by build_layout.py; checks the stored sha."""
    layout_dir = pathlib.Path(layout_dir)
    maze = np.load(layout_dir / 'maze.npy')
    meta = json.loads((layout_dir / 'layout.json').read_text())
    assert layout_sha(maze) == meta['sha'], 'maze.npy does not match layout.json'
    return maze, meta


def bfs_to(maze, goal):
    """{cell: BFS steps to `goal`} (the maze is undirected)."""
    return bfs_distances(maze, goal)
