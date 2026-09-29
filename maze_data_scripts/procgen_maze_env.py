"""Procedurally generated grid maze matching `config/task/procgen_maze.yaml`.

The task config declares an [3, 11, 11] rgb observation, `agent_pos` [2], `goal_pos` [2]
and a continuous [2] action. No env in the repo produced it -- the generator lived in the
external `l2s` package -- so the conventions are defined here:

* **Layout.** An 11x11 grid is a perfect maze over 5x5 cells (grid = 2*cells + 1), carved by
  a recursive backtracker. `maze_map[i, j] == 1` is wall, `0` is free, matching the
  `maze_map` convention `QIteration` in `procgen_maze_expert` reads.
* **Position.** Continuous `(row, col)` in grid coordinates, so an action chunk is smooth.
  `xy` and `ij` are therefore the same space, and `xy_to_ij` is a round -- the expert
  written against gymnasium-robotics' maze API works unchanged.
* **Action.** A displacement clipped to [-1, 1] and scaled by `step_scale`, following
  OGBench's PointEnv (`action = 0.2 * action`). Blocked axes slide rather than stick.
* **Image.** An actual RGB picture, one pixel per grid cell. Agent and goal use the
  Okabe-Ito blue/orange rather than red/green so a rendered maze stays readable to a
  colourblind reader.
"""
from collections import deque

import numpy as np

# Okabe-Ito: distinguishable under every common form of colour blindness.
COLOR_WALL = (0, 0, 0)
COLOR_FREE = (255, 255, 255)
COLOR_AGENT = (0, 114, 178)    # blue
COLOR_GOAL = (230, 159, 0)     # orange

# 23x23 holds 241 free cells, so 200 distinct goals can be drawn for a single fixed
# start. The 11x11 it started at holds only 49. `config/task/procgen_maze.yaml`'s
# `image_shape` must match whatever this is.
GRID_SIZE = 23
N_CELLS = (GRID_SIZE - 1) // 2  # 11x11 cells carved into a 23x23 grid


def generate_maze(rng, grid_size=GRID_SIZE):
    """Recursive-backtracker perfect maze. Returns (grid_size, grid_size) uint8, 1 = wall."""
    n_cells = (grid_size - 1) // 2
    maze = np.ones((grid_size, grid_size), dtype=np.uint8)

    start = (int(rng.integers(n_cells)), int(rng.integers(n_cells)))
    visited = {start}
    stack = [start]
    maze[2 * start[0] + 1, 2 * start[1] + 1] = 0

    while stack:
        ci, cj = stack[-1]
        neighbours = []
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < n_cells and 0 <= nj < n_cells and (ni, nj) not in visited:
                neighbours.append((ni, nj))
        if not neighbours:
            stack.pop()
            continue
        ni, nj = neighbours[int(rng.integers(len(neighbours)))]
        # Knock out the wall between the two cells.
        maze[ci + ni + 1, cj + nj + 1] = 0
        maze[2 * ni + 1, 2 * nj + 1] = 0
        visited.add((ni, nj))
        stack.append((ni, nj))

    return maze


def maze_id(maze_map):
    """Stable identity of a layout, so train can exclude the held-out eval mazes."""
    return np.packbits(maze_map.astype(bool).ravel()).tobytes().hex()


def free_cells(maze_map):
    """The (row, col) grid coordinates an agent or goal may occupy."""
    return [tuple(c) for c in np.argwhere(maze_map == 0)]


def bfs_distances(maze_map, start_ij):
    """Steps along the maze from `start_ij` to every reachable free cell, as {(i,j): d}.

    Distance along corridors, not Euclidean: it is what decides whether a goal is actually
    far away rather than merely on the other side of a wall.
    """
    start = tuple(int(x) for x in start_ij)
    dist = {start: 0}
    queue = deque([start])
    h, w = maze_map.shape
    while queue:
        i, j = queue.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (i + di, j + dj)
            if nb not in dist and 0 <= nb[0] < h and 0 <= nb[1] < w and maze_map[nb] == 0:
                dist[nb] = dist[(i, j)] + 1
                queue.append(nb)
    return dist


def make_maze_split(n_mazes, seed, exclude_ids=frozenset(), grid_size=GRID_SIZE):
    """`n_mazes` distinct layouts, none of them in `exclude_ids`.

    Deterministic in `seed`: the eval split is reproducible from its seed alone, and the
    train split is reproducible from its seed plus the eval ids it was told to avoid.
    """
    rng = np.random.default_rng(seed)
    mazes, seen = [], set(exclude_ids)
    while len(mazes) < n_mazes:
        maze = generate_maze(rng, grid_size=grid_size)
        mid = maze_id(maze)
        if mid in seen:
            continue
        seen.add(mid)
        mazes.append(maze)
    return mazes


class ProcgenMazeEnv:
    """Minimal env exposing the attributes `WaypointController` needs.

    `maze_map`, `cur_goal_xy`, `xy_to_ij` and `ij_to_xy` mirror the gymnasium-robotics maze
    API the given expert was written against, so that expert needs no changes.
    """

    def __init__(self, mazes, step_scale=0.25, goal_tol=0.3, seed=0):
        self.mazes = list(mazes)
        self.step_scale = step_scale
        self.goal_tol = goal_tol
        self.rng = np.random.default_rng(seed)
        self.maze_map = None
        self.cur_goal_xy = None
        self.agent_xy = None
        self.maze_index = None

    # -- geometry -----------------------------------------------------------------
    def xy_to_ij(self, xy):
        return np.rint(np.asarray(xy, dtype=np.float64)).astype(int)

    def ij_to_xy(self, ij):
        return np.asarray(ij, dtype=np.float64)

    def _is_free(self, ij):
        i, j = int(ij[0]), int(ij[1])
        if not (0 <= i < self.maze_map.shape[0] and 0 <= j < self.maze_map.shape[1]):
            return False
        return self.maze_map[i, j] == 0

    # -- rollout ------------------------------------------------------------------
    def reset(self, maze_index=None, start_ij=None, goal_ij=None):
        """`start_ij` / `goal_ij` pin an endpoint; whatever is left as None is sampled."""
        self.maze_index = (
            int(self.rng.integers(len(self.mazes))) if maze_index is None else maze_index)
        self.maze_map = self.mazes[self.maze_index]

        if start_ij is None or goal_ij is None:
            cells = free_cells(self.maze_map)
            start_idx, goal_idx = self.rng.choice(len(cells), size=2, replace=False)
        self.agent_xy = np.asarray(
            cells[start_idx] if start_ij is None else start_ij, dtype=np.float64)
        self.cur_goal_xy = np.asarray(
            cells[goal_idx] if goal_ij is None else goal_ij, dtype=np.float64)
        return self.get_obs()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        delta = self.step_scale * action

        # Slide along whichever axes are not blocked, rather than refusing the whole move.
        proposed = self.agent_xy + delta
        if not self._is_free(self.xy_to_ij(proposed)):
            proposed = self.agent_xy + np.array([delta[0], 0.0])
            if not self._is_free(self.xy_to_ij(proposed)):
                proposed = self.agent_xy + np.array([0.0, delta[1]])
                if not self._is_free(self.xy_to_ij(proposed)):
                    proposed = self.agent_xy
        self.agent_xy = proposed

        success = bool(np.linalg.norm(self.agent_xy - self.cur_goal_xy) <= self.goal_tol)
        # Reward matches OGBench's singletask convention: -1 failure, 0 success.
        return self.get_obs(), (0.0 if success else -1.0), success

    # -- observation --------------------------------------------------------------
    def render(self):
        """(11, 11, 3) uint8 RGB. HWC: the dataset moves the channel axis itself."""
        img = np.empty((*self.maze_map.shape, 3), dtype=np.uint8)
        img[:] = COLOR_FREE
        img[self.maze_map == 1] = COLOR_WALL
        gi, gj = self.xy_to_ij(self.cur_goal_xy)
        img[gi, gj] = COLOR_GOAL
        ai, aj = self.xy_to_ij(self.agent_xy)
        img[ai, aj] = COLOR_AGENT
        return img

    def get_obs(self):
        return {
            'image': self.render(),
            'agent_pos': self.agent_xy.copy(),
            'goal_pos': self.cur_goal_xy.copy(),
        }
