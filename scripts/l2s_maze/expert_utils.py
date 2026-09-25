"""Expert rollouts on the fixed layout, with Q-iteration cached per goal.

`QIteration` is rebuilt per episode by `ProcgenMazeExpert` and costs ~1s; with one layout it
depends only on the goal, so a solved instance is reused across episodes sharing a goal.
"""
import numpy as np

from maze_layout import REPO  # noqa: F401  (puts maze_data_scripts on sys.path)
from pointmaze_expert import QIteration
from procgen_maze_env import ProcgenMazeEnv
from procgen_maze_expert import ProcgenMazeExpert

STEP_SCALE = 0.25
GOAL_TOL = 0.3


def make_env(maze, seed=0):
    return ProcgenMazeEnv([maze], step_scale=STEP_SCALE, goal_tol=GOAL_TOL, seed=seed)


class CachedExpert:
    """Runs the (dither-free) waypoint expert from a given (start, goal)."""

    def __init__(self, maze):
        self.maze = maze
        self.env = make_env(maze)
        self._solvers = {}

    def _solver(self, goal):
        goal = tuple(int(x) for x in goal)
        if goal not in self._solvers:
            self._solvers[goal] = QIteration(maze=self.env)
        return self._solvers[goal]

    def rollout(self, start, goal, max_steps=1000, record=False):
        """Returns (success, n_steps, episode). `episode` holds per-step obs/actions if `record`."""
        obs = self.env.reset(maze_index=0, start_ij=start, goal_ij=goal)
        expert = ProcgenMazeExpert(self.env)
        expert.controller.maze_solver = self._solver(goal)
        ep = {'image': [], 'agent_pos': [], 'goal_pos': [], 'action': []} if record else None
        for t in range(max_steps):
            action = expert.compute_action(obs)
            if record:
                ep['image'].append(obs['image'])
                ep['agent_pos'].append(obs['agent_pos'])
                ep['goal_pos'].append(obs['goal_pos'])
                ep['action'].append(action)
            obs, _, success = self.env.step(action)
            if success:
                return True, t + 1, ep
        return False, max_steps, ep


def as_cell(x):
    return tuple(int(v) for v in np.rint(np.asarray(x)))
