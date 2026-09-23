"""Waypoint expert for the procgen grid maze.

`QIteration` and `WaypointController` are REUSED from `pointmaze_expert` rather than
forked: both are written against `maze_map` plus `xy_to_ij` / `ij_to_xy` / `cur_goal_xy`,
which `ProcgenMazeEnv` deliberately provides. The only thing this task adds is that the
layout AND the goal change every episode, so the solver has to be rebuilt per reset --
`QIteration` caches `q_values` and short-circuits `compute_reward_matrix` once
`rew_matrix` is non-zero, so one controller can only ever solve one (maze, goal) pair.
"""
import numpy as np

from pointmaze_expert import QIteration, WaypointController  # noqa: F401  (re-exported)


def episode_seed(env):
    """A seed fixed by WHICH episode this is, not by how many steps have been taken.

    Deriving it from the env's own rng would work but would consume that stream, making
    the next reset's start/goal depend on rollout length. The layout and the endpoints
    already identify the episode, so hash those instead.
    """
    key = (env.maze_index, *np.rint(env.agent_xy).astype(int),
           *np.rint(env.cur_goal_xy).astype(int))
    return abs(hash(key)) % (2 ** 32)


class NoDither:
    """Stands in for the expert's rng so waypoints are the exact cell centres."""

    def uniform(self, size=None):
        return np.zeros(size)


class ProcgenMazeExpert:
    """Per-episode waypoint controller. Rebuild it after every `env.reset()`.

    Dither is OFF by default: demonstrations run down the middle of each corridor. Pass
    `dither=True` to restore `WaypointController`'s 0.2-cell waypoint jitter, which costs
    1-4 steps an episode and is the only source of action diversity in the demos.
    """

    # p=4 with the env's step_scale=0.25 puts the loop gain at 1.0. The shipped p=10 gives
    # 2.5, and a discrete P loop above 2 overshoots: it turned the 0.2-cell waypoint dither
    # into a 0.43-cell ring, since the d term in `WaypointController` is commented out and
    # nothing damps it.
    DEFAULT_GAINS = {"p": 4.0, "d": -1.0}

    def __init__(self, env, waypoint_threshold=0.3, gains=None, rng=None, dither=False):
        self.env = env
        if rng is None:
            rng = np.random.default_rng(episode_seed(env)) if dither else NoDither()
        self.controller = WaypointController(
            env,
            gains=gains if gains is not None else dict(self.DEFAULT_GAINS),
            waypoint_threshold=waypoint_threshold,
            rng=rng,
        )

    def compute_action(self, obs):
        """obs dict from ProcgenMazeEnv -> action in [-1, 1]^2."""
        return self.controller.compute_action({
            "achieved_goal": np.asarray(obs["agent_pos"], dtype=np.float64),
            "desired_goal": np.asarray(obs["goal_pos"], dtype=np.float64),
        })


def solve_length(env, max_steps=300, waypoint_threshold=0.3, dither=False):
    """Steps the expert needs from the current reset state, or None if it fails."""
    expert = ProcgenMazeExpert(env, waypoint_threshold=waypoint_threshold, dither=dither)
    obs = env.get_obs()
    for step in range(max_steps):
        obs, _, success = env.step(expert.compute_action(obs))
        if success:
            return step + 1
    return None


if __name__ == "__main__":
    from procgen_maze_env import ProcgenMazeEnv, make_maze_split

    env = ProcgenMazeEnv(make_maze_split(5, seed=0), seed=0)
    for episode in range(5):
        env.reset()
        n = solve_length(env)
        print(f"episode {episode}: maze {env.maze_index} "
              f"{'solved in %d steps' % n if n else 'FAILED'}")
