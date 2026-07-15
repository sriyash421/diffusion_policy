"""Feedback wrapper + parallel-env helpers for the online search policy.

``PushTFeedbackWrapper`` injects the per-step ``feedback`` signal (see
``feedback_util.compute_feedback_from_pose``) into the obs, computed live from the
env's pymunk block body. The same env-free util lets the dataset produce an identical
signal from the stored ``block_pos``.
"""
import numpy as np
import gym
from gym import spaces

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.env.pusht.feedback_util import (
    compute_feedback_from_pose, GOAL_POSE, FEEDBACK_DIM, N_KEYPOINTS)  # noqa: F401
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv


class PushTFeedbackWrapper(gym.ObservationWrapper):
    """Adds ``obs['feedback']`` (goal-vs-achieved T keypoint displacement)."""

    def __init__(self, env, goal_pose=None):
        super().__init__(env)
        self._goal_pose = goal_pose
        obs_spaces = dict(self.env.observation_space.spaces)
        obs_spaces['feedback'] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(FEEDBACK_DIM,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)

    def observation(self, obs):
        base = self.unwrapped
        block_pose = np.array([
            base.block.position[0],
            base.block.position[1],
            base.block.angle % (2 * np.pi),
        ], dtype=np.float32)
        goal = self._goal_pose if self._goal_pose is not None else base.goal_pose
        obs = dict(obs)
        obs['feedback'] = compute_feedback_from_pose(block_pose, goal_pose=goal)
        return obs


def make_env_fn(n_obs_steps=1, n_action_steps=1, max_episode_steps=300,
                legacy=True, render_size=96):
    """Thunk building ``MultiStepWrapper(PushTFeedbackWrapper(PushTImageEnv))``."""
    def _fn():
        env = PushTImageEnv(legacy=legacy, render_size=render_size)
        env = PushTFeedbackWrapper(env)
        env = MultiStepWrapper(
            env,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps)
        return env
    return _fn


def build_parallel(n_envs=32, use_async=True, **env_kwargs):
    """Vectorized PushT envs with feedback, reusing the repo's vector envs."""
    env_fns = [make_env_fn(**env_kwargs) for _ in range(n_envs)]
    if use_async:
        return AsyncVectorEnv(env_fns)
    return SyncVectorEnv(env_fns)


def set_reset_states(vec_env, states):
    """Set per-env ``reset_to_state`` (called before ``vec_env.reset()``).

    Args:
        vec_env: an Async/SyncVectorEnv of MultiStepWrapper-wrapped PushT envs.
        states: list of 5-D ``[agent_x, agent_y, block_x, block_y, block_angle]``.
    """
    import dill

    def _make_setter(state):
        state = np.asarray(state, dtype=np.float64)
        def _fn(env):
            env.unwrapped.reset_to_state = state
        return _fn

    args_list = [(dill.dumps(_make_setter(s)),) for s in states]
    vec_env.call_each('run_dill_function', args_list=args_list)
