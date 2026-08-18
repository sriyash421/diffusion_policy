"""Feedback-based verifier for the offline PushT diffusion-search policy.

The maze diffusion-search policy scores each candidate action with an external
``l2s.verifier.MazeVerifier`` (analytic, maze-only). PushT has no such analytic map from
(state, action-sequence) to outcome, so this verifier **simulates** the candidate in a
deterministic PushT sim and scores the result with the *same feedback signal the online
search policy uses*: ``feedback_util.compute_feedback_from_pose`` (goal-vs-achieved T
keypoint displacement).

``get_value(obs_dict, action) -> (B,)`` returns ``value = -mean_kp ||goal_kp - achieved_kp||``
(negative mean per-keypoint distance, in pixels) at the block pose reached after executing
the candidate action sequence from the obs state. It is ``0`` iff the block ends at the goal
and decreases the further the T is from the goal -- i.e. higher value == closer to goal.

``rollout(obs_dict, action, render=False) -> (value (B,), state (B, STATE_DIM)[, image])``
returns that same scalar *plus* the sim state reached at the end of the candidate chunk --
the "subgoal" the candidate lands on -- laid out as ``[agent_pos (2), feedback (16)]``, and
with ``render=True`` also the rendered subgoal observation ``(B, 3, 96, 96)``. All of it
comes from the same simulation, so the extra signals cost no extra sim steps (``render``
adds one render per chunk, not per step).

The reached state also backs the search-context modes of ``PushTDiffusionSearchPolicy``:
``value`` rescales the scalar from it, and ``subgoal``/``subgoal_value`` pair it with the
rendered frame and embed the whole observation through the policy's own obs encoder.
"""
from typing import Dict

import numpy as np
import torch

import gym
import dill

from diffusion_policy.env.pusht.pusht_env import PushTEnv
from diffusion_policy.env.pusht.feedback_util import (
    compute_feedback_from_pose, block_pose_from_feedback, keypoints_at_pose,
    T_VERTS, GOAL_POSE, N_KEYPOINTS)
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv, force_close
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv


class _DillEnv(gym.Wrapper):
    """Adds run_dill_function/get_attr so the vector env can set per-env reset states."""

    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        return dill.loads(dill_fn)(self)

    def render_obs(self):
        """The current state rendered as an (H, W, 3) uint8 frame.

        Must be a real method, not inherited via gym.Wrapper.__getattr__: the vector env
        worker resolves remote calls with ``getattr(env, name)`` on the wrapper.
        """
        return self.env._render_frame(mode='rgb_array')


def _make_verifier_env(legacy, render_size):
    """Picklable thunk building a PushTEnv for the verifier pool.

    ``render_action=False`` is REQUIRED, not cosmetic: PushTEnv defaults it to True, which
    draws a red action cross into every rendered frame. The subgoal frames are fed to the
    policy's obs encoder, so they must match ``PushTImageEnv._get_obs`` -- the pipeline that
    produced the dataset's ``img`` array and the eval-time obs -- or the encoder sees
    out-of-distribution images. Stepping never renders; only ``render_obs`` does.
    """
    def _fn():
        return _DillEnv(PushTEnv(
            legacy=legacy, render_action=False, render_size=render_size))
    return _fn


class PushTVerifier:
    # layout of the rollout state returned by `rollout`: [agent_pos (2), feedback (16)].
    # agent_pos/feedback are exactly the two low_dim policy obs keys, so the state is
    # normalizable with the policy's own normalizer (see PushTDiffusionSearchPolicy).
    AGENT_DIM = 2
    STATE_DIM = AGENT_DIM + 2 * N_KEYPOINTS   # 18

    def __init__(self, n_envs: int = 32, legacy: bool = False, use_async: bool = True,
                 verifier_steps: int = None, render_size: int = 96, **kwargs):
        """
        Args:
            n_envs: size of the persistent vectorized PushT sim pool; candidates are
                scored in chunks of this size (the envs step in parallel).
            legacy: must be False so ``reset_to_state`` round-trips a recorded state
                exactly (see runner/eval_bon notes); kept configurable for parity.
            use_async: run the pool with AsyncVectorEnv (parallel worker processes).
                Set False for a single-process SyncVectorEnv (debugging).
            verifier_steps: optional cap on how many action steps to simulate
                (None = all of what is passed in). The caller already slices the action
                down to the executed window, so this should normally stay None -- a cap
                shorter than that window silently truncates the evaluated chunk.
            render_size: edge length of the subgoal frames from ``rollout(render=True)``.
                Must match the policy's image shape_meta (96) or the encoder rejects them.
        """
        self.n_envs = n_envs
        self.legacy = legacy
        self.use_async = use_async
        self.verifier_steps = verifier_steps
        self.render_size = render_size
        # Lazily built vectorized pool, reused across get_value calls: the workers are
        # only forked on the first rollout (i.e. inside the first compute_loss), which
        # is why _get_vec uses a forkserver context -- by then the parent has CUDA
        # initialized, and forking a CUDA-initialized process can deadlock.
        self._vec = None

    def _get_vec(self):
        if self._vec is None:
            env_fns = [_make_verifier_env(self.legacy, self.render_size)
                       for _ in range(self.n_envs)]
            if self.use_async:
                # forkserver: workers are forked from a clean server process, NOT from the
                # (CUDA-initialized) training process -- fork-after-CUDA can deadlock. The
                # server imports torch once, so this is far cheaper than 'spawn'.
                self._vec = AsyncVectorEnv(env_fns, context='forkserver')
            else:
                self._vec = SyncVectorEnv(env_fns)
        return self._vec

    def _set_reset_states(self, vec, states):
        def _make_setter(state):
            state = np.asarray(state, dtype=np.float64)
            def _fn(env):
                env.unwrapped.reset_to_state = state
            return _fn
        vec.call_each('run_dill_function',
                      args_list=[(dill.dumps(_make_setter(s)),) for s in states])

    @staticmethod
    def _reset_states_from_obs(obs_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """(B, 5) env reset state [agent_x, agent_y, block_x, block_y, angle].

        Uses the last obs step. CALLER CONTRACT: the obs window handed in must END at
        the state the candidate action is taken from -- normally a single step, sliced
        by ``PushTDiffusionSearchPolicy._verifier_inputs``. Do NOT pass a raw training
        batch: those carry the full ``horizon`` (16) steps, so ``[:, -1]`` would be the
        state ~14 control steps ahead of the one the policy conditioned on.

        The block pose is reconstructed from ``feedback`` (exactly -- feedback is an
        invertible function of the pose, see ``block_pose_from_feedback``). This is the
        single path for both training and eval, so the two produce identical resets;
        ``feedback`` is a declared obs key, so no privileged reset carrier is needed.
        """
        agent_last = obs_dict['agent_pos'][:, -1].detach().cpu().numpy().astype(np.float64)
        feedback_last = obs_dict['feedback'][:, -1].detach().cpu().numpy()
        block_last = block_pose_from_feedback(feedback_last)  # (B, 3)
        return np.concatenate([agent_last, block_last], axis=-1)  # (B, 5)

    @torch.no_grad()
    def get_value(self, obs_dict: Dict[str, torch.Tensor],
                  action: torch.Tensor) -> torch.Tensor:
        """Scalar score of a candidate action sequence per batch element.

        Thin wrapper over ``rollout`` keeping the original scalar-only interface (the
        interface the maze ``MazeVerifier`` also exposes).
        """
        return self.rollout(obs_dict, action)[0]

    @torch.no_grad()
    def rollout(self, obs_dict: Dict[str, torch.Tensor],
                action: torch.Tensor, render: bool = False):
        """Simulate a candidate action sequence per batch element.

        Each candidate is simulated in a deterministic PushT sim (reset to the obs state)
        by stepping its action sequence; the value is ``-mean per-keypoint distance`` of
        the achieved T from the goal T, and the state is the sim state reached at the end
        of the chunk. The pool of ``n_envs`` sims steps in parallel, so a batch of
        candidates is simulated in ceil(B / n_envs) parallel rounds. The env's own obs
        already carries agent pos (obs[:, :2]) and the block pose (obs[:, 2:5]), so the
        low-dim outputs need no render.

        Args:
            obs_dict: must contain ``agent_pos`` (B, To, 2) and ``feedback`` (B, To, 16),
                from which the block pose is reconstructed. Same at train and eval time.
            action: (B, horizon, 2) UNNORMALIZED agent-target positions (pixel coords).
            render: also return the subgoal frame. Costs ONE render per chunk (not per
                step), taken after the last step; leave False when unused.
        Returns:
            ``(value, state)``, or ``(value, state, image)`` when ``render``:
            ``value`` (B,); ``state`` (B, STATE_DIM) laid out as
            ``[agent_pos (2), feedback (16)]``; ``image`` (B, 3, render_size,
            render_size) float32 in [0, 1], channel-first -- byte-for-byte the obs
            ``PushTImageEnv`` would emit at that state. All on ``action``'s device/dtype.
        """
        states = self._reset_states_from_obs(obs_dict)          # (B, 5)
        actions = action.detach().cpu().numpy().astype(np.float64)  # (B, H, 2)
        B, H, _ = actions.shape
        n_steps = H if self.verifier_steps is None else min(H, self.verifier_steps)

        vec = self._get_vec()
        values = np.empty(B, dtype=np.float32)
        end_states = np.empty((B, self.STATE_DIM), dtype=np.float32)
        images = np.empty(
            (B, 3, self.render_size, self.render_size), dtype=np.float32) \
            if render else None
        for start in range(0, B, self.n_envs):
            end = min(start + self.n_envs, B)
            m = end - start
            # pad the final partial chunk up to n_envs; padded results are ignored
            chunk_states = states[start:end]
            chunk_actions = actions[start:end]
            if m < self.n_envs:
                pad = self.n_envs - m
                chunk_states = np.concatenate(
                    [chunk_states, np.repeat(chunk_states[-1:], pad, axis=0)], axis=0)
                chunk_actions = np.concatenate(
                    [chunk_actions, np.repeat(chunk_actions[-1:], pad, axis=0)], axis=0)

            self._set_reset_states(vec, list(chunk_states))
            obs = vec.reset()                                   # (n_envs, 5)
            for t in range(n_steps):
                obs, _, _, _ = vec.step(chunk_actions[:, t])    # obs (n_envs, 5)
            obs = np.asarray(obs)
            agent_pos = obs[:, :2]                              # (n_envs, 2)
            block_pose = obs[:, 2:5]                            # (n_envs, 3): x, y, angle
            feedback = compute_feedback_from_pose(block_pose)   # (n_envs, 16)
            dist = np.linalg.norm(
                feedback.reshape(-1, N_KEYPOINTS, 2), axis=-1).mean(axis=-1)  # (n_envs,)
            values[start:end] = -dist[:m].astype(np.float32)
            end_states[start:end] = np.concatenate(
                [agent_pos[:m], feedback[:m]], axis=-1).astype(np.float32)

            if render:
                # one render per chunk, at the state the action chunk landed on. Same
                # transform as PushTImageEnv._get_obs: HWC uint8 -> CHW float32 in [0,1].
                frames = np.asarray(vec.call('render_obs'), dtype=np.float32)  # (n,H,W,3)
                images[start:end] = np.moveaxis(frames[:m] / 255.0, -1, 1)

        out = (
            torch.as_tensor(values, dtype=action.dtype, device=action.device),
            torch.as_tensor(end_states, dtype=action.dtype, device=action.device),
        )
        if render:
            out = out + (
                torch.as_tensor(images, dtype=action.dtype, device=action.device),)
        return out

    def close(self):
        # force_close, not _vec.close(): a plain close() first tries to drain whatever call
        # is in flight, with no timeout, so a worker that died mid-reply hangs teardown
        # forever. That happened -- a training run sat wedged inside this call for 13.5h
        # holding a GPU, with SLURM still reporting it RUNNING and the original traceback
        # swallowed by the stuck unwind. See force_close for the full account.
        if self._vec is not None:
            vec, self._vec = self._vec, None
            force_close(vec)
