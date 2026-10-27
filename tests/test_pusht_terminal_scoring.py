"""Terminal semantics shared by the simulated verifier and candidate ranking."""
import copy
import unittest
from types import SimpleNamespace

import gym
import numpy as np
import torch

from diffusion_policy.common.selection_util import select_candidate
from diffusion_policy.env.pusht.feedback_util import compute_feedback_from_pose
from diffusion_policy.env.pusht.pusht_verifier import (
    _DillEnv, PushTVerifier, PushTWaypointVerifier, TERMINAL_SUCCESS_SCORE,
    CROSS_CANDIDATE_VALUES,
    WAYPOINT_CONTEXT_SCALE,
)
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.policy.pusht_search_mixin import PushTSearchMixin


class ScriptedEnv(gym.Env):
    """Would move away again if stepped after its second-step success."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), dtype=np.float64)
    action_space = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float64)

    def reset(self):
        self.steps = 0
        self.state = self.reset_to_state.copy()
        return self.state.copy()

    def step(self, action):
        self.steps += 1
        self.state[0] = self.steps
        self.state[2] += 1
        done = action[0] > 0 and self.steps == 2
        return self.state.copy(), float(done), done, {}

    def _render_frame(self, mode):
        return np.full((8, 8, 3), self.steps, dtype=np.uint8)


class CountingTracker:
    def __init__(self):
        self.score = 0.0
        self.pusher = SimpleNamespace(
            done=False, i=0, waypoints=np.zeros((1, 1, 2)), tol_px=np.ones(1))
        self.t = None

    def copy(self):
        return copy.copy(self)

    def update(self, agent, block):
        self.score += 0.1


class PolicyHarness(PushTSearchMixin):
    consumes_search_context = False

    def __init__(self, mode):
        self.search_kwargs = {'verifier_value': mode, 'search_context': 'value'}

    def _verifier_inputs(self, obs, action):
        return obs, action

    def _normalize_value(self, state):
        return state[:, 0] / 10


class TerminalScoringTest(unittest.TestCase):
    def make_inputs(self):
        pose = np.tile([256., 260., 0.2], (3, 1))
        obs = {
            'agent_pos': torch.zeros(3, 1, 2),
            'feedback': torch.tensor(compute_feedback_from_pose(pose)[:, None]).float(),
        }
        actions = torch.ones(3, 4, 2)
        actions[1] = -1
        return obs, actions

    def make_verifier(self, mode='t_goal', waypoint=False, terminal_scoring=True):
        cls = PushTWaypointVerifier if waypoint else PushTVerifier
        verifier = cls(n_envs=2, use_async=False, render_size=8, value_fn=mode,
                       terminal_scoring=terminal_scoring)
        verifier._vec = SyncVectorEnv([
            lambda: _DillEnv(ScriptedEnv(), terminal_scoring=terminal_scoring)] * 2)
        self.addCleanup(verifier.close)
        if waypoint:
            verifier.set_trackers([CountingTracker() for _ in range(3)])
        return verifier

    def test_mixed_batch_padding_terminal_state_image_and_reset(self):
        verifier = self.make_verifier()
        obs, actions = self.make_inputs()
        value, state, image, success = verifier.rollout(
            obs, actions, render=True, return_success=True)
        self.assertEqual(success.tolist(), [True, False, True])
        self.assertEqual(state[:, 0].tolist(), [2., 4., 2.])
        torch.testing.assert_close(image[:, 0, 0, 0], torch.tensor([2., 4., 2.]) / 255)
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue((value <= 0).all())
        score = verifier.get_value(obs, actions)
        self.assertEqual(score[[0, 2]].tolist(), [TERMINAL_SUCCESS_SCORE] * 2)
        torch.testing.assert_close(score[1], value[1])
        _, state, success = verifier.rollout(obs, -torch.ones_like(actions), return_success=True)
        self.assertFalse(success.any())
        self.assertEqual(state[:, 0].tolist(), [4., 4., 4.])

    def test_waypoint_tracker_stops_at_terminal_step(self):
        verifier = self.make_verifier('wp_v3', waypoint=True)
        value, _ = verifier.rollout(*self.make_inputs())
        torch.testing.assert_close(value, torch.tensor([.2, .4, .2]))
        self.assertEqual([t.score for t in verifier._trackers], [0., 0., 0.])

    def test_legacy_mode_keeps_full_chunk_and_original_scores(self):
        self.assertFalse(PushTVerifier().terminal_scoring)
        for mode, waypoint in [('t_goal', False), ('wp_v3', True), ('wp_v3_la_tgn', True)]:
            with self.subTest(mode=mode):
                verifier = self.make_verifier(mode, waypoint, terminal_scoring=False)
                value, state, image, success = verifier.rollout(
                    *self.make_inputs(), render=True, return_success=True)
                self.assertEqual(state[:, 0].tolist(), [4., 4., 4.])
                self.assertFalse(success.any())
                torch.testing.assert_close(image[:, 0, 0, 0], torch.full((3,), 4 / 255))
                torch.testing.assert_close(verifier.get_value(*self.make_inputs()), value)
                if mode == 'wp_v3':
                    torch.testing.assert_close(value, torch.full((3,), .4))
                _, score, _, _ = PolicyHarness(mode)._score_candidates(
                    verifier, *self.make_inputs())
                torch.testing.assert_close(score, value)

    def test_changing_version_rebuilds_pool(self):
        verifier = self.make_verifier(terminal_scoring=False)
        verifier.set_terminal_scoring(True)
        self.assertIsNone(verifier._vec)
        self.assertTrue(verifier.terminal_scoring)

    def test_selection_priority_does_not_enter_model_context(self):
        for mode, waypoint in [('t_goal', False), ('wp_v3', True),
                               ('wp_v3_la_tgn', True)]:
            with self.subTest(mode=mode):
                verifier = self.make_verifier(mode, waypoint)
                context, score, subgoal, terms = PolicyHarness(mode)._score_candidates(
                    verifier, *self.make_inputs(), want_subgoals=True)
                expected = (subgoal['value'] / WAYPOINT_CONTEXT_SCALE.get(mode, 1.)
                            if waypoint else torch.tensor([.2, .4, .2]))
                torch.testing.assert_close(context, expected)
                self.assertEqual(score[[0, 2]].tolist(), [TERMINAL_SUCCESS_SCORE] * 2)
                self.assertTrue(torch.isfinite(terms).all())
                self.assertTrue((subgoal['value'] < TERMINAL_SUCCESS_SCORE).all())

    def test_fusion_preserves_success_and_nonterminal_scores(self):
        for mode in ('armTd',):
            with self.subTest(mode=mode):
                terms = torch.tensor([[[100., 100., 100.], [0., 0., 0.], [1., 1., 1.]]])
                if mode == 'armTd':
                    terms = terms[..., :2]
                baseline = CROSS_CANDIDATE_VALUES[mode](terms)
                policy = PolicyHarness(mode)
                torch.testing.assert_close(policy._fuse_scores(-torch.ones(1, 3), terms), baseline)
                scores = policy._fuse_scores(torch.tensor([[2., -1., -1.]]), terms)
                self.assertEqual(scores.argmax(1).item(), 0)
                torch.testing.assert_close(scores[:, 1:], baseline[:, 1:])
                self.assertTrue(torch.isfinite(scores).all())
                actions = torch.arange(3.).reshape(1, 3, 1, 1)
                self.assertEqual(select_candidate(actions, scores).item(), 0)
                self.assertTrue(torch.isfinite(select_candidate(actions, scores, mode='softmax')).all())


if __name__ == '__main__':
    unittest.main()
