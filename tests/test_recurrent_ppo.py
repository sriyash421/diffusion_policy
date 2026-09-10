"""Regression tests for recurrent_ppo, replacing the ad-hoc scratchpad scripts.

Each test pins one property the audit established, so a future edit that breaks it fails loudly
rather than silently changing what an arm measures.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch as th

from recurrent_ppo.corrupt_policy import (CorruptingExtractor, features_extractor_kwargs,
                                          make_obs_noise_scheduler, policy_for)
from recurrent_ppo.pusht_gym import WS, PushTGymEnv

GOAL_STATE = np.array([256.0, 200.0, 256.0, 256.0, np.pi / 4])   # block sitting on the goal pose


def _solved_env(reward_mode, steps=8):
    env = PushTGymEnv(obs_type="keypoint", reward_mode=reward_mode, max_episode_steps=steps)
    env.reset(seed=0)
    env.env.reset_to_state = GOAL_STATE.copy()
    env.env.reset()
    env._elapsed_steps, env._solved, env._max_reward = 0, False, 0.0
    return env


def test_dense_reward_never_terminates_on_success():
    """Terminating under dense reward would forfeit a stream worth 1.0/step -- the severe bug."""
    env = _solved_env("dense")
    rewards, terminated = [], False
    for _ in range(8):
        _, r, term, trunc, info = env.step(np.zeros(2))
        rewards.append(r)
        terminated |= term
        if term or trunc:
            break
    assert not terminated
    assert rewards == [1.0] * len(rewards)
    assert info["is_success"] is True          # sticky, even though the episode did not end


def test_sparse_reward_pays_once_then_terminates():
    env = _solved_env("sparse")
    rewards = []
    for _ in range(8):
        _, r, term, _, info = env.step(np.zeros(2))
        rewards.append(r)
        if term:
            break
    assert rewards == [1.0]
    assert info["is_success"] is True


def test_solving_earlier_scores_higher_under_dense():
    """The inverse of the bug: the discounted return must reward speed."""
    gamma, horizon = 0.99, 40
    returns = []
    for solve_at in (0, 20):
        env = _solved_env("dense", steps=horizon)
        away = np.array([256.0, 50.0, 60.0, 60.0, 0.0])
        rs = []
        for i in range(horizon):
            env.env.reset_to_state = (away if i < solve_at else GOAL_STATE).copy()
            env.env.reset()
            env._elapsed_steps = i
            rs.append(env.step(np.zeros(2))[1])
        returns.append(sum(gamma ** i * r for i, r in enumerate(rs)))
    assert returns[0] > returns[1]


@pytest.mark.parametrize("obs_type", ["keypoint", "image"])
def test_observations_stay_in_the_declared_box(obs_type):
    env = PushTGymEnv(obs_type=obs_type, max_episode_steps=40)
    env.reset(seed=0)
    for _ in range(15):
        obs, _ = env.reset()
        for _ in range(40):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            vec = obs if obs_type == "keypoint" else obs["agent_pos"]
            assert vec.min() >= -1.0 and vec.max() <= 1.0
            if term or trunc:
                break
    if obs_type == "image":
        # uint8 on purpose: SB3's extractor does the /255 itself
        assert env.observation_space["image"].dtype == np.uint8


def test_delta_action_is_an_offset_and_absolute_is_a_target():
    env = PushTGymEnv(obs_type="keypoint", action_mode="delta", delta_scale=32.0)
    env.reset(seed=0)
    here = np.array(env.env.agent.position)
    assert np.allclose(env._convert_action(np.zeros(2)), here)           # a = 0 holds position
    assert np.allclose(env._convert_action(np.array([1.0, -1.0])) - here, [32.0, -32.0])
    assert np.all(env._convert_action(np.array([50.0, 50.0])) <= WS)     # clipped into the arena

    absolute = PushTGymEnv(obs_type="keypoint", action_mode="absolute")
    assert np.allclose(absolute._convert_action(np.array([-1.0, 1.0])), [0.0, WS])


@pytest.mark.parametrize("mode,expected_run", [("iid", 2.0), ("persistent", 20.0)])
def test_occlusion_modes_share_a_marginal_and_differ_in_structure(mode, expected_run):
    """The whole point of the pair: same amount missing, different distribution over time."""
    env = PushTGymEnv(obs_type="keypoint", keypoint_visible_rate=0.5, occlusion=mode,
                      occlusion_persistence=20.0, max_episode_steps=2000)
    assert env.env.keypoint_visible_rate == 1.0        # shared repo env left untouched
    env.reset(seed=1)
    obs, _ = env.reset()
    visible, runs, current = [], [], 0
    for _ in range(8000):
        obs, _, term, trunc, _ = env.step(np.zeros(2))
        mask = obs[20:20 + 18:2] > 0
        visible.append(mask.mean())
        if not mask[0]:
            current += 1
        elif current:
            runs.append(current)
            current = 0
        if term or trunc:
            obs, _ = env.reset()
    assert np.mean(visible) == pytest.approx(0.5, abs=0.03)
    assert np.mean(runs) == pytest.approx(expected_run, rel=0.25)


def test_corruption_matches_the_repo_mixin_when_unscaled():
    """A copy kept honest: if ObsCorruptionMixin changes, this fails instead of drifting."""
    from diffusion_policy.model.common.obs_corruption import ObsCorruptionMixin

    class Reference(ObsCorruptionMixin):
        def __init__(self):
            self.obs_noise_scheduler = make_obs_noise_scheduler()
            self.training = True
            self._init_corruption(True, True)

    extractor = CorruptingExtractor(gym.spaces.Box(-1, 1, (64,)), t_max=1000).eval()
    features = th.randn(64, 64)
    th.manual_seed(7)
    noise = th.randn_like(features) * extractor.feature_std       # buffer is still all ones
    timesteps = th.randint(0, extractor.t_max, (features.shape[0],)).long()
    mine = extractor.scheduler.add_noise(features, noise, timesteps)
    th.manual_seed(7)
    assert th.equal(mine, Reference().corrupt_obs_features(features))


def test_scaling_makes_the_snr_independent_of_feature_magnitude():
    """Why a level chosen from the VAE renders transfers to a ResNet arm at all."""
    space, ratios = gym.spaces.Box(-1, 1, (64,)), []
    for scale in (0.03, 0.44, 30.0, 9188.0):
        extractor = CorruptingExtractor(space, t_max=200).train()
        features = th.randn(4096, 64) * scale
        extractor._update_feature_std(features)
        th.manual_seed(0)
        noised = extractor.scheduler.add_noise(
            features, th.randn_like(features) * extractor.feature_std,
            th.full((features.shape[0],), 100, dtype=th.long))
        residual = noised - features * extractor.scheduler.alphas_cumprod[100].sqrt()
        ratios.append(float(features.var() / residual.var()))
    assert np.allclose(ratios, ratios[0], rtol=1e-3)


def test_q_head_cannot_move_the_policy():
    """The separate-optimizer guarantee, checked rather than asserted in a comment."""
    space = gym.spaces.Box(-1, 1, (40,))
    policy = policy_for("keypoint")(
        space, gym.spaces.Box(-1, 1, (2,)), lambda _: 3e-4, lstm_hidden_size=32,
        **features_extractor_kwargs("keypoint", corrupt_obs=True, t_max=200))
    q_params = {id(p) for p in policy.q_net.parameters()}
    policy_params = {id(p) for group in policy.optimizer.param_groups for p in group["params"]}
    assert q_params and not (q_params & policy_params)
    assert hasattr(policy, "q_optimizer")
