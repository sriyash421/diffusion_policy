"""Regression tests for recurrent_ppo, replacing the ad-hoc scratchpad scripts.

Each test pins one property the audit established, so a future edit that breaks it fails loudly
rather than silently changing what an arm measures.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch as th

from recurrent_ppo.corrupt_policy import (CorruptingExtractor, KeypointExtractor,
                                          features_extractor_kwargs, make_obs_noise_scheduler,
                                          policy_for)
from recurrent_ppo.pusht_gym import (AUG_NOISE_KEY, AUG_T_KEY, STATE_KEY, WS,
                                     AugmentationDraw, PushTGymEnv)

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
        # uint8 on purpose: SB3's preprocess_obs does the /255 itself
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


def _corrupt_space(dim, t_max=1000):
    return gym.spaces.Dict({
        STATE_KEY: gym.spaces.Box(-1, 1, (dim,)),
        AUG_NOISE_KEY: gym.spaces.Box(-np.inf, np.inf, (dim,)),
        AUG_T_KEY: gym.spaces.Box(0.0, float(t_max), (1,)),
    })


def test_corruption_matches_the_repo_mixin_when_unscaled():
    """A copy kept honest: if ObsCorruptionMixin changes, this fails instead of drifting.

    Calls `forward` itself, rather than re-implementing its body here -- the earlier version
    restated the operation in the test, so reordering it in the source would not have failed.
    """
    from diffusion_policy.model.common.obs_corruption import ObsCorruptionMixin

    class Reference(ObsCorruptionMixin):
        def __init__(self):
            self.obs_noise_scheduler = make_obs_noise_scheduler()
            self.training = True
            self._init_corruption(True, True)

    dim = 64
    extractor = CorruptingExtractor(_corrupt_space(dim), inner_class=KeypointExtractor,
                                    t_max=1000).eval()
    features = th.randn(32, dim)
    # the draw the mixin makes internally, made here instead and handed in with the observation
    th.manual_seed(7)
    noise = th.randn_like(features)                     # feature_std is still all ones
    timesteps = th.randint(0, 1000, (features.shape[0],)).long()
    mine = extractor({STATE_KEY: features, AUG_NOISE_KEY: noise,
                      AUG_T_KEY: timesteps.float().unsqueeze(1)})
    th.manual_seed(7)
    assert th.equal(mine, Reference().corrupt_obs_features(features))


def test_the_draw_is_replayed_rather_than_redrawn():
    """The property PPO's importance ratio depends on: same transition, same corruption.

    The noise used to be drawn inside `forward`, so each of PPO's --n-epochs passes over a
    stored transition corrupted it differently and `exp(log_prob - old_log_prob)` compared two
    observations rather than two policies.
    """
    dim = 64
    extractor = CorruptingExtractor(_corrupt_space(dim), inner_class=KeypointExtractor, t_max=200)
    obs = {STATE_KEY: th.randn(8, dim), AUG_NOISE_KEY: th.randn(8, dim),
           AUG_T_KEY: th.randint(0, 200, (8, 1)).float()}
    extractor.train()                     # the mode PPO's update epochs run in
    assert th.equal(extractor(obs), extractor(obs))
    extractor.eval()                      # and the mode rollout collection runs in
    assert th.equal(extractor(obs), extractor(obs))


def test_feature_std_is_frozen_outside_collection():
    """A std that moved between epochs would re-scale an already-stored transition's noise."""
    dim = 64
    extractor = CorruptingExtractor(_corrupt_space(dim), inner_class=KeypointExtractor, t_max=200)
    obs = {STATE_KEY: th.randn(256, dim) * 5.0, AUG_NOISE_KEY: th.randn(256, dim),
           AUG_T_KEY: th.randint(0, 200, (256, 1)).float()}
    extractor.train()
    before = extractor.feature_std.clone()
    extractor(obs)
    assert th.equal(extractor.feature_std, before)      # update_std defaults to False
    extractor.update_std = True                          # what FeatureStdWindow opens
    extractor(obs)
    assert not th.equal(extractor.feature_std, before)


def test_a_single_row_batch_cannot_poison_the_feature_std():
    """SB3 encodes ONE terminal observation to bootstrap a truncated episode.

    `std()` over a single sample is NaN and `clamp_min` does not repair a NaN, so an
    unguarded update puts NaN in the buffer permanently and every action mean becomes NaN
    from that step on. Dense-reward episodes always truncate, so this path is not an edge
    case -- it is hit every 300 steps, and the eval env's single env hits it every step.
    """
    dim = 64
    extractor = CorruptingExtractor(_corrupt_space(dim), inner_class=KeypointExtractor, t_max=200)
    extractor.update_std = True
    one = {STATE_KEY: th.randn(1, dim), AUG_NOISE_KEY: th.randn(1, dim),
           AUG_T_KEY: th.randint(0, 200, (1, 1)).float()}
    out = extractor(one)
    assert th.isfinite(extractor.feature_std).all()
    assert th.isfinite(out).all()
    assert not bool(extractor.feature_std_inited)      # nothing estimable from one sample

    many = {STATE_KEY: th.randn(32, dim) * 3.0, AUG_NOISE_KEY: th.randn(32, dim),
            AUG_T_KEY: th.randint(0, 200, (32, 1)).float()}
    extractor(many)
    assert bool(extractor.feature_std_inited)
    assert th.isfinite(extractor.feature_std).all()
    # and a single row AFTER initialisation must leave the estimate untouched, not wreck it
    before = extractor.feature_std.clone()
    extractor(one)
    assert th.equal(extractor.feature_std, before)


@pytest.mark.parametrize("obs_type,corrupt", [("keypoint", True), ("image", False), ("image", True)])
def test_evaluate_actions_reproduces_the_collected_log_prob(obs_type, corrupt):
    """A stored transition must ENCODE identically either side of SB3's training-mode flip.

    Collection runs the policy in eval mode and the update in train mode, which is what used to
    swap both the corruption noise and the image arm's random crop underneath the ratio. This
    covers the CLEAN image arm too, where only the crop was at fault.

    The features are asserted bitwise, not the log-prob within a tolerance, and deliberately:
    measured against the old redrawing behaviour, the features moved by up to 2.34 (0.30 on the
    clean image arm) while the resulting ratio moved only 1.5e-5, because an untrained head
    barely reads the difference. A log-prob tolerance is therefore a weak probe at
    initialisation -- it would let a reintroduced redraw pass. The ratio is checked too, as the
    end-to-end consequence, but the feature identity is the assertion with teeth.
    """
    from sb3_contrib.common.recurrent.type_aliases import RNNStates

    from recurrent_ppo.corrupt_policy import aug_for
    from recurrent_ppo.pusht_gym import AUG_CROP_KEY

    env = PushTGymEnv(obs_type=obs_type, max_episode_steps=20)
    aug = aug_for(obs_type, corrupt, render_size=96)
    env = AugmentationDraw(env, **aug)
    obs, _ = env.reset(seed=0)

    policy = policy_for(obs_type, corrupt)(
        env.observation_space, env.action_space, lambda _: 3e-4, lstm_hidden_size=16,
        **features_extractor_kwargs(obs_type, corrupt, t_max=200))
    # raw, not pre-divided: the policy's own preprocess_obs does the /255 on the image space
    obs_t = {k: th.as_tensor(np.asarray(v)).unsqueeze(0).float() for k, v in obs.items()}
    if obs_type == "image":
        assert AUG_CROP_KEY in obs_t
    starts = th.ones(1)
    shape = (policy.lstm_actor.num_layers, 1, policy.lstm_actor.hidden_size)
    z = lambda: (th.zeros(shape), th.zeros(shape))
    states = RNNStates(z(), z())

    def encode():
        with th.no_grad():
            features = policy.extract_features(obs_t)
        return features[0] if isinstance(features, tuple) else features

    policy.set_training_mode(False)                     # collection
    collected = encode()
    with th.no_grad():
        actions, _, old_log_prob, _ = policy.forward(obs_t, states, starts)
    policy.set_training_mode(True)                      # the update epochs
    assert th.equal(encode(), collected)
    with th.no_grad():
        _, log_prob, _ = policy.evaluate_actions(obs_t, actions, states, starts)
    assert th.allclose(th.exp(log_prob - old_log_prob), th.ones(1), atol=1e-6)


def test_scaling_makes_the_snr_independent_of_feature_magnitude():
    """Why a level chosen from the VAE renders transfers to a ResNet arm at all."""
    space, ratios = _corrupt_space(64, t_max=200), []
    for scale in (0.03, 0.44, 30.0, 9188.0):
        extractor = CorruptingExtractor(space, inner_class=KeypointExtractor, t_max=200).train()
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
    space = _corrupt_space(40, t_max=200)
    policy = policy_for("keypoint", corrupt_obs=True)(
        space, gym.spaces.Box(-1, 1, (2,)), lambda _: 3e-4, lstm_hidden_size=32,
        **features_extractor_kwargs("keypoint", corrupt_obs=True, t_max=200))
    q_params = {id(p) for p in policy.q_net.parameters()}
    policy_params = {id(p) for group in policy.optimizer.param_groups for p in group["params"]}
    assert q_params and not (q_params & policy_params)
    assert hasattr(policy, "q_optimizer")


def test_the_scheduler_still_matches_the_config_it_copies():
    """`make_obs_noise_scheduler` duplicates the YAML by value, so pin it to the YAML.

    Nothing else would notice the two drifting apart: the mixin comparison above takes its
    scheduler from this same function, so both sides would move together.
    """
    import yaml

    with open("diffusion_policy/config/train_pusht_diffusion_search.yaml") as f:
        cfg = yaml.safe_load(f)
    declared = dict(cfg["policy"]["obs_noise_scheduler"])
    assert declared.pop("_target_") == "diffusers.schedulers.scheduling_ddpm.DDPMScheduler"
    mine = make_obs_noise_scheduler().config
    for key, value in declared.items():
        assert mine[key] == value, f"{key}: config says {value!r}, make_obs_noise_scheduler {mine[key]!r}"
