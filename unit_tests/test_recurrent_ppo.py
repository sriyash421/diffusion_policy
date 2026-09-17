"""Regression tests for recurrent_ppo, replacing the ad-hoc scratchpad scripts.

Each test pins one property the audit established, so a future edit that breaks it fails loudly
rather than silently changing what an arm measures.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch as th

from recurrent_ppo.corrupt_policy import (CorruptingExtractor, KeypointExtractor, aug_for,
                                          features_extractor_kwargs, make_obs_noise_scheduler,
                                          policy_for)
from recurrent_ppo.pusht_gym import (AGENT_BOUNDS, AUG_CROP_KEY, AUG_NOISE_KEY, AUG_T_KEY,
                                     STATE_KEY, WS, AugmentationDraw, PushTGymEnv)

GOAL_STATE = np.array([256.0, 200.0, 256.0, 256.0, np.pi / 4])   # block sitting on the goal pose


def _corrupt_space(dim, t_max=1000):
    return gym.spaces.Dict({
        STATE_KEY: gym.spaces.Box(-1, 1, (dim,)),
        AUG_NOISE_KEY: gym.spaces.Box(-np.inf, np.inf, (dim,)),
        AUG_T_KEY: gym.spaces.Box(0.0, float(t_max), (1,)),
    })


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


@pytest.mark.parametrize("obs_type", ["keypoint", "state", "image"])
def test_observations_stay_in_the_declared_box(obs_type):
    env = PushTGymEnv(obs_type=obs_type, max_episode_steps=40)
    env.reset(seed=0)
    for _ in range(15):
        obs, _ = env.reset()
        for _ in range(40):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            # the flat arms are normalised; the image arm went image-only, so it has no
            # [-1, 1] vector left to check -- `contains` is the whole claim there
            if obs_type != "image":
                assert obs.min() >= -1.0 and obs.max() <= 1.0
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

    # both modes address the agent's USABLE region, not the image bounds. The agent is a kinematic
    # body, so pymunk's walls do not stop it -- this clip is the only thing that does, and letting
    # it reach [0, WS] let two runs park outside the arena where the block cannot be touched.
    lo, hi = AGENT_BOUNDS
    assert lo > 0 and hi < WS
    # delta clips the ACTION to +/-1 first, so a huge action is a bounded offset from where the
    # agent is -- never a jump to the edge. What must hold is that the target stays in the region.
    env.env.reset_to_state = np.array([lo + 1.0, hi - 1.0, 300.0, 300.0, 0.0])
    env.env.reset()
    for extreme in ([-50.0, 50.0], [50.0, -50.0]):
        target = env._convert_action(np.array(extreme))
        assert np.all(target >= lo) and np.all(target <= hi)

    absolute = PushTGymEnv(obs_type="keypoint", action_mode="absolute")
    assert np.allclose(absolute._convert_action(np.array([-1.0, 1.0])), [lo, hi])
    assert np.allclose(absolute._convert_action(np.zeros(2)), [(lo + hi) / 2] * 2)


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
        # the chain itself, not the observation. The mask used to ride along as 20 extra
        # features and could be read back out of obs[20:38:2]; it is now APPLIED to the
        # keypoints instead, so an occluded one is indistinguishable from one at the arena
        # centre -- which is the property under test everywhere else, and would make this
        # measurement wrong. `_visible` is what the mask is built from.
        mask = np.asarray(env._visible, dtype=bool)
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


def test_an_unreachable_persistent_combination_is_announced_not_clipped(capsys):
    """The two modes are only comparable while they share a marginal, and they cannot always.

    Holding a mean hidden run of `persistence` at a low visible rate needs
    p(visible->hidden) = (1-v)/(v*persistence) > 1, which no rate can be. The clip that follows
    silently moves the stationary visible fraction off `visible_rate`, so a `persistent` arm
    stops being matched with the `iid` arm it is supposed to be read against -- the one property
    test_occlusion_modes_share_a_marginal_and_differ_in_structure exists to pin. Announcing it
    is the difference between a known limitation and a wrong comparison.
    """
    # 1/(1 + 20) = 0.0476 is the lowest rate a mean run of 20 can hold
    env = PushTGymEnv(obs_type="keypoint", keypoint_visible_rate=0.02, occlusion="persistent",
                      occlusion_persistence=20.0)
    out = capsys.readouterr().out
    assert "[WARN]" in out and "NOT matched" in out
    # and the warning tells the truth about what was actually built
    p_hv, p_vh = env._p_hidden_to_visible, env._p_visible_to_hidden
    assert p_vh == 1.0                                     # clipped
    assert f"{p_hv / (p_hv + 1.0):.3f}" in out
    assert env._stationary_visible == pytest.approx(p_hv / (p_hv + p_vh))
    assert env._stationary_visible > 0.02                  # NOT the requested rate

    # the reachable side of the same boundary stays silent and stays matched
    env = PushTGymEnv(obs_type="keypoint", keypoint_visible_rate=0.5, occlusion="persistent",
                      occlusion_persistence=20.0)
    assert "[WARN]" not in capsys.readouterr().out
    assert env._stationary_visible == pytest.approx(0.5)

    # `iid` can never reach it: p_vh is 1 - v there, always a rate
    env = PushTGymEnv(obs_type="keypoint", keypoint_visible_rate=0.02, occlusion="iid",
                      occlusion_persistence=20.0)
    assert "[WARN]" not in capsys.readouterr().out
    assert env._stationary_visible == pytest.approx(0.02)


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
    extractor = CorruptingExtractor(_corrupt_space(dim), inner_class=KeypointExtractor).eval()
    features = th.randn(32, dim)
    # the draw the mixin makes internally, made here instead and handed in with the observation
    th.manual_seed(7)
    noise = th.randn_like(features)                     # feature_std is still all ones
    timesteps = th.randint(0, 1000, (features.shape[0],)).long()
    mine = extractor({STATE_KEY: features, AUG_NOISE_KEY: noise,
                      AUG_T_KEY: timesteps.float().unsqueeze(1)})
    th.manual_seed(7)
    assert th.equal(mine, Reference().corrupt_obs_features(features))


def test_scaling_makes_the_snr_independent_of_feature_magnitude():
    """Why a level chosen from the VAE renders transfers to a ResNet arm at all."""
    space, ratios = _corrupt_space(64, t_max=200), []
    for scale in (0.03, 0.44, 30.0, 9188.0):
        extractor = CorruptingExtractor(space, inner_class=KeypointExtractor).train()
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
        **features_extractor_kwargs("keypoint", corrupt_obs=True))
    q_params = {id(p) for p in policy.q_net.parameters()}
    policy_params = {id(p) for group in policy.optimizer.param_groups for p in group["params"]}
    assert q_params and not (q_params & policy_params)
    assert hasattr(policy, "q_optimizer")


def test_feedforward_q_head_cannot_move_the_policy():
    """The same separate-optimizer guarantee, for the non-recurrent policy."""
    policy = policy_for("keypoint", recurrent=False)(
        gym.spaces.Box(-1, 1, (160,)), gym.spaces.Box(-1, 1, (2,)), lambda _: 3e-4,
        **features_extractor_kwargs("keypoint", corrupt_obs=True))
    q_params = {id(p) for p in policy.q_net.parameters()}
    policy_params = {id(p) for group in policy.optimizer.param_groups for p in group["params"]}
    assert q_params and not (q_params & policy_params)
    assert hasattr(policy, "q_optimizer")


@pytest.mark.parametrize("space", [
    gym.spaces.Box(-1, 1, (40,), dtype=np.float32),                       # any flat Box
    gym.spaces.Box(0, 255, (3, 8, 8), dtype=np.uint8),                    # a channels-first image
    gym.spaces.Dict({"image": gym.spaces.Box(0, 255, (3, 8, 8), dtype=np.uint8),
                     "agent_pos": gym.spaces.Box(-1, 1, (2,), dtype=np.float32)}),
])
def test_frame_stacker_matches_vec_frame_stack(space):
    """The video rollout stacks by hand, because its env is not a VecEnv.

    Both the AXIS and the order are invisible when wrong -- it simply feeds the policy a
    differently-arranged input than it trained on -- so this pins them against SB3's own
    StackedObservations, which is also what FrameStacker delegates to. The test is therefore
    about the num_envs=1 adapter around it: the batch axis, and the dict handling.
    """
    from stable_baselines3.common.vec_env.stacked_observations import StackedObservations

    from recurrent_ppo.callbacks import FrameStacker

    n_stack = 4
    reference = StackedObservations(1, n_stack, space)
    ours = FrameStacker(n_stack, space)
    frames = [space.sample() for _ in range(6)]

    def batch(obs):
        return {k: v[None] for k, v in obs.items()} if isinstance(obs, dict) else obs[None]

    def same(theirs, mine):
        if isinstance(mine, dict):
            return all(np.array_equal(theirs[k][0], mine[k]) for k in mine)
        return np.array_equal(theirs[0], mine)

    assert same(reference.reset(batch(frames[0])), ours.reset(frames[0]))
    for frame in frames[1:]:
        theirs, _ = reference.update(batch(frame), np.zeros(1, dtype=bool), [{}])
        mine = ours.update(frame)
        assert same(theirs, mine), "stacked observations diverged"

    # the newest frame is LAST, which is the half of the convention a symmetric test would miss
    flat = mine["agent_pos"] if isinstance(mine, dict) else mine
    newest = frames[-1]["agent_pos"] if isinstance(mine, dict) else frames[-1]
    axis = 0 if getattr(reference, "channels_first", False) else -1
    if not isinstance(mine, dict):
        assert np.array_equal(np.take(flat, range(-newest.shape[axis], 0), axis=axis), newest)
    else:
        assert np.array_equal(flat[-newest.shape[0]:], newest)


@pytest.mark.parametrize("n_stack,expected", [(1, 20), (4, 80)])
def test_stack_arch_widens_the_observation(n_stack, expected):
    """What the policy is actually handed, through the same wrap() training and play both use."""
    from recurrent_ppo.arch import StackArch
    from recurrent_ppo.pusht_gym import build_vec_env

    venv = build_vec_env(obs_type="keypoint", n_envs=1, seed=0, use_subproc=False,
                         max_episode_steps=10)
    try:
        wrapped = StackArch().wrap(venv, {"n_stack": n_stack})
        assert wrapped.observation_space.shape == (expected,)
    finally:
        venv.close()


# ------------------------------------------------------------------ the state arm

def test_state_obs_is_six_dimensional_and_matches_pushtenv():
    """The state arm is PushTEnv's own observation, with only the angle re-encoded."""
    env = PushTGymEnv(obs_type="state", max_episode_steps=20)
    obs, _ = env.reset(seed=0)
    raw = env.env._get_obs()
    assert env.observation_space.shape == (6,)
    assert np.allclose(obs[:4], np.asarray(raw[:4]) / (WS / 2) - 1.0, atol=1e-5)
    assert np.allclose(obs[4:], [np.cos(raw[4]), np.sin(raw[4])], atol=1e-6)


def test_state_angle_has_no_discontinuity_at_the_wrap():
    """Why the angle is (cos, sin) and not a scalar.

    `block.angle % 2*pi` means 0.01 and 6.27 rad are the SAME pose. Under a scalar encoding they
    land at opposite ends of [-1, 1] -- the furthest apart two values can be -- and the policy
    has to learn that the coordinate wraps from the handful of episodes where the block rotates
    through zero.
    """
    env = PushTGymEnv(obs_type="state", max_episode_steps=20)
    env.reset(seed=0)

    def angle_obs(theta):
        env.env.reset_to_state = np.array([256.0, 200.0, 256.0, 256.0, theta])
        env.env.reset()
        return env._convert_obs(env.env._get_obs())[4:]

    near_zero, near_two_pi = angle_obs(0.01), angle_obs(2 * np.pi - 0.01)
    assert np.linalg.norm(near_zero - near_two_pi) < 0.05
    # and it is not degenerate: half a turn away is far
    assert np.linalg.norm(near_zero - angle_obs(np.pi)) > 1.9


# ------------------------------------------------------------------ the fixed evaluation set

def _eval_starts(env, n_episodes, seed=None):
    """The first observation of n consecutive episodes, as evaluate_policy would meet them."""
    if seed is not None:
        env.seed(seed)
    return np.array([env.reset()[0].copy() for _ in range(n_episodes)])


def test_evaluation_replays_the_same_episodes_when_reseeded():
    """A fixed benchmark set, so two checkpoints differ by the policy and not by the draw.

    SB3 seeds an eval env once at construction and never again -- neither EvalCallback nor
    evaluate_policy mentions `seed` -- so without this every evaluation sampled fresh episodes and
    consecutive points were unpaired.
    """
    from recurrent_ppo.pusht_gym import build_vec_env

    env = build_vec_env(obs_type="keypoint", n_envs=1, seed=10_000, use_subproc=False,
                        max_episode_steps=20, agent_near_block_prob=1.0)
    try:
        assert np.allclose(_eval_starts(env, 6, seed=10_000), _eval_starts(env, 6, seed=10_000))
        # and that it is re-seeding that does it, not the env being degenerate
        assert not np.allclose(_eval_starts(env, 6, seed=10_000), _eval_starts(env, 6))
    finally:
        env.close()


def test_the_fixed_eval_set_does_not_depend_on_the_policy():
    """The set must be the same for every checkpoint and every arm, not merely stable in one run.

    Episode starts come from the env's own RNG, and reset's rejection sampling consumes a
    policy-independent number of draws -- so stepping different actions in between cannot shift
    which episodes the next evaluation sees.
    """
    from recurrent_ppo.pusht_gym import build_vec_env

    env = build_vec_env(obs_type="keypoint", n_envs=1, seed=10_000, use_subproc=False,
                        max_episode_steps=20, agent_near_block_prob=1.0)
    try:
        baseline = _eval_starts(env, 5, seed=10_000)
        env.seed(10_000)
        env.reset()
        for _ in range(37):                       # an arbitrary, different "policy"
            env.step(np.array([[0.7, -0.4]], dtype=np.float32))
        assert np.allclose(_eval_starts(env, 5, seed=10_000), baseline)
    finally:
        env.close()


def test_clean_eval_callback_scores_a_fixed_policy_identically_twice():
    """The re-seed reaches the real callback path, not just a hand-driven env.

    With the policy held fixed, two evaluations must return the SAME mean reward. Before the
    fix they differed, because each drew a fresh sample of episodes.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import Logger

    from recurrent_ppo.callbacks import CleanEvalCallback
    from recurrent_ppo.pusht_gym import build_vec_env

    env = build_vec_env(obs_type="keypoint", n_envs=1, seed=10_000, use_subproc=False,
                        max_episode_steps=25, reward_mode="delta", agent_near_block_prob=1.0)
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, device="cpu", seed=0)
    model.set_logger(Logger(folder=None, output_formats=[]))

    def run(eval_seed):
        cb = CleanEvalCallback(env, n_eval_episodes=4, eval_freq=1, deterministic=True,
                               warn=False, verbose=0, eval_seed=eval_seed)
        cb.init_callback(model)
        scores = []
        for _ in range(2):
            cb.on_step()
            scores.append(cb.last_mean_reward)
        return scores

    fixed = run(10_000)
    assert fixed[0] == pytest.approx(fixed[1]), f"fixed set drifted: {fixed}"
    # and the old behaviour is what it replaces: without a seed the two samples differ
    assert run(None)[0] != pytest.approx(run(None)[1])
    env.close()
def test_the_draw_is_replayed_rather_than_redrawn():
    """The property PPO's importance ratio depends on: same transition, same corruption.

    The noise used to be drawn inside `forward`, so each of PPO's --n-epochs passes over a
    stored transition corrupted it differently and `exp(log_prob - old_log_prob)` compared two
    observations rather than two policies.
    """
    dim = 64
    extractor = CorruptingExtractor(_corrupt_space(dim), inner_class=KeypointExtractor)
    obs = {STATE_KEY: th.randn(8, dim), AUG_NOISE_KEY: th.randn(8, dim),
           AUG_T_KEY: th.randint(0, 200, (8, 1)).float()}
    extractor.train()                     # the mode PPO's update epochs run in
    assert th.equal(extractor(obs), extractor(obs))
    extractor.eval()                      # and the mode rollout collection runs in
    assert th.equal(extractor(obs), extractor(obs))


def test_feature_std_is_frozen_outside_collection():
    """A std that moved between epochs would re-scale an already-stored transition's noise."""
    dim = 64
    extractor = CorruptingExtractor(_corrupt_space(dim), inner_class=KeypointExtractor)
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
    extractor = CorruptingExtractor(_corrupt_space(dim), inner_class=KeypointExtractor)
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

    The crop half is now structural rather than replayed: PPO centre-crops in both phases, so
    there is no offset to carry and nothing a mode flip could change. The assertion stays
    because the NOISE half is still replayed from the observation, and because a reintroduced
    random crop would fail here first.

    The features are asserted bitwise, not the log-prob within a tolerance, and deliberately:
    measured against the old redrawing behaviour, the features moved by up to 2.34 (0.30 on the
    clean image arm) while the resulting ratio moved only 1.5e-5, because an untrained head
    barely reads the difference. A log-prob tolerance is therefore a weak probe at
    initialisation -- it would let a reintroduced redraw pass. The ratio is checked too, as the
    end-to-end consequence, but the feature identity is the assertion with teeth.
    """
    from sb3_contrib.common.recurrent.type_aliases import RNNStates

    env = PushTGymEnv(obs_type=obs_type, max_episode_steps=20)
    aug = aug_for(obs_type, corrupt, render_size=96)
    if aug:
        env = AugmentationDraw(env, **aug)
    obs, _ = env.reset(seed=0)

    policy = policy_for(obs_type, recurrent=True, corrupt_obs=corrupt)(
        env.observation_space, env.action_space, lambda _: 3e-4, lstm_hidden_size=16,
        **features_extractor_kwargs(obs_type, corrupt))
    # raw, not pre-divided: the policy's own preprocess_obs does the /255 on the image space
    obs_t = {k: th.as_tensor(np.asarray(v)).unsqueeze(0).float() for k, v in obs.items()}
    if obs_type == "image":
        # PPO carries NO crop offset: the arm centre-crops in both phases, so the transform is
        # a property of the extractor rather than of the observation. See aug_for.
        assert AUG_CROP_KEY not in obs_t
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


@pytest.mark.parametrize("obs_type,n_stack,expected", [
    ("state", 1, 6), ("state", 4, 24),
    ("keypoint", 1, 20), ("keypoint", 4, 80),
    ("image", 1, 512), ("image", 4, 2048),
])
def test_the_draw_is_as_wide_as_the_features_it_corrupts(obs_type, n_stack, expected):
    """The noise vector must match what the extractor produces from a STACKED observation.

    Every arm widens with the stack, but for different reasons and it is easy to get wrong: a
    flat observation is concatenated whole, while the image arm runs each frame through the one
    shared ResNet and concatenates the 512-d results. Since that arm went image-only there is
    nothing concatenated alongside, so it is 512*n_stack and not (512+2)*n_stack. A mismatch
    crashes at the first forward, i.e. only once a run is already on a GPU.
    """
    from recurrent_ppo.arch import ARCHS
    from recurrent_ppo.config import DEFAULTS
    from recurrent_ppo.pusht_gym import VecAugmentationDraw, build_vec_env, env_kwargs_from

    cfg = dict(DEFAULTS)
    cfg.update(obs=obs_type, n_stack=n_stack, corrupt_obs=True, delta_scale=33.0)
    venv = ARCHS["stack"].wrap(build_vec_env(obs_type=obs_type, n_envs=2, seed=0,
                                             use_subproc=False,
                                             **env_kwargs_from(cfg, obs_type)), cfg)
    aug = aug_for(obs_type, True, render_size=cfg["render_size"], n_stack=n_stack, snr=1.92)
    assert aug["feature_dim"] == expected
    venv = VecAugmentationDraw(venv, seed=0, **aug)
    kw = features_extractor_kwargs(obs_type, True)
    ext = kw["features_extractor_class"](venv.observation_space, **kw["features_extractor_kwargs"])
    assert ext.features_dim == expected
    batch = {k: th.as_tensor(np.asarray(v)).float() for k, v in venv.reset().items()}
    if "image" in batch:
        batch["image"] = batch["image"] / 255.0
    assert ext(batch).shape == (2, expected)
    venv.close()


def test_a_target_snr_pins_one_level():
    """--corrupt-snr addresses a level by what it MEANS, not by an index into the schedule."""
    from recurrent_ppo.corrupt_policy import snr_to_timestep
    from recurrent_ppo.pusht_gym import aug_draw

    assert snr_to_timestep(1.92) == 200
    assert snr_to_timestep(3.67) == 150
    fixed = aug_for("keypoint", True, snr=1.92)
    assert (fixed["t_min"], fixed["t_max"]) == (200, 201)
    rng = np.random.default_rng(0)
    assert {int(aug_draw(rng, **fixed)[AUG_T_KEY][0]) for _ in range(200)} == {200}
    # without it, the level is still drawn across the range
    drawn = aug_for("keypoint", True, t_max=200)
    assert (drawn["t_min"], drawn["t_max"]) == (0, 200)
    assert len({int(aug_draw(rng, **drawn)[AUG_T_KEY][0]) for _ in range(200)}) > 1


@pytest.mark.parametrize("n_stack", [1, 4])
def test_the_terminal_observation_carries_the_draw(n_stack):
    """SB3 encodes infos["terminal_observation"] to bootstrap a truncated episode.

    That observation reaches the same extractor as any other, so it needs the draw's keys. It
    did not have them, and because dense-reward episodes ALWAYS truncate, every corrupted run
    died at its first episode boundary -- a few minutes in, on a GPU, well past any import check.
    """
    from recurrent_ppo.arch import ARCHS
    from recurrent_ppo.config import DEFAULTS
    from recurrent_ppo.pusht_gym import VecAugmentationDraw, build_vec_env, env_kwargs_from

    horizon = 8
    cfg = dict(DEFAULTS)
    cfg.update(obs="keypoint", n_stack=n_stack, corrupt_obs=True, delta_scale=33.0,
               max_episode_steps=horizon)
    venv = ARCHS["stack"].wrap(build_vec_env(obs_type="keypoint", n_envs=2, seed=0,
                                             use_subproc=False,
                                             **env_kwargs_from(cfg, "keypoint")), cfg)
    venv = VecAugmentationDraw(venv, seed=0,
                               **aug_for("keypoint", True, n_stack=n_stack, snr=1.92))
    keys = set(venv.observation_space.spaces)
    venv.reset()
    saw_terminal = False
    for _ in range(horizon + 2):
        _, _, dones, infos = venv.step(np.zeros((2, 2), dtype=np.float32))
        for done, info in zip(dones, infos):
            if done and "terminal_observation" in info:
                saw_terminal = True
                assert set(info["terminal_observation"]) == keys
                assert info["terminal_observation"][AUG_NOISE_KEY].shape == \
                    venv.observation_space[AUG_NOISE_KEY].shape
    assert saw_terminal, "the horizon should have truncated at least one episode"
    venv.close()


def test_the_draw_tolerates_a_tuple_infos():
    """SubprocVecEnv hands back infos as a TUPLE; DummyVecEnv hands back a list.

    The terminal-observation fix assigned into infos[i] directly, which works under
    DummyVecEnv and raises TypeError under SubprocVecEnv -- so the test that covered the
    logic passed while every real run (which uses subprocesses) died. A stub is used rather
    than a real SubprocVecEnv because the container type is the whole point and forkserver
    startup is not.
    """
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv

    from recurrent_ppo.pusht_gym import VecAugmentationDraw

    dim = 8

    class TupleInfosVecEnv(VecEnv):
        def __init__(self):
            super().__init__(2, gym.spaces.Box(-1, 1, (dim,)), gym.spaces.Box(-1, 1, (2,)))
        def reset(self):
            return np.zeros((2, dim), dtype=np.float32)
        def step_async(self, actions):
            pass
        def step_wait(self):
            # a tuple, and a terminal_observation on the env that just finished
            infos = ({"terminal_observation": np.ones(dim, dtype=np.float32)}, {})
            return (np.zeros((2, dim), dtype=np.float32), np.zeros(2),
                    np.array([True, False]), infos)
        def close(self): pass
        def get_attr(self, name, indices=None): return [None, None]  # VecEnv.__init__ reads render_mode
        def set_attr(self, name, value, indices=None): pass
        def env_method(self, name, *a, indices=None, **k): return []
        def env_is_wrapped(self, cls, indices=None): return [False, False]

    env = VecAugmentationDraw(TupleInfosVecEnv(), feature_dim=dim, t_min=200, t_max=201, seed=0)
    obs, _, dones, infos = env.step(np.zeros((2, 2), dtype=np.float32))
    assert set(obs) == {STATE_KEY, AUG_NOISE_KEY, AUG_T_KEY}
    term = infos[0]["terminal_observation"]
    assert set(term) == {STATE_KEY, AUG_NOISE_KEY, AUG_T_KEY}
    assert term[AUG_NOISE_KEY].shape == (dim,) and int(term[AUG_T_KEY][0]) == 200


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_the_ppo_image_arm_centre_crops_in_both_modes(mode):
    """PPO supplies no crop offset, so it centre-crops while acting AND while learning.

    Two things are asserted because either alone would pass for the wrong reason: `aug_for`
    emits no `crop_span`, so no `aug_crop` key can reach the observation; and with no key the
    extractor takes the CENTRE crop rather than CropRandomizer's `self.training` branch. The
    second is what makes the first safe -- a deterministic transform cannot be redrawn between
    rollout and update, so PPO's ratio stays a ratio of policies without any replay machinery.
    """
    from recurrent_ppo.corrupt_policy import ST_CROP, STResNetExtractor, aug_for

    assert aug_for("image", corrupt_obs=False, render_size=96) is None
    assert "crop_span" not in (aug_for("image", corrupt_obs=True, render_size=96) or {})

    space = gym.spaces.Dict({"image": gym.spaces.Box(0, 255, (3, 96, 96), np.uint8)})
    extractor = STResNetExtractor(space)
    assert not extractor.replay_crop
    extractor.train(mode == "train")

    th.manual_seed(0)
    img = th.rand(1, 3, 96, 96)
    got = extractor._crop(img, {"image": img})

    lo = (96 - ST_CROP) // 2
    expected = img[:, :, lo:lo + ST_CROP, lo:lo + ST_CROP]
    assert th.equal(got, expected), f"{mode}: not the centre crop"
    # and it is stable across calls, which is the property the ratio depends on
    assert th.equal(extractor._crop(img, {"image": img}), got)


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_a_supplied_crop_offset_wins_over_the_training_flag(mode):
    """A stored offset is honoured in BOTH modes -- `self.training` is never consulted.

    This is the contract the OFF-POLICY arms run on. SAC wraps its env in VecAugmentationDraw
    so the offset rides in the observation, and `sac.score.obs_for_arm` supplies the centre one
    at deployment; both rely on the supplied offset beating CropRandomizer's own mode branch.
    SB3 has that flag backwards for this purpose -- False while collecting, True during the
    update -- so without this the critic and the BON head would be fit on randomly-cropped
    features against targets produced under a centre crop.

    The property is easy to lose by accident -- deleting one `_forced_offsets` line restores
    CropRandomizer's default -- and nothing else would fail if it did. PPO no longer takes this
    path at all: it supplies no offset and centre-crops, which
    test_the_ppo_image_arm_centre_crops_in_both_modes covers.
    """
    from recurrent_ppo.corrupt_policy import ST_CROP, STResNetExtractor

    space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, (3, 96, 96), np.uint8),
        AUG_CROP_KEY: gym.spaces.Box(0.0, float(96 - ST_CROP - 1), (2,), np.float32),
    })
    extractor = STResNetExtractor(space)
    assert extractor.replay_crop
    extractor.train(mode == "train")

    th.manual_seed(0)
    img = th.rand(1, 3, 96, 96)
    obs = {"image": img, AUG_CROP_KEY: th.tensor([[3.0, 11.0]])}
    got = extractor._crop(img, obs)

    # the crop the stored offset names, computed independently of CropRandomizer
    expected = img[:, :, 3:3 + ST_CROP, 11:11 + ST_CROP]
    assert th.equal(got, expected), f"{mode}: not the stored offset"

    # and it is genuinely NOT the centre crop, which is what would come back on a regression
    lo = (96 - ST_CROP) // 2
    assert not th.equal(got, img[:, :, lo:lo + ST_CROP, lo:lo + ST_CROP])

    # a different stored offset must give a different crop -- i.e. the offset is really read
    other = dict(obs, **{AUG_CROP_KEY: th.tensor([[0.0, 0.0]])})
    assert not th.equal(extractor._crop(img, other), got)


# ------------------------------------------------------------------ the image arm is image only
def test_the_image_arm_observation_carries_no_agent_pos():
    """agent_pos is absent from the image arm, in all three places that used to encode its width.

    Keeping it made this a 514-d observation: the ResNet's 512 features plus the arm's exact
    position in closed form. `task/pusht_image_search_imgonly.yaml` forbids exactly that for the
    diffusion-policy arms -- they assert no low_dim key is declared, because handing the policy
    the pose analytically is a strictly stronger observation than the standard PushT-image
    setup. An RL arm with the privilege could not be compared against an offline arm without it,
    which defeats the point of mirroring ST's encoder in the first place.
    """
    from recurrent_ppo.config import DEFAULTS
    from recurrent_ppo.pusht_gym import PushTGymEnv

    env = PushTGymEnv(obs_type="image", render_size=96)
    try:
        assert set(env.observation_space.spaces) == {"image"}, \
            f"image arm observation space is {sorted(env.observation_space.spaces)}"
        obs, _ = env.reset(seed=0)
        assert set(obs) == {"image"}, f"_convert_obs emitted {sorted(obs)}"
        assert obs["image"].dtype == np.uint8
    finally:
        env.close()

    # the extractor's width, and the noise drawn to match it, agree at every stack depth
    kw = features_extractor_kwargs("image", False)
    ext = kw["features_extractor_class"](env.observation_space,
                                        **kw["features_extractor_kwargs"])
    assert ext.features_dim == 512
    for n_stack in (1, DEFAULTS["n_stack"]):
        aug = aug_for("image", True, render_size=96, n_stack=n_stack, snr=1.92)
        assert aug["feature_dim"] == 512 * n_stack, f"n_stack={n_stack}"

# ---------------------------------------------------------------- the image arm, stacked
@pytest.mark.parametrize("n_stack", [1, 4])
def test_image_encoder_runs_the_shared_resnet_per_frame(n_stack):
    """VecFrameStack concatenates on the CHANNEL axis, and the pretrained conv1 takes 3.

    Before this, `--obs image --n-stack 4` handed a 12-channel tensor to a 3-channel kernel and
    died. ST resolves the same problem the same way for its n_obs_steps: reshape
    (B, To, C, H, W) to (B*To, C, H, W), run the ONE encoder, concatenate the features. Widening
    conv1 instead would discard the pretrained kernel, which is the whole reason this encoder
    matches ST's.
    """
    from stable_baselines3.common.vec_env.stacked_observations import StackedObservations

    from recurrent_ppo.corrupt_policy import STResNetExtractor

    base = gym.spaces.Dict({"image": gym.spaces.Box(0, 255, (3, 96, 96), dtype=np.uint8)})
    space = base if n_stack == 1 else StackedObservations(1, n_stack, base).stacked_observation_space
    extractor = STResNetExtractor(space).eval()

    assert extractor.resnet.conv1.in_channels == 3, "the pretrained kernel must be used as trained"
    assert extractor.features_dim == 512 * n_stack, "image-only: no agent_pos alongside"
    out = extractor({"image": th.rand(2, 3 * n_stack, 96, 96)})
    assert out.shape == (2, extractor.features_dim)


def test_image_stack_keeps_frames_oldest_first():
    """The reshape must not scramble the stack: frame k of the channel axis becomes row k."""
    n_stack = 4
    image = th.zeros(1, 3 * n_stack, 8, 8)
    for k in range(n_stack):
        image[0, 3 * k:3 * k + 3] = k                 # frame k is a constant image of value k
    rows = image.reshape(1 * n_stack, 3, 8, 8)[:, 0, 0, 0]
    assert rows.tolist() == [0.0, 1.0, 2.0, 3.0], "oldest-first order lost in the reshape"


def test_every_frame_of_a_stack_gets_the_SAME_stored_crop():
    """One offset per transition, shared across the stack -- and replayed exactly.

    The draw is per stacked observation: VecAugmentationDraw sits outside `arch.wrap` and stores
    a single (2,) offset, which is what makes PPO's ratio a ratio of policies on the image arm.
    Per-frame encoding turns one observation into n_frames rows, so the offset is repeated
    rather than redrawn -- a per-frame offset would need the draw itself to widen to
    (n_frames, 2), changing the observation space the rollout buffer stores.
    """
    from stable_baselines3.common.vec_env.stacked_observations import StackedObservations

    from recurrent_ppo.corrupt_policy import STResNetExtractor

    n_stack, crop_span = 4, 96 - 76
    base = gym.spaces.Dict({"image": gym.spaces.Box(0, 255, (3, 96, 96), dtype=np.uint8)})
    stacked = StackedObservations(1, n_stack, base).stacked_observation_space
    space = gym.spaces.Dict({**stacked.spaces,
                             AUG_CROP_KEY: gym.spaces.Box(0.0, float(crop_span - 1), shape=(2,),
                                                          dtype=np.float32)})
    extractor = STResNetExtractor(space)
    assert extractor.replay_crop, "the stored offset must be picked up when it is present"
    extractor.train()                          # the mode PPO's update epochs run in

    # every frame of the stack is the SAME image, so if each frame were cropped differently
    # their features would differ
    frame = th.rand(1, 3, 96, 96)
    obs = {"image": frame.repeat(1, n_stack, 1, 1),
           AUG_CROP_KEY: th.tensor([[3.0, 7.0]])}
    out = extractor(obs)[0].reshape(n_stack, 512)
    for k in range(1, n_stack):
        assert th.allclose(out[0], out[k], atol=1e-5), "frames of one stack were cropped differently"

    # and the whole thing replays: same transition, same features, across epochs
    assert th.equal(extractor(obs), extractor(obs))


# ------------------------------------------------------- the shared eval episode set
def test_a_supplied_reset_state_is_used_verbatim():
    """`reset_to_state` must survive BOTH rejection tests, or episodes are silently renumbered.

    This is what lets an RL arm be scored on the same episodes as the diffusion-policy arms:
    the recorded first frame of each held-out demo episode, replayed exactly. The failure mode
    it guards is quiet. `block_zero_coverage` redraws any start whose block already overlaps the
    goal -- 28.7% of uniform draws do, and the demonstrations were never filtered for it -- so
    without the bypass, env k would come up on a DIFFERENT episode than the caller asked for,
    with nothing raising and every cross-arm comparison off by that episode.
    """
    env = PushTGymEnv(obs_type="state", block_zero_coverage=True)
    try:
        # one reset first: goal_pose is assigned in PushTEnv._setup, which runs on reset, so
        # it does not exist on a freshly constructed env
        env.reset(seed=0)
        # a state whose block sits exactly on the goal, i.e. coverage 1.0 -- the case
        # block_zero_coverage exists to reject
        goal = np.asarray(env.env.goal_pose, dtype=np.float64)
        state = np.array([256.0, 256.0, goal[0], goal[1], goal[2]], dtype=np.float64)
        env.reset_to_state = state
        env.reset(seed=0)

        got = np.array([*env.env.agent.position, *env.env.block.position, env.env.block.angle])
        assert np.allclose(got, state, atol=1e-6), f"redrawn: asked {state}, got {got}"
        assert env._block_coverage() > 0.9, \
            "the fixture no longer exercises the block_zero_coverage branch"

        # and it PERSISTS: same env, same state on the next reset, matching PushTEnv's contract
        env.reset(seed=1)
        again = np.array([*env.env.agent.position, *env.env.block.position, env.env.block.angle])
        assert np.allclose(again, state, atol=1e-6), "reset_to_state was consumed, not persistent"
    finally:
        env.close()


def test_clearing_reset_to_state_restores_sampling():
    """Setting it back to None must return the env to its own draw, not freeze the last state."""
    env = PushTGymEnv(obs_type="state")
    try:
        env.reset_to_state = np.array([100.0, 100.0, 200.0, 200.0, 0.5])
        env.reset(seed=0)
        pinned = np.array([*env.env.agent.position, *env.env.block.position])

        env.reset_to_state = None
        seen = set()
        for s in range(6):
            env.reset(seed=s)
            seen.add(tuple(np.round([*env.env.agent.position, *env.env.block.position], 3)))
        assert len(seen) > 1, "still pinned after clearing reset_to_state"
        assert tuple(np.round(pinned, 3)) not in seen or len(seen) > 1
    finally:
        env.close()


def test_pinning_reaches_the_inner_env_through_the_wrapper_chain():
    """`_pin` must actually move the state into PushTGymEnv, not onto the Monitor around it.

    A VecEnv's `set_attr` is a plain `setattr` on the outermost per-env wrapper, so the naive
    spelling puts `reset_to_state` on `Monitor` and every episode is still sampled -- while the
    results table is labelled with manifest episode indices. Nothing raises. This asserts the
    two envs come up on exactly the states they were handed, which is the only way to tell.
    """
    from recurrent_ppo.eval_episodes import _pin
    from recurrent_ppo.pusht_gym import build_vec_env

    want = [np.array([120.0, 130.0, 240.0, 250.0, 0.3]),
            np.array([300.0, 310.0, 180.0, 190.0, 1.1])]
    venv = build_vec_env(obs_type="state", n_envs=2, seed=0, use_subproc=False)
    try:
        _pin(venv, want)                      # raises if it did not land
        venv.reset()
        got = [np.array([*e.env.agent.position, *e.env.block.position, e.env.block.angle])
               for e in (venv.envs[0].unwrapped, venv.envs[1].unwrapped)]
        for i, (w, g) in enumerate(zip(want, got)):
            assert np.allclose(w, g, atol=1e-6), f"env {i}: pinned {w}, reset to {g}"
    finally:
        venv.close()


def test_manifest_states_match_the_offline_runners_episodes():
    """The RL arms must resolve the SAME episodes the diffusion-policy arms roll out.

    `states_from_manifest` names the manifest directly because an RL arm has no hydra cfg, but
    it must agree episode-for-episode with what `get_episode_init_states` gives the offline
    runner off the same manifest -- otherwise 'same episodes' is a claim, not a fact.
    """
    from diffusion_policy.common.replay_buffer import ReplayBuffer
    from diffusion_policy.dataset.pusht_image_dataset import (
        get_episode_init_states, load_split_manifest, masks_from_manifest)
    from recurrent_ppo.config import DEMO_ZARR
    from recurrent_ppo.eval_episodes import states_from_manifest

    split_file = "diffusion_policy/config/splits/pusht_seed42_train30.json"
    states, idxs = states_from_manifest(split_file, "test")
    assert len(states) == 50 and len(idxs) == 50

    rb = ReplayBuffer.copy_from_path(DEMO_ZARR, keys=["agent_pos", "block_pos"])
    ends = np.asarray(rb.episode_ends[:])
    manifest = load_split_manifest(split_file, episode_ends=ends)
    _, _, test_mask = masks_from_manifest(manifest, len(ends))
    assert np.array_equal(idxs, np.nonzero(test_mask)[0])
    assert np.allclose(states, get_episode_init_states(rb, test_mask))


# ------------------------------------------------------------------ BC on the PPO policy
def test_bc_batching_matches_sb3s_sequence_layout():
    """`_pad_batch` must produce what `_process_sequence` reshapes, or the LSTM sees nonsense.

    SB3 does a plain `features.reshape((n_seq, -1, input_size))`, taking n_seq from the LSTM
    state's batch dimension. So the flattened order has to be episode-major and every sequence
    in a batch the same length -- neither of which raises if you get it wrong. The check is that
    a per-episode forward and a batched forward agree on the valid steps.
    """
    from sb3_contrib.common.recurrent.type_aliases import RNNStates

    from recurrent_ppo.arch import ARCHS
    from recurrent_ppo.bc import _pad_batch
    from recurrent_ppo.config import DEFAULTS
    from recurrent_ppo.pusht_gym import build_vec_env, env_kwargs_from

    cfg = dict(DEFAULTS, obs="state", n_stack=1, corrupt_obs=False, num_envs=2, device="cpu")
    cfg["delta_scale"] = 61.0
    venv = build_vec_env(obs_type="state", n_envs=2, seed=0, use_subproc=False,
                         **env_kwargs_from(cfg, "state"))
    try:
        agent = ARCHS["lstm"].build(venv, cfg, log_dir=None)
        policy = agent.policy
        policy.set_training_mode(False)

        rng = np.random.default_rng(0)
        d = venv.observation_space.shape[0]
        seqs = [(rng.standard_normal((7, d)).astype(np.float32),
                 rng.uniform(-1, 1, (7, 2)).astype(np.float32)),
                (rng.standard_normal((4, d)).astype(np.float32),     # shorter: gets padded
                 rng.uniform(-1, 1, (4, 2)).astype(np.float32))]

        obs, act, starts, valid, zeros = _pad_batch(policy, seqs, agent.device)
        assert obs.shape[0] == 2 * 7 and valid.sum() == 7 + 4, "padding or masking is wrong"
        with th.no_grad():
            _, batched, _ = policy.evaluate_actions(obs, act, RNNStates(zeros, zeros), starts)

        # the same two episodes, one at a time, each from its own zero state
        for i, (o, a) in enumerate(seqs):
            n = len(a)
            shape = (policy.lstm_actor.num_layers, 1, policy.lstm_actor.hidden_size)
            z = lambda: (th.zeros(shape), th.zeros(shape))          # noqa: E731
            with th.no_grad():
                _, alone, _ = policy.evaluate_actions(
                    th.as_tensor(o), th.as_tensor(a), RNNStates(z(), z()),
                    th.as_tensor(np.r_[1.0, np.zeros(n - 1)], dtype=th.float32))
            got = batched[i * 7:i * 7 + n]
            assert th.allclose(got, alone, atol=1e-4), \
                f"episode {i}: batched log-probs differ from the per-episode forward"
    finally:
        venv.close()
