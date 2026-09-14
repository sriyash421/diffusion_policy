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
            vec = obs["agent_pos"] if obs_type == "image" else obs
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


def test_feedforward_q_head_cannot_move_the_policy():
    """The same separate-optimizer guarantee, for the non-recurrent policy."""
    policy = policy_for("keypoint", recurrent=False)(
        gym.spaces.Box(-1, 1, (160,)), gym.spaces.Box(-1, 1, (2,)), lambda _: 3e-4,
        **features_extractor_kwargs("keypoint", corrupt_obs=True, t_max=200))
    q_params = {id(p) for p in policy.q_net.parameters()}
    policy_params = {id(p) for group in policy.optimizer.param_groups for p in group["params"]}
    assert q_params and not (q_params & policy_params)
    assert hasattr(policy, "q_optimizer")


@pytest.mark.parametrize("space", [
    gym.spaces.Box(-1, 1, (40,), dtype=np.float32),                       # the keypoint obs
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


@pytest.mark.parametrize("n_stack,expected", [(1, 40), (4, 160)])
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

    env = PushTGymEnv(obs_type=obs_type, max_episode_steps=20)
    aug = aug_for(obs_type, corrupt, render_size=96)
    env = AugmentationDraw(env, **aug)
    obs, _ = env.reset(seed=0)

    policy = policy_for(obs_type, recurrent=True, corrupt_obs=corrupt)(
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
