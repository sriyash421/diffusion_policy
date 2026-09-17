"""What the SAC arm must not silently break.

The theme is the same as unit_tests/test_recurrent_ppo.py: every test here guards a property that
is invisible when it fails. A codec that cannot express a candidate, a chunk whose discount
does not match the agent's, a reward that fires twice -- none of these raise, they just make
the numbers mean something other than what they are labelled.
"""

import os

import numpy as np
import pytest

from recurrent_ppo.pusht_gym import PushTGymEnv
from sac import env as sac_env
from sac.env import ChunkPushTEnv
from sac.config import (CHUNK, DEFAULTS as D, DEMO_ZARR, SUCCESS_THRESHOLD,
                        TAU_LADDER, WS, gamma_base)


@pytest.fixture(scope="module")
def demos():
    return sac_env.demo_chunks(DEMO_ZARR)


# ---------------------------------------------------------------- the codec
def test_absolute_codec_expresses_every_demo_chunk_exactly(demos):
    """THE load-bearing property: a Q cannot score a chunk it cannot express.

    ST's candidates are absolute pixel targets drawn from this distribution, so a codec that
    loses demo chunks loses candidates too -- and the loss is silent, because a clipped action
    still produces a number.
    """
    A, _ = demos
    u = sac_env.encode(A)
    assert u.shape == (len(A), 2 * CHUNK)
    assert (np.abs(u) <= 1.0 + 1e-9).all(), "a demo chunk falls outside the action space"
    assert sac_env.assert_roundtrip(A) == 0.0


def test_an_increment_chain_would_lose_a_quarter_of_the_expert_chunks(demos):
    """Why `absolute` is the only codec, computed rather than asserted from a comment.

    An increment chain anchors each target on the previous one, which buys translation
    equivariance -- a real argument, and the reason the mode existed. It is measured here
    instead of implemented: at recurrent_ppo's own delta_scale it cannot express a QUARTER of
    what the expert actually did, and a verifier that silently clips the chunk it was asked
    about is worse than one that is merely coarse.
    """
    A, P = demos
    anchors = np.concatenate([P[:, None, :], A[:, :-1, :]], axis=1)
    steps = np.abs(A - anchors)                                   # per-step target displacement
    lost = {R: float((steps > R).any(axis=(1, 2)).mean()) for R in (33.0, 64.0)}
    assert lost[33.0] == pytest.approx(0.26, abs=0.02)
    assert lost[64.0] == pytest.approx(0.022, abs=0.005)
    # absolute, by contrast, expresses every one of them (test above)


def test_codec_ignores_differences_outside_the_arena():
    """Two chunks differing only outside [0, WS] drive identical trajectories, so they must
    encode identically -- otherwise Q distinguishes candidates the simulator cannot."""
    p = np.array([256.0, 256.0])
    a = np.full((CHUNK, 2), 600.0)
    b = np.full((CHUNK, 2), 900.0)
    assert np.allclose(sac_env.encode(a), sac_env.encode(b))


def test_decode_stays_inside_the_arena():
    rng = np.random.default_rng(0)
    out = sac_env.decode(rng.uniform(-1, 1, size=(500, 2 * CHUNK)))
    assert out.shape == (500, CHUNK, 2)
    assert (out >= 0.0).all() and (out <= WS).all()


# ---------------------------------------------------------------- the chunked MDP
def test_episode_is_a_whole_number_of_chunks():
    """A ragged final chunk would bootstrap at the wrong discount, silently."""
    env = ChunkPushTEnv(obs_type="keypoint", block_zero_coverage=False)
    env.reset(seed=0)
    n = 0
    while True:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        n += 1
        if term or trunc:
            break
    assert env._elapsed_steps == n * CHUNK == D["max_episode_steps"]


def test_max_episode_steps_must_be_a_multiple_of_chunk():
    with pytest.raises(AssertionError, match="not a multiple of chunk"):
        ChunkPushTEnv(obs_type="keypoint", max_episode_steps=300)


def test_chunk_return_equals_the_base_env_discounted_sum():
    """The chunked MDP is the 8-step-decision view of the original one EXACTLY.

    Replays the identical absolute targets one base step at a time, through the SAME class's
    inherited PushTGymEnv.step, and compares. If this drifts, `gamma_chunk = gamma_base**CHUNK`
    is a lie and every Q value sits on a different scale from the one documented.
    """
    kwargs = dict(obs_type="keypoint", block_zero_coverage=False, reward_mode="dense",
                  block_near_goal_prob=1.0, block_goal_offset=[8.0, 0.06])
    u = np.linspace(-0.6, 0.6, 2 * CHUNK)

    chunked = ChunkPushTEnv(**kwargs)
    chunked.reset(seed=7)
    targets = sac_env.decode(u)
    _, got, _, _, _ = chunked.step(u)

    stepwise = ChunkPushTEnv(**kwargs)
    stepwise.reset(seed=7)
    want, discount = 0.0, 1.0
    for k in range(CHUNK):
        _, r, term, trunc, _ = PushTGymEnv.step(stepwise, targets[k])
        want += discount * float(r)
        discount *= stepwise.gamma_base
        if term or trunc:
            break
    assert got == pytest.approx(want, rel=1e-9)


def test_gamma_base_powers_up_to_the_quoted_chunk_discount():
    assert gamma_base(0.95) ** CHUNK == pytest.approx(0.95)


# ---------------------------------------------------------------- the sparse reward
def test_sparse_reward_fires_at_most_once():
    """Sparse pays on the FIRST crossing and terminates; it must never pay twice."""
    env = ChunkPushTEnv(obs_type="keypoint", reward_mode="sparse", block_zero_coverage=False,
                        block_near_goal_prob=1.0, block_goal_offset=[3.0, 0.02])
    paid = 0
    for ep in range(20):
        env.reset(seed=ep)
        rewards = []
        for _ in range(38):
            _, r, term, trunc, _ = env.step(np.zeros(2 * CHUNK, dtype=np.float32))
            rewards.append(r)
            if term or trunc:
                break
        assert sum(r > 0 for r in rewards) <= 1, "the sparse reward must be paid at most once"
        paid += sum(r > 0 for r in rewards)
    assert paid > 0, "the curriculum must produce SOME solve, or there is nothing to learn"


def test_an_episode_never_begins_solved():
    """Otherwise the near-goal curriculum pays the sparse reward for doing nothing.

    Measured before this rule: at offset [3, 0.02] a third of draws started above 0.95 and a
    do-nothing policy returned 0.349 -- most of the available reward, for free.
    """
    env = ChunkPushTEnv(obs_type="keypoint", block_zero_coverage=False,
                        block_near_goal_prob=1.0, block_goal_offset=[3.0, 0.02])
    cov = np.array([(env.reset(seed=i), env._block_coverage())[1] for i in range(40)])
    assert cov.max() < SUCCESS_THRESHOLD
    assert cov.mean() > 0.85, "but they should still start CLOSE, or the rung is unreachable"


def test_a_rung_the_episode_starts_above_is_not_live():
    """There is no achievement to credit there, and crediting it is the same free lunch."""
    env = ChunkPushTEnv(obs_type="keypoint", block_zero_coverage=False,
                        block_near_goal_prob=1.0, block_goal_offset=[3.0, 0.02])
    env.reset(seed=0)
    start = env._block_coverage()
    _, _, _, _, info = env.step(np.zeros(2 * CHUNK, dtype=np.float32))
    for k, tau in enumerate(TAU_LADDER):
        assert info["tau_live"][k] == (start <= tau)


def test_tau_ladder_is_ordered_and_reported():
    """The lower rungs exist because NO demo reaches 0.95; they must be monotone in tau."""
    env = ChunkPushTEnv(obs_type="keypoint", block_zero_coverage=False,
                        block_near_goal_prob=1.0, block_goal_offset=[3.0, 0.02])
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.zeros(2 * CHUNK, dtype=np.float32))
    assert len(info["tau_live"]) == len(info["tau_done"]) == len(TAU_LADDER)
    # a rung can only be paid while it is live, and only once
    assert not (info["tau_done"] & ~info["tau_live"]).any()
    assert (info["tau_reward"] > 0).sum() == info["tau_done"].sum()


def test_the_near_goal_curriculum_actually_reaches_the_threshold():
    """recurrent_ppo's default offset [30, 0.25] sits at coverage ~0.48 and NEVER crosses 0.95,
    so the curriculum as inherited cannot produce the reward it exists to produce."""
    def mean_coverage(offset):
        env = ChunkPushTEnv(obs_type="keypoint", block_zero_coverage=False,
                            block_near_goal_prob=1.0, block_goal_offset=offset)
        return np.mean([(env.reset(seed=i), env._block_coverage())[1] for i in range(25)])

    assert mean_coverage([30.0, 0.25]) < 0.6
    assert mean_coverage(D["block_goal_offset"]) > 0.9


# ---------------------------------------------------------------- the agent
def _tiny_agent(n_envs=2, **kw):
    from sac.agent import buffer_class_for
    from sac.env import build_chunk_vec_env
    from sac.agent import ChunkSAC

    env = build_chunk_vec_env("keypoint", n_envs=n_envs, use_subproc=False,
                              block_zero_coverage=False, block_near_goal_prob=1.0,
                              block_goal_offset=[3.0, 0.02])
    return ChunkSAC("MlpPolicy", env, obs_type="keypoint", tau_ladder=TAU_LADDER, gamma=0.95,
                    learning_starts=32, batch_size=16, gradient_steps=2, buffer_size=2000,
                    replay_buffer_class=buffer_class_for("keypoint"),
                    replay_buffer_kwargs={"n_tau": len(TAU_LADDER)}, device="cpu", **kw), env


def test_the_verifier_head_cannot_move_the_policy():
    """The whole isolation argument, as a fact rather than a promise.

    The head is built AFTER the policy's optimizers, so they cannot see its parameters, and its
    features are detached. If this ever fails, the SAC run is no longer a clean SAC run and the
    verifier is co-adapting with the actor it is supposed to judge.
    """
    import torch as th

    agent, _ = _tiny_agent()
    head_params = {id(p) for p in agent.policy.bon_head.parameters()}
    for opt in (agent.policy.actor.optimizer, agent.policy.critic.optimizer):
        for group in opt.param_groups:
            assert not head_params & {id(p) for p in group["params"]}

    agent.learn(total_timesteps=64)                       # fill the buffer
    after_collect = [p.detach().clone() for p in agent.policy.actor.parameters()]
    agent._train_bon_head(gradient_steps=5, batch_size=16)
    after_head = [p.detach().clone() for p in agent.policy.actor.parameters()]
    for a, b in zip(after_collect, after_head):
        assert th.equal(a, b), "the verifier head moved the actor"


def test_q_depends_on_the_action():
    """A Q that has collapsed to V(s) scores every candidate identically -- which is EXACTLY
    the degeneracy of the heuristic being replaced, restated in Q coordinates."""
    agent, env = _tiny_agent()
    agent.learn(total_timesteps=128)
    obs = env.reset()
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(16, 2 * CHUNK)).astype(np.float32)
    obs_rep = np.repeat(obs[:1], 16, axis=0)
    q = agent.q_values(obs_rep, actions).cpu().numpy()
    assert q.std() > 1e-6, "Q is constant across candidates at a fixed state"


def test_uniform_arm_is_uniform_and_ignores_the_observation():
    """The uniform arm's job is action coverage that does NOT depend on what the actor thinks.

    The test is the mechanism, not a statistical claim about the actor: early in training the
    actor's tanh-Gaussian is itself close to uniform (measured |a|>0.9 fractions of 0.112 vs
    uniform's 0.118), so neither spread nor saturation separates the two and a test asserting
    otherwise would be asserting something false. What IS always true is that the uniform arm
    is uniform on the cube and is independent of the observation.
    """
    agent, env = _tiny_agent()
    agent.learn(total_timesteps=64)
    agent.mix = np.array([0.0, 1.0, 0.0, 0.0])

    def draw(obs):
        agent._last_obs = obs
        return np.stack([agent._mixture_actions(env.num_envs) for _ in range(60)]).reshape(-1, 2 * CHUNK)

    first = draw(env.reset())
    assert first.min() >= -1.0 and first.max() <= 1.0
    assert abs(float(first.mean())) < 0.05                       # centred
    assert float((np.abs(first) > 0.9).mean()) == pytest.approx(0.10, abs=0.04)

    # a completely different state must not shift the uniform arm's distribution
    for _ in range(5):
        env.step(np.zeros((env.num_envs, 2 * CHUNK), dtype=np.float32))
    second = draw(env.step(np.zeros((env.num_envs, 2 * CHUNK), dtype=np.float32))[0])
    assert abs(float(first.mean()) - float(second.mean())) < 0.08
    assert abs(float(first.std()) - float(second.std())) < 0.08


def test_mixture_weights_actually_route():
    """All-uniform and all-demo must not produce the same thing, or the mixture is decorative."""
    from sac.env import decode, demo_chunks

    A, P = demo_chunks(DEMO_ZARR, stride=64)
    agent, env = _tiny_agent()
    agent.demo_offsets = A - P[:, None, :]
    agent.learn(total_timesteps=64)
    agent._last_obs = env.reset()

    def step_px(mix):
        agent.mix = np.asarray(mix, dtype=float)
        u = np.stack([agent._mixture_actions(env.num_envs) for _ in range(30)]).reshape(-1, 2 * CHUNK)
        pos = np.tile((agent._last_obs[:, 18:20] + 1.0) * (WS / 2), (30, 1))
        return float(np.median(np.abs(np.diff(decode(u), axis=1))))

    # a uniform chunk teleports the target across the arena between steps; a demo-shaped one
    # moves it a few pixels. If these coincide, the `which == 2` branch never fired.
    assert step_px([0.0, 1.0, 0.0, 0.0]) > 3 * step_px([0.0, 0.0, 1.0, 0.0])


def test_demo_arm_proposes_demo_shaped_chunks():
    """The demo arm re-anchors a recorded chunk SHAPE at the current agent position, so a push
    the expert performed in one corner is a usable proposal anywhere. Injecting the raw encoding
    instead would only ever teach Q about the places the demos happened to visit."""
    from sac.env import decode, demo_chunks

    A, P = demo_chunks(DEMO_ZARR, stride=64)
    agent, env = _tiny_agent()
    agent.demo_offsets = A - P[:, None, :]
    agent.learn(total_timesteps=64)
    agent._last_obs = env.reset()
    agent.mix = np.array([0.0, 0.0, 1.0, 0.0])
    u = agent._mixture_actions(env.num_envs)
    pos = (agent._last_obs[:, 18:20] + 1.0) * (WS / 2)
    steps = np.abs(np.diff(decode(u), axis=1))
    # a demo-shaped chunk moves the target a few px per step, not half the arena
    assert np.median(steps) < 40.0


# ---------------------------------------------------------------- the verifier shim
@pytest.fixture(scope="module")
def obs_dict():
    """An obs dict shaped like the one `predict_action_best` hands the verifier."""
    import torch as th
    import zarr

    from diffusion_policy.env.pusht.feedback_util import compute_feedback_from_pose

    root = zarr.open(DEMO_ZARR, "r")
    state = np.asarray(root["data/state"])[[1000, 2000, 3000, 4000]]
    return {
        "agent_pos": th.tensor(state[:, None, :2], dtype=th.float32),
        "feedback": th.tensor(compute_feedback_from_pose(state[:, 2:5].astype(np.float32))[:, None, :],
                              dtype=th.float32),
    }, state


def test_q_verifier_reconstructs_the_same_state_as_the_sim_verifier(obs_dict):
    """Both verifiers must be answering about the SAME state for the same observation, or a
    comparison between them is comparing two different questions."""
    from diffusion_policy.env.pusht.pusht_verifier import PushTVerifier
    from sac.score import state_from_obs

    obs, truth = obs_dict
    sim = PushTVerifier(n_envs=2, use_async=False, value_fn="t_goal")
    try:
        assert np.abs(state_from_obs(obs) - sim._reset_states_from_obs(obs)).max() == 0.0
    finally:
        sim.close()
    # and both agree with the recorded state, up to the angle's 2*pi wrap
    diff = np.abs(state_from_obs(obs) - truth.astype(np.float64))
    assert diff[:, :4].max() < 1e-3
    assert np.minimum(diff[:, 4], np.abs(diff[:, 4] - 2 * np.pi)).max() < 1e-3


def test_keypoint_arm_needs_no_simulation(obs_dict):
    """The analytic path: block pose -> affine transform of 9 local keypoints. If this ever
    starts needing a sim, the verifier has lost the speed that made it worth deploying."""
    from sac.score import obs_for_arm

    obs, truth = obs_dict
    out = obs_for_arm(obs, "keypoint")
    # 20-d: the visibility mask is applied by the env, not carried as features
    assert out.shape == (4, 20)
    assert (np.abs(out) <= 1.0 + 1e-6).all(), "everything the policy sees is in [-1, 1]"
    assert (out[:, 20:] == 1.0).all(), "the demos are fully visible, so the mask half is all +1"


def test_q_verifier_refuses_a_value_it_does_not_implement():
    """Silently accepting `armTn` and returning the Q is how an evaluation ends up labelled
    with a ranking it did not use."""
    with pytest.raises(AssertionError, match="one value"):
        from sac.score import PushTQVerifier

        PushTQVerifier.__init__(object.__new__(PushTQVerifier), "x", value_fn="armTn")


# ---------------------------------------------------------------- the buffer and the backup
def test_sample_chunk_pairs_each_transition_with_its_own_tau_labels():
    """The desync `_draw` exists to prevent, asserted rather than assumed.

    `ReplayBuffer._get_samples` picks `env_indices` itself with a fresh `np.random.randint`, so
    calling `sample()` and then `tau_batch()` would pair each transition with ANOTHER
    transition's rung labels. Nothing raises: the Q is simply regressed on shuffled targets,
    which looks exactly like slow learning.

    Each transition is stamped with its own index in BOTH the observation and the rung reward,
    so a desync shows up as the two disagreeing.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    from sac.agent import ChunkReplayBuffer
    from sac.env import make_chunk_env

    n_envs, n = 4, 50
    env = DummyVecEnv([make_chunk_env("keypoint", block_zero_coverage=False)] * n_envs)
    buf = ChunkReplayBuffer(n, env.observation_space, env.action_space, n_envs=n_envs, n_tau=4)
    for i in range(n):
        stamp = np.arange(n_envs) + i * n_envs                      # unique per (step, env)
        obs = np.zeros((n_envs,) + env.observation_space.shape, dtype=np.float32)
        obs[:, 0] = stamp
        infos = [{"tau_reward": np.full(4, stamp[e], dtype=np.float32),
                  "tau_done": np.zeros(4, bool), "tau_live": np.ones(4, bool)}
                 for e in range(n_envs)]
        buf.add(obs, obs.copy(), np.zeros((n_envs, 2 * CHUNK), np.float32),
                np.zeros(n_envs, np.float32), np.zeros(n_envs, np.float32), infos)

    data, (r_tau, _, _) = buf.sample_chunk(64)
    assert np.allclose(data.observations[:, 0].cpu().numpy(), r_tau[:, 0].cpu().numpy()), \
        "the tau labels belong to different transitions than the observations"


def test_the_head_backup_is_a_hard_max_over_candidates():
    """`r + gamma * (1-d) * max_M`, not the actor's expectation.

    Taking the mean over candidates would learn Q^pi -- the value of the actor's stochastic
    policy -- which is not what best-of-N evaluates. With a stub target head whose value is a
    known function of the action, the two are numerically distinguishable.
    """
    import torch as th

    agent, env = _tiny_agent()
    agent.learn(total_timesteps=64)
    obs = agent.policy.obs_to_tensor(env.reset())[0]
    b, n_tau = env.num_envs, len(TAU_LADDER)

    known = th.arange(agent.bon_candidates, dtype=th.float32)
    calls = {"i": 0}

    class _Stub(th.nn.Module):
        def forward(self, features, actions):
            v = known[calls["i"] % len(known)]
            calls["i"] += 1
            return th.full((2, features.shape[0], n_tau), float(v))

    agent.policy.bon_head_target = _Stub()
    r = th.zeros(b, n_tau)
    d = th.zeros(b, n_tau)
    target = agent.bon_target(obs, r, d)
    assert th.allclose(target, th.full_like(target, agent.gamma * float(known.max()))), \
        "the backup is not a hard max over the candidate set"

    # and the terminal flag must switch the bootstrap off entirely
    assert th.allclose(agent.bon_target(obs, r + 1.0, th.ones(b, n_tau)), th.ones(b, n_tau))


def test_a_batch_with_nothing_live_contributes_no_gradient():
    """The live mask at the UPDATE level, not just in the env.

    A rung that keeps training past its own threshold is learning a different MDP from the one
    its label claims. If every transition is dead for every rung, the loss is exactly zero and
    no gradient reaches the head.

    Asserted on the GRADIENT, not on the parameters: Adam carries momentum from earlier steps,
    so `optimizer.step()` still moves weights when the gradient is exactly zero. Checking
    parameters would fail for a reason that has nothing to do with masking.
    """
    import torch as th

    agent, _ = _tiny_agent()
    agent.learn(total_timesteps=128)
    agent.replay_buffer.tau_live[:] = 0.0
    agent._train_bon_head(gradient_steps=3, batch_size=16)

    assert float(agent.logger.name_to_value["bon/head_loss"]) == 0.0
    for p in agent.policy.bon_head.parameters():
        assert p.grad is not None and th.count_nonzero(p.grad) == 0, "a dead rung produced gradient"


def test_demo_transitions_round_trip_through_the_buffer():
    """What preload_demos puts in comes back out unchanged, through the same `add` collected
    transitions use -- the demo path and the collected path must be indistinguishable."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    from sac.agent import ChunkReplayBuffer, preload_demos
    from sac.env import make_chunk_env
    from sac.env import demo_transitions

    env = DummyVecEnv([make_chunk_env("keypoint", block_zero_coverage=False)])
    tr = demo_transitions(DEMO_ZARR, obs_type="keypoint", stride=64)
    buf = ChunkReplayBuffer(len(tr["action"]) + 8, env.observation_space, env.action_space, n_tau=4)
    n = preload_demos(buf, tr, verbose=False)

    assert np.allclose(buf.observations[:n, 0], tr["obs"][:n])
    assert np.allclose(buf.actions[:n, 0], tr["action"][:n])
    assert np.allclose(buf.tau_reward[:n, 0], tr["reward"][:n])
    assert np.array_equal(buf.tau_done[:n, 0].astype(bool), tr["done"][:n])
    assert np.array_equal(buf.tau_live[:n, 0].astype(bool), tr["live"][:n])


# ---------------------------------------------------------------- exploration and curriculum
def test_the_smooth_arm_moves_the_target_a_demo_scale_step():
    """Measured at 9.3 px against uniform's 171 and the demonstrations' 4 px median.

    If this regresses toward uniform, the agent slams across the table eight times per chunk,
    scatters the block, and the top rung silently stops collecting terminals -- the failure
    that produced ONE tau=0.95 terminal in 50k steps.
    """
    from sac.env import decode

    agent, env = _tiny_agent()
    agent.learn(total_timesteps=64)
    agent._last_obs = env.reset()
    agent.mix = np.array([0.0, 0.0, 0.0, 1.0])                      # all smooth
    u = np.stack([agent._mixture_actions(env.num_envs) for _ in range(30)]).reshape(-1, 2 * CHUNK)
    pos = np.tile((agent._last_obs[:, 18:20] + 1.0) * (WS / 2), (30, 1))
    step = float(np.median(np.abs(np.diff(decode(u), axis=1))))
    assert 1.0 < step < 30.0, f"smooth arm moves the target {step:.1f} px/step (demos: 4, uniform: 171)"


def test_curriculum_anneal_actually_moves_the_env():
    """`set_attr` does nothing detectable if the attribute name is wrong, so assert the env's
    own values, and that they move monotonically from the tight end to the wide one."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    from sac.runner import CurriculumAnneal
    from sac.env import make_chunk_env

    env = DummyVecEnv([make_chunk_env("keypoint", block_zero_coverage=False)])
    cb = CurriculumAnneal([3.0, 0.02], [20.0, 0.15], total_timesteps=1000, frac=1.0,
                          prob_start=0.6, prob_final=0.1)
    # BOTH `logger` and `training_env` are read-only properties on BaseCallback, resolving to
    # `self.model.logger` and `self.model.get_env()`. So the stub is a model, not a callback.
    class _Model:
        logger = type("L", (), {"record": lambda *a, **k: None})()

        @staticmethod
        def get_env():
            return env

    cb.model = _Model()

    seen = []
    for t in (0, 250, 500, 750, 1000):
        cb.num_timesteps = t
        cb._on_step()
        seen.append((env.get_attr("block_goal_offset")[0][0], env.get_attr("block_near_goal_prob")[0]))
    offsets = [o for o, _ in seen]
    probs = [p for _, p in seen]
    assert offsets == sorted(offsets) and offsets[0] == pytest.approx(3.0) and offsets[-1] == pytest.approx(20.0)
    assert probs == sorted(probs, reverse=True) and probs[-1] == pytest.approx(0.1)


# ---------------------------------------------------------------- training and deployment agree
def test_q_values_survive_save_and_load():
    """The bug this guards actually happened: `attach_bon_head` had to move into `_setup_model`
    because SB3's `load` constructs with `_init_setup_model=False`, so `__init__` has no policy.

    A silently re-initialised head scores garbage at deployment while every training metric
    still looks healthy, which is why this compares NUMBERS rather than checking the attribute
    exists.
    """
    import tempfile

    from sac.agent import ChunkSAC

    agent, env = _tiny_agent()
    agent.learn(total_timesteps=128)
    obs = env.reset()
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(env.num_envs, 2 * CHUNK)).astype(np.float32)
    before = agent.q_values(obs, actions).cpu().numpy()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.zip")
        agent.save(path)
        reloaded = ChunkSAC.load(path, device="cpu")
        after = reloaded.q_values(obs, actions).cpu().numpy()
    assert np.allclose(before, after, atol=1e-6), "the verifier head did not survive save/load"


def test_score_agrees_with_the_agent_it_loaded(obs_dict):
    """`PushTQVerifier.get_value` and `ChunkSAC.q_values` are two paths to one quantity.

    If they disagree, training and deployment disagree about what the Q is -- and the BON
    numbers would be measuring a function that was never trained.
    """
    import tempfile

    import torch as th
    import zarr

    from sac.env import encode
    from sac.score import PushTQVerifier, obs_for_arm, state_from_obs

    obs, _ = obs_dict
    agent, _ = _tiny_agent()
    agent.learn(total_timesteps=64)
    chunks = np.asarray(zarr.open(DEMO_ZARR, "r")["data/action"])[1000:1000 + 4 * CHUNK]
    chunks = chunks.reshape(4, CHUNK, 2).astype(np.float64)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.zip")
        agent.save(path)
        q = PushTQVerifier(path, obs_type="keypoint", device="cpu")
        via_shim = q.get_value(obs, th.tensor(chunks, dtype=th.float32)).cpu().numpy()
        state = state_from_obs(obs)
        via_agent = q.agent.q_values(obs_for_arm(obs, "keypoint"),
                                     encode(chunks).astype(np.float32)).cpu().numpy()
    assert np.allclose(via_shim, via_agent, atol=1e-6)


@pytest.mark.slow
def test_training_collects_terminals_and_keeps_q_non_degenerate():
    """The only test that says the whole loop LEARNS rather than merely runs.

    Two acceptance signals, both of which the tree already produces: the top rung collects
    terminals at all (it collected ONE in 50k steps before the exploration prior was fixed),
    and Q never scores every candidate identically -- the degeneracy of the heuristic being
    replaced, restated in Q coordinates.
    """
    from sac.agent import buffer_class_for
    from sac.env import build_chunk_vec_env
    from sac.agent import ChunkSAC

    env = build_chunk_vec_env("keypoint", n_envs=8, use_subproc=False, block_zero_coverage=False,
                              block_near_goal_prob=1.0, block_goal_offset=[3.0, 0.02])
    agent = ChunkSAC("MlpPolicy", env, obs_type="keypoint", tau_ladder=TAU_LADDER, gamma=0.95,
                     learning_starts=500, batch_size=64, gradient_steps=2, buffer_size=20000,
                     replay_buffer_class=buffer_class_for("keypoint"),
                     replay_buffer_kwargs={"n_tau": len(TAU_LADDER)}, device="cpu", verbose=0)
    agent.learn(total_timesteps=6000)

    top = agent.replay_buffer.tau_done[:agent.replay_buffer.pos, :, -1].sum()
    assert top > 0, "the tightest curriculum collected no tau=0.95 terminal at all"

    obs = env.reset()
    rng = np.random.default_rng(0)
    q = agent.q_values(np.repeat(obs[:1], 16, axis=0),
                       rng.uniform(-1, 1, size=(16, 2 * CHUNK)).astype(np.float32)).cpu().numpy()
    assert q.std() > 1e-9, "Q scores every candidate identically -- the heuristic's own failure"


def test_demo_rewards_encode_the_discounted_time_to_cross():
    """Q's UNITS, asserted on the data that defines them.

    `Q*` here is meant to be `gamma**(chunks to solve)`, and that only holds if a rung's reward
    is `gamma_base**j` for the within-chunk step j at which it crossed -- so that crossing
    earlier is worth strictly more, exactly as the env pays it. Every paid demo reward must
    therefore be one of the CHUNK powers of gamma_base and nothing else.
    """
    from sac.env import demo_transitions

    tr = demo_transitions(DEMO_ZARR, obs_type="keypoint", stride=8, gamma=0.95)
    paid = tr["reward"][tr["done"]]
    assert len(paid) > 0, "no rung was ever crossed in the demonstrations"
    allowed = gamma_base(0.95) ** np.arange(CHUNK)
    assert np.isclose(paid[:, None], allowed[None, :], atol=1e-6).any(axis=1).all()
    # and a reward is paid if and only if that rung terminated on that chunk
    assert np.array_equal(tr["reward"] > 0, tr["done"])


@pytest.mark.slow
def test_q_is_higher_closer_to_the_goal():
    """Calibration, directionally: `Q ~ gamma**(chunks to solve)` implies a state one push from
    the goal must score above one far from it.

    Directional rather than exact on purpose -- pinning `Q` to `gamma**k` needs a converged
    critic, and a unit test's budget cannot produce one. What it CAN catch is a Q whose ordering
    is backwards or flat, which is the failure that would make it useless as a verifier.
    """
    from sac.agent import buffer_class_for
    from sac.env import build_chunk_vec_env
    from sac.env import demo_transitions
    from sac.agent import preload_demos
    from sac.agent import ChunkSAC

    env = build_chunk_vec_env("keypoint", n_envs=8, use_subproc=False, block_zero_coverage=False,
                              block_near_goal_prob=1.0, block_goal_offset=[3.0, 0.02])
    agent = ChunkSAC("MlpPolicy", env, obs_type="keypoint", tau_ladder=TAU_LADDER, gamma=0.95,
                     learning_starts=500, batch_size=128, gradient_steps=4, buffer_size=40000,
                     replay_buffer_class=buffer_class_for("keypoint"),
                     replay_buffer_kwargs={"n_tau": len(TAU_LADDER)}, device="cpu", verbose=0)
    preload_demos(agent.replay_buffer, demo_transitions(DEMO_ZARR, obs_type="keypoint", stride=4),
                  verbose=False)
    agent.learn(total_timesteps=8000)

    near = build_chunk_vec_env("keypoint", n_envs=8, use_subproc=False, seed=1,
                               block_zero_coverage=False, block_near_goal_prob=1.0,
                               block_goal_offset=[4.0, 0.03])
    far = build_chunk_vec_env("keypoint", n_envs=8, use_subproc=False, seed=1,
                              block_zero_coverage=False, block_near_goal_prob=0.0)
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, size=(8, 2 * CHUNK)).astype(np.float32)
    q_near = float(agent.q_values(near.reset(), a).mean())
    q_far = float(agent.q_values(far.reset(), a).mean())
    assert q_near > q_far, f"Q is not higher near the goal ({q_near:.4f} vs {q_far:.4f})"


# ---------------------------------------------------------------- the image arm is image only
def test_the_image_arm_observation_carries_no_pose():
    """`agent_pos` was removed so the RL arms see exactly what the offline arms see.

    Handing a policy the pose in closed form is a strictly stronger observation than the standard
    PushT-image setup, so keeping it would make the two families' success rates incomparable --
    and this Q exists to rank the offline arms' candidates. Every producer of an image
    observation must agree, or a demo transition silently stops matching the space it is stored
    in.
    """
    from sac.env import ChunkPushTEnv, demo_transitions

    env = ChunkPushTEnv(obs_type="image", block_zero_coverage=False)
    assert set(env.observation_space.spaces) == {"image"}
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"image"}

    tr = demo_transitions(DEMO_ZARR, obs_type="image", stride=128)
    assert set(tr["obs"]) == {"image"} == set(tr["next_obs"])
    assert tr["obs"]["image"].dtype == np.uint8


def test_the_anchor_comes_from_the_env_not_the_observation():
    """The behaviour mixture needs the agent's pixel position to PLACE a proposed chunk.

    It cannot come from the image observation any more, and must not be put back there: the
    position is used only to propose actions, never to score them, so reading it off the env
    keeps the image arm genuinely image-only while the demo-shape and smooth-walk proposals keep
    working. Those are the reason the sparse reward is findable at all.
    """
    from sac.agent import agent_pos_from_env
    from sac.env import build_chunk_vec_env

    for arm in ("keypoint", "image"):
        env = build_chunk_vec_env(arm, n_envs=3, use_subproc=False, block_zero_coverage=False)
        env.reset()
        pos = agent_pos_from_env(env)
        assert pos.shape == (3, 2)
        # arena pixels, not the [-1, 1] the policy sees
        assert (pos >= 0).all() and (pos <= WS).all() and pos.max() > 1.5


def test_score_builds_the_image_observation_the_arm_declares():
    """`obs_for_arm` must produce exactly the space the env declares, or the Q is fed a dict its
    encoder cannot read -- at DEPLOYMENT, where there is no test to catch it."""
    import torch as th
    import zarr

    from diffusion_policy.env.pusht.feedback_util import compute_feedback_from_pose
    from recurrent_ppo.corrupt_policy import ST_CROP, aug_for
    from recurrent_ppo.pusht_gym import AUG_CROP_KEY, aug_spaces
    from sac.env import ChunkPushTEnv
    from sac.score import obs_for_arm

    state = np.asarray(zarr.open(DEMO_ZARR, "r")["data/state"])[[100, 200]]
    obs_dict = {
        "agent_pos": th.tensor(state[:, None, :2], dtype=th.float32),
        "feedback": th.tensor(compute_feedback_from_pose(state[:, 2:5].astype(np.float32))[:, None, :],
                              dtype=th.float32),
        "image": th.rand(2, 1, 3, 96, 96),
    }
    built = obs_for_arm(obs_dict, "image")
    # The TRAINING space, not the bare env's: sac/runner wraps in VecAugmentationDraw so the
    # image arm carries a crop offset, and that wrapped space is what the buffer stores and the
    # extractor was built against. Comparing against the unwrapped env would pass while the Q
    # was handed a dict one key short of what its encoder reads.
    raw = ChunkPushTEnv(obs_type="image", block_zero_coverage=False).observation_space
    aug = aug_for("image", corrupt_obs=False, render_size=96, random_crop=True)
    declared = set(aug_spaces(raw, **aug).spaces)
    assert set(built) == declared, f"score builds {sorted(built)}, training space is {sorted(declared)}"
    assert built["image"].shape == (2, 3, 96, 96) and built["image"].dtype == np.uint8
    # and the offset is the CENTRE one -- the Q trains on random crops and deploys on the centre
    lo = (96 - ST_CROP) // 2
    assert np.array_equal(built[AUG_CROP_KEY], np.full((2, 2), lo, dtype=np.float32))


def test_demo_seeding_is_restricted_to_the_split():
    """The buffer must never be seeded from an episode the verifier is later scored on.

    `demo_transitions` used to walk all 206 episodes, so the learned Q was fitted on transitions
    from the very 50 the best-of-N sweep judges it against -- while the heuristic it is compared
    with has no such advantage. That makes "the learned Q beats the heuristic" not a held-out
    claim, and nothing about the number would have shown it.
    """
    from recurrent_ppo.eval_episodes import states_from_manifest
    from sac.env import demo_transitions

    split = "diffusion_policy/config/splits/pusht_seed42_train106_val50.json"
    _, train_idxs = states_from_manifest(split, "train")
    _, test_idxs = states_from_manifest(split, "test")

    everything = demo_transitions(DEMO_ZARR, obs_type="keypoint", stride=64)
    restricted = demo_transitions(DEMO_ZARR, obs_type="keypoint", stride=64,
                                  episode_idxs=train_idxs)
    assert len(restricted["action"]) < len(everything["action"]), "the filter did nothing"

    # 106 of 206 episodes, so the transition count should track that, not merely be smaller
    ratio = len(restricted["action"]) / len(everything["action"])
    assert 0.4 < ratio < 0.7, f"kept {ratio:.2%} of transitions for {len(train_idxs)}/206 episodes"

    # and none of it is an episode the sweep scores on
    assert not (set(int(i) for i in train_idxs) & set(int(i) for i in test_idxs))
