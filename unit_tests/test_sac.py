"""What the SAC arm must not silently break.

The theme is the same as tests/test_recurrent_ppo.py: every test here guards a property that
is invisible when it fails. A codec that cannot express a candidate, a chunk whose discount
does not match the agent's, a reward that fires twice -- none of these raise, they just make
the numbers mean something other than what they are labelled.
"""

import numpy as np
import pytest

from recurrent_ppo.pusht_gym import PushTGymEnv
from sac import chunk_codec as cc
from sac.chunk_env import ChunkPushTEnv
from sac.config import (CHUNK, DEFAULTS as D, DEMO_ZARR, SUCCESS_THRESHOLD,
                        TAU_LADDER, WS, gamma_base)


@pytest.fixture(scope="module")
def demos():
    return cc.demo_chunks(DEMO_ZARR)


# ---------------------------------------------------------------- the codec
def test_absolute_codec_expresses_every_demo_chunk_exactly(demos):
    """THE load-bearing property: a Q cannot score a chunk it cannot express.

    ST's candidates are absolute pixel targets drawn from this distribution, so a codec that
    loses demo chunks loses candidates too -- and the loss is silent, because a clipped action
    still produces a number.
    """
    A, P = demos
    chunks, comps = cc.representable(A, P, mode="absolute")
    assert chunks == 1.0 and comps == 1.0
    assert cc.assert_roundtrip(A, P, mode="absolute") == 0.0


def test_increment_codec_loses_chunks_and_the_scale_says_how_many(demos):
    """The documented reason `absolute` is the default rather than `increment`."""
    A, P = demos
    at33, _ = cc.representable(A, P, mode="increment", scale=33.0)
    at64, _ = cc.representable(A, P, mode="increment", scale=64.0)
    # recurrent_ppo's measured delta_scale loses a quarter of real expert chunks
    assert at33 == pytest.approx(0.74, abs=0.02)
    assert at64 == pytest.approx(0.978, abs=0.005)
    assert at33 < at64 < 1.0


def test_codec_ignores_differences_outside_the_arena():
    """Two chunks differing only outside [0, WS] drive identical trajectories, so they must
    encode identically -- otherwise Q distinguishes candidates the simulator cannot."""
    p = np.array([256.0, 256.0])
    a = np.full((CHUNK, 2), 600.0)
    b = np.full((CHUNK, 2), 900.0)
    assert np.allclose(cc.encode(a, p), cc.encode(b, p))


@pytest.mark.parametrize("mode,scale", [("absolute", 1.0), ("increment", 64.0)])
def test_decode_stays_inside_the_arena(mode, scale):
    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, size=(500, 2 * CHUNK))
    p = rng.uniform(0, WS, size=(500, 2))
    out = cc.decode(u, p, mode=mode, scale=scale)
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
        ChunkPushTEnv(obs_type="state", max_episode_steps=300)


def test_chunk_return_equals_the_base_env_discounted_sum():
    """The chunked MDP is the 8-step-decision view of the original one EXACTLY.

    Replays the identical absolute targets one base step at a time, through the SAME class's
    inherited PushTGymEnv.step, and compares. If this drifts, `gamma_chunk = gamma_base**CHUNK`
    is a lie and every Q value sits on a different scale from the one documented.
    """
    kwargs = dict(obs_type="state", block_zero_coverage=False, reward_mode="dense",
                  block_near_goal_prob=1.0, block_goal_offset=[8.0, 0.06])
    u = np.linspace(-0.6, 0.6, 2 * CHUNK)

    chunked = ChunkPushTEnv(**kwargs)
    chunked.reset(seed=7)
    targets = cc.decode(u, np.asarray(chunked.env.agent.position), mode="absolute")
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
    env = ChunkPushTEnv(obs_type="state", reward_mode="sparse", block_zero_coverage=False,
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
    env = ChunkPushTEnv(obs_type="state", block_zero_coverage=False,
                        block_near_goal_prob=1.0, block_goal_offset=[3.0, 0.02])
    cov = np.array([(env.reset(seed=i), env._block_coverage())[1] for i in range(40)])
    assert cov.max() < SUCCESS_THRESHOLD
    assert cov.mean() > 0.85, "but they should still start CLOSE, or the rung is unreachable"


def test_a_rung_the_episode_starts_above_is_not_live():
    """There is no achievement to credit there, and crediting it is the same free lunch."""
    env = ChunkPushTEnv(obs_type="state", block_zero_coverage=False,
                        block_near_goal_prob=1.0, block_goal_offset=[3.0, 0.02])
    env.reset(seed=0)
    start = env._block_coverage()
    _, _, _, _, info = env.step(np.zeros(2 * CHUNK, dtype=np.float32))
    for k, tau in enumerate(TAU_LADDER):
        assert info["tau_live"][k] == (start <= tau)


def test_tau_ladder_is_ordered_and_reported():
    """The lower rungs exist because NO demo reaches 0.95; they must be monotone in tau."""
    env = ChunkPushTEnv(obs_type="state", block_zero_coverage=False,
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
        env = ChunkPushTEnv(obs_type="state", block_zero_coverage=False,
                            block_near_goal_prob=1.0, block_goal_offset=offset)
        return np.mean([(env.reset(seed=i), env._block_coverage())[1] for i in range(25)])

    assert mean_coverage([30.0, 0.25]) < 0.6
    assert mean_coverage(D["block_goal_offset"]) > 0.9


# ---------------------------------------------------------------- the agent
def _tiny_agent(n_envs=2, **kw):
    from sac.buffers import buffer_class_for
    from sac.chunk_env import build_chunk_vec_env
    from sac.sac import ChunkSAC

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
    from sac.chunk_codec import decode, demo_chunks

    A, P = demo_chunks(DEMO_ZARR, stride=64)
    agent, env = _tiny_agent()
    agent.demo_offsets = A - P[:, None, :]
    agent.learn(total_timesteps=64)
    agent._last_obs = env.reset()

    def step_px(mix):
        agent.mix = np.asarray(mix, dtype=float)
        u = np.stack([agent._mixture_actions(env.num_envs) for _ in range(30)]).reshape(-1, 2 * CHUNK)
        pos = np.tile((agent._last_obs[:, 18:20] + 1.0) * (WS / 2), (30, 1))
        return float(np.median(np.abs(np.diff(decode(u, pos), axis=1))))

    # a uniform chunk teleports the target across the arena between steps; a demo-shaped one
    # moves it a few pixels. If these coincide, the `which == 2` branch never fired.
    assert step_px([0.0, 1.0, 0.0, 0.0]) > 3 * step_px([0.0, 0.0, 1.0, 0.0])


def test_demo_arm_proposes_demo_shaped_chunks():
    """The demo arm re-anchors a recorded chunk SHAPE at the current agent position, so a push
    the expert performed in one corner is a usable proposal anywhere. Injecting the raw encoding
    instead would only ever teach Q about the places the demos happened to visit."""
    from sac.chunk_codec import decode, demo_chunks

    A, P = demo_chunks(DEMO_ZARR, stride=64)
    agent, env = _tiny_agent()
    agent.demo_offsets = A - P[:, None, :]
    agent.learn(total_timesteps=64)
    agent._last_obs = env.reset()
    agent.mix = np.array([0.0, 0.0, 1.0, 0.0])
    u = agent._mixture_actions(env.num_envs)
    pos = (agent._last_obs[:, 18:20] + 1.0) * (WS / 2)
    steps = np.abs(np.diff(decode(u, pos), axis=1))
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
    assert out.shape == (4, 40)
    assert (np.abs(out) <= 1.0 + 1e-6).all(), "everything the policy sees is in [-1, 1]"
    assert (out[:, 20:] == 1.0).all(), "the demos are fully visible, so the mask half is all +1"


def test_q_verifier_refuses_a_value_it_does_not_implement():
    """Silently accepting `armTn` and returning the Q is how an evaluation ends up labelled
    with a ranking it did not use."""
    with pytest.raises(AssertionError, match="one value"):
        from sac.score import PushTQVerifier

        PushTQVerifier.__init__(object.__new__(PushTQVerifier), "x", value_fn="armTn")
