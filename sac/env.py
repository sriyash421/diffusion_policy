"""The task: the chunked MDP, how an action is encoded, and the demonstrations as transitions.

One agent decision is the action chunk ST/BC actually execute, so `Q(s, chunk)` is a drop-in
for `PushTVerifier.get_value` and TD backs up with `gamma**CHUNK`.

The three things here belong together because they must agree exactly. The codec defines what
an action IS; the env is the only place a chunk becomes 8 env steps; and the demo loader exists
to produce precisely the transitions the env produces, from recorded data instead of from a
rollout. Those two drifting apart is the failure this package cannot have -- the demo-seeded
and self-collected halves of the replay buffer would be on different scales with nothing
raising.
"""

import hashlib
import os

import gymnasium
import numpy as np
from gymnasium import spaces

from recurrent_ppo.pusht_gym import PushTGymEnv
from sac.config import (CHUNK, DEFAULTS as D, DEMO_ZARR, SUCCESS_THRESHOLD, TAU_LADDER, WS,
                        ENV_KEYS, KEYPOINT_ONLY_KEYS, gamma_base)

RESET_TRIES = 20
CACHE_DIR = os.path.expanduser("~/.cache/sac_pusht")


# ==================================================================== the action codec
# Between ST's absolute action chunks and the flat action a Q is defined on.
#
# INVERTIBILITY IS THE REQUIREMENT. The Q exists to rank chunks a diffusion policy proposes, and
# those arrive as absolute pixel targets `(B, 8, 2)` -- `PushTVerifier.get_value`'s second
# argument. A codec that cannot express such a chunk cannot score it, and no amount of training
# fixes that. Measured over all 24,208 eight-step demo windows, the fraction NOT representable:
#
#     absolute over [0, WS]           0.00%      <- what this is
#     absolute over AGENT_BOUNDS      1.16%      demo targets reach 511 px; AGENT_BOUNDS ends at 490
#     increment chain at R = 64 px    2.20%
#     increment chain at R = 33 px   26.01%      recurrent_ppo's measured delta_scale
#
# An increment chain anchors each target on the previous one, which makes the action a SHAPE
# rather than a location and buys translation equivariance. It was implemented, measured and
# dropped: at any scale small enough to be a useful exploration prior it cannot express a quarter
# of what the expert actually did, and a verifier that silently clips the chunk it was asked about
# is worse than one that is merely coarse. `unit_tests/test_sac.py` keeps the measurement, so that
# number stays an executable fact rather than a claim in a comment.
#
# Because absolute needs no anchor, `encode` and `decode` are pure per-element maps -- no agent
# position, no mode, no scale. The anchor the demo-shape and smooth-walk samplers use is the
# sampler's business (see `sac/agent.py`), not the codec's.
#
# Clipping is part of the codec on BOTH sides, so `encode` is defined on `clip(a, 0, WS)`: two
# candidates differing only outside the arena drive byte-identical trajectories and must
# therefore receive the same Q.


def action_dim(chunk=CHUNK):
    return 2 * chunk


def decode(u, chunk=CHUNK):
    """(..., 2*chunk) in [-1, 1] -> (..., chunk, 2) absolute pixel targets."""
    u = np.clip(np.asarray(u, dtype=np.float64), -1.0, 1.0)
    u = u.reshape(*u.shape[:-1], chunk, 2)
    return np.clip((u + 1.0) * 0.5 * WS, 0.0, WS)


def encode(action, chunk=CHUNK):
    """(..., chunk, 2) absolute pixel targets -> (..., 2*chunk) in [-1, 1]. Exact inverse."""
    a = np.clip(np.asarray(action, dtype=np.float64), 0.0, WS)
    u = a / (WS / 2.0) - 1.0
    return u.reshape(*u.shape[:-2], 2 * chunk)


def assert_roundtrip(action, chunk=CHUNK, atol=1e-4):
    """Proof that a real chunk survives encode -> decode. Run at startup, not trusted."""
    a = np.clip(np.asarray(action, dtype=np.float64), 0.0, WS)
    err = float(np.abs(decode(encode(a, chunk), chunk) - a).max())
    if err > atol:
        raise AssertionError(
            f"the chunk codec is not invertible: max error {err:.6g} px. A Q cannot score a "
            f"chunk it cannot express, so this is fatal rather than a warning.")
    return err


def demo_chunks(zarr_path, chunk=CHUNK, stride=1):
    """(actions (N, chunk, 2), agent_pos (N, 2)) over every in-episode window of the demos.

    Windowed WITHIN episodes: a window spanning an episode boundary joins two unrelated
    trajectories, and the chunk it produces is something no expert ever did. `agent_pos` is not
    needed by the codec; it is returned because the demo-shape sampler anchors on it.
    """
    import zarr

    root = zarr.open(str(zarr_path), "r")
    act = np.asarray(root["data/action"], dtype=np.float64)
    pos = np.asarray(root["data/agent_pos"], dtype=np.float64)
    ends = np.asarray(root["meta/episode_ends"])
    starts = np.concatenate([[0], ends[:-1]])
    idx = np.concatenate([np.arange(s, e - chunk + 1, stride) for s, e in zip(starts, ends)])
    return act[idx[:, None] + np.arange(chunk)], pos[idx]


# ==================================================================== the chunked MDP
# PushT where one agent decision is the action chunk ST/BC actually execute.
#
# A SUBCLASS of PushTGymEnv, not a wrapper, and with no edits to recurrent_ppo. The reset
# rejection sampling, the observation encoding, the reward modes, the occlusion chain and the
# is_success/max_reward bookkeeping are all inherited verbatim, so the SAC arm and the PPO arm
# cannot drift on what the TASK is. `_convert_action` is the single seam where an action becomes
# pixels, and it is the only method overridden for the action change.
#
# The chunked MDP is the 8-step-decision view of the original one, EXACTLY and not approximately:
# `step` returns sum_k gamma_base^k r_k and the agent is given gamma_chunk = gamma_base**CHUNK.
# `max_episode_steps` is a multiple of CHUNK for the same reason -- a ragged final chunk would
# bootstrap at gamma**4 while the code says gamma**8, silently, and worst at the truncation
# boundary where the critic has the least data.


class ChunkPushTEnv(PushTGymEnv):
    """PushTGymEnv whose action is a whole chunk of `chunk` absolute targets."""

    def __init__(self, chunk=CHUNK, gamma=D["gamma"], tau_ladder=TAU_LADDER,
                 max_episode_steps=D["max_episode_steps"],
                 max_reset_coverage=D["max_reset_coverage"], **kwargs):
        assert max_episode_steps % chunk == 0, (
            f"max_episode_steps={max_episode_steps} is not a multiple of chunk={chunk}; a ragged "
            f"final chunk bootstraps at the wrong discount. Use {chunk * round(max_episode_steps / chunk)}.")
        # `reward` defaults to sparse here (sac/config), which is what makes Q* the time-to-solve
        # value the verifier wants. The base class still owns what each mode means.
        kwargs.setdefault("reward_mode", D["reward"])
        super().__init__(max_episode_steps=max_episode_steps, **kwargs)
        self.chunk = int(chunk)
        self.gamma_base = gamma_base(gamma, chunk)
        self.tau_ladder = tuple(float(t) for t in tau_ladder)
        self.max_reset_coverage = float(max_reset_coverage)
        # flat, not (chunk, 2): SB3's SAC needs a 1-D Box.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim(chunk),), dtype=np.float32)

    @property
    def agent_position(self):
        """The agent's arena-pixel position, as a plain array.

        A property rather than `get_attr("agent")`: the agent is a pymunk Body, which does not
        pickle, so reading it across a SubprocVecEnv has to return numbers. Used only by the
        behaviour mixture to place a proposed chunk -- never by the policy or the Q head, which
        see exactly what their observation arm defines.
        """
        return np.asarray(self.env.agent.position, dtype=np.float64)

    def _convert_action(self, action):
        """Inside a chunk the action is ALREADY an absolute pixel target.

        Overrides the base class's [-1,1] -> AGENT_BOUNDS mapping. Clipped to the arena rather
        than to AGENT_BOUNDS: demo targets reach 511 px and the ST eval path does not clip, so a
        Q bounded to AGENT_BOUNDS could not express 1.16% of real chunks. AGENT_BOUNDS exists
        because dense-reward runs parked outside the arena and got paid for it; under a sparse
        terminating reward parking earns exactly 0, so that exploit does not exist here.
        """
        return np.clip(np.asarray(action, dtype=np.float64), 0.0, WS)

    def reset(self, *, seed=None, options=None):
        # AN EPISODE MUST NOT BEGIN SOLVED. The near-goal curriculum exists to put the block
        # within reach of the threshold, but at the offsets that actually produce a tau=0.95
        # terminal it overshoots: at [3, 0.02] fully 33% of draws are ALREADY above 0.95, and
        # PushTEnv then pays the sparse +1 on the first step for doing nothing. Measured: mean
        # return 0.349 from a do-nothing policy, i.e. the curriculum was handing out most of
        # the reward for free. The tau ladder already refuses to credit such a start (see
        # `_crossed` below); this makes the env's own reward agree.
        for attempt in range(RESET_TRIES):
            obs, info = super().reset(seed=seed if attempt == 0 else None, options=options)
            if self._block_coverage() <= self.max_reset_coverage:
                break
        else:
            print(f"[WARN] {RESET_TRIES} draws all started above coverage "
                  f"{self.max_reset_coverage}; widen --block-goal-offset.")
        # per-rung: has THIS EPISODE already exceeded tau? A transition after the crossing
        # belongs to an MDP that has terminated, and is dropped from that rung's minibatch.
        # Seeded from the RESET coverage, so a rung the episode starts above is never live:
        # there is no achievement to learn there, and paying for it would be a free lunch the
        # near-goal curriculum hands out by construction. the demo loader below applies the same rule.
        self._crossed = self._block_coverage() > np.asarray(self.tau_ladder)
        self._chunk_max_coverage = 0.0
        return obs, info

    def step(self, action):
        """One chunk: `chunk` base steps, discounted-summed, stopping early on episode end."""
        targets = decode(action, chunk=self.chunk)
        total, discount = 0.0, 1.0
        terminated = truncated = False
        info = {}
        taus = np.asarray(self.tau_ladder)
        self._chunk_max_coverage = 0.0          # over THIS chunk's reached states
        # live is read BEFORE this chunk runs: a rung crossed on an earlier chunk is done with.
        live = ~self._crossed.copy()
        tau_reward = np.zeros(len(taus), dtype=np.float32)
        tau_done = np.zeros(len(taus), dtype=bool)
        for k in range(self.chunk):
            obs, reward, terminated, truncated, info = super().step(targets[k])
            total += discount * float(reward)
            coverage = self._block_coverage()
            # the rungs this base step crosses for the first time this episode
            fresh = live & ~self._crossed & (coverage > taus)
            tau_reward[fresh] = discount      # = gamma_base**k, so crossing earlier pays more
            tau_done[fresh] = True
            self._crossed |= coverage > taus
            self._chunk_max_coverage = max(self._chunk_max_coverage, coverage)
            discount *= self.gamma_base
            if terminated or truncated:
                break
        # `max_reward` / `is_success` stay the base class's, so the episode score remains the
        # quantity pusht_image_runner reports and the PPO arms are compared on.
        info["chunk_max_coverage"] = self._chunk_max_coverage
        info["tau_reward"] = tau_reward
        info["tau_done"] = tau_done
        info["tau_live"] = live
        return obs, total, terminated, truncated, info


def make_chunk_env(obs_type="keypoint", seed=0, rank=0, monitor_path=None, **env_kwargs):
    """Thunk for a single seeded, Monitor-wrapped chunked env."""
    def _init():
        from stable_baselines3.common.monitor import Monitor

        env = ChunkPushTEnv(obs_type=obs_type, **env_kwargs)
        env.action_space.seed(seed + rank)
        return Monitor(env, filename=monitor_path,
                       info_keywords=("is_success", "max_reward", "chunk_max_coverage"))

    return _init


def build_chunk_vec_env(obs_type="keypoint", n_envs=16, seed=0, use_subproc=True,
                        monitor_dir=None, **env_kwargs):
    """The vectorised chunked env both train and play build, so they cannot drift."""
    import os

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = []
    for rank in range(n_envs):
        monitor_path = None if monitor_dir is None else os.path.join(monitor_dir, f"env_{rank}")
        fns.append(make_chunk_env(obs_type=obs_type, seed=seed, rank=rank,
                                  monitor_path=monitor_path, **env_kwargs))
    # forkserver: pymunk/pygame state does not survive a plain fork cleanly
    venv = SubprocVecEnv(fns, start_method="forkserver") if use_subproc and n_envs > 1 else DummyVecEnv(fns)
    venv.seed(seed)
    return venv


def env_kwargs_from(cfg, obs_type):
    """The ChunkPushTEnv kwargs an arm takes, out of a flat config dict (CLI args or args.yaml)."""

    kwargs = {k: cfg[k] for k in ENV_KEYS}
    kwargs["reward_mode"] = cfg["reward"]
    kwargs["block_zero_coverage"] = cfg["block_zero_coverage"]
    for key in ("gamma", "tau_ladder"):
        kwargs[key] = cfg[key]
    if obs_type == "keypoint":
        kwargs.update({k: cfg[k] for k in KEYPOINT_ONLY_KEYS})
    return kwargs


# ==================================================================== the demonstrations
# The 206 demonstrations as chunk transitions, for every rung of the tau ladder.
#
# WHY THE LADDER EXISTS, in one measurement: replaying every demo state through PushTEnv and
# taking coverage exactly as `PushTGymEnv._block_coverage` does, the best any demonstration ever
# achieves is **0.9018**, and the fraction reaching each threshold is
#
#     0.80  93.7%        0.85  51.0%        0.90  1.0%        0.95  0.0%
#
# `PushTEnv.success_threshold` is 0.95 and BON is scored on it. So a replay buffer seeded with
# demonstrations and a single 0.95 head carries **zero** positive reward: every transition is
# `r=0, done=False`, and a Q regressed on that learns `Q = 0` -- the identical "same score for
# every candidate" degeneracy as the heuristic being replaced, reached more expensively. The
# lower rungs are the same MDP at an easier threshold, and they have signal the demonstrations
# already contain.
#
# For head `tau` a transition is LIVE until the episode first exceeds `tau`; it is paid
# `gamma_base**j` on the chunk containing that crossing (j = the within-chunk base step, so
# crossing earlier is worth strictly more, matching the env) and is dropped from that head's
# minibatch afterwards. That is exactly the terminate-at-tau MDP, so `Q_0.95` is unchanged by
# the presence of the others.
#
# Observations are read STRAIGHT from the zarr, not re-rendered. Verified byte-for-byte against
# a live env reset to the same state: `data/img` (float32 in [0, 255]) maps to the uint8 CHW obs
# `PushTGymEnv._convert_obs` produces with max difference **0**, and `data/keypoint` matches the
# live 9 global keypoints to 1.3e-5 (float32 precision). So there is no re-render cost and, more
# importantly, no distribution shift between demo-seeded and self-collected transitions.


def _normalise(p):
    """Arena pixels -> [-1, 1], as PushTGymEnv._normalise does. One definition, imported shape."""
    return np.clip(np.asarray(p, dtype=np.float32) / (WS / 2) - 1.0, -1.0, 1.0)


def frame_coverage(zarr_path=DEMO_ZARR, cache=True):
    """Goal coverage at every demo frame, cached. ~25k shapely intersections, so not per-run."""
    import zarr

    root = zarr.open(str(zarr_path), "r")
    state = np.asarray(root["data/state"], dtype=np.float64)
    key = hashlib.md5(state.tobytes()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"coverage_{key}.npy")
    if cache and os.path.exists(path):
        return np.load(path)

    from diffusion_policy.env.pusht.pusht_env import PushTEnv, pymunk_to_shapely

    env = PushTEnv(legacy=False, render_action=False)
    env.reset_to_state = state[0]
    env.seed(0)
    env.reset()
    goal_geom = pymunk_to_shapely(env._get_goal_pose_body(env.goal_pose), env.block.shapes)
    out = np.empty(len(state), dtype=np.float32)
    for i, s in enumerate(state):
        env._set_state(s)
        block = pymunk_to_shapely(env.block, env.block.shapes)
        out[i] = goal_geom.intersection(block).area / goal_geom.area
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, out)
    return out


def _obs_from_zarr(root, idx, obs_type):
    """The observation PushTGymEnv would emit at these demo frames, without re-simulating."""
    if obs_type == "keypoint":
        kps = np.asarray(root["data/keypoint"])[idx].reshape(len(idx), -1)   # 9 global kps
        agent = np.asarray(root["data/agent_pos"])[idx]
        flat = _normalise(np.concatenate([kps, agent], axis=-1))             # 20, in [-1, 1]
        # the demos are fully visible, so the mask half is all +1 -- the {-1,+1} encoding
        # PushTGymEnv._convert_obs uses, not {0,1}
        return np.concatenate([flat, np.ones_like(flat)], axis=-1)           # 40
    if obs_type == "image":
        # zarr img is float32 in [0, 255] HWC; the obs is uint8 CHW. Verified identical.
        # IMAGE ONLY: `agent_pos` was removed from this arm so it sees exactly what the offline
        # diffusion-policy arms see. A demo transition that still carried it would not match the
        # observation space, and the buffer would store a shape the policy cannot read.
        return {"image": np.moveaxis(np.asarray(root["data/img"])[idx], -1, 1).astype(np.uint8)}
    raise ValueError(f"unknown obs_type {obs_type!r}")


def demo_transitions(zarr_path=DEMO_ZARR, obs_type="keypoint", chunk=CHUNK, stride=1,
                     gamma=0.95, tau_ladder=TAU_LADDER, frac=1.0):
    """Chunk transitions with a per-tau (reward, done, live) triple.

    Returns a dict with `obs`, `action`, `next_obs`, `reward` (N, n_tau), `done` (N, n_tau),
    `live` (N, n_tau) and `chunk_max_coverage` (N,).
    """
    import zarr

    root = zarr.open(str(zarr_path), "r")
    action = np.asarray(root["data/action"], dtype=np.float64)
    ends = np.asarray(root["meta/episode_ends"])
    starts = np.concatenate([[0], ends[:-1]])
    cov = frame_coverage(zarr_path)
    gb = gamma_base(gamma, chunk)
    taus = np.asarray(tau_ladder, dtype=np.float32)

    t0, rew, done, live, cmax = [], [], [], [], []
    for s, e in zip(starts, ends):
        # first frame at which the episode exceeds each tau; len(cov) if it never does
        crossed = cov[s:e] > taus[:, None]                       # (n_tau, T)
        first = np.where(crossed.any(1), crossed.argmax(1), e - s)   # (n_tau,) episode-relative
        for t in range(s, e - chunk, stride):
            j = np.arange(1, chunk + 1)                          # states reached in this chunk
            rel = (t - s) + j                                    # episode-relative reached frames
            r = np.zeros(len(taus), np.float32)
            d = np.zeros(len(taus), bool)
            # live: the episode has not already crossed tau BEFORE this chunk's first reached
            # state. A transition after the crossing belongs to an MDP that has terminated.
            lv = first >= rel[0]
            for k, f in enumerate(first):
                if lv[k] and f <= rel[-1]:                       # the crossing is inside this chunk
                    r[k] = gb ** int(np.argmax(rel >= f))        # earlier crossing pays more
                    d[k] = True
            t0.append(t)
            rew.append(r)
            done.append(d)
            live.append(lv)
            cmax.append(cov[t + 1:t + 1 + chunk].max())

    t0 = np.asarray(t0)
    if frac < 1.0:
        keep = np.random.default_rng(0).permutation(len(t0))[:int(frac * len(t0))]
        t0, rew, done, live, cmax = t0[keep], np.asarray(rew)[keep], np.asarray(done)[keep], \
            np.asarray(live)[keep], np.asarray(cmax)[keep]

    chunks = action[t0[:, None] + np.arange(chunk)]              # (N, chunk, 2) absolute targets
    return {
        "obs": _obs_from_zarr(root, t0, obs_type),
        "next_obs": _obs_from_zarr(root, t0 + chunk, obs_type),
        "action": encode(chunks, chunk=chunk).astype(np.float32),
        "reward": np.asarray(rew, np.float32),
        "done": np.asarray(done, bool),
        "live": np.asarray(live, bool),
        "chunk_max_coverage": np.asarray(cmax, np.float32),
        "tau_ladder": tuple(float(t) for t in tau_ladder),
    }


def summarise(tr):
    """What each rung actually contributes -- printed at load, so a dead rung is loud."""
    lines = [f"[INFO] {len(tr['action'])} demo chunk transitions"]
    for k, tau in enumerate(tr["tau_ladder"]):
        live, pos = tr["live"][:, k], tr["done"][:, k]
        note = "" if pos.sum() else "   <-- NO POSITIVE REWARD; this rung teaches Q = 0"
        lines.append(f"       tau={tau:.2f}  live {live.mean():6.1%}  terminals {int(pos.sum()):5d}"
                     f"  ({pos.mean():.2%} of transitions){note}")
    return "\n".join(lines)
