"""PushT where one agent decision is the action chunk ST/BC actually execute.

A SUBCLASS of PushTGymEnv, not a wrapper, and with no edits to recurrent_ppo. The reset
rejection sampling, the observation encoding, the reward modes, the occlusion chain and the
is_success/max_reward bookkeeping are all inherited verbatim, so the SAC arm and the PPO arm
cannot drift on what the TASK is. `_convert_action` is the single seam where an action becomes
pixels, and it is the only method overridden for the action change.

The chunked MDP is the 8-step-decision view of the original one, EXACTLY and not approximately:
`step` returns sum_k gamma_base^k r_k and the agent is given gamma_chunk = gamma_base**CHUNK.
`max_episode_steps` is a multiple of CHUNK for the same reason -- a ragged final chunk would
bootstrap at gamma**4 while the code says gamma**8, silently, and worst at the truncation
boundary where the critic has the least data.
"""

import gymnasium
import numpy as np
from gymnasium import spaces

from recurrent_ppo.pusht_gym import PushTGymEnv
from sac.chunk_codec import action_dim, decode
from sac.config import CHUNK, DEFAULTS as D, SUCCESS_THRESHOLD, TAU_LADDER, WS, gamma_base

RESET_TRIES = 20


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
        # near-goal curriculum hands out by construction. demo_buffer applies the same rule.
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
    from sac.config import ENV_KEYS, KEYPOINT_ONLY_KEYS

    kwargs = {k: cfg[k] for k in ENV_KEYS}
    kwargs["reward_mode"] = cfg["reward"]
    kwargs["block_zero_coverage"] = cfg["block_zero_coverage"]
    for key in ("gamma", "tau_ladder"):
        kwargs[key] = cfg[key]
    if obs_type == "keypoint":
        kwargs.update({k: cfg[k] for k in KEYPOINT_ONLY_KEYS})
    return kwargs
