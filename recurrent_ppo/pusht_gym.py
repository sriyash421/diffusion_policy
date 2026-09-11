"""PushT wrapped for Stable-Baselines3 / sb3-contrib.

`diffusion_policy.env.pusht` is written against gym 0.21: a 4-tuple `step`, a `reset` that
takes no seed, `render(mode)` with no default, and no time limit. SB3 2.x speaks gymnasium.
Everything that gap costs lives here and nowhere else, so train.py and play.py build the
same env through `make_env` and cannot drift apart.

Two observation variants, matching the ST arms:

    keypoint  9 block T keypoints plus the agent xy (the 20-d lowdim obs of
              config/task/pusht_lowdim.yaml), followed by the 20-d visibility mask that
              says which of those entries `keypoint_visible_rate` occluded. 40 numbers.
    image     PushTImageEnv's {image (3, 96, 96), agent_pos (2)}.

EVERYTHING THE POLICY SEES IS IN [-1, 1]. Positions are scaled against the arena's known
bounds, the visibility mask is mapped to {-1, +1} rather than left at {0, 1}, and the image
is uint8 [0, 255] for the one reason that SB3's NatureCNN then does the /255 itself. Known
constants, so no VecNormalize: nothing to estimate and no running statistics to keep in sync
between training and play.
"""

import gymnasium
import numpy as np
from gymnasium import spaces

from shapely.geometry import Point

from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv

# arena is 512x512; obs positions and action targets both live in it
WS = 512.0
OBS_TYPES = ("keypoint", "image")
ACTION_MODES = ("delta", "absolute")
REWARD_MODES = ("dense", "sparse", "shaped")
OCCLUSION_MODES = ("iid", "persistent")
# fallback only, for when the demonstrations are not on disk; --delta-scale auto measures it
DEFAULT_DELTA_SCALE = 32.0
DEMO_ZARR = "data/pusht_cchi_v7_replay.zarr"
# PushTEnv draws the agent from [50, 450] and the block from [100, 400]. The demonstrations in
# data/pusht_cchi_v7_replay.zarr keep every agent start inside that agent range, but a quarter of
# their BLOCK starts fall outside [100, 400] (measured span x 66-440, y 116-486). Training on the
# narrower box means the policy never resets where a quarter of the demonstrated episodes begin,
# so this widens to cover them, with a little margin.
DEFAULT_AGENT_START_RANGE = (50.0, 450.0)
DEFAULT_BLOCK_START_RANGE = (60.0, 490.0)
# A block CENTRE inside the range above can still put the T's arms through a wall, which spawns
# the block overlapping it and pushes keypoints outside the arena. Redraw when that happens:
# ~74% of draws are accepted, so this costs about 1.4 resets, and it covers 201 of the 206
# demonstrated starts (the other 5 genuinely poke out of the arena).
SPAWN_TRIES = 20
# The dense reward is a function of the BLOCK pose alone -- the agent's position never enters it --
# so until the agent touches the block the return is fixed at reset and there is no gradient toward
# making contact at all. Measured on the first 2M-step run: a random policy contacts the block on
# 1.5% of steps, the trained one on 1.1%, and the reward never changed in 19 of 20 episodes.
# Starting some episodes with the agent already beside the block puts contact within a few steps.
AGENT_RADIUS = 15.0
NEAR_TRIES = 40
# THE AGENT IS A KINEMATIC BODY (pusht_env.py add_circle), so pymunk's walls do not stop it -- it
# passes straight through them; they only constrain the block. The action target clip below is
# therefore the ONLY thing bounding the agent, and it has to be the region the agent can usefully
# occupy rather than the image bounds. Clipping to [0, WS] let the policy park at (0, 512),
# outside the arena, physically unable to reach the block and unpenalised for it -- which is
# exactly where both 2M-step runs converged, from every start.
WALL_INNER = 7.0                       # segments at 5 and 506, radius 2
AGENT_BOUNDS = (WALL_INNER + AGENT_RADIUS, WS - WALL_INNER - AGENT_RADIUS)   # (22, 489)
# The goal pose is a constant for every PushT episode (PushTEnv._setup).
GOAL_POSE = np.array([256.0, 256.0, np.pi / 4])


class PushTGymEnv(gymnasium.Env):
    """gymnasium view of a single PushT env."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        obs_type="keypoint",
        max_episode_steps=300,
        render_size=96,
        keypoint_visible_rate=1.0,
        action_mode="delta",
        delta_scale=DEFAULT_DELTA_SCALE,
        reward_mode="dense",
        shaping_coef=1.0,
        shaping_gamma=0.99,
        occlusion="iid",
        occlusion_persistence=20.0,
        agent_start_range=DEFAULT_AGENT_START_RANGE,
        block_start_range=DEFAULT_BLOCK_START_RANGE,
        agent_near_block_prob=0.0,
        agent_block_gap=(20.0, 80.0),
        block_near_goal_prob=0.0,
        block_goal_offset=(30.0, 0.25),
        legacy=False,
        render_mode=None,
    ):
        assert obs_type in OBS_TYPES, f"unknown obs_type {obs_type!r}, expected one of {OBS_TYPES}"
        assert action_mode in ACTION_MODES, f"unknown action_mode {action_mode!r}, expected {ACTION_MODES}"
        assert reward_mode in REWARD_MODES, f"unknown reward_mode {reward_mode!r}, expected {REWARD_MODES}"
        assert occlusion in OCCLUSION_MODES, f"unknown occlusion {occlusion!r}, expected {OCCLUSION_MODES}"
        self.reward_mode = reward_mode
        self.shaping_coef = float(shaping_coef)
        self.shaping_gamma = float(shaping_gamma)
        self.occlusion = occlusion
        self.obs_type = obs_type
        self.max_episode_steps = max_episode_steps
        self.action_mode = action_mode
        self.delta_scale = float(delta_scale)
        self.agent_start_range = tuple(float(x) for x in agent_start_range)
        self.block_start_range = tuple(float(x) for x in block_start_range)
        self.agent_near_block_prob = float(agent_near_block_prob)
        self.agent_block_gap = tuple(float(x) for x in agent_block_gap)
        self.block_near_goal_prob = float(block_near_goal_prob)
        self.block_goal_offset = tuple(float(x) for x in block_goal_offset)
        self.render_mode = render_mode

        if obs_type == "keypoint":
            # The inner env is PINNED to full visibility and we own the occlusion, for two
            # reasons. PushTKeypointsEnv is shared with the diffusion-policy training and must not
            # change behaviour; and its own dropout is redrawn i.i.d. inside _get_obs, which is
            # the one thing a persistent mode has to replace.
            self.env = PushTKeypointsEnv(
                legacy=legacy,
                render_size=render_size,
                keypoint_visible_rate=1.0,
                render_action=False,
            )
            # PushTKeypointsEnv stacks [obs, visibility mask]; we keep both halves
            self._half = self.env.observation_space.shape[0] // 2
            # obs half is [block keypoints (2 each), agent xy]; only the keypoints are occludable,
            # matching the inner env, which always marks agent_pos visible
            self._n_kps = (self._half - 2) // 2
            self._p_hidden_to_visible, self._p_visible_to_hidden = self._occlusion_rates(
                float(keypoint_visible_rate), float(occlusion_persistence))
            self._visible = np.ones(self._n_kps, dtype=bool)
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(2 * self._half,), dtype=np.float32)
        else:
            self.env = PushTImageEnv(legacy=legacy, render_size=render_size)
            self.observation_space = spaces.Dict({
                "image": spaces.Box(0, 255, shape=(3, render_size, render_size), dtype=np.uint8),
                "agent_pos": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            })

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._elapsed_steps = 0
        self._max_reward = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # We draw the initial state ourselves and hand it over as reset_to_state, for two
        # reasons. PushTEnv.reset() re-derives its state from RandomState(self._seed) and never
        # advances it, so without intervention every episode is the SAME episode. And its block
        # range is [100, 400], which excludes a quarter of the demonstrated block starts.
        for _ in range(SPAWN_TRIES):
            state = self._sample_state()
            self.env.reset_to_state = state
            # seeded per episode: PushTKeypointsEnv draws its visibility mask from this RNG
            self.env.seed(int(self.np_random.integers(0, 2**31 - 1)))
            obs = self.env.reset()
            if not self._block_inside_arena():
                continue
            # guarded so prob=0 consumes no draw, i.e. it is bit-identical to not having the
            # curriculum at all rather than merely equivalent in distribution
            if self.agent_near_block_prob <= 0.0 or self.np_random.random() >= self.agent_near_block_prob:
                break
            # the block's geometry is only known once it is placed, so the curriculum draw is a
            # second reset rather than part of _sample_state
            near = self._sample_agent_near_block()
            if near is None:
                continue
            state[:2] = near
            self.env.reset_to_state = state
            obs = self.env.reset()
            break
        else:
            print(f"[WARN] {SPAWN_TRIES} spawn draws all put the block through a wall; "
                  f"using the last one. Narrow --block-start-range.")
        self._elapsed_steps = 0
        self._max_reward = 0.0
        self._solved = False
        self._prev_phi = self._potential()
        if self.obs_type == "keypoint":
            # start the chain AT its stationary distribution, so episode step 0 is not
            # systematically cleaner than the rest
            self._visible = self.np_random.random(self._n_kps) < self._stationary_visible
        return self._convert_obs(obs), {}

    @property
    def _stationary_visible(self):
        total = self._p_hidden_to_visible + self._p_visible_to_hidden
        return 1.0 if total <= 0 else self._p_hidden_to_visible / total

    def _occlusion_rates(self, visible_rate, persistence):
        """Two-state Markov rates whose STATIONARY visible fraction is `visible_rate`, in both modes.

        `iid` is the degenerate case of the same chain -- p(hidden->visible) = v gives a mean
        hidden run of 1/v, i.e. 2 steps at v=0.5, exactly what PushTKeypointsEnv does per step.
        `persistent` sets that mean run to `persistence` instead. One chain, so the two arms are
        matched on HOW MUCH is missing and differ only in how it is distributed over time.
        """
        v = float(np.clip(visible_rate, 0.0, 1.0))
        if v >= 1.0:
            return 1.0, 0.0                                   # never hidden
        if v <= 0.0:
            return 0.0, 1.0                                   # never visible
        p_hv = v if self.occlusion == "iid" else 1.0 / max(persistence, 1.0)
        return p_hv, float(np.clip(p_hv * (1.0 - v) / v, 0.0, 1.0))

    def _step_occlusion(self):
        """Advance the per-keypoint chain one step and return the (2 * n_kps,) obs-half mask."""
        flip = self.np_random.random(self._n_kps) < np.where(
            self._visible, self._p_visible_to_hidden, self._p_hidden_to_visible)
        self._visible = np.where(flip, ~self._visible, self._visible)
        return np.repeat(self._visible.astype(np.float32), 2)

    def _sample_state(self):
        agent_lo, agent_hi = self.agent_start_range
        block_lo, block_hi = self.block_start_range
        if self.block_near_goal_prob > 0.0 and self.np_random.random() < self.block_near_goal_prob:
            # A reverse curriculum: start the block part-way solved so the agent can reach the
            # goal, and see a reward for doing so, within an episode. With a uniform block start
            # the first 2M-step run never once crossed the success threshold.
            pos_offset, angle_offset = self.block_goal_offset
            block = GOAL_POSE[:2] + self.np_random.uniform(-pos_offset, pos_offset, size=2)
            angle = GOAL_POSE[2] + self.np_random.uniform(-angle_offset, angle_offset)
        else:
            block = self.np_random.uniform(block_lo, block_hi, size=2)
            # PushTEnv draws randn()*2pi - pi here, which is a wrapped normal with sigma = 2pi:
            # uniform to ~9 decimal places once taken mod 2pi. Drawn uniformly and said plainly.
            angle = self.np_random.uniform(0.0, 2.0 * np.pi)
        return np.array([
            self.np_random.uniform(agent_lo, agent_hi),
            self.np_random.uniform(agent_lo, agent_hi),
            block[0], block[1], angle,
        ])

    def _sample_agent_near_block(self):
        """An agent start a short clear gap from the block's SURFACE, or None if no draw landed.

        Distance to the surface, not to the centroid: the T is asymmetric, so a fixed radius would
        put the agent inside one arm and a wide gap from the other. The lower bound clears the
        agent's own radius, so it never spawns overlapping.
        """
        geom = pymunk_to_shapely(self.env.block, self.env.block.shapes)
        centre = np.asarray(geom.centroid.coords[0])
        lo_gap, hi_gap = self.agent_block_gap
        agent_lo, agent_hi = self.agent_start_range
        reach = float(np.hypot(*(np.asarray(geom.bounds[2:]) - centre))) + hi_gap
        for _ in range(NEAR_TRIES):
            candidate = centre + self.np_random.uniform(-reach, reach, size=2)
            if not (agent_lo <= candidate[0] <= agent_hi and agent_lo <= candidate[1] <= agent_hi):
                continue
            if lo_gap <= geom.distance(Point(candidate)) <= hi_gap:
                return candidate
        return None

    def _potential(self):
        """Phi(s) = -(agent-to-block distance) / WS, the potential the shaping is built from.

        Distance to the block's CENTRE, not its surface. Surface distance flatlines at zero the
        moment the agent touches, so it stops rewarding the push it just earned; the centre keeps
        pulling through contact. It is also 15x cheaper per step (+1% against +15% of an env step),
        and under potential-based shaping the exact shape of Phi is free -- any potential leaves
        the optimal policy unchanged, so this is chosen for the gradient it gives, not for
        correctness.
        """
        agent = np.asarray(self.env.agent.position)
        block = np.asarray(self.env.block.position)
        return -float(np.linalg.norm(agent - block)) / WS

    def _block_inside_arena(self):
        verts = np.array([self.env.block.local_to_world(v)
                          for shape in self.env.block.shapes for v in shape.get_vertices()])
        return bool((verts >= 0.0).all() and (verts <= WS).all())

    def step(self, action):
        obs, reward, done, info = self.env.step(self._convert_action(action))
        self._elapsed_steps += 1
        # ALWAYS the env's own dense reward, in both modes, so the episode score stays one
        # comparable quantity: max normalised coverage, as pusht_image_runner reports it.
        self._max_reward = max(self._max_reward, float(reward))
        newly_solved = bool(done) and not self._solved
        self._solved = self._solved or bool(done)

        # Termination is tied to the reward mode, because that is what makes each self-consistent.
        # Under `dense` the reward saturates at 1.0/step while `done` needs coverage STRICTLY
        # above 0.95, so terminating on success would forfeit a stream worth more than the whole
        # approach -- the optimal policy would be to stop just short of solving. Under `sparse`
        # there is nothing left to earn, so terminating costs nothing.
        if self.reward_mode == "dense":
            reward, terminated = float(reward), False
        elif self.reward_mode == "shaped":
            # Potential-based shaping (Ng, Harada & Russell 1999): F = gamma*Phi(s') - Phi(s)
            # PROVABLY leaves the optimal policy unchanged, which a raw distance bonus does not.
            # It exists because PushT's own reward is a function of the block pose alone, so
            # nothing the agent does before contact changes its return -- and the 2M-step runs
            # duly learned to drive into a corner and stop.
            phi = self._potential()
            reward = float(reward) + self.shaping_coef * (self.shaping_gamma * phi - self._prev_phi)
            self._prev_phi = phi
            terminated = False
        else:
            reward, terminated = (1.0 if newly_solved else 0.0), self._solved
        truncated = (not terminated) and self._elapsed_steps >= self.max_episode_steps
        # added to, not replacing, the env's own info (pos_agent, block_pose, n_contacts...).
        # is_success is STICKY: SB3 reads it off the top-level info at episode end, and under
        # `dense` the episode does not end when the task is solved.
        info["is_success"] = self._solved
        info["max_reward"] = self._max_reward
        return self._convert_obs(obs), reward, terminated, truncated, info

    def render(self):
        return self.env.render("rgb_array")

    def close(self):
        self.env.close()

    def _convert_obs(self, obs):
        if self.obs_type == "keypoint":
            # normalise FIRST, then mask, so an occluded keypoint reads as 0.0 (the arena
            # centre) rather than -1.0 (a corner). The mask is what disambiguates it.
            norm = self._normalise(obs[: self._half])
            # our mask, not the inner env's -- it is pinned to full visibility. agent_pos is
            # never occluded, matching the convention PushTKeypointsEnv itself uses.
            mask = np.ones(self._half, dtype=np.float32)
            mask[: 2 * self._n_kps] = self._step_occlusion()
            return np.concatenate([norm * mask, 2.0 * mask - 1.0])
        return {
            # PushTImageEnv hands back float32 in [0, 1]; SB3 normalises uint8 itself and the
            # rollout buffer is 4x smaller for it.
            "image": (obs["image"] * 255).astype(np.uint8),
            "agent_pos": self._normalise(obs["agent_pos"]),
        }

    @staticmethod
    def _normalise(positions):
        """Arena pixels -> [-1, 1], clipped so the declared observation space is not a lie.

        A block pressed into a wall can push a keypoint a little past the arena edge -- about
        0.004% of entries, by at most ~20px. The demonstrations do it too (they reach 521px).
        Spawn overlaps, the structural cause, are handled by redrawing in reset().
        """
        return np.clip(np.asarray(positions, dtype=np.float32) / (WS / 2) - 1.0, -1.0, 1.0)

    def _convert_action(self, action):
        """[-1, 1]^2 -> an absolute target in [0, WS]^2, which is what PushT's PD controller takes.

        `delta` is the default because the useful target is a waypoint near the agent, and the
        human demonstrations move it a median of 8px per step: an absolute-position Gaussian at
        std 1 would explore with a standard deviation of 256px, half the table.
        """
        lo, hi = AGENT_BOUNDS
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.action_mode == "delta":
            target = np.asarray(self.env.agent.position, dtype=np.float64) + action * self.delta_scale
            return np.clip(target, lo, hi)
        # absolute spans the same usable region, so the two modes address the same set of points
        return lo + (action + 1.0) * 0.5 * (hi - lo)


def demo_action_steps(zarr_path=DEMO_ZARR):
    """Per-axis |target displacement| of every step in the human demonstrations, in pixels.

    PushT actions are absolute targets, so differencing them within an episode gives the target
    displacement -- the same quantity a delta action commands, which is what makes the two
    comparable. Differenced WITHIN episodes: across an episode boundary the difference is
    between two unrelated states.
    """
    import zarr

    root = zarr.open(zarr_path, "r")
    action = np.asarray(root["data/action"])
    ends = np.asarray(root["meta/episode_ends"])
    starts = np.concatenate([[0], ends[:-1]])
    return np.abs(np.concatenate([np.diff(action[s:e], axis=0) for s, e in zip(starts, ends)])).ravel()


def delta_scale_from_demos(zarr_path=DEMO_ZARR, percentile=99.0):
    """How far |a| = 1 should move the target, measured rather than guessed.

    The percentile answers "what counts as a big step for a human here": at p99 the full action
    range spans essentially everything the demonstrations do (median 4px, p99 33px) without the
    rare 135px outlier stretching the scale so that ordinary moves live in the first 3% of it.
    Falls back to DEFAULT_DELTA_SCALE, loudly, when the dataset is not on disk -- this is a
    convenience for picking a number, not a dependency of training.
    """
    try:
        return float(np.percentile(demo_action_steps(zarr_path), percentile))
    except Exception as exc:
        print(f"[WARN] Could not measure the demo action steps from {zarr_path} ({exc}); "
              f"falling back to --delta-scale {DEFAULT_DELTA_SCALE}.")
        return DEFAULT_DELTA_SCALE


def make_env(obs_type="keypoint", seed=0, rank=0, monitor_path=None, **env_kwargs):
    """Thunk for a single seeded, Monitor-wrapped env. Pass it to Dummy/SubprocVecEnv."""
    def _init():
        from stable_baselines3.common.monitor import Monitor

        env = PushTGymEnv(obs_type=obs_type, **env_kwargs)
        env.action_space.seed(seed + rank)
        return Monitor(env, filename=monitor_path, info_keywords=("is_success", "max_reward"))

    return _init


def build_vec_env(obs_type="keypoint", n_envs=16, seed=0, use_subproc=True, monitor_dir=None, **env_kwargs):
    """The vectorised env both scripts train and evaluate on."""
    import os

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = []
    for rank in range(n_envs):
        monitor_path = None if monitor_dir is None else os.path.join(monitor_dir, f"env_{rank}")
        fns.append(make_env(obs_type=obs_type, seed=seed, rank=rank, monitor_path=monitor_path, **env_kwargs))
    # forkserver: pymunk/pygame state does not survive a plain fork cleanly
    venv = SubprocVecEnv(fns, start_method="forkserver") if use_subproc and n_envs > 1 else DummyVecEnv(fns)
    # ONE seeding path for training and play alike. SB3 hands seed + idx to each env at the next
    # reset and then clears them, which is the gymnasium convention; seeding inside the thunk as
    # well would leave the two scripts on different paths, since only training re-seeds via the
    # algorithm's set_random_seed.
    venv.seed(seed)
    return venv


def env_kwargs_from(cfg, obs_type):
    """The PushTGymEnv kwargs an arm takes, out of a flat config dict (CLI args or args.yaml).

    keypoint_visible_rate belongs to PushTKeypointsEnv only, so the image arm must not be handed
    it -- the one place that asymmetry is expressed, rather than at every call site.
    """
    kwargs = {
        "max_episode_steps": cfg["max_episode_steps"],
        "render_size": cfg["render_size"],
        "action_mode": cfg["action_mode"],
        "delta_scale": cfg["delta_scale"],
        "agent_start_range": cfg["agent_start_range"],
        "block_start_range": cfg["block_start_range"],
        "reward_mode": cfg["reward"],
        "shaping_coef": cfg["shaping_coef"],
        "shaping_gamma": cfg["gamma"],
        "agent_near_block_prob": cfg["agent_near_block_prob"],
        "agent_block_gap": cfg["agent_block_gap"],
        "block_near_goal_prob": cfg["block_near_goal_prob"],
        "block_goal_offset": cfg["block_goal_offset"],
    }
    if obs_type == "keypoint":
        kwargs["keypoint_visible_rate"] = cfg["keypoint_visible_rate"]
        kwargs["occlusion"] = cfg["occlusion"]
        kwargs["occlusion_persistence"] = cfg["occlusion_persistence"]
    return kwargs
