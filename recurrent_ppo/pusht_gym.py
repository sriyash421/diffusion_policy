"""PushT wrapped for Stable-Baselines3 / sb3-contrib.

`diffusion_policy.env.pusht` is written against gym 0.21: a 4-tuple `step`, a `reset` that
takes no seed, `render(mode)` with no default, and no time limit. SB3 2.x speaks gymnasium.
Everything that gap costs lives here and nowhere else, so train.py and play.py build the
same env through `make_env` and cannot drift apart.

Two observation variants, matching the ST arms:

    keypoint  9 block T keypoints plus the agent xy (the 20-d lowdim obs of
              diffusion_policy/config/task/pusht_lowdim.yaml), followed by the 20-d
              visibility mask saying which of those `keypoint_visible_rate` occluded.
              40 numbers.
    image     PushTImageEnv's {image (3, 96, 96), agent_pos (2)}.

EVERYTHING THE POLICY SEES IS IN [-1, 1]. Positions are scaled against the arena's known
bounds, the visibility mask is mapped to {-1, +1} rather than left at {0, 1}, and the image
is uint8 [0, 255] for the one reason that SB3's `preprocess_obs` then does the /255 itself,
and the rollout buffer is 4x smaller for it. Known constants, so no VecNormalize on the
observation: nothing to estimate and no running statistics to keep in sync between
training and play.
"""

import gymnasium
import numpy as np
from gymnasium import spaces

from shapely.geometry import Point

from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv

from diffusion_policy.env.pusht.feedback_util import (GOAL_KEYPOINTS, arm_to_t_distance,
                                                     keypoints_at_pose, t_goal_distance)
from recurrent_ppo.config import (ACTION_MODES, AGENT_BOUNDS, AGENT_RADIUS, DEFAULTS as D,
                                  DELTA_SCALE_FALLBACK, DEMO_ZARR, ENV_KEYS, GOAL_POSE, KEYPOINT_ONLY_KEYS, NEAR_TRIES,
                                  OBS_TYPES, OCCLUSION_MODES, REWARD_MODES, SHAPING_POTENTIALS,
                                  SPAWN_TRIES, WS)

# Keys of the per-transition AUGMENTATION DRAW (see AugmentationDraw). Prefixed so they cannot
# collide with an observation key, and named in one place because the extractor reads them back.
STATE_KEY = "state"
AUG_NOISE_KEY = "aug_noise"
AUG_T_KEY = "aug_t"
AUG_CROP_KEY = "aug_crop"


class PushTGymEnv(gymnasium.Env):
    """gymnasium view of a single PushT env."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        obs_type=D["obs"],
        max_episode_steps=D["max_episode_steps"],
        render_size=D["render_size"],
        keypoint_visible_rate=D["keypoint_visible_rate"],
        action_mode=D["action_mode"],
        delta_scale=DELTA_SCALE_FALLBACK,
        reward_mode=D["reward"],
        shaping_coef=D["shaping_coef"],
        shaping_potential=D["shaping_potential"],
        progress_coef=D["progress_coef"],
        success_bonus=D["success_bonus"],
        block_zero_coverage=D["block_zero_coverage"],
        shaping_gamma=D["gamma"],
        occlusion=D["occlusion"],
        occlusion_persistence=D["occlusion_persistence"],
        agent_start_range=D["agent_start_range"],
        block_start_range=D["block_start_range"],
        agent_near_block_prob=D["agent_near_block_prob"],
        agent_block_gap=D["agent_block_gap"],
        block_near_goal_prob=D["block_near_goal_prob"],
        block_goal_offset=D["block_goal_offset"],
        legacy=False,
        render_mode=None,
    ):
        assert obs_type in OBS_TYPES, f"unknown obs_type {obs_type!r}, expected one of {OBS_TYPES}"
        assert action_mode in ACTION_MODES, f"unknown action_mode {action_mode!r}, expected {ACTION_MODES}"
        assert reward_mode in REWARD_MODES, f"unknown reward_mode {reward_mode!r}, expected {REWARD_MODES}"
        assert occlusion in OCCLUSION_MODES, f"unknown occlusion {occlusion!r}, expected {OCCLUSION_MODES}"
        self.reward_mode = reward_mode
        assert shaping_potential in SHAPING_POTENTIALS, \
            f"unknown shaping_potential {shaping_potential!r}, expected {SHAPING_POTENTIALS}"
        self.shaping_potential = shaping_potential
        self.shaping_coef = float(shaping_coef)
        self.progress_coef = float(progress_coef)
        self.success_bonus = float(success_bonus)
        self.block_zero_coverage = bool(block_zero_coverage)
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
            # zero coverage at reset: 28.7% of uniform block draws already overlap the goal, and
            # under a level reward that overlap is paid for every step of the episode
            if self.block_zero_coverage and self._block_coverage() > 0.0:
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
        self._prev_distance = self._t_goal_distance()
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

    def _t_goal_distance(self):
        """Mean per-keypoint distance of the achieved T from the goal pose, in pixels."""
        block = self.env.block
        pose = np.array([block.position[0], block.position[1], block.angle], dtype=np.float32)
        return float(t_goal_distance((GOAL_KEYPOINTS - keypoints_at_pose(pose)).reshape(-1)))

    def _block_coverage(self):
        """Fraction of the goal T the block currently covers -- PushTEnv's own reward quantity."""
        goal_body = self.env._get_goal_pose_body(self.env.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.env.block.shapes)
        block_geom = pymunk_to_shapely(self.env.block, self.env.block.shapes)
        return goal_geom.intersection(block_geom).area / goal_geom.area

    def _potential(self):
        """Phi(s), in units of WS. Reuses feedback_util so these terms mean exactly what the
        repo's verifier means by them, rather than a second definition of the same quantity.

        `feedback` is the goal-vs-achieved per-keypoint displacement, so t_goal captures the
        block's POSITION AND ROTATION error and is 0 only at the goal pose. A centroid distance
        would be blind to rotation, which is half of what PushT asks for.

        arm-to-T is the only term that varies before the agent touches the block; t_goal alone is
        flat until the block moves, which is why `arm_t` exists as the combination.
        """
        block = self.env.block
        pose = np.array([block.position[0], block.position[1], block.angle], dtype=np.float32)
        feedback = (GOAL_KEYPOINTS - keypoints_at_pose(pose)).reshape(-1)
        if self.shaping_potential == "arm":
            distance = arm_to_t_distance(np.asarray(self.env.agent.position), feedback)
        elif self.shaping_potential == "t_goal":
            distance = t_goal_distance(feedback)
        else:
            distance = t_goal_distance(feedback) + arm_to_t_distance(
                np.asarray(self.env.agent.position), feedback)
        return -float(distance) / WS

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
        elif self.reward_mode == "delta":
            # Pays for CHANGE, not level: exactly 0 when the T does not move, however well or
            # badly it happens to be placed. That is the fix for dense's free lunch -- under
            # dense, doing nothing from a lucky reset paid a return of 92.7, more than any
            # policy earned by acting. Summed over an episode this telescopes to
            # (d_start - d_end), i.e. total progress, so it cannot be farmed by loitering.
            distance = self._t_goal_distance()
            reward = self.progress_coef * (self._prev_distance - distance) / WS
            if newly_solved:
                reward += self.success_bonus
            self._prev_distance = distance
            terminated = False
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
    Falls back to DELTA_SCALE_FALLBACK, loudly, when the dataset is not on disk -- this is a
    convenience for picking a number, not a dependency of training.
    """
    try:
        return float(np.percentile(demo_action_steps(zarr_path), percentile))
    except Exception as exc:
        print(f"[WARN] Could not measure the demo action steps from {zarr_path} ({exc}); "
              f"falling back to --delta-scale {DELTA_SCALE_FALLBACK}.")
        return DELTA_SCALE_FALLBACK


class AugmentationDraw(gymnasium.ObservationWrapper):
    """Carry the per-transition corruption noise and crop offset IN the observation.

    These are learner-side draws, not environment state, and it is worth saying plainly why
    they are emitted here anyway: PPO re-encodes every stored transition once per epoch, and
    SB3's only per-transition channel into a features extractor is the observation it was
    stored with. A draw made inside the extractor is therefore a DIFFERENT draw on every
    epoch, so `ratio = exp(log_prob - old_log_prob)` compares pi(a | draw 2) against
    pi_old(a | draw 1) and stops being a policy ratio at all. Drawing here puts the draw in
    the rollout buffer, where all `--n-epochs` passes read back the same numbers and
    reproduce the same noise and the same crop.

    The draw is attached to the observation the action was chosen FROM, which is the
    transition it belongs to: SB3 stores `self._last_obs` alongside that action.

    `feature_dim` and `crop_span` are learner-side facts (the encoder's output width, and
    `render_size - crop`), so they are passed in rather than inferred.
    """

    def __init__(self, env, feature_dim=None, t_max=None, crop_span=None):
        super().__init__(env)
        self.feature_dim = feature_dim
        self.t_max = t_max
        self.crop_span = crop_span
        inner = env.observation_space
        # a Box observation has to be nested under a key to sit beside the draw; a Dict one
        # already has keys, and the ST image extractor indexes them by name, so it is untouched
        spaces_map = dict(inner.spaces) if isinstance(inner, spaces.Dict) else {STATE_KEY: inner}
        if feature_dim is not None:
            spaces_map[AUG_NOISE_KEY] = spaces.Box(-np.inf, np.inf, shape=(feature_dim,), dtype=np.float32)
            # float, not int: it is an integer timestep, but a float box keeps every buffer and
            # normaliser on one dtype path. The extractor casts it back with .long().
            spaces_map[AUG_T_KEY] = spaces.Box(0.0, float(t_max), shape=(1,), dtype=np.float32)
        if crop_span is not None:
            spaces_map[AUG_CROP_KEY] = spaces.Box(0.0, float(crop_span - 1), shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict(spaces_map)

    def observation(self, obs):
        out = dict(obs) if isinstance(obs, dict) else {STATE_KEY: obs}
        if self.feature_dim is not None:
            out[AUG_NOISE_KEY] = self.np_random.standard_normal(self.feature_dim).astype(np.float32)
            out[AUG_T_KEY] = np.array([self.np_random.integers(0, self.t_max)], dtype=np.float32)
        if self.crop_span is not None:
            # [0, span), matching CropRandomizer's own sampler exactly -- it scales rand() by
            # (image - crop) and truncates, so the last offset it can draw is span - 1. A wider
            # range here would make this a different augmentation than ST's.
            out[AUG_CROP_KEY] = self.np_random.integers(0, self.crop_span, size=2).astype(np.float32)
        return out


def make_env(obs_type="keypoint", seed=0, rank=0, monitor_path=None, aug=None, **env_kwargs):
    """Thunk for a single seeded, Monitor-wrapped env. Pass it to Dummy/SubprocVecEnv.

    `aug` is the AugmentationDraw kwargs, or None to leave the observation alone.
    """
    def _init():
        from stable_baselines3.common.monitor import Monitor

        env = PushTGymEnv(obs_type=obs_type, **env_kwargs)
        if aug:
            env = AugmentationDraw(env, **aug)
        env.action_space.seed(seed + rank)
        return Monitor(env, filename=monitor_path, info_keywords=("is_success", "max_reward"))

    return _init


def build_vec_env(obs_type="keypoint", n_envs=16, seed=0, use_subproc=True, monitor_dir=None,
                  aug=None, **env_kwargs):
    """The vectorised env both scripts train and evaluate on."""
    import os

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = []
    for rank in range(n_envs):
        monitor_path = None if monitor_dir is None else os.path.join(monitor_dir, f"env_{rank}")
        fns.append(make_env(obs_type=obs_type, seed=seed, rank=rank, monitor_path=monitor_path,
                            aug=aug, **env_kwargs))
    # forkserver: pymunk/pygame state does not survive a plain fork cleanly
    venv = SubprocVecEnv(fns, start_method="forkserver") if use_subproc and n_envs > 1 else DummyVecEnv(fns)
    # ONE seeding path for training and play alike. SB3 hands seed + idx to each env at the next
    # reset and then clears them, which is the gymnasium convention; seeding inside the thunk as
    # well would leave the two scripts on different paths, since only training re-seeds via the
    # algorithm's set_random_seed.
    venv.seed(seed)
    return venv


def env_kwargs_from(cfg, obs_type):
    """The PushTGymEnv kwargs an arm takes, out of a flat config dict (CLI args or args.yaml)."""
    kwargs = {k: cfg[k] for k in ENV_KEYS}
    kwargs["reward_mode"] = cfg["reward"]
    kwargs["shaping_coef"] = cfg.get("shaping_coef", D["shaping_coef"])
    kwargs["shaping_potential"] = cfg.get("shaping_potential", D["shaping_potential"])
    for key in ("progress_coef", "success_bonus", "block_zero_coverage"):
        kwargs[key] = cfg.get(key, D[key])
    kwargs["shaping_gamma"] = cfg.get("gamma", D["gamma"])
    if obs_type == "keypoint":
        kwargs.update({k: cfg[k] for k in KEYPOINT_ONLY_KEYS})
    return kwargs
