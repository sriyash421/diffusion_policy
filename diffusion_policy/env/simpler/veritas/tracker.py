"""The live per-episode tracker for SimplerEnv, the env wrapper that advances it, and
the per-frame targets encoding the verifier reads.

Targets vector (what ``VeritasPixelVerifier`` consumes, ``TARGETS_DIM`` floats):
``[n_valid, tol_px_active, u0, v0, u1, v1, ..., u_{K-1}, v_{K-1}]`` for the active
waypoint and the next ``K-1`` (upstream ``lookahead_k``); unused slots are -1 and
``n_valid == 0`` means the plan is done. At eval the wrapper appends the source
object's live pixel ``[u_src, v_src]`` (-1 if unknown; ``TARGETS_DIM_WITH_SOURCE``),
which upstream scores against once the plan is done. It is the state a candidate chunk is scored
against, so it travels with the obs: the eval loop reads it from the wrapper, and
training reads it from a precomputed per-frame array (``SimplerTrackerSnapshotStore``).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import gym
import numpy as np

from diffusion_policy.env.simpler.veritas.camera import CameraModel, tcp_xyz
from diffusion_policy.env.simpler.veritas.hints import object_hints
from diffusion_policy.veritas.schemas import VeritasPlan
from diffusion_policy.veritas.tracker import (TrackerParams, WaypointTracker,
                                              anchored_from_plan)

LOOKAHEAD_K = 3
TARGETS_DIM = 2 + 2 * LOOKAHEAD_K
TARGETS_DIM_WITH_SOURCE = TARGETS_DIM + 2
EMPTY_TARGETS = np.full(TARGETS_DIM, -1.0, dtype=np.float32)
EMPTY_TARGETS[0] = 0.0


def targets_vector(tracker: Optional[WaypointTracker], k: int = LOOKAHEAD_K) -> np.ndarray:
    """Encode the tracker's upcoming waypoints; ``None``/done -> ``n_valid = 0``."""
    out = np.full(2 + 2 * k, -1.0, dtype=np.float32)
    out[0] = 0.0
    if tracker is None or tracker.done:
        return out
    idxs = tracker.upcoming(k)
    out[0] = len(idxs)
    out[1] = tracker.tol_px[tracker.i]
    for j, i in enumerate(idxs):
        out[2 + 2 * j: 4 + 2 * j] = tracker.waypoints[i, 0]
    return out


def decode_targets(vec) -> tuple:
    """(upcoming (n_valid, 2), tol_px) from a targets vector."""
    vec = np.asarray(vec, dtype=np.float64)
    n = int(round(vec[0]))
    return vec[2:2 + 2 * n].reshape(n, 2), float(vec[1])


class SimplerEpisodeTracker:
    """An anchored ``WaypointTracker`` fed world TCP positions through the camera."""

    def __init__(self, plan: VeritasPlan, camera: CameraModel, start_xyz,
                 params: Optional[TrackerParams] = None):
        self.plan = plan
        self.camera = camera
        self.params = params or TrackerParams.for_image_width(camera.image_hw[1])
        self.tracker = anchored_from_plan(plan, camera.project(start_xyz), self.params)
        self.last = None

    def update(self, tcp_world_xyz) -> dict:
        px = self.camera.project(np.asarray(tcp_world_xyz, dtype=np.float64))
        self.last = self.tracker.update(px[None, :])
        return self.last

    def targets_vector(self, k: int = LOOKAHEAD_K) -> np.ndarray:
        return targets_vector(self.tracker, k)

    @property
    def progress(self) -> float:
        return self.tracker.progress

    @property
    def done(self) -> bool:
        return self.tracker.done


class VeritasTrackerWrapper(gym.Wrapper):
    """Plans at reset and advances the tracker on EVERY executed env step.

    ``plan_provider(seed, frame=..., instruction=..., eef_pixel=..., object_hints=...)``
    returns the episode's ``VeritasPlan`` (see ``veritas.plan_store.CachedPlanProvider``).
    The wrapper must sit directly on the SimplerEnv gym env, below any action-chunk
    or time-limit wrapper, so it sees each step.
    """

    def __init__(self, env, camera: CameraModel,
                 plan_provider: Callable[..., VeritasPlan],
                 params: Optional[TrackerParams] = None, lookahead_k: int = LOOKAHEAD_K,
                 hint_provider: Optional[Callable] = None):
        super().__init__(env)
        self.camera = camera
        self.plan_provider = plan_provider
        # None: object hints from sim poses. Else ``hint_provider(env_base, frame,
        # instruction, eef_pixel) -> (hints, meta)`` (upstream Grounded-SAM-2), run only
        # when the plan is not already cached.
        self.hint_provider = hint_provider
        self.params = params
        self.lookahead_k = lookahead_k
        self.episode: Optional[SimplerEpisodeTracker] = None
        self.last_seed = None

    def reset(self, *, seed=None, options=None, **kwargs):
        out = self.env.reset(seed=seed, options=options, **kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        self.camera.assert_matches(obs)
        self.last_seed = seed
        from simpler_env.utils.env.observation_utils import \
            get_image_from_maniskill2_obs_dict
        frame = get_image_from_maniskill2_obs_dict(self.env.unwrapped, obs)
        start = tcp_xyz(obs)
        instruction = self.env.unwrapped.get_language_instruction()
        eef_pixel = self.camera.project(start).tolist()
        if self.hint_provider is None:
            hint_kw = dict(object_hints=object_hints(self.env.unwrapped, self.camera))
        else:
            hp, base = self.hint_provider, self.env.unwrapped
            hint_kw = dict(hint_fn=lambda: hp(base, frame, instruction, eef_pixel))
        plan = self.plan_provider(seed, frame=frame, instruction=instruction,
                                  eef_pixel=eef_pixel, **hint_kw)
        self.episode = SimplerEpisodeTracker(plan, self.camera, start, self.params)
        return out

    def step(self, action):
        out = self.env.step(action)
        obs = out[0]
        if self.episode is not None:
            self.episode.update(tcp_xyz(obs))
        return out

    def source_pixel(self) -> np.ndarray:
        """The source object's live pixel from its sim pose, ``[-1, -1]`` if unknown."""
        src = [h['uv'] for h in object_hints(self.env.unwrapped, self.camera)
               if h['role'] == 'source']
        return np.asarray(src[0] if src else (-1.0, -1.0), dtype=np.float32)

    def targets_vector(self) -> np.ndarray:
        """Tracker targets plus the source pixel (``TARGETS_DIM_WITH_SOURCE`` floats)."""
        base = (self.episode.targets_vector(self.lookahead_k) if self.episode is not None
                else EMPTY_TARGETS.copy())
        return np.concatenate([base, self.source_pixel()])

    def tracker_status(self) -> Dict[str, float]:
        if self.episode is None:
            return {}
        t = self.episode.tracker
        return {'wp_progress': t.progress, 'wp_active_idx': int(t.i), 'wp_n': int(t.n),
                'wp_done': bool(t.done), 'wp_score': float(t.last_score)}


class SimplerTrackerSnapshotStore:
    """Per-frame targets for planned demo episodes, replayed from the recorded TCP path.

    THE CANONICAL ANCHOR/UPDATE CONVENTION (matches ``VeritasTrackerWrapper``): anchor
    at the episode's reset row with ZERO updates, then one ``update`` per following
    row. Rows of episodes without a plan are ``EMPTY_TARGETS`` (``n_valid = 0``).
    """

    def __init__(self, ee_xyz, episode_ends, plans_by_ep: Dict[int, VeritasPlan],
                 camera: CameraModel, params: Optional[TrackerParams] = None,
                 lookahead_k: int = LOOKAHEAD_K):
        ee_xyz = np.asarray(ee_xyz, dtype=np.float64)[:, :3]
        ends = np.asarray(episode_ends)
        starts = np.concatenate([[0], ends[:-1]])
        self.targets = np.tile(EMPTY_TARGETS, (len(ee_xyz), 1))
        self.final_progress: Dict[int, float] = {}
        for e, plan in plans_by_ep.items():
            s, t = int(starts[e]), int(ends[e])
            ep = SimplerEpisodeTracker(plan, camera, ee_xyz[s], params)
            self.targets[s] = ep.targets_vector(lookahead_k)
            for f in range(s + 1, t):
                ep.update(ee_xyz[f])
                self.targets[f] = ep.targets_vector(lookahead_k)
            self.final_progress[int(e)] = ep.progress
