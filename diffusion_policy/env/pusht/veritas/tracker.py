"""Waypoint-progress trackers for the Veritas PushT plans (v5 dual, v3 pusher-only),
on the 96x96 obs image.

``WaypointTracker`` is a line-for-line port of ``VeritasVerifier.update``
(veritas/src/agent/veritas_verifier/veritas_verifier.py): hysteresis hold, min_hold,
stall detection, the ahead/stalled/transit skip rules, and the
``0.7*progress + 0.3*exp(-d/sigma)`` score. Two deliberate departures:

* **Waypoints are (K, 2) keypoint sets**, not single points, and the distance is the
  MEAN per-keypoint distance -- the metric ``t_goal_distance`` and the v5 prompt's
  ``tol_px`` are defined in. At K=1 (the pusher track) this is exactly Veritas's
  Euclidean distance, so nothing changes for it. Veritas's class cannot express this:
  its schema fixes ``uv`` to one point, and a flattened 16-D L2 norm would sit ~sqrt(8)
  above the tolerance scale.
* **Replan is dropped**, as it already is in Veritas's own eval ("Replan disabled for
  static trace per requirement"); the plan is fixed per episode.

Pixel-distance constants are Veritas's, rescaled by 96/640 (the same ratio the prompt
tolerances were rescaled by; Veritas tunes for a 640-wide frame). Frame-count constants
are per env step and stay as they are.

``DualTracker`` runs one tracker per v5 list -- ``pusher_waypoints`` against the agent
pixel, ``t_poses`` (via ``pose_keypoints_px``) against the achieved T keypoints -- and
scores their MEAN. Both tracks are 0..1, so the unweighted mean keeps them on equal
footing; the pusher track is what varies before contact, when every candidate's T is
identical.
"""

from __future__ import annotations

import json
import pathlib
from collections import deque
from typing import List, Optional

import gym
import numpy as np

from diffusion_policy.env.pusht.feedback_util import keypoints_at_pose
from diffusion_policy.env.pusht.veritas.pusht_client import (
    WORKSPACE_SIZE, pose_keypoints_px)
from diffusion_policy.env.pusht.veritas.schemas import VeritasDualPlan, VeritasPlan

IMAGE_SIZE = 96
# Veritas's pixel constants (sigma 50, skip margin 10, stall eps 1) are tuned for a
# 640-wide frame; distances here live on 96 px.
_SCALE = IMAGE_SIZE / 640.0
SIGMA_PX = 50.0 * _SCALE           # 7.5
SKIP_MARGIN_PX = 10.0 * _SCALE     # 1.5
STALL_EPS_PX = 1.0 * _SCALE        # 0.15
HYSTERESIS_RATIO = 1.25
STALL_HIST_LEN = 6
STALL_FRAMES = 6
SKIP_FRAMES = 6
TRANSIT_SKIP_FRAMES = 3
TRANSIT_SKIP_T = 0.4
SKIP_PENALTY = 0.05
PROGRESS_WEIGHT = 0.7
CLOSENESS_WEIGHT = 0.3


def agent_px_from_pos(agent_pos) -> np.ndarray:
    """(..., 2) workspace agent position -> (..., 1, 2) pixel keypoint set."""
    px = np.asarray(agent_pos, dtype=np.float64) * (IMAGE_SIZE / WORKSPACE_SIZE)
    return px[..., None, :]


def t_kps_px_from_pose(block_pose) -> np.ndarray:
    """(..., 3) workspace block pose -> (..., 8, 2) pixel T keypoints (T_VERTS order)."""
    kps = keypoints_at_pose(np.asarray(block_pose, dtype=np.float64))
    return kps.astype(np.float64) * (IMAGE_SIZE / WORKSPACE_SIZE)


class WaypointTracker:
    """Progress along one waypoint list. See the module docstring for provenance."""

    def __init__(self, waypoints_px: np.ndarray, tol_px, min_hold, skippable):
        """
        Args:
            waypoints_px: (N, K, 2) pixel keypoint sets, one per waypoint. The caller
                prepends the start anchor (Veritas's ``initialize_episode`` behaviour)
                before constructing; nothing is prepended here.
            tol_px / min_hold / skippable: per-waypoint, length N.
        """
        self.waypoints = np.asarray(waypoints_px, dtype=np.float64)
        assert self.waypoints.ndim == 3 and self.waypoints.shape[-1] == 2, \
            f'waypoints_px must be (N, K, 2), got {self.waypoints.shape}'
        n = len(self.waypoints)
        self.tol_px = np.asarray(tol_px, dtype=np.float64)
        self.min_hold = np.asarray(min_hold, dtype=int)
        self.skippable = np.asarray(skippable, dtype=bool)
        assert self.tol_px.shape == (n,) and self.min_hold.shape == (n,) \
            and self.skippable.shape == (n,)
        self.i = 0
        self.hold = 0
        self.skip_hold_next = 0
        self.transit_hold = 0
        self.d_hist: deque = deque(maxlen=STALL_HIST_LEN)
        self.stall_counter = 0
        self.last_score = 0.0
        self.last_d_px = float('nan')

    @property
    def n(self) -> int:
        return len(self.waypoints)

    @property
    def done(self) -> bool:
        return self.i >= self.n

    def copy(self) -> 'WaypointTracker':
        """Independent copy for scoring a candidate without touching the live state.
        The waypoint arrays are immutable here and shared, so this is cheap."""
        new = object.__new__(WaypointTracker)
        new.__dict__.update(self.__dict__)
        new.d_hist = deque(self.d_hist, maxlen=STALL_HIST_LEN)
        return new

    def _d(self, point: np.ndarray, idx: int) -> float:
        """Mean per-keypoint distance from ``point`` (K, 2) to waypoint ``idx``."""
        return float(np.linalg.norm(point - self.waypoints[idx], axis=-1).mean())

    def update(self, point) -> dict:
        """Advance on one observed frame. ``point`` is (K, 2) pixels."""
        point = np.asarray(point, dtype=np.float64)
        if self.done:
            self.last_score = 1.0
            return {'done': True, 'score': 1.0, 'active_idx': self.i, 'd_px': 0.0,
                    'skip_reason': None}

        d_px = self._d(point, self.i)
        tol_on = self.tol_px[self.i]
        tol_off = HYSTERESIS_RATIO * tol_on

        # Hysteresis hold: dead-band between tol_on and tol_off avoids chatter.
        if d_px < tol_on:
            self.hold += 1
        elif d_px > tol_off:
            self.hold = 0

        # Completion
        if self.hold >= self.min_hold[self.i]:
            self._advance()
            if self.done:
                self.last_score, self.last_d_px = 1.0, d_px
                return {'done': True, 'score': 1.0, 'active_idx': self.i,
                        'd_px': d_px, 'skip_reason': None}
            d_px = self._d(point, self.i)
            tol_on = self.tol_px[self.i]

        # Stall detection
        self.d_hist.append(d_px)
        if len(self.d_hist) == self.d_hist.maxlen:
            mean_d = float(np.mean(self.d_hist))
            trend = float((self.d_hist[-1] - self.d_hist[0]) / self.d_hist.maxlen)
            if (mean_d > tol_on) and (trend >= -STALL_EPS_PX):
                self.stall_counter += 1
            else:
                self.stall_counter = 0

        # Skip-ahead (max 1 step). The segment projection runs on the flattened
        # (2K,) vectors; at K=1 that is Veritas's own point projection.
        skip_reason = None
        if self.i + 1 < self.n and self.skippable[self.i + 1]:
            d_next = self._d(point, self.i + 1)
            if d_next < self.tol_px[self.i + 1]:
                self.skip_hold_next += 1
            else:
                self.skip_hold_next = 0

            seg = (self.waypoints[self.i + 1] - self.waypoints[self.i]).ravel()
            seg_len2 = float(np.dot(seg, seg))
            rel = (point - self.waypoints[self.i]).ravel()
            t_along = float(np.dot(rel, seg)) / seg_len2 if seg_len2 > 1e-6 else 0.0
            if t_along > TRANSIT_SKIP_T and d_px > tol_on:
                self.transit_hold += 1
            else:
                self.transit_hold = 0
            transit_skip = self.transit_hold >= TRANSIT_SKIP_FRAMES

            ahead = d_next + SKIP_MARGIN_PX < d_px
            next_satisfied = self.skip_hold_next >= max(int(self.min_hold[self.i + 1]), 1)
            stalled_flag = self.stall_counter >= STALL_FRAMES
            if (transit_skip
                    or (stalled_flag and next_satisfied)
                    or (ahead and next_satisfied and self.skip_hold_next >= SKIP_FRAMES)):
                skip_reason = ('transit_skip' if transit_skip
                               else ('stalled_skip' if stalled_flag else 'ahead_skip'))
                self._advance()
                if not self.done:
                    d_px = self._d(point, self.i)

        # Scoring
        progress = self.i / max(1, self.n)
        closeness = float(np.exp(-d_px / SIGMA_PX)) if np.isfinite(d_px) else 0.0
        score = PROGRESS_WEIGHT * progress + CLOSENESS_WEIGHT * closeness
        if skip_reason is not None:
            score = max(0.0, min(1.0, score - SKIP_PENALTY))
        if self.done:
            score = 1.0
        self.last_score, self.last_d_px = score, d_px
        return {'done': self.done, 'score': score, 'active_idx': self.i,
                'd_px': d_px, 'skip_reason': skip_reason}

    def _advance(self):
        self.i += 1
        self.hold = 0
        self.skip_hold_next = 0
        self.transit_hold = 0
        self.d_hist.clear()
        self.stall_counter = 0


def _anchored(kps_list: List[np.ndarray], tols, holds, skips, start_kps: np.ndarray):
    """Prepend the start anchor and pin the endpoints, as Veritas's
    ``initialize_episode`` does: anchor at the current position with the first
    waypoint's tol, min_hold 1; first and last skippable forced False."""
    wps = np.stack([np.asarray(start_kps, dtype=np.float64)] + list(kps_list))
    tols = np.concatenate([[tols[0]], tols])
    holds = np.concatenate([[1], holds])
    skips = np.concatenate([[False], skips])
    skips[-1] = False
    return WaypointTracker(wps, tols, holds, skips)


class DualTracker:
    """The tracks of one episode's plan: pusher waypoints, and T poses when it has them.

    A v5 ``VeritasDualPlan`` carries both tracks; a v3 ``VeritasPlan`` (pusher points
    only, the ``wp_v3*`` values) has no T track, and ``self.t`` is then None -- the
    score is the pusher track's alone, so one class serves both and every consumer
    (env wrapper, verifier, demo replay) is agnostic to which plan kind is loaded.

    ``update`` takes the WORKSPACE agent position and block pose -- exactly what the
    sim's low-dim obs / info carry -- and converts to pixels here, so every caller
    shares one conversion.
    """

    def __init__(self, plan, agent_pos, block_pose):
        if isinstance(plan, VeritasDualPlan):
            pusher_wps, t_poses = plan.pusher_waypoints, plan.t_poses
            assert pusher_wps and t_poses, \
                'v5 plan must carry both pusher_waypoints and t_poses'
        else:
            assert isinstance(plan, VeritasPlan) and plan.waypoints, \
                f'plan must be a VeritasDualPlan or a non-empty VeritasPlan, got {plan!r}'
            pusher_wps, t_poses = plan.waypoints, None
        self.plan = plan
        self.pusher = _anchored(
            [np.asarray(wp.uv, dtype=np.float64)[None, :] for wp in pusher_wps],
            [wp.tol_px for wp in pusher_wps],
            [wp.min_hold for wp in pusher_wps],
            [wp.skippable for wp in pusher_wps],
            agent_px_from_pos(agent_pos))
        self.t = None if t_poses is None else _anchored(
            [pose_keypoints_px(wp, IMAGE_SIZE) for wp in t_poses],
            [wp.tol_px for wp in t_poses],
            [wp.min_hold for wp in t_poses],
            [wp.skippable for wp in t_poses],
            t_kps_px_from_pose(block_pose))

    def copy(self) -> 'DualTracker':
        new = object.__new__(DualTracker)
        new.plan = self.plan
        new.pusher = self.pusher.copy()
        new.t = None if self.t is None else self.t.copy()
        return new

    def update(self, agent_pos, block_pose) -> dict:
        """One env step: workspace agent position (2,) and block pose (3,)."""
        p = self.pusher.update(agent_px_from_pos(agent_pos)[0])
        t = None if self.t is None else self.t.update(t_kps_px_from_pose(block_pose))
        return {'pusher': p, 't': t, 'score': self.score}

    @property
    def score(self) -> float:
        """Mean of the present 0..1 track scores -- the wp_v5 / wp_v3 ranking scalar."""
        if self.t is None:
            return self.pusher.last_score
        return 0.5 * (self.pusher.last_score + self.t.last_score)

    @property
    def progress(self) -> tuple:
        """(pusher, t) fraction of waypoints latched, anchor included; t is NaN for a
        pusher-only plan."""
        return (self.pusher.i / max(1, self.pusher.n),
                float('nan') if self.t is None else self.t.i / max(1, self.t.n))


# Where each waypoint value's plans live by default. The version in the value NAMES the
# plan kind the run is scored against (see pusht_verifier.WAYPOINT_VALUES), so the
# default cannot pair a value with the wrong prompt's plans. Lives HERE rather than in
# eval_search_pusht.py (which re-exports it) so the training workspace can resolve it
# without paying that module's ~88s/440MB import.
WAYPOINT_PLAN_DIRS = {
    'wp_v5': 'media/veritas_pusht/v5_pusher_and_t',
    'wp_v3': 'media/veritas_pusht/v3_end_outside',
}


def load_split_plans(plan_dir, idxs, split, value):
    """One plan per episode of a split, refusing on ANY gap.

    A missing or failed plan is a hole in the split, not something to skip over: a rate
    computed on the episodes that happen to have plans is not comparable to anything.
    The plan KIND is validated against the value -- wp_v5* declares the dual-track
    signal, so scoring it against pusher-only plans (or the reverse) would let the run
    directory lie about what ranked it.
    """
    plan_dir = pathlib.Path(plan_dir)
    want_dual = value.startswith('wp_v5')
    plans, missing = [], []
    for i in idxs:
        path = plan_dir / f'ep{int(i)}.json'
        try:
            plan = load_plan(path)
        except (FileNotFoundError, ValueError):
            missing.append(int(i))
            continue
        if isinstance(plan, VeritasDualPlan) != want_dual:
            raise ValueError(
                f'{path} is a {"dual" if not want_dual else "pusher-only"} plan but '
                f'verifier value {value} declares the '
                f'{"dual-track (v5)" if want_dual else "pusher-only (v3)"} signal. '
                f'Point the plan dir at the matching prompt\'s plans.')
        plans.append(plan)
    if missing:
        raise FileNotFoundError(
            f'{split}: no usable plan for episode(s) {missing} in {plan_dir}/. Generate '
            f'them with scripts/veritas_pusht_overlays.py (re-run until every episode '
            f'has a `plan` field); a partial split cannot be scored comparably.')
    return plans


class TrackerSnapshotStore:
    """The live tracker state at EVERY recorded frame of the planned episodes.

    THE CANONICAL ANCHOR/UPDATE CONVENTION -- every consumer must match it or its
    snapshots are one update out of step with what eval's verifier branches from:
    anchor ``DualTracker(plan, agent_pos[s], block_pos[s])`` at the episode's reset row
    ``s`` with ZERO updates, then one ``update`` per POST-STEP row ``s+1 ..`` -- exactly
    ``VeritasTrackerWrapper`` (reset builds, each real step updates). Every recorded row
    is replayed: the tracker's hold/stall counters are per-frame, so skipping rows makes
    a different tracker. (``scripts/veritas_tracker_demo_check.py`` also updates on the
    anchor row -- a diagnostic-only deviation; do not copy it.)

    Built once per run: one demo replay per episode, one light copy per frame (the
    waypoint arrays are shared across copies -- see ``WaypointTracker.copy``). Lookup
    hands out a fresh ``.copy()`` every time, so the stored snapshots can never be
    advanced by a caller.
    """

    def __init__(self, agent_pos, block_pos, episode_ends, plans_by_ep):
        """
        Args:
            agent_pos: (N, 2) workspace agent positions, all episodes concatenated.
            block_pos: (N, 3) workspace block poses.
            episode_ends: (E,) exclusive episode end frames (ReplayBuffer convention).
            plans_by_ep: {episode_idx: plan} -- only these episodes get snapshots.
        """
        agent_pos = np.asarray(agent_pos, dtype=np.float64)
        block_pos = np.asarray(block_pos, dtype=np.float64)
        self._ends = np.asarray(episode_ends)
        starts = np.concatenate([[0], self._ends[:-1]])
        self._snaps = {}
        for e, plan in plans_by_ep.items():
            s, t = int(starts[e]), int(self._ends[e])
            tracker = DualTracker(plan, agent_pos[s], block_pos[s])
            snaps = [tracker.copy()]                 # frame s: the zero-update anchor
            for f in range(s + 1, t):
                tracker.update(agent_pos[f], block_pos[f])
                snaps.append(tracker.copy())
            self._snaps[int(e)] = snaps

    @classmethod
    def from_replay_buffer(cls, replay_buffer, plans_by_ep):
        return cls(replay_buffer['agent_pos'], replay_buffer['block_pos'],
                   replay_buffer.episode_ends[:], plans_by_ep)

    def for_frame(self, frame: int) -> DualTracker:
        """A fresh copy of the tracker state at absolute ``frame``."""
        # side='right': episode_ends are EXCLUSIVE, so frame ends[k]-1 is episode k's
        # last row and frame ends[k] is episode k+1's first.
        e = int(np.searchsorted(self._ends, frame, side='right'))
        if e not in self._snaps:
            raise KeyError(
                f'frame {frame} is in episode {e}, which this store holds no plan for; '
                f'it was built over episodes {sorted(self._snaps)}.')
        start = 0 if e == 0 else int(self._ends[e - 1])
        return self._snaps[e][frame - start].copy()

    def for_dataset_index(self, dataset, i: int) -> DualTracker:
        """The tracker state at dataset sample ``i``'s DECISION frame.

        Keys off ``dataset.sampler.indices`` -- never off arithmetic over episode
        lengths: a transition-filtered dataset drops rows, and each split copy rebuilds
        its own sampler, so the index -> frame map belongs to the dataset in hand.
        """
        from diffusion_policy.dataset.pusht_image_dataset import decision_frame
        buffer_start, _, sample_start, _ = dataset.sampler.indices[i]
        frame = decision_frame(int(buffer_start), int(sample_start), dataset.n_obs_steps)
        return self.for_frame(frame)


def load_plan(path):
    """The clipped plan out of an overlay-script ``ep{idx}.json`` -- a
    ``VeritasDualPlan`` (v5, both tracks) or a ``VeritasPlan`` (v3, pusher only),
    told apart by which fields the record carries.

    ``plan`` is the field ``scripts/veritas_pusht_overlays.py`` writes back after
    clipping; a json without it is a failed Gemini call and is refused rather than
    fallen back from (Veritas's own fallback -- one waypoint at the image centre --
    would silently steer scoring toward the goal).
    """
    rec = json.loads(pathlib.Path(path).read_text())
    if 'plan' not in rec:
        raise ValueError(f'{path}: no clipped `plan` recorded -- the Gemini call failed; '
                         f're-run scripts/veritas_pusht_overlays.py for this episode.')
    plan = rec['plan']
    if 'pusher_waypoints' in plan:
        return VeritasDualPlan(**plan)
    return VeritasPlan(**plan)


class VeritasTrackerWrapper(gym.Wrapper):
    """Advances the episode's DualTracker on EVERY real env step.

    Sits INSIDE MultiStepWrapper, which is the whole point: the outer wrappers only
    retain the last ``n_obs_steps`` of info, so the per-step trace of an 8-step action
    chunk is not recoverable outside -- this wrapper sees each step as it happens. This
    is the passive Veritas feedback, and its tracker is the state candidates branch
    from (the verifier copies it per candidate).

    The plan is handed in like ``reset_to_state``: the eval loop's dill'd init fn calls
    ``set_veritas_plan`` (forwarded down the wrapper stack by gym's ``__getattr__``)
    before ``reset``. With no plan set the wrapper is inert, so it can sit in the env
    stack unconditionally.
    """

    def __init__(self, env):
        super().__init__(env)
        self.veritas_plan = None
        self.tracker = None

    def set_veritas_plan(self, plan: Optional[VeritasDualPlan]):
        self.veritas_plan = plan

    def _state(self):
        u = self.unwrapped
        return (np.array(u.agent.position, dtype=np.float64),
                np.array(list(u.block.position) + [u.block.angle], dtype=np.float64))

    def sim_state(self):
        """(agent_pos (2,), block_pose (3,)) in workspace units. Public because gym's
        Wrapper.__getattr__ refuses underscored names, so `env.call('_state')` through
        a vector env kills the worker."""
        return self._state()

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        if self.veritas_plan is not None:
            agent_pos, block_pose = self._state()
            self.tracker = DualTracker(self.veritas_plan, agent_pos, block_pose)
        else:
            self.tracker = None
        return obs

    def step(self, action):
        out = super().step(action)
        if self.tracker is not None:
            agent_pos, block_pose = self._state()
            self.tracker.update(agent_pos, block_pose)
        return out


class ShadowTrackerWrapper(gym.Wrapper):
    """Extra DualTrackers on OTHER plans, advanced on every real step; analysis only.

    The rollout ranks with VeritasTrackerWrapper's tracker. This one lets the same
    candidates also be scored as another waypoint value would score them (e.g. wp_v5
    while wp_v3 steers), from that value's own live episode progress. Nothing reads it
    during a rollout, so adding it changes no trajectory. Attribute names differ from
    VeritasTrackerWrapper's so gym's __getattr__ forwarding cannot confuse the two.
    """

    def __init__(self, env):
        super().__init__(env)
        self.shadow_plans = {}
        self.shadow_trackers = {}

    def set_shadow_plans(self, plans: dict):
        """{value name: plan}; set before reset, like set_veritas_plan."""
        self.shadow_plans = dict(plans or {})

    _state = VeritasTrackerWrapper._state

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        agent_pos, block_pose = self._state()
        self.shadow_trackers = {k: DualTracker(p, agent_pos, block_pose)
                                for k, p in self.shadow_plans.items()}
        return obs

    def step(self, action):
        out = super().step(action)
        if self.shadow_trackers:
            agent_pos, block_pose = self._state()
            for tr in self.shadow_trackers.values():
                tr.update(agent_pos, block_pose)
        return out
