"""Waypoint-progress trackers for the Veritas PushT plans (v5 dual, v3 pusher-only),
on the 96x96 obs image.

``WaypointTracker`` (now in ``diffusion_policy.veritas.tracker``) is a line-for-line
port of ``VeritasVerifier.update``
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
from diffusion_policy.veritas.tracker import (  # noqa: F401  (WaypointTracker re-exported)
    TrackerParams, WaypointTracker, anchored)

IMAGE_SIZE = 96
# Veritas's pixel constants (sigma 50, skip margin 10, stall eps 1) are tuned for a
# 640-wide frame; distances here live on 96 px. Rescaled by the shared tracker.
PARAMS = TrackerParams.for_image_width(IMAGE_SIZE)
SIGMA_PX = PARAMS.sigma_px             # 7.5
SKIP_MARGIN_PX = PARAMS.skip_margin_px # 1.5
STALL_EPS_PX = PARAMS.stall_eps_px     # 0.15


def _anchored(kps_list, tols, holds, skips, start_kps):
    return anchored(kps_list, tols, holds, skips, start_kps, params=PARAMS)


def agent_px_from_pos(agent_pos) -> np.ndarray:
    """(..., 2) workspace agent position -> (..., 1, 2) pixel keypoint set."""
    px = np.asarray(agent_pos, dtype=np.float64) * (IMAGE_SIZE / WORKSPACE_SIZE)
    return px[..., None, :]


def t_kps_px_from_pose(block_pose) -> np.ndarray:
    """(..., 3) workspace block pose -> (..., 8, 2) pixel T keypoints (T_VERTS order)."""
    kps = keypoints_at_pose(np.asarray(block_pose, dtype=np.float64))
    return kps.astype(np.float64) * (IMAGE_SIZE / WORKSPACE_SIZE)


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
