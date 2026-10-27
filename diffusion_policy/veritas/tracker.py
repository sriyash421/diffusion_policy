"""Waypoint-progress tracker: a port of ``VeritasVerifier.update``
(veritas/src/agent/veritas_verifier/veritas_verifier.py): hysteresis hold, min_hold,
stall detection, the ahead/stalled/transit skip rules, and the
``0.7*progress + 0.3*exp(-d/sigma)`` score.

Departures from upstream, shared by every env integration:

* **Waypoints are (K, 2) keypoint sets** and the distance is the MEAN per-keypoint
  distance. At K=1 (a single end-effector pixel) this is exactly Veritas's Euclidean
  distance, so single-point plans behave as upstream.
* **Replan is dropped**, as it already is in Veritas's own eval; the plan is fixed per
  episode.
* **Pixel constants are parameters** (``TrackerParams``). Upstream tunes them for a
  640-wide frame; ``TrackerParams.for_image_width(w)`` rescales the pixel-valued ones
  by ``w / 640`` and leaves the frame-count ones alone.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import List, Optional

import numpy as np

UPSTREAM_IMAGE_WIDTH = 640.0


@dataclasses.dataclass(frozen=True)
class TrackerParams:
    """Upstream ``VeritasVerifier`` constants. Pixel-valued fields are for a 640-wide
    frame; use ``for_image_width`` for anything else."""

    sigma_px: float = 50.0
    skip_margin_px: float = 10.0
    stall_eps_px: float = 1.0
    hysteresis_ratio: float = 1.25
    stall_hist_len: int = 6
    stall_frames: int = 6
    skip_frames: int = 6
    transit_skip_frames: int = 3
    transit_skip_t: float = 0.4
    skip_penalty: float = 0.05
    progress_weight: float = 0.7
    closeness_weight: float = 0.3

    @classmethod
    def for_image_width(cls, width: float) -> 'TrackerParams':
        s = float(width) / UPSTREAM_IMAGE_WIDTH
        return cls(sigma_px=50.0 * s, skip_margin_px=10.0 * s, stall_eps_px=1.0 * s)


class WaypointTracker:
    """Progress along one waypoint list. See the module docstring for provenance."""

    def __init__(self, waypoints_px: np.ndarray, tol_px, min_hold, skippable,
                 params: Optional[TrackerParams] = None):
        """
        Args:
            waypoints_px: (N, K, 2) pixel keypoint sets, one per waypoint. The caller
                prepends the start anchor (Veritas's ``initialize_episode`` behaviour)
                before constructing; nothing is prepended here. See ``anchored``.
            tol_px / min_hold / skippable: per-waypoint, length N.
            params: pixel/frame constants; default is upstream's 640-px tuning.
        """
        self.params = params or TrackerParams()
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
        self.d_hist: deque = deque(maxlen=self.params.stall_hist_len)
        self.stall_counter = 0
        self.last_score = 0.0
        self.last_d_px = float('nan')

    @property
    def n(self) -> int:
        return len(self.waypoints)

    @property
    def done(self) -> bool:
        return self.i >= self.n

    @property
    def progress(self) -> float:
        """Fraction of waypoints latched, anchor included."""
        return self.i / max(1, self.n)

    def copy(self) -> 'WaypointTracker':
        """Independent copy for scoring a candidate without touching the live state.
        The waypoint arrays are immutable here and shared, so this is cheap."""
        new = object.__new__(WaypointTracker)
        new.__dict__.update(self.__dict__)
        new.d_hist = deque(self.d_hist, maxlen=self.params.stall_hist_len)
        return new

    def upcoming(self, k: int) -> List[int]:
        """Indices of the active waypoint and the next ``k-1`` (Veritas's
        ``_veritas_active_waypoint`` lookahead); empty once done."""
        return list(range(self.i, min(self.i + k, self.n)))

    def _d(self, point: np.ndarray, idx: int) -> float:
        """Mean per-keypoint distance from ``point`` (K, 2) to waypoint ``idx``."""
        return float(np.linalg.norm(point - self.waypoints[idx], axis=-1).mean())

    def update(self, point) -> dict:
        """Advance on one observed frame. ``point`` is (K, 2) pixels."""
        P = self.params
        point = np.asarray(point, dtype=np.float64)
        if self.done:
            self.last_score = 1.0
            return {'done': True, 'score': 1.0, 'active_idx': self.i, 'd_px': 0.0,
                    'skip_reason': None}

        d_px = self._d(point, self.i)
        tol_on = self.tol_px[self.i]
        tol_off = P.hysteresis_ratio * tol_on

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
            if (mean_d > tol_on) and (trend >= -P.stall_eps_px):
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
            if t_along > P.transit_skip_t and d_px > tol_on:
                self.transit_hold += 1
            else:
                self.transit_hold = 0
            transit_skip = self.transit_hold >= P.transit_skip_frames

            ahead = d_next + P.skip_margin_px < d_px
            next_satisfied = self.skip_hold_next >= max(int(self.min_hold[self.i + 1]), 1)
            stalled_flag = self.stall_counter >= P.stall_frames
            if (transit_skip
                    or (stalled_flag and next_satisfied)
                    or (ahead and next_satisfied and self.skip_hold_next >= P.skip_frames)):
                skip_reason = ('transit_skip' if transit_skip
                               else ('stalled_skip' if stalled_flag else 'ahead_skip'))
                self._advance()
                if not self.done:
                    d_px = self._d(point, self.i)

        # Scoring
        progress = self.i / max(1, self.n)
        closeness = float(np.exp(-d_px / P.sigma_px)) if np.isfinite(d_px) else 0.0
        score = P.progress_weight * progress + P.closeness_weight * closeness
        if skip_reason is not None:
            score = max(0.0, min(1.0, score - P.skip_penalty))
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


def anchored(kps_list: List[np.ndarray], tols, holds, skips, start_kps: np.ndarray,
             params: Optional[TrackerParams] = None) -> WaypointTracker:
    """Prepend the start anchor and pin the endpoints, as Veritas's
    ``initialize_episode`` does: anchor at the current position with the first
    waypoint's tol, min_hold 1; first and last skippable forced False."""
    wps = np.stack([np.asarray(start_kps, dtype=np.float64)] + list(kps_list))
    tols = np.concatenate([[tols[0]], tols])
    holds = np.concatenate([[1], holds])
    skips = np.concatenate([[False], skips])
    skips[-1] = False
    return WaypointTracker(wps, tols, holds, skips, params=params)


def anchored_from_plan(plan, start_px, params: Optional[TrackerParams] = None
                       ) -> WaypointTracker:
    """``anchored`` for a single-point ``VeritasPlan``: every waypoint is a (1, 2)
    keypoint set and the anchor is the current end-effector pixel ``start_px`` (2,)."""
    wps = plan.waypoints
    assert wps, 'plan has no waypoints'
    return anchored(
        [np.asarray(wp.uv, dtype=np.float64)[None, :] for wp in wps],
        [wp.tol_px for wp in wps], [wp.min_hold for wp in wps],
        [wp.skippable for wp in wps],
        np.asarray(start_px, dtype=np.float64)[None, :], params=params)
