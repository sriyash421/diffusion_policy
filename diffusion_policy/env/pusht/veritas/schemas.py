"""Pydantic schemas for the Veritas waypoint plan (ported from veritas/src/agent/veritas_verifier/schemas.py)."""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel


class VeritasWaypoint(BaseModel):
    """Absolute pixel waypoint in the 96x96 obs image."""

    uv: Tuple[float, float]
    # Veritas's 60 px is ~9.4% of a 640-wide frame; 9 px is the same fraction of 96.
    tol_px: float = 9.0
    min_hold: int = 2
    skippable: bool = True
    weight: float = 1.0


class VeritasPlan(BaseModel):
    """Waypoint plan returned by Gemini."""

    waypoints: List[VeritasWaypoint]
    confidence: float = 0.0
    notes: Optional[str] = None


class VeritasPoseWaypoint(BaseModel):
    """Intermediate T pose: keypoint centroid (pixels) + angle (deg, clockwise on screen)."""

    centroid_uv: Tuple[float, float]
    angle_deg: float
    tol_px: float = 6.0  # mean keypoint distance at which the pose counts as reached
    min_hold: int = 1
    skippable: bool = True


class VeritasPosePlan(BaseModel):
    """T-pose plan returned by Gemini."""

    waypoints: List[VeritasPoseWaypoint]
    confidence: float = 0.0
    notes: Optional[str] = None


class VeritasDualPlan(BaseModel):
    """Two independent lists from one call: a pusher path and a T-pose sequence."""

    pusher_waypoints: List[VeritasWaypoint]
    t_poses: List[VeritasPoseWaypoint]
    confidence: float = 0.0
    notes: Optional[str] = None
