"""Pydantic schemas for the Veritas waypoint plan.

Ported from veritas/src/agent/veritas_verifier/schemas.py with the upstream defaults,
which are tuned for a 640-wide frame. Envs with a different frame size subclass
``VeritasWaypoint`` and override the pixel defaults (see ``env/pusht/veritas/schemas``).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel


class VeritasWaypoint(BaseModel):
    """Absolute pixel waypoint in the planning frame."""

    uv: Tuple[float, float]
    tol_px: float = 60.0
    min_hold: int = 2
    skippable: bool = True
    weight: float = 1.0


class VeritasPlan(BaseModel):
    """Waypoint plan returned by Gemini."""

    waypoints: List[VeritasWaypoint]
    confidence: float = 0.0
    notes: Optional[str] = None
