"""Veritas waypoint planning for PushT, on the policy's 96x96 image obs.

Port of ``VeritasGeminiClient.call_veritas_plan`` (veritas/src/agent/veritas_verifier/
gemini_client.py). What differs from Veritas and why:
  * The prompt names the pusher, block and goal; PushT has no language instruction.
  * tol_px guidance is rescaled to 96 px (Veritas's ~60 px is tuned for 640x480).
  * Object hints come from state, not Grounded-SAM-2, so the objects/bbox calls are dropped.
  * Waypoints are clipped to the image; Veritas does no bounds check.
  * Failure returns None. Veritas falls back to one waypoint at the image centre, which
    in PushT is the goal T, so a failed call would silently steer toward the goal.

Pixel coords at render size S are workspace coords times S/512, with no flip
(PushTEnv._render_frame; positive_y_is_up=False).
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional

import numpy as np

from diffusion_policy.env.pusht.feedback_util import (
    GOAL_POSE, T_VERTS, block_pose_from_feedback, keypoints_at_pose,
    keypoints_from_feedback, t_center_from_feedback, t_goal_distance)
from diffusion_policy.env.pusht.veritas.gemini_client import GeminiClient
from diffusion_policy.env.pusht.veritas.schemas import (
    VeritasDualPlan, VeritasPlan, VeritasPosePlan)

log = logging.getLogger(__name__)

WORKSPACE_SIZE = 512.0

PUSHT_INSTRUCTION = (
    "Push the grey T-shaped block until it exactly covers the light-green T-shaped "
    "target, matching both position and orientation. Only the blue circle moves; it "
    "pushes the block by contact and cannot grasp or pull.")

# Veritas's hand-written schema, unchanged: Gemini's schema subset rejects
# minItems/maximum/minimum, so only types and `required` are kept.
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "waypoints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "uv": {"type": "array", "items": {"type": "number"}},
                    "tol_px": {"type": "number"},
                    "min_hold": {"type": "integer"},
                    "skippable": {"type": "boolean"},
                    "weight": {"type": "number"},
                },
                "required": ["uv"],
            },
        },
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": ["waypoints"],
}


# Prompt templates, filled by build_prompt. Each results folder stores the template it used.
# v1_veritas_port: Veritas's plan prompt with the PushT substitutions (module docstring).
PROMPT_V1 = """Image size: {W}x{H} (u in [0,{W}), v in [0,{H}]). Current pusher pixel: {agent_uv}. Objects (name/role/uv): T block(object) uv={t_uv}; goal T(target) uv={g_uv}
Generate 5-10 absolute pixel waypoints (u,v) for the blue circular pusher to complete the task.
Return JSON only matching schema:
waypoints: [{{"uv": [u,v], "tol_px": float, "min_hold": int, "skippable": bool, "weight": float}}],
confidence: float, notes: optional.
Instruction: {instruction}
Top-down view of the whole table. Pixel coords (u right, v down) are absolute; fractional values are allowed.
Pick tol_px (pixel-radius the verifier uses to consider a waypoint reached):
use ~9 pixels per waypoint by default. Adjust within ~7-12 only when you
have a clear reason — slightly tighter (~7) for precise contact with the block,
slightly looser (~11-12) for coarse approach transits. Stay close to 9.
Pick min_hold (consecutive frames the pusher must stay within tol_px before the
waypoint counts as reached) as a small integer: 1 for transit waypoints,
2-3 for waypoints where the pusher contacts or pushes the block.
Set skippable=true for every intermediate waypoint so the verifier can skip
ahead when one is approximately satisfied. Set skippable=false only for the
final push waypoint — that one must latch for success.
"""

# v2_push_rules: PushT-specific pushing rules plus T/goal angles.
PROMPT_V2 = """You are planning a path for the blue circular pusher in the Push-T task.
Image: {W}x{H} pixels, fixed top-down camera, origin top-left, u right, v down.
State (pixels): pusher at {agent_uv}. Grey T: center {t_uv}, angle {t_deg} deg.
Green goal T: center {g_uv}, angle {g_deg} deg.
The pusher can only push (no grasping). The task succeeds when the grey T covers
the green goal T in both position and rotation.

Return 5-10 waypoints (u,v) for the PUSHER, in order:
- To move the T, first go to the side of the T opposite the direction it must
  move, then push through it toward the goal.
- To rotate the T, push near the end of one of its bars, not through its center.
- When repositioning between pushes, route AROUND the T, not through it.
- The last waypoint is where the pusher is when the T reaches the goal.

tol_px: about {tol_default} px (~6% of image width). Use ~{tol_tight} px for waypoints
where the pusher must contact a specific part of the T, ~{tol_loose} px for repositioning.
min_hold: 1 for repositioning, 2 for contact/push waypoints.
skippable: true for all but the final waypoint.
Return JSON only:
{{"waypoints":[{{"uv":[u,v],"tol_px":float,"min_hold":int,"skippable":bool,"weight":float}}],
  "confidence":float,"notes":"optional"}}
"""

# v3_end_outside: v2 with the final waypoint kept outside the goal T.
PROMPT_V3 = PROMPT_V2.replace(
    "- The last waypoint is where the pusher is when the T reaches the goal.\n",
    "- The last waypoint is where the pusher is when the T reaches the goal. not insde the goal\n")
assert PROMPT_V3 != PROMPT_V2

# v4_t_keypoints: plan T POSES, tracked by the repo's own T keypoints (feedback_util.T_VERTS,
# the 8 corners behind t_goal_distance). Adapted from a keypoint-verifier example: no
# contact sites, and the centroid is the corner mean (block-local (0, 45)).
PROMPT_V4 = """You are planning how the grey T block should move in the Push-T task.
A blue circular pusher pushes the T across a flat table (it can only push, never grasp).
The task succeeds when the grey T covers the green goal T in BOTH position and rotation.

Image: {W}x{H} pixels, fixed top-down camera, origin top-left, u to the right, v down.

T geometry: a horizontal bar {bar_len_px}x{bar_w_px} px with a stem {stem_w_px}x{stem_len_px} px
hanging from its middle. The T is tracked by its 8 corner keypoints, in this order:
bar_left_bottom, bar_right_bottom, bar_right_top, bar_left_top,
stem_top_left, stem_bottom_left, stem_bottom_right, stem_top_right.
Pose = (centroid u, centroid v, angle); the centroid is the mean of the 8 keypoints.
Angle 0 means the bar is on top and the stem points straight down (+v); positive
angles rotate CLOCKWISE on screen.

Current state (pixels, degrees):
- Grey T pose: centroid {t_uv}, angle {t_deg}
- Grey T keypoints: {t_kps}
- Green goal T pose: centroid {g_uv}, angle {g_deg}
- Green goal T keypoints: {g_kps}
- Grey-to-goal distance (mean keypoint distance): {d_goal} px
- Pusher at {agent_uv}

Plan 3-8 intermediate poses of the grey T, ending exactly at the goal pose.
Rules:
- Each step moves the centroid at most ~{max_step_px} px and turns at most ~30 degrees.
- Take the shortest rotation direction to the goal angle (difference within +/-180 degrees).
- Keep the entire T inside the image with a margin of ~{margin_px} px.
- Rotate while the T has room around it; small rotations near the goal are hard to fix.
- Every step must be producible by pushing: the pusher pushes from the side opposite
  the direction of motion, and rotates the T by pushing near a bar end or the stem side.

Tolerances are the MEAN KEYPOINT DISTANCE (pixels) at which a pose counts as reached:
- tol_px ~{tol_mid_px} for intermediate poses, ~{tol_final_px} for the final (goal) pose.
- min_hold: 1 for intermediate poses, 2 for the final pose.
- skippable: true for every intermediate pose, false for the final pose.

Return JSON only:
{{"waypoints": [{{"centroid_uv": [u, v], "angle_deg": float,
                 "tol_px": float, "min_hold": int, "skippable": bool}}],
  "confidence": float, "notes": "optional"}}
"""

# v5_pusher_and_t: v4's T-pose plan and v3's pusher path (rules verbatim), as two
# independent lists in one response.
PROMPT_V5 = """You are planning the Push-T task. A blue circular pusher pushes the grey T block
across a flat table (it can only push, never grasp). The task succeeds when the grey T
covers the green goal T in BOTH position and rotation.

Image: {W}x{H} pixels, fixed top-down camera, origin top-left, u to the right, v down.

T geometry: a horizontal bar {bar_len_px}x{bar_w_px} px with a stem {stem_w_px}x{stem_len_px} px
hanging from its middle. The T is tracked by its 8 corner keypoints, in this order:
bar_left_bottom, bar_right_bottom, bar_right_top, bar_left_top,
stem_top_left, stem_bottom_left, stem_bottom_right, stem_top_right.
Pose = (centroid u, centroid v, angle); the centroid is the mean of the 8 keypoints.
Angle 0 means the bar is on top and the stem points straight down (+v); positive
angles rotate CLOCKWISE on screen.

Current state (pixels, degrees):
- Grey T pose: centroid {t_uv}, angle {t_deg}
- Grey T keypoints: {t_kps}
- Green goal T pose: centroid {g_uv}, angle {g_deg}
- Green goal T keypoints: {g_kps}
- Grey-to-goal distance (mean keypoint distance): {d_goal} px
- Pusher at {agent_uv}

Return TWO plans in one response.

1. t_poses: 3-8 intermediate poses of the grey T, ending exactly at the goal pose.
- Each step moves the centroid at most ~{max_step_px} px and turns at most ~30 degrees.
- Take the shortest rotation direction to the goal angle (difference within +/-180 degrees).
- Keep the entire T inside the image with a margin of ~{margin_px} px.
- Rotate while the T has room around it; small rotations near the goal are hard to fix.
- Every step must be producible by pushing: the pusher pushes from the side opposite
  the direction of motion, and rotates the T by pushing near a bar end or the stem side.
- tol_px is the MEAN KEYPOINT DISTANCE (pixels) at which a pose counts as reached:
  ~{tol_mid_px} for intermediate poses, ~{tol_final_px} for the final (goal) pose.
- min_hold: 1 for intermediate poses, 2 for the final pose.
- skippable: true for every intermediate pose, false for the final pose.

2. pusher_waypoints: 5-10 waypoints (u,v) for the PUSHER, in order, that move the T
   through those poses:
- To move the T, first go to the side of the T opposite the direction it must
  move, then push through it toward the goal.
- To rotate the T, push near the end of one of its bars, not through its center.
- When repositioning between pushes, route AROUND the T, not through it.
- The last waypoint is where the pusher is when the T reaches the goal. not insde the goal
- tol_px: about {tol_default} px (~6% of image width). Use ~{tol_tight} px for waypoints
  where the pusher must contact a specific part of the T, ~{tol_loose} px for repositioning.
- min_hold: 1 for repositioning, 2 for contact/push waypoints.
- skippable: true for all but the final waypoint.

Return JSON only:
{{"t_poses": [{{"centroid_uv": [u, v], "angle_deg": float,
               "tol_px": float, "min_hold": int, "skippable": bool}}],
  "pusher_waypoints": [{{"uv": [u, v], "tol_px": float, "min_hold": int,
                        "skippable": bool, "weight": float}}],
  "confidence": float, "notes": "optional"}}
"""

PROMPTS = {'v1_veritas_port': PROMPT_V1, 'v2_push_rules': PROMPT_V2,
           'v3_end_outside': PROMPT_V3, 'v4_t_keypoints': PROMPT_V4,
           'v5_pusher_and_t': PROMPT_V5}
# What each prompt returns: pusher points (default), T poses, or both.
PLAN_KINDS = {'v4_t_keypoints': 'pose', 'v5_pusher_and_t': 'dual'}


def plan_kind(prompt_name: str) -> str:
    return PLAN_KINDS.get(prompt_name, 'pusher')

# v2 tolerances as fractions of image width: 6 / 5 / 8 px at 96, Veritas's 60/50/80 ratios.
TOL_FRAC = {'tol_default': 0.06, 'tol_tight': 0.05, 'tol_loose': 0.08}
# v4 step, margin and tolerances as fractions of image width (from the example).
POSE_FRAC = {'max_step_px': 0.2, 'margin_px': 0.05, 'tol_mid_px': 0.06, 'tol_final_px': 0.02}

POSE_SCHEMA = {
    "type": "object",
    "properties": {
        "waypoints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "centroid_uv": {"type": "array", "items": {"type": "number"}},
                    "angle_deg": {"type": "number"},
                    "tol_px": {"type": "number"},
                    "min_hold": {"type": "integer"},
                    "skippable": {"type": "boolean"},
                },
                "required": ["centroid_uv", "angle_deg"],
            },
        },
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": ["waypoints"],
}

DUAL_SCHEMA = {
    "type": "object",
    "properties": {
        "t_poses": POSE_SCHEMA["properties"]["waypoints"],
        "pusher_waypoints": PLAN_SCHEMA["properties"]["waypoints"],
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": ["t_poses", "pusher_waypoints"],
}

# kind -> (pydantic plan class, Gemini response schema)
PLAN_TYPES = {'pusher': (VeritasPlan, PLAN_SCHEMA), 'pose': (VeritasPosePlan, POSE_SCHEMA),
              'dual': (VeritasDualPlan, DUAL_SCHEMA)}

# T_VERTS order: bar (0-3), stem (4-7).
KP_NAMES = ['bar_left_bottom', 'bar_right_bottom', 'bar_right_top', 'bar_left_top',
            'stem_top_left', 'stem_bottom_left', 'stem_bottom_right', 'stem_top_right']
T_CENTROID_LOCAL = T_VERTS.mean(axis=0).astype(np.float64)   # (0, 45)


def _wrap_deg(rad) -> float:
    """Radians -> degrees in [-180, 180). Sim convention: v points down, so positive
    angles appear clockwise on screen."""
    return float((np.degrees(rad) + 180.0) % 360.0 - 180.0)


def obs_to_gemini_inputs(image, agent_pos, feedback) -> Dict:
    """One obs step -> the frame and state hints Gemini is given.

    Args:
        image: (3, S, S) float in [0, 1], the policy's image obs.
        agent_pos: (2,) workspace coords.
        feedback: (16,) goal-relative keypoint displacement.
    Returns:
        dict with ``frame`` (S, S, 3) uint8; ``pusher_uv`` / ``block_uv`` / ``goal_uv``
        in S-pixel coords; ``block_deg`` / ``goal_deg``.
    """
    image = np.asarray(image, dtype=np.float32)
    feedback = np.asarray(feedback)
    frame = (np.moveaxis(image, 0, -1) * 255).round().clip(0, 255).astype(np.uint8)
    scale = frame.shape[0] / WORKSPACE_SIZE
    return {
        'frame': frame,
        'pusher_uv': np.asarray(agent_pos, dtype=np.float64) * scale,
        'block_uv': t_center_from_feedback(feedback) * scale,
        'goal_uv': keypoints_at_pose(GOAL_POSE).mean(axis=0).astype(np.float64) * scale,
        'block_deg': _wrap_deg(block_pose_from_feedback(feedback)[2]),
        'goal_deg': _wrap_deg(GOAL_POSE[2]),
        'block_kps_uv': keypoints_from_feedback(feedback).astype(np.float64) * scale,
        'goal_kps_uv': keypoints_at_pose(GOAL_POSE).astype(np.float64) * scale,
        'goal_dist_px': float(t_goal_distance(feedback)) * scale,
    }


def pose_from_centroid(centroid_uv, angle_deg, image_size: int = 96) -> np.ndarray:
    """(3,) workspace body pose [x, y, theta] of a T whose keypoint centroid is at
    ``centroid_uv`` (pixels) with ``angle_deg``. Inverse of the centroid offset R(theta)@(0, 45)."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    offset = np.array([c * T_CENTROID_LOCAL[0] - s * T_CENTROID_LOCAL[1],
                       s * T_CENTROID_LOCAL[0] + c * T_CENTROID_LOCAL[1]])
    xy = np.asarray(centroid_uv, dtype=np.float64) * (WORKSPACE_SIZE / image_size) - offset
    return np.array([xy[0], xy[1], theta])


def pose_keypoints_px(wp, image_size: int = 96) -> np.ndarray:
    """(8, 2) pixel keypoints of one T-pose waypoint, in T_VERTS order."""
    pose = pose_from_centroid(wp.centroid_uv, wp.angle_deg, image_size)
    return keypoints_at_pose(pose).astype(np.float64) * (image_size / WORKSPACE_SIZE)


def waypoints_to_workspace(plan: VeritasPlan, image_size: int = 96) -> np.ndarray:
    """(N, 2) waypoints in PushT action units (0-512)."""
    uv = np.asarray([wp.uv for wp in plan.waypoints], dtype=np.float64)
    return uv * (WORKSPACE_SIZE / image_size)


def _fmt_uv(uv) -> str:
    return f"[{uv[0]:.1f}, {uv[1]:.1f}]"


def _fmt_kps(kps) -> str:
    return "{" + ", ".join(f"{n}: {_fmt_uv(uv)}" for n, uv in zip(KP_NAMES, kps)) + "}"


def build_prompt(template: str, g: Dict) -> str:
    """Fill a prompt template from ``obs_to_gemini_inputs`` output. Placeholders a template
    does not use are ignored by str.format."""
    h, w = g['frame'].shape[:2]
    s = w / WORKSPACE_SIZE
    return template.format(
        W=w, H=h,
        agent_uv=_fmt_uv(g['pusher_uv']),
        t_uv=_fmt_uv(g['block_uv']), t_deg=f"{g['block_deg']:.0f}",
        g_uv=_fmt_uv(g['goal_uv']), g_deg=f"{g['goal_deg']:.0f}",
        instruction=PUSHT_INSTRUCTION,
        t_kps=_fmt_kps(g['block_kps_uv']), g_kps=_fmt_kps(g['goal_kps_uv']),
        d_goal=f"{g['goal_dist_px']:.1f}",
        bar_len_px=f"{120 * s:.1f}", bar_w_px=f"{30 * s:.1f}",
        stem_w_px=f"{30 * s:.1f}", stem_len_px=f"{90 * s:.1f}",
        **{k: round(f * w) for k, f in TOL_FRAC.items()},
        **{k: round(f * w) for k, f in POSE_FRAC.items()})


def gemini_hints(g: Dict) -> Dict:
    """The state hints in ``g`` as JSON-able values, for logs."""
    return {k: (list(map(float, g[k])) if k.endswith('_uv') else float(g[k]))
            for k in ('pusher_uv', 'block_uv', 'goal_uv', 'block_deg', 'goal_deg',
                      'goal_dist_px')}


class PushTVeritasClient(GeminiClient):
    """Gemini client for PushT Veritas plans."""

    def call_pusht_plan(
        self,
        frame: np.ndarray,
        prompt: str,
        hints: Optional[Dict] = None,
        log_dir: Optional[str] = None,
        log_name: str = "veritas_plan",
        kind: str = 'pusher',
    ):
        """Ask Gemini for a waypoint plan on ``frame`` ((S, S, 3) uint8) with ``prompt``.
        ``kind`` ('pusher' | 'pose' | 'dual', see PLAN_TYPES) picks the plan type.

        Returns the plan with every uv clipped to the image, or None if every attempt
        failed. With ``log_dir``, writes ``<log_dir>/<log_name>.json`` holding the model,
        the hints, the prompt, the raw response and the token usage (incl. thinking).
        """
        h, w = frame.shape[:2]
        parts = [prompt, self._image_to_part(frame)]
        generation_config = {
            "response_mime_type": "application/json",
            "response_schema": PLAN_TYPES[kind][1],
        }

        def _on_success(parsed, response_text):
            self._save_vlm_log(json.dumps({
                'model': self.model_name,
                'image_size': [w, h],
                'hints': hints,
                'prompt': prompt,
                'response': json.loads(response_text),
                'usage': self.last_usage,
            }, indent=2), log_dir, log_name)

        plan = self._call_with_retry(
            parts, generation_config, PLAN_TYPES[kind][0],
            label="PushTVeritasPlan", on_success=_on_success)
        if plan is None:
            return None
        return clip_plan(plan, w, h)


def clip_plan(plan, w: int, h: int):
    """Clip every waypoint uv (or T-pose centroid) into [0, w-1] x [0, h-1], logging any
    that moved."""
    n_clipped = 0
    wps = (plan.t_poses + plan.pusher_waypoints if isinstance(plan, VeritasDualPlan)
           else plan.waypoints)
    for wp in wps:
        field = 'centroid_uv' if hasattr(wp, 'centroid_uv') else 'uv'
        uv = getattr(wp, field)
        u = min(max(uv[0], 0.0), w - 1.0)
        v = min(max(uv[1], 0.0), h - 1.0)
        if (u, v) != tuple(uv):
            n_clipped += 1
        setattr(wp, field, (u, v))
    if n_clipped:
        log.warning(f"[PushTVeritasPlan] clipped {n_clipped}/{len(wps)} "
                    f"waypoints into the {w}x{h} image")
    return plan
