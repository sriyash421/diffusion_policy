"""Gemini waypoint planning for SimplerEnv frames: upstream's prompt and schema
(veritas/src/agent/veritas_verifier/gemini_client.py ``call_veritas_plan``) verbatim,
with the plan clipped to the image and NO centre-of-image fallback (a failed call
returns None and the record carries no ``plan``; see ``veritas.plan_store``).
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

import numpy as np

from diffusion_policy.veritas.gemini_client import GeminiClient
from diffusion_policy.veritas.schemas import VeritasPlan

log = logging.getLogger(__name__)

PROMPT_V1_UPSTREAM = """Generate 5-10 absolute pixel waypoints (u,v) for the robot end-effector to complete the task.
Return JSON only matching schema:
waypoints: [{"uv": [u,v], "tol_px": float, "min_hold": int, "skippable": bool, "weight": float}],
confidence: float, notes: optional.
Instruction: {instruction}.
Camera is fixed; pixel coords are absolute.
Pick tol_px (pixel-radius the verifier uses to consider a waypoint reached):
use ~60 pixels per waypoint by default. Adjust within ~50-80 only when you
have a clear reason — slightly tighter (~50) for precise contact/placement,
slightly looser (~70-80) for coarse approach transits. Stay close to 60.
Pick min_hold (consecutive frames the EE must stay within tol_px before the
waypoint counts as reached) as a small integer: 1 for transit waypoints,
2-3 for contact/placement waypoints that need a brief settle.
Set skippable=true for every intermediate waypoint so the verifier can skip
ahead when one is approximately satisfied. Set skippable=false only for the
final contact/placement waypoint — that one must latch for success.
"""

# v1_upstream_exact: same text, but the context line prints the eef pixel and object uvs
# unrounded (upstream's f-string of the raw lists) and the schema requires only ``uv``
# (upstream HEAD). Used with the Grounded-SAM-2 object hints (VERITAS_HINTS=gsam2).
PROMPTS = {'v1_upstream': PROMPT_V1_UPSTREAM, 'v1_upstream_exact': PROMPT_V1_UPSTREAM}

# google-genai's schema accepts only types and `required`; tol_px is required because
# the verifier reads it directly (upstream comment).
PLAN_SCHEMA = {
    'type': 'object',
    'properties': {
        'waypoints': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'uv': {'type': 'array', 'items': {'type': 'number'}},
                    'tol_px': {'type': 'number'},
                    'min_hold': {'type': 'integer'},
                    'skippable': {'type': 'boolean'},
                    'weight': {'type': 'number'},
                },
                'required': ['uv', 'tol_px'],
            },
        },
        'confidence': {'type': 'number'},
        'notes': {'type': 'string'},
    },
    'required': ['waypoints'],
}


PLAN_SCHEMA_EXACT = json.loads(json.dumps(PLAN_SCHEMA))
PLAN_SCHEMA_EXACT['properties']['waypoints']['items']['required'] = ['uv']


def build_prompt(template: str, frame: np.ndarray, instruction: str,
                 eef_pixel: Optional[List[float]] = None,
                 object_hints: Optional[List[dict]] = None, raw_numbers: bool = False) -> str:
    h, w = frame.shape[:2]
    extra = (f'Image size: {w}x{h} (u in [0,{w}), v in [0,{h}]).' if raw_numbers  # upstream
             else f'Image size: {w}x{h} (u in [0,{w}), v in [0,{h})).')
    if eef_pixel is not None and len(eef_pixel) == 2:
        px = list(eef_pixel) if raw_numbers else [round(float(p), 1) for p in eef_pixel]
        extra += f' Current end-effector pixel: {px}.'
    if object_hints:
        extra += ' Objects (name/role/uv): ' + '; '.join(
            f"{o.get('name', 'unknown')}({o.get('role', 'other')}) uv={o.get('uv')}"
            for o in object_hints)
    body = template.replace('{instruction}', instruction or 'No instruction provided')
    return extra + '\n' + body


def clip_plan(plan: VeritasPlan, image_hw) -> VeritasPlan:
    h, w = image_hw
    for wp in plan.waypoints:
        u, v = wp.uv
        wp.uv = (float(np.clip(u, 0, w - 1)), float(np.clip(v, 0, h - 1)))
    return plan


class SimplerVeritasClient(GeminiClient):
    """One structured Gemini call -> a clipped ``VeritasPlan`` or None."""

    def call_plan(self, frame: np.ndarray, instruction: str,
                  eef_pixel: Optional[List[float]] = None,
                  object_hints: Optional[List[dict]] = None,
                  prompt: str = 'v1_upstream', log_dir: Optional[str] = None,
                  log_name: str = 'veritas_plan') -> Optional[VeritasPlan]:
        exact = prompt == 'v1_upstream_exact'
        text = build_prompt(PROMPTS[prompt], frame, instruction, eef_pixel, object_hints,
                            raw_numbers=exact)
        self.last_prompt = text
        parts = [text, self._image_to_part(frame)]
        cfg = {'response_mime_type': 'application/json',
               'response_schema': PLAN_SCHEMA_EXACT if exact else PLAN_SCHEMA}

        def _on_success(parsed, response_text):
            self._save_vlm_log(response_text, log_dir, log_name)

        # 503 "high demand" spikes outlast the base client's 3 attempts; back off
        # exponentially for longer (GEMINI_MAX_RETRIES, default 8: ~10 s -> ~20 min total).
        plan = self._call_with_retry(parts, cfg, VeritasPlan, label='VeritasPlan',
                                     max_retries=int(os.environ.get('GEMINI_MAX_RETRIES', '8')),
                                     initial_retry_delay=5.0, on_success=_on_success)
        if plan is None or not plan.waypoints:
            return None
        return clip_plan(plan, frame.shape[:2])


def make_live_planner(client: SimplerVeritasClient, prompt: str = 'v1_upstream',
                      log_dir: Optional[str] = None, **provenance):
    """A ``CachedPlanProvider`` planner: one Gemini call at env reset, returning the
    record dict (raw usage + clipped plan + provenance) the store persists by seed."""
    def _plan(seed, frame, instruction, eef_pixel=None, object_hints=None, hint_fn=None):
        hint_meta = None
        if hint_fn is not None:              # upstream Grounded-SAM-2 hints, only on a miss
            object_hints, hint_meta = hint_fn()
        plan = client.call_plan(frame, instruction, eef_pixel=eef_pixel,
                                object_hints=object_hints, prompt=prompt,
                                log_dir=log_dir, log_name=f'seed{int(seed)}_raw')
        rec = dict(model=client.model_name, image_size=list(frame.shape[:2][::-1]),
                   prompt=client.last_prompt, hints=object_hints, eef_pixel=eef_pixel,
                   usage=client.last_usage, seed=int(seed), instruction=instruction,
                   live=True, **provenance)
        if hint_meta is not None:
            rec['hint_meta'] = hint_meta
        if plan is not None:
            rec['plan'] = plan.model_dump()
        return rec
    return _plan
