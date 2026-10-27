"""Upstream Veritas object hints for the planning prompt: Gemini picks the objects, the
Grounded-SAM-2 tracking server localizes them, Gemini bboxes fill the gaps.

A line-for-line port of veritas ``src/agent/eval.py`` ``_initialize_episode`` (the
waypoint-px branch), with the prompts, schemas and fallbacks of
``src/agent/veritas_verifier/gemini_client.py`` (``call_veritas_objects``,
``call_detect_object_bboxes``) and ``src/agent/utils/verifier_helpers.get_object_names``.
The tracker itself (``object_tracker`` / ``tracker_adapter`` / ``tracking_server*``) is
vendored verbatim next to this file. Kept on purpose, as upstream does it:
  * a hint's uv is the centre of the SAM2 mask's tight bbox, not the mask centroid;
  * a detection is named by the FIRST object with its role (two objects sharing a role
    both get the first name, and the second then goes to the Gemini bbox fallback);
  * one ``TrackerAdapter`` lives for the whole process, so ``last_bboxes`` from an
    earlier episode can stand in for an object the detector misses.
Only the retry policy differs: Gemini calls use this repo's client (longer 503 backoff).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from diffusion_policy.env.simpler.veritas.gsam.schema_clean import clean_schema_for_gemini

log = logging.getLogger(__name__)

SAM2_CONFIG = 'sam2.1/sam2.1_hiera_l'
BOX_THRESHOLD, TEXT_THRESHOLD = 0.4, 0.3


class VeritasObject(BaseModel):
    """Object (name/role) to track for Veritas planning."""

    name: str
    role: Optional[str] = None
    gdino_prompt: Optional[str] = None


class VeritasObjectPlan(BaseModel):
    """Object selection returned by Gemini prior to waypoint planning."""

    objects: List[VeritasObject]
    confidence: float = 0.0
    notes: Optional[str] = None


def get_object_names(env) -> List[Tuple[str, str]]:
    """(name, role) of the sim's target / source / obj (upstream ``get_object_names``)."""
    names: List[Tuple[str, str]] = []

    def _process_name(raw_name: str) -> str:
        if not raw_name:
            return ""
        parts = raw_name.split("_")
        prefixes_to_remove = ["opened", "light", "generated", "modified", "objaverse", "bridge", "baked", "v2"]
        parts = [p for p in parts if p not in prefixes_to_remove]
        parts = [p for p in parts if not p.replace(".", "").isdigit() and "cm" not in p.lower()]
        name = " ".join(parts).strip()
        return name if name else raw_name

    for attr, default, role in (('episode_target_obj', 'target', 'target'),
                                ('episode_source_obj', 'source', 'source'),
                                ('obj', 'obj', 'obj')):
        try:
            if hasattr(env, attr) and getattr(env, attr) is not None:
                processed_name = _process_name(getattr(getattr(env, attr), "name", default))
                if processed_name:
                    names.append((processed_name, role))
        except Exception:
            pass
    try:
        drawer_id = getattr(env, "drawer_id", None)
        if drawer_id:
            names.append((drawer_id, "drawer"))
    except Exception:
        pass
    return names


def call_veritas_objects(client, instruction: str, frame: np.ndarray,
                         log_dir: Optional[str] = None,
                         eef_pixel: Optional[List[float]] = None) -> Optional[VeritasObjectPlan]:
    """First VLM call: choose which objects to track (upstream prompt verbatim)."""
    h, w = frame.shape[:2]
    prompt = [
        (
            "Select the minimal set of objects the robot must interact with to finish the task. "
            "Return JSON only matching the schema. Choose concrete names and brief roles. "
            "For each object, also fill `gdino_prompt` with the object's visible properties "
            "— for example color, — as 2-4 words you can see in "
            "this image. "
            "Do not return waypoints here—only objects."
        ),
        f"Task: {instruction if instruction else 'No instruction provided'}",
        f"Image size: {w}x{h} (u in [0,{w}), v in [0,{h}]).",
    ]
    if eef_pixel and len(eef_pixel) == 2:
        prompt.append(f"Current end-effector pixel (u,v): {eef_pixel}.")
    parts = [("\n".join(prompt)), client._image_to_part(frame)]
    cfg = {'response_mime_type': 'application/json',
           'response_schema': clean_schema_for_gemini(VeritasObjectPlan.model_json_schema())}

    def _on_success(parsed, response_text):
        client._save_vlm_log(response_text, log_dir, 'veritas_objects')

    return client._call_with_retry(
        parts, cfg, VeritasObjectPlan, label='VeritasObjects',
        max_retries=int(os.environ.get('GEMINI_MAX_RETRIES', '8')),
        initial_retry_delay=5.0, on_success=_on_success)


def call_detect_object_bboxes(client, frame: np.ndarray, object_names: List[str],
                              log_dir: Optional[str] = None) -> Optional[Dict[str, List[float]]]:
    """Gemini bbox fallback -> {name: [x_min, y_min, x_max, y_max]} px (upstream verbatim)."""
    h, w = frame.shape[:2]
    object_list = ", ".join(object_names)
    prompt = f"""Detect the bounding boxes for these objects in the image: {object_list}.
Return JSON with format:
{{
  "objects": [
    {{"name": "object_name", "box_2d": [ymin, xmin, ymax, xmax]}}
  ]
}}
The box_2d should be normalized to 0-1000 (ymin, xmin, ymax, xmax).
Image size: {w}x{h} pixels."""
    parts = [prompt, client._image_to_part(frame)]
    schema_dict = {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "box_2d": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["name", "box_2d"],
                },
            },
        },
        "required": ["objects"],
    }
    cfg = {"response_mime_type": "application/json", "response_schema": schema_dict}
    max_retries = int(os.environ.get('GEMINI_MAX_RETRIES', '8'))
    retry_delay = 5.0
    for attempt in range(max_retries):
        try:
            response = client._generate(parts, cfg)
            response_text = response.text
            client._save_vlm_log(response_text, log_dir, "object_bboxes")
            parsed = json.loads(response_text)
            result = {}
            for obj in parsed.get("objects", []):
                name = obj.get("name", "").strip()
                box_normalized = obj.get("box_2d", [])
                if len(box_normalized) == 4 and name:
                    ymin_norm, xmin_norm, ymax_norm, xmax_norm = box_normalized
                    result[name] = [int(xmin_norm / 1000 * w), int(ymin_norm / 1000 * h),
                                    int(xmax_norm / 1000 * w), int(ymax_norm / 1000 * h)]
            return result if result else None
        except Exception as e:
            log.warning(f"[ObjectBboxes] attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt + 1 == max_retries:
                break
            time.sleep(retry_delay)
            retry_delay *= 1.5
    return None


class UpstreamHintProvider:
    """Per-process Grounded-SAM-2 tracker + the per-episode hint procedure."""

    def __init__(self, client, log_dir: Optional[str] = None):
        from diffusion_policy.env.simpler.veritas.gsam.object_tracker import ObjectTracker
        from diffusion_policy.env.simpler.veritas.gsam.tracker_adapter import TrackerAdapter
        import torch
        gsam = os.environ['GROUNDED_SAM2_PATH']
        self.client = client
        self.log_dir = log_dir
        self.tracker = ObjectTracker(
            sam2_checkpoint=os.environ.get(
                'SAM2_CHECKPOINT', os.path.join(gsam, 'checkpoints', 'sam2.1_hiera_large.pt')),
            model_cfg=SAM2_CONFIG,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            prompt_type='box', box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            grounded_sam2_python=os.environ.get('GROUNDED_SAM2_PYTHON'))
        self.adapter = TrackerAdapter(self.tracker)

    def __call__(self, env_base, frame: np.ndarray, instruction: str,
                 eef_pixel: Optional[List[float]]) -> Tuple[Optional[List[dict]], Dict[str, Any]]:
        """(object_hints, provenance) for the first frame of an episode."""
        meta: Dict[str, Any] = {'hint_source': 'gsam2_upstream'}
        object_plan = None
        object_names_with_roles: List[Tuple[str, str]] = []
        object_names: List[str] = []
        try:
            object_plan = call_veritas_objects(self.client, instruction, frame,
                                               log_dir=self.log_dir, eef_pixel=eef_pixel)
            if object_plan and getattr(object_plan, "objects", None):
                object_names_with_roles = [(o.name, o.role or "other") for o in object_plan.objects]
                object_names = [n for n, _ in object_names_with_roles]
        except Exception as e:
            log.warning(f"Waypoint-px object selection failed: {e}")
        meta['objects_plan'] = object_plan.model_dump() if object_plan is not None else None
        meta['objects_usage'] = getattr(self.client, 'last_usage', None)
        if not object_names_with_roles:
            try:
                object_names_with_roles = get_object_names(env_base)
                object_names = [n for n, _ in object_names_with_roles]
                meta['objects_from_sim'] = True
            except Exception:
                object_names_with_roles, object_names = [], []

        object_hints = None
        n_tracked = 0
        if object_names_with_roles:
            try:
                gdino_prompts = ([getattr(o, "gdino_prompt", None) for o in object_plan.objects]
                                 if object_plan and getattr(object_plan, "objects", None) else None)
                if self.adapter.initialize(frame, object_names_with_roles,
                                           gdino_prompts=gdino_prompts):
                    detections = self.adapter.track_frame(frame)
                    tracked_object_names = set()
                    object_hints = []
                    for det in detections or []:
                        if det.center is not None and len(det.center) == 2:
                            uv = det.center.tolist() if isinstance(det.center, np.ndarray) else det.center
                            bbox = (det.bbox.tolist() if det.bbox is not None
                                    and isinstance(det.bbox, np.ndarray) else det.bbox)
                            role = getattr(det, "role", "other")
                            name = None
                            if self.adapter.object_names:
                                for n, r in object_names_with_roles:
                                    if r == role:
                                        name = n
                                        break
                                if name is None and len(self.adapter.object_names) > len(object_hints):
                                    name = self.adapter.object_names[len(object_hints)]
                            if name:
                                tracked_object_names.add(name)
                            object_hints.append({"name": name or "unknown", "role": role,
                                                 "uv": uv, "bbox": bbox, "source": "gsam2"})
                    n_tracked = len(object_hints)
                    missing = [n for n, _ in object_names_with_roles if n not in tracked_object_names]
                    if missing:
                        vlm_bboxes = call_detect_object_bboxes(self.client, frame, missing,
                                                               log_dir=self.log_dir)
                        if vlm_bboxes:
                            for obj_name, obj_role in object_names_with_roles:
                                if obj_name not in tracked_object_names and obj_name in vlm_bboxes:
                                    b = vlm_bboxes[obj_name]
                                    if len(b) == 4:
                                        object_hints.append({
                                            "name": obj_name, "role": obj_role,
                                            "uv": [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0],
                                            "bbox": b, "source": "gemini_bbox"})
            except Exception as e:
                log.warning(f"Failed to get object hints from TrackerAdapter: {e}", exc_info=True)
                object_hints = None

        if not object_hints and object_names:
            vlm_bboxes = call_detect_object_bboxes(self.client, frame, object_names,
                                                   log_dir=self.log_dir)
            if vlm_bboxes:
                object_hints = []
                for obj_name, obj_role in object_names_with_roles:
                    b = vlm_bboxes.get(obj_name)
                    if b is not None and len(b) == 4:
                        object_hints.append({
                            "name": obj_name, "role": obj_role,
                            "uv": [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0],
                            "bbox": b, "source": "gemini_bbox"})
        meta.update(n_from_gsam2=n_tracked,
                    n_from_gemini_bbox=len(object_hints or []) - n_tracked)
        return object_hints, meta
