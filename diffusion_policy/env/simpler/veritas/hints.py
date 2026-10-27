"""Object hints for the planning prompt from simulator state.

Upstream Veritas gets object pixels from Grounded-SAM-2 or a Gemini bbox call; here
the source and target objects' sim poses are projected with the fixed camera. The
names are the task's own vocabulary so the prompt reads as upstream's
``name(role) uv=[u, v]`` hints.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from diffusion_policy.env.simpler.veritas.camera import CameraModel

# SimplerEnv model ids -> the noun the instruction uses.
OBJECT_NAMES: Dict[str, str] = {
    'eggplant': 'eggplant',
    'dummy_sink_target_plane': 'basket',
    'bridge_carrot_generated_modified': 'carrot',
    'bridge_plate_objaverse_larger': 'plate',
    'green_cube_3cm': 'green cube',
    'yellow_cube_3cm': 'yellow cube',
    'bridge_spoon_generated_modified': 'spoon',
    'table_cloth_generated_shorter': 'towel',
}


def _name(obj) -> str:
    model_id = getattr(obj, 'name', None) or ''
    for key, noun in OBJECT_NAMES.items():
        if key in model_id:
            return noun
    return model_id or 'object'


def object_hints(env_base, camera: CameraModel) -> List[dict]:
    """``[{"name", "role", "uv"}]`` for the episode's source and target objects.

    ``env_base`` is the unwrapped ``PutOnInSceneEnv`` (``episode_source_obj`` /
    ``episode_target_obj`` are sapien actors; ``.pose.p`` is the world position).
    """
    hints = []
    for role, attr in (('source', 'episode_source_obj'), ('target', 'episode_target_obj')):
        obj = getattr(env_base, attr, None)
        if obj is None:
            continue
        uv = camera.project(np.asarray(obj.pose.p, dtype=np.float64))
        if not np.all(np.isfinite(uv)):
            continue
        hints.append({'name': _name(obj), 'role': role,
                      'uv': [round(float(uv[0]), 1), round(float(uv[1]), 1)]})
    return hints
