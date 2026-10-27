"""Upstream Grounded-SAM-2 object-hint port (VERITAS_HINTS=gsam2), with a fake tracker
and a fake Gemini client: no GPU, no network."""
import json
import types

import numpy as np

from diffusion_policy.env.simpler.veritas.gsam import upstream_hints as uh
from diffusion_policy.env.simpler.veritas.gsam.tracker_adapter import TrackerAdapter
from diffusion_policy.env.simpler.veritas.simpler_client import build_prompt


class _FakeTracker:
    """ObjectTracker stand-in: bboxes per prompt name, or nothing for unseen ones."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.init_calls = []

    def initialize(self, frame, names, gdino_prompts=None):
        self.init_calls.append((list(names), gdino_prompts))
        return True

    def track(self, frame):
        return {n: {'bbox': b} for n, b in self.boxes.items()}


class _FakeClient:
    def __init__(self, objects, bboxes=None):
        self.objects, self.bboxes, self.last_usage = objects, bboxes or {}, {'n': 1}
        self.bbox_requests = []

    def _image_to_part(self, frame):
        return 'IMG'

    def _save_vlm_log(self, *a):
        pass

    def _call_with_retry(self, parts, cfg, schema, **kw):
        assert parts[0].startswith('Select the minimal set of objects')
        return schema.model_validate({'objects': self.objects})

    def _generate(self, parts, cfg):
        names = parts[0].split('these objects in the image: ')[1].split('.\n')[0].split(', ')
        self.bbox_requests.append(names)
        h, w = 480, 640
        objs = [{'name': n, 'box_2d': [b[1] / h * 1000, b[0] / w * 1000, b[3] / h * 1000,
                                       b[2] / w * 1000]}
                for n, b in self.bboxes.items() if n in names]
        return types.SimpleNamespace(text=json.dumps({'objects': objs}))


def _provider(client, tracker):
    p = uh.UpstreamHintProvider.__new__(uh.UpstreamHintProvider)
    p.client, p.log_dir, p.tracker, p.adapter = client, None, tracker, TrackerAdapter(tracker)
    return p


FRAME = np.zeros((480, 640, 3), np.uint8)


def test_hints_are_mask_bbox_centres_with_gdino_prompts():
    client = _FakeClient([{'name': 'eggplant', 'role': 'source', 'gdino_prompt': 'purple object'},
                          {'name': 'yellow_basket', 'role': 'target', 'gdino_prompt': 'yellow basket'}])
    tracker = _FakeTracker({'eggplant': [100, 200, 140, 260], 'yellow basket': [300, 100, 420, 180]})
    hints, meta = _provider(client, tracker)(None, FRAME, 'put eggplant into yellow basket', [1.5, 2.25])
    assert tracker.init_calls == [(['eggplant', 'yellow basket'], ['purple object', 'yellow basket'])]
    assert [(h['name'], h['role'], h['uv']) for h in hints] == [
        ('eggplant', 'source', [120.0, 230.0]), ('yellow_basket', 'target', [360.0, 140.0])]
    assert meta['n_from_gsam2'] == 2 and meta['n_from_gemini_bbox'] == 0 and not client.bbox_requests


def test_missed_object_and_shared_role_go_to_gemini_bbox():
    # upstream quirk: both 'target' detections are named after the FIRST target object, so
    # the second target is "missing" and asked for via the Gemini bbox call
    client = _FakeClient([{'name': 'green cube', 'role': 'source'},
                          {'name': 'yellow cube', 'role': 'target'},
                          {'name': 'table', 'role': 'target'}],
                         bboxes={'table': [0, 300, 640, 480]})
    tracker = _FakeTracker({'green cube': [10, 10, 30, 30], 'yellow cube': [50, 50, 70, 70],
                            'table': [0, 300, 640, 480]})
    hints, meta = _provider(client, tracker)(None, FRAME, 'stack', None)
    assert [h['name'] for h in hints] == ['green cube', 'yellow cube', 'yellow cube', 'table']
    assert client.bbox_requests == [['table']]
    assert hints[-1]['source'] == 'gemini_bbox' and meta['n_from_gemini_bbox'] == 1


def test_exact_prompt_formatting_is_unrounded():
    hints = [{'name': 'eggplant', 'role': 'source', 'uv': [120.25, 230.0]}]
    text = build_prompt('BODY {instruction}', FRAME, 'go', [312.123456, 120.7], hints,
                        raw_numbers=True)
    assert text.splitlines()[0] == ('Image size: 640x480 (u in [0,640), v in [0,480]). '
                                    'Current end-effector pixel: [312.123456, 120.7]. '
                                    'Objects (name/role/uv): eggplant(source) uv=[120.25, 230.0]')
