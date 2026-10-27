"""Cached Gemini assessment of observed manipulation phases (no candidate VLM calls)."""
import hashlib
import json
import os
from pathlib import Path
from typing import Literal
import numpy as np
from pydantic import BaseModel, Field
from diffusion_policy.veritas.gemini_client import GeminiClient, genai_types
from diffusion_policy.policy.veritas_phase_verifier import PHASES, VERSION

PROMPT = """You are a visual manipulation verifier, not a task-success oracle.
Task: {instruction}
Examine the timestamped OBSERVED frames in order. Robot TCP pixel coordinates and
finger joint positions are proprioception; previous gripper command is an intention,
not evidence of grasp. Image coordinates: u rightwards, v downwards; width {width},
height {height}. All output pixel coordinates must be in this image coordinate frame.
Identify the current NEXT required phase: approach, grasp, lift, transport, align,
lower, release, verify, or unknown. Identify source and target by visible appearance.
Require evidence: held means object is visibly between fingers AND follows their
motion or is clearly suspended in the grasp. Closing or proximity alone is insufficient.
Support contact means the source appears seated on the target, ready for release;
2D overlap alone is insufficient. Stable stack requires released fingers and observed
stability across frames, ideally after withdrawing the gripper. If hidden/ambiguous,
report unknown. Do not assert observed success from hypothetical movement.
If a grasp failed or object fell, return to approach/grasp; do not advance by elapsed
steps. Until possession is verified, do not direct an empty gripper to the goal.
Return next_tcp_uv: a small, physically meaningful end-effector target for the NEXT
8 control actions, not the eventual object goal. For grasp target between fingers
around the source; for lift target above current grasp; for transport/align target
above destination maintaining object clearance; for lower target seated placement;
for release keep pose; for verify withdraw upward without disturbing the stack.
Set tolerance_px around the width of the source object (10 to 60 pixels), NOT a broad
60-pixel default that skips grasp. Treat planned trajectories as hypothetical.
Confidence describes confidence in phase AND target. Give evidence with observed
frame timestamps; mention uncertainty explicitly. Return only the supplied JSON schema.
"""

Tri = Literal['true', 'false', 'unknown']
class Assessment(BaseModel):
    phase: Literal['approach','grasp','lift','transport','align','lower','release','verify','unknown']
    source: str
    target: str
    next_tcp_uv: list[float] = Field(min_length=2, max_length=2)
    tolerance_px: float = Field(ge=10, le=60)
    confidence: float = Field(ge=0, le=1)
    held: Tri
    support_contact: Tri
    stable_stack: Tri
    evidence: str
    failure: str


class PhaseClient(GeminiClient):
    def _generate(self, parts, generation_config):
        # Explicit temperature, bounded output and no candidate-dependent prompts.
        return self.client.models.generate_content(model=self.model_name, contents=parts,
            config=genai_types.GenerateContentConfig(temperature=0,
                max_output_tokens=2048, response_mime_type='application/json',
                response_schema=Assessment))


class PhaseObserver:
    def __init__(self, camera, cache_dir, trace_dir):
        self.camera = camera
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
        self.trace = Path(trace_dir); self.trace.mkdir(parents=True, exist_ok=True)
        self.client = PhaseClient()
        self.frames = []
        self.seed = None
        self.last_record = None
        self.requests = 0

    def reset(self, seed):
        self.seed = int(seed); self.frames = []; self.last_record = None

    def push(self, frame, t, tcp, fingers, last_gripper):
        if t % 4:
            return
        meta = dict(t=int(t), tcp_uv=self.camera.project(np.asarray(tcp)[:3]).tolist(),
                    finger_joints=np.asarray(fingers).tolist(), previous_open_command=float(last_gripper))
        self.frames.append((np.asarray(frame, dtype=np.uint8).copy(), meta))
        self.frames = self.frames[-3:]

    def assess(self, instruction):
        if not self.frames:
            raise RuntimeError('phase verifier missing observed frames')
        h,w = self.frames[-1][0].shape[:2]
        prompt = PROMPT.format(instruction=instruction, width=w, height=h)
        identity = dict(version=VERSION, model=self.client.model_name, prompt=prompt,
                        observations=[m for _,m in self.frames])
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
        for frame,_ in self.frames: digest.update(frame.tobytes())
        key = digest.hexdigest(); path = self.cache / (key + '.json')
        if path.exists():
            record = json.loads(path.read_text()); assessment = Assessment.model_validate(record['assessment'])
        else:
            parts = [prompt]
            for frame,meta in self.frames:
                parts.extend([json.dumps(meta), self.client._image_to_part(frame)])
            assessment = self.client._call_with_retry(parts, {}, Assessment,
                label='visual_phase', max_retries=3)
            if assessment is None:
                raise RuntimeError('phase verifier Gemini assessment failed')
            if not (0 <= assessment.next_tcp_uv[0] < w and 0 <= assessment.next_tcp_uv[1] < h):
                raise RuntimeError('phase verifier target outside image')
            record = dict(identity, assessment=assessment.model_dump(), usage=self.client.last_usage)
            tmp = path.with_suffix('.%s.tmp' % os.getpid())
            tmp.write_text(json.dumps(record,indent=2)); tmp.replace(path)
            self.requests += 1
        self.last_record = dict(record, cache_key=key, seed=self.seed, t=self.frames[-1][1]['t'])
        return np.asarray([*assessment.next_tcp_uv, assessment.tolerance_px, PHASES.index(assessment.phase),
            assessment.confidence, float(assessment.held=='true'), float(assessment.support_contact=='true'),
            float(assessment.stable_stack=='true')],dtype=np.float32)

    def record_selection(self, pred, verifier, obs_dict):
        import torch
        action = torch.as_tensor(pred['candidate_actions'], device=next(iter(obs_dict.values())).device)
        obs = {k:v.expand(action.shape[0], *v.shape[1:]) for k,v in obs_dict.items() if torch.is_tensor(v)}
        parts = {k:v.detach().cpu().tolist() for k,v in verifier.components(obs,action).items()}
        record = dict(self.last_record, selected_index=pred['selected_index'], components=parts,
                      candidates=pred['candidate_actions'].tolist(), values=pred['values'])
        with (self.trace / ('seed%d.jsonl' % self.seed)).open('a') as f:
            f.write(json.dumps(record)+'\n')
