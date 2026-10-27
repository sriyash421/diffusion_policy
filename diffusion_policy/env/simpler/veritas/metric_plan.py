"""One fresh Gemini plan after every actual reset, with a shared fail-closed dollar ledger."""
import fcntl
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from PIL import Image
from google import genai
from google.genai import types
from diffusion_policy.veritas.gemini_client import load_api_keys

VERSION = 'veritas_metric_plan_v1'
PROMPT = '''Plan this robot task from the initial image: {instruction}.
Return source/target identities, relationship (on or in), a pixel waypoint for each
phase, desired gripper state, and metric parameters in meters. Use exactly these
phases in order: approach, grasp, lift, transport, lower, release, retreat.
No further VLM calls will be made. State metrics will update source/target positions,
check object-gripper co-motion, lift, release, and stable placement. The executor
must not advance merely because an empty gripper reached a waypoint.
For uv use NORMALIZED image coordinates in [0,1000]: u right, v down.
Return uv=[1000*x/image_width,1000*y/image_height], NOT native pixels.
The actual image width is {w}, height {h}; the caller converts coordinates back.
The objects and robot are tabletop scale. Choose grasp_radius_m for TCP proximity
(typically .03-.05), lift_clearance_m (.04-.08), goal_xy_tolerance_m (.02-.04 for
stacking, .04-.07 for a basket), goal_height_offset_m (source CENTER relative to
target CENTER at final placement; around .03 for equal cubes, near 0 for basket),
and grasp_offset_z_m (TCP relative to source CENTER at grasp; near 0).
Do not claim that a commanded close proves grasp. Release requires the object at
the placement pose; terminal completion requires released, stationary object and
withdrawn gripper. Return JSON only.'''

class Stage(BaseModel):
    phase: Literal['approach','grasp','lift','transport','lower','release','retreat']
    uv: list[float] = Field(min_length=2,max_length=2)
    gripper: Literal['open','closed']
class MetricPlan(BaseModel):
    source: str
    target: str
    relationship: Literal['on','in']
    stages: list[Stage] = Field(min_length=7,max_length=7)
    grasp_radius_m: float = Field(ge=.02,le=.07)
    lift_clearance_m: float = Field(ge=.03,le=.10)
    goal_xy_tolerance_m: float = Field(ge=.015,le=.08)
    goal_height_offset_m: float = Field(ge=-.04,le=.08)
    grasp_offset_z_m: float = Field(ge=-.025,le=.04)
    notes: str


def atomic(path, data):
    tmp=path.with_suffix('.%s.tmp'%os.getpid());tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

class PlanProvider:
    version=VERSION
    schema=MetricPlan

    def make_prompt(self, instruction, w, h):
        return PROMPT.format(instruction=instruction,w=w,h=h)

    def validate_plan(self, plan):
        if [s.phase for s in plan.stages]!=['approach','grasp','lift','transport','lower','release','retreat']:
            raise ValueError('incorrect phase order')

    def plan_points(self, plan):
        return plan.stages

    def __init__(self, root, task, rollout_dir, max_requests_per_episode=3,
                 budget_group=None, group_cap_usd=None):
        self.root=Path(root);self.task=task
        self.cache=Path(rollout_dir)/'plans';self.cache.mkdir(parents=True,exist_ok=True)
        self.ledger=self.root/'gemini_cost.json'
        self.model='gemini-3.1-flash-lite'
        self.max_requests_per_episode=max_requests_per_episode
        self.budget_group=budget_group;self.group_cap_usd=group_cap_usd

    def reserve(self, episode):
        with (self.root/'gemini_cost.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            d=json.loads(self.ledger.read_text()) if self.ledger.exists() else dict(
                cap_usd=100, input_usd_per_million=.75,output_usd_per_million=3.75,
                pricing_source='https://ai.google.dev/gemini-api/docs/pricing',requests=[])
            d['pricing_by_model']={'gemini-3.8-flash': [.75,3.75], 'gemini-3.1-flash-lite': [.25,1.50]}
            spent=sum(r['charged_or_reserved_usd'] for r in d['requests'])
            if self.max_requests_per_episode is not None and sum(r['episode']==episode for r in d['requests'])>=self.max_requests_per_episode:
                raise RuntimeError('metric verifier: maximum %d Gemini requests per episode reached'%self.max_requests_per_episode)
            if self.budget_group is not None:
                groups=d.setdefault('budget_groups',{})
                old_cap=groups.setdefault(self.budget_group,float(self.group_cap_usd))
                if old_cap!=float(self.group_cap_usd):raise RuntimeError('metric verifier: group budget cap changed')
                group_spent=sum(r['charged_or_reserved_usd'] for r in d['requests'] if r.get('budget_group')==self.budget_group)
                if group_spent+.10>old_cap:raise RuntimeError('metric verifier: group dollar budget exhausted')
            # Reserve $0.10 BEFORE each request. One <=640x480 image and <=12000
            # text characters plus max4096 output tokens are below this bound at
            # pinned pricing. Uncertain failed
            # requests retain their reservation; SDK retries are disabled.
            if spent+.10>100: raise RuntimeError('metric verifier: Gemini $100 budget exhausted')
            i=len(d['requests']);d['requests'].append(dict(episode=episode,model=self.model,
                charged_or_reserved_usd=.10,status='reserved',
                input_usd_per_million=.25,output_usd_per_million=1.50))
            if self.budget_group is not None:d['requests'][-1]['budget_group']=self.budget_group
            atomic(self.ledger,d)
            return i

    def settle(self, i, usage):
        with (self.root/'gemini_cost.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);d=json.loads(self.ledger.read_text())
            if usage:
                n_in=int(usage.get('prompt_token_count') or 0)
                n_out=int(usage.get('candidates_token_count') or 0)+int(usage.get('thoughts_token_count') or 0)
                entry=d['requests'][i]
                cost=(n_in*entry.get('input_usd_per_million',.75)+n_out*entry.get('output_usd_per_million',3.75))/1e6
                if n_in>0:
                    d['requests'][i].update(charged_or_reserved_usd=cost,status='usage_recorded',usage=usage)
            atomic(self.ledger,d)

    def get(self, seed, frame, instruction):
        path=self.cache/('seed%d.json'%seed)
        with path.with_suffix('.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            keys=load_api_keys()
            if not keys: raise RuntimeError('metric verifier: no Gemini key configured')

            h,w=frame.shape[:2];prompt=self.make_prompt(instruction,w,h)
            assert len(prompt)<12000 and w*h<=640*480
            # One get() occurs after each actual env.reset; retries share this reset ID.
            reset_id=getattr(self,'rollout_id',None) or uuid.uuid4().hex
            episode=self.model+':'+str(self.cache.resolve())+':'+str(seed)+':'+reset_id
            for attempt in range(3):
                i=self.reserve(episode)
                client=genai.Client(api_key=keys[(int(seed)+attempt)%len(keys)][1],
                    http_options=types.HttpOptions(timeout=120000,
                        retry_options=types.HttpRetryOptions(attempts=1)))
                try:
                    res=client.models.generate_content(model=self.model,contents=[prompt,Image.fromarray(frame)],
                        config=types.GenerateContentConfig(temperature=0,seed=42,max_output_tokens=4096,
                            thinking_config=types.ThinkingConfig(thinking_level='low'),
                            response_mime_type='application/json',response_schema=self.schema))
                    self.settle(i,res.usage_metadata.model_dump(mode='json') if res.usage_metadata else None)
                    (self.cache / ('seed%d_request%d_raw.txt'%(seed,i))).write_text(res.text or '')
                    plan=self.schema.model_validate_json(res.text)
                    self.validate_plan(plan)
                    # Match upstream plan handling: clip geometric suggestions,
                    # rather than paying for another identical response.
                    points=self.plan_points(plan)
                    clipped=sum(not (0<=s.uv[0]<=1000 and 0<=s.uv[1]<=1000) for s in points)
                    for stage in points:
                        stage.uv=[max(0,min(1000,stage.uv[0]))*(w-1)/1000,
                                  max(0,min(1000,stage.uv[1]))*(h-1)/1000]
                    record=dict(version=self.version,model=self.model,instruction=instruction,prompt=prompt,
                        frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),plan=plan.model_dump(),request_index=i,
                        stored_coordinate_system='native_pixels',response_coordinate_system='normalized_1000',
                        reset_id=reset_id,clipped_waypoints=clipped)
                    atomic(path,record)
                    atomic(self.cache / ('seed%d_request%d.json'%(seed,i)),record)
                    Image.fromarray(frame).save(self.cache / ('seed%d_request%d.png'%(seed,i)))
                    return plan
                except Exception as error:
                    print('metric verifier request failed:', type(error).__name__, str(error), flush=True)
                    if attempt < 2: time.sleep(10 * (attempt+1))
                    if attempt==2: raise RuntimeError('metric verifier: plan generation failed') from error
            raise AssertionError('unreachable')
