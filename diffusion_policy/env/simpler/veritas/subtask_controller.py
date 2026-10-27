"""Metric-triggered subtask planning; observed positions veto advancement.

One stacking pilot. Gemini produces trajectories, never completion judgments.
All requests share the real rollout ID and the trial's $10 budget group.
"""
import json
import uuid
from pathlib import Path
import numpy as np
from diffusion_policy.env.simpler.veritas.pixel_gripper_plan import PixelGripperPlanProvider
from diffusion_policy.policy.veritas_gripper_verifier import PixelGripperTracker

STAGES=('approach_source','pick_source','transport_source','stack_source')
INSTRUCTIONS=(
    'Move the OPEN gripper to the green cube, ready to grasp. Do not transport it yet.',
    'Grasp the green cube, CLOSE the gripper, and lift it clearly off its initial support.',
    'Carry the grasped green cube above the yellow cube. Keep the gripper CLOSED.',
    'Lower the green cube onto the yellow cube, OPEN the gripper, then withdraw without disturbing the stack.')


class PositionChecks:
    def __init__(self):
        self.initial_z=None;self.history=[];self.held=False

    def update(self,tcp,source,target,opened):
        tcp=np.asarray(tcp)[:3];src=np.asarray(source)[:3];tgt=np.asarray(target)[:3]
        if self.initial_z is None:self.initial_z=float(src[2])
        self.history.append((tcp.copy(),src.copy()));self.history=self.history[-9:]
        near=float(np.linalg.norm(tcp-src));lift=float(src[2]-self.initial_z)
        coupled=False
        for (pt,ps),(ct,cs) in zip(self.history,self.history[1:]):
            ds=cs-ps;dt=ct-pt
            coupled |= np.linalg.norm(ds)>.002 and np.linalg.norm(ds-dt)<.012
        if not opened and near<.07 and lift>.012 and coupled:self.held=True
        if opened or near>.08 or lift<.005:self.held=False
        xy=float(np.linalg.norm(src[:2]-tgt[:2]));height=float(src[2]-tgt[2])
        stable=len(self.history)>=3 and all(np.linalg.norm(b[1]-a[1])<.005
            for a,b in zip(self.history[-3:],self.history[-2:]))
        checks=[near<.05,bool(self.held),bool(self.held) and xy<.06 and height>.02,
                xy<.025 and abs(height-.03)<.015 and stable and opened]
        return dict(valid_by_stage=[bool(x) for x in checks],held=bool(self.held),
            tcp_source_distance_m=near,source_target_distance_m=float(np.linalg.norm(src-tgt)),
            source_target_xy_m=xy,source_target_height_m=height,source_lift_m=lift,
            object_stable=bool(stable),gripper_open=bool(opened),co_motion=bool(coupled))


class SubtaskPlanProvider(PixelGripperPlanProvider):
    version='veritas_subtask_trajectory_plan_v1'
    stage=0
    feedback={}

    def make_prompt(self,instruction,w,h):
        return f'''Whole task: {instruction}
Subtasks in order: move to green cube; pick up green cube; carry above yellow cube;
stack green cube on yellow cube and release.
Generate a short 5-10 waypoint END-EFFECTOR trajectory ONLY for CURRENT subtask
{self.stage+1}: {INSTRUCTIONS[self.stage]}
Do not include subsequent subtasks. Use the CURRENT image, not the initial scene.
The completion metric is reaching these waypoints in order within tol_px for
min_hold observed frames; actual object positions independently validate completion.
If the preceding trajectory completed but validation failed, repair this same
subtask. Closing fingers or reaching a point alone does not prove object possession.
Every waypoint needs uv, tol_px, min_hold, skippable, weight, gripper (open/closed).
uv are absolute image locations on NORMALIZED [0,1000] coordinates, u right,v down;
the caller converts to the native {w}x{h} image. tol_px uses native pixels, typically
50-60; min_hold 1 for transit,2-3 for contact. Last waypoint must not be skippable.
gripper is the desired state on arrival, held until the next waypoint is reached.
Observed position checks and prior-plan status: {json.dumps(self.feedback)}
Return JSON only matching the supplied schema.'''


class SubtaskController:
    stages=STAGES
    provider_class=SubtaskPlanProvider
    budget_group="subtask_one_episode_20261004"
    def __init__(self,root,out,seed,instruction,camera):
        self.seed=seed;self.instruction=instruction;self.camera=camera;self.stage=0
        self.provider=self.provider_class(root,'stack',out,max_requests_per_episode=None,
            budget_group=self.budget_group,group_cap_usd=10.)
        self.provider.rollout_id=uuid.uuid4().hex
        self.checks=PositionChecks();self.current=None;self.complete=False
        self.last_request=-100;self.last_progress=0.;self.last_progress_t=0
        self.plan_calls=0;self.budget_exhausted=False;self.events=[]
        self.event_path=Path(out)/f'seed{seed}_subtask_events.jsonl'
        self.metrics={}

    def _plan(self,frame,t,tcp,reason,checks):
        self.provider.stage=self.stage
        self.provider.feedback=dict(reason=reason,t=t,position_checks=checks,
            previous_plan_progress=None if self.current is None else self.current.episode.progress,
            tcp_uv=self.camera.project(np.asarray(tcp)[:3]).tolist())
        try:plan=self.provider.get(self.seed,frame,self.instruction)
        except RuntimeError as error:
            if 'budget exhausted' in str(error):
                self.budget_exhausted=True
                if self.current is None:raise
                return
            raise
        self.plan_calls+=1;self.last_request=t;self.last_progress_t=t;self.last_progress=0.
        self.current=PixelGripperTracker(plan,self.camera,np.asarray(tcp)[:3],initial_open=checks["gripper_open"])
        event=dict(t=t,stage=self.stages[self.stage],reason=reason,position_checks=checks,
            plan_call=self.plan_calls,request_index=json.loads((self.provider.cache/f'seed{self.seed}.json').read_text())['request_index'])
        self.events.append(event)
        with self.event_path.open('a') as f:f.write(json.dumps(event)+'\n')

    def observe(self,frame,t,tcp,src,tgt,opened,can_request):
        checks=self.checks.update(tcp,src,tgt,opened)
        if self.current is None:self._plan(frame,t,tcp,'initial_subtask',checks)
        state=self.current.update(tcp)
        progress=self.current.episode.progress
        if progress>self.last_progress:
            self.last_progress_t=t;self.last_progress=progress
        old_tracker=self.current
        if can_request and not self.complete and t>self.last_request and not self.budget_exhausted:
            done=self.current.episode.done
            if done and checks['valid_by_stage'][self.stage]:
                event=dict(t=t,stage=self.stages[self.stage],event='metric_and_positions_validated',position_checks=checks)
                self.events.append(event)
                with self.event_path.open('a') as f:f.write(json.dumps(event)+'\n')
                self.stage+=1
                if self.stage==len(STAGES):self.complete=True
                else:self._plan(frame,t,tcp,'next_subtask_after_validation',checks)
                if self.current is not old_tracker:state=self.current.update(tcp)
            elif done and t-self.last_request>=8:
                self._plan(frame,t,tcp,'repair_current_subtask_position_validation_failed',checks)
                if self.current is not old_tracker:state=self.current.update(tcp)
            elif t-self.last_progress_t>=24 and t-self.last_request>=24:
                self._plan(frame,t,tcp,'repair_current_subtask_waypoint_progress_stalled',checks)
                if self.current is not old_tracker:state=self.current.update(tcp)
        self.metrics=dict(phase=STAGES[min(self.stage,3)],held=checks['held'],
            wp_progress=self.current.episode.progress,wp_done=bool(self.current.episode.done),
            position_checks=checks,subtasks_verified=self.stage,all_subtasks_verified=self.complete,
            plan_calls=self.plan_calls,budget_exhausted=self.budget_exhausted,
            version='veritas_subtask_trajectory_v1')
        return state
