"""Two-stage diagnostic: pixel guidance, physical completion, fresh stage plans."""
import json
from diffusion_policy.env.simpler.veritas.subtask_controller import SubtaskController, SubtaskPlanProvider


class TwoSubtaskPlanProvider(SubtaskPlanProvider):
    version='veritas_two_subtask_plan_v2'

    def make_prompt(self,instruction,w,h):
        goal=(
            'Approach the green cube with open fingers, close to grasp, and lift just '
            '2-4 cm clear of its support. Stop this trajectory while holding the cube.',
            'The green cube is ALREADY GRASPED AND LIFTED. Carry it above the yellow '
            'cube, lower onto its top, OPEN fingers to release, then withdraw. Do not '
            'repeat the grasp or add unnecessary lifting. Include the release explicitly.'
        )[self.stage]
        return f'''Task: {instruction}. There are exactly TWO subtasks:
1. grasp and lift; 2. place on target, release, and withdraw.
Generate 5-10 END-EFFECTOR waypoints only for current subtask {self.stage+1}:
{goal}
Use the CURRENT image and observed metrics below. Positions describe current state;
waypoints guide candidate ranking, but physical metrics determine completion.
Grasp completion requires observed lift and object/gripper co-motion. Placement
requires object alignment, correct stack height, release, stability, and withdrawal.
Every waypoint needs uv, tol_px, min_hold, skippable, weight, gripper (open/closed).
uv uses NORMALIZED [0,1000], u right, v down; converted to {w}x{h} native pixels.
tol_px uses native pixels (typically 50-60). min_hold is 1 for transit, 2 for contact.
Last waypoint is not skippable. gripper specifies state on arrival, retained until
the next waypoint. Keep closed throughout carrying; open only at placement.
Current observations and repair reason: {json.dumps(self.feedback)}
Return only JSON matching the schema.'''


class TwoSubtaskController(SubtaskController):
    stages=('grasp_and_lift','place_and_release')
    provider_class=TwoSubtaskPlanProvider
    budget_group='subtask_two_one_episode_20261004'

    @staticmethod
    def physical_complete(stage, checks):
        if stage==0:
            return bool(checks['held'] and checks['source_lift_m']>=.02)
        return bool(checks['valid_by_stage'][3] and checks['tcp_source_distance_m']>=.05)

    @staticmethod
    def physical_progress(stage,checks):
        if stage==0:
            return (min(max(checks['source_lift_m'],0)/.02,1) if checks['held']
                    else -checks['tcp_source_distance_m'])
        # Meter-valued error: lateral alignment and target stack height.
        return -(checks['source_target_xy_m']+abs(checks['source_target_height_m']-.03))

    def observe(self,frame,t,tcp,src,tgt,opened,can_request):
        checks=self.checks.update(tcp,src,tgt,opened)
        if self.current is None:
            self._plan(frame,t,tcp,'initial_subtask',checks)
            self.best_physical_progress=self.physical_progress(self.stage,checks)
        state=self.current.update(tcp)
        progress=self.physical_progress(min(self.stage,1),checks)
        if progress>self.best_physical_progress+.005:
            self.best_physical_progress=progress;self.last_progress_t=t
        if can_request and not self.complete and not self.budget_exhausted and t>self.last_request:
            previous=self.current
            if self.physical_complete(self.stage,checks):
                event=dict(t=t,stage=self.stages[self.stage],event='physical_metrics_validated',
                    pixel_plan_done=bool(self.current.episode.done),position_checks=checks)
                self.events.append(event)
                with self.event_path.open('a') as f:f.write(json.dumps(event)+'\n')
                self.stage+=1
                if self.stage==2:self.complete=True
                else:
                    self._plan(frame,t,tcp,'next_subtask_after_physical_validation',checks)
                    self.best_physical_progress=self.physical_progress(self.stage,checks)
            elif t-self.last_progress_t>=24 and t-self.last_request>=24:
                self._plan(frame,t,tcp,'repair_current_subtask_physical_progress_stalled',checks)
                self.best_physical_progress=progress
            if self.current is not previous:state=self.current.update(tcp)
        self.metrics=dict(phase=self.stages[min(self.stage,1)],held=checks['held'],
            wp_progress=self.current.episode.progress,wp_done=bool(self.current.episode.done),
            physical_progress=progress,position_checks=checks,subtasks_verified=self.stage,
            all_subtasks_verified=self.complete,plan_calls=self.plan_calls,
            budget_exhausted=self.budget_exhausted,version='veritas_two_subtask_physical_v2')
        return state
