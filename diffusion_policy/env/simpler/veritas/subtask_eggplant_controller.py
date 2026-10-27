"""Eggplant adaptation of the two-subtask diagnostic, unchanged candidate scoring."""
import numpy as np
from diffusion_policy.env.simpler.veritas.subtask_controller import PositionChecks
from diffusion_policy.env.simpler.veritas.subtask_two_controller import TwoSubtaskController,TwoSubtaskPlanProvider

class EggplantPlanProvider(TwoSubtaskPlanProvider):
    version='veritas_two_subtask_eggplant_plan_v2'
    def make_prompt(self,instruction,w,h):
        prompt=super().make_prompt(instruction,w,h)
        return (prompt.replace('green cube','eggplant').replace('yellow cube','yellow basket')
            .replace('the cube','the eggplant').replace('lower onto its top','lower into its interior')
            .replace('correct stack height','height inside the basket'))

class BasketPositionChecks(PositionChecks):
    def __init__(self,source_bbox,target_bbox):
        super().__init__()
        self.target_half=np.asarray(target_bbox)[:3]/2
        self.goal_height=float((np.asarray(source_bbox)[2]+np.asarray(target_bbox)[2])/2)
    def update(self,tcp,source,target,opened):
        checks=super().update(tcp,source,target,opened)
        offset=np.asarray(source)[:3]-np.asarray(target)[:3]
        inside=bool(np.all(np.abs(offset[:2])<=self.target_half[:2]))
        height_ok=bool(abs(offset[2]-self.goal_height)<.025)
        checks.update(basket_center_inside=inside,basket_height_valid=height_ok,
                      target_support_height_m=self.goal_height)
        checks['valid_by_stage'][3]=bool(inside and height_ok and checks['object_stable'] and opened)
        return checks

class EggplantSubtaskController(TwoSubtaskController):
    provider_class=EggplantPlanProvider
    budget_group='subtask_eggplant_one_episode_20261004'
    @staticmethod
    def physical_progress(stage,checks):
        if stage==0:return TwoSubtaskController.physical_progress(stage,checks)
        return -(checks['source_target_xy_m']+abs(checks['source_target_height_m']-checks['target_support_height_m']))
