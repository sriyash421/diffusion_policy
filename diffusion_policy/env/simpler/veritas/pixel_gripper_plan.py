"""Fresh waypoint plan, augmented only with desired gripper state at each point."""
from typing import Literal
from pydantic import Field
from diffusion_policy.veritas.schemas import VeritasWaypoint,VeritasPlan
from diffusion_policy.env.simpler.veritas.metric_plan import PlanProvider
from diffusion_policy.env.simpler.veritas.simpler_client import PROMPT_V1_UPSTREAM


class GripWaypoint(VeritasWaypoint):
    uv: list[float] = Field(min_length=2,max_length=2)
    gripper: Literal['open','closed']


class GripPlan(VeritasPlan):
    waypoints: list[GripWaypoint]=Field(min_length=5,max_length=10)


class PixelGripperPlanProvider(PlanProvider):
    version='veritas_pixel_gripper_plan_v1'
    schema=GripPlan
    context=''

    def make_prompt(self,instruction,w,h):
        prompt=PROMPT_V1_UPSTREAM.replace('{instruction}',instruction)
        return prompt+f'''\nFor each waypoint add gripper="open" or "closed": the desired command
on arrival at that waypoint, retained until the next waypoint is reached.
Include the grasp closure and placement release at their appropriate locations.
Coordinate convention for this response: uv are absolute image locations expressed
on a NORMALIZED [0,1000] grid (u right, v down), NOT native pixels.
The caller converts uv to the {w}x{h} image. tol_px remains in native image pixels.
Use the same spatial trace regardless of how candidate actions will be scored.
{self.context}'''

    def validate_plan(self,plan):
        pass  # Bounds and required open/close labels are checked by GripPlan.

    def plan_points(self,plan):
        return plan.waypoints
