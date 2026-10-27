"""Wide-reset eggplant scene for SimplerEnv, registered on import.

Upstream SimplerEnv / ManiSkill2_real2sim (the ``SimplerEnv`` submodule, pinned to
simpler-env/SimplerEnv 4ab7178) only ships ``PutEggplantInBasketScene-v0``, whose
eggplant spawns on a tiny 2x4 grid. This module adds the wide variant used to collect
``/gscratch/robotics/harine/data/eggplant_reset_core_x-0.22_-0.06_y0.02_0.26`` and maps
it to the ``widowx_put_eggplant_in_basket_wide`` task name, so::

    import diffusion_policy.env.simpler.wide_eggplant  # noqa: F401
    env = simpler_env.make("widowx_put_eggplant_in_basket_wide")

The class is copied verbatim from the RoboMonkey ManiSkill2_real2sim commit 12f55b1
(``harine/wide-eggplant``); keep it identical so resets reproduce the collected starts.
"""

import numpy as np
import simpler_env
from mani_skill2_real2sim.envs.custom_scenes.put_on_in_scene import (
    PutEggplantInBasketScene, PutOnBridgeInSceneEnv)
from mani_skill2_real2sim.utils.registration import register_env
from transforms3d.euler import euler2quat

TASK_NAME = "widowx_put_eggplant_in_basket_wide"
ENV_ID = "PutEggplantInBasketWideScene-v0"


@register_env(ENV_ID, max_episode_steps=120)
class PutEggplantInBasketWideScene(PutEggplantInBasketScene):
    """Like PutEggplantInBasketScene, but the eggplant spawns across a wide reachable
    region of the sink basin instead of the original tiny 2x4 grid. The basket footprint
    is excluded so the eggplant never starts in/touching the basket. All other behavior
    (sink placement, evaluate, prepackaged config, lighting, instruction) is inherited.
    """

    def __init__(self, **kwargs):
        source_obj_name = "eggplant"
        target_obj_name = "dummy_sink_target_plane"  # invisible
        target_xy = np.array([-0.125, 0.025])

        # Reachable-core region inside the sink basin (world XY). The sink envelope is
        # X[-0.2945,-0.0255], Y[-0.0665,0.3265]; we stay well inside it and within reach.
        x_lo, x_hi = -0.22, -0.06
        y_lo, y_hi = 0.02, 0.26
        num_x, num_y = 12, 16  # ~1.5 cm spacing

        # Exclude the basket footprint (dummy_sink_target_plane bbox half-extents
        # [0.065, 0.04]) expanded by ~eggplant half-length + clearance, so the eggplant
        # never starts in/touching the basket.
        basket_half = np.array([0.065, 0.04])
        clear = 0.05
        excl_lo = target_xy - basket_half - clear
        excl_hi = target_xy + basket_half + clear

        grid_pos = []
        for x in np.linspace(x_lo, x_hi, num_x):
            for y in np.linspace(y_lo, y_hi, num_y):
                if (excl_lo[0] <= x <= excl_hi[0]) and (excl_lo[1] <= y <= excl_hi[1]):
                    continue  # skip cells overlapping the basket
                grid_pos.append(np.array([x, y]))

        xy_configs = [np.stack([pos, target_xy], axis=0) for pos in grid_pos]

        quat_configs = [
            np.array([
                euler2quat(0, 0, 0, 'sxyz'),
                [1, 0, 0, 0]
            ]),
            np.array([
                euler2quat(0, 0, 1 * np.pi / 4, 'sxyz'),
                [1, 0, 0, 0]
            ]),
            np.array([
                euler2quat(0, 0, -1 * np.pi / 4, 'sxyz'),
                [1, 0, 0, 0]
            ]),
        ]

        # Call the grandparent __init__ directly: PutEggplantInBasketScene.__init__ would
        # rebuild the narrow grid, so we bypass it and pass the wide configs ourselves.
        PutOnBridgeInSceneEnv.__init__(
            self,
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            rgb_always_overlay_objects=['sink', 'dummy_sink_target_plane'],
            **kwargs,
        )


if TASK_NAME not in simpler_env.ENVIRONMENTS:
    simpler_env.ENVIRONMENTS.append(TASK_NAME)
simpler_env.ENVIRONMENT_MAP[TASK_NAME] = (ENV_ID, {})
