"""The fixed SimplerEnv camera as a pinhole model, plus the robot base rotation.

``3rd_view_camera`` is mounted on the WidowX base link and the base is placed at a fixed
per-task pose (``custom_scenes/base_env.py`` ``robot_init_options``), so K, the
world->camera extrinsic and the base rotation are constants of a task. They are
captured once from a live reset (``scripts/simpler/veritas/capture_simpler_camera.py``) into
a json that training can use without a simulator, and re-checked against every live
reset by ``assert_matches``.
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import numpy as np

CAMERA_NAME = '3rd_view_camera'


def quat_wxyz_to_R(q) -> np.ndarray:
    """sapien quat order [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in q[:4])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def project(world_xyz, K, T_wc) -> np.ndarray:
    """world_xyz (..., 3) -> pixel (..., 2); points behind the camera -> NaN."""
    world_xyz = np.asarray(world_xyz, dtype=np.float64)
    shape = world_xyz.shape[:-1]
    pts = world_xyz.reshape(-1, 3)
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    cam = pts @ R.T + t
    z = cam[:, 2]
    safe = z > 1e-6
    uv = np.full((cam.shape[0], 2), np.nan, dtype=np.float64)
    uv[safe, 0] = (K[0, 0] * cam[safe, 0] + K[0, 2] * z[safe]) / z[safe]
    uv[safe, 1] = (K[1, 1] * cam[safe, 1] + K[1, 2] * z[safe]) / z[safe]
    return uv.reshape(*shape, 2)


class CameraModel:
    """Pinhole camera (K, world->camera T) at the native render size, and the robot
    base pose whose rotation maps base-frame action deltas into world."""

    def __init__(self, K, T_wc, image_hw, base_pose, camera_name: str = CAMERA_NAME):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.T_wc = np.asarray(T_wc, dtype=np.float64).reshape(4, 4)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.base_pose = np.asarray(base_pose, dtype=np.float64).reshape(7)  # xyz + wxyz
        self.camera_name = camera_name

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_obs(cls, obs: dict, camera_name: str = CAMERA_NAME) -> 'CameraModel':
        cam = obs['camera_param'][camera_name]
        img = obs['image'][camera_name]['rgb']
        return cls(cam['intrinsic_cv'], cam['extrinsic_cv'], img.shape[:2],
                   obs['agent']['base_pose'], camera_name)

    @classmethod
    def load(cls, path) -> 'CameraModel':
        d = json.loads(pathlib.Path(path).read_text())
        return cls(d['K'], d['T_wc'], d['image_hw'], d['base_pose'],
                   d.get('camera_name', CAMERA_NAME))

    def save(self, path, **extra) -> None:
        d = dict(K=self.K.tolist(), T_wc=self.T_wc.tolist(), image_hw=list(self.image_hw),
                 base_pose=self.base_pose.tolist(), camera_name=self.camera_name, **extra)
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, indent=2))

    # ------------------------------------------------------------------ geometry

    @property
    def R_world_from_base(self) -> np.ndarray:
        return quat_wxyz_to_R(self.base_pose[3:7])

    def project(self, world_xyz) -> np.ndarray:
        return project(world_xyz, self.K, self.T_wc)

    def assert_matches(self, obs: dict, tol: float = 1e-5) -> None:
        """Refuse to score a live env whose camera/base differ from the captured model:
        every pixel target would be silently off."""
        other = CameraModel.from_obs(obs, self.camera_name)
        for name, a, b in (('K', self.K, other.K), ('T_wc', self.T_wc, other.T_wc),
                           ('base_pose', self.base_pose, other.base_pose)):
            if not np.allclose(a, b, atol=tol):
                raise ValueError(f'live {name} differs from the captured camera model:\n'
                                 f'captured={a}\nlive={b}')
        if other.image_hw != self.image_hw:
            raise ValueError(f'live frame is {other.image_hw}, captured {self.image_hw}')


def tcp_xyz(obs: dict) -> np.ndarray:
    """World-frame TCP position from a SimplerEnv obs (``obs['extra']['tcp_pose']`` is
    xyz + quat wxyz; the same quantity the zarr stores as ``end_effector_pose``)."""
    return np.asarray(obs['extra']['tcp_pose'], dtype=np.float64)[:3]
