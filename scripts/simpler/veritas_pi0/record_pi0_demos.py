#!/usr/bin/env python
"""Record successful raw-pi0 rollouts as demos, in the OpenVLA collection's zarr layout.

Runs the official EvalAgent through ``run_pi0_eval.LoggedEvalAgent`` (seeded resets, resume,
wandb) and wraps ``env.step`` to record the pre-step state of every step, the same fields as
``data/_collection_code/.../collect_trajectories.py``:

  data/actions (T,7)   Bridge format: xyz, euler rpy, gripper {0 closed, 1 open}
  data/rewards, data/dones
  data/obs/{agentview_image (224), end_effector_pose, insertive_asset_pose, receptive_asset_pose,
            insertive_asset_in_receptive_asset_frame, binary_contact (check_grasp),
            last_gripper_action, last_arm_action, arm_joint_pos, joint_vel,
            end_effector_vel_lin_ang_b}
  meta/episode_ends

An episode is kept if any step is rewarded; it is cut after the first rewarded step (the OpenVLA
collector stops there, while the official loop finishes the 4-step chunk). Each kept demo also
gets a clean 3rd-view mp4 (no overlays) and a line in ``manifest.jsonl``. Recording stops once
``+n_demos`` are kept.

    EVAL_SCRIPT=record_pi0_demos.py EXTRA="+n_demos=200" EVAL_SEEDS=seeds.json \\
        OUT_DIR=$STEP2/<task>/pi0_demos sbatch eval_pi0.sbatch
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_pi0_eval  # noqa: E402  (sets up the veritas import path)

ZARR_NAME = "demos.zarr"
CHUNK = 1024


def _pose_vec(pose):
    return np.concatenate([np.asarray(pose.p, np.float32), np.asarray(pose.q, np.float32)])


def _to_bridge(env_action):
    """Env action (xyz, axis-angle, gripper +-1) -> Bridge action, inverting the adapter's
    postprocess (euler2axangle, 2*(g>0.5)-1)."""
    from transforms3d.euler import axangle2euler
    a = np.asarray(env_action, np.float64)
    angle = float(np.linalg.norm(a[3:6]))
    rpy = axangle2euler(a[3:6] / angle, angle) if angle > 1e-12 else (0.0, 0.0, 0.0)
    return np.concatenate([a[:3], rpy, [1.0 if a[6] > 0 else 0.0]]).astype(np.float32)


class DemoWriter:
    """Appends episodes to one zarr store; episode_ends is written last, so a store cut short
    mid-append is repaired on open by trimming every array back to episode_ends[-1]."""

    def __init__(self, path, env_name, instruction):
        import zarr
        from numcodecs import Blosc
        self.comp = Blosc(cname="zstd", clevel=3)
        self.root = zarr.open_group(path, mode="a")
        self.root.attrs.update(env_name=env_name, task_description=instruction, source="pi0")
        self.data = self.root.require_group("data")
        self.obs = self.data.require_group("obs")
        meta = self.root.require_group("meta")
        if "episode_ends" not in meta:
            meta.create_dataset("episode_ends", shape=(0,), chunks=(CHUNK,), dtype=np.int64,
                                compressor=self.comp)
        self.ends = meta["episode_ends"]
        end = int(self.ends[-1]) if len(self.ends) else 0
        for g in (self.data, self.obs):
            for _, arr in g.arrays():
                if arr.shape[0] != end:
                    arr.resize((end,) + arr.shape[1:])

    @property
    def n_episodes(self):
        return len(self.ends)

    def _append(self, group, name, x):
        x = np.asarray(x)
        if name not in group:
            group.create_dataset(name, shape=(0,) + x.shape[1:], chunks=(CHUNK,) + x.shape[1:],
                                 dtype=x.dtype, compressor=self.comp)
        group[name].append(x)

    def append(self, data, obs):
        for k, v in data.items():
            self._append(self.data, k, v)
        for k, v in obs.items():
            self._append(self.obs, k, v)
        end = (int(self.ends[-1]) if len(self.ends) else 0) + len(data["actions"])
        self.ends.append(np.array([end], np.int64))
        return end


class DemoRecordAgent(run_pi0_eval.LoggedEvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.n_demos = int(cfg.get("n_demos", 200))
        self.demo_video_dir = os.path.join(self.log_dir, "demo_videos")
        os.makedirs(self.demo_video_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.log_dir, "manifest.jsonl")
        self.writer = DemoWriter(os.path.join(self.log_dir, ZARR_NAME), cfg.env.task,
                                 self.env.get_language_instruction())
        self.manifest = self._read_manifest()
        self._wrap_env()

    def _read_manifest(self):
        rows = []
        if os.path.exists(self.manifest_path):
            rows = [json.loads(l) for l in open(self.manifest_path) if l.strip()]
        rows = [r for r in rows if r["demo"] < self.writer.n_episodes]   # drop unwritten tails
        assert len(rows) == self.writer.n_episodes, (len(rows), self.writer.n_episodes)
        with open(self.manifest_path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        return rows

    def _done(self):
        return len(self.manifest) >= self.n_demos

    # ------------------------------------------------------------------ recording
    def _wrap_env(self):
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
        env = self.env
        base = env.unwrapped
        real_reset, real_step = env.reset, env.step

        def reset(*args, **kwargs):
            obs, info = real_reset(*args, **kwargs)
            self._obs, self._steps = obs, []
            self._prev = np.zeros(7, np.float32)
            return obs, info

        def step(action):
            obs = self._obs
            src, tgt = base.episode_source_obj, base.episode_target_obj
            tcp = base.tcp
            qpos = np.asarray(base.agent.robot.get_qpos(), np.float32)
            qvel = np.asarray(base.agent.robot.get_qvel(), np.float32)
            R = tcp.pose.to_transformation_matrix()[:3, :3]
            vel_b = np.concatenate([R.T @ np.asarray(tcp.get_velocity()),
                                    R.T @ np.asarray(tcp.get_angular_velocity())])
            frame = np.asarray(get_image_from_maniskill2_obs_dict(base, obs), np.uint8)
            bridge = _to_bridge(action)
            rec = dict(
                frame=frame, actions=bridge,
                end_effector_pose=np.asarray(obs["extra"]["tcp_pose"], np.float32).reshape(-1)[:7],
                insertive_asset_pose=_pose_vec(src.pose), receptive_asset_pose=_pose_vec(tgt.pose),
                insertive_asset_in_receptive_asset_frame=_pose_vec(tgt.pose.inv() * src.pose),
                binary_contact=np.float32(base.agent.check_grasp(src)),
                last_gripper_action=self._prev[6:7].copy(), last_arm_action=self._prev[:6].copy(),
                arm_joint_pos=qpos[:6], joint_vel=qvel[:6], end_effector_vel_lin_ang_b=vel_b.astype(np.float32))
            out = real_step(action)
            obs2, reward = out[0], out[1]
            rec["rewards"] = np.float32(reward)
            rec["final_frame"] = np.asarray(get_image_from_maniskill2_obs_dict(base, obs2), np.uint8)
            self._steps.append(rec)
            self._obs, self._prev = obs2, bridge
            return out

        env.reset, env.step = reset, step

    def _after_episode(self, cnt, seed, success):
        steps = self._steps
        rewarded = [i for i, r in enumerate(steps) if r["rewards"] > 0]
        self._last = dict(any_success=bool(rewarded), first_success_step=rewarded[0] + 1 if rewarded else None,
                          kept=False, demo=None)
        if not rewarded or self._done() or any(r["seed"] == seed for r in self.manifest):
            return
        steps = steps[:rewarded[0] + 1]
        import cv2
        import imageio
        data = dict(actions=np.stack([s["actions"] for s in steps]),
                    rewards=np.array([s["rewards"] for s in steps], np.float32),
                    dones=np.arange(len(steps)) == len(steps) - 1)
        keys = ["end_effector_pose", "insertive_asset_pose", "receptive_asset_pose",
                "insertive_asset_in_receptive_asset_frame", "binary_contact", "last_gripper_action",
                "last_arm_action", "arm_joint_pos", "joint_vel", "end_effector_vel_lin_ang_b"]
        obs = {k: np.stack([np.asarray(s[k], np.float32) for s in steps]) for k in keys}
        obs["agentview_image"] = np.stack([cv2.resize(s["frame"], (224, 224), interpolation=cv2.INTER_AREA)
                                           for s in steps])
        demo = self.writer.n_episodes
        video = os.path.join(self.demo_video_dir, f"demo{demo:03d}_seed{seed}.mp4")
        frames = [s["frame"] for s in steps] + [steps[-1]["final_frame"]]
        imageio.mimsave(video, frames, fps=5)
        end = self.writer.append(data, obs)
        row = dict(demo=demo, seed=seed, episode=cnt, out_start=end - len(steps), out_end=end,
                   steps=len(steps), steps_run=len(self._steps), success_last_step=bool(success),
                   video=os.path.basename(video))
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self.manifest.append(row)
        self._last.update(kept=True, demo=demo)
        print(f"[record] demo {demo + 1}/{self.n_demos} seed {seed}: {len(steps)} steps", flush=True)

    def _extra_row_fields(self):
        return self._last

    def _finish(self):
        k = len(self.manifest)
        print(f"[record] {k} demos in {os.path.join(self.log_dir, ZARR_NAME)}")
        super()._finish()


if __name__ == "__main__":
    run_pi0_eval.main(agent_cls=DemoRecordAgent)
