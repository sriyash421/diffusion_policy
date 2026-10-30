#!/usr/bin/env python
"""Replay the strict eggplant demos in the official veritas SimplerEnv and record what pi0 sees.

For every episode of a split manifest: reset by the manifest seed, gate the reset against the
zarr (TCP, source and target object positions), then step the recorded zarr actions (Bridge
format: xyz, rpy, gripper {0,1}; converted exactly as eval does) and store, aligned to the zarr
rows, the observation pi0 gets at eval time:

  image     (T, 224, 224, 3) uint8   3rd_view_camera frame, cv2 LANCZOS4 resize (adapter)
  proprio   (T, 7)  float32          BridgeSimplerAdapter.preprocess_proprio: xyz, rpy, openness
  eef_pos   (T, 8)  float32          obs['agent']['eef_pos'] (base-frame pose + gripper width)
  tcp_world (T, 3)  float32          obs['extra']['tcp_pose'][:3]
  tcp_err   (T,)    float32          |tcp_world - zarr end_effector_pose[:3]| per row (metres)

The gripper openness is the one quantity the zarr lacks; the TCP error tells how faithful the
replay is. One ``ep{idx:04d}.npz`` per episode (skipped if present) plus ``replay_summary.json``.
Runs in the veritas venv (scripts/simpler/veritas_pi0/step2_env.sh).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

VERITAS_ROOT = os.environ.get("VERITAS_ROOT", "/gscratch/robotics/harine/veritas")
if VERITAS_ROOT not in sys.path:
    sys.path.insert(0, VERITAS_ROOT)

import cv2  # noqa: E402
import zarr  # noqa: E402
from transforms3d.euler import euler2axangle  # noqa: E402


def convert_maniskill(action: np.ndarray) -> np.ndarray:
    """Bridge-format zarr action -> widowx env action, as scripts/simpler/eval_diffusion does:
    rpy -> axis-angle, gripper [0,1] -> {-1,+1} at 0.5 (the same rule as the official
    BridgeSimplerAdapter.postprocess)."""
    a = np.asarray(action, dtype=np.float32).copy()
    axis, angle = euler2axangle(float(a[3]), float(a[4]), float(a[5]))
    a[3:6] = np.asarray(axis, dtype=np.float32) * np.float32(angle)
    a[6] = 1.0 if a[6] > 0.5 else -1.0
    return a


def make_adapter(task: str):
    """The official BridgeSimplerAdapter, instantiated from config/eval/bridge.yaml like
    EvalAgent does, so preprocessing is the eval one by construction."""
    from diffusion_policy.pi0_context import official
    return official.make_adapter(official.compose_eval_cfg(task))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", default="all",
                    help="'all' (every manifest episode), a split name list like 'train,val' or"
                         " explicit indices '3,7,9'")
    ap.add_argument("--task", default=None, help="default: manifest env_task")
    args = ap.parse_args()

    man = json.load(open(args.split_file))
    task = args.task or man["env_task"]
    if args.episodes == "all":
        eps = sorted(int(k) for k in man["episodes"])
    elif all(p.strip() in ("train", "val", "test") for p in args.episodes.split(",")):
        eps = sorted(set(i for p in args.episodes.split(",") for i in man[p.strip()]))
    else:
        eps = [int(x) for x in args.episodes.split(",")]
    os.makedirs(args.out, exist_ok=True)

    root = zarr.open(man["zarr_path"], "r")
    ends = root["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    actions = root["data/actions"]
    ee_zarr = root["data/obs/end_effector_pose"]
    src_zarr = root["data/obs/insertive_asset_pose"]
    tgt_zarr = root["data/obs/receptive_asset_pose"]

    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    env = simpler_env.make(task)
    base = env.unwrapped
    adapter = make_adapter(task)
    image_size = tuple(adapter.image_size)

    summary_path = os.path.join(args.out, "replay_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    t_start = time.time()
    for n, ep in enumerate(eps):
        out_path = os.path.join(args.out, f"ep{ep:04d}.npz")
        if os.path.exists(out_path) and str(ep) in summary:
            continue
        info = man["episodes"][str(ep)]
        seed = int(info["seed"])
        s, e = int(starts[ep]), int(ends[ep])
        T = e - s
        acts = actions[s:e]
        ee_rows = ee_zarr[s:e, :3]

        obs, _ = env.reset(seed=seed)
        gate = dict(
            ee=float(np.linalg.norm(np.asarray(obs["extra"]["tcp_pose"])[:3] - ee_rows[0])),
            source=float(np.linalg.norm(np.asarray(base.episode_source_obj.pose.p) - src_zarr[s, :3])),
            target=float(np.linalg.norm(np.asarray(base.episode_target_obj.pose.p) - tgt_zarr[s, :3])),
        )
        images = np.zeros((T, *image_size, 3), dtype=np.uint8)
        proprio = np.zeros((T, 7), dtype=np.float32)
        eef = np.zeros((T, 8), dtype=np.float32)
        tcp = np.zeros((T, 3), dtype=np.float32)
        for t in range(T):
            frame = get_image_from_maniskill2_obs_dict(env, obs)
            images[t] = cv2.resize(frame, image_size, interpolation=cv2.INTER_LANCZOS4)
            proprio[t] = adapter.preprocess_proprio(obs)
            eef[t] = obs["agent"]["eef_pos"]
            tcp[t] = np.asarray(obs["extra"]["tcp_pose"])[:3]
            if t < T - 1:
                obs, _, _, _, _ = env.step(convert_maniskill(acts[t]))
        tcp_err = np.linalg.norm(tcp - ee_rows, axis=1).astype(np.float32)
        np.savez_compressed(out_path, image=images, proprio=proprio, eef_pos=eef,
                            tcp_world=tcp, tcp_err=tcp_err, seed=seed, zarr_start=s, zarr_end=e)
        summary[str(ep)] = dict(seed=seed, T=T, gate=gate, tcp_err_max=float(tcp_err.max()),
                                tcp_err_mean=float(tcp_err.mean()),
                                openness_min=float(proprio[:, 6].min()),
                                openness_max=float(proprio[:, 6].max()),
                                instruction=env.get_language_instruction())
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"[{n + 1}/{len(eps)}] ep {ep} seed {seed} T={T} gate ee={gate['ee'] * 1e3:.2f}mm "
              f"src={gate['source'] * 1e3:.1f}mm tgt={gate['target'] * 1e3:.1f}mm "
              f"tcp_err max={tcp_err.max() * 1e3:.2f}mm  ({time.time() - t_start:.0f}s)", flush=True)

    errs = np.array([summary[str(ep)]["tcp_err_max"] for ep in eps if str(ep) in summary])
    gates = np.array([summary[str(ep)]["gate"]["ee"] for ep in eps if str(ep) in summary])
    print(f"done: {len(errs)} episodes; reset EE err max {gates.max() * 1e3:.2f} mm; replay TCP err "
          f"max {errs.max() * 1e3:.2f} mm, median of per-episode max {np.median(errs) * 1e3:.2f} mm; "
          f"{int((errs <= 1e-3).sum())} episodes within 1 mm")


if __name__ == "__main__":
    main()
