#!/usr/bin/env python
"""Successful demos as planning examples: for N strict-passing TRAIN demos per task (pi0_demos),
the first frame, the projected end-effector path and the pixels the demo actually passed
through at its key moments, in the plan schema:
  1 pre-grasp   4 steps before the gripper first closes
  2 grasp       the frame it closes
  3 lift        4 steps after
  4 carry       halfway (in steps) between lift and the end
  5 place       the last frame (the task is rewarded; the gripper opens here or just after)
Used by plan_prompts.py (few-shot variant) and prompt_gen.py (Gemini writes the prompt).

    python demo_examples.py [--n 5]   (veritas venv)
Out: $STEP2/<task>/pi0_demos/examples/ex{k}.{json,png} and sheet.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_astar_viz import ASTAR, caption, ipt, put  # noqa: E402
from plot_n_sweep import STEP2  # noqa: E402
from prompt_offline import DP_ROOT  # noqa: E402, F401  (puts the repo on sys.path)
from diffusion_policy.env.simpler.veritas.camera import CameraModel  # noqa: E402

TASKS = ["eggplant", "stack", "carrot", "spoon"]
NAMES = ["pre-grasp", "grasp", "lift", "carry", "place"]
LEAD = 4


def key_frames(acts, ee):
    g = int(np.where(acts[:, 6] < 0.5)[0][0])
    T = len(acts)
    lift = min(g + LEAD, T - 1)
    return [max(0, g - LEAD), g, lift, (lift + T - 1) // 2, T - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    a = ap.parse_args()
    import zarr
    for t in a.tasks:
        src = f"{STEP2}/{t}/pi0_demos"
        out = f"{src}/examples"
        os.makedirs(out, exist_ok=True)
        tj = json.load(open(f"{src}/plans_h4/veritas_targets.json"))
        cam = CameraModel.load(tj["camera_file"])
        z = zarr.open(f"{src}/demos.zarr", "r")
        ends = z["meta/episode_ends"][:]
        starts = np.r_[0, ends[:-1]]
        man = {r["demo"]: r for r in (json.loads(l) for l in open(f"{src}/manifest.jsonl") if l.strip())}
        train = json.load(open(f"{src}/split_strict.json"))["train"][:a.n]
        tiles = []
        for k, ep in enumerate(train):
            s, e = int(starts[ep]), int(ends[ep])
            ee = np.asarray(z["data/obs/end_effector_pose"][s:e, :3], np.float64)
            acts = np.asarray(z["data/actions"][s:e])
            rec = json.load(open(f"{src}/plans_h4/ep{ep}.json"))
            frame = imageio.mimread(f"{src}/demo_videos/{man[ep]['video']}", memtest=False)[0]
            path = cam.project(ee)
            kf = key_frames(acts, ee)
            wps = [dict(name=n, uv=[round(float(u), 1) for u in path[f]], step=int(f)) for n, f in zip(NAMES, kf)]
            ex = dict(episode=int(ep), seed=rec["seed"], instruction=rec["instruction"], image_size=rec["image_size"],
                      eef_pixel=[round(float(u), 1) for u in rec["eef_pixel"]], hints=rec["hints"], T=e - s,
                      gripper_close_step=kf[1], path_px=np.round(path, 1).tolist(), waypoints=wps,
                      png=f"ex{k}.png")
            json.dump(ex, open(f"{out}/ex{k}.json", "w"), indent=1)
            img = np.ascontiguousarray(frame[..., :3])
            for q in path:
                cv2.circle(img, ipt(q), 2, ASTAR, -1)
            for j, w in enumerate(wps):
                cv2.circle(img, ipt(w["uv"]), 6, (230, 159, 0), 2, cv2.LINE_AA)
                put(img, f"{j + 1} {w['name']}", (w["uv"][0] + 7, w["uv"][1] - 6), 0.4)
            cv2.circle(img, ipt(rec["eef_pixel"]), 7, (255, 255, 255), 2)
            for h in rec["hints"]:
                cv2.drawMarker(img, ipt(h["uv"]), (0, 158, 115), cv2.MARKER_CROSS, 14, 2)
                put(img, h["name"], (h["uv"][0] + 8, h["uv"][1] + 14), 0.4)
            cv2.imwrite(f"{out}/ex{k}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            tiles.append(caption(img, [f"{t} ep{ep} seed {rec['seed']} | T={e - s}, closes at step {kf[1]}",
                                       "dots = EE path, numbered rings = key moments, crosses = objects, white = EE start"]))
            print(t, ep, [(w["name"], w["uv"]) for w in wps], flush=True)
        cv2.imwrite(f"{out}/sheet.png", cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
