#!/usr/bin/env python
"""Gemini plans and the verifier's choice against A*, on a few recorded pi0 demos.

For each demo and each planner model (plans from ``veritas/plan_simpler_episodes.py``):
  <out>/ep{idx}_plans.png   first frame per model: numbered waypoints with their tolerance
                            circles, and the demo's whole end-effector path (A*) as dots
  <out>/ep{idx}.mp4         one frame per 4-step decision, one panel per model: upcoming
                            waypoints, the 4 base-pi0 candidate endpoints (numbered squares),
                            the verifier's pick (ring), the A* endpoint (X) and the EE (circle)
Tracker targets are replayed per model from the demo's TCP path, exactly as
``build_veritas_targets.py`` does, so each panel shows what that model's plan would score.

    python plan_astar_viz.py --task stack --source pi0_demos --episodes 3 17 42 \\
        --models gemini-3.8-flash gemini-3-flash-preview gemini-2.5-flash-lite
(veritas venv; CPU)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [DP_ROOT, HERE]
from diffusion_policy.env.simpler.veritas.camera import CameraModel  # noqa: E402
from diffusion_policy.env.simpler.veritas.tracker import (  # noqa: E402
    SimplerTrackerSnapshotStore, decode_targets)
from diffusion_policy.policy.veritas_verifier import VeritasPixelVerifier  # noqa: E402
from diffusion_policy.veritas.plan_store import has_plan, load_plan  # noqa: E402
from plot_n_sweep import STEP2  # noqa: E402
from precompute_proprio_replay import convert_maniskill  # noqa: E402

H = 4
# Okabe-Ito, RGB; every element also has its own marker shape and a text label
WP, ACTIVE, CAND, PICK, ASTAR, EE = ((153, 153, 153), (230, 159, 0), (86, 180, 233),
                                     (0, 114, 178), (213, 94, 0), (255, 255, 255))


def put(img, text, xy, scale=0.45, color=(0, 0, 0), bg=(255, 255, 255)):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(xy[0]), int(xy[1])
    cv2.rectangle(img, (x - 1, y - h - 2), (x + w + 1, y + 3), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def ipt(uv):
    return int(round(uv[0])), int(round(uv[1]))


def draw_plan(img, plan, faint=False):
    for i, wp in enumerate(plan.waypoints):
        c = WP if faint else ACTIVE
        cv2.circle(img, ipt(wp.uv), int(wp.tol_px), c, 1, cv2.LINE_AA)
        cv2.circle(img, ipt(wp.uv), 4, c, -1)
        put(img, str(i + 1), (wp.uv[0] + 5, wp.uv[1] - 5), 0.4)


def caption(img, lines):
    strip = np.full((18 * len(lines) + 6, img.shape[1], 3), 255, np.uint8)
    for j, l in enumerate(lines):
        cv2.putText(strip, l, (6, 16 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([img, strip], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--source", default="pi0_demos")
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True,
                    help="plan dirs <source>/plan_viz/<model>; 'pipeline' = <source>/plans_h4")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    src = f"{STEP2}/{a.task}/{a.source}"
    out = a.out or f"{STEP2}/plan_viz/{a.task}_{a.source}"
    os.makedirs(out, exist_ok=True)
    tj = json.load(open(f"{src}/plans_h4/veritas_targets.json"))
    cam = CameraModel.load(tj["camera_file"])
    ver = VeritasPixelVerifier(camera_file=tj["camera_file"], lookahead_k=tj["lookahead_k"],
                               lookahead_alpha=tj["lookahead_alpha"], default_tol_px=60.0,
                               horizon=H, normalize="none", plan_done_mode="zero", health_check=False)
    import imageio
    import zarr
    z = zarr.open(f"{src}/demos.zarr", "r")
    ends = z["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    man = {r["demo"]: r for r in map(json.loads, open(f"{src}/manifest.jsonl")) if r}
    summary = {}
    for ep in a.episodes:
        s, e = int(starts[ep]), int(ends[ep])
        T = e - s
        ee = np.asarray(z["data/obs/end_effector_pose"][s:e, :3], np.float64)
        acts = np.asarray(z["data/actions"][s:e])
        frames = imageio.mimread(f"{src}/demo_videos/{man[ep]['video']}", memtest=False)
        cand = np.load(f"{src}/candidates_ct/ep{ep:04d}.npz")["ctx_actions_env"]     # (T, 4, H, 7)
        path_px = cam.project(ee)
        plans = {}
        for m in a.models:
            p = f"{src}/plans_h4/ep{ep}.json" if m == "pipeline" else f"{src}/plan_viz/{m}/ep{ep}.json"
            if has_plan(p):
                plans[m] = load_plan(p)
        # static: first frame per model
        panels = []
        for m, plan in plans.items():
            img = frames[0].copy()
            for q in path_px:
                cv2.circle(img, ipt(q), 2, ASTAR, -1)
            draw_plan(img, plan)
            cv2.circle(img, ipt(path_px[0]), 7, EE, 2)
            panels.append(caption(img, [f"{os.path.basename(m)}: {len(plan.waypoints)} waypoints",
                                        "numbered circles = waypoints (radius = tol), dots = demo EE path (A*)"]))
        if panels:
            cv2.imwrite(f"{out}/ep{ep}_plans.png", cv2.cvtColor(np.concatenate(panels, 1), cv2.COLOR_RGB2BGR))
        # per decision, per model
        ts = np.arange(0, T - H + 1, H)
        astar = np.stack([[convert_maniskill(x) for x in acts[t:t + H]] for t in ts])
        allc = torch.as_tensor(np.concatenate([cand[ts], astar[:, None]], 1).reshape(-1, H, 7), dtype=torch.float32)
        tcp = torch.as_tensor(np.repeat(ee[ts], 5, 0), dtype=torch.float32)
        endp = ver.endpoints(dict(end_effector_pose=tcp), allc).numpy().reshape(len(ts), 5, 3)
        endp_px = cam.project(endp)
        video, stats = [], {}
        per_model = {}
        for m, plan in plans.items():
            tg = SimplerTrackerSnapshotStore(ee, [T], {0: plan}, cam).targets[ts]
            cost = ver.cost(dict(end_effector_pose=tcp, veritas_targets=torch.as_tensor(
                np.repeat(tg, 5, 0), dtype=torch.float32)), allc).numpy().reshape(len(ts), 5)
            d = np.linalg.norm(endp[:, :4] - endp[:, 4:], axis=-1)
            inf = np.ptp(cost[:, :4], 1) > 0
            stats[m] = dict(decisions=int(len(ts)), informative=int(inf.sum()),
                            pick_is_closest=int(((cost[:, :4].argmin(1) == d.argmin(1)) & inf).sum()),
                            astar_best=int(((cost[:, 4] < cost[:, :4].min(1)) & inf).sum()))
            per_model[m] = (tg, cost)
        for k, t in enumerate(ts):
            row = []
            for m, plan in plans.items():
                tg, cost = per_model[m]
                img = frames[t].copy()
                draw_plan(img, plan, faint=True)
                up, tol = decode_targets(tg[k])
                for j, uv in enumerate(up):
                    cv2.circle(img, ipt(uv), int(tol), ACTIVE, 2 if j == 0 else 1, cv2.LINE_AA)
                tie = np.ptp(cost[k, :4]) == 0
                pick = int(cost[k, :4].argmin())
                for c in range(4):
                    u, v = ipt(endp_px[k, c])
                    cv2.rectangle(img, (u - 4, v - 4), (u + 4, v + 4), CAND, -1)
                    put(img, str(c + 1), (u + 6, v + 4), 0.38)
                if not tie:
                    cv2.circle(img, ipt(endp_px[k, pick]), 11, PICK, 2, cv2.LINE_AA)
                cv2.drawMarker(img, ipt(endp_px[k, 4]), ASTAR, cv2.MARKER_TILTED_CROSS, 14, 2)
                put(img, "A*", (endp_px[k, 4][0] + 8, endp_px[k, 4][1] - 6), 0.4)
                cv2.circle(img, ipt(cam.project(ee[t])), 7, EE, 2)
                rank = int((cost[k, :4] < cost[k, 4]).sum()) + 1
                row.append(caption(img, [
                    f"{os.path.basename(m)} | step {t}/{T} | " + ("all tied (no ranking)" if tie else f"pick = cand {pick + 1}"),
                    "costs " + " ".join(f"{x:.0f}" for x in cost[k, :4]) + f" | A* {cost[k, 4]:.0f} (rank {rank}/5)",
                    "squares = candidate endpoints, ring = pick, X = A*, white = EE now"]))
            video.append(np.concatenate(row, 1))
        if video:
            imageio.mimsave(f"{out}/ep{ep}.mp4", video, fps=1)
        summary[str(ep)] = dict(seed=man[ep]["seed"], T=T, models=stats)
        print(ep, json.dumps(stats), flush=True)
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
