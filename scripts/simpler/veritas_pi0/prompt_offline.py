#!/usr/bin/env python
"""Score plan sets (model x prompt variant) offline on the strict-passing val demos.

Each plan is replayed through the tracker along the demo's own TCP path, and the 4 base-pi0
candidates plus A* are scored at every frame (as astar_vs_verifier.py does). Per plan set:
  leave_early   the tracker is past the grasp waypoint when the demo first closes the gripper
                (grasp waypoint = the plan waypoint nearest the EE pixel at that moment); the
                failure seen in eval: the verifier then pulls the arm away from the object
  lead_steps    how many steps before the grasp the tracker left it (leave_early demos)
  leave_how     latched (within tol for min_hold frames) or the skip rule that left it
  completed     a successful demo latches the whole plan (else the plan asks for something
                the expert never does)
  astar_best    A* beats all 4 candidates (chance 20%); top1 = pick is closest to A* (25%)
  pregrasp_dz   pick minus candidate-mean endpoint height change (mm) in the 8 steps before
                the grasp; > 0 = the verifier holds the arm up instead of descending
  wp_on_path    fraction of plan waypoints within their tol of the demo's EE path, and the
                final waypoint's distance to the demo's last EE pixel: does the plan look
                like the successful trajectory?

    python prompt_offline.py [--tasks ...] [--sources pi0_demos pi0beta_demos]   (veritas venv)
Plan sets: 'pipeline' = <source>/plans_h4 (gemini-3.8-flash, official), and every
<source>/plan_var/<model>__<variant>/.  Out: $STEP2/expert_analysis/prompt_offline.{json,md}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [DP_ROOT, HERE]
from diffusion_policy.env.simpler.veritas.camera import CameraModel  # noqa: E402
from diffusion_policy.env.simpler.veritas.tracker import SimplerEpisodeTracker  # noqa: E402
from diffusion_policy.policy.veritas_verifier import VeritasPixelVerifier  # noqa: E402
from diffusion_policy.veritas.plan_store import has_plan, load_plan  # noqa: E402
from plot_n_sweep import STEP2  # noqa: E402
from precompute_proprio_replay import convert_maniskill  # noqa: E402

H = 4
PRE = 8


def replay(plan, cam, ee, k):
    """Per-frame targets vector, active waypoint index (anchor = index 0) and skip reason."""
    tr = SimplerEpisodeTracker(plan, cam, ee[0])
    tg, idx, why = [tr.targets_vector(k)], [tr.tracker.i], [None]
    for f in range(1, len(ee)):
        why.append(tr.update(ee[f]).get("skip_reason"))
        tg.append(tr.targets_vector(k))
        idx.append(tr.tracker.i)
    return np.stack(tg), np.array(idx), why, tr.tracker


def episode(src, ep, plan, cam, ver, k, z, starts, ends):
    s, e = int(starts[ep]), int(ends[ep])
    T = e - s
    ee = np.asarray(z["data/obs/end_effector_pose"][s:e, :3], np.float64)
    acts = np.asarray(z["data/actions"][s:e])
    tg, idx, why, tracker = replay(plan, cam, ee, k)
    closed = np.where(acts[:, 6] < 0.5)[0]
    path = cam.project(ee)
    dpath = np.linalg.norm(path[None] - tracker.waypoints[1:, 0][:, None], axis=-1).min(1)   # per plan wp
    r = dict(completed=bool(tracker.done), T=T, wp_on_path=float((dpath <= tracker.tol_px[1:]).mean()),
             wp_path_px=dpath.tolist(), final_wp_to_end_px=float(np.linalg.norm(tracker.waypoints[-1, 0] - path[-1])))
    if len(closed):
        g = int(closed[0])
        px = cam.project(ee[g])
        d = np.linalg.norm(tracker.waypoints[1:, 0] - px, axis=-1)          # plan waypoints only
        j = int(d.argmin()) + 1
        past = np.where(idx > j)[0]
        r.update(grasp_step=g, grasp_wp=j, grasp_wp_dist_px=float(d.min()),
                 leave_early=bool(idx[g] > j), lead_steps=int(g - past[0]) if idx[g] > j else 0,
                 leave_how=(why[past[0]] or "latched") if idx[g] > j else None)
    cand = np.load(f"{src}/candidates_ct/ep{ep:04d}.npz")["ctx_actions_env"]
    ts = np.arange(T - H + 1)
    astar = np.stack([[convert_maniskill(a) for a in acts[t:t + H]] for t in ts])
    allc = torch.as_tensor(np.concatenate([cand[ts], astar[:, None]], 1).reshape(-1, H, 7), dtype=torch.float32)
    obs = dict(end_effector_pose=torch.as_tensor(np.repeat(ee[ts], 5, 0), dtype=torch.float32),
               veritas_targets=torch.as_tensor(np.repeat(tg[ts], 5, 0), dtype=torch.float32))
    cost = ver.cost(obs, allc).numpy().reshape(len(ts), 5)
    endp = ver.endpoints(obs, allc).numpy().reshape(len(ts), 5, 3)
    inf = np.ptp(cost[:, :4], 1) > 1e-6
    pick = cost[:, :4].argmin(1)
    dist = np.linalg.norm(endp[:, :4] - endp[:, 4:], axis=-1)
    dz = 1e3 * (endp[:, :4, 2] - ee[ts, None, 2])
    pre = np.zeros(len(ts), bool)
    if "grasp_step" in r:
        pre[max(0, r["grasp_step"] - PRE):r["grasp_step"] + 1] = True
    r.update(n=int(inf.sum()), astar_best=int(((cost[:, 4] < cost[:, :4].min(1) - 1e-6) & inf).sum()),
             top1=int(((pick == dist.argmin(1)) & inf).sum()),
             pregrasp_dz=(dz[np.arange(len(ts)), pick] - dz.mean(1))[pre & inf].tolist())
    return r


def plan_sets(src):
    sets = {"gemini-3.8-flash__official": f"{src}/plans_h4"}
    for d in sorted(glob.glob(f"{src}/plan_var/*__*")):
        sets[os.path.basename(d)] = d
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["eggplant", "stack", "carrot", "spoon"])
    ap.add_argument("--sources", nargs="+", default=["pi0_demos", "pi0beta_demos"])
    ap.add_argument("--out", default=f"{STEP2}/expert_analysis")
    a = ap.parse_args()
    import zarr
    res = {}
    for t in a.tasks:
        for s in a.sources:
            src = f"{STEP2}/{t}/{s}"
            tj = json.load(open(f"{src}/plans_h4/veritas_targets.json"))
            cam = CameraModel.load(tj["camera_file"])
            ver = VeritasPixelVerifier(camera_file=tj["camera_file"], lookahead_k=tj["lookahead_k"],
                                       lookahead_alpha=tj["lookahead_alpha"], default_tol_px=60.0,
                                       horizon=H, normalize="none", plan_done_mode="zero", health_check=False)
            z = zarr.open(f"{src}/demos.zarr", "r")
            ends = z["meta/episode_ends"][:]
            starts = np.r_[0, ends[:-1]]
            val = json.load(open(f"{src}/split_strict.json"))["val"]
            for name, d in plan_sets(src).items():
                rows = {}
                for ep in val:
                    p = f"{d}/ep{ep}.json"
                    if has_plan(p):
                        rows[ep] = episode(src, ep, load_plan(p), cam, ver, tj["lookahead_k"], z, starts, ends)
                res.setdefault(name, {}).setdefault(t, {})[s] = rows
                print(t, s, name, len(rows), flush=True)
    json.dump(res, open(f"{a.out}/prompt_offline.json", "w"), indent=1)

    def agg(rows):
        rs = [r for src in rows.values() for r in src.values()]
        if not rs:
            return None
        g = [r for r in rs if "leave_early" in r]
        n = sum(r["n"] for r in rs)
        dz = np.concatenate([r["pregrasp_dz"] for r in rs]) if rs else np.zeros(0)
        le = [r for r in g if r["leave_early"]]
        how = {h: sum(r["leave_how"] == h for r in le) / max(1, len(le))
               for h in ("latched", "transit_skip", "stalled_skip", "ahead_skip")}
        return dict(demos=len(rs), leave_early=len(le) / max(1, len(g)), how=how,
                    lead=float(np.mean([r["lead_steps"] for r in le])) if le else 0.0,
                    completed=np.mean([r["completed"] for r in rs]),
                    astar_best=sum(r["astar_best"] for r in rs) / max(1, n),
                    top1=sum(r["top1"] for r in rs) / max(1, n),
                    dz=float(dz.mean()) if len(dz) else float("nan"),
                    on_path=float(np.mean([r["wp_on_path"] for r in rs])),
                    final_px=float(np.median([r["final_wp_to_end_px"] for r in rs])),
                    wp_px=float(np.median([r["grasp_wp_dist_px"] for r in g])) if g else float("nan"))
    md = ["Val split, strict-passing demos, pi0 and Bridge-Beta pooled.", "",
          "| plan set | task | demos | tracker left grasp wp early | lead (steps) "
          "| how: latched / transit / stalled / ahead | plan completed "
          "| A* best (20%) | top-1 (25%) | pre-grasp pick dz (mm) | grasp wp to EE at grasp (px) "
          "| wps on demo path | final wp to demo end (px) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in res:
        for t in a.tasks + ["all"]:
            rows = ({f"{tt}/{s}": v for tt in a.tasks for s, v in res[name].get(tt, {}).items()}
                    if t == "all" else res[name].get(t, {}))
            m = agg(rows)
            if m:
                md.append(f"| {name} | {t} | {m['demos']} | {100 * m['leave_early']:.0f}% | {m['lead']:.1f} | "
                          + " / ".join(f"{100 * v:.0f}" for v in m["how"].values()) + " | "
                          f"{100 * m['completed']:.0f}% | {100 * m['astar_best']:.0f}% | {100 * m['top1']:.0f}% | "
                          f"{m['dz']:+.2f} | {m['wp_px']:.0f} | {100 * m['on_path']:.0f}% | {m['final_px']:.0f} |")
    open(f"{a.out}/prompt_offline.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
