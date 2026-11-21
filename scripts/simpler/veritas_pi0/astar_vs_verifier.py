#!/usr/bin/env python
"""Does the Veritas pixel verifier rank base-pi0 candidates the way the expert (A*) would?

On every frame t of the expert demos that have candidates (``$STEP2/<task>/candidates_ct``,
4 base-pi0 chunks per frame scored by precompute_context_candidates.py), A* = the demo's own next
4 actions (zarr ``data/actions[t:t+4]``, in env format). ``--source openvla`` (default) uses the
OpenVLA strict demos; ``pi0_demos`` / ``pi0beta_demos`` the recorded pi0 demos
(``$STEP2/<task>/<source>/{plans_h4,replay,candidates_ct}``). Candidates are always base pi0.
Distances to A*:
  endpoint  world-xyz distance between the chunk endpoints (what the verifier scores)
  xyz_rmse  RMSE over the 4x3 xyz deltas
  norm7     RMSE over all 7 dims, each divided by the task's expert per-dim std
A* is also scored by the same verifier call, so it can be ranked against the candidates.

Frames where the verifier gives all four candidates the same cost (plan done, or every endpoint
within tol of every lookahead waypoint) cannot rank anything; they are counted and left out of the
ranking metrics. CIs: bootstrap over demos.

    python astar_vs_verifier.py [--tasks eggplant stack carrot spoon] [--source pi0_demos]
Out: $STEP2/expert_analysis/astar_vs_verifier[_<source>].{json,md,png}   (veritas venv)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
DP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [DP_ROOT, HERE]
from diffusion_policy.policy.veritas_verifier import VeritasPixelVerifier  # noqa: E402
from plot_n_sweep import STEP2  # noqa: E402
from precompute_proprio_replay import convert_maniskill  # noqa: E402

H = 4
DISTS = ("endpoint", "xyz_rmse", "norm7")
N_BOOT = 2000


def task_dir(task, source):
    return f"{STEP2}/{task}" if source == "openvla" else f"{STEP2}/{task}/{source}"


def load_task(task, source):
    """Per-frame arrays over all demos with candidates; `ep` indexes the demo of each frame."""
    tdir = task_dir(task, source)
    tj = json.load(open(f"{tdir}/plans_h4/veritas_targets.json"))
    targets_all = np.load(f"{tdir}/plans_h4/veritas_targets.npy")
    import zarr
    actions = np.asarray(zarr.open(tj["zarr_path"], "r")["data/actions"])
    ver = VeritasPixelVerifier(camera_file=tj["camera_file"], lookahead_k=tj["lookahead_k"],
                               lookahead_alpha=tj["lookahead_alpha"], default_tol_px=60.0,
                               horizon=tj["horizon"], normalize="none", plan_done_mode="zero",
                               health_check=False)
    out = {k: [] for k in ("ep", "cand", "astar", "grip_open", "cost", "cost_astar", "stored_cost",
                           "ep_cand", "ep_astar", "tcp_next", "tcp")}
    for f in sorted(glob.glob(f"{tdir}/candidates_ct/ep*.npz")):
        ep = int(os.path.basename(f)[2:6])
        c = np.load(f)
        rp = np.load(f"{tdir}/replay/ep{ep:04d}.npz")
        s, e = int(c["zarr_start"]), int(c["zarr_end"])
        T = e - s
        if T <= H:
            continue
        ts = np.arange(T - H + 1)                                    # A* window inside the demo
        astar = np.stack([[convert_maniskill(a) for a in actions[s + t:s + t + H]] for t in ts])
        cand = c["ctx_actions_env"][ts]                              # (F, 4, H, 7)
        tcp, tg = rp["tcp_world"][ts], targets_all[s:e][ts]
        F, N = cand.shape[:2]
        allc = np.concatenate([cand, astar[:, None]], 1).reshape(F * (N + 1), H, 7)
        obs = dict(end_effector_pose=torch.as_tensor(np.repeat(tcp, N + 1, 0)),
                   veritas_targets=torch.as_tensor(np.repeat(tg, N + 1, 0)))
        act = torch.as_tensor(allc, dtype=torch.float32)
        cost = ver.cost(obs, act).numpy().reshape(F, N + 1)
        endp = ver.endpoints(obs, act).numpy().reshape(F, N + 1, 3)
        out["ep"].append(np.full(F, ep))
        out["cand"].append(cand)
        out["astar"].append(astar)
        out["grip_open"].append(actions[s + ts, 6] > 0.5)
        out["cost"].append(cost[:, :N])
        out["cost_astar"].append(cost[:, N])
        out["stored_cost"].append(c["ctx_cost"][ts])
        out["ep_cand"].append(endp[:, :N])
        out["ep_astar"].append(endp[:, N])
        nxt = np.minimum(ts + H, T - 1)
        out["tcp_next"].append(np.where((ts + H < T)[:, None], rp["tcp_world"][nxt], np.nan))
        out["tcp"].append(tcp)
    return {k: np.concatenate(v) for k, v in out.items()}


def distances(d):
    diff = d["cand"] - d["astar"][:, None]                           # (F, 4, H, 7)
    sd = d["astar"].reshape(-1, 7).std(0) + 1e-8
    return dict(endpoint=np.linalg.norm(d["ep_cand"] - d["ep_astar"][:, None], axis=-1),
                xyz_rmse=np.sqrt((diff[..., :3] ** 2).mean((-1, -2))),
                norm7=np.sqrt(((diff / sd) ** 2).mean((-1, -2))))


def frame_metrics(d, dist):
    """Per-frame metrics on the frames whose four candidate costs are not all tied."""
    cost, ca = d["cost"], d["cost_astar"]
    informative = np.ptp(cost, axis=1) > 1e-6
    pick = cost.argmin(1)
    best = dist.argmin(1)
    r_cost = rankdata(cost, axis=1)
    r_dist = rankdata(dist, axis=1)
    rc, rd = r_cost - r_cost.mean(1, keepdims=True), r_dist - r_dist.mean(1, keepdims=True)
    rho = (rc * rd).sum(1) / np.sqrt((rc ** 2).sum(1) * (rd ** 2).sum(1) + 1e-12)
    rank_astar = rankdata(np.concatenate([cost, ca[:, None]], 1), axis=1)[:, -1]   # 1 = best, ties averaged
    F = len(cost)
    pick_d = dist[np.arange(F), pick]
    return dict(informative=informative,
                top1=(pick == best).astype(float),
                spearman=rho,
                astar_rank=rank_astar,
                astar_best=(ca < cost.min(1) - 1e-6).astype(float),
                astar_tied_best=(ca <= cost.min(1) + 1e-6).astype(float),
                regret_pick=pick_d - dist.min(1),
                regret_random=dist.mean(1) - dist.min(1))


def pooled_corr(d, dist, mask):
    """Within-frame demeaned Pearson of (-cost) vs distance: negative = agrees with A*."""
    v = -d["cost"][mask]
    x = dist[mask]
    v, x = v - v.mean(1, keepdims=True), x - x.mean(1, keepdims=True)
    return float((v * x).sum() / np.sqrt((v ** 2).sum() * (x ** 2).sum() + 1e-12))


def boot(values, ep, rng):
    """Frame-weighted mean and 95% CI by resampling demos."""
    eps = np.unique(ep)
    sums = np.array([values[ep == e].sum() for e in eps])
    cnts = np.array([(ep == e).sum() for e in eps])
    idx = rng.integers(0, len(eps), (N_BOOT, len(eps)))
    bs = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    return float(values.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def analyse(task, source, rng):
    d = load_task(task, source)
    dists = distances(d)
    gap = np.linalg.norm(d["ep_astar"] - d["tcp_next"], axis=-1)
    res = dict(n_demos=int(len(np.unique(d["ep"]))), n_frames=int(len(d["ep"])),
               stored_cost_max_abs_diff=float(np.abs(d["cost"] - d["stored_cost"]).max()),
               astar_endpoint_vs_tcp_next_median_m=float(np.nanmedian(gap)),
               candidate_endpoint_spread_m=float(np.median(np.ptp(d["ep_cand"], axis=1).max(-1))),
               by_dist={})
    for name in DISTS:
        m = frame_metrics(d, dists[name])
        inf = m["informative"]
        r = dict(frac_informative=float(inf.mean()),
                 pooled_corr_value_vs_dist=pooled_corr(d, dists[name], inf),
                 median_dist_closest=float(np.median(dists[name].min(1))),
                 median_dist_spread=float(np.median(np.ptp(dists[name], axis=1))))
        for k in ("top1", "spearman", "astar_rank", "astar_best", "astar_tied_best",
                  "regret_pick", "regret_random"):
            r[k] = boot(m[k][inf], d["ep"][inf], rng)
        for phase, pm in (("gripper_open", d["grip_open"]), ("gripper_closed", ~d["grip_open"])):
            sel = inf & pm
            r[phase] = dict(n=int(sel.sum()), top1=boot(m["top1"][sel], d["ep"][sel], rng),
                            astar_best=boot(m["astar_best"][sel], d["ep"][sel], rng)) if sel.sum() else None
        res["by_dist"][name] = r
    return res


def fmt(t, pct=False):
    s = 100 if pct else 1
    return f"{s * t[0]:.{1 if pct else 2}f} [{s * t[1]:.{1 if pct else 2}f}, {s * t[2]:.{1 if pct else 2}f}]"


def plot(all_res, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = list(all_res)
    series = [  # key, label, chance, colour, marker, hatch
        ("top1", "verifier pick = closest to A*", 0.25, "#0072B2", "o", ""),
        ("astar_best", "A* beats all 4 candidates", None, "#D55E00", "s", "//"),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.35
    for j, (k, lab, chance, col, mk, hatch) in enumerate(series):
        for i, t in enumerate(tasks):
            m, lo, hi = all_res[t]["by_dist"]["endpoint"][k]
            x = i + (j - 0.5) * w
            ax.bar(x, m, w, color=col, hatch=hatch, edgecolor="black", lw=0.6, label=lab if i == 0 else None)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="black", capsize=2, lw=0.8)
            ax.text(x, hi + 0.01, f"{100 * m:.0f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0.25, color="#0072B2", ls="--", lw=1, label="chance 1/4 (pick = closest)")
    ax.axhline(0.2, color="#D55E00", ls=":", lw=1.2, label="chance 1/5 (A* best of 5)")
    ax.set_xticks(range(len(tasks)), tasks)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of informative frames")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["eggplant", "stack", "carrot", "spoon"])
    ap.add_argument("--source", default="openvla", choices=["openvla", "pi0_demos", "pi0beta_demos"])
    ap.add_argument("--out", default=f"{STEP2}/expert_analysis")
    a = ap.parse_args()
    name = "astar_vs_verifier" + ("" if a.source == "openvla" else f"_{a.source}")
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    all_res = {}
    for t in a.tasks:
        if not glob.glob(f"{task_dir(t, a.source)}/candidates_ct/ep*.npz"):
            print(f"{t}: no candidates_ct yet, skipped")
            continue
        all_res[t] = analyse(t, a.source, rng)
        print(t, json.dumps({k: v for k, v in all_res[t].items() if k != "by_dist"}))
    md = ["Distance = endpoint (world xyz, m). Mean [95% CI over demos], informative frames only.", "",
          "| task | demos/frames | informative | top-1 (chance 25%) | Spearman | A* rank of 5 (chance 3) | "
          "A* best (strict) | A* tied-best | regret pick / random (mm) | pooled corr(value, dist) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in all_res.items():
        e = r["by_dist"]["endpoint"]
        md.append(f"| {t} | {r['n_demos']}/{r['n_frames']} | {100 * e['frac_informative']:.0f}% | "
                  f"{fmt(e['top1'], True)} | {fmt(e['spearman'])} | {fmt(e['astar_rank'])} | "
                  f"{fmt(e['astar_best'], True)} | {fmt(e['astar_tied_best'], True)} | "
                  f"{1e3 * e['regret_pick'][0]:.1f} / {1e3 * e['regret_random'][0]:.1f} | "
                  f"{e['pooled_corr_value_vs_dist']:+.3f} |")
    md += ["", "Other distances (top-1 %, Spearman, pooled corr):", "",
           "| task | xyz_rmse | norm7 |", "|---|---|---|"]
    for t, r in all_res.items():
        cells = [f"{fmt(r['by_dist'][k]['top1'], True)}, {r['by_dist'][k]['spearman'][0]:+.2f}, "
                 f"{r['by_dist'][k]['pooled_corr_value_vs_dist']:+.3f}" for k in ("xyz_rmse", "norm7")]
        md.append(f"| {t} | {cells[0]} | {cells[1]} |")
    md += ["", "By phase (endpoint distance; A* gripper open = approach, closed = carry):", "",
           "| task | open: n, top-1, A* best | closed: n, top-1, A* best |", "|---|---|---|"]
    for t, r in all_res.items():
        e = r["by_dist"]["endpoint"]
        cell = lambda p: "-" if not p else f"{p['n']}, {100 * p['top1'][0]:.0f}%, {100 * p['astar_best'][0]:.0f}%"
        md.append(f"| {t} | {cell(e['gripper_open'])} | {cell(e['gripper_closed'])} |")
    md += ["", "Checks: " + "; ".join(
        f"{t} stored-cost max diff {r['stored_cost_max_abs_diff']:.2g}, A* endpoint vs tcp[t+4] median "
        f"{1e3 * r['astar_endpoint_vs_tcp_next_median_m']:.1f} mm, candidate endpoint spread "
        f"{1e3 * r['candidate_endpoint_spread_m']:.1f} mm" for t, r in all_res.items())]
    open(f"{a.out}/{name}.md", "w").write("\n".join(md) + "\n")
    json.dump(all_res, open(f"{a.out}/{name}.json", "w"), indent=1)
    plot(all_res, f"{a.out}/{name}.png")
    print("\n".join(md))


if __name__ == "__main__":
    main()
