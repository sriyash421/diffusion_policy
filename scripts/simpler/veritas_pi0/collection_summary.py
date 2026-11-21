#!/usr/bin/env python
"""pi0 demo collections (record_pi0_demos.py): episodes run, successes, strict-filter pass, and
the start positions of the successful trajectories. Reads <task>/<source>/{episodes.jsonl,
demos.zarr, strict_kept.json} (strict_kept.json from strict_stats.py).

    python collection_summary.py      # -> $STEP2/expert_analysis/collection_summary.{md,json,png}
Plot: one panel per task x policy, source object start (x, y in the robot base frame, cm). Tasks
with fixed layouts get one marker per start position labelled strict/successes; eggplant (random
starts) gets every demo, circle = strict pass, x = strict fail.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_n_sweep import STEP2, wilson  # noqa: E402

TASKS = ["eggplant", "stack", "carrot", "spoon"]
SOURCES = {"pi0_demos": "pi0 (Bridge-uniform)", "pi0beta_demos": "pi0 (Bridge-Beta)"}


def summary(task, src):
    d = f"{STEP2}/{task}/{src}"
    rows = [json.loads(l) for l in open(f"{d}/episodes.jsonl") if l.strip()]
    kept = set(json.load(open(f"{d}/strict_kept.json"))["kept"])
    import zarr
    z = zarr.open(f"{d}/demos.zarr", "r")
    ends = z["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    src_xy = z["data/obs/insertive_asset_pose"][:][starts, :2]
    tgt_xy = z["data/obs/receptive_asset_pose"][:][starts, :2]
    n_succ = sum(bool(r["any_success"]) for r in rows)
    return dict(run=len(rows), success=n_succ, success_rate=n_succ / len(rows), success_ci=wilson(n_succ, len(rows)),
                demos=len(ends), strict=len(kept), strict_rate=len(kept) / len(ends),
                strict_of_run=len(kept) / len(rows) * (len(ends) / n_succ),   # strict pass per episode run
                steps_median=float(np.median(ends - starts)),
                src_xy=src_xy.tolist(), tgt_xy=tgt_xy.tolist(), strict_pass=[i in kept for i in range(len(ends))])


def main():
    res = {t: {s: summary(t, s) for s in SOURCES} for t in TASKS}
    out = f"{STEP2}/expert_analysis/collection_summary"
    json.dump(res, open(f"{out}.json", "w"))
    md = ["Collection stopped at 200 successes per task and policy (seeds 400000+ / 410000+). Every success "
          "was recorded; strict = passes the task's strict filter (strict_stats.py).", "",
          "| task | policy | episodes run | successes | success rate | strict pass of successes | "
          "strict successes per episode run | median demo length |", "|---|---|---|---|---|---|---|---|"]
    for t in TASKS:
        for s, lab in SOURCES.items():
            r = res[t][s]
            md.append(f"| {t} | {lab} | {r['run']} | {r['success']} | {100 * r['success_rate']:.0f}% "
                      f"[{100 * r['success_ci'][0]:.0f}, {100 * r['success_ci'][1]:.0f}] | {r['strict']}/{r['demos']} = "
                      f"{100 * r['strict_rate']:.0f}% | {100 * r['strict_of_run']:.0f}% | {r['steps_median']:.0f} |")
    open(f"{out}.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for i, (s, lab) in enumerate(SOURCES.items()):
        for j, t in enumerate(TASKS):
            ax, r = axes[i, j], res[t][s]
            sx, tx = 100 * np.array(r["src_xy"]), 100 * np.array(r["tgt_xy"])
            ok = np.array(r["strict_pass"])
            key = np.round(sx)   # the tasks' fixed layouts, 1 cm bins
            uniq = np.unique(key, axis=0)
            if len(uniq) <= 40:   # one annotated marker per start position
                for q, u in enumerate(uniq):
                    m = (key == u).all(1)
                    k, n = int(ok[m].sum()), int(m.sum())
                    ax.scatter(*u, s=12 * n, color="#0072B2", alpha=0.35, edgecolors="black")
                    near = np.where((np.abs(uniq - u).max(1) <= 2) & (np.arange(len(uniq)) != q))[0]
                    dy = 0 if not len(near) else (8 if q < near[0] else -8)   # neighbours within 2 cm: stagger labels
                    ax.annotate(f"{k}/{n}", u, (0, dy), textcoords="offset points", ha="center", va="center", fontsize=7)
                ax.set_title(f"{t} | {lab}\n{r['success']}/{r['run']} succeeded, {r['strict']}/{r['demos']} strict", fontsize=8)
            else:   # continuous starts: every demo
                ax.scatter(sx[ok, 0], sx[ok, 1], s=18, color="#0072B2", marker="o", alpha=0.6, label=f"strict pass ({ok.sum()})")
                ax.scatter(sx[~ok, 0], sx[~ok, 1], s=36, color="#D55E00", marker="x", label=f"strict fail ({(~ok).sum()})")
                ax.legend(fontsize=7, loc="lower left")
                ax.set_title(f"{t} | {lab}\n{r['success']}/{r['run']} succeeded, {r['strict']}/{r['demos']} strict; "
                             f"target at ({tx[0, 0]:.0f}, {tx[0, 1]:.0f})", fontsize=8)
            lo, hi = sx.min(0) - 2, sx.max(0) + 2
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("source object start x (cm, base frame)")
            if j == 0:
                ax.set_ylabel("y (cm)")
    fig.suptitle("Start positions of the recorded successful demos (200 per panel); label = strict-pass / successes at that start")
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=150)


if __name__ == "__main__":
    main()
