#!/usr/bin/env python
"""Which way does the Veritas pick lean? On every informative decision of the strict-passing demos
(the frames astar_vs_verifier.py ranks), compare the verifier's pick with the mean of the 4 base-pi0
candidates on: gripper closed (fraction of the 4 steps), endpoint height change dz, endpoint
displacement, and the same three for A*. Decisions are split by the demo's phase:
  approach  A* gripper open, no close within the next 8 steps
  pre-grasp A* open, the demo closes within the next 8 steps (incl. this chunk)
  carry     A* closed, no open within the next 8 steps
  release   A* closed, the demo opens within the next 8 steps
Positive "pick - mean" on closed = the verifier prefers closing; on dz = prefers moving up.

    python verifier_bias.py [--sources openvla pi0_demos pi0beta_demos]   (veritas venv)
Out: $STEP2/expert_analysis/verifier_bias.{json,md}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from astar_vs_verifier import boot, load_task  # noqa: E402
from plot_n_sweep import STEP2  # noqa: E402

TASKS = ["eggplant", "stack", "carrot", "spoon"]
LOOK = 8
PHASES = ("approach", "pre-grasp", "carry", "release")


def phases(d):
    """Phase per frame from the demo's own gripper commands (A* chunk step 0 onwards)."""
    g_open = d["astar"][:, 0, 6] > 0                     # env format: +1 open, -1 closed
    ep = d["ep"]
    ph = np.empty(len(ep), object)
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        go = g_open[idx]
        for j, i in enumerate(idx):
            fut = go[j:j + LOOK]
            if go[j]:
                ph[i] = "pre-grasp" if (~fut).any() else "approach"
            else:
                ph[i] = "release" if fut.any() else "carry"
    return ph


def features(chunks, endp, tcp):
    """chunks (F, N, H, 7) env actions, endp (F, N, 3) -> dict of (F, N)."""
    disp = endp - tcp[:, None]
    return dict(closed=(chunks[..., 6] < 0).mean(-1), dz_mm=1e3 * disp[..., 2],
                disp_mm=1e3 * np.linalg.norm(disp, axis=-1))


def analyse(task, source, rng):
    d = load_task(task, source)
    inf = np.ptp(d["cost"], 1) > 1e-6
    ph = phases(d)
    fc = features(d["cand"], d["ep_cand"], d["tcp"])
    fa = features(d["astar"][:, None], d["ep_astar"][:, None], d["tcp"])
    pick = d["cost"].argmin(1)
    F = len(pick)
    res = {}
    for p in PHASES + ("all",):
        sel = inf & ((ph == p) if p != "all" else True)
        if sel.sum() < 10:
            continue
        r = dict(n=int(sel.sum()))
        for k in fc:
            delta = fc[k][np.arange(F), pick] - fc[k].mean(1)
            r[k] = dict(pick_minus_mean=boot(delta[sel], d["ep"][sel], rng),
                        mean=float(fc[k][sel].mean()), astar=float(fa[k][sel, 0].mean()))
        # decisions where candidates disagree on closing: does the pick close more or less often?
        dis = sel & (np.ptp(fc["closed"], 1) > 0)
        if dis.sum() >= 10:
            pc = fc["closed"][np.arange(F), pick] > 0
            r["close_disagree"] = dict(n=int(dis.sum()), pick_closes=float(pc[dis].mean()),
                                       cand_closes=float((fc["closed"][dis] > 0).mean()))
        res[p] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["openvla", "pi0_demos", "pi0beta_demos"])
    ap.add_argument("--out", default=f"{STEP2}/expert_analysis")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    res = {s: {t: analyse(t, s, rng) for t in TASKS} for s in a.sources}
    json.dump(res, open(f"{a.out}/verifier_bias.json", "w"), indent=1)
    md = ["Pick minus candidate mean [95% CI over demos]; cand mean / A* mean in brackets.", "",
          "| source | task | phase | n | gripper closed (frac) | dz (mm) | displacement (mm) "
          "| when candidates disagree on closing: pick closes / candidates close |",
          "|---|---|---|---|---|---|---|---|"]
    f = lambda r, k, s: (f"{s * r[k]['pick_minus_mean'][0]:+.2f} [{s * r[k]['pick_minus_mean'][1]:+.2f}, "
                         f"{s * r[k]['pick_minus_mean'][2]:+.2f}] ({s * r[k]['mean']:.2f} / {s * r[k]['astar']:.2f})")
    for s, rt in res.items():
        for t, rp in rt.items():
            for p, r in rp.items():
                cd = r.get("close_disagree")
                md.append(f"| {s} | {t} | {p} | {r['n']} | {f(r, 'closed', 1)} | {f(r, 'dz_mm', 1)} | "
                          f"{f(r, 'disp_mm', 1)} | "
                          + (f"{100 * cd['pick_closes']:.0f}% / {100 * cd['cand_closes']:.0f}% (n={cd['n']})"
                             if cd else "-") + " |")
    open(f"{a.out}/verifier_bias.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
