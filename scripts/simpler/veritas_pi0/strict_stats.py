#!/usr/bin/env python
"""Strict-filter pass rates of successful demos, per task and per rule, for OpenVLA vs pi0 demos.

Each task uses the rule set its OpenVLA strict dataset was built with, through the filters' own
metric functions (``scripts/simpler/data_collect/``):
  eggplant  3 stages: distance (net decrease, backtrack) -> gripper carry (<= 3 cm) ->
            first-try grasp (closes to contact, release/retry, contact gap, gripper drift)
  stack (xy distance), spoon, carrot (xyz)  ``filter_strict_success.keep_episode``
(Carrot's OpenVLA dataset was success-only, never strict-filtered; the spoon rules apply.)

Per source: n successes, % kept, the first rule failed (filter order), and the % passing each
rule on its own. Sanity: a strict dataset passes 100%; the OpenVLA stack and spoon raw
collections give the dataset counts (280/428, 510/796).

    python strict_stats.py            # -> $STEP2/expert_analysis/strict_stats.{md,json,png}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "data_collect"))
import filter_first_try_grasp as fg  # noqa: E402
import filter_strict_distance_success as fd  # noqa: E402
import filter_strict_success as fs  # noqa: E402
from expert_vs_pi0 import RAW  # noqa: E402
from plot_n_sweep import STEP2  # noqa: E402

DATA = "/gscratch/robotics/harine/data"
STRICT = {"eggplant": f"{DATA}/strict_distance_success_split", "stack": f"{DATA}/stack_cube_strict_split",
          "spoon": f"{DATA}/spoon_on_towel_strict", "carrot": f"{DATA}/carrot_on_plate_strict"}
DIST = {"eggplant": "xyz", "stack": "xy", "spoon": "xyz", "carrot": "xyz"}
ARGS = SimpleNamespace(backtrack_abs=0.012, backtrack_rel=0.06, end_over_d0_max=0.90,
                       max_closes_to_contact=1, max_reopen_runs=0, max_contact_gap=1,
                       max_dge_rise=0.015, gripper_carry_max=0.03)
RULES = {
    "eggplant": ["net_increase", "backtrack", "gripper_carry", "no_contact", "not_first_try",
                 "dropped"],
    "other": ["net_increase", "backtrack", "no_contact", "not_first_try", "release_retry",
              "dropped_contact", "gripper_drift", "post_grasp_increase"],
}
SOURCES = {  # label -> (zarr glob per task, colour, hatch)
    "OpenVLA": (None, "#E69F00", "//"),
    "pi0 (Bridge-uniform)": ("{task}/pi0_demos/demos.zarr", "#0072B2", ""),
    "pi0 (Bridge-Beta)": ("{task}/pi0beta_demos/demos.zarr", "#009E73", ".."),
}


def episodes(paths):
    """Yield per-episode arrays of every successful episode in the zarr stores."""
    import zarr
    for p in paths:
        z = zarr.open(p, "r")
        ends = z["meta/episode_ends"][:]
        o = z["data/obs"]
        rew = z["data/rewards"][:]
        ins, rec = o["insertive_asset_pose"][:, :3], o["receptive_asset_pose"][:, :3]
        eef = o["end_effector_pose"][:, :3]
        contact = np.asarray(o["binary_contact"][:]).reshape(len(rew), -1)[:, 0].astype(int)
        grip = np.asarray(o["last_gripper_action"][:]).reshape(len(rew), -1)[:, 0].astype(np.float64)
        for s, e in zip(np.r_[0, ends[:-1]], ends):
            if rew[s:e].max() > 0:
                yield (ins[s:e].astype(np.float64), rec[s:e].astype(np.float64),
                       eef[s:e].astype(np.float64), contact[s:e], grip[s:e])


def check_eggplant(ins, rec, eef, contact, grip):
    """{rule: passed} for the 3-stage eggplant chain (each rule evaluated on its own)."""
    d_eb = np.linalg.norm(ins - rec, axis=1)
    d_ge = np.linalg.norm(eef - ins, axis=1)
    ok = {}
    m = fd.distance_metrics(d_eb)
    tol = max(ARGS.backtrack_abs, ARGS.backtrack_rel * m["d0"])
    ok["net_increase"] = m["end_over_d0"] <= ARGS.end_over_d0_max
    ok["backtrack"] = m["backtrack"] <= tol
    t_grasp, t_place = int(np.argmin(d_ge)), int(np.argmin(d_eb))
    hi = max(t_place, t_grasp) + 1
    ok["gripper_carry"] = float(d_ge[t_grasp:hi].max()) <= ARGS.gripper_carry_max
    g = fg.grasp_metrics(grip, contact, d_ge, d_eb)
    ok["no_contact"] = g is not None
    ok["not_first_try"] = g is not None and (g["closes_to_contact"] <= ARGS.max_closes_to_contact
                                             and g["reopen_runs2"] <= ARGS.max_reopen_runs)
    ok["dropped"] = g is not None and (g["max_contact_gap"] <= ARGS.max_contact_gap
                                       and g["dge_rise"] <= ARGS.max_dge_rise)
    return ok


def check_other(ins, rec, eef, contact, grip, dist_mode):
    m, _, _ = fs.episode_metrics(ins, rec, eef, contact, grip, dist_mode)
    if m is None:
        return {r: False for r in RULES["other"]}
    tol = max(ARGS.backtrack_abs, ARGS.backtrack_rel * m["d0"])
    c = m.get("contacted", False)
    return {
        "net_increase": m["end_over_d0"] <= ARGS.end_over_d0_max,
        "backtrack": m["backtrack"] <= tol,
        "no_contact": c,
        "not_first_try": c and m["closes_to_contact"] <= ARGS.max_closes_to_contact,
        "release_retry": c and m["reopen_runs2"] <= ARGS.max_reopen_runs,
        "dropped_contact": c and m["max_contact_gap"] <= ARGS.max_contact_gap,
        "gripper_drift": c and m["dge_rise"] <= ARGS.max_dge_rise,
        "post_grasp_increase": c and m["post_grasp_backtrack"] <= tol,
    }


def stats(task, paths):
    """Summary dict plus the first failed rule (None = kept) of every successful episode."""
    rules = RULES["eggplant" if task == "eggplant" else "other"]
    rows = [check_eggplant(*ep) if task == "eggplant" else check_other(*ep, DIST[task])
            for ep in episodes(paths)]
    n = len(rows)
    if not n:
        return None, []
    fails = [next((k for k in rules if not r[k]), None) for r in rows]
    kept = fails.count(None)
    return dict(n=n, kept=kept, rate=kept / n, first_fail={k: fails.count(k) for k in rules},
                pass_alone={k: sum(r[k] for r in rows) / n for k in rules}), fails


def source_paths(label, task):
    if label == "OpenVLA":
        return [p for p, _ in RAW[task][0] if os.path.exists(p)]
    p = os.path.join(STEP2, SOURCES[label][0].format(task=task))
    return [p] if os.path.exists(p) else []


def plot(res, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_n_sweep import wilson
    tasks = list(res)
    labels = list(SOURCES)
    w = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for j, lab in enumerate(labels):
        col, hatch = SOURCES[lab][1:]
        for i, t in enumerate(tasks):
            r = res[t].get(lab)
            if not r:
                continue
            x = i + (j - (len(labels) - 1) / 2) * w
            lo, hi = wilson(r["kept"], r["n"])
            ax.bar(x, r["rate"], w, color=col, hatch=hatch, edgecolor="black", lw=0.6,
                   label=lab if lab not in ax.get_legend_handles_labels()[1] else None)
            ax.errorbar(x, r["rate"], yerr=[[r["rate"] - lo], [hi - r["rate"]]], color="black",
                        capsize=2, lw=0.8)
            ax.text(x, hi + 0.01, f"{100 * r['rate']:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(tasks)), tasks)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("successes passing the strict filter")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(labels), fontsize=8,
              frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{STEP2}/expert_analysis")
    ap.add_argument("--sanity", action="store_true", help="also run on the strict datasets (expect 100%%)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res, md = {}, []
    prev_f = f"{a.out}/strict_stats.json"
    prev = json.load(open(prev_f)) if os.path.exists(prev_f) else {}
    for task in ("eggplant", "stack", "carrot", "spoon"):
        res[task] = {}
        srcs = list(SOURCES) + (["strict dataset"] if a.sanity else [])
        for lab in srcs:
            paths = ([os.path.join(STRICT[task], f) for f in os.listdir(STRICT[task]) if f.endswith(".zarr")]
                     if lab == "strict dataset" else source_paths(lab, task))
            r, fails = stats(task, paths) if paths else (None, [])
            if r is None and lab == "OpenVLA":   # raw collections deleted: keep the last result
                r = prev.get(task, {}).get(lab)
            if r and lab in SOURCES and lab != "OpenVLA":   # recorded demos: every episode is a success
                json.dump(dict(kept=[i for i, f in enumerate(fails) if f is None],
                               first_fail={str(i): f for i, f in enumerate(fails) if f}),
                          open(os.path.join(os.path.dirname(paths[0]), "strict_kept.json"), "w"))
            if r:
                res[task][lab] = r
                print(f"{task:8s} {lab:22s} {r['kept']}/{r['n']} = {100 * r['rate']:.1f}%", flush=True)
    for task, d in res.items():
        rules = RULES["eggplant" if task == "eggplant" else "other"]
        md += [f"### {task}", "", "| source | successes | kept | " + " | ".join(rules) + " |",
               "|---|---|---|" + "---|" * len(rules)]
        for lab, r in d.items():
            md.append(f"| {lab} | {r['n']} | {r['kept']} ({100 * r['rate']:.1f}%) | " + " | ".join(
                f"{100 * r['pass_alone'][k]:.0f}% ({r['first_fail'][k]})" for k in rules) + " |")
        md.append("")
    md.append("Rule cells: % of successes passing that rule alone (episodes whose first failure it is).")
    open(f"{a.out}/strict_stats.md", "w").write("\n".join(md) + "\n")
    json.dump(res, open(f"{a.out}/strict_stats.json", "w"), indent=1)
    plot(res, f"{a.out}/strict_stats.png")
    print("\n".join(md))


if __name__ == "__main__":
    main()
