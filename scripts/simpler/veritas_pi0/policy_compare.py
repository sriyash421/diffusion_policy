#!/usr/bin/env python
"""OpenVLA vs pi0 Bridge-uniform vs pi0 Bridge-Beta as demo sources, per task:
raw success, strict-filter pass of the successes, and how the Veritas verifier ranks base-pi0
candidates against each source's A*. Reads the outputs of expert_vs_pi0.py --write-seeds,
record_pi0_demos.py, strict_stats.py and astar_vs_verifier.py [--source ...].

    python policy_compare.py      # -> $STEP2/expert_analysis/policy_compare.{md,json,png}
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_n_sweep import STEP2, wilson  # noqa: E402

OUT = f"{STEP2}/expert_analysis"
TASKS = ["eggplant", "stack", "carrot", "spoon"]
SOURCES = {  # label -> (demo dir, A* json suffix, colour, hatch)
    "OpenVLA": (None, "", "#E69F00", "//"),
    "pi0 (Bridge-uniform)": ("pi0_demos", "_pi0_demos", "#0072B2", ""),
    "pi0 (Bridge-Beta)": ("pi0beta_demos", "_pi0beta_demos", "#009E73", ".."),
}


def raw_success(task, demo_dir):
    if demo_dir is None:
        oc = json.load(open(f"{STEP2}/{task}/openvla_outcome.json"))
        return sum(oc.values()), len(oc)
    rows = [json.loads(l) for l in open(f"{STEP2}/{task}/{demo_dir}/episodes.jsonl") if l.strip()]
    return sum(bool(r["any_success"]) for r in rows), len(rows)


def collect():
    strict = json.load(open(f"{OUT}/strict_stats.json"))
    res = {}
    for t in TASKS:
        res[t] = {}
        for lab, (d, suf, _, _) in SOURCES.items():
            k, n = raw_success(t, d)
            r = dict(raw=dict(k=k, n=n, rate=k / n, ci=wilson(k, n)))
            st = strict.get(t, {}).get(lab)
            if st:
                r["strict"] = dict(k=st["kept"], n=st["n"], rate=st["rate"], ci=wilson(st["kept"], st["n"]))
            f = f"{OUT}/astar_vs_verifier{suf}.json"
            if os.path.exists(f) and t in json.load(open(f)):
                e = json.load(open(f))[t]["by_dist"]["endpoint"]
                r["astar"] = {k: e[k] for k in ("top1", "astar_best", "spearman", "regret_pick", "regret_random")}
                r["astar"]["carry_top1"] = e["gripper_closed"]["top1"] if e["gripper_closed"] else None
                r["astar"]["approach_top1"] = e["gripper_open"]["top1"] if e["gripper_open"] else None
            res[t][lab] = r
    return res


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("raw success rate", lambda r: r.get("raw") and (r["raw"]["rate"], *r["raw"]["ci"]), None),
              ("successes passing strict filter", lambda r: r.get("strict") and (r["strict"]["rate"], *r["strict"]["ci"]), None),
              ("verifier pick = closest to A*", lambda r: r.get("astar") and tuple(r["astar"]["top1"]), 0.25),
              ("A* beats all 4 candidates", lambda r: r.get("astar") and tuple(r["astar"]["astar_best"]), 0.2)]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    w = 0.8 / len(SOURCES)
    for ax, (title, get, chance) in zip(axes, panels):
        for j, (lab, (_, _, col, hatch)) in enumerate(SOURCES.items()):
            for i, t in enumerate(TASKS):
                v = get(res[t][lab])
                if not v:
                    continue
                m, lo, hi = v
                x = i + (j - (len(SOURCES) - 1) / 2) * w
                ax.bar(x, m, w, color=col, hatch=hatch, edgecolor="black", lw=0.6,
                       label=lab if (i == 0 and ax is axes[0]) else None)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="black", capsize=2, lw=0.8)
                ax.text(x, hi + 0.01, f"{100 * m:.0f}", ha="center", va="bottom", fontsize=6)
        if chance:
            ax.axhline(chance, color="black", ls="--", lw=1)
            title += f"\n(dashed line = chance {chance:.2f})"
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(TASKS)), TASKS, fontsize=8)
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("fraction (95% CI)")
    fig.legend(loc="lower center", ncol=len(SOURCES), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)


def main():
    res = collect()
    pct = lambda v: "-" if v is None else f"{100 * v:.0f}%"
    md = ["| task | source | raw success | strict pass | top-1 vs A* (chance 25%) | A* best of 5 (chance 20%) "
          "| top-1 approach / carry | regret pick / random (mm) |", "|---|---|---|---|---|---|---|---|"]
    for t in TASKS:
        for lab, r in res[t].items():
            a = r.get("astar")
            md.append(f"| {t} | {lab} | {r['raw']['k']}/{r['raw']['n']} = {pct(r['raw']['rate'])} | "
                      + (f"{r['strict']['k']}/{r['strict']['n']} = {pct(r['strict']['rate'])}" if "strict" in r else "-")
                      + " | " + (" | ".join([pct(a["top1"][0]), pct(a["astar_best"][0]),
                                             f"{pct(a['approach_top1'] and a['approach_top1'][0])} / "
                                             f"{pct(a['carry_top1'] and a['carry_top1'][0])}",
                                             f"{1e3 * a['regret_pick'][0]:.1f} / {1e3 * a['regret_random'][0]:.1f}"])
                                 if a else "- | - | - | -") + " |")
    md += ["", "Raw success: OpenVLA over its whole collection; pi0 over the episodes recorded until 200 "
           "successes. A* metrics on informative frames of up to 120 strict-passing demos; candidates are always "
           "base pi0 (Bridge-uniform)."]
    open(f"{OUT}/policy_compare.md", "w").write("\n".join(md) + "\n")
    json.dump(res, open(f"{OUT}/policy_compare.json", "w"), indent=1)
    plot(res, f"{OUT}/policy_compare.png")
    print("\n".join(md))


if __name__ == "__main__":
    main()
