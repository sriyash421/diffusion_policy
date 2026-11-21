#!/usr/bin/env python
"""Success vs N for the submit_bon_sweep.sh arms: one panel per task, one curve per policy
(pi0 Bridge-uniform, Bridge-Beta), N=1 = raw. Paired on the same 50 seeds, so the table also
gives raw-only / verifier-only flips per arm. The official-prompt seeded sweep of uniform pi0
(<task>/eval/{raw_n1,veritas_n<N>}_seeded, same seeds) is drawn as the baseline, and a pooled
block (all 4 tasks, 200 episodes) closes the table.

    python plot_bon_sweep.py [--sweep bon_gemini_written]
Out: $STEP2/expert_analysis/<sweep>.{png,md,json}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_n_sweep import STEP2, wilson  # noqa: E402

TASKS = ["eggplant", "stack", "carrot", "spoon"]
NS = [1, 2, 4, 8, 16, 64]
POLICIES = {"pi0": ("pi0 (Bridge-uniform)", "#0072B2", "o", "-"), "beta": ("pi0 (Bridge-Beta)", "#D55E00", "s", "--"),
            "official": ("pi0 (Bridge-uniform), official prompt", "#000000", "^", ":")}


def arm_dir(sweep, t, p, n):
    if p == "official":
        return f"{STEP2}/{t}/eval/" + ("raw_n1_seeded" if n == 1 else f"veritas_n{n}_seeded")
    return f"{STEP2}/{t}/eval/{sweep}/" + (f"{p}_raw_n1" if n == 1 else f"{p}_veritas_n{n}")


def rows(path, seeds=None):
    f = f"{path}/episodes.jsonl"
    if not os.path.exists(f):
        return {}
    out = {}
    for l in open(f):
        if l.strip():
            r = json.loads(l)
            if seeds is None or r["seed"] in seeds:
                out.setdefault(r["seed"], bool(r["success"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="bon_gemini_written")
    a = ap.parse_args()
    res, md = {}, [f"Sweep `{a.sweep}`, 50 eval seeds per arm, paired. Flips are vs the policy's own raw N=1 arm. "
                   "The official-prompt rows are the earlier `*_seeded` arms of uniform pi0 on the same seeds (raw N=1 taken "
                   "from this sweep where that arm ran on other seeds).", "",
                   "| task | policy | N | success | 95% CI | n | raw-only / verifier-only flips |", "|---|---|---|---|---|---|---|"]
    pooled = {}   # (p, n) -> {(task, seed): success}
    for t in TASKS:
        seeds = set(json.load(open(f"{STEP2}/{t}/eval_seeds_50.json"))["test"])
        for p, (lab, *_) in POLICIES.items():
            base = rows(arm_dir(a.sweep, t, p, 1), seeds)
            for n in NS:
                r = rows(arm_dir(a.sweep, t, p, n), seeds)
                if not r and p == "official" and n == 1:   # raw pi0 does not use the prompt: the sweep's raw arm
                    r = base = rows(arm_dir(a.sweep, t, "pi0", 1), seeds)
                if not r:
                    continue
                pooled.setdefault((p, n), {}).update({(t, s): v for s, v in r.items()})
                k, m = sum(r.values()), len(r)
                lo, hi = wilson(k, m)
                both = [s for s in r if s in base]
                ro = sum(base[s] and not r[s] for s in both)
                vo = sum(r[s] and not base[s] for s in both)
                res.setdefault(t, {}).setdefault(p, {})[n] = dict(k=k, n=m, rate=k / m, ci=[lo, hi], raw_only=ro, ver_only=vo)
                md.append(f"| {t} | {lab} | {n} | {100 * k / m:.0f}% | [{100 * lo:.0f}, {100 * hi:.0f}] | {m} | "
                          + ("-" if n == 1 else f"{ro} / {vo}") + " |")
    for p, (lab, *_) in POLICIES.items():
        base = pooled.get((p, 1), {})
        for n in NS:
            r = pooled.get((p, n))
            if not r:
                continue
            k, m = sum(r.values()), len(r)
            lo, hi = wilson(k, m)
            both = [s for s in r if s in base]
            ro = sum(base[s] and not r[s] for s in both)
            vo = sum(r[s] and not base[s] for s in both)
            res.setdefault("pooled", {}).setdefault(p, {})[n] = dict(k=k, n=m, rate=k / m, ci=[lo, hi], raw_only=ro, ver_only=vo)
            md.append(f"| all 4 tasks | {lab} | {n} | {100 * k / m:.0f}% | [{100 * lo:.0f}, {100 * hi:.0f}] | {m} | "
                      + ("-" if n == 1 else f"{ro} / {vo}") + " |")
    out = f"{STEP2}/expert_analysis/{a.sweep}"
    json.dump(res, open(f"{out}.json", "w"), indent=1)
    open(f"{out}.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, t in zip(axes, TASKS):
        for p, (lab, col, mk, ls) in POLICIES.items():
            pts = sorted(res.get(t, {}).get(p, {}).items())
            if not pts:
                continue
            ns = [n for n, _ in pts]
            y = [v["rate"] for _, v in pts]
            ax.errorbar(ns, y, yerr=[[v["rate"] - v["ci"][0] for _, v in pts], [v["ci"][1] - v["rate"] for _, v in pts]],
                        color=col, marker=mk, ls=ls, capsize=2, label=lab if ax is axes[0] else None)
            ax.text(ns[-1], y[-1], " " + ("official" if p == "official" else lab.split("(")[1][:-1]), fontsize=7, va="center", color=col)
        ax.set_xscale("log", base=2)
        ax.set_xticks(NS, [str(n) for n in NS])
        ax.set_title(t)
        ax.set_xlabel("N (1 = raw)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("success (95% CI)")
    fig.suptitle(f"Veritas best-of-N, prompt {a.sweep.replace('bon_', '')}, 50 seeded episodes per point")
    fig.legend(loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(f"{out}.png", dpi=150)


if __name__ == "__main__":
    main()
