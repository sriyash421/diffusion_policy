#!/usr/bin/env python
"""Expert (OpenVLA collection) vs pi0 success, per task.

OpenVLA: every episode of the raw collections (executed sample 0, no verifier), success =
max reward > 0 (``dones`` also marks truncation), seed = 100000 + START_INDEX[shard] + ep.
Eggplant's main raw collection was deleted; its separate 1000-episode ``eggplant_in_basket_val``
collection (same expert, same task, seeds 150000-150999) stands in.

pi0 (``$STEP2/<task>``): raw N=1 and + Veritas N=5 on the 50 eval seeds, attempt 1 of the
N=5 collection (seeds 200000-200299), and raw N=1 on 200 OpenVLA seeds (``expert_cmp/``), paired
with OpenVLA's outcome on the same seed (McNemar, exact).

    python expert_vs_pi0.py --write-seeds     # outcome table + openvla_seeds_200.json per task
    python expert_vs_pi0.py                   # table, json and figure under $STEP2/expert_analysis
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_n_sweep import STEP2, eval_seeds, rate, wilson  # noqa: E402

DATA = "/gscratch/robotics/harine/data"
# task -> raw shards (zarr path, START_INDEX), strict dataset episodes
RAW = {
    "eggplant": ([(f"{DATA}/eggplant_in_basket_val/val{i}.zarr", s)
                  for i, s in enumerate((50000, 50100, 50400, 50700))], None),
    "stack": ([(f"{DATA}/_archive/stack_cube/state{i}.zarr", s) for i, s in enumerate((0, 1000))], 280),
    "carrot": ([(f"{DATA}/carrot_on_plate/state{i}.zarr", s)
                for i, s in enumerate((0, 628, 2000, 3000))], None),
    "spoon": ([(f"{DATA}/_archive/spoon_on_towel/state{i}.zarr", s) for i, s in enumerate((0, 1000))], 510),
}
START_SEED = 100000
N_PAIRED = 200
ARMS = [  # (label, arm dirs tried in order, seeds key)
    ("pi0 raw", ["eval/raw_n1_eval50", "eval/raw_n1_seeded"], "eval50"),
    ("pi0 + Veritas N=5", ["eval/veritas_n5_eval50", "eval/veritas_n5_seeded"], "eval50"),
]
STYLE = {  # colour + hatch, plus a direct label on every bar
    "OpenVLA (collection)": ("#E69F00", "//"),
    "pi0 raw": ("#0072B2", ""),
    "pi0 + Veritas N=5": ("#009E73", ".."),
    "pi0 + Veritas collect (attempt 1)": ("#CC79A7", "xx"),
}


def openvla_outcomes(task):
    """{seed: (success, steps)} over every collected episode of the task's raw shards."""
    import zarr
    out = {}
    for path, start in RAW[task][0]:
        z = zarr.open(path, "r")
        ends = z["meta/episode_ends"][:]
        rew = z["data/rewards"][:]
        for i, (s, e) in enumerate(zip(np.r_[0, ends[:-1]], ends)):
            out[START_SEED + start + i] = (bool(rew[s:e].max() > 0), int(e - s))
    return out


def write_seeds():
    rng = np.random.default_rng(0)
    for task in RAW:
        oc = openvla_outcomes(task)
        with open(f"{STEP2}/{task}/openvla_outcome.json", "w") as f:
            json.dump({str(k): v[0] for k, v in sorted(oc.items())}, f)
        pick = sorted(int(s) for s in rng.choice(sorted(oc), N_PAIRED, replace=False))
        with open(f"{STEP2}/{task}/openvla_seeds_{N_PAIRED}.json", "w") as f:
            json.dump({"test": pick}, f)
        k = sum(v[0] for v in oc.values())
        print(f"{task}: OpenVLA {k}/{len(oc)} = {k / len(oc):.3f}; paired subset "
              f"{sum(oc[s][0] for s in pick)}/{N_PAIRED}")


def first_rows(path, key="seed"):
    rows = {}
    f = os.path.join(path, "episodes.jsonl")
    if os.path.exists(f):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                rows.setdefault(r[key], r)
    return rows


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def summarize(task):
    oc = json.load(open(f"{STEP2}/{task}/openvla_outcome.json"))
    k, n = sum(oc.values()), len(oc)
    res = {"OpenVLA (collection)": dict(k=k, n=n, rate=k / n, ci=wilson(k, n), seeds="raw collection")}
    if RAW[task][1]:
        res["OpenVLA (collection)"]["strict_pass"] = RAW[task][1] / n
    seeds50 = eval_seeds(task)
    for label, dirs, _ in ARMS:
        for d in dirs:
            r = rate(f"{STEP2}/{task}/{d}", seeds50)
            if r:
                p, lo, hi, m = r
                res[label] = dict(k=round(p * m), n=m, rate=p, ci=(lo, hi), seeds="eval 300000-300049", arm=d)
                break
    rows = first_rows(f"{STEP2}/{task}/collect_runs/collect_n5")
    if rows:
        k1 = sum(bool(r["attempt_successes"][0]) for r in rows.values())
        res["pi0 + Veritas collect (attempt 1)"] = dict(k=k1, n=len(rows), rate=k1 / len(rows),
                                                         ci=wilson(k1, len(rows)), seeds="collect 200000-200299")
    paired = None
    sf = f"{STEP2}/{task}/openvla_seeds_{N_PAIRED}.json"
    rows = first_rows(f"{STEP2}/{task}/expert_cmp/raw_n1_openvla{N_PAIRED}")
    if os.path.exists(sf) and rows:
        seeds = [s for s in json.load(open(sf))["test"] if s in rows]
        o = np.array([oc[str(s)] for s in seeds])
        p = np.array([bool(rows[s]["success"]) for s in seeds])
        b, c = int((o & ~p).sum()), int((~o & p).sum())
        paired = dict(n=len(seeds), openvla=int(o.sum()), pi0=int(p.sum()),
                      openvla_only=b, pi0_only=c, mcnemar_p=mcnemar_exact(b, c))
    return res, paired


def plot(all_res, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = list(all_res)
    labels = list(STYLE)
    w = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for j, lab in enumerate(labels):
        col, hatch = STYLE[lab]
        for i, t in enumerate(tasks):
            r = all_res[t]["arms"].get(lab)
            if not r:
                continue
            x = i + (j - (len(labels) - 1) / 2) * w
            ax.bar(x, r["rate"], w, color=col, hatch=hatch, edgecolor="black", linewidth=0.6,
                   label=lab if i == 0 else None)
            ax.errorbar(x, r["rate"], yerr=[[r["rate"] - r["ci"][0]], [r["ci"][1] - r["rate"]]],
                        color="black", capsize=2, lw=0.8)
            ax.text(x, r["ci"][1] + 0.01, f"{100 * r['rate']:.0f}", ha="center", va="bottom", fontsize=7)
    for i, t in enumerate(tasks):
        pr = all_res[t].get("paired")
        if pr:
            ax.text(i, -0.12, f"same {pr['n']} OpenVLA seeds:\nOpenVLA {pr['openvla']} vs pi0 raw {pr['pi0']}\n"
                    f"McNemar p={pr['mcnemar_p']:.1g}", ha="center", va="top", fontsize=7,
                    transform=ax.transData)
    ax.set_xticks(range(len(tasks)), tasks)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("success rate (Wilson 95%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=len(labels), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-seeds", action="store_true")
    ap.add_argument("--out", default=f"{STEP2}/expert_analysis")
    a = ap.parse_args()
    if a.write_seeds:
        write_seeds()
        return
    os.makedirs(a.out, exist_ok=True)
    all_res = {}
    md = ["| task | arm | seeds | success | 95% CI |", "|---|---|---|---|---|"]
    for task in RAW:
        res, paired = summarize(task)
        all_res[task] = dict(arms=res, paired=paired)
        for lab, r in res.items():
            md.append(f"| {task} | {lab} | {r['seeds']} | {r['k']}/{r['n']} = {100 * r['rate']:.1f}% | "
                      f"[{100 * r['ci'][0]:.1f}, {100 * r['ci'][1]:.1f}] |")
    md += ["", "Paired (raw pi0 N=1 on OpenVLA collection seeds):", "",
           "| task | n | OpenVLA | pi0 | OpenVLA only | pi0 only | McNemar p |", "|---|---|---|---|---|---|---|"]
    for task, v in all_res.items():
        p = v["paired"]
        if p:
            md.append(f"| {task} | {p['n']} | {p['openvla']} | {p['pi0']} | {p['openvla_only']} | "
                      f"{p['pi0_only']} | {p['mcnemar_p']:.3g} |")
    strict = [f"{t} {100 * v['arms']['OpenVLA (collection)']['strict_pass']:.1f}%"
              for t, v in all_res.items() if "strict_pass" in v["arms"]["OpenVLA (collection)"]]
    md += ["", "OpenVLA strict-filter pass rate (clean one-try successes / all episodes): " + ", ".join(strict)]
    open(f"{a.out}/expert_vs_pi0.md", "w").write("\n".join(md) + "\n")
    json.dump(all_res, open(f"{a.out}/expert_vs_pi0.json", "w"), indent=1)
    plot(all_res, f"{a.out}/expert_vs_pi0.png")
    print("\n".join(md))


if __name__ == "__main__":
    main()
