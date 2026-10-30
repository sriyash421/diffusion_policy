#!/usr/bin/env python
"""Success rate vs number of candidates N per task, one panel per task, from the 50-seed arms
under $STEP2/<task>/eval/<arm>/ (pi0 arms) and optional diffusion-policy arm directories.

    python scripts/simpler/veritas_pi0/plot_n_sweep.py --out n_sweep.png [--ckpt A_argmax_xtask_25000]
        [--dp-glob "/path/to/dp/<task>/veritas_bon_n{N}"]

Curves (colour + marker + line style, one legend under the panels):
  pi0 + verifier         veritas_n{N}_seeded (N=1: raw_n1_seeded)
  DP + verifier          --dp-glob pattern with {task} and {N} (episodes.jsonl inside)
  pi0 + context, untrained   ctx_sliding_n{N}_untrained
  fine-tuned + context       ctx_sliding_n{N}_<ckpt>  (one pair of curves per --ckpt)
  fine-tuned, no context     ckpt_bon_n{N}_<ckpt>
Wilson 95% intervals as error bars.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEP2 = "/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2"
TASKS = [("eggplant", "put eggplant in basket"), ("stack", "stack cube"),
         ("carrot", "carrot on plate"), ("spoon", "spoon on towel (unseen)")]
NS = [1, 2, 4, 8, 16, 32, 64]
STYLES = {   # colour, marker, line style (Okabe-Ito palette, never hue alone)
    "pi0 + verifier": ("#000000", "o", "-"),
    "DP + verifier": ("#E69F00", "s", "--"),
    "pi0 + context, untrained": ("#56B4E9", "^", ":"),
}
FT_STYLES = [("#009E73", "D", "-."), ("#CC79A7", "v", (0, (3, 1, 1, 1)))]
NOCTX_STYLES = [("#D55E00", "x", (0, (5, 2))), ("#0072B2", "+", (0, (1, 1)))]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def eval_seeds(task):
    with open(f"{STEP2}/{task}/eval_seeds_50.json") as f:
        return {s for v in json.load(f).values() for s in v}


def rate(path, seeds):
    """(success rate, lo, hi, n) over the rows of an arm dir's episodes.jsonl whose seed is in
    `seeds` (first row per seed), or None."""
    f = os.path.join(path, "episodes.jsonl")
    if not os.path.exists(f):
        return None
    rows = {}
    for l in open(f):
        if l.strip():
            r = json.loads(l)
            if r.get("seed") in seeds:
                rows.setdefault(r["seed"], r)
    if not rows:
        return None
    k = sum(bool(r["success"]) for r in rows.values())
    lo, hi = wilson(k, len(rows))
    return k / len(rows), lo, hi, len(rows)


def series(fmt, task, ns=NS):
    pts = []
    seeds = eval_seeds(task)
    for n in ns:
        cands = [fmt.format(task=task, N=n)]
        if n == 1 and "{N}" in fmt and "veritas_n" in fmt:
            for raw in ("raw_n1_eval50", "raw_n1_seeded"):
                cands.append(fmt.replace("veritas_n{N}_seeded", raw).format(task=task, N=n))
        for c in cands:
            m = sorted(glob.glob(c))
            r = rate(m[0], seeds) if m else None
            if r:
                pts.append((n, *r))
                break
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="n_sweep.png")
    ap.add_argument("--ckpt", action="append", default=[], help="<run>_<step> tag of a fine-tuned sweep")
    ap.add_argument("--dp-glob", default=None, help="DP arm dir pattern with {task} and {N}")
    ap.add_argument("--title", default="SimplerEnv WidowX, 50 seeded episodes per point")
    args = ap.parse_args()

    curves = {"pi0 + verifier": f"{STEP2}/{{task}}/eval/veritas_n{{N}}_seeded"}
    if args.dp_glob:
        curves["DP + verifier"] = args.dp_glob
    curves["pi0 + context, untrained"] = f"{STEP2}/{{task}}/eval/ctx_sliding_n{{N}}_untrained"
    styles = dict(STYLES)
    for i, ck in enumerate(args.ckpt):
        name = f"fine-tuned + context ({ck})"
        curves[name] = f"{STEP2}/{{task}}/eval/ctx_sliding_n{{N}}_{ck}"
        styles[name] = FT_STYLES[i % len(FT_STYLES)]
        name = f"fine-tuned, no context ({ck})"
        curves[name] = f"{STEP2}/{{task}}/eval/ckpt_bon_n{{N}}_{ck}"
        styles[name] = NOCTX_STYLES[i % len(NOCTX_STYLES)]

    handles = {}
    fig, axes = plt.subplots(1, len(TASKS), figsize=(4.2 * len(TASKS), 3.8), sharey=True)
    for ax, (task, title) in zip(axes, TASKS):
        for name, fmt in curves.items():
            pts = series(fmt, task)
            if not pts:
                continue
            col, mk, ls = styles[name]
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            err = [[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]]
            h = ax.errorbar(xs, ys, yerr=err, color=col, marker=mk, linestyle=ls, capsize=2, lw=1.4, ms=5, label=name)
            handles.setdefault(name, h)
        ax.set_xscale("log", base=2)
        ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS])
        ax.set_xlabel("candidates per decision N")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(0, 1)
    fig.suptitle(args.title, fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=min(len(handles), 4), fontsize=8,
               frameon=False)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    for task, _ in TASKS:
        for name, fmt in curves.items():
            pts = series(fmt, task)
            if pts:
                print(f"{task:9s} {name:40s} " + " ".join(f"N{p[0]}={p[1]:.2f}" for p in pts))


if __name__ == "__main__":
    main()
