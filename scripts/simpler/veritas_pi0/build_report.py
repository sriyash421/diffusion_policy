#!/usr/bin/env python
"""Build or refresh the W&B report of the Veritas-on-pi0 campaign: plots only, four sections
(Veritas reproduction; success on train and test tasks with checkpoint choice; action RMSE of the
context fine-tune; expert and verifier checks). Every plot is redrawn from the result files, except
section 4's, which are the PNGs written by expert_vs_pi0.py and astar_vs_verifier.py; rerunning
updates the report in place (its URL is kept in $STEP2/report/report_url.txt).

    <vae_pushT_l2s env>/python scripts/simpler/veritas_pi0/build_report.py
    python scripts/simpler/veritas_pi0/build_report.py --select    # print best-train / best-test steps
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_n_sweep import STEP2, eval_seeds, rate, wilson  # noqa: E402

ENTITY, PROJECT = "l2sml", "simpler_veritas"
STEP1 = "/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_20261027"
OUT = f"{STEP2}/report"
RUN = "S_prefix_sx"
RUNS = ["S_prefix_sx", "A_argmax_sx"]
STEPS = [5000, 10000, 15000, 20000, 25000]
TRAIN = ["eggplant", "stack", "carrot"]
PAPER = {"eggplant": (81.7, 89.6), "stack": (31.3, 46.7), "carrot": (61.3, 71.7), "spoon": (63.8, 80.4)}


def forge(url):
    return url.replace("https://wandb.ai/", "https://forge.coreweave.com/wandb/")


def ks(task, arm):
    r = rate(f"{STEP2}/{task}/eval/{arm}", eval_seeds(task))
    return (round(r[0] * r[3]), r[3]) if r else None


def pooled(tasks, arm):
    """(successes, episodes) summed over tasks, None unless every task has all 50 episodes."""
    parts = [ks(t, arm) for t in tasks]
    if any(p is None or p[1] < 50 for p in parts):
        return None
    return sum(p[0] for p in parts), sum(p[1] for p in parts)


def select(run=RUN):
    """Best step on the train tasks and on spoon by context successes (ties -> earlier step)."""
    out = {}
    for name, tasks in (("best_train", TRAIN), ("best_test", ["spoon"])):
        scored = [(pooled(tasks, f"ctx_seq_n5_{run}_{s}"), s) for s in STEPS]
        scored = [(p[0], s) for p, s in scored if p]
        if scored:
            out[name] = max(scored, key=lambda x: (x[0], -x[1]))[1]
    return out


def plot_reproduction(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = ["eggplant", "stack", "carrot", "spoon"]
    ours = {}
    for t in ("eggplant", "stack"):     # official 240-episode protocol
        ours[t] = []
        for a in ("raw_n1", "veritas_n5"):
            rows = [json.loads(l) for l in open(f"{STEP1}/{t}/{a}/episodes.jsonl") if l.strip()]
            ours[t].append((sum(bool(r["success"]) for r in rows), len(rows)))
    for t in ("carrot", "spoon"):       # 50 seeded episodes
        ours[t] = [ks(t, "raw_n1_seeded"), ks(t, "veritas_n5_seeded")]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(tasks))
    w = 0.2
    bars = [("ours: pi0", 0, "white", ""), ("ours: pi0 + Veritas", 1, "#0072B2", ""),
            ("paper: pi0", 0, "white", "///"), ("paper: pi0 + Veritas", 1, "#E69F00", "///")]
    for j, (lab, i, col, hatch) in enumerate(bars):
        if lab.startswith("ours"):
            vals = [ours[t][i][0] / ours[t][i][1] for t in tasks]
            ci = [wilson(*ours[t][i]) for t in tasks]
            err = [[v - c[0] for v, c in zip(vals, ci)], [c[1] - v for v, c in zip(vals, ci)]]
        else:
            vals = [PAPER[t][i] / 100 for t in tasks]
            err = None
        ax.bar(x + (j - 1.5) * w, vals, w, color=col, edgecolor="black", hatch=hatch, label=lab,
               yerr=err, capsize=2)
    labels = [f"{t}\n({'240 episodes' if t in ('eggplant', 'stack') else '50 seeded'})" for t in tasks]
    ax.set_xticks(x, labels)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    ax.set_title("Veritas on open-pi0 (Bridge-uniform), N = 5 candidates", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_checkpoints(path, sel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"S_prefix_sx": "#0072B2", "A_argmax_sx": "#D55E00"}
    markers = {"S_prefix_sx": "o", "A_argmax_sx": "s"}
    names = {"S_prefix_sx": "context = first 4 candidates", "A_argmax_sx": "context = 4 non-picked candidates"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    handles = {}
    for ax, (title, tasks, key) in zip(axes, [("train tasks (eggplant + stack + carrot)", TRAIN, "best_train"),
                                               ("test task: spoon (never trained on)", ["spoon"], "best_test")]):
        for run in RUNS:
            for arm, ls, lab in (("ctx_seq", "-", "with context"), ("mix_bon", "--", "context masked")):
                pts = [(s, pooled(tasks, f"{arm}_n5_{run}_{s}")) for s in STEPS]
                pts = [(s, p) for s, p in pts if p]
                if not pts:
                    continue
                xs = [s / 1000 for s, _ in pts]
                ys = [p[0] / p[1] for _, p in pts]
                ci = [wilson(*p) for _, p in pts]
                h = ax.errorbar(xs, ys, yerr=[[y - c[0] for y, c in zip(ys, ci)], [c[1] - y for y, c in zip(ys, ci)]],
                                color=colors[run], marker=markers[run], linestyle=ls, capsize=2, lw=1.5, ms=5,
                                mfc=colors[run] if ls == "-" else "white")
                handles.setdefault(f"{names[run]}, {lab}", h)
        if key in sel:
            ax.axvline(sel[key] / 1000, color="black", linestyle=":", lw=1.2)
            ax.annotate(f"chosen, first-4 context ({key.replace('_', ' ')})", (sel[key] / 1000, 0.97), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, va="top")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training step (k)")
        ax.set_xticks([s / 1000 for s in STEPS])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("success rate, N = 5")
    axes[0].set_ylim(0, 1)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(sel):
    figs = {}
    p = f"{OUT}/reproduction.png"
    plot_reproduction(p)
    figs["repro/bars"] = p
    p = f"{OUT}/checkpoints.png"
    plot_checkpoints(p, sel)
    figs["ckpt/success_vs_step"] = p
    py = "/gscratch/robotics/harine/miniconda3/envs/dqc_jax/bin/python"
    sweep = [py, f"{HERE}/plot_n_sweep.py", "--dp-glob", f"{STEP2}/{{task}}/dp_eval/*/n{{N}}"]
    for key, step in sel.items():
        if key == "best_test" and step == sel.get("best_train"):
            continue
        p = f"{OUT}/n_sweep_{RUN}_{step}.png"
        subprocess.run(sweep + ["--ckpt", f"{RUN}_{step}", "--out", p, "--title",
                                f"{RUN} at {step // 1000}k, 50 seeded episodes per point"],
                       check=True, stdout=subprocess.DEVNULL)
        figs[f"sweep/{key}"] = p
    subprocess.run([py, f"{HERE}/action_rmse.py", "--plot"], check=True, stdout=subprocess.DEVNULL)
    p = f"{STEP2}/action_rmse/action_rmse.png"
    if os.path.exists(p):
        figs["rmse/curves"] = p
    for key, name in (("expert/success", "expert_vs_pi0"), ("expert/astar", "astar_vs_verifier")):
        p = f"{STEP2}/expert_analysis/{name}.png"
        if os.path.exists(p):
            figs[key] = p
    return figs


def media_run(figs):
    import wandb
    state_f = f"{OUT}/media_run.json"
    state = json.load(open(state_f)) if os.path.exists(state_f) else {}
    run = wandb.init(entity=ENTITY, project=PROJECT, name="step2-report-media", id=state.get("id"),
                     resume="allow", dir=OUT, job_type="report", group="report")
    run.log({k: wandb.Image(p) for k, p in figs.items()})
    rid = run.id
    run.finish()
    json.dump({"id": rid}, open(state_f, "w"))
    return rid


def blocks(rid, figs, sel):
    import wandb_workspaces.reports.v2 as wr

    def media(key, title, h=9):
        return wr.PanelGrid(hide_run_sets=True,
                            runsets=[wr.Runset(entity=ENTITY, project=PROJECT, filters=f'ID == "{rid}"')],
                            panels=[wr.MediaBrowser(title=title, media_keys=[key], num_columns=1, mode="gallery",
                                                    gallery_axis="run", layout=wr.Layout(x=0, w=24, h=h))])

    M = wr.MarkdownBlock
    out = [
        M(text="Open-pi0 (Bridge-uniform) in SimplerEnv WidowX. Fine-tunes train on eggplant, stack and carrot "
          "and are tested on spoon. Error bars are 95% Wilson intervals."),
        M(text="# 1 · Veritas reproduction"),
        M(text="Official Veritas code; Gemini plans; the verifier picks one of 5 pi0 candidates per decision. "
          "Hatched bars are the paper's numbers."),
        media("repro/bars", "pi0 vs pi0 + Veritas", h=8),
        M(text="# 2 · Success rates on train and test tasks"),
        M(text="**Fine-tune:** pi0 LoRA, given up to 4 earlier candidate chunks with their verifier feedback. "
          "**With context:** 4 base candidates, then 1 fine-tuned candidate conditioned on them; the verifier "
          "picks among the 5. **Context masked:** the same with the context hidden."),
        M(text="## Checkpoint choice\nBest train checkpoint: most successes with context summed over the "
          "train tasks. Best test checkpoint: most successes with context on spoon."),
        media("ckpt/success_vs_step", "success vs training step", h=9),
    ]
    for key, title in (("best_train", "best train checkpoint"), ("best_test", "best test checkpoint")):
        if f"sweep/{key}" in figs:
            out += [M(text=f"## N sweep · {title} ({RUN}, {sel[key] // 1000}k)\nFine-tuned + context: all N "
                      "candidates generated in sequence, each seeing the previous 4 with feedback. Fine-tuned, no "
                      "context: N independent candidates. DP: diffusion policy trained on each task's demos "
                      "(none exist for spoon)."),
                    media(f"sweep/{key}", title, h=8)]
        elif key == "best_test" and "best_test" in sel:
            out.append(M(text=f"## N sweep · {title}\nSame checkpoint as the best train checkpoint."))
    out += [
        M(text="## Training left\nBoth fine-tunes finished their full schedule (25k steps); nothing is left to "
          "train. Remaining work is evaluation only."),
        M(text="# 3 · Action RMSE of the context fine-tune\n**Training data (confirmed from the run configs: "
          "success_only = true):** decisions from successful pi0 + Veritas attempts only, collected on eggplant, "
          "stack and carrot. Target = the verifier's pick among 5 candidates. The last 30 collection seeds of "
          "each task are held out. Spoon decisions are collected the same way and never trained on.\n\n"
          "RMSE between one sampled chunk and the target chunk, fixed noise, normalised action space. Dashed: no "
          "context. Solid: the 4 context blocks the run was trained with."),
    ]
    if "rmse/curves" in figs:
        out.append(media("rmse/curves", "RMSE vs training step", h=8))
    else:
        out.append(M(text="*RMSE curves pending.*"))
    if "expert/success" in figs:
        out += [
            M(text="# 4 · Expert and verifier checks"),
            M(text="## The expert vs pi0\nThe collected datasets come from OpenVLA (one sample executed, no "
              "verifier). Its rate is over every collected episode (eggplant: its separate validation "
              "collection, since the main raw collection was deleted). The pi0 arms use the 50 eval seeds and "
              "the 300 collection seeds. Below each task: raw pi0 rerun on 200 of OpenVLA's own seeds, paired "
              "with OpenVLA's outcome on each seed."),
            media("expert/success", "OpenVLA vs pi0 success", h=9),
        ]
    if "expert/astar" in figs:
        out += [
            M(text="## Does the verifier rank like the expert?\nOn each demo frame, A* = the expert's next 4 "
              "actions. Blue: the verifier's top pick among pi0's 4 candidates is the one whose endpoint is "
              "closest to A*'s. Orange: the verifier scores A* above all 4 candidates. Frames where all 4 "
              "candidates tie are excluded. Dashed and dotted lines are chance."),
            media("expert/astar", "verifier agreement with A*", h=8),
        ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true", help="print the chosen steps as JSON and exit")
    args = ap.parse_args()
    sel = select()
    if args.select:
        print(json.dumps(sel))
        return
    import wandb_workspaces.reports.v2 as wr
    os.makedirs(OUT, exist_ok=True)
    figs = make_figures(sel)
    rid = media_run(figs)
    url_f = f"{OUT}/report_url.txt"
    report = wr.Report.from_url(open(url_f).read().strip()) if os.path.exists(url_f) else \
        wr.Report(entity=ENTITY, project=PROJECT)
    report.title = "Veritas on pi0 in SimplerEnv"
    report.description = ("Reproduction; success on train and test tasks; action RMSE of the context fine-tune; "
                          "expert and verifier checks.")
    report.width = "fixed"
    report.blocks = blocks(rid, figs, sel)
    report.save()
    open(url_f, "w").write(report.url)
    print(forge(report.url))


if __name__ == "__main__":
    main()
