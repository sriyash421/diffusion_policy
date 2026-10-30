#!/usr/bin/env python
"""Does the context model improve its candidates with each generation?

On held-out decision frames (val collection seeds of the train tasks; image, proprio, TCP and
the plan targets are stored, so no env is needed) generate N candidates sequentially, chunk i
conditioned on the previous min(i, Kc) chunks and their verifier values (the ctx_sliding eval
mode), and record every candidate's value in generation order. Arms:

  base       frozen pi0, N i.i.d. samples (flat by construction: the reference)
  untrained  PiZeroContext at init (LoRA identity, random context encoders, bias 0)
  ft_iid     fine-tuned checkpoint with the context masked (n_ctx 0): the fine-tune alone
  ft_ctx     fine-tuned checkpoint, sliding context: the arm that should trend upward
  seq        the ctx_seq "second turn": 4 base chunks as context, then 4 conditioned draws from
             the checkpoint (and 4 unconditioned ones, same noise) -> does the conditioned
             chunk beat the base chunks it saw?

Writes <out>/probe.json; ``--plot <json>`` draws value vs generation index per task (mean with
95% CI, plus "P(chunk i beats the best of its context)").
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

DP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, DP_ROOT)

STEP2 = "/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2"
SHORT = {"eggplant": "widowx_put_eggplant_in_basket", "stack": "widowx_stack_cube",
         "carrot": "widowx_carrot_on_plate"}
ARMS = ["base", "untrained", "ft_iid", "ft_ctx"]


def load_frames(task, n_val_seeds, per_task, rng):
    """Decision frames of the last n_val_seeds collection seeds (never trained on)."""
    files = sorted(glob.glob(os.path.join(STEP2, task, "collect", "ep*_a1.npz")))
    seeds = sorted(set(int(os.path.basename(f)[2:8]) for f in files))[-n_val_seeds:]
    frames = []
    for f in files:
        if int(os.path.basename(f)[2:8]) not in seeds:
            continue
        d = np.load(f)
        for i in range(int(d["n_decisions"])):
            if d["targets"][i, 0] > 0:   # a plan with an active waypoint (else every cost is 0)
                frames.append(dict(image=d["image"][i], proprio=d["proprio"][i], tcp=d["tcp"][i],
                                   targets=d["targets"][i], tol=float(d["tol"][i]),
                                   stored_values=d["values"][i]))
    idx = rng.choice(len(frames), min(per_task, len(frames)), replace=False)
    return [frames[i] for i in sorted(idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{STEP2}/eggplant/train/A_argmax_xtask/ckpt_25000.pt")
    ap.add_argument("--out", default=f"{STEP2}/probe_feedback/A_argmax_xtask_25000")
    ap.add_argument("--tasks", default="eggplant,stack,carrot")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--per-task", type=int, default=40)
    ap.add_argument("--n-val-seeds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", default=None, help="probe.json to plot (no GPU)")
    args = ap.parse_args()
    if args.plot:
        return plot(args.plot)

    import torch
    from diffusion_policy.pi0_context.official import (DEFAULT_CKPT, clip_value, compose_eval_cfg, frame_inputs,
                                                       load_pizero, make_adapter, model_masks)
    from diffusion_policy.pi0_context.pizero_context import build_pizero_context
    from diffusion_policy.pi0_context.rollout_dataset import INSTRUCTIONS
    from diffusion_policy.policy.veritas_verifier import VeritasPixelVerifier
    from omegaconf import open_dict

    rng = np.random.default_rng(args.seed)
    cfg = compose_eval_cfg()
    adapter = make_adapter(cfg)
    base, _ = load_pizero(cfg, DEFAULT_CKPT)
    with open_dict(cfg):
        cfg.context_attn_bias_init = 0.0
    untrained = build_pizero_context(cfg, DEFAULT_CKPT, lora=True).eval()
    ft = build_pizero_context(cfg, DEFAULT_CKPT, lora=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    res = ft.load_state_dict(ck["trainable"], strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    ft.eval()
    Kc, H, A, N = ft.num_context_blocks, ft.horizon_steps, ft.action_dim, args.n
    print(f"ckpt step {ck.get('step')} bias {ft.ctx_attn_bias.item():.2f}", flush=True)

    def score(verifier, fr, raws):
        chunks = np.stack([adapter.postprocess(r) for r in raws])
        obs = dict(end_effector_pose=torch.as_tensor(np.repeat(fr["tcp"][None], len(raws), 0)),
                   veritas_targets=torch.as_tensor(np.repeat(fr["targets"][None], len(raws), 0)))
        cost = verifier.cost(obs, torch.as_tensor(chunks, dtype=torch.float32)).numpy()
        return clip_value(-cost / fr["tol"])

    def sliding(model, inp, verifier, fr, use_ctx):
        with torch.inference_mode():
            prefix = model.prefix_caches(inp["input_ids"], inp["pixel_values"], inp["attention_mask"], inp["proprios"])
        raws, vals = [], []
        for i in range(N):
            n_ctx = min(i, Kc) if use_ctx else 0
            ca = np.zeros((1, Kc, H, A), np.float32); cv = np.zeros((1, Kc), np.float32)
            if n_ctx:
                ca[0, :n_ctx] = np.stack(raws[i - n_ctx:i]); cv[0, :n_ctx] = vals[i - n_ctx:i]
            with torch.inference_mode():
                out = model.infer_action_with_context(
                    inp["input_ids"], inp["pixel_values"], inp["attention_mask"], inp["proprios"],
                    torch.as_tensor(ca, device="cuda"), torch.as_tensor(cv, device="cuda"),
                    torch.tensor([n_ctx], device="cuda"), prefix_caches=prefix)
            raw = out[0].float().cpu().numpy()
            raws.append(raw); vals.append(float(score(verifier, fr, [raw])[0]))
        return np.array(vals, np.float32)

    results = {}
    for short in args.tasks.split(","):
        task = SHORT[short]
        verifier = VeritasPixelVerifier(
            camera_file=os.path.join(DP_ROOT, "diffusion_policy/config/veritas", f"camera_{task}.json"),
            lookahead_k=3, lookahead_alpha=0.5, default_tol_px=60.0, horizon=4, normalize="none",
            plan_done_mode="zero", health_check=False)
        frames = load_frames(short, args.n_val_seeds, args.per_task, rng)
        print(f"{short}: {len(frames)} frames", flush=True)
        per = {a: [] for a in ARMS}
        stored, seq = [], []
        for k, fr in enumerate(frames):
            inp = frame_inputs(adapter, fr["image"][None], fr["proprio"][None], INSTRUCTIONS[task])
            masks = model_masks(base, inp["attention_mask"])
            rep = lambda t: t.expand(N, *t.shape[1:]).contiguous()
            with torch.inference_mode():
                out = base.infer_action(input_ids=rep(inp["input_ids"]), pixel_values=rep(inp["pixel_values"]),
                                        proprios=rep(inp["proprios"]), **{k2: rep(v) for k2, v in masks.items()})
            per["base"].append(score(verifier, fr, out.float().cpu().numpy()))
            per["untrained"].append(sliding(untrained, inp, verifier, fr, True))
            per["ft_iid"].append(sliding(ft, inp, verifier, fr, False))
            per["ft_ctx"].append(sliding(ft, inp, verifier, fr, True))
            # second turn: context = the first 4 base chunks, 4 conditioned / unconditioned draws
            base_raw = out[:4].float().cpu().numpy()
            ca = np.zeros((4, Kc, H, A), np.float32); cv = np.zeros((4, Kc), np.float32)
            ca[:, :4] = base_raw[None]; cv[:, :4] = per["base"][-1][:4][None]
            rep4 = lambda t: t.expand(4, *t.shape[1:]).contiguous()
            torch.manual_seed(1000 + k); noise = torch.randn(4, H, A, device="cuda")
            with torch.inference_mode():
                cond = ft.infer_action_with_context(rep4(inp["input_ids"]), rep4(inp["pixel_values"]), rep4(inp["attention_mask"]),
                                                    rep4(inp["proprios"]), torch.as_tensor(ca, device="cuda"),
                                                    torch.as_tensor(cv, device="cuda"), torch.full((4,), 4, device="cuda"), noise=noise)
                unc = ft.infer_action_with_context(rep4(inp["input_ids"]), rep4(inp["pixel_values"]), rep4(inp["attention_mask"]),
                                                   rep4(inp["proprios"]), noise=noise)
            seq.append(dict(base=per["base"][-1][:4].tolist(), cond=score(verifier, fr, cond.float().cpu().numpy()).tolist(),
                            unc=score(verifier, fr, unc.float().cpu().numpy()).tolist()))
            stored.append(clip_value(fr["stored_values"]).mean())
            if k % 10 == 0:
                print(f"  frame {k}: " + " ".join(f"{a} {np.mean(per[a][-1]):.2f}" for a in ARMS)
                      + f" stored {stored[-1]:.2f}", flush=True)
        results[short] = {a: np.stack(per[a]).tolist() for a in ARMS}
        results[short]["stored_mean"] = float(np.mean(stored))
        results[short]["seq"] = seq
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "probe.json")
    json.dump(dict(ckpt=args.ckpt, step=ck.get("step"), n=N, results=results), open(path, "w"))
    print(f"wrote {path}")
    summarize(results, N)


def summarize(results, N):
    for task, r in results.items():
        print(f"== {task}")
        for a in ARMS:
            v = np.array(r[a])                      # (frames, N)
            first, last = v[:, :4].mean(), v[:, -4:].mean()
            best_ctx = np.array([[v[f, max(0, i - 4):i].max() if i else -np.inf for i in range(N)] for f in range(len(v))])
            beats = (v[:, 1:] > best_ctx[:, 1:]).mean()
            print(f"  {a:10s} mean value first4 {first:6.2f} last4 {last:6.2f}  "
                  f"P(chunk beats best of its window) {beats:.2f}  best-of-N {v.max(1).mean():6.2f}")
        if r.get("seq"):
            b = np.array([x["base"] for x in r["seq"]]); c = np.array([x["cond"] for x in r["seq"]]); u = np.array([x["unc"] for x in r["seq"]])
            print(f"  second turn (4 base -> conditioned): base mean {b.mean():6.2f} best {b.max(1).mean():6.2f} | "
                  f"conditioned mean {c.mean():6.2f}  unconditioned (same noise) {u.mean():6.2f} | "
                  f"P(cond > best base) {(c > b.max(1, keepdims=True)).mean():.2f}  P(unc > best base) {(u > b.max(1, keepdims=True)).mean():.2f}  "
                  f"P(cond > unc) {(c > u).mean():.2f}")


def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.load(open(path))
    results, N = d["results"], d["n"]
    summarize(results, N)
    style = {"base": ("#000000", "o", "-"), "untrained": ("#56B4E9", "^", ":"),
             "ft_iid": ("#E69F00", "s", "--"), "ft_ctx": ("#009E73", "D", "-.")}
    tasks = list(results)
    fig, axes = plt.subplots(2, len(tasks), figsize=(4.2 * len(tasks), 6.5), squeeze=False)
    for j, task in enumerate(tasks):
        for a in ARMS:
            v = np.array(results[task][a])
            x = np.arange(1, N + 1)
            m, se = v.mean(0), v.std(0) / np.sqrt(len(v))
            col, mk, ls = style[a]
            axes[0, j].errorbar(x, m, yerr=1.96 * se, color=col, marker=mk, ls=ls, ms=4, capsize=2, label=a)
            best = np.maximum.accumulate(v, axis=1)
            axes[1, j].plot(x, best.mean(0), color=col, marker=mk, ls=ls, ms=4, label=a)
        axes[0, j].set_title(f"{task} ({len(v)} held-out frames)", fontsize=10)
        axes[0, j].set_xlabel("generation index i"); axes[1, j].set_xlabel("candidates so far")
        axes[0, j].grid(alpha=0.3); axes[1, j].grid(alpha=0.3)
    axes[0, 0].set_ylabel("verifier value of chunk i (−px err / tol)")
    axes[1, 0].set_ylabel("best value among the first i")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle(f"{os.path.basename(os.path.dirname(path))}: does chunk i improve on its context?", fontsize=10)
    fig.tight_layout()
    out = path.replace(".json", ".png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
