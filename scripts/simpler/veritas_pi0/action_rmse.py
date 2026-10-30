#!/usr/bin/env python
"""RMSE between the chunk a context fine-tune samples and the target chunk a* (the verifier's
pick among the 5 base candidates), per checkpoint, on success-only collected decisions.

Splits: ``train`` (train-task decisions the run trained on), ``val`` (the last 30 collection
seeds of each train task, never trained on), ``spoon`` (every successful spoon decision; spoon is
never trained on). Context is the run's own (``--ctx-source`` from its args.json): n_ctx 0 and 4.
One chunk per decision with fixed noise, RMSE over frames x 4 steps x 7 dims in the model's
normalised action space. Step 0 = the base model (LoRA B=0, context encoder at init, bias 0).

    python scripts/simpler/veritas_pi0/action_rmse.py --run S_prefix_sx --splits train,val
    python scripts/simpler/veritas_pi0/action_rmse.py --plot     # figure from the saved JSONs (no GPU)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

STEP2 = "/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2"
OUT = f"{STEP2}/action_rmse"
SHORT = {"eggplant": "widowx_put_eggplant_in_basket", "stack": "widowx_stack_cube",
         "carrot": "widowx_carrot_on_plate", "spoon": "widowx_spoon_on_towel"}
STEPS = [0, 5000, 10000, 15000, 20000, 25000]


def datasets(splits, ctx_source):
    from diffusion_policy.pi0_context.rollout_dataset import RolloutFrames
    kw = dict(success_only=True, n_ctx_blocks=4, context=ctx_source)
    train_dirs = {SHORT[s]: f"{STEP2}/{s}/collect" for s in ("eggplant", "stack", "carrot")}
    out = {}
    for sp in splits:
        if sp == "spoon":   # all collection seeds: n_val_seeds covers every seed
            out[sp] = RolloutFrames({SHORT["spoon"]: f"{STEP2}/spoon/collect"}, "val", n_val_seeds=10 ** 6, **kw)
        else:
            out[sp] = RolloutFrames(train_dirs, sp, n_val_seeds=30, **kw)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="S_prefix_sx")
    ap.add_argument("--splits", default="train,val,spoon")
    ap.add_argument("--steps", default=",".join(map(str, STEPS)))
    ap.add_argument("--frames", type=int, default=512, help="decisions per split (fixed random subset)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        return plot()

    import torch
    from omegaconf import open_dict
    from diffusion_policy.pi0_context.context_dataset import BatchEncoder
    from diffusion_policy.pi0_context.official import DEFAULT_CKPT, compose_eval_cfg, make_adapter
    from diffusion_policy.pi0_context.pizero_context import build_pizero_context

    run_dir = f"{STEP2}/eggplant/train/{args.run}"
    targs = json.load(open(f"{run_dir}/args.json"))
    assert targs["success_only"], "run was not trained on success-only data"
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{args.run}.json"
    res = json.load(open(path)) if os.path.exists(path) else {}

    cfg = compose_eval_cfg(overrides=[f"lora_r={targs['lora_r']}", "lora_dropout=0"])
    with open_dict(cfg):
        cfg.num_context_blocks = targs["n_ctx_blocks"]
        cfg.context_attn_bias_init = targs["bias_init"]
    adapter = make_adapter(cfg)
    model = build_pizero_context(cfg, DEFAULT_CKPT, lora=True).eval()
    data = datasets(args.splits.split(","), targs["ctx_source"])
    for sp, ds in data.items():
        print(f"{sp}: {ds.summary()}", flush=True)
        res.setdefault(sp, {})["n_decisions"] = len(ds)
    H, A = model.horizon_steps, model.action_dim

    # ascending from the fresh model: every checkpoint holds all trainable tensors
    for step in sorted(int(s) for s in args.steps.split(",")):
        if step:
            ck = torch.load(f"{run_dir}/ckpt_{step}.pt", map_location="cpu", weights_only=False)
            assert int(ck["step"]) == step
            # eval() merges LoRA into the base weights and skips the LoRA branch afterwards:
            # unmerge (train) before loading, merge (eval) after
            model.train()
            bad = model.load_state_dict(ck["trainable"], strict=False).unexpected_keys
            assert not bad, bad
        model.eval()
        for sp, ds in data.items():
            enc = BatchEncoder(adapter, ds.instructions, "cuda")
            idx_all = np.random.default_rng(0).permutation(len(ds))[:args.frames]
            for k in (0, 4):
                key = f"ctx{k}"
                if str(step) in res[sp].get(key, {}):
                    continue
                se, n = 0.0, 0
                for i in range(0, len(idx_all), args.batch):
                    idx = idx_all[i:i + args.batch]
                    b = enc(ds.batch(idx, np.full(len(idx), k)))
                    g = torch.Generator(device="cuda").manual_seed(int(idx[0]))
                    noise = torch.randn(len(idx), H, A, device="cuda", generator=g)
                    with torch.inference_mode():
                        pred = model.infer_action_with_context(
                            b["input_ids"], b["pixel_values"], b["attention_mask"], b["proprios"],
                            b["ctx_actions"].float(), b["ctx_values"].float(), b["n_ctx"], noise=noise)
                    se += float(((pred.float() - b["actions"].float()) ** 2).sum())
                    n += pred.numel()
                res[sp].setdefault(key, {})[str(step)] = float(np.sqrt(se / n))
                print(f"step {step:5d} {sp:5s} n_ctx {k}: rmse {res[sp][key][str(step)]:.4f}", flush=True)
                json.dump(res, open(path + ".tmp", "w"), indent=1)
                os.replace(path + ".tmp", path)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r for r in ("S_prefix_sx", "A_argmax_sx") if os.path.exists(f"{OUT}/{r}.json")]
    if not runs:
        print("no RMSE results yet")
        return
    panels = [("train", "train tasks, training decisions"), ("val", "train tasks, held-out seeds"),
              ("spoon", "spoon (unseen task)")]
    colors = {"S_prefix_sx": "#0072B2", "A_argmax_sx": "#D55E00"}
    markers = {"S_prefix_sx": "o", "A_argmax_sx": "s"}
    styles = {"ctx0": ("--", "no context"), "ctx4": ("-", "4 context blocks")}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    handles = {}
    for ax, (sp, title) in zip(axes, panels):
        for r in runs:
            d = json.load(open(f"{OUT}/{r}.json")).get(sp, {})
            for key, (ls, lab) in styles.items():
                pts = sorted((int(s), v) for s, v in d.get(key, {}).items())
                if pts:
                    h, = ax.plot([p[0] / 1000 for p in pts], [p[1] for p in pts], color=colors[r],
                                 marker=markers[r], linestyle=ls, lw=1.6, ms=5)
                    handles.setdefault(f"{r}, {lab}", h)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training step (k); 0 = base pi0")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("RMSE to target chunk (normalised actions)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    p = f"{OUT}/action_rmse.png"
    fig.savefig(p, dpi=150)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
