#!/usr/bin/env python
"""Fine-tune PiZeroContext (LoRA on the action expert + context encoder) on the step-2 frames.

Loss: the official flow-matching loss on the query chunk, with ``--ctx-mode`` deciding how
many best-first context blocks each sample sees (``uniform``: 0..Kc per sample, the context
arm; ``none``: always 0, the masked-context control; ``full``: always Kc). Flow time is
sampled uniformly like the bridge_uniform checkpoint was trained. VLM frozen, bf16 autocast,
AdamW, warmup + cosine. Checkpoints hold only the trainable parameters (~55 MB) and the
optimizer; ``latest.pt`` in --out resumes (ckpt partition safe). Validation every
--val-every steps on the val split at n_ctx 0 and Kc with fixed t and noise, logged to wandb
(l2sml/simpler_veritas) with per-n_ctx train losses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

DP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if DP_ROOT not in sys.path:
    sys.path.insert(0, DP_ROOT)

from diffusion_policy.pi0_context.context_dataset import BatchEncoder, ContextFrames  # noqa: E402
from diffusion_policy.pi0_context.official import compose_eval_cfg, make_adapter  # noqa: E402
from diffusion_policy.pi0_context.pizero_context import build_pizero_context  # noqa: E402

STEP2 = "/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2/eggplant"


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="run name, also the subdir of --out-root")
    ap.add_argument("--ctx-mode", choices=["uniform", "none", "full"], default="uniform")
    ap.add_argument("--dataset", choices=["demos", "rollouts"], default="demos",
                    help="demos: expert-chunk targets (ContextFrames); rollouts: verifier-argmax targets "
                         "from collected pi0+verifier decisions (RolloutFrames)")
    ap.add_argument("--tasks", default="eggplant", help="rollouts: comma list of task short names")
    ap.add_argument("--collect-root", default="/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2")
    ap.add_argument("--success-only", action="store_true")
    ap.add_argument("--ctx-source", choices=["others", "prefix"], default="others",
                    help="rollouts: context = the non-chosen candidates (A) or the generation-order "
                         "prefix of all candidates, chosen one included (S, matches the sliding eval)")
    ap.add_argument("--multi-demos", action="store_true",
                    help="demos: use the per-task layout under --collect-root even for a single task")
    ap.add_argument("--n-val-seeds", type=int, default=30)
    ap.add_argument("--split-file", default="/gscratch/robotics/harine/RoboMonkey_data/experiments/position_holdout_20261009/splits/eggplant/position_holdout_train100.json")
    ap.add_argument("--replay-dir", default=f"{STEP2}/replay")
    ap.add_argument("--candidates-dir", default=f"{STEP2}/candidates")
    ap.add_argument("--out-root", default=f"{STEP2}/train")
    ap.add_argument("--checkpoint", default="models/bridge_uniform_step19296_2024-12-26_22-31_42.pt")
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bias-lr", type=float, default=1e-2,
                    help="lr of the scalar context-attention bias (init -8; Adam moves it ~lr/step)")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--n-ctx-blocks", type=int, default=4)
    ap.add_argument("--bias-init", type=float, default=0.0,
                    help="context-attention bias init (-8 gated the context shut in the first runs)")
    ap.add_argument("--max-tcp-err-mm", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--wandb-mode", default="online")
    ap.add_argument("--wandb-project", default="simpler_veritas")
    ap.add_argument("--wandb-group", default="step2_eggplant")
    return ap.parse_args()


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    p = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.lr * (args.lr_min_ratio + (1 - args.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def trainable_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def main():
    args = parse()
    out = os.path.join(args.out_root, args.arm)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda")

    cfg = compose_eval_cfg(overrides=[f"lora_r={args.lora_r}", f"lora_dropout={args.lora_dropout}"])
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.num_context_blocks = args.n_ctx_blocks
        cfg.context_attn_bias_init = args.bias_init
    adapter = make_adapter(cfg)
    model = build_pizero_context(cfg, args.checkpoint, lora=True, device=dev)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"trainable {n_train / 1e6:.2f}M params", flush=True)

    short2task = {"eggplant": "widowx_put_eggplant_in_basket", "stack": "widowx_stack_cube",
                  "carrot": "widowx_carrot_on_plate", "spoon": "widowx_spoon_on_towel"}
    if args.dataset == "demos" and args.tasks == "eggplant" and not args.multi_demos:
        train = ContextFrames(args.split_file, args.replay_dir, args.candidates_dir, adapter, ("train",),
                              max_tcp_err_mm=args.max_tcp_err_mm, n_ctx_blocks=args.n_ctx_blocks)
        val = ContextFrames(args.split_file, args.replay_dir, args.candidates_dir, adapter, ("val",),
                            max_tcp_err_mm=args.max_tcp_err_mm, n_ctx_blocks=args.n_ctx_blocks)
        print(f"train frames {len(train)} ({train.n_dropped} dropped by tcp_err), val frames {len(val)}", flush=True)
        enc = BatchEncoder(adapter, train.instruction, dev)
    elif args.dataset == "demos":
        # per-task demo data under <collect-root>/<task>/{replay,candidates_ct} + its split manifest
        from diffusion_policy.pi0_context.context_dataset import MultiTaskFrames
        SPLITS = {"eggplant": "/gscratch/robotics/harine/RoboMonkey_data/experiments/position_holdout_20261009/splits/eggplant/position_holdout_train100.json",
                  "stack": "/gscratch/robotics/harine/RoboMonkey_data/experiments/position_holdout_20261009/splits/stack/position_holdout_train100.json",
                  "carrot": os.path.join(DP_ROOT, "diffusion_policy/config/splits/simpler/carrot/random_train100.json")}
        def part(s, split):
            return ContextFrames(SPLITS[s], os.path.join(args.collect_root, s, "replay"),
                                 os.path.join(args.collect_root, s, "candidates_ct"), adapter, (split,),
                                 max_tcp_err_mm=args.max_tcp_err_mm, n_ctx_blocks=args.n_ctx_blocks)
        train = MultiTaskFrames({short2task[s]: part(s, "train") for s in args.tasks.split(",")})
        val = MultiTaskFrames({short2task[s]: part(s, "val") for s in args.tasks.split(",")})
        print(f"train: {train.summary()}\nval: {val.summary()}", flush=True)
        enc = BatchEncoder(adapter, train.instructions, dev)
    else:
        from diffusion_policy.pi0_context.rollout_dataset import RolloutFrames
        dirs = {short2task[s]: os.path.join(args.collect_root, s, "collect") for s in args.tasks.split(",")}
        kw = dict(n_val_seeds=args.n_val_seeds, success_only=args.success_only, n_ctx_blocks=args.n_ctx_blocks,
                  context=args.ctx_source)
        train = RolloutFrames(dirs, "train", **kw)
        val = RolloutFrames(dirs, "val", **kw)
        print(f"train: {train.summary()}\nval: {val.summary()}", flush=True)
        enc = BatchEncoder(adapter, train.instructions, dev)

    main_params = [p for p in params if p is not model.ctx_attn_bias]
    opt = torch.optim.AdamW([dict(params=main_params, lr=args.lr, base_lr=args.lr),
                             dict(params=[model.ctx_attn_bias], lr=args.bias_lr, base_lr=args.bias_lr,
                                  weight_decay=0.0)],
                            lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    step = 0
    run_id = hashlib.sha1(f"{args.arm}|{out}".encode()).hexdigest()[:16]
    latest = os.path.join(out, "latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        missing = model.load_state_dict(ck["trainable"], strict=False).unexpected_keys
        assert not missing, missing
        opt.load_state_dict(ck["optimizer"])
        step = int(ck["step"])
        rng = np.random.default_rng(args.seed + step)
        print(f"resumed at step {step} from {latest}", flush=True)

    import wandb
    run = wandb.init(entity="l2sml", project=args.wandb_project, group=args.wandb_group, name=args.arm,
                     id=run_id, resume="allow", mode=args.wandb_mode, dir=out,
                     config=dict(vars(args), n_trainable=n_train, train_frames=len(train), val_frames=len(val),
                                 slurm_job_id=os.environ.get("SLURM_JOB_ID"), kind="pi0_context_train"))
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    # fixed validation batches (same frames, t, noise every time)
    vrng = np.random.default_rng(1234)
    vidx = [vrng.choice(len(val), args.batch, replace=False) for _ in range(args.val_batches)]
    vt = [torch.rand(args.batch, generator=torch.Generator().manual_seed(100 + i)).to(dev) for i in range(args.val_batches)]

    def validate():
        model.eval()
        res = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_bf16):
            for k in (0, args.n_ctx_blocks):
                losses = []
                for i, idx in enumerate(vidx):
                    b = enc(val.batch(idx, np.full(args.batch, k)))
                    torch.manual_seed(500 + i)
                    losses.append(model(b["input_ids"], b["pixel_values"], b["attention_mask"], b["proprios"],
                                        b["actions"], vt[i], b["ctx_actions"], b["ctx_values"], b["n_ctx"]).item())
                res[f"val/loss_ctx{k}"] = float(np.mean(losses))
        model.train()
        return res

    def save(tag):
        ck = dict(step=step, trainable=trainable_state(model), optimizer=opt.state_dict(), args=vars(args),
                  wandb_id=run_id)
        torch.save(ck, os.path.join(out, f"{tag}.pt.tmp"))
        os.replace(os.path.join(out, f"{tag}.pt.tmp"), os.path.join(out, f"{tag}.pt"))

    model.train()
    bucket = {k: [] for k in range(args.n_ctx_blocks + 1)}
    t_log = time.time()
    while step < args.steps:
        idx = rng.integers(0, len(train), size=args.batch)
        n_ctx = train.sample_n_ctx(rng, args.batch, args.ctx_mode)
        b = enc(train.batch(idx, n_ctx))
        t = ((torch.rand(1) + torch.arange(args.batch) / args.batch) % (1 - 1e-5)).to(dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args) * g["base_lr"] / args.lr
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_bf16):
            per_sample = model(b["input_ids"], b["pixel_values"], b["attention_mask"], b["proprios"],
                               b["actions"], t, b["ctx_actions"], b["ctx_values"], b["n_ctx"], reduce=False)
        loss = per_sample.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        step += 1
        ps = per_sample.detach().float().cpu().numpy()
        for k in np.unique(n_ctx):
            bucket[int(k)].extend(ps[n_ctx == k].tolist())
        if step % args.log_every == 0:
            dt = (time.time() - t_log) / args.log_every
            t_log = time.time()
            log = {"train/loss": loss.item(), "train/grad_norm": float(gn), "train/lr": lr_at(step, args),
                   "train/step_s": dt, "train/ctx_attn_bias": model.ctx_attn_bias.item(),
                   "train/gpu_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            for k, v in bucket.items():
                if v:
                    log[f"train/loss_ctx{k}"] = float(np.mean(v)); bucket[k] = []
            run.log(log, step=step)
            print(f"step {step} loss {loss.item():.4f} gn {float(gn):.2f} {dt:.2f}s/step "
                  f"bias {model.ctx_attn_bias.item():.2f}", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            v = validate()
            run.log(v, step=step)
            print(f"step {step} " + " ".join(f"{k} {x:.4f}" for k, x in v.items()), flush=True)
        if step % args.ckpt_every == 0 or step == args.steps:
            save(f"ckpt_{step}")
            save("latest")
    run.finish()
    print("done", flush=True)


if __name__ == "__main__":
    main()
