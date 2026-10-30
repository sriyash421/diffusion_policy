#!/usr/bin/env python
"""Offline context for the pi0 context model: N candidate chunks per recorded frame from the
frozen base pi0, scored by the Veritas pixel verifier.

Reads the replay frames (precompute_proprio_replay.py) so the model sees exactly its eval
observation, samples ``--n-samples`` chunks per frame with the official ``infer_action``
(fp32, 10 flow steps, independent noise per sample), and scores each chunk's end effector
endpoint against the frame's ``veritas_targets`` row (H=4 window, lookahead 3, alpha 0.5,
value z-scored with the H=4 expert stats). Per episode ``ep{idx:04d}.npz``:

  ctx_actions_norm (T, N, 4, 7) float32   model-space chunks (what the context block embeds)
  ctx_actions_env  (T, N, 4, 7) float32   adapter.postprocess of the same (xyz, rotvec, +-1)
  ctx_values       (T, N)       float32   normalised verifier value, higher is better
  ctx_cost         (T, N)       float32   raw pixel cost

Resumes per episode. Runs in the veritas venv via step2_job.sbatch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

DP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if DP_ROOT not in sys.path:
    sys.path.insert(0, DP_ROOT)

from diffusion_policy.pi0_context.official import (compose_eval_cfg, frame_inputs,  # noqa: E402
                                                   load_pizero, make_adapter, model_masks)
from diffusion_policy.policy.veritas_verifier import VeritasPixelVerifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--targets", required=True, help="veritas_targets.npy (its .json beside it)")
    ap.add_argument("--value", choices=["cost_over_tol", "zscore"], default="cost_over_tol",
                    help="feedback scalar: task-agnostic -cost/tol_px (default) or the per-task z-score")
    ap.add_argument("--instruction", default=None, help="default: manifest instruction / eggplant")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", default="train,val")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--frames-per-batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera-file", default=os.path.join(
        DP_ROOT, "diffusion_policy/config/veritas/camera_widowx_put_eggplant_in_basket.json"))
    ap.add_argument("--checkpoint", default="models/bridge_uniform_step19296_2024-12-26_22-31_42.pt")
    ap.add_argument("--limit-episodes", type=int, default=0, help="smoke: stop after this many")
    args = ap.parse_args()

    man = json.load(open(args.split_file))
    if all(p in ("train", "val", "test") for p in args.episodes.split(",")):
        eps = sorted(set(i for p in args.episodes.split(",") for i in man[p]))
    else:
        eps = [int(x) for x in args.episodes.split(",")]
    if args.limit_episodes:
        eps = eps[:args.limit_episodes]
    os.makedirs(args.out, exist_ok=True)
    instruction = args.instruction or man.get("instruction", "put eggplant into yellow basket")

    targets_all = np.load(args.targets)
    tj = json.load(open(os.path.splitext(args.targets)[0] + ".json"))
    assert tj["episode_ends_checksum"] == man["episode_ends_checksum"], "targets/zarr mismatch"
    verifier = VeritasPixelVerifier(camera_file=args.camera_file, lookahead_k=tj["lookahead_k"],
                                    lookahead_alpha=tj["lookahead_alpha"], default_tol_px=60.0,
                                    horizon=tj["horizon"], normalize="none", plan_done_mode="zero")

    def value_of(cost, targets_rows):
        if args.value == "zscore":
            return (-cost - tj["value_mean"]) / tj["value_std"]
        tol = np.where(targets_rows[:, 1] > 0, targets_rows[:, 1], 60.0)
        return -cost / tol

    cfg = compose_eval_cfg(man["env_task"])
    model, _ = load_pizero(cfg, args.checkpoint)
    adapter = make_adapter(cfg)
    torch.manual_seed(args.seed)
    N, FB = args.n_samples, args.frames_per_batch
    t0 = time.time()
    for n, ep in enumerate(eps):
        out_path = os.path.join(args.out, f"ep{ep:04d}.npz")
        if os.path.exists(out_path):
            continue
        rp = np.load(os.path.join(args.replay_dir, f"ep{ep:04d}.npz"))
        images, proprio, tcp = rp["image"], rp["proprio"], rp["tcp_world"]
        s, e = int(rp["zarr_start"]), int(rp["zarr_end"])
        targets = targets_all[s:e]
        T = len(images)
        norm = np.zeros((T, N, 4, 7), np.float32)
        envs = np.zeros((T, N, 4, 7), np.float32)
        vals = np.zeros((T, N), np.float32)
        costs = np.zeros((T, N), np.float32)
        for f0 in range(0, T, FB):
            sl = slice(f0, min(f0 + FB, T))
            B = sl.stop - sl.start
            inp = frame_inputs(adapter, images[sl], proprio[sl], instruction)
            # N samples per frame: frame-major repeat so samples of one frame are adjacent
            inp = {k: v.repeat_interleave(N, dim=0) for k, v in inp.items()}
            masks = model_masks(model, inp.pop("attention_mask"))
            with torch.inference_mode():
                chunks = model.infer_action(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                                            proprios=inp["proprios"], **masks)
            chunks = chunks.float().cpu().numpy().reshape(B, N, 4, 7)
            env_chunks = np.stack([[adapter.postprocess(c) for c in row] for row in chunks]).astype(np.float32)
            obs = dict(end_effector_pose=torch.as_tensor(np.repeat(tcp[sl], N, axis=0)),
                       veritas_targets=torch.as_tensor(np.repeat(targets[sl], N, axis=0)))
            act = torch.as_tensor(env_chunks.reshape(B * N, 4, 7))
            c_flat = verifier.cost(obs, act).numpy()
            v = value_of(c_flat, np.repeat(targets[sl], N, axis=0)).reshape(B, N)
            c = c_flat.reshape(B, N)
            norm[sl], envs[sl], vals[sl], costs[sl] = chunks, env_chunks, v, c
        np.savez_compressed(out_path, ctx_actions_norm=norm, ctx_actions_env=envs, ctx_values=vals,
                            ctx_cost=costs, zarr_start=s, zarr_end=e)
        print(f"[{n + 1}/{len(eps)}] ep {ep} T={T} value mean {vals.mean():.2f} "
              f"best-of-{N} mean {vals.max(1).mean():.2f} ({time.time() - t0:.0f}s)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
