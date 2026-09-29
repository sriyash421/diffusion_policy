"""Decode the SD-VAE latent corruption back to images, so a noise level can be CHOSEN by eye.

The audit's finding is that a DDPM timestep `t` only means something if the thing being noised
has a known scale. The frozen SD VAE is the one space in this repo where it does: x0.18215 puts
the PushT latent at ~unit variance, which is exactly what the schedule assumes. This renders the
round trip at a ladder of `t` so the question "how much of the observation survives" is answered
by looking rather than by a table.

The gate, borrowed from scripts/decode_obs_latents.py: a corrupted latent should decode to a
BLURRED, UNCERTAIN T -- still a plausible scene -- not to noise. A level whose decode is noise is
not a hard observation, it is an absent one.

    python -m recurrent_ppo.scripts.render_noise_levels

The forward chain matches SDVAEEncoder exactly (render 96 -> centre-crop 72 -> [0,1]*2-1 ->
encode -> posterior MEAN -> x0.18215); the decode is its exact inverse.
"""

import os

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import AutoencoderKL
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from recurrent_ppo.pusht_gym import PushTGymEnv

SCALING_FACTOR = 0.18215          # SDVAEEncoder's default; calibrated on PushT frames
CROP = 72                         # crop_shape in the ST image configs -> a 4x9x9 = 324-d latent
DEFAULT_LADDER = [0, 50, 100, 150, 200, 300, 400, 600, 800, 999]


def sample_frames(n, seed, render_size):
    """Frames from the env the agent actually sees, not from the demo dataset."""
    env = PushTGymEnv(obs_type="image", render_size=render_size, max_episode_steps=300)
    obs, _ = env.reset(seed=seed)
    frames = []
    while len(frames) < n:
        for _ in range(env.np_random.integers(5, 40)):      # decorrelate the frames
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                obs, _ = env.reset()
        frames.append(obs["image"].astype(np.float32) / 255.0)   # (3, H, W) in [0, 1]
    env.close()
    return np.stack(frames)


def centre_crop(x, size):
    _, _, h, w = x.shape
    top, left = (h - size) // 2, (w - size) // 2
    return x[:, :, top:top + size, left:left + size]


def main():
    ap = argparse.ArgumentParser(description="Render SD-VAE latent corruption at a ladder of timesteps.")
    ap.add_argument("-n", "--n-frames", type=int, default=6, help="columns in the panel")
    ap.add_argument("--timesteps", type=int, nargs="+", default=DEFAULT_LADDER, help="rows in the panel")
    ap.add_argument("--render-size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="recurrent_ppo/noise_levels.png")
    args = ap.parse_args()

    print(f"[INFO] sampling {args.n_frames} frames")
    frames = sample_frames(args.n_frames, args.seed, args.render_size)
    x = torch.from_numpy(frames).to(args.device)
    x = centre_crop(x, CROP) * 2.0 - 1.0                    # get_image_range_normalizer

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(args.device).eval()
    sched = DDPMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                          beta_schedule="linear", prediction_type="epsilon")
    ab = sched.alphas_cumprod.numpy()

    with torch.no_grad():
        latent = vae.encode(x).latent_dist.mean * SCALING_FACTOR
    print(f"[INFO] latent {tuple(latent.shape)} -> {latent[0].numel()} dims | "
          f"mean {latent.mean():.3f}  std {latent.std():.3f}   "
          f"(SDVAEEncoder documents mean 0.48 / std 1.00)")

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    rows = []
    for t in args.timesteps:
        noise = torch.randn(latent.shape, generator=g).to(args.device)
        noised = sched.add_noise(latent, noise, torch.full((latent.shape[0],), t, dtype=torch.long))
        with torch.no_grad():
            img = vae.decode(noised / SCALING_FACTOR).sample
        rows.append(((img + 1) / 2).clamp(0, 1).cpu().numpy())

    original = ((x + 1) / 2).clamp(0, 1).cpu().numpy()
    panel = [("original\n(no VAE)", original)] + [
        (f"t={t}\n" + r"$\sqrt{\bar\alpha}$=" + f"{np.sqrt(ab[t]):.3f}\nSNR={ab[t]/(1-ab[t]+1e-12):7.2f}", r)
        for t, r in zip(args.timesteps, rows)]

    nrow, ncol = len(panel), args.n_frames
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.5 * ncol, 1.55 * nrow))
    axes = np.atleast_2d(axes)
    for i, (label, imgs) in enumerate(panel):
        for j in range(ncol):
            ax = axes[i, j]
            ax.imshow(np.transpose(imgs[j], (1, 2, 0)))
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center", labelpad=42)
    fig.suptitle("PushT observation through the frozen SD VAE, corrupted at DDPM timestep t\n"
                 "the gate: a usable level still decodes to a blurred T, not to noise", fontsize=11)
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"[INFO] saved {args.out}")


if __name__ == "__main__":
    main()
