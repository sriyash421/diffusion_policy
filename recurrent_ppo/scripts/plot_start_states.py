"""Where episodes start, and how far a step moves the agent.

Two questions this answers, both of which set defaults in train.py:

  * What does RL explore from, versus what the demonstrations start from? The RL reset
    distribution is uniform over a box; the demos are wherever a human happened to begin.
  * How big is a useful action? The human's step distribution is what --delta-scale is set
    from, and it is what makes an absolute-position Gaussian look as coarse as it is.

    python -m recurrent_ppo.scripts.plot_start_states
"""

import os

import argparse

parser = argparse.ArgumentParser(description="Plot PushT start states and demo action steps.")
parser.add_argument("--n-samples", type=int, default=2000, help="RL resets to sample.")
parser.add_argument("--zarr", type=str, default="data/pusht_cchi_v7_replay.zarr", help="Demo dataset.")
parser.add_argument("--out", type=str, default=None, help="Output png. Default: next to this script.")
args_cli = parser.parse_args()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from recurrent_ppo.config import DEFAULTS, WS
from recurrent_ppo.pusht_gym import PushTGymEnv, delta_scale_from_demos, demo_action_steps

# the start boxes live in config now, with every other default
DEFAULT_AGENT_START_RANGE = tuple(DEFAULTS["agent_start_range"])
DEFAULT_BLOCK_START_RANGE = tuple(DEFAULTS["block_start_range"])

# Okabe-Ito, and never hue alone: RL is always a circle / solid, demos always a cross / dashed.
RL_C, DEMO_C = "#0072B2", "#D55E00"
GOAL = np.array([256.0, 256.0, np.pi / 4])


def sample_rl_starts(n):
    """Ground-truth start states from the env we actually train on, not the normalised obs."""
    env = PushTGymEnv(obs_type="keypoint")
    env.reset(seed=0)
    out = np.empty((n, 5))
    for i in range(n):
        env.reset()
        out[i] = list(env.env.agent.position) + list(env.env.block.position) + [env.env.block.angle]
    env.close()
    return out


def load_demo_starts(path):
    root = zarr.open(path, "r")
    state = np.asarray(root["data/state"])
    ends = np.asarray(root["meta/episode_ends"])
    return state[np.concatenate([[0], ends[:-1]])]


def scatter_panel(ax, rl, demo, box, title):
    outside = ((demo < box[0]) | (demo > box[1])).any(axis=1)
    ax.add_patch(plt.Rectangle((box[0], box[0]), box[1] - box[0], box[1] - box[0],
                               fill=False, ec=RL_C, ls="-", lw=1.5,
                               label=f"RL sampling box [{box[0]:.0f},{box[1]:.0f}]"))
    ax.scatter(rl[:, 0], rl[:, 1], s=4, c=RL_C, marker="o", alpha=0.25, label="RL resets")
    ax.scatter(demo[:, 0], demo[:, 1], s=28, c=DEMO_C, marker="x", lw=1.4, label="demo episode starts")
    ax.plot(GOAL[0], GOAL[1], marker="*", ms=18, c="k", ls="none", label="goal pose")
    ax.set(xlim=(0, WS), ylim=(0, WS), title=title, xlabel="x (px)", ylabel="y (px)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower right")
    # the box exists to contain the demonstrated starts; say whether it does
    verdict = (f"{outside.sum()}/{len(demo)} demo starts lie OUTSIDE the RL box"
               if outside.any() else f"all {len(demo)} demo starts lie inside the RL box")
    ax.set_title(f"{title}\n{verdict}", fontsize=10)


def main():
    rl = sample_rl_starts(args_cli.n_samples)
    demo_state, demo_steps = load_demo_starts(args_cli.zarr), demo_action_steps(args_cli.zarr)
    print(f"[INFO] {len(rl)} RL resets, {len(demo_state)} demo episodes, {len(demo_steps)} demo per-axis steps")

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    scatter_panel(axes[0, 0], rl[:, :2], demo_state[:, :2], DEFAULT_AGENT_START_RANGE, "Agent start position")
    scatter_panel(axes[0, 1], rl[:, 2:4], demo_state[:, 2:4], DEFAULT_BLOCK_START_RANGE,
                  "Block start position\n(PushTEnv's own range is the narrower [100,400])")

    # block start angle. The env draws randn()*2pi - pi, a Gaussian -- but a wrapped normal with
    # sigma = 2pi is uniform to ~9 decimal places (first Fourier coefficient exp(-sigma^2/2) ~ 3e-9),
    # so mod 2pi it is uniform in every way that matters.
    ax = axes[1, 0]
    bins = np.linspace(0, 2 * np.pi, 37)
    ax.hist(rl[:, 4] % (2 * np.pi), bins=bins, density=True, histtype="step", lw=2, ls="-",
            color=RL_C, label="RL resets")
    ax.hist(demo_state[:, 4] % (2 * np.pi), bins=bins, density=True, histtype="step", lw=2, ls="--",
            color=DEMO_C, label="demo episode starts")
    ax.axhline(1 / (2 * np.pi), color="k", ls=":", lw=1.5, label="uniform")
    ax.axvline(GOAL[2], color="k", lw=1.5, label="goal angle (pi/4)")
    ax.set(title="Block start angle (mod 2$\\pi$)", xlabel="angle (rad)", ylabel="density")
    ax.legend(fontsize=8)

    # the panel that sets --delta-scale
    ax = axes[1, 1]
    per_axis = demo_steps
    ax.hist(per_axis, bins=np.linspace(0, 80, 81), density=True, color=DEMO_C, alpha=0.55,
            label="demo per-axis step |dx|")
    for pct, style in ((95, "--"), (99, "-.")):
        v = np.percentile(per_axis, pct)
        ax.axvline(v, color=DEMO_C, ls=style, lw=2, label=f"demo p{pct} = {v:.0f} px")
    scale = delta_scale_from_demos(args_cli.zarr)
    for label, px, style in ((f"delta, std=1 ({scale:.0f} px)", scale, "-"),
                             (f"delta, log_std_init=-1 ({scale * np.exp(-1.0):.0f} px)", scale * np.exp(-1.0), ":")):
        ax.axvline(px, color=RL_C, ls=style, lw=2, label=label)
    ax.annotate("absolute action, std=1\n= 256 px, off this axis\n(half the table)",
                xy=(0.97, 0.55), xycoords="axes fraction", ha="right", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec=RL_C, ls="-"))
    ax.set(title="How far one action moves the agent", xlabel="per-axis displacement (px)",
           ylabel="density", xlim=(0, 80))
    ax.legend(fontsize=8)

    fig.suptitle("PushT: RL reset distribution vs human demonstrations", fontsize=14)
    fig.tight_layout()
    out = args_cli.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "start_states.png")
    fig.savefig(out, dpi=130)
    print(f"[INFO] Saved: {out}")


if __name__ == "__main__":
    main()
