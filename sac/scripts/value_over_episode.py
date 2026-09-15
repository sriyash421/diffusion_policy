"""EVAL TEST 3 -- what the verifiers say while the arm is NOT touching the T.

This is the mechanism test, and the one the blindness statistics make directly falsifiable.
`value_t_goal` is a function of where the T ends up and nothing else, so on a decision where no
candidate reaches the block, EVERY candidate returns the identical number -- a measured 0.0000
px spread. Those decisions are 28-35% of all of them and 68-73% of the approach phase. There is
currently no signal at all to rank candidates by on any of them.

The figures are the direct picture of whether a learned Q fixes that:

  frames.png   the render at each no-contact decision, with the n candidate chunks drawn on it,
               the Q-best highlighted and the Q-worst dashed.
  values.png   per-frame spread ACROSS candidates for each verifier. `t_goal` is a flat line at
               zero here by construction; the question is whether Q's is not.

Colour is never the only channel: each verifier gets its own line style and marker and is
labelled directly at the end of its trace, and candidate rank is drawn as width plus an
explicit text label as well as shade.

    python sac/scripts/value_over_episode.py -c <st.ckpt> --q logs/sac/<arm>/<run>/model.zip
"""

import json
import pathlib
import sys

import click
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib                                                             # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                               # noqa: E402

from diffusion_policy.env.pusht.feedback_util import (compute_feedback_from_pose,   # noqa: E402
                                                      t_goal_distance)
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv          # noqa: E402
from diffusion_policy.env.pusht.pusht_verifier import VALUE_FNS               # noqa: E402
from eval_search_pusht import get_split_states, load_policy                   # noqa: E402
from sac.score import PushTQVerifier                                          # noqa: E402

# Each verifier gets a style as well as a colour, so the figure survives being printed in
# greyscale or read by someone who cannot separate the hues (repo rule).
STYLES = {"q": ("-", "o", "#0072B2"), "t_goal": ("--", "s", "#D55E00"),
          "armTn": (":", "^", "#009E73")}


def _obs_dict(state, image, device, To=2):
    """The obs dict a verifier receives, from a raw PushT state and its rendered frame."""
    state = np.asarray(state, dtype=np.float32)[None]
    fb = compute_feedback_from_pose(state[:, 2:5])
    t = lambda x: torch.tensor(np.repeat(x[:, None], To, axis=1), dtype=torch.float32, device=device)  # noqa: E731
    out = {"agent_pos": t(state[:, :2]), "feedback": t(fb)}
    if image is not None:
        out["image"] = t(image[None])
    return out


@click.command()
@click.option('-c', '--checkpoint', required=True, help='the ST or BC checkpoint')
@click.option('--q', 'q_ckpt', required=True)
@click.option('--n', 'n_actions', default=8, show_default=True)
@click.option('--frames', 'n_frames', default=50, show_default=True)
@click.option('--episode', default=0, show_default=True, help='index into the held-out split')
@click.option('--split', type=click.Choice(['val', 'test']), default='test', show_default=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--seed', default=42, show_default=True)
@click.option('-o', '--out', default='sac_eval/value_over_episode', show_default=True)
def main(checkpoint, q_ckpt, n_actions, n_frames, episode, split, device, seed, out):
    outdir = pathlib.Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    policy, cfg = load_policy(checkpoint, device)
    q = PushTQVerifier(q_ckpt, device=device)
    To, Ta = policy.n_obs_steps, policy.n_action_steps

    states, _ = get_split_states(cfg, split, pathlib.Path(checkpoint).resolve().parent.parent)
    env = PushTImageEnv(legacy=False, render_size=96)
    env.reset_to_state = np.asarray(states[episode], dtype=np.float64)
    env.seed(seed)
    obs = env.reset()

    torch.manual_seed(seed)
    np.random.seed(seed)
    rows, panels = [], []
    step = 0
    while len(rows) < n_frames and step < 300:
        info = env._get_info()
        state = np.concatenate([info["pos_agent"], info["block_pose"]])
        no_contact = int(np.sum(info["n_contacts"])) == 0
        od = _obs_dict(state, obs["image"], device, To)
        with torch.no_grad(), policy._crop_scope():
            acts, _, sc = policy.predict_n_actions(
                od, verifier=policy.verifier, n_actions=n_actions, return_scores=True)
        chunks = acts[:, :, To - 1:To - 1 + Ta]                       # the executed window
        flat = chunks.reshape(-1, Ta, 2)
        rep = {k: v.repeat_interleave(n_actions, dim=0) for k, v in od.items()}
        qv = q.get_value(rep, flat).float().cpu().numpy()
        # the sim values, recomputed per candidate from the reached state the policy already got
        sims = {"t_goal": sc.float().cpu().numpy().ravel()}

        if no_contact:
            rows.append({"step": step, "q": qv.tolist(), **{k: v.tolist() for k, v in sims.items()},
                         "t_goal_ref": float(t_goal_distance(
                             od["feedback"][:, -1].cpu().numpy()))})
            panels.append((env.render("rgb_array"), chunks[0].float().cpu().numpy(), qv, step))
        # execute the argmax candidate, as BON would
        best = int(np.argmax(sims["t_goal"]))
        for a in chunks[0, best].float().cpu().numpy():
            obs, _, done, _ = env.step(np.clip(a, 0, 512))
            step += 1
            if done:
                break
        if step >= 300:
            break

    print(f"[INFO] {len(rows)} no-contact decisions captured from episode {episode}")
    if not rows:
        print("[WARN] the arm touched the block at every decision; nothing to plot.")
        return

    # ---- frames.png: candidates drawn on the render, ranked by Q
    cols = 5
    n = min(len(panels), n_frames)
    fig, axes = plt.subplots((n + cols - 1) // cols, cols,
                             figsize=(3.0 * cols, 3.0 * ((n + cols - 1) // cols)))
    for ax, (frame, chunk, qv, st) in zip(np.atleast_1d(axes).ravel(), panels[:n]):
        ax.imshow(frame, extent=[0, 512, 512, 0])
        order = np.argsort(-qv)
        for rank, j in enumerate(order):
            best, worst = rank == 0, rank == len(order) - 1
            ax.plot(chunk[j][:, 0], chunk[j][:, 1],
                    linestyle="-" if best else ("--" if worst else "-"),
                    linewidth=2.6 if best else (1.6 if worst else 1.0),
                    color="#0072B2" if best else ("#D55E00" if worst else "#999999"),
                    marker="o" if best else None, markersize=3, zorder=3 if best else 2)
        # rank as TEXT as well as style, so the ordering does not depend on seeing colour
        ax.annotate(f"best Q {qv.max():.3f}\nworst {qv.min():.3f}\nspread {qv.std():.4f}",
                    (6, 20), fontsize=6, color="black",
                    bbox=dict(fc="white", alpha=0.75, lw=0))
        ax.set_title(f"step {st}", fontsize=7)
        ax.set_xlim(0, 512); ax.set_ylim(512, 0); ax.set_xticks([]); ax.set_yticks([])
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"No-contact decisions: {n_actions} candidates, solid blue = Q's pick, "
                 f"dashed orange = Q's worst", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "frames.png", dpi=130)
    plt.close(fig)

    # ---- values.png: spread across candidates, per verifier, per frame
    steps = [r["step"] for r in rows]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name in ("q", "t_goal"):
        if name not in rows[0]:
            continue
        ls, mk, c = STYLES[name]
        spread = [float(np.std(r[name])) for r in rows]
        ax0.plot(steps, spread, ls, marker=mk, ms=3.5, color=c, label=name)
        ax0.annotate(name, (steps[-1], spread[-1]), color=c, fontsize=9,
                     xytext=(6, 0), textcoords="offset points", va="center")
        rng = [float(np.max(r[name]) - np.min(r[name])) for r in rows]
        ax1.plot(steps, rng, ls, marker=mk, ms=3.5, color=c, label=name)
        ax1.annotate(name, (steps[-1], rng[-1]), color=c, fontsize=9,
                     xytext=(6, 0), textcoords="offset points", va="center")
    ax0.set_ylabel("std of the score\nACROSS candidates")
    ax1.set_ylabel("max - min\nACROSS candidates")
    ax1.set_xlabel("episode step (no-contact decisions only)")
    ax0.set_title("If a curve sits at zero, that verifier expressed NO preference and argmax "
                  "was a tie-break", fontsize=10)
    for ax in (ax0, ax1):
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "values.png", dpi=140)
    plt.close(fig)

    blind = {name: float(np.mean([np.std(r[name]) <= 1e-9 for r in rows]))
             for name in ("q", "t_goal") if name in rows[0]}
    (outdir / "values.json").write_text(json.dumps(
        {"checkpoint": checkpoint, "q": q_ckpt, "episode": episode, "rows": rows,
         "blind_fraction": blind}, indent=2))
    print("blind fraction on these no-contact decisions: "
          + "  ".join(f"{k}={v:.1%}" for k, v in blind.items()))
    print(f"wrote {outdir}/frames.png, values.png, values.json")


if __name__ == '__main__':
    main()
