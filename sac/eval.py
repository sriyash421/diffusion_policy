"""The four evaluations, as subcommands.

    python sac/eval.py rank-expert -c <st_or_bc.ckpt> --q logs/sac/keypoint/<run>/model.zip
    python sac/eval.py frames      -c <ckpt> --q <sac.zip> --frames 50
    python sac/eval.py bon-sweep   -c <ckpt> --q <sac.zip> --rankers q,t_goal,armTn --max-n 16

ALL OF THEM RESOLVE EPISODES THROUGH ST'S OWN SPLIT. `eval_search_pusht.get_split_states`
returns the 50 held-out DEMO episodes named by the committed manifest, with its checksum
validation and its `splits.json` cross-check -- so an eval can never silently score a checkpoint
against a partition it was not trained under. This is deliberately NOT `recurrent_ppo`'s "fixed
eval set", which is seeded resets of PushTGymEnv: fixed in ST's STYLE, but not ST's SET.

WHAT IS SWAPPED, AND WHAT IS DELIBERATELY NOT. `_score_candidates` returns
`(context, value, subgoal, terms)`: `context` is what ST conditions its NEXT candidate on,
`value` is what argmax ranks by. Only `value` is replaced. ST was trained with a `t_goal`-shaped
search context, so handing it a Q-shaped context at eval would put it off its training
distribution and confound "Q ranks better" with "ST was given an input it has never seen". BC
(`PushTUNetSearchPolicy`) ignores the context entirely, which makes it the cleaner first read.
"""

import json
import pathlib
import sys

import click
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib                                                             # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                               # noqa: E402

from diffusion_policy.common.replay_buffer import ReplayBuffer                # noqa: E402
from diffusion_policy.env.pusht.feedback_util import (compute_feedback_from_pose,   # noqa: E402
                                                      t_goal_distance)
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv          # noqa: E402
from eval_search_pusht import (_eval_split_at_n, build_envs, get_split_states,  # noqa: E402
                               load_policy)
from sac.score import PushTQVerifier                                          # noqa: E402
from scripts.astar_uniform_walk import expert_moved
from scripts.verifier_ranks_expert import (BLIND, CLASSES, T_GOAL, _stats,    # noqa: E402
                                           build_batch, classify, sample_points)

# Each verifier gets a line style and a marker as well as a colour, and is labelled directly at
# its trace, so every figure survives greyscale and colour-blind readers (repo rule).
STYLES = {"q": ("-", "o", "#0072B2"), "t_goal": ("--", "s", "#D55E00"),
          "armTn": (":", "^", "#009E73"), "d_t_goal": ("-.", "D", "#CC79A7")}


@click.group()
def cli():
    """Evaluate a learned Q against the distance heuristic it replaces."""



def _q_scores(q, obs, chunks, To, Ta):
    """Q on the EXECUTED window of each candidate -- the window `_verifier_inputs` slices.

    Scoring the whole horizon instead would score actions the policy never executes, and
    scoring `[:Ta]` would replay one already-past step and miss the last executed one.
    """
    now = {k: v[:, To - 1:To] for k, v in obs.items()}
    flat = chunks.reshape(-1, *chunks.shape[-2:])[:, To - 1:To - 1 + Ta]
    rep = {k: v.repeat_interleave(chunks.shape[1], dim=0) for k, v in now.items()} \
        if chunks.ndim == 4 else now
    return q.get_value(rep, flat).float().cpu().numpy().reshape(chunks.shape[:-2])



def _obs_dict(state, image, device, To=2):
    """The obs dict a verifier receives, from a raw PushT state and its rendered frame."""
    state = np.asarray(state, dtype=np.float32)[None]
    fb = compute_feedback_from_pose(state[:, 2:5])
    t = lambda x: torch.tensor(np.repeat(x[:, None], To, axis=1), dtype=torch.float32, device=device)  # noqa: E731
    out = {"agent_pos": t(state[:, :2]), "feedback": t(fb)}
    if image is not None:
        out["image"] = t(image[None])
    return out



def install_q_ranker(policy, q, skip_context_sim=False):
    """Replace ONLY the ranking scalar in `_score_candidates`. Returns a restore callable.

    Monkeypatched on the instance rather than edited into `pusht_search_mixin`, so nothing that
    ST and BC are trained and evaluated with changes on disk, and an ordinary run is bit-for-bit
    what it was.
    """
    original = policy._score_candidates
    To, Ta = policy.n_obs_steps, policy.n_action_steps
    keys = ("agent_pos", "feedback", "image")

    def patched(verifier, obs_dict, action, want_subgoals=False):
        now = {k: v[:, To - 1:To] for k, v in obs_dict.items() if k in keys}
        exec_action = action[:, To - 1:To - 1 + Ta]
        qv = q.get_value(now, exec_action)
        if skip_context_sim:
            # BC never reads the context, so the sim call is pure overhead. `terms` is only
            # used by cross-candidate values, which are not in play here.
            zeros = torch.zeros_like(qv)
            return zeros, qv.to(zeros.dtype), None, None
        context, value, subgoal, terms = original(verifier, obs_dict, action, want_subgoals)
        return context, qv.to(value.device).to(value.dtype), subgoal, terms

    policy._score_candidates = patched
    return lambda: setattr(policy, "_score_candidates", original)


def install_v_ranker(policy, v):
    """Rank on a learned V evaluated at the state the chunk REACHES. Returns a restore callable.

    NO `skip_context_sim` COUNTERPART, and that is not an omission. Q's shortcut is valid
    because Q scores (state, chunk) directly and needs no rollout; V takes no action, so the
    rollout IS its input -- skipping the sim would leave it nothing to evaluate. `v.rollout`
    therefore calls the sim verifier itself, and the original `_score_candidates` still runs so
    the CONTEXT ST conditions on stays the t_goal-shaped one it was trained with.
    """
    original = policy._score_candidates
    To, Ta = policy.n_obs_steps, policy.n_action_steps
    keys = ("agent_pos", "feedback", "image")

    def patched(verifier, obs_dict, action, want_subgoals=False):
        now = {k: val[:, To - 1:To] for k, val in obs_dict.items() if k in keys}
        exec_action = action[:, To - 1:To - 1 + Ta]
        vv = v.get_value(now, exec_action)
        context, value, subgoal, terms = original(verifier, obs_dict, action, want_subgoals)
        return context, vv.to(value.device).to(value.dtype), subgoal, terms

    policy._score_candidates = patched
    return lambda: setattr(policy, "_score_candidates", original)


def set_sim_value(policy, value_fn):
    """Point the sim verifier at one of VALUE_FNS, both places that must move together."""
    policy.search_kwargs["verifier_value"] = value_fn
    built = policy.__dict__.get("_verifier") or policy.__dict__.get("verifier")
    if built is not None:
        built.value_fn = value_fn


@cli.command("rank-expert")
@click.option('-c', '--checkpoint', required=True, help='the ST or BC checkpoint')
@click.option('--q', 'q_ckpt', default=None, help='the SAC checkpoint holding the learned Q')
@click.option('--v', 'v_ckpt', default=None,
              help='a PPO checkpoint; its V is evaluated at the state each chunk reaches. '
                   'Give --q, --v, or both -- both scores the SAME decisions under each.')
@click.option('--arm', default='unknown')
@click.option('--n', 'n_actions', default=16, show_default=True)
@click.option('--episodes', default=20, show_default=True)
@click.option('--per-episode', default=8, show_default=True)
@click.option('--split', type=click.Choice(['val', 'test']), default='test', show_default=True)
@click.option('--batch', default=8, show_default=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--seed', default=42, show_default=True)
@click.option('--out', default=None, help='write the stats as JSON here')
def rank_expert(checkpoint, q_ckpt, v_ckpt, arm, n_actions, episodes, per_episode, split, batch,
                device, seed, out):
    """Where the EXPERT action ranks among the policy's candidates, under both verifiers.

    Best-of-n is only as good as the thing ranking, and the only ground truth for "good action"
    is the recorded demonstration. Both verifiers score the SAME decisions, the SAME candidates
    and the SAME a*, split by the existing blind/partial/informative classes. The blind column
    is the headline: a Q that only matches the heuristic overall but ranks a* well where the
    heuristic is silent is already the win.
    """
    if not (q_ckpt or v_ckpt):
        raise SystemExit('give --q, --v, or both: there is no learned scorer to compare against')
    policy, cfg = load_policy(checkpoint, device)
    learned = {}
    if q_ckpt:
        learned['q'] = PushTQVerifier(q_ckpt, device=device)
    if v_ckpt:
        from recurrent_ppo.value_score import PushTVVerifier

        # the policy's OWN sim verifier, so V is scored on the same reached states the
        # heuristic column is computed from
        learned['v'] = PushTVVerifier(v_ckpt, policy.verifier, device=device)
    To, Ta, H = policy.n_obs_steps, policy.n_action_steps, cfg.policy.horizon

    run_dir = pathlib.Path(checkpoint).resolve().parent.parent
    _, ep_idxs = get_split_states(cfg, split, run_dir=run_dir)      # ST's OWN held-out episodes
    ep_idxs = list(ep_idxs)[:episodes]
    rb = ReplayBuffer.copy_from_path(cfg.task.dataset.zarr_path,
                                     keys=['img', 'agent_pos', 'action', 'block_pos'])
    pts = sample_points(rb, ep_idxs, per_episode, To, H, np.random.default_rng(seed))
    print(f'{arm}: {len(pts)} decision points from {len(ep_idxs)} {split} episodes, n={n_actions}')

    sim_star, sim_cand, t_cand, ep_of, ref_of = [], [], [], [], []
    learned_star = {k: [] for k in learned}
    learned_cand = {k: [] for k in learned}
    pose_of = []                      # block poses over each executed window, for the T-moved split
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        for b0 in range(0, len(pts), batch):
            chunk = pts[b0:b0 + batch]
            obs, star = build_batch(rb, chunk, To, H, torch.device(device))
            ref_of.append(t_goal_distance(obs['feedback'][:, To - 1].cpu().numpy()))
            # the DEMO's own block poses over the window it executed, for the T-moved split
            bp = np.asarray(rb['block_pos'])
            pose_of.extend(bp[i:i + Ta + 1] for _, i in chunk)
            seeder = getattr(policy, 'set_sample_seeds', None)
            if seeder is not None:
                seeder([seed * 1_000_003 + b0 + k for k in range(len(chunk))])
            with torch.no_grad(), policy._crop_scope():
                feats = policy._encode_obs_features(obs)
                acts, _, sc, tm = policy.predict_n_actions(
                    obs, verifier=policy.verifier, n_actions=n_actions, return_scores=True,
                    obs_features=feats, return_terms=True)
                # a* through THE SAME call the candidates go through -- same verifier, same
                # window. A separate scoring path could drift from the ranking actually applied.
                out_star = policy._score_candidates(policy.verifier, obs, star)
            sim_star.append(out_star[1].float().cpu().numpy())
            sim_cand.append(sc.float().cpu().numpy())
            t_cand.append(tm.float().cpu().numpy())
            for name, scorer in learned.items():
                learned_star[name].append(_q_scores(scorer, obs, star, To, Ta))
                learned_cand[name].append(_q_scores(scorer, obs, acts, To, Ta))
            ep_of += [e for e, _ in chunk]
            print(f'  {min(b0 + batch, len(pts))}/{len(pts)}', end='\r', flush=True)
    finally:
        for obj in (policy, *learned.values()):
            close = getattr(obj, 'close', None)
            if close is not None:
                try:
                    close()
                except Exception as exc:
                    print(f'warning: close failed: {exc}')

    eps = np.array(ep_of)
    refs = np.concatenate(ref_of)
    cls, n_movers, _ = classify(np.concatenate(t_cand)[:, :, T_GOAL], refs)
    # TWO SPLITS, SIDE BY SIDE, because they answer different questions and neither subsumes
    # the other:
    #   classify       did the CANDIDATES move the T -- i.e. did the heuristic have anything to
    #                  say. Targets the failure being fixed, but depends on which candidates
    #                  were drawn, so it differs per arm and per n.
    #   expert_moved   did the DEMO's own T move over this window. A property of the state, so
    #                  the split is byte-identical across every arm compared.
    moved = expert_moved(pose_of, Ta)
    phase = np.where(moved, 'block_moving', 'block_still')
    scores = {'sim': (np.concatenate(sim_star), np.concatenate(sim_cand))}
    for name in learned:
        scores[name] = (np.concatenate(learned_star[name]),
                        np.concatenate(learned_cand[name]))

    report = {'checkpoint': checkpoint, 'q': q_ckpt, 'v': v_ckpt, 'arm': arm, 'n': n_actions,
              'split': split, 'episodes': len(ep_idxs), 'decisions': int(len(eps)),
              'class_fractions': {c: float((cls == c).mean()) for c in CLASSES},
              'phase_fractions': {p: float((phase == p).mean())
                                  for p in ('block_still', 'block_moving')}}
    print(f"\ndecisions: {len(eps)}   " + "  ".join(
        f"{c}={float((cls == c).mean()):.1%}" for c in CLASSES))
    print("  (blind = the sim verifier gave EVERY candidate the same score; argmax there is a "
          "tie-break)\n")

    hdr = f"{'subset':<14}{'n':>6}  {'verifier':<10}{'p_best':>18}{'mean_rank':>18}"
    names = ('sim',) + tuple(learned)

    def table(title, labels, assign, key):
        print(f"\n== {title} ==")
        print(hdr)
        print('-' * len(hdr))
        for subset in labels:
            sel = np.ones(len(eps), bool) if subset == 'all' else (assign == subset)
            if sel.sum() < 2 or len(np.unique(eps[sel])) < 2:
                continue
            report.setdefault(key, {})[subset] = {}
            for name in names:
                sc, cd = scores[name]
                st = _stats(sc[sel], cd[sel], higher_is_better=True, ep_ids=eps[sel], seed=seed)
                report[key][subset][name] = st
                pb, mr = st['p_best'], st['mean_rank']
                print(f"{subset:<14}{int(sel.sum()):>6}  {name:<10}"
                      f"{pb['v']:>8.3f} [{pb['ci'][0]:.2f},{pb['ci'][1]:.2f}]"
                      f"{mr['v']:>8.2f} [{mr['ci'][0]:.1f},{mr['ci'][1]:.1f}]")
            print()

    # candidate-dependent: where the heuristic had anything to say
    table('by candidate informativeness', ('all',) + CLASSES, cls, 'by_class')
    # state-defined and arm-independent: did the demo's own T move
    table('by whether the T moved (demo)', ('block_still', 'block_moving'), phase, 'by_phase')
    print(f"mean_rank is out of {n_actions} candidates; 0 means a* beat every one, "
          f"{n_actions / 2:.1f} is the exchangeability null (ties count as half).")

    if out:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out).write_text(json.dumps(report, indent=2))
        print(f'wrote {out}')


@cli.command("frames")
@click.option('-c', '--checkpoint', required=True, help='the ST or BC checkpoint')
@click.option('--q', 'q_ckpt', required=True)
@click.option('--n', 'n_actions', default=8, show_default=True)
@click.option('--frames', 'n_frames', default=50, show_default=True)
@click.option('--episode', default=0, show_default=True, help='index into the held-out split')
@click.option('--split', type=click.Choice(['val', 'test']), default='test', show_default=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--seed', default=42, show_default=True)
@click.option('-o', '--out', default='sac_eval/value_over_episode', show_default=True)
def frames(checkpoint, q_ckpt, n_actions, n_frames, episode, split, device, seed, out):
    """What the verifiers say while the arm is NOT touching the T.

    `value_t_goal` is a function of where the T ends up and nothing else, so on a decision where
    no candidate reaches the block every candidate returns the identical number. Those are
    28-35% of all decisions and 68-73% of the approach phase, and on every one of them there is
    currently no signal to rank by at all. The figures are the direct picture of whether a
    learned Q fixes that.
    """
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


@cli.command("bon-sweep")
@click.option('-c', '--checkpoint', required=True)
@click.option('--q', 'q_ckpt', default=None, help='SAC checkpoint; required if `q` is a ranker')
@click.option('--rankers', default='q,t_goal,armTn', show_default=True,
              help='comma-separated: any of q, v, t_goal, d_t_goal, armTn')
@click.option('--v', 'v_ckpt', default=None,
              help='PPO checkpoint; required if `v` is a ranker. Its V is evaluated at the '
                   'state each candidate chunk reaches.')
@click.option('--max-n', default=16, show_default=True)
@click.option('--split', type=click.Choice(['val', 'test']), default='test', show_default=True)
@click.option('--n-envs', default=25, show_default=True)
@click.option('--max-steps', default=300, show_default=True)
@click.option('--episodes', default=None, type=int, help='cap, for a smoke run')
@click.option('--skip-context-sim', is_flag=True, help='BC only: it ignores the search context')
@click.option('-d', '--device', default='cuda:0')
@click.option('--seed', default=42, show_default=True)
@click.option('-o', '--out', default='sac_eval/bon_sweep', show_default=True)
def bon_sweep(checkpoint, q_ckpt, rankers, v_ckpt, max_n, split, n_envs, max_steps, episodes,
         skip_context_sim, device, seed, out):
    """The headline: does the learned Q improve a policy through best-of-N?

    n in {1..64} on ST and BC, ranked by the learned Q and by the sim heuristics it replaces, on
    the SAME held-out episodes with the SAME per-episode sampling noise -- so the arms are paired
    episode by episode and the only thing that differs is the ranker. At n=1 every ranker must
    give identical numbers; if they do not, the harness is leaking.
    """
    outdir = pathlib.Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    policy, cfg = load_policy(checkpoint, device)
    if not hasattr(policy, 'predict_action_best'):
        raise SystemExit(f'{type(policy).__name__} has no predict_action_best, so best-of-n is '
                         'undefined for it.')
    rankers = tuple(r.strip() for r in str(rankers).split(',') if r.strip())
    states, idxs = get_split_states(cfg, split, pathlib.Path(checkpoint).resolve().parent.parent)
    if episodes:
        states, idxs = states[:episodes], idxs[:episodes]
    n_list = [2 ** k for k in range(int(np.log2(max_n)) + 1)]
    print(f'[INFO] {len(idxs)} {split} episodes, n in {n_list}, rankers {list(rankers)}')

    q = PushTQVerifier(q_ckpt, device=device) if 'q' in rankers else None
    if 'q' in rankers and q_ckpt is None:
        raise SystemExit('--q is required when `q` is among the rankers')
    if 'v' in rankers and v_ckpt is None:
        raise SystemExit('--v is required when `v` is among the rankers')
    vv = None
    if 'v' in rankers:
        from recurrent_ppo.value_score import PushTVVerifier

        # the SAME sim verifier the heuristic rankers use, so every ranker is compared on
        # identical reached states rather than on separately simulated ones
        vv = PushTVVerifier(v_ckpt, policy.verifier, device=device)

    env = build_envs(min(n_envs, len(idxs)), policy.n_obs_steps, policy.n_action_steps, max_steps)
    curves = {}
    try:
        for ranker in rankers:
            restore = None
            if ranker == 'q':
                restore = install_q_ranker(policy, q, skip_context_sim)
            elif ranker == 'v':
                restore = install_v_ranker(policy, vv)
            else:
                set_sim_value(policy, ranker)
            try:
                rates, cis = [], []
                for n in n_list:
                    # the SAME per-episode noise under every ranker, so the arms are paired
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    rate, ci, _ = _eval_split_at_n(env, policy, states, device, n,
                                                   min(n_envs, len(idxs)), seed,
                                                   label=f'{ranker} n={n}')
                    rates.append(float(rate))
                    cis.append([float(ci[0]), float(ci[1])])
                    print(f'  {ranker:<8} n={n:<3} success {rate:.3f} '
                          f'[{ci[0]:.3f}, {ci[1]:.3f}]')
                curves[ranker] = {'n': n_list, 'success': rates, 'ci': cis}
            finally:
                if restore is not None:
                    restore()
    finally:
        env.close()
        # vv TOO: PushTVVerifier holds a sim env of its own, and leaking it leaves a pygame
        # process per sweep.
        for obj in (policy, q, vv):
            close = getattr(obj, 'close', None)
            if close is not None:
                try:
                    close()
                except Exception as exc:
                    print(f'warning: close failed: {exc}')

    # BOTH checkpoints recorded, not just q. A v-ranked curve whose JSON does not say which V
    # produced it is a number that cannot be attributed later.
    report = {'checkpoint': checkpoint, 'q': q_ckpt, 'v': v_ckpt, 'split': split,
              'rankers': list(rankers), 'seed': seed, 'max_n': max_n,
              'episodes': [int(i) for i in idxs], 'curves': curves}
    (outdir / 'bon_curves.json').write_text(json.dumps(report, indent=2))
    _plot(curves, outdir / 'bon_curves.png', checkpoint)
    print(f'wrote {outdir}/bon_curves.json, bon_curves.png')


def _plot(curves, path, title):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # style AND colour AND a direct label, so the figure survives greyscale (repo rule)
    styles = {'q': ('-', 'o', '#0072B2'), 't_goal': ('--', 's', '#D55E00'),
              'armTn': (':', '^', '#009E73'), 'd_t_goal': ('-.', 'D', '#CC79A7'),
              'v': ((0, (3, 1, 1, 1, 1, 1)), 'v', '#E69F00')}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, c in curves.items():
        ls, mk, col = styles.get(name, ('-', 'x', '#444444'))
        lo = [a for a, _ in c['ci']]
        hi = [b for _, b in c['ci']]
        # linestyle= AS A KEYWORD, not the third positional: that slot is a format
        # STRING, and a dash-pattern tuple lands there as y data instead.
        ax.plot(c['n'], c['success'], linestyle=ls, marker=mk, color=col, label=name, lw=2)
        ax.fill_between(c['n'], lo, hi, color=col, alpha=0.13, lw=0)
        ax.annotate(name, (c['n'][-1], c['success'][-1]), color=col, fontsize=10,
                    xytext=(7, 0), textcoords='offset points', va='center')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('n candidates')
    ax.set_ylabel('success rate (max coverage >= 0.95)')
    ax.set_title(f'Best-of-N by ranker\n{pathlib.Path(title).parent.parent.name}', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    cli()
