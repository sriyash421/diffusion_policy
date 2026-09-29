"""Behaviour-clone the PPO policy itself, so PPO can fine-tune it.

NOT a separate BC model whose weights are later copied across. `diffusion_policy`'s
`LSTMBCPolicy` was built to mirror this architecture, and mirroring turned out not to be
identity: read off its checkpoint, `lstm.weight_ih_l0` was (512, 20) against the PPO arm's
(512, 40), and `mean_head` was (32, 128) -- a 16-step chunk -- against `action_net`'s (2, 128).
Only the trunk and `log_std` lined up, about 33k parameters of a much larger policy.

So this trains the RecurrentPPO policy OBJECT, through `policy.evaluate_actions`, and saves a
RecurrentPPO checkpoint. Fine-tuning is then `RecurrentPPO.load(...).learn()` -- there is
nothing to translate, because it is the same network throughout.

TWO CONVERSIONS, and both have a way of being silently wrong:

  observation  the demo frames must become what THIS arm sees. `sac.env.obs_from_zarr` already
               does it for both arms and is reused rather than reimplemented.
  action       PushT demos are absolute pixel targets; this arm commands
               `target = agent_pos + a * delta_scale`. So the BC target is
               `(action - agent_pos) / delta_scale`, clipped -- NOT the difference between
               consecutive targets, which is a smaller quantity (see pusht_gym.demo_action_steps).

Selection is on ROLLOUT SUCCESS over the manifest's val episodes, not on held-out log-prob: a
policy can fit the demonstrations and still act badly, which is roughly what the existing BC
arms showed, their val loss peaking long before their test score did.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch as th

from recurrent_ppo.eval_episodes import score_on_states, states_from_manifest


def demo_sequences(zarr_path, episode_idxs, obs_type, delta_scale):
    """One (obs, action) pair per demo episode, in THIS arm's observation and action units.

    Returns a list of (obs, act) where obs is (T, ...) and act is (T, 2) in [-1, 1]. Kept as
    whole episodes rather than shuffled transitions because the LSTM's state is only meaningful
    along one.
    """
    import zarr

    from sac.env import obs_from_zarr

    root = zarr.open(str(zarr_path), "r")
    ends = np.asarray(root["meta/episode_ends"])
    starts = np.concatenate([[0], ends[:-1]])
    action = np.asarray(root["data/action"], dtype=np.float64)
    agent = np.asarray(root["data/agent_pos"], dtype=np.float64)

    out = []
    for ep in episode_idxs:
        idx = np.arange(starts[ep], ends[ep])
        obs = obs_from_zarr(root, idx, obs_type)
        # what `_convert_action` inverts: a = (target - agent_pos) / delta_scale, clipped to the
        # action space. 18.8% of steps exceeded the old 33px scale; at the recalibrated 61px
        # (p99 of exactly this quantity) the clip is rare, but it is still applied because the
        # env applies it -- a target the policy cannot command is not a target it should be
        # trained to output.
        act = np.clip((action[idx] - agent[idx]) / float(delta_scale), -1.0, 1.0)
        out.append((obs, act.astype(np.float32)))
    return out


def _pad_batch(policy, seqs, device):
    """Episodes -> the (n_seq * L) flattened layout `_process_sequence` reshapes.

    It does a plain `features.reshape((n_seq, -1, input_size))`, so every sequence in a batch
    must be the SAME length. Episodes are padded to the batch's longest and the padded steps are
    masked out of the loss; PushT demos run ~120-160 frames, so the waste is small.
    """
    lengths = [len(a) for _, a in seqs]
    L, B = max(lengths), len(seqs)

    def pad(x):
        out = np.zeros((B, L) + x[0].shape[1:], dtype=x[0].dtype)
        for i, v in enumerate(x):
            out[i, :len(v)] = v
        return out

    first = seqs[0][0]
    if isinstance(first, dict):
        obs = {k: pad([o[k] for o, _ in seqs]) for k in first}
        obs_t = {k: th.as_tensor(v.reshape((B * L,) + v.shape[2:]), device=device) for k, v in obs.items()}
    else:
        flat = pad([o for o, _ in seqs])
        obs_t = th.as_tensor(flat.reshape(B * L, -1), device=device, dtype=th.float32)

    act = pad([a for _, a in seqs])
    act_t = th.as_tensor(act.reshape(B * L, -1), device=device, dtype=th.float32)

    starts = np.zeros((B, L), dtype=np.float32)
    starts[:, 0] = 1.0                      # each row is its own episode, from its own step 0
    valid = np.zeros((B, L), dtype=np.float32)
    for i, n in enumerate(lengths):
        valid[i, :n] = 1.0

    shape = (policy.lstm_actor.num_layers, B, policy.lstm_actor.hidden_size)
    zeros = lambda: (th.zeros(shape, device=device), th.zeros(shape, device=device))  # noqa: E731
    return (obs_t, act_t,
            th.as_tensor(starts.reshape(-1), device=device),
            th.as_tensor(valid.reshape(-1), device=device),
            zeros())


def bc_epoch(agent, seqs, optimizer, batch_episodes, rng):
    """One pass over the demonstrations. Returns mean negative log-likelihood.

    The loss is the ACTOR's alone. `evaluate_actions` also computes `values`, but the critic has
    no behaviour-cloning target -- BC says what to do, never how good a state is -- so it is
    left at its init and learned during PPO fine-tuning, from on-policy rollouts. That is also
    why the warm start cannot contaminate V: the critic never sees a demonstration.
    """
    from sb3_contrib.common.recurrent.type_aliases import RNNStates

    policy = agent.policy
    policy.set_training_mode(True)
    order = rng.permutation(len(seqs))
    total, n = 0.0, 0
    for a in range(0, len(order), batch_episodes):
        batch = [seqs[i] for i in order[a:a + batch_episodes]]
        obs, act, starts, valid, zeros = _pad_batch(policy, batch, agent.device)
        _, log_prob, _ = policy.evaluate_actions(obs, act, RNNStates(zeros, zeros), starts)
        loss = -(log_prob * valid).sum() / valid.sum().clamp_min(1.0)
        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        total += float(loss) * len(batch)
        n += len(batch)
    return total / max(n, 1)


def evaluate(agent, venv, states, num_envs, max_steps):
    """Mean max-coverage over the val episodes -- the quantity every other arm is scored on."""
    agent.policy.set_training_mode(False)
    rows = score_on_states(agent, venv, states, num_envs, deterministic=True, max_steps=max_steps)
    return (float(np.mean([r["max_reward"] for r in rows])),
            float(np.mean([r["is_success"] for r in rows])))


def main():
    import argparse
    import json

    from recurrent_ppo.arch import ARCHS
    from recurrent_ppo.config import DEFAULTS, DEMO_ZARR
    from recurrent_ppo.pusht_gym import build_vec_env, delta_scale_from_demos, env_kwargs_from

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default="keypoint", choices=("keypoint", "state", "image"))
    ap.add_argument("--split-file",
                    default="diffusion_policy/config/splits/pusht_seed42_train106_val50.json",
                    help="Train on this manifest's TRAIN split, select on its VAL split.")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-episodes", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--num-envs", type=int, default=10, help="parallel envs for the val rollout")
    ap.add_argument("--lstm-hidden-size", type=int, default=DEFAULTS["lstm_hidden_size"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True, help="directory for bc_best.zip / bc_last.zip")
    args = ap.parse_args()

    cfg = dict(DEFAULTS, obs=args.obs, n_stack=1, corrupt_obs=False, device=args.device,
               lstm_hidden_size=args.lstm_hidden_size, seed=args.seed)
    cfg["delta_scale"] = delta_scale_from_demos(DEMO_ZARR, cfg["delta_percentile"])
    print(f"[INFO] delta_scale {cfg['delta_scale']:.1f} px -- the BC target is "
          f"(action - agent_pos)/delta_scale, the quantity this arm commands")

    train_states, train_idxs = states_from_manifest(args.split_file, "train")
    val_states, val_idxs = states_from_manifest(args.split_file, "val")
    print(f"[INFO] {len(train_idxs)} train episodes, {len(val_idxs)} val, from {args.split_file}")

    seqs = demo_sequences(DEMO_ZARR, train_idxs, args.obs, cfg["delta_scale"])
    clipped = float(np.mean([np.mean(np.abs(a) >= 1.0 - 1e-6) for _, a in seqs]))
    print(f"[INFO] {len(seqs)} demo episodes, {sum(len(a) for _, a in seqs)} steps; "
          f"{clipped:.2%} of action components sit at the clip")

    venv = build_vec_env(obs_type=args.obs, n_envs=args.num_envs, seed=args.seed + 10_000,
                         use_subproc=False, **env_kwargs_from(cfg, args.obs))
    agent = ARCHS["lstm"].build(venv, cfg, log_dir=None)
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.out, exist_ok=True)
    history, best = [], -np.inf
    for epoch in range(1, args.epochs + 1):
        nll = bc_epoch(agent, seqs, optimizer, args.batch_episodes, rng)
        row = {"epoch": epoch, "nll": nll}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            cov, suc = evaluate(agent, venv, val_states, args.num_envs,
                               cfg["max_episode_steps"])
            row.update(val_max_reward=cov, val_success=suc)
            # SELECTED ON ROLLOUT, not on NLL. A policy can fit the demonstrations and still act
            # badly; the existing BC arms' val loss peaked long before their test score did.
            if cov > best:
                best = cov
                agent.save(os.path.join(args.out, "bc_best.zip"))
                row["saved"] = True
            print(f"epoch {epoch:4d}  nll {nll:8.4f}  val_max_reward {cov:.4f}  "
                  f"val_success {suc:.3f}{'  <- best' if row.get('saved') else ''}", flush=True)
        else:
            print(f"epoch {epoch:4d}  nll {nll:8.4f}", flush=True)
        history.append(row)

    agent.save(os.path.join(args.out, "bc_last.zip"))
    with open(os.path.join(args.out, "bc_history.json"), "w") as f:
        json.dump({"args": vars(args), "delta_scale": cfg["delta_scale"],
                   "train_episodes": [int(i) for i in train_idxs],
                   "val_episodes": [int(i) for i in val_idxs],
                   "best_val_max_reward": best, "history": history}, f, indent=2)
    venv.close()
    print(f"[INFO] best val max_reward {best:.4f}; wrote {args.out}/bc_best.zip")


if __name__ == "__main__":
    main()
