"""Running one: the training loop, its diagnostics, and the entry point.

    python sac/runner.py --obs keypoint
    python sac/runner.py --obs image --num-envs 8
    python sac/runner.py --obs keypoint --tau-ladder 0.95      # the single-head version

The run-directory bookkeeping is imported from `recurrent_ppo.run_io` rather than
re-implemented: a run's directory is the record of what it WAS, and two copies of that logic is
how a resume silently continues one experiment under another's name.

THE DIAGNOSTICS ARE THE POINT of this file as much as the loop is. The failure this package has
to avoid is a Q that looks healthy by every SAC metric and is useless as a verifier, and that
happens two ways, both invisible to `critic_loss`:

  * **Q collapses to V(s)** -- it scores every candidate at a state identically, which is
    EXACTLY the degeneracy of the heuristic being replaced, restated in Q coordinates.
    `bon/q_spread_zero_frac` measures it, and the number to beat is the heuristic's own: blind
    on 28-35% of decisions overall and 68-73% during approach.
  * **Nothing to learn from.** Under a sparse reward at a threshold NO demonstration reaches,
    every rung can sit at zero positives forever. `buffer/reward_rate_tau*` says so on day one
    rather than after a week.
"""

import contextlib
import os
import sys
from datetime import datetime

# Running this file directly puts `sac/` on sys.path, not the repo root, so `import sac` fails.
# It has to happen BEFORE the imports below, which is why it is not inside `__main__`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, LogEveryNTimesteps

from diffusion_policy.common.slot_stats_util import BLIND_EPS
from recurrent_ppo.corrupt_policy import aug_for
from recurrent_ppo.eval_episodes import states_from_manifest
from recurrent_ppo.run_io import check_conflicts, dump_args, load_args
from recurrent_ppo.runner import _claim
from sac.agent import ChunkSAC, buffer_class_for, preload_demos
from sac.config import DEFAULTS, DEMO_ZARR, IDENTITY_KEYS, buffer_size_for, parse_train_args
from sac.env import (assert_roundtrip, build_chunk_vec_env, demo_chunks, demo_transitions,
                     env_kwargs_from, summarise)

LOG_ROOT = "sac"


# ==================================================================== the diagnostics


class CurriculumAnneal(BaseCallback):
    """Widen the near-goal reset offset from `start` to `final` over the first `frac` of training.

    The tight end exists because nothing else produces a tau=0.95 terminal -- no demonstration
    ever reaches 0.95, and at recurrent_ppo's [30, 0.25] the near-goal reset sits at coverage
    0.48 and never crosses. The wide end exists because the FINAL Q has to be accurate on the
    real start distribution, not on the curriculum's.
    """

    def __init__(self, start, final, total_timesteps, frac, prob_start=None, prob_final=None):
        super().__init__()
        self.start, self.final = np.asarray(start, float), np.asarray(final, float)
        self.prob_start, self.prob_final = prob_start, prob_final
        self.horizon = max(1.0, float(total_timesteps) * float(frac))

    def _on_step(self):
        t = min(1.0, self.num_timesteps / self.horizon)
        offset = (1 - t) * self.start + t * self.final
        self.training_env.set_attr("block_goal_offset", tuple(offset))
        self.logger.record("curriculum/block_goal_offset_pos", float(offset[0]))
        if self.prob_start is not None:
            # The PROBABILITY anneals too, not just the offset. Annealing the offset alone
            # leaves only `block_near_goal_prob` of episodes anywhere near the threshold from
            # the first step, and measured over a 50k-step run that collected ONE tau=0.95
            # terminal in total -- the top rung had nothing to learn from. Start with most
            # episodes near the goal and hand the distribution back as the agent improves.
            prob = (1 - t) * self.prob_start + t * self.prob_final
            self.training_env.set_attr("block_near_goal_prob", float(prob))
            self.logger.record("curriculum/block_near_goal_prob", float(prob))
        return True


class LadderStats(BaseCallback):
    """Per-rung positive-reward rate in the buffer. A rung at zero teaches Q = 0."""

    def __init__(self, tau_ladder, freq=10_000):
        super().__init__()
        self.tau_ladder, self.freq = tuple(tau_ladder), int(freq)
        self._next = 0

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next = self.num_timesteps + self.freq
        buf = self.model.replay_buffer
        if not hasattr(buf, "tau_done"):
            return True
        upper = buf.buffer_size if buf.full else buf.pos
        if upper == 0:
            return True
        done, live = buf.tau_done[:upper], buf.tau_live[:upper]
        for k, tau in enumerate(self.tau_ladder):
            n_live = max(1.0, live[..., k].sum())
            self.logger.record(f"buffer/reward_rate_tau{tau:.2f}", float(done[..., k].sum() / n_live))
            self.logger.record(f"buffer/terminals_tau{tau:.2f}", float(done[..., k].sum()))
        return True


class QSpreadProbe(BaseCallback):
    """Does Q distinguish candidates at a fixed state? The replacement metric.

    Proposes `n_candidates` actor samples at each probe state and measures the spread of Q
    across them. `q_spread_zero_frac` is the fraction of states where that spread is below
    `BLIND_EPS` -- the same threshold `slot_stats_util` uses to call a heuristic decision blind,
    so the two numbers are directly comparable.

    `spread_within / spread_across` is the resolution check. All n candidates share a state and
    differ by at most one chunk of progress, so if the within-state spread is tiny against the
    across-state spread, the candidates sit inside the regression's noise floor and gamma is
    too high to resolve them.
    """

    def __init__(self, env, freq=50_000, n_states=64, n_candidates=16, seed=0):
        super().__init__()
        self.env, self.freq = env, int(freq)
        self.n_states, self.n_candidates, self.seed = int(n_states), int(n_candidates), int(seed)
        self._next = 0

    def _on_step(self):
        if self.freq <= 0 or self.num_timesteps < self._next:
            return True
        self._next = self.num_timesteps + self.freq
        self.env.seed(self.seed)
        obs = self.env.reset()
        rows = []
        while len(rows) < self.n_states:
            obs_t, _ = self.model.policy.obs_to_tensor(obs)
            with th.no_grad():
                feats = self.model._features(obs_t)
                cand = self.model._candidate_actions(obs_t, self.n_candidates)   # (n, B, A)
                q = th.stack([self.model.policy.bon_head.mean(feats, cand[i])[:, -1]
                              for i in range(self.n_candidates)], dim=0)         # (n, B)
            rows.append(q.T.cpu().numpy())                                       # (B, n)
            actions, _ = self.model.predict(obs, deterministic=False)
            obs, _, _, _ = self.env.step(actions)
        q = np.concatenate(rows, axis=0)[:self.n_states]                          # (S, n)
        within = q.std(axis=1)
        self.logger.record("bon/q_spread_zero_frac", float((within <= BLIND_EPS).mean()))
        self.logger.record("bon/q_spread_within", float(np.median(within)))
        self.logger.record("bon/q_spread_across", float(q.mean(axis=1).std()))
        across = max(1e-12, float(q.mean(axis=1).std()))
        self.logger.record("bon/q_resolution", float(np.median(within) / across))
        return True


class FixedEval(BaseCallback):
    """Success on a FIXED episode set, re-seeded before every evaluation.

    SB3 seeds an eval env once at construction and never again, so consecutive points are
    unpaired draws and most of the movement between checkpoints is the draw rather than the
    policy. Re-seeding replays the same episodes in the same order. This is ST's convention
    (`test_start_seed`); note it is fixed in ST's STYLE, not ST's SET -- the ST/BC eval set is
    the 50 held-out demo episodes, and only sac/scripts use that.
    """

    def __init__(self, env, freq, n_episodes, seed, prefix="eval"):
        super().__init__()
        self.env, self.freq, self.n_episodes = env, int(freq), int(n_episodes)
        self.seed, self.prefix = int(seed), prefix
        self._next = int(freq)

    def _on_step(self):
        if self.freq <= 0 or self.num_timesteps < self._next:
            return True
        self._next = self.num_timesteps + self.freq
        self.env.seed(self.seed)
        n_envs = self.env.num_envs
        budget = [(self.n_episodes + i) // n_envs for i in range(n_envs)]
        done_count = [0] * n_envs
        successes, coverage, returns = [], [], []
        obs = self.env.reset()
        while any(d < b for d, b in zip(done_count, budget)):
            actions, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, infos = self.env.step(actions)
            for i, (d, info) in enumerate(zip(dones, infos)):
                if d and done_count[i] < budget[i]:
                    done_count[i] += 1
                    successes.append(float(info.get("is_success", 0.0)))
                    coverage.append(float(info.get("max_reward", 0.0)))
                    returns.append(float(info["episode"]["r"]) if "episode" in info else 0.0)
        self.logger.record(f"{self.prefix}/success_rate", float(np.mean(successes)))
        # max coverage is what pusht_image_runner reports, so this is the number comparable
        # with the diffusion-policy arms
        self.logger.record(f"{self.prefix}/max_coverage", float(np.mean(coverage)))
        self.logger.record(f"{self.prefix}/mean_reward", float(np.mean(returns)))
        return True


def arm_tags(cfg):
    """The labels describing what a run IS, derived from the config rather than typed."""
    tags = [f"obs-{cfg['obs']}", f"reward-{cfg['reward']}", f"gamma-{cfg['gamma']}", "algo-sac"]
    if cfg["block_near_goal_prob"] > 0:
        tags.append("init-near-goal")
    if cfg["demo_seed_frac"] > 0:
        tags.append("demo-seeded")
    if len(cfg["tau_ladder"]) > 1:
        tags.append(f"ladder-{len(cfg['tau_ladder'])}")
    return tags


# ==================================================================== the training loop


def _policy_kwargs(cfg):
    kwargs = {"net_arch": [int(x) for x in str(cfg["net_arch"]).split(",")],
              "n_critics": cfg["n_critics"]}
    if cfg["obs"] == "image":
        # ST's own encoder, so the Q learns from the representation the policies it verifies use
        from recurrent_ppo.corrupt_policy import STResNetExtractor

        kwargs["features_extractor_class"] = STResNetExtractor
        kwargs["share_features_extractor"] = bool(cfg["share_features_extractor"])
    return kwargs


def _build_agent(cfg, env, log_dir, demo_offsets):
    policy = "MultiInputPolicy" if cfg["obs"] == "image" else "MlpPolicy"
    return ChunkSAC(
        policy, env,
        obs_type=cfg["obs"], tau_ladder=cfg["tau_ladder"], demo_chunks=demo_offsets,
        mix=cfg["mix"], smooth_scale=cfg["smooth_scale"], bon_n_critics=cfg["n_critics"],
        gamma=cfg["gamma"], learning_rate=cfg["learning_rate"], batch_size=cfg["batch_size"],
        learning_starts=cfg["learning_starts"], train_freq=cfg["train_freq"],
        gradient_steps=cfg["gradient_steps"], tau=cfg["tau"], ent_coef=cfg["ent_coef"],
        target_entropy=cfg["target_entropy"], buffer_size=cfg["buffer_size"],
        replay_buffer_class=buffer_class_for(cfg["obs"]),
        replay_buffer_kwargs={"n_tau": len(cfg["tau_ladder"])},
        policy_kwargs=_policy_kwargs(cfg), tensorboard_log=log_dir,
        seed=cfg["seed"], device=cfg["device"], verbose=1)


def train(args_cli):
    # DEFAULTS underneath, so a setting with no flag (block_zero_coverage, which MUST be False
    # here) is still present and still recorded in args.yaml. A missing key is otherwise only
    # discovered at the call site that needs it.
    cfg = dict(DEFAULTS, **vars(args_cli))
    cfg["buffer_size"] = buffer_size_for(cfg["obs"], cfg["buffer_size"])
    resuming = args_cli.checkpoint is not None

    if resuming:
        log_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
        check_conflicts(load_args(log_dir), cfg, keys=IDENTITY_KEYS)
        print(f"[INFO] Resuming run in: {log_dir}")
    else:
        run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.abspath(args_cli.log_dir
                                  or os.path.join("logs", LOG_ROOT, cfg["obs"], run_info))
        log_dir = _claim(log_dir)
        dump_args(log_dir, cfg)
        print(f"[INFO] Logging experiment in directory: {log_dir}")
    with open(os.path.join(log_dir, "command.txt"), "a") as f:
        f.write(" ".join([sys.executable] + sys.argv) + "\n")

    run = None
    if args_cli.wandb:
        import wandb

        tags = sorted(set(args_cli.wandb_tags) | set(arm_tags(cfg)))
        run = wandb.init(entity=args_cli.wandb_entity, project=args_cli.wandb_project,
                         group=args_cli.wandb_group, name=os.path.basename(log_dir), tags=tags,
                         config=cfg, sync_tensorboard=True, dir=log_dir,
                         resume="allow" if resuming else None)
        print(f"[INFO] W&B: {run.url}  tags: {tags}")

    # THE INVERTIBILITY PROOF, before anything depends on it: a Q cannot score a chunk it
    # cannot express, and ST's candidates are absolute targets from this distribution.
    A, P = demo_chunks(DEMO_ZARR)
    err = assert_roundtrip(A)
    print(f"[INFO] chunk codec round-trips the demonstrations to {err:.2e} px")
    demo_offsets = A - P[:, None, :]          # chunk SHAPES, re-anchorable anywhere

    env_kwargs = env_kwargs_from(cfg, cfg["obs"])
    # The crop draw, on BOTH envs. It is what carries a per-transition offset in the
    # observation, so the image arm crops identically while collecting and while updating
    # rather than letting CropRandomizer branch on SB3's training-mode flag, which is False in
    # collect_rollouts and True in train(). `aug_for` returns None for the keypoint arm, so
    # that arm is untouched by construction. `corrupt_obs=False`: this arm studies the
    # verifier, not observation noise.
    aug = aug_for(cfg["obs"], corrupt_obs=False, render_size=cfg["render_size"],
                  random_crop=True)
    env = build_chunk_vec_env(obs_type=cfg["obs"], n_envs=cfg["num_envs"], seed=cfg["seed"],
                              use_subproc=not cfg["dummy_vec_env"], monitor_dir=log_dir,
                              aug=aug, **env_kwargs)
    # evaluation is on the REAL start distribution, never the curriculum's: the curriculum is
    # scaffolding for learning, not the task being measured.
    eval_kwargs = dict(env_kwargs, block_near_goal_prob=0.0)
    eval_env = build_chunk_vec_env(obs_type=cfg["obs"], n_envs=min(cfg["num_envs"], 8),
                                   seed=cfg["seed"] + 10_000,
                                   use_subproc=not cfg["dummy_vec_env"], aug=aug, **eval_kwargs)

    if resuming:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        agent = ChunkSAC.load(args_cli.checkpoint, env, device=cfg["device"])
    else:
        agent = _build_agent(cfg, env, log_dir, demo_offsets)
        if cfg["demo_seed_frac"] > 0:
            # `train` holds out only `split_file`'s own test/val half. The best-of-N sweep
            # scores on the GEOMETRIC manifests, which cut the same 206 episodes differently --
            # seed-42's 106 train episodes cover 27 of blq137's 50 test episodes and 24 of
            # brd60's 50 -- so it bought a half-clean eval set, which is worse than none:
            # the bias lands unevenly across strata and only on the learned side, since the
            # heuristic has no training set. `all` makes it uniform and declared.
            if cfg["demo_episodes"] == "all":
                train_idxs = None
                print("[INFO] demo seeding from ALL episodes -- EVERY eval episode is in the "
                      "buffer, uniformly. Q numbers are not held out; say so beside them.")
            else:
                _, train_idxs = states_from_manifest(cfg["split_file"], "train")
                print(f"[INFO] demo seeding restricted to {len(train_idxs)} TRAIN episodes of "
                      f"{cfg['split_file']} -- this does NOT hold out the geometric splits")
            # the SAME crop_span the live env draws with, or the demo half of the buffer
            # would not match the space it is stored in -- see obs_from_zarr
            tr = demo_transitions(DEMO_ZARR, obs_type=cfg["obs"], gamma=cfg["gamma"],
                                  tau_ladder=cfg["tau_ladder"], frac=cfg["demo_seed_frac"],
                                  crop_span=(aug or {}).get("crop_span"),
                                  episode_idxs=train_idxs)
            print(summarise(tr))
            preload_demos(agent.replay_buffer, tr)

    callbacks = [
        CheckpointCallback(save_freq=max(1, cfg["save_freq"] // cfg["num_envs"]),
                           save_path=log_dir, name_prefix="model", verbose=2),
        LogEveryNTimesteps(n_steps=cfg["log_interval"]),
        LadderStats(cfg["tau_ladder"], freq=cfg["log_interval"]),
        CurriculumAnneal(cfg["block_goal_offset"], cfg["block_goal_offset_final"],
                         cfg["total_timesteps"], cfg["curriculum_anneal_frac"],
                         cfg["block_near_goal_prob"], cfg["block_near_goal_prob_final"]),
        FixedEval(eval_env, cfg["eval_freq"], cfg["n_eval_episodes"], cfg["seed"] + 10_000),
        QSpreadProbe(eval_env, freq=cfg["bon_probe_freq"], seed=cfg["seed"] + 20_000),
    ]

    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(total_timesteps=cfg["total_timesteps"], callback=callbacks,
                    log_interval=None, reset_num_timesteps=not resuming)
    agent.save(os.path.join(log_dir, "model"))
    print(f"[INFO] Saved: {os.path.join(log_dir, 'model.zip')}")
    env.close()
    eval_env.close()
    if run is not None:
        run.finish()
    return log_dir


if __name__ == "__main__":
    train(parse_train_args())
