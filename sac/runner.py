"""The training and evaluation loops.

Mirrors recurrent_ppo/runner.py, including `_claim` and the args.yaml bookkeeping, which are
imported rather than re-implemented: a run directory is the record of what a run WAS, and two
copies of that logic is how a resume silently continues one experiment under another's name.
"""

import contextlib
import os
import sys
from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps

from recurrent_ppo.run_io import check_conflicts, dump_args, get_checkpoint_path, load_args
from recurrent_ppo.runner import _claim
from sac.buffers import buffer_class_for, preload_demos
from sac.callbacks import CurriculumAnneal, FixedEval, LadderStats, QSpreadProbe, arm_tags
from sac.chunk_codec import assert_roundtrip, demo_chunks
from sac.chunk_env import build_chunk_vec_env, env_kwargs_from
from sac.config import DEFAULTS, DEMO_ZARR, IDENTITY_KEYS, buffer_size_for
from sac.demo_buffer import demo_transitions, summarise
from sac.sac import ChunkSAC

LOG_ROOT = "sac"


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
    env = build_chunk_vec_env(obs_type=cfg["obs"], n_envs=cfg["num_envs"], seed=cfg["seed"],
                              use_subproc=not cfg["dummy_vec_env"], monitor_dir=log_dir,
                              **env_kwargs)
    # evaluation is on the REAL start distribution, never the curriculum's: the curriculum is
    # scaffolding for learning, not the task being measured.
    eval_kwargs = dict(env_kwargs, block_near_goal_prob=0.0)
    eval_env = build_chunk_vec_env(obs_type=cfg["obs"], n_envs=min(cfg["num_envs"], 8),
                                   seed=cfg["seed"] + 10_000,
                                   use_subproc=not cfg["dummy_vec_env"], **eval_kwargs)

    if resuming:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        agent = ChunkSAC.load(args_cli.checkpoint, env, device=cfg["device"])
    else:
        agent = _build_agent(cfg, env, log_dir, demo_offsets)
        if cfg["demo_seed_frac"] > 0:
            tr = demo_transitions(DEMO_ZARR, obs_type=cfg["obs"], gamma=cfg["gamma"],
                                  tau_ladder=cfg["tau_ladder"], frac=cfg["demo_seed_frac"])
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


def play(args_cli):
    """Evaluate a checkpoint on the fixed episode set, reading how the env was shaped."""
    if args_cli.checkpoint is None:
        root = os.path.abspath(os.path.join("logs", LOG_ROOT, args_cli.obs or DEFAULTS["obs"]))
        checkpoint_path = get_checkpoint_path(root, args_cli.use_last_checkpoint)
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    saved = load_args(log_dir) or {}
    if not saved:
        print(f"[WARN] No params/args.yaml in {log_dir}; falling back to today's defaults, which "
              "may not be what this checkpoint trained on.")
    cfg = dict(DEFAULTS, **saved)
    for key in ("obs", "render_size"):
        override = getattr(args_cli, key, None)
        if override is not None and override != cfg.get(key):
            print(f"[INFO] Overriding {key}: run recorded {cfg.get(key)!r}, using {override!r}")
            cfg[key] = override

    env_kwargs = dict(env_kwargs_from(cfg, cfg["obs"]), block_near_goal_prob=0.0)
    env = build_chunk_vec_env(obs_type=cfg["obs"], n_envs=args_cli.num_envs, seed=args_cli.seed,
                              use_subproc=not args_cli.video, **env_kwargs)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    agent = ChunkSAC.load(checkpoint_path, env, device=args_cli.device)

    env.seed(args_cli.seed)                  # the fixed set: the same episodes every time
    n_envs = env.num_envs
    budget = [(args_cli.n_episodes + i) // n_envs for i in range(n_envs)]
    done_count = [0] * n_envs
    successes, coverage = [], []
    obs = env.reset()
    while any(d < b for d, b in zip(done_count, budget)):
        actions, _ = agent.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(actions)
        for i, (d, info) in enumerate(zip(dones, infos)):
            if d and done_count[i] < budget[i]:
                done_count[i] += 1
                successes.append(float(info.get("is_success", 0.0)))
                coverage.append(float(info.get("max_reward", 0.0)))
    print(f"[RESULT] {len(successes)} episodes  success {np.mean(successes):.3f}  "
          f"max coverage {np.mean(coverage):.3f}")
    env.close()
    return {"success_rate": float(np.mean(successes)), "max_coverage": float(np.mean(coverage))}
