"""The training and evaluation loops, shared by both architectures.

This is the body that used to be train.py's and play.py's `main()`. It was moved here when the
frame-stacking arm was added, because the two arms exist to be COMPARED: anything duplicated
between them -- the reward, the curriculum, the evaluation protocol, the seed offsets, the run
bookkeeping -- is a place where the comparison can quietly stop being like-for-like. Everything
architecture-specific is behind the `arch` object (see arch.py), which has four methods.
"""

import contextlib
import os
import sys
from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize

from recurrent_ppo.callbacks import (ActionDiagnostics, CleanEvalCallback, QHead, RolloutVideo,
                                     SecondEval, arm_tags)
from recurrent_ppo.config import DEFAULTS, TRAINING_HPARAMS
from recurrent_ppo.corrupt_policy import corrupting_extractor, set_corruption
from recurrent_ppo.pusht_gym import build_vec_env, delta_scale_from_demos, env_kwargs_from
from recurrent_ppo.run_io import (check_conflicts, dump_args, get_checkpoint_path, load_args,
                                  vecnormalize_path_for)


def train(args_cli, arch):
    """Train one arm. `arch` is LstmArch or StackArch."""
    cfg = vars(args_cli)
    cfg["arch"] = arch.name
    resuming = args_cli.checkpoint is not None

    # resolve --delta-scale BEFORE anything records or compares it, so params/args.yaml holds the
    # number the envs were actually built with and play.py never has to re-measure it
    if cfg["delta_scale"] == "auto":
        cfg["delta_scale"] = delta_scale_from_demos(args_cli.demo_zarr, args_cli.delta_percentile)
        print(f"[INFO] --delta-scale auto -> {cfg['delta_scale']:.1f} px "
              f"(p{args_cli.delta_percentile:g} of the demo per-axis step)")
    else:
        cfg["delta_scale"] = float(cfg["delta_scale"])

    arm = f"{args_cli.obs}{'_corrupt' if args_cli.corrupt_obs else '_clean'}"
    if resuming:
        # continue the run the checkpoint belongs to, rather than opening a new directory that
        # would claim to be a fresh experiment
        log_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
        check_conflicts(load_args(log_dir), cfg)
        print(f"[INFO] Resuming run in: {log_dir}")
    else:
        # --corrupt-obs is part of the run IDENTITY, not just a knob: two noise regimes sharing
        # a directory would overwrite each other's checkpoints. So is the architecture, which is
        # why each one has its own log root.
        run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.abspath(args_cli.log_dir or os.path.join("logs", arch.log_root, arm, run_info))
        os.makedirs(log_dir, exist_ok=True)
        dump_args(log_dir, cfg)
        print(f"[INFO] Logging experiment in directory: {log_dir}")

    with open(os.path.join(log_dir, "command.txt"), "a") as f:
        f.write(" ".join([sys.executable] + sys.argv) + "\n")

    run = None
    if args_cli.wandb:
        import wandb

        tags = sorted(set(args_cli.wandb_tags) | set(arm_tags(cfg)))
        run = wandb.init(
            entity=args_cli.wandb_entity,
            project=args_cli.wandb_project,
            group=args_cli.wandb_group,
            name=f"{os.path.basename(log_dir)}_{arch.name}_{arm}" if not resuming else os.path.basename(log_dir),
            tags=tags,
            config=cfg,
            # SB3 already writes these metrics to tensorboard; let wandb mirror that rather than
            # instrumenting the training loop a second time
            sync_tensorboard=True,
            dir=log_dir,
            resume="allow" if resuming else None,
        )
        print(f"[INFO] W&B: {run.url}")
        print(f"[INFO] W&B tags: {tags}")

    env_kwargs = env_kwargs_from(cfg, args_cli.obs)

    # create the vectorised pushT environment
    env = arch.wrap(build_vec_env(
        obs_type=args_cli.obs,
        n_envs=args_cli.num_envs,
        seed=args_cli.seed,
        use_subproc=not args_cli.dummy_vec_env,
        monitor_dir=log_dir,
        **env_kwargs,
    ), cfg)

    # obs are already scaled to [-1, 1] against known bounds, so only the reward is ever normalised
    saved_vecnorm = vecnormalize_path_for(args_cli.checkpoint) if resuming else None
    if saved_vecnorm is not None and saved_vecnorm.exists():
        print(f"[INFO] Loading saved normalization: {saved_vecnorm}")
        env = VecNormalize.load(str(saved_vecnorm), env)
        env.training = True
        env.norm_reward = args_cli.norm_reward
    elif args_cli.norm_reward:
        print("[INFO] Normalizing reward.")
        env = VecNormalize(env, training=True, norm_obs=False, norm_reward=True,
                           gamma=args_cli.gamma, clip_reward=np.inf)

    if resuming:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        agent = arch.load(args_cli.checkpoint, env, cfg)
        saved = load_args(log_dir) or {}
        ignored = [k for k in TRAINING_HPARAMS if k in saved and saved[k] != cfg[k]]
        if ignored:
            print("[WARN] These come from the checkpoint, not the command line, and your values "
                  "are being ignored:\n" + "\n".join(
                      f"  {k}: using {saved[k]!r}, you passed {cfg[k]!r}" for k in ignored))
        ckpt_corrupt = corrupting_extractor(agent.policy) is not None
        if ckpt_corrupt != args_cli.corrupt_obs:
            raise SystemExit(
                f"[ERROR] The checkpoint trained with corrupt_obs={ckpt_corrupt}, but "
                f"--corrupt-obs was {'given' if args_cli.corrupt_obs else 'not given'}. That is the "
                "run's identity and cannot change on a resume."
            )
    else:
        agent = arch.build(env, cfg, log_dir)

    # callbacks for agent. save_freq counts agent steps, of which each is num_envs env steps.
    callbacks = [
        CheckpointCallback(
            save_freq=max(args_cli.save_freq // args_cli.num_envs, 1),
            save_path=log_dir,
            name_prefix="model",
            save_vecnormalize=isinstance(env, VecNormalize),
            verbose=2,
        ),
        LogEveryNTimesteps(n_steps=args_cli.log_interval),
        ActionDiagnostics(),
        QHead(),
    ]
    if args_cli.video_freq > 0:
        if run is None:
            raise SystemExit("[ERROR] --video-freq needs --wandb: the video has nowhere else to go.")
        callbacks.append(RolloutVideo(env_kwargs, args_cli.obs, args_cli.video_freq,
                                      args_cli.video_length, args_cli.seed + 20_000,
                                      episodes=args_cli.video_episodes,
                                      stride=args_cli.video_stride,
                                      n_stack=arch.video_n_stack(cfg)))
    if args_cli.eval_freq > 0:
        # Two evaluations, on the two start distributions, so neither question is answered by
        # the other's number: `match` is the one the policy trained on, `off` is the real task.
        real_kwargs = dict(env_kwargs, agent_near_block_prob=0.0, block_near_goal_prob=0.0)
        matched = args_cli.eval_curriculum == "match"
        eval_env_kwargs = dict(env_kwargs) if matched else real_kwargs
        second_kwargs = real_kwargs if matched else dict(env_kwargs)
        second_prefix = "eval_real" if matched else "eval_train"
        eval_freq = max(args_cli.eval_freq // args_cli.num_envs, 1)
        eval_env = arch.wrap(build_vec_env(
            obs_type=args_cli.obs,
            n_envs=1,
            # a disjoint seed block, so the evaluation episodes are not the training ones
            seed=args_cli.seed + 10_000,
            use_subproc=False,
            **eval_env_kwargs,
        ), cfg)
        if isinstance(env, VecNormalize):
            # EvalCallback syncs the statistics across, which requires both sides to be wrapped
            eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=False,
                                    gamma=args_cli.gamma, clip_reward=np.inf)
        callbacks.append(CleanEvalCallback(
            eval_env,
            n_eval_episodes=args_cli.n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_dir,
            best_model_save_path=log_dir,
            deterministic=True,
            verbose=1,
        ))
        second_env = arch.wrap(build_vec_env(obs_type=args_cli.obs, n_envs=1, seed=args_cli.seed + 30_000,
                                             use_subproc=False, **second_kwargs), cfg)
        if isinstance(env, VecNormalize):
            second_env = VecNormalize(second_env, training=False, norm_obs=False, norm_reward=False,
                                      gamma=args_cli.gamma, clip_reward=np.inf)
        callbacks.append(SecondEval(second_env, second_prefix, eval_freq * args_cli.num_envs,
                                    args_cli.n_eval_episodes))

    # train the agent
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=args_cli.total_timesteps,
            callback=callbacks,
            log_interval=None,
            # a resume continues the step counter; restarting it would overwrite the earlier
            # checkpoints and restart the tensorboard curve at zero
            reset_num_timesteps=not resuming,
        )

    # save the final model
    agent.save(os.path.join(log_dir, "model"))
    print("Saving to:")
    print(os.path.join(log_dir, "model.zip"))

    if isinstance(env, VecNormalize):
        print("Saving normalization")
        env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))

    env.close()
    if run is not None:
        run.finish()


# how the env was shaped. A key is overridable only if the entry point exposes a flag for it;
# getattr's default keeps the two from having to be kept in step by hand.
RESOLVE_KEYS = ("obs", "max_episode_steps", "render_size", "keypoint_visible_rate",
                "occlusion", "occlusion_persistence", "reward", "shaping_coef",
                "shaping_potential", "progress_coef", "success_bonus", "block_zero_coverage",
                "action_mode", "delta_scale", "agent_start_range", "block_start_range",
                "agent_near_block_prob", "agent_block_gap", "block_near_goal_prob",
                "block_goal_offset", "gamma")


def resolve_config(run_dir, args_cli, arch):
    """The run's recorded environment configuration, with any CLI flag overriding it out loud."""
    keys = RESOLVE_KEYS + arch.resolve_keys
    saved = load_args(run_dir) or {}
    if not saved:
        print(f"[WARN] No params/args.yaml in {run_dir}; falling back to defaults, which may not "
              "be what this checkpoint trained on.")
    missing = [k for k in keys if k not in saved]
    if missing:
        print(f"[WARN] This run predates {len(missing)} key(s); assuming today's default for each. "
              f"See recurrent_ppo_runs_sep11.md for what it actually used.")
        for k in missing:
            print(f"         {k} = {DEFAULTS[k]!r}")
    cfg = {}
    for key in keys:
        override = getattr(args_cli, key, None)
        recorded = saved.get(key, DEFAULTS[key])
        if override is not None and override != recorded:
            print(f"[INFO] Overriding {key}: run recorded {recorded!r}, using {override!r}")
        cfg[key] = recorded if override is None else override
    cfg["device"] = args_cli.device
    return cfg


def play(args_cli, arch):
    """Evaluate a checkpoint of one arm, and optionally record it."""
    if args_cli.checkpoint is None:
        log_root_path = os.path.abspath(os.path.join(
            "logs", arch.log_root, f"{args_cli.obs or 'keypoint'}_{args_cli.arm}"))
        checkpoint_path = get_checkpoint_path(log_root_path, args_cli.use_last_checkpoint)
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(os.path.abspath(checkpoint_path))

    cfg = resolve_config(log_dir, args_cli, arch)
    env_kwargs = env_kwargs_from(cfg, cfg["obs"])

    # video frames come off one env in this process, so it forces a single in-process env
    num_envs = args_cli.num_envs
    if args_cli.video:
        num_envs = 1
        env_kwargs["render_mode"] = "rgb_array"
        print("[INFO] --video forces a single in-process environment.")
    env = arch.wrap(build_vec_env(
        obs_type=cfg["obs"],
        n_envs=num_envs,
        seed=args_cli.seed,
        use_subproc=not args_cli.video,
        **env_kwargs,
    ), cfg)

    # normalize environment (if needed)
    vec_norm_path = vecnormalize_path_for(checkpoint_path)
    if vec_norm_path.exists():
        print(f"Loading saved normalization: {vec_norm_path}")
        env = VecNormalize.load(str(vec_norm_path), env)
        #  do not update them at test time
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False

    print(f"Loading checkpoint from: {checkpoint_path}")
    agent = arch.load(checkpoint_path, env, cfg)
    # the policy was BUILT with whatever corruption it trained under; this is the eval question
    set_corruption(agent.policy, args_cli.corrupt_obs_eval)
    extractor = corrupting_extractor(agent.policy)
    print(f"[INFO] Observation corruption during this rollout: "
          f"{'None (clean checkpoint)' if extractor is None else extractor.enabled}")

    # A per-env budget, not a global count. Stopping at the first N episodes across all envs
    # over-samples the SHORT ones, and success is exactly what ends an episode early -- so a
    # global count reports a success rate biased upward.
    budget = [(args_cli.n_episodes + i) // num_envs for i in range(num_envs)]
    done_count = [0] * num_envs
    successes, scores, lengths, frames = [], [], [], []

    obs = env.reset()
    # the LSTM hidden state and the mask that zeroes it on a new episode. A non-recurrent policy
    # accepts both and ignores them, so the loop is the same for either architecture.
    states = None
    episode_starts = np.ones((num_envs,), dtype=bool)
    while any(d < b for d, b in zip(done_count, budget)):
        actions, states = agent.predict(
            obs,
            state=states,
            episode_start=episode_starts,
            deterministic=not args_cli.stochastic,
        )
        # BEFORE the step: a VecEnv auto-resets a finished env inside step(), so a frame taken
        # afterwards would already show the next episode
        if args_cli.video and len(frames) < args_cli.video_length:
            frames.append(env.render("rgb_array"))

        obs, _, dones, infos = env.step(actions)
        # a done env is reset by the VecEnv, so its NEXT observation starts a new episode
        episode_starts = dones

        for i, (done, info) in enumerate(zip(dones, infos)):
            if not done or done_count[i] >= budget[i]:
                continue
            done_count[i] += 1
            successes.append(bool(info["is_success"]))
            scores.append(float(info["max_reward"]))
            lengths.append(int(info["episode"]["l"]))

    print(f"\nEpisodes:     {len(successes)} ({budget[0]}-{budget[-1]} per env over {num_envs} envs)")
    print(f"Success rate: {np.mean(successes):.3f}")
    print(f"Score (max normalised coverage): {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
    print(f"Episode length: {np.mean(lengths):.1f}")

    if args_cli.video:
        import imageio

        video_path = os.path.join(log_dir, "videos", "play.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, frames, fps=10)
        print(f"Saved video to: {video_path}")

    env.close()
