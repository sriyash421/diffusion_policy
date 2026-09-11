"""Train a recurrent PPO policy and value function on PushT.

Two observation arms, `--obs keypoint` (block T keypoints + agent xy, plus their visibility
mask) and `--obs image`, and two noise regimes: clean by default, and `--corrupt-obs` for ST's
flat DDPM corruption of the encoded observation. Everything env-side lives in pusht_gym.py, the
corruption in corrupt_policy.py, the run-directory bookkeeping in run_io.py.

    # phase 1 -- clean
    python recurrent_ppo/train.py --obs keypoint
    # phase 2 -- noised observations
    python recurrent_ppo/train.py --obs keypoint --corrupt-obs

Evaluation is always CLEAN, in both arms, so every reported number is a clean-observation
number and the arms differ only in how they trained.
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Train a RecurrentPPO agent on PushT.")
# environment
parser.add_argument("--obs", type=str, default="keypoint", choices=["keypoint", "image"], help="Observation type.")
parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments.")
parser.add_argument("--max-episode-steps", type=int, default=300, help="Episode truncation length.")
parser.add_argument("--render-size", type=int, default=96, help="Render size; also the image obs resolution.")
parser.add_argument("--keypoint-visible-rate", type=float, default=1.0,
                    help="Fraction of keypoints visible per step (keypoint obs). Occluded entries are zeroed "
                         "and flagged in the observation's mask half.")
parser.add_argument("--occlusion", type=str, default="iid", choices=["iid", "persistent"],
                    help="How occlusion is distributed in TIME at the same visibility rate. iid redraws every "
                         "step (mean hidden run 1/rate, ~2 steps); persistent hides a keypoint for a stretch, "
                         "which is the partial observability recurrence exists for.")
parser.add_argument("--occlusion-persistence", type=float, default=20.0,
                    help="Mean hidden run in steps under --occlusion persistent.")
parser.add_argument("--reward", type=str, default="dense", choices=["dense", "sparse"],
                    help="dense: coverage/0.95 every step, episode always runs to the horizon. sparse: +1 on "
                         "the first solve, then terminate. Termination is tied to the mode -- terminating "
                         "under dense would forfeit a reward stream worth more than the whole approach.")
parser.add_argument("--action-mode", type=str, default="delta", choices=["delta", "absolute"],
                    help="delta: the action is an offset from the agent's current position. absolute: it is a "
                         "target anywhere in the arena, which makes std=1 explore over half the table.")
parser.add_argument("--delta-scale", type=str, default="auto",
                    help="Pixels moved per axis at |a|=1 in delta mode, or 'auto' to measure it from the "
                         "demonstrations. The resolved number is recorded in params/args.yaml.")
parser.add_argument("--delta-percentile", type=float, default=99.0,
                    help="Percentile of the demo per-axis step that --delta-scale auto resolves to.")
parser.add_argument("--demo-zarr", type=str, default="data/pusht_cchi_v7_replay.zarr",
                    help="Demonstrations --delta-scale auto measures.")
parser.add_argument("--agent-start-range", type=float, nargs=2, default=[50.0, 450.0], metavar=("LO", "HI"),
                    help="Uniform range each agent start coordinate is drawn from. PushTEnv's own default.")
parser.add_argument("--block-start-range", type=float, nargs=2, default=[60.0, 490.0], metavar=("LO", "HI"),
                    help="Uniform range each block start coordinate is drawn from. Wider than PushTEnv's "
                         "[100,400], which excludes a quarter of the demonstrated block starts.")
parser.add_argument("--agent-near-block-prob", type=float, default=0.0,
                    help="Fraction of episodes that start with the agent a short gap from the block. The dense "
                         "reward depends on the BLOCK pose alone, so with a uniform start the return is fixed "
                         "at reset until contact happens by chance -- measured at 1.5%% of steps. Opt-in: this "
                         "is a curriculum choice, not a bug fix.")
parser.add_argument("--agent-block-gap", type=float, nargs=2, default=[20.0, 80.0], metavar=("LO", "HI"),
                    help="Clear distance from the block's surface for those starts, in px.")
parser.add_argument("--block-near-goal-prob", type=float, default=0.0,
                    help="Fraction of episodes that start with the block part-way to the goal. A reverse "
                         "curriculum: with a uniform block start the first 2M-step run never once crossed the "
                         "success threshold, so the agent never saw what solving pays. Opt-in.")
parser.add_argument("--block-goal-offset", type=float, nargs=2, default=[30.0, 0.25], metavar=("PX", "RAD"),
                    help="Max position and angle offset from the goal pose for those starts. The default "
                         "leaves mean coverage 0.50 (p90 0.74) -- clearly unsolved, but reachable.")
parser.add_argument("--dummy-vec-env", action="store_true", default=False, help="Run envs in-process (debugging).")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment and the agent.")
# observation corruption
parser.add_argument("--corrupt-obs", action="store_true", default=False, help="Noise the encoded obs (ST flat arm).")
parser.add_argument("--corrupt-t-max", type=int, default=200,
                    help="Corruption timesteps are drawn from U[0, this). 200 was chosen from the VAE render "
                         "gate: t=100 is the edge at which the block's orientation stops being readable, so "
                         "the median draw sits on that edge. The old U[0,1000) put 90%% of draws past it.")
# agent
parser.add_argument("--total-timesteps", type=int, default=2_000_000, help="Total environment steps to train for.")
parser.add_argument("--n-steps", type=int, default=128, help="Rollout length per environment.")
parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size.")
parser.add_argument("--n-epochs", type=int, default=10, help="Optimisation epochs per rollout.")
parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate.")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clipping range.")
parser.add_argument("--ent-coef", type=float, default=0.01,
                    help="Entropy bonus. Default 0.01, not SB3's 0.0: measured on the first 2M-step run, the "
                         "action std collapsed monotonically 0.365 -> 0.108 while the task was never solved, "
                         "so nothing resisted the entropy decay.")
parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coefficient.")
parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm.")
parser.add_argument("--log-std-init", type=float, default=-1.0,
                    help="Initial log std of the action distribution. Under --action-mode delta, 0.0 means one "
                         "std is --delta-scale pixels. Default -1.0, measured: at 0.0 32%% of action "
                         "components clip against the box, at -1.0 0.7%%.")
parser.add_argument("--net-arch", type=str, default="128,128", help="Hidden sizes after the LSTM, comma separated.")
parser.add_argument("--lstm-hidden-size", type=int, default=128, help="LSTM hidden size.")
parser.add_argument("--n-lstm-layers", type=int, default=1, help="Number of LSTM layers.")
parser.add_argument("--shared-lstm", action="store_true", default=False, help="Share one LSTM between actor and critic.")
parser.add_argument("--no-norm-reward", dest="norm_reward", action="store_false", default=True,
                    help="Disable reward normalisation. On by default: dense returns reach ~274 undiscounted, "
                         "and vf_coef puts that raw scale straight into the loss.")
parser.add_argument("--target-kl", type=float, default=None, help="Stop the update early past this KL (SB3 default: off).")
parser.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "linear"],
                    help="Linear decays the learning rate to 0 over training. SB3's default is constant; "
                         "annealing is an arm to run, not a fix to apply.")
# bookkeeping
parser.add_argument("--wandb", action="store_true", default=False, help="Log to Weights & Biases.")
parser.add_argument("--wandb-entity", type=str, default="l2sml", help="W&B entity.")
parser.add_argument("--wandb-project", type=str, default="recurrent_ppo", help="W&B project.")
parser.add_argument("--wandb-group", type=str, default=None, help="W&B group, the closest thing to a folder.")
parser.add_argument("--wandb-tags", type=str, nargs="*", default=[],
                    help="Extra W&B tags. The arm's own labels (obs type, reward, corruption, occlusion) are "
                         "DERIVED and always added, so a tag cannot disagree with the run it names.")
parser.add_argument("--log-dir", type=str, default=None, help="Log directory. Default: logs/recurrent_ppo/<arm>/<time>.")
parser.add_argument("--log-interval", type=int, default=10_000, help="Log data every n timesteps.")
parser.add_argument("--save-freq", type=int, default=100_000, help="Checkpoint every n timesteps.")
parser.add_argument("--eval-freq", type=int, default=0, help="Evaluate every n timesteps (0 disables).")
parser.add_argument("--n-eval-episodes", type=int, default=20, help="Episodes per evaluation.")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue training from a checkpoint, in its own run directory.")
parser.add_argument("--device", type=str, default="auto", help="Torch device.")
args_cli = parser.parse_args()

"""Rest everything follows."""

import contextlib
from datetime import datetime

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize

import torch as th
from stable_baselines3.common.utils import explained_variance

from recurrent_ppo.corrupt_policy import (corrupting_extractor, features_extractor_kwargs,
                                          policy_for)
from recurrent_ppo.pusht_gym import build_vec_env, delta_scale_from_demos, env_kwargs_from
from recurrent_ppo.run_io import check_conflicts, dump_args, load_args, vecnormalize_path_for

# the training hyperparameters a checkpoint carries: RecurrentPPO.load rebuilds the agent from
# them, so a CLI value given on a resume would be silently ignored rather than applied
TRAINING_HPARAMS = ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma",
                    "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm")



class ActionDiagnostics(BaseCallback):
    """Log how much of the sampled action mass falls outside the action space.

    PPO stores the UNCLIPPED Gaussian samples in the rollout buffer and clips only on its way
    into the env, so the buffer is the one place the raw exploration distribution is visible.
    These three series are what decides whether the Gaussian needs replacing with a Beta:
    corner_frac staying high while returns plateau is the symptom, and nothing else reports it.
    """

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        actions = self.model.rollout_buffer.actions
        low, high = self.model.action_space.low, self.model.action_space.high
        outside = (actions < low) | (actions > high)
        self.logger.record("rollout/action_clip_frac", float(outside.mean()))
        self.logger.record("rollout/action_corner_frac", float(outside.all(axis=-1).mean()))
        self.logger.record("rollout/action_abs_mean", float(np.abs(actions).mean()))


class QHead(BaseCallback):
    """Regress Q(history, a) on the returns PPO fits V to -- alongside the update, never inside it.

    Runs at rollout end, before `train()`, so the buffer is full and untouched. Everything upstream
    of the head is detached inside `q_values`, and the head carries its own optimizer, so this
    cannot perturb the policy gradient. `q_explained_variance` is the honest read on it: the Q is
    on-policy, so it is only trustworthy near the behaviour policy.
    """

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        policy = self.model.policy
        if not hasattr(policy, "q_net"):
            return
        losses, preds, targets = [], [], []
        for data in self.model.rollout_buffer.get(self.model.batch_size):
            mask = data.mask > 1e-8
            q = policy.q_values(data.observations, data.actions, data.lstm_states.vf, data.episode_starts)
            loss = th.nn.functional.mse_loss(q[mask], data.returns[mask])
            policy.q_optimizer.zero_grad()
            loss.backward()
            policy.q_optimizer.step()
            losses.append(loss.item())
            preds.append(q[mask].detach())
            targets.append(data.returns[mask])
        self.logger.record("train/q_loss", float(np.mean(losses)))
        self.logger.record("train/q_explained_variance",
                           float(explained_variance(th.cat(preds).cpu().numpy(), th.cat(targets).cpu().numpy())))


class CleanEvalCallback(EvalCallback):
    """EvalCallback that evaluates with observation corruption switched OFF.

    The evaluation shares the training policy object, so corruption cannot be a construction-time
    choice here -- it has to be toggled around the rollout. Clean is the convention: every
    reported number is then a clean-observation number and the arms differ only in how they
    trained. play.py's --corrupt-obs-eval opts back in.
    """

    def _on_step(self):
        extractor = corrupting_extractor(self.model.policy)
        was_on = extractor is not None and extractor.enabled
        if was_on:
            extractor.enabled = False
        try:
            return super()._on_step()
        finally:
            if was_on:
                extractor.enabled = True

def arm_tags(cfg):
    """The labels that describe what this run IS, derived from the config rather than typed.

    pusht_base.yaml keeps `obs_noise_tag: {clean, obs_noised}` by hand and has a startup check
    (`_check_obs_noise_labels`) precisely because hand-set tags drifted from the arm they named.
    Deriving them removes the failure instead of checking for it, and keeps the same vocabulary
    so these runs read like the diffusion-policy ones in the same workspace.
    """
    tags = [
        f"obs-{cfg['obs']}",
        "obs_noised" if cfg["corrupt_obs"] else "clean",
        f"reward-{cfg['reward']}",
        f"action-{cfg['action_mode']}",
    ]
    if cfg["agent_near_block_prob"] > 0:
        tags.append("init-near-block")
    if cfg["block_near_goal_prob"] > 0:
        tags.append("init-near-goal")
    if cfg["obs"] == "keypoint" and cfg["keypoint_visible_rate"] < 1.0:
        tags.append(f"occl-{cfg['occlusion']}")
    return tags


def main():
    """Train with a sb3-contrib recurrent agent."""
    cfg = vars(args_cli)
    resuming = args_cli.checkpoint is not None

    # resolve --delta-scale BEFORE anything records or compares it, so params/args.yaml holds the
    # number the envs were actually built with and play.py never has to re-measure it
    if cfg["delta_scale"] == "auto":
        cfg["delta_scale"] = delta_scale_from_demos(args_cli.demo_zarr, args_cli.delta_percentile)
        print(f"[INFO] --delta-scale auto -> {cfg['delta_scale']:.1f} px "
              f"(p{args_cli.delta_percentile:g} of the demo per-axis step)")
    else:
        cfg["delta_scale"] = float(cfg["delta_scale"])

    if resuming:
        # continue the run the checkpoint belongs to, rather than opening a new directory that
        # would claim to be a fresh experiment
        log_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
        arm = f"{args_cli.obs}{'_corrupt' if args_cli.corrupt_obs else '_clean'}"
        check_conflicts(load_args(log_dir), cfg)
        print(f"[INFO] Resuming run in: {log_dir}")
    else:
        # --corrupt-obs is part of the run IDENTITY, not just a knob: two noise regimes sharing
        # a directory would overwrite each other's checkpoints.
        arm = f"{args_cli.obs}{'_corrupt' if args_cli.corrupt_obs else '_clean'}"
        run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.abspath(args_cli.log_dir or os.path.join("logs", "recurrent_ppo", arm, run_info))
        os.makedirs(log_dir, exist_ok=True)
        dump_args(log_dir, cfg)
        print(f"[INFO] Logging experiment in directory: {log_dir}")

    with open(os.path.join(log_dir, "command.txt"), "a") as f:
        f.write(" ".join([sys.executable] + sys.argv) + "\n")

    run = None
    if args_cli.wandb:
        import wandb

        run = wandb.init(
            entity=args_cli.wandb_entity,
            project=args_cli.wandb_project,
            group=args_cli.wandb_group,
            name=f"{os.path.basename(log_dir)}_{arm}" if not resuming else os.path.basename(log_dir),
            tags=sorted(set(args_cli.wandb_tags) | set(arm_tags(cfg))),
            config=cfg,
            # SB3 already writes these metrics to tensorboard; let wandb mirror that rather than
            # instrumenting the training loop a second time
            sync_tensorboard=True,
            dir=log_dir,
            resume="allow" if resuming else None,
        )
        print(f"[INFO] W&B: {run.url}")
        print(f"[INFO] W&B tags: {sorted(set(args_cli.wandb_tags) | set(arm_tags(cfg)))}")

    env_kwargs = env_kwargs_from(cfg, args_cli.obs)

    # create the vectorised pushT environment
    env = build_vec_env(
        obs_type=args_cli.obs,
        n_envs=args_cli.num_envs,
        seed=args_cli.seed,
        use_subproc=not args_cli.dummy_vec_env,
        monitor_dir=log_dir,
        **env_kwargs,
    )

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

    # create agent from sb3-contrib
    if resuming:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        agent = RecurrentPPO.load(args_cli.checkpoint, env, print_system_info=True, device=args_cli.device)
        saved = load_args(log_dir) or {}
        ignored = [k for k in TRAINING_HPARAMS
                   if k in saved and saved[k] != cfg[k]]
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
        net_arch = [int(x) for x in args_cli.net_arch.split(",") if x]
        policy_kwargs = {
            "net_arch": dict(pi=net_arch, vf=net_arch),
            "lstm_hidden_size": args_cli.lstm_hidden_size,
            "n_lstm_layers": args_cli.n_lstm_layers,
            # a shared LSTM and a critic LSTM are mutually exclusive in sb3-contrib
            "shared_lstm": args_cli.shared_lstm,
            "enable_critic_lstm": not args_cli.shared_lstm,
            "log_std_init": args_cli.log_std_init,
            # corruption arrives as a features extractor, through SB3's own extension point
            **features_extractor_kwargs(args_cli.obs, args_cli.corrupt_obs, args_cli.corrupt_t_max),
        }
        lr = args_cli.learning_rate
        learning_rate = (lambda progress: progress * lr) if args_cli.lr_schedule == "linear" else lr
        agent = RecurrentPPO(
            policy_for(args_cli.obs),
            env,
            n_steps=args_cli.n_steps,
            batch_size=args_cli.batch_size,
            n_epochs=args_cli.n_epochs,
            learning_rate=learning_rate,
            gamma=args_cli.gamma,
            gae_lambda=args_cli.gae_lambda,
            clip_range=args_cli.clip_range,
            ent_coef=args_cli.ent_coef,
            vf_coef=args_cli.vf_coef,
            max_grad_norm=args_cli.max_grad_norm,
            target_kl=args_cli.target_kl,
            policy_kwargs=policy_kwargs,
            seed=args_cli.seed,
            device=args_cli.device,
            verbose=1,
            tensorboard_log=log_dir,
        )

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
    if args_cli.eval_freq > 0:
        # Evaluation always uses the REAL start distribution, even when training uses the
        # near-block curriculum: a curriculum is a way to learn the task, not a redefinition of
        # it, and evaluating on it would make the number incomparable with a uniform-start run.
        eval_env_kwargs = dict(env_kwargs, agent_near_block_prob=0.0, block_near_goal_prob=0.0)
        eval_env = build_vec_env(
            obs_type=args_cli.obs,
            n_envs=1,
            # a disjoint seed block, so the evaluation episodes are not the training ones
            seed=args_cli.seed + 10_000,
            use_subproc=False,
            **eval_env_kwargs,
        )
        if isinstance(env, VecNormalize):
            # EvalCallback syncs the statistics across, which requires both sides to be wrapped
            eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=False,
                                    gamma=args_cli.gamma, clip_reward=np.inf)
        callbacks.append(CleanEvalCallback(
            eval_env,
            n_eval_episodes=args_cli.n_eval_episodes,
            eval_freq=max(args_cli.eval_freq // args_cli.num_envs, 1),
            log_path=log_dir,
            best_model_save_path=log_dir,
            deterministic=True,
            verbose=1,
        ))

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


if __name__ == "__main__":
    main()
