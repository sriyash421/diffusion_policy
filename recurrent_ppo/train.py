"""Train a recurrent PPO policy and value function on PushT.

Two observation arms, `--obs keypoint` (block T keypoints + agent xy, plus their visibility
mask) and `--obs image`, and two noise regimes: clean by default, and `--corrupt-obs` for ST's
flat DDPM corruption of the encoded observation. Everything env-side lives in pusht_gym.py, the
corruption in corrupt_policy.py, the run-directory bookkeeping in run_io.py.

    # phase 1 -- clean
    python -m recurrent_ppo.train --obs keypoint
    # phase 2 -- noised observations
    python -m recurrent_ppo.train --obs keypoint --corrupt-obs

Evaluation is always CLEAN, in both arms, so every reported number is a clean-observation
number and the arms differ only in how they trained.
"""

import argparse
import os
import sys

# the defaults the flags below carry. Imported up here, not with the heavy imports, because
# build_parser() needs it and nothing else at module scope does.
from recurrent_ppo.config import DEFAULTS as D


def build_parser():
    """The CLI, built in a function so that importing this module does not parse sys.argv.
    The argparse-before-heavy-imports layout this file inherited needs no module-level
    parse: there is no simulator that must start before the imports.
    """
    parser = argparse.ArgumentParser(description="Train a RecurrentPPO agent on PushT.")
    # environment
    parser.add_argument("--obs", type=str, default=D["obs"], choices=["keypoint", "image"], help="Observation type.")
    parser.add_argument("--num-envs", type=int, default=D["num_envs"], help="Number of parallel environments.")
    parser.add_argument("--max-episode-steps", type=int, default=D["max_episode_steps"], help="Episode truncation length.")
    parser.add_argument("--render-size", type=int, default=D["render_size"], help="Render size; also the image obs resolution.")
    parser.add_argument("--keypoint-visible-rate", type=float, default=D["keypoint_visible_rate"],
                        help="Fraction of keypoints visible per step (keypoint obs). Occluded entries are zeroed "
                             "and flagged in the observation's mask half.")
    parser.add_argument("--occlusion", type=str, default=D["occlusion"], choices=["iid", "persistent"],
                        help="How occlusion is distributed in TIME at the same visibility rate. iid redraws every "
                             "step (mean hidden run 1/rate, ~2 steps); persistent hides a keypoint for a stretch, "
                             "which is the partial observability recurrence exists for.")
    parser.add_argument("--occlusion-persistence", type=float, default=D["occlusion_persistence"],
                        help="Mean hidden run in steps under --occlusion persistent.")
    parser.add_argument("--reward", type=str, default=D["reward"], choices=["dense", "sparse", "shaped", "delta"],
                        help="dense: coverage/0.95 every step, episode always runs to the horizon. sparse: +1 on "
                             "the first solve, then terminate. Termination is tied to the mode -- terminating "
                             "under dense would forfeit a reward stream worth more than the whole approach. "
                             "shaped: dense plus potential-based shaping toward the block, which is what gives "
                             "the agent any gradient at all before it makes contact.")
    parser.add_argument("--progress-coef", type=float, default=D["progress_coef"],
                        help="Scale on the --reward delta progress term. The T starts a mean 167px from the goal "
                             "pose (0.326 of the arena), so 30 makes a full solve worth about 10 in progress.")
    parser.add_argument("--success-bonus", type=float, default=D["success_bonus"],
                        help="Paid once, on the first step coverage exceeds 0.95. At the default it is worth about "
                             "as much as the entire approach, so finishing is not a rounding error on progress.")
    parser.add_argument("--allow-goal-overlap", dest="block_zero_coverage", action="store_false",
                        default=D["block_zero_coverage"],
                        help="Allow block starts that already overlap the goal. Rejecting them is the DEFAULT: "
                             "34.5%% of uniform draws overlap, and any level-valued reward pays for that overlap "
                             "every step of the episode, which is how a do-nothing policy scored 92.7.")
    parser.add_argument("--shaping-potential", type=str, default=D["shaping_potential"],
                        choices=["t_goal", "arm_t", "arm"],
                        help="What the shaping potential measures. t_goal: mean per-keypoint distance of the T "
                             "from the goal pose, so it sees rotation as well as position and keeps paying after "
                             "contact. arm: distance to the block, the only term that varies BEFORE contact. "
                             "arm_t: both, which is the repo's own verifier value.")
    parser.add_argument("--shaping-coef", type=float, default=D["shaping_coef"],
                        help="Weight on the shaping term (--reward shaped). At 1.0 a full-arena approach is worth "
                             "~0.5 total, against an episode-return spread of +/-6.7 driven by the random initial "
                             "pose -- the signal would be buried in the noise it has to be told apart from. 10.0 "
                             "puts a full approach at ~5, comparable to that spread. Worth sweeping.")
    parser.add_argument("--action-mode", type=str, default=D["action_mode"], choices=["delta", "absolute"],
                        help="delta: the action is an offset from the agent's current position. absolute: it is a "
                             "target anywhere in the arena, which makes std=1 explore over half the table.")
    parser.add_argument("--delta-scale", type=str, default=D["delta_scale"],
                        help="Pixels moved per axis at |a|=1 in delta mode, or 'auto' to measure it from the "
                             "demonstrations. The resolved number is recorded in params/args.yaml.")
    parser.add_argument("--delta-percentile", type=float, default=D["delta_percentile"],
                        help="Percentile of the demo per-axis step that --delta-scale auto resolves to.")
    parser.add_argument("--demo-zarr", type=str, default=D["demo_zarr"],
                        help="Demonstrations --delta-scale auto measures.")
    parser.add_argument("--agent-start-range", type=float, nargs=2, default=D["agent_start_range"], metavar=("LO", "HI"),
                        help="Uniform range each agent start coordinate is drawn from. PushTEnv's own default.")
    parser.add_argument("--block-start-range", type=float, nargs=2, default=D["block_start_range"], metavar=("LO", "HI"),
                        help="Uniform range each block start coordinate is drawn from. Wider than PushTEnv's "
                             "[100,400], which excludes a quarter of the demonstrated block starts.")
    parser.add_argument("--agent-near-block-prob", type=float, default=D["agent_near_block_prob"],
                        help="Fraction of episodes that start with the agent a short gap from the block. The dense "
                             "reward depends on the BLOCK pose alone, so with a uniform start the return is fixed "
                             "at reset until contact happens by chance -- measured at 1.5%% of steps. Opt-in: this "
                             "is a curriculum choice, not a bug fix.")
    parser.add_argument("--agent-block-gap", type=float, nargs=2, default=D["agent_block_gap"], metavar=("LO", "HI"),
                        help="Distance from the block's SURFACE to the agent CENTRE, so clearance is this minus "
                             "the 15px agent radius. Measured with a random policy: at [20,80] the median episode "
                             "takes 58 steps to make first contact, at [16,25] it takes 3. Below ~15.5 the agent "
                             "can spawn interpenetrating.")
    parser.add_argument("--block-near-goal-prob", type=float, default=D["block_near_goal_prob"],
                        help="Fraction of episodes that start with the block part-way to the goal. A reverse "
                             "curriculum: with a uniform block start the first 2M-step run never once crossed the "
                             "success threshold, so the agent never saw what solving pays. Opt-in.")
    parser.add_argument("--block-goal-offset", type=float, nargs=2, default=D["block_goal_offset"], metavar=("PX", "RAD"),
                        help="Max position and angle offset from the goal pose for those starts. The default "
                             "leaves mean coverage 0.50 (p90 0.74) -- clearly unsolved, but reachable.")
    parser.add_argument("--dummy-vec-env", action="store_true", default=D["dummy_vec_env"], help="Run envs in-process (debugging).")
    parser.add_argument("--seed", type=int, default=D["seed"], help="Seed used for the environment and the agent.")
    # observation corruption
    parser.add_argument("--corrupt-obs", action="store_true", default=D["corrupt_obs"], help="Noise the encoded obs (ST flat arm).")
    parser.add_argument("--corrupt-t-max", type=int, default=D["corrupt_t_max"],
                        help="Corruption timesteps are drawn from U[0, this). 200 was chosen from the VAE render "
                             "gate: t=100 is the edge at which the block's orientation stops being readable, so "
                             "the median draw sits on that edge. The old U[0,1000) put 90%% of draws past it.")
    # agent
    parser.add_argument("--total-timesteps", type=int, default=D["total_timesteps"], help="Total environment steps to train for.")
    parser.add_argument("--n-steps", type=int, default=D["n_steps"], help="Rollout length per environment.")
    parser.add_argument("--batch-size", type=int, default=D["batch_size"], help="Minibatch size.")
    parser.add_argument("--n-epochs", type=int, default=D["n_epochs"], help="Optimisation epochs per rollout.")
    parser.add_argument("--learning-rate", type=float, default=D["learning_rate"], help="Adam learning rate.")
    parser.add_argument("--gamma", type=float, default=D["gamma"], help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=D["gae_lambda"], help="GAE lambda.")
    parser.add_argument("--clip-range", type=float, default=D["clip_range"], help="PPO clipping range.")
    parser.add_argument("--ent-coef", type=float, default=D["ent_coef"],
                        help="Entropy bonus. Not SB3's 0.0: measured on the first 2M-step run, the action std "
                             "collapsed monotonically 0.365 -> 0.108 while the task was never solved, so nothing "
                             "resisted the entropy decay. 0.0005 is a light touch -- watch train/std early, since "
                             "it may not be enough to hold the entropy up on its own.")
    parser.add_argument("--vf-coef", type=float, default=D["vf_coef"], help="Value loss coefficient.")
    parser.add_argument("--max-grad-norm", type=float, default=D["max_grad_norm"], help="Gradient clipping norm.")
    parser.add_argument("--log-std-init", type=float, default=D["log_std_init"],
                        help="Initial log std of the action distribution. Under --action-mode delta, 0.0 means one "
                             "std is --delta-scale pixels. Default -1.0, measured: at 0.0 32%% of action "
                             "components clip against the box, at -1.0 0.7%%.")
    parser.add_argument("--net-arch", type=str, default=D["net_arch"], help="Hidden sizes after the LSTM, comma separated.")
    parser.add_argument("--q-net-arch", type=str, default=D["q_net_arch"],
                        help="Hidden sizes of the Q head, comma separated. It is a deliverable of the clean "
                             "arm, so it is configurable like everything else that is.")
    parser.add_argument("--q-lr", type=float, default=D["q_lr"], help="Adam learning rate of the Q head.")
    parser.add_argument("--lstm-hidden-size", type=int, default=D["lstm_hidden_size"], help="LSTM hidden size.")
    parser.add_argument("--n-lstm-layers", type=int, default=D["n_lstm_layers"], help="Number of LSTM layers.")
    parser.add_argument("--shared-lstm", action="store_true", default=D["shared_lstm"], help="Share one LSTM between actor and critic.")
    parser.add_argument("--no-norm-reward", dest="norm_reward", action="store_false", default=D["norm_reward"],
                        help="Disable reward normalisation. On by default: dense returns reach ~274 undiscounted, "
                             "and vf_coef puts that raw scale straight into the loss.")
    parser.add_argument("--target-kl", type=float, default=D["target_kl"], help="Stop the update early past this KL (SB3 default: off).")
    parser.add_argument("--lr-schedule", type=str, default=D["lr_schedule"], choices=["constant", "linear"],
                        help="Linear decays the learning rate to 0 over training. SB3's default is constant; "
                             "annealing is an arm to run, not a fix to apply.")
    # bookkeeping
    parser.add_argument("--wandb", action="store_true", default=D["wandb"], help="Log to Weights & Biases.")
    parser.add_argument("--wandb-entity", type=str, default=D["wandb_entity"], help="W&B entity.")
    parser.add_argument("--wandb-project", type=str, default=D["wandb_project"], help="W&B project.")
    parser.add_argument("--wandb-group", type=str, default=D["wandb_group"], help="W&B group, the closest thing to a folder.")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=D["wandb_tags"],
                        help="Extra W&B tags. The arm's own labels (obs type, reward, corruption, occlusion) are "
                             "DERIVED and always added, so a tag cannot disagree with the run it names.")
    parser.add_argument("--log-dir", type=str, default=D["log_dir"], help="Log directory. Default: logs/recurrent_ppo/<arm>/<time>.")
    parser.add_argument("--log-interval", type=int, default=D["log_interval"], help="Log data every n timesteps.")
    parser.add_argument("--save-freq", type=int, default=D["save_freq"], help="Checkpoint every n timesteps.")
    parser.add_argument("--eval-freq", type=int, default=D["eval_freq"], help="Evaluate every n timesteps (0 disables).")
    parser.add_argument("--video-freq", type=int, default=D["video_freq"],
                        help="Record one rollout to W&B every n TRAINING ITERATIONS -- one rollout collection plus "
                             "its update, i.e. one block in the training log, n_steps * num_envs env steps "
                             "(0 disables). Needs --wandb. At the defaults a 2M-step run is 976 iterations, so "
                             "24 gives 40 clips. The scalar rollout/* series already sync from tensorboard; this "
                             "is the picture of what they describe.")
    parser.add_argument("--video-length", type=int, default=D["video_length"], help="Max frames per EPISODE in a clip.")
    parser.add_argument("--video-episodes", type=int, default=D["video_episodes"], help="Episodes per clip.")
    parser.add_argument("--video-stride", type=int, default=D["video_stride"],
                        help="Keep one frame every N env steps. The clip plays at the control rate regardless, so "
                             "a stride of 5 is a 5x-speed view of the same episode -- 60 frames for a 300-step "
                             "episode rather than 300.")
    parser.add_argument("--eval-corrupt", action="store_true", default=False,
                        help="ALSO evaluate with the observation corruption on, logged under eval_corrupt/. "
                             "For the --corrupt-obs arm this is the number that says whether the POMDP was "
                             "solved; the clean eval/ series stays alongside it so the two are comparable.")
    parser.add_argument("--n-eval-episodes", type=int, default=D["n_eval_episodes"], help="Episodes per evaluation.")
    parser.add_argument("--eval-curriculum", type=str, default=D["eval_curriculum"], choices=["match", "off"],
                        help="match: eval/ uses the SAME start distribution as training, so it measures what was "
                             "actually trained. off: eval/ uses the real uniform distribution. Either way the "
                             "other one is logged too, as eval_real/ or eval_train/ -- the matched number says "
                             "whether the policy improved, the real one says whether it can do the task.")
    parser.add_argument("--checkpoint", type=str, default=D["checkpoint"], help="Continue training from a checkpoint, in its own run directory.")
    parser.add_argument("--device", type=str, default=D["device"], help="Torch device.")
    return parser

import contextlib
from datetime import datetime

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, LogEveryNTimesteps
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

import torch as th
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import explained_variance

from recurrent_ppo.corrupt_policy import (aug_for, corrupting_extractor,
                                          features_extractor_kwargs, policy_for)
from recurrent_ppo.pusht_gym import build_vec_env, delta_scale_from_demos, env_kwargs_from
from recurrent_ppo.run_io import check_conflicts, dump_args, load_args, vecnormalize_path_for

# the training hyperparameters a checkpoint carries: RecurrentPPO.load rebuilds the agent from
# them, so a CLI value given on a resume would be silently ignored rather than applied
from recurrent_ppo.config import TRAINING_HPARAMS



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

    `data.returns` is the target, which is what makes this Q^pi rather than Q*. A Bellman backup
    would go exactly here, in place of that target; see `_QHeadMixin`.
    """

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        policy = self.model.policy
        if not hasattr(policy, "q_net"):
            return
        low = th.as_tensor(self.model.action_space.low, device=self.model.device)
        high = th.as_tensor(self.model.action_space.high, device=self.model.device)
        losses, preds, targets = [], [], []
        for data in self.model.rollout_buffer.get(self.model.batch_size):
            mask = data.mask > 1e-8
            # the buffer holds the UNCLIPPED Gaussian sample, but the env was stepped with the
            # clipped one, so that is the action these returns are a function of
            actions = th.clamp(data.actions, low, high)
            q = policy.q_values(data.observations, actions, data.lstm_states.vf, data.episode_starts)
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


class FeatureStdWindow(BaseCallback):
    """Open the corruption's feature-std EMA during collection and freeze it for the update.

    The EMA tracks the encoder's output magnitude, which for an end-to-end ResNet drifts by
    orders of magnitude over a run -- that drift is the whole reason the noise is scaled by it.
    But it has to hold still WITHIN one update: PPO re-encodes each stored transition once per
    epoch, and a std that moved between epochs would apply a different corruption each time,
    which is exactly what carrying the draw in the observation exists to prevent.
    """

    def _on_training_start(self):
        self._set(False)

    def _on_rollout_start(self):
        self._set(True)

    def _on_rollout_end(self):
        self._set(False)

    def _on_step(self):
        return True
    def _set(self, value):
        extractor = corrupting_extractor(self.model.policy)
        if extractor is not None:
            extractor.update_std = value


class SecondEval(BaseCallback):
    """A second evaluation on the OTHER start distribution, logged under its own prefix.

    Whichever distribution `eval/` uses, this reports the other. Matching training says whether
    the policy is improving at what it practises; the real uniform distribution says whether that
    is the task. Reporting only one of those is how a curriculum run flatters or libels itself --
    the nb=1.0 arm pushes the block 47.9px with its curricula on and 15.4px without.
    """

    def __init__(self, env, prefix, freq, n_episodes):
        super().__init__()
        self.env, self.prefix, self.freq, self.n_episodes = env, prefix, freq, n_episodes
        self._next = freq

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next = self.num_timesteps + self.freq
        extractor = corrupting_extractor(self.model.policy)
        was_on = extractor is not None and extractor.enabled
        if was_on:
            extractor.enabled = False
        try:
            rewards, lengths = evaluate_policy(self.model, self.env, n_eval_episodes=self.n_episodes,
                                               deterministic=True, return_episode_rewards=True, warn=False)
        finally:
            if was_on:
                extractor.enabled = True
        self.logger.record(f"{self.prefix}/mean_reward", float(np.mean(rewards)))
        self.logger.record(f"{self.prefix}/mean_ep_length", float(np.mean(lengths)))
        return True


class RolloutVideo(BaseCallback):
    """Record one deterministic episode to W&B every `video_freq` steps.

    Its own env, on the REAL start distribution with the curricula off and observations clean --
    the same convention CleanEvalCallback uses, so the video shows the task as reported rather
    than the easier one being trained on. Frames are taken BEFORE each step, and the env is a raw
    PushTGymEnv rather than a VecEnv, so no auto-reset can splice the next episode in.
    """

    def __init__(self, env_kwargs, obs_type, freq, length, seed, episodes=1, stride=1, aug=None):
        super().__init__()
        # the corrupted arms' policies read the draw back out of the observation, so this env
        # has to emit it too -- a raw PushTGymEnv here is an observation-space mismatch
        self.aug = aug
        self.env_kwargs = dict(env_kwargs, agent_near_block_prob=0.0, block_near_goal_prob=0.0,
                               render_mode="rgb_array")
        if obs_type == "keypoint":
            # the keypoint observation does not depend on render_size, so the video can be legible
            self.env_kwargs["render_size"] = 512
        self.obs_type, self.freq, self.length, self.seed = obs_type, freq, length, seed
        self.episodes, self.stride = episodes, max(1, stride)
        self.env = None
        self._iteration = 0

    def _on_step(self):
        return True
    def _on_rollout_end(self):
        # one call per training iteration, which is exactly the unit the frequency is expressed in
        self._iteration += 1
        if self._iteration % self.freq == 0:
            self._record()

    def _record(self):
        import imageio
        import wandb

        from recurrent_ppo.pusht_gym import PushTGymEnv

        if self.env is None:
            from recurrent_ppo.pusht_gym import AugmentationDraw

            self.env = PushTGymEnv(obs_type=self.obs_type, **self.env_kwargs)
            if self.aug:
                self.env = AugmentationDraw(self.env, **self.aug)
            self.env.reset(seed=self.seed)
        frames, returns, successes = [], [], []
        for _ in range(self.episodes):
            obs, _ = self.env.reset()
            # the LSTM state is reset PER EPISODE, exactly as it is in a vec env rollout -- one
            # clip spanning several episodes must not carry memory across their boundaries
            state, starts, reward = None, np.ones(1, dtype=bool), 0.0
            for step in range(self.length):
                # BEFORE the step, and only every `stride`-th one
                if step % self.stride == 0:
                    frames.append(self.env.render())
                batched = {k: v[None] for k, v in obs.items()} if isinstance(obs, dict) else obs[None]
                action, state = self.model.policy.predict(batched, state=state, episode_start=starts,
                                                          deterministic=True)
                obs, r, terminated, truncated, info = self.env.step(action[0])
                starts, reward = np.zeros(1, dtype=bool), reward + r
                if terminated or truncated:
                    break
            returns.append(reward)
            successes.append(bool(info["is_success"]))
        path = os.path.join(self.logger.dir or ".", f"rollout_{self.num_timesteps}.mp4")
        # imageio, not wandb's own encoder: moviepy is not installed and this avoids the dependency
        imageio.mimsave(path, frames, fps=10, macro_block_size=1)
        caption = (f"iter {self._iteration} | {self.num_timesteps} steps | {len(returns)} episode(s) | "
                   f"1 frame per {self.stride} steps ({self.stride}x speed) | "
                   f"returns {' '.join(f'{r:.1f}' for r in returns)} | "
                   f"successes {sum(successes)}/{len(successes)}")
        wandb.log({"rollout/video": wandb.Video(path, caption=caption)})
        self.logger.record("rollout/video_return_mean", float(np.mean(returns)))
        os.remove(path)


class CleanEvalCallback(EvalCallback):
    """EvalCallback that evaluates with observation corruption switched OFF, on FIXED episodes.

    The evaluation shares the training policy object, so corruption cannot be a construction-time
    choice here -- it has to be toggled around the rollout. Clean is the convention: every
    `eval/` number is then a clean-observation number and the arms differ only in how they
    trained. `--eval-corrupt` adds the other half, and play.py's --corrupt-obs-eval opts in too.

    The env is re-seeded before every evaluation. SB3 hands the seeds over at the next reset and
    then clears them, so without this each eval point draws a different set of episodes and the
    curve mixes policy improvement with episode luck -- and the clean and corrupted series would
    not be scoring the same task.
    """

    def __init__(self, *args, eval_seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_seed = eval_seed

    def _on_step(self):
        evaluating = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if evaluating and self.eval_seed is not None:
            self.eval_env.seed(self.eval_seed)
        extractor = corrupting_extractor(self.model.policy)
        was_on = extractor is not None and extractor.enabled
        if was_on:
            extractor.enabled = False
        try:
            return super()._on_step()
        finally:
            if was_on:
                extractor.enabled = True


class CorruptEvalCallback(BaseCallback):
    """Evaluate WITH the corruption on, under its own `eval_corrupt/` keys.

    Not an EvalCallback subclass: that class hardcodes the `eval/` log keys, so two of them
    would overwrite each other. This one only measures -- it saves nothing and nominates no
    checkpoint, so the two series stay directly comparable at every step.
    """

    def __init__(self, eval_env, n_eval_episodes, eval_freq, eval_seed=None,
                 deterministic=True, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.eval_seed = eval_seed
        self.deterministic = deterministic

    def _on_step(self):
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        extractor = corrupting_extractor(self.model.policy)
        if extractor is None:
            return True
        was_on, was_updating = extractor.enabled, extractor.update_std
        extractor.enabled = True
        # an evaluation must not move the training statistics it is measuring against
        extractor.update_std = False
        try:
            # the same seed the clean eval just used, so the two series score the SAME episodes
            # and the gap between them is the corruption rather than the draw
            if self.eval_seed is not None:
                self.eval_env.seed(self.eval_seed)
            sync_envs_normalization(self.training_env, self.eval_env)
            rewards, lengths = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic, return_episode_rewards=True, warn=False)
        finally:
            extractor.enabled, extractor.update_std = was_on, was_updating
        self.logger.record("eval_corrupt/mean_reward", float(np.mean(rewards)))
        self.logger.record("eval_corrupt/mean_ep_length", float(np.mean(lengths)))
        if self.verbose:
            print(f"[EVAL corrupt] mean_reward={np.mean(rewards):.2f} "
                  f"+/- {np.std(rewards):.2f} over {self.n_eval_episodes} episodes")
        return True


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
    args_cli = build_parser().parse_args()
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
    # the per-transition corruption/crop draw rides in the observation; see AugmentationDraw
    aug = aug_for(args_cli.obs, args_cli.corrupt_obs, render_size=args_cli.render_size,
                  t_max=args_cli.corrupt_t_max)

    # sb3-contrib pads sequences into fixed minibatches and requires this to divide exactly
    rollout = args_cli.n_steps * args_cli.num_envs
    if rollout % args_cli.batch_size:
        raise SystemExit(
            f"[ERROR] n_steps * num_envs = {rollout} is not a multiple of batch_size "
            f"{args_cli.batch_size}; sb3-contrib's recurrent buffer requires that. Adjust "
            "--batch-size, --n-steps or --num-envs.")

    # create the vectorised pushT environment
    env = build_vec_env(
        obs_type=args_cli.obs,
        n_envs=args_cli.num_envs,
        seed=args_cli.seed,
        use_subproc=not args_cli.dummy_vec_env,
        monitor_dir=log_dir,
        aug=aug,
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
            "q_net_arch": tuple(int(x) for x in args_cli.q_net_arch.split(",") if x),
            "q_lr": args_cli.q_lr,
            # corruption arrives as a features extractor, through SB3's own extension point
            **features_extractor_kwargs(args_cli.obs, args_cli.corrupt_obs, args_cli.corrupt_t_max),
        }
        lr = args_cli.learning_rate
        learning_rate = (lambda progress: progress * lr) if args_cli.lr_schedule == "linear" else lr
        agent = RecurrentPPO(
            policy_for(args_cli.obs, args_cli.corrupt_obs),
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
        FeatureStdWindow(),
        ActionDiagnostics(),
        QHead(),
    ]
    if args_cli.video_freq > 0:
        if run is None:
            raise SystemExit("[ERROR] --video-freq needs --wandb: the video has nowhere else to go.")
        callbacks.append(RolloutVideo(env_kwargs, args_cli.obs, args_cli.video_freq,
                                      args_cli.video_length, args_cli.seed + 20_000,
                                      episodes=args_cli.video_episodes,
                                      stride=args_cli.video_stride, aug=aug))
    if args_cli.eval_freq > 0:
        # a disjoint seed block, so the evaluation episodes are not the training ones
        eval_seed = args_cli.seed + 10_000
        # Two evaluations, on the two start distributions, so neither question is answered by
        # the other's number: `match` is the one the policy trained on, `off` is the real task.
        real_kwargs = dict(env_kwargs, agent_near_block_prob=0.0, block_near_goal_prob=0.0)
        matched = args_cli.eval_curriculum == "match"
        eval_env_kwargs = dict(env_kwargs) if matched else real_kwargs
        second_kwargs = real_kwargs if matched else dict(env_kwargs)
        second_prefix = "eval_real" if matched else "eval_train"
        eval_env = build_vec_env(
            obs_type=args_cli.obs,
            n_envs=1,
            seed=eval_seed,
            use_subproc=False,
            aug=aug,
            **eval_env_kwargs,
        )
        if isinstance(env, VecNormalize):
            # EvalCallback syncs the statistics across, which requires both sides to be wrapped
            eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=False,
                                    gamma=args_cli.gamma, clip_reward=np.inf)
        # no best_model_save_path: the run RECORDS eval numbers, it does not nominate a
        # checkpoint from them. Which step to evaluate is a decision passed in explicitly.
        callbacks.append(CleanEvalCallback(
            eval_env,
            n_eval_episodes=args_cli.n_eval_episodes,
            eval_freq=max(args_cli.eval_freq // args_cli.num_envs, 1),
            eval_seed=eval_seed,
            log_path=log_dir,
            deterministic=True,
            verbose=1,
        ))
        if args_cli.eval_corrupt:
            if not args_cli.corrupt_obs:
                raise SystemExit("[ERROR] --eval-corrupt needs a policy that has corruption to "
                                 "turn on; this run was not started with --corrupt-obs.")
            callbacks.append(CorruptEvalCallback(
                eval_env,
                n_eval_episodes=args_cli.n_eval_episodes,
                eval_freq=max(args_cli.eval_freq // args_cli.num_envs, 1),
                eval_seed=eval_seed,
                deterministic=True,
                verbose=1,
            ))
        second_env = build_vec_env(obs_type=args_cli.obs, n_envs=1, seed=args_cli.seed + 30_000,
                                   use_subproc=False, aug=aug, **second_kwargs)
        if isinstance(env, VecNormalize):
            second_env = VecNormalize(second_env, training=False, norm_obs=False, norm_reward=False,
                                      gamma=args_cli.gamma, clip_reward=np.inf)
        callbacks.append(SecondEval(second_env, second_prefix,
                                    max(args_cli.eval_freq // args_cli.num_envs, 1) * args_cli.num_envs,
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


if __name__ == "__main__":
    main()
