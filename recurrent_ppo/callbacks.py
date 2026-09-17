"""Callbacks shared by both architectures, and the tags that describe a run.

These were inside train.py while there was one entry point. There are now two -- the LSTM arm
and the frame-stacking arm in recurrent_ppo/ppo -- and the whole point of that comparison is
that the two differ ONLY in the architecture. A callback copied into each would be one more
thing that can silently diverge, so every one of them lives here and neither arm defines its own.

Nothing here knows which architecture it is running under. The two places that would have had to
ask are pushed to where the answer already lives: the Q head's per-batch unpacking is
`policy.q_batch`, and the video's frame stacking comes in as an `n_stack` argument.
"""

import os

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations

from recurrent_ppo.corrupt_policy import corrupting_extractor
from recurrent_ppo.pusht_gym import STATE_KEY, aug_draw


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
    """Regress Q(s, a) on the returns PPO fits V to -- alongside the update, never inside it.

    Runs at rollout end, before `train()`, so the buffer is full and untouched. Everything
    upstream of the head is detached inside `q_values`, and the head carries its own optimizer, so
    this cannot perturb the policy gradient. `q_explained_variance` is the honest read on it: the
    Q is on-policy, so it is only trustworthy near the behaviour policy.

    `policy.q_batch` is what makes this architecture-agnostic -- the recurrent buffer yields
    padded sequences with a mask and an LSTM state, the flat one yields neither, and the policy
    is the object that already knows which it is.
    """

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        policy = self.model.policy
        if not hasattr(policy, "q_net"):
            return
        losses, preds, targets = [], [], []
        for data in self.model.rollout_buffer.get(self.model.batch_size):
            q, target = policy.q_batch(data)
            loss = th.nn.functional.mse_loss(q, target)
            policy.q_optimizer.zero_grad()
            loss.backward()
            policy.q_optimizer.step()
            losses.append(loss.item())
            preds.append(q.detach())
            targets.append(target)
        self.logger.record("train/q_loss", float(np.mean(losses)))
        self.logger.record("train/q_explained_variance",
                           float(explained_variance(th.cat(preds).cpu().numpy(), th.cat(targets).cpu().numpy())))


class SecondEval(BaseCallback):
    """A second evaluation on the OTHER start distribution, logged under its own prefix.

    Whichever distribution `eval/` uses, this reports the other. Matching training says whether
    the policy is improving at what it practises; the real uniform distribution says whether that
    is the task. Reporting only one of those is how a curriculum run flatters or libels itself --
    the nb=1.0 arm pushes the block 47.9px with its curricula on and 15.4px without.
    """

    def __init__(self, env, prefix, freq, n_episodes, seed):
        super().__init__()
        self.env, self.prefix, self.freq, self.n_episodes = env, prefix, freq, n_episodes
        self.seed = seed
        self._next = freq

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next = self.num_timesteps + self.freq
        # the same episodes every time -- see CleanEvalCallback for why
        self.env.seed(self.seed)
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


class FrameStacker:
    """VecFrameStack's stacking for ONE un-vectorised env, for the video rollout.

    The video env is a raw PushTGymEnv rather than a VecEnv, so VecFrameStack cannot wrap it, and
    a policy trained on `n_stack` frames would be handed one. This drives SB3's own
    `StackedObservations` at num_envs=1 rather than reimplementing it: which axis to stack on is
    decided by `is_image_space`, not by the array's rank, and a hand-rolled version that got that
    or the oldest-first order wrong would be invisible -- it would simply train a worse policy.
    """

    def __init__(self, n_stack, observation_space):
        self.n_stack = max(1, int(n_stack))
        self.stacked = (None if self.n_stack == 1
                        else StackedObservations(1, self.n_stack, observation_space))

    @property
    def observation_space(self):
        return self.stacked.stacked_observation_space if self.stacked else None

    def reset(self, obs):
        if self.stacked is None:
            return obs
        return self._unbatch(self.stacked.reset(self._batch(obs)))

    def update(self, obs):
        if self.stacked is None:
            return obs
        # never with done=True: an episode boundary calls reset() instead, which is what zeroes
        # the stack the way VecFrameStack does on a vec env's auto-reset
        stacked, _ = self.stacked.update(self._batch(obs), np.zeros(1, dtype=bool), [{}])
        return self._unbatch(stacked)

    @staticmethod
    def _batch(obs):
        return {k: v[None] for k, v in obs.items()} if isinstance(obs, dict) else obs[None]

    @staticmethod
    def _unbatch(obs):
        return {k: v[0] for k, v in obs.items()} if isinstance(obs, dict) else obs[0]


class RolloutVideo(BaseCallback):
    """Record one deterministic episode to W&B every `video_freq` iterations.

    Its own env, on the REAL start distribution with the curricula off and observations clean --
    the same convention CleanEvalCallback uses, so the video shows the task as reported rather
    than the easier one being trained on. Frames are taken BEFORE each step, and the env is a raw
    PushTGymEnv rather than a VecEnv, so no auto-reset can splice the next episode in.
    """

    def __init__(self, env_kwargs, obs_type, freq, length, seed, episodes=1, stride=1, n_stack=1,
                 aug=None):
        super().__init__()
        self.env_kwargs = dict(env_kwargs, agent_near_block_prob=0.0, block_near_goal_prob=0.0,
                               render_mode="rgb_array")
        if obs_type != "image":
            # a lowdim observation does not depend on render_size, so the video can be legible
            self.env_kwargs["render_size"] = 512
        self.obs_type, self.freq, self.length, self.seed = obs_type, freq, length, seed
        self.episodes, self.stride = episodes, max(1, stride)
        self.n_stack = n_stack
        # the corrupted arms read the draw back out of the observation, so the video env has
        # to emit it too -- and after stacking, exactly as VecAugmentationDraw does
        self.aug = aug
        self.rng = np.random.default_rng(seed)
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
            self.env = PushTGymEnv(obs_type=self.obs_type, **self.env_kwargs)
            self.env.reset(seed=self.seed)
        stacker = FrameStacker(self.n_stack, self.env.observation_space)
        frames, returns, successes = [], [], []
        for _ in range(self.episodes):
            obs, _ = self.env.reset()
            # memory is reset PER EPISODE, exactly as it is in a vec env rollout -- one clip
            # spanning several episodes must not carry the LSTM state or the frame stack across
            # their boundaries
            obs = stacker.reset(obs)
            state, starts, reward = None, np.ones(1, dtype=bool), 0.0
            for step in range(self.length):
                # BEFORE the step, and only every `stride`-th one
                if step % self.stride == 0:
                    frames.append(self.env.render())
                batched = {k: v[None] for k, v in obs.items()} if isinstance(obs, dict) else obs[None]
                if self.aug:
                    if not isinstance(batched, dict):
                        batched = {STATE_KEY: batched}
                    batched.update(aug_draw(self.rng, n=1, **self.aug))
                # a non-recurrent policy accepts and ignores state/episode_start, so one call
                # serves both architectures
                action, state = self.model.policy.predict(batched, state=state, episode_start=starts,
                                                          deterministic=True)
                obs, r, terminated, truncated, info = self.env.step(action[0])
                obs = stacker.update(obs)
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


class CleanEvalCallback(EvalCallback):
    """EvalCallback that evaluates with observation corruption switched OFF, on a FIXED set.

    Corruption: the evaluation shares the training policy object, so it cannot be a
    construction-time choice here -- it has to be toggled around the rollout. Clean is the
    convention: every reported number is then a clean-observation number and the arms differ only
    in how they trained. play.py's --corrupt-obs-eval opts back in.

    Fixed set: SB3 seeds an eval env once, at construction, and neither EvalCallback nor
    evaluate_policy touches the seed again -- so every evaluation drew a FRESH sample and
    consecutive points were unpaired. At 20 episodes that is a standard error of 0.057 on a
    success rate near 0.07, which is most of the jitter seen between checkpoints. Re-seeding
    immediately before each evaluation replays the same episodes in the same order, so two
    checkpoints differ by the policy rather than by the draw. This is ST's convention
    (`test_start_seed`, pusht_keypoints_runner.py:111), and the base seed already matches its
    10000.

    The start states are policy-independent -- reset draws come from the env's own RNG and
    rejection sampling consumes a policy-independent number of them -- so the set is the same for
    every checkpoint and every arm, not merely stable within a run.
    """

    def __init__(self, *args, eval_seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_seed = eval_seed

    def _on_step(self):
        if self.eval_seed is not None and self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
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
        f"arch-{cfg['arch']}",
    ]
    if cfg["arch"] == "stack":
        tags.append(f"nstack-{cfg['n_stack']}")
    if cfg["agent_near_block_prob"] > 0:
        tags.append("init-near-block")
    if cfg["block_near_goal_prob"] > 0:
        tags.append("init-near-goal")
    if cfg["obs"] == "keypoint" and cfg["keypoint_visible_rate"] < 1.0:
        tags.append(f"occl-{cfg['occlusion']}")
    return tags


class UnfreezeEncoder(BaseCallback):
    """Hold the BC-initialised encoder still, then let PPO have it.

    PPO's first updates are the highest-variance ones it will ever take -- advantages estimated
    from a critic that is still at its initialisation, since BC has no target for a critic and
    trains the actor alone. Letting those gradients straight into a representation that took a
    supervised run to learn is how a warm start becomes indistinguishable from a cold one.

    ONLY THE IMAGE ARM HAS ANYTHING TO FREEZE. KeypointExtractor is `nn.Flatten` and carries no
    parameters, so on the keypoint and state arms this is a no-op and says so rather than
    implying an effect it cannot have.
    """

    def __init__(self, n_steps, verbose=0):
        super().__init__(verbose)
        self.n_steps = int(n_steps)
        self._frozen = False

    def freeze(self, policy):
        params = [p for p in policy.features_extractor.parameters()]
        for p in params:
            p.requires_grad_(False)
        self._frozen = bool(params)
        n = sum(p.numel() for p in params)
        print(f"[INFO] BC-init: froze {n} encoder parameters for the first {self.n_steps} steps"
              if params else
              "[INFO] BC-init: the encoder is parameter-free on this arm; nothing to freeze")
        return self._frozen

    def _on_step(self):
        if self._frozen and self.num_timesteps >= self.n_steps:
            for p in self.model.policy.features_extractor.parameters():
                p.requires_grad_(True)
            self._frozen = False
            print(f"[INFO] BC-init: encoder unfrozen at {self.num_timesteps} steps", flush=True)
        return True
