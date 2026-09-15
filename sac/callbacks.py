"""Diagnostics that answer the BON question during training, not only after it.

The failure this package exists to avoid is a Q that looks healthy by every SAC metric and is
useless as a verifier. Two ways that happens, both invisible to `critic_loss`:

  * **Q collapses to V(s).** It scores every candidate at a state identically -- which is
    EXACTLY the degeneracy of the heuristic being replaced, restated in Q coordinates.
    `bon/q_spread_zero_frac` is the direct measurement, and the number to beat is the
    heuristic's own: 28-35% of decisions overall, 68-73% during approach.
  * **Nothing to learn from.** Under a sparse reward at a threshold no demonstration reaches,
    every rung can sit at zero positives forever. `buffer/reward_rate_tau*` says so on day one
    rather than after a week.
"""

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

from diffusion_policy.common.slot_stats_util import BLIND_EPS


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
    tags = [f"obs-{cfg['obs']}", f"reward-{cfg['reward']}", f"chunk-{cfg['chunk_action_mode']}",
            f"gamma-{cfg['gamma']}", "algo-sac"]
    if cfg["block_near_goal_prob"] > 0:
        tags.append("init-near-goal")
    if cfg["demo_seed_frac"] > 0:
        tags.append("demo-seeded")
    if len(cfg["tau_ladder"]) > 1:
        tags.append(f"ladder-{len(cfg['tau_ladder'])}")
    return tags
