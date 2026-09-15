"""Every SAC default, in one place.

Mirrors `recurrent_ppo/config.py` and for the same reason: a default that lives in the env
signature, the argparse and a fallback table drifts between the three, and the disagreement
only shows up as an evaluation quietly measuring a different task from the one trained.

The ARENA facts are re-exported from recurrent_ppo rather than restated -- they are properties
of PushT, and two copies could disagree. The DEFAULTS are NOT shared: `reward`, `gamma`,
`max_episode_steps` and `block_zero_coverage` all mean something different here, and importing
PPO's values would silently give this package the wrong task.
"""

import numpy as np

from recurrent_ppo.config import (AGENT_BOUNDS, AGENT_RADIUS, CONTROL_HZ, DEMO_ZARR, GOAL_POSE,
                                  NEAR_TRIES, SPAWN_TRIES, SUCCESS_THRESHOLD, WALL_INNER, WS)

__all__ = ["AGENT_BOUNDS", "AGENT_RADIUS", "CONTROL_HZ", "DEMO_ZARR", "GOAL_POSE", "NEAR_TRIES",
           "SPAWN_TRIES", "SUCCESS_THRESHOLD", "WALL_INNER", "WS", "CHUNK", "TAU_LADDER",
           "CHUNK_ACTION_MODES", "DEFAULTS", "ENV_KEYS", "IDENTITY_KEYS"]

# ------------------------------------------------------------------ the chunk
# One agent decision is the action chunk ST/BC actually execute: n_action_steps=8 in
# pusht_base.yaml. Q(s, chunk) is then a drop-in for PushTVerifier.get_value, which receives
# exactly this window (pusht_search_mixin._verifier_inputs slices action[To-1 : To-1+Ta]).
CHUNK = 8

# How a 16-d action in [-1, 1] becomes 8 absolute pixel targets. INVERTIBILITY IS THE
# REQUIREMENT, not a nicety: a verifier that cannot express the chunk it is handed is not a
# verifier, and ST's candidates are absolute targets drawn from the demo distribution.
# Measured over all 24,208 eight-step demo windows -- fraction NOT representable exactly:
#     absolute over [0, WS]          0.00%     <- the default, exact
#     absolute over AGENT_BOUNDS     1.16%
#     increment chain at R=64 px     2.20%
#     increment chain at R=33 px    26.01%     <- recurrent_ppo's delta_scale
# `increment` is kept because it buys translation equivariance (Q need not relearn "push left"
# at every arena location), which is a real argument for sample efficiency -- but it is not
# the default, because it is the one that can fail to represent a real candidate.
CHUNK_ACTION_MODES = ("absolute", "increment")

# The success threshold ladder. 0.95 is PushTEnv's own and the one BON is scored on; it is the
# deliverable. The lower rungs exist because NO demonstration ever reaches 0.95 -- measured max
# over all 206 episodes is 0.9018, and the fractions reaching each rung are:
#     0.80  93.7%      0.85  51.0%      0.90  1.0%      0.95  0.0%
# so a buffer seeded with demonstrations carries zero positive reward at 0.95 alone, and a Q
# regressed on it learns Q = 0: the same "identical score for every candidate" degeneracy as
# the heuristic being replaced. Each rung is a separate head on a shared trunk, and head tau
# sees the terminate-at-tau MDP exactly, so Q_0.95 is unchanged by the others' presence.
TAU_LADDER = (0.80, 0.85, 0.90, 0.95)

# ------------------------------------------------------------------ DEFAULTS
DEFAULTS = {
    # environment
    "obs": "keypoint",                  # the two arms are `keypoint` and `image`
    "num_envs": 16,
    # 304, not 300: the nearest multiple of CHUNK, so every non-terminal chunk is exactly 8
    # base steps and gamma_chunk = gamma_base**8 is exact rather than approximate. A ragged
    # final chunk would bootstrap at gamma**4 while the code says gamma**8 -- silent, and worst
    # at the truncation boundary where the critic has the least data. ST evaluates at 300; the
    # difference is shared by every candidate at a given state, so it cannot flip a ranking.
    "max_episode_steps": 38 * CHUNK,
    "render_size": 96,
    "keypoint_visible_rate": 1.0,
    "occlusion": "iid",
    "occlusion_persistence": 20.0,
    # sparse is the POINT of this package: Q* under it is ~gamma^(chunks to solve), so any
    # pre-contact signal is earned by the TD backup and cannot be dismissed as a re-encoded
    # distance heuristic. `delta` is a diagnostic arm only and must never be the headline Q.
    "reward": "sparse",
    "chunk_action_mode": "absolute",
    "chunk_scale": 64.0,                # px per unit, `increment` mode only
    "tau_ladder": list(TAU_LADDER),
    # The demos span agent [50, 449] and block x [66, 440] / y [116, 486] -- all 206 starts lie
    # inside these, where PushTEnv's own [100, 400] block box excludes 49 of them (23.8%). Wider
    # than the demo starts on purpose: a verifier has to be accurate on states a policy VISITS,
    # and demo trajectories reach agent y=511 and block y=486.
    "agent_start_range": [22.0, 490.0],
    "block_start_range": [50.0, 490.0],
    "agent_near_block_prob": 0.0,
    "agent_block_gap": [16.0, 25.0],
    # The only mechanism that produces a tau=0.95 terminal at all, since the demos never do.
    # Measured coverage at reset by offset: [30, 0.25] -> mean 0.48 and 0% above 0.95, which is
    # recurrent_ppo's default and cannot produce the reward it exists to produce. [3, 0.02] ->
    # mean 0.94 and 33% above 0.95. Annealed outward over training so the final Q is on the
    # real start distribution.
    "block_near_goal_prob": 0.15,
    "block_goal_offset": [3.0, 0.02],
    "block_goal_offset_final": [20.0, 0.15],
    "curriculum_anneal_frac": 0.4,      # fraction of training over which the offset widens
    # MUST be False here. PushTGymEnv.reset rejects any draw with coverage > 0, which is every
    # near-goal draw: it burns all 20 SPAWN_TRIES, warns, and falls through to the last draw
    # anyway. The states still arrive, but at the cost of 20 wasted resets and warning spam.
    "block_zero_coverage": False,
    # An episode must not begin solved: reject resets already at or above the threshold, else
    # the near-goal curriculum pays the sparse reward for doing nothing.
    "max_reset_coverage": SUCCESS_THRESHOLD,
    "dummy_vec_env": False,
    "seed": 0,
    # agent. gamma is quoted PER CHUNK; the env discount is gamma**(1/CHUNK).
    # All n BON candidates share a state and differ by at most one chunk of progress, so what
    # must be resolved is the per-chunk contrast (1 - gamma_chunk). At 0.9227 (gamma_base=0.99)
    # that is 7.7% but Q at episode start is 0.047, so early states are all ~0 and mutually
    # indistinguishable; at 0.97 the contrast falls to 3.0%, below what MSE regression reliably
    # resolves. 0.95 sits between: 5.0% contrast, Q ~0.14 at episode start.
    "gamma": 0.95,
    "total_timesteps": 2_000_000,       # in CHUNK steps, i.e. 16M base env steps
    "buffer_size": None,                # None -> per-arm default, see buffer_size_for()
    "batch_size": 512,
    "learning_rate": 3e-4,
    "learning_starts": 20_000,
    "train_freq": 1,
    "gradient_steps": 8,
    "tau": 0.005,
    "n_critics": 2,
    "ent_coef": "auto_0.1",
    # SB3 defaults target_entropy to -dim(A) = -16, a heuristic calibrated on dense-reward
    # locomotion. A 16-d chunk has ~2 effective degrees of freedom, and -16 pins alpha so high
    # that the soft MDP stops resembling the MDP. Worth sweeping.
    "target_entropy": -8.0,
    "net_arch": "256,256",
    "share_features_extractor": True,
    # behaviour mixture for the replay buffer -- the off-distribution coverage BON needs. Q is
    # asked to rank DIFFUSION-POLICY candidates, which the actor never proposes, so a buffer
    # fed only by the actor yields a Q that is accurate exactly where it is not needed.
    "mix_actor": 0.55,
    "mix_uniform": 0.20,
    "mix_demo": 0.20,
    "mix_wide": 0.05,
    "demo_seed_frac": 1.0,              # fraction of the 24,208 demo chunks preloaded
    "reset_from_demos": 0.0,
    # bookkeeping
    "wandb": False,
    "wandb_entity": "l2sml",
    "wandb_project": "sac_pusht",
    "wandb_group": None,
    "wandb_tags": [],
    "log_dir": None,
    "log_interval": 10_000,
    "save_freq": 100_000,
    "eval_freq": 50_000,
    "n_eval_episodes": 20,
    "bon_probe_freq": 0,                # 0 = off; needs a candidate bank on disk
    "checkpoint": None,
    "device": "auto",
}

# The image arm stores obs AND next_obs in full: DictReplayBuffer asserts against
# optimize_memory_usage, so the duplication cannot be turned off. At 96x96x3 uint8 that is
# 54.1 KiB per transition -- 300k is 16.6 GB, 1M would be 55 GB on a 125 GB box also running
# forkserver workers. The keypoint arm is 396 B per transition and is free by comparison.
_BUFFER_SIZE = {"keypoint": 2_000_000, "image": 300_000}


def buffer_size_for(obs_type, requested=None):
    return int(requested) if requested else _BUFFER_SIZE[obs_type]


# The PushTGymEnv kwargs this package passes through. keypoint_visible_rate / occlusion belong
# to PushTKeypointsEnv only, so the image arm must not be handed them.
ENV_KEYS = ("max_episode_steps", "render_size", "agent_start_range", "block_start_range",
            "agent_near_block_prob", "agent_block_gap", "block_near_goal_prob",
            "block_goal_offset")
KEYPOINT_ONLY_KEYS = ("keypoint_visible_rate", "occlusion", "occlusion_persistence")

# What changes what a checkpoint IS, as opposed to how a run is driven. Resuming with any of
# these altered would continue one experiment under another's name.
IDENTITY_KEYS = (
    "obs", "reward", "gamma", "max_episode_steps", "render_size", "keypoint_visible_rate",
    "occlusion", "occlusion_persistence", "chunk_action_mode", "chunk_scale", "tau_ladder",
    "agent_start_range", "block_start_range", "agent_near_block_prob", "agent_block_gap",
    "block_near_goal_prob", "block_goal_offset", "block_goal_offset_final", "max_reset_coverage",
    "curriculum_anneal_frac", "block_zero_coverage", "net_arch", "n_critics",
    "target_entropy", "mix_actor", "mix_uniform", "mix_demo", "mix_wide", "demo_seed_frac",
)


def gamma_base(gamma_chunk, chunk=CHUNK):
    """The per-env-step discount whose CHUNK-th power is the quoted per-chunk discount."""
    return float(np.power(float(gamma_chunk), 1.0 / chunk))
