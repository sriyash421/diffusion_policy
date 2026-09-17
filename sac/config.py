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
           "DEFAULTS", "ENV_KEYS", "IDENTITY_KEYS"]

# ------------------------------------------------------------------ the chunk
# One agent decision is the action chunk ST/BC actually execute: n_action_steps=8 in
# pusht_base.yaml. Q(s, chunk) is then a drop-in for PushTVerifier.get_value, which receives
# exactly this window (pusht_search_mixin._verifier_inputs slices action[To-1 : To-1+Ta]).
CHUNK = 8

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
    "block_near_goal_prob": 0.60,
    "block_near_goal_prob_final": 0.10,
    "block_goal_offset": [3.0, 0.02],
    "block_goal_offset_final": [20.0, 0.15],
    "curriculum_anneal_frac": 0.7,      # fraction of training over which the offset widens
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
    # actor / uniform / demo-shape / smooth-random-walk. Uniform is DELIBERATELY small: in
    # absolute coordinates a uniform chunk jumps the target ~171 px per step against the demos'
    # 4 px median, and measured at the tightest curriculum it solves 0/40 where a demo-shaped
    # chunk solves 3/40 -- it destroys the near-goal states the curriculum exists to create.
    # It is kept at all because Q has to learn those chunks are bad.
    "mix_actor": 0.40,
    "mix_uniform": 0.10,
    "mix_demo": 0.25,
    "mix_smooth": 0.25,
    "smooth_scale": 12.0,
    "demo_seed_frac": 1.0,              # fraction of the demo chunks preloaded
    # WHICH demonstrations may be preloaded. Only this manifest's TRAIN episodes reach
    # the buffer, so the learned Q is never fitted on transitions from the episodes the
    # best-of-N sweep scores it on.
    "split_file": "diffusion_policy/config/splits/pusht_seed42_train106_val50.json",
    "reset_from_demos": 0.0,
    # bookkeeping
    "wandb": False,
    "wandb_entity": "l2sml",
    "wandb_project": "sac",
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
    "occlusion", "occlusion_persistence", "tau_ladder",
    "agent_start_range", "block_start_range", "agent_near_block_prob", "agent_block_gap",
    "block_near_goal_prob", "block_near_goal_prob_final", "block_goal_offset",
    "block_goal_offset_final", "max_reset_coverage",
    "curriculum_anneal_frac", "block_zero_coverage", "net_arch", "n_critics",
    "target_entropy", "mix_actor", "mix_uniform", "mix_demo", "mix_smooth", "smooth_scale", "demo_seed_frac",
    # WHICH demonstrations seeded the buffer is part of what a run IS: resuming with a different
    # manifest would swap the data under a checkpoint and leave nothing on disk saying so.
    "split_file",
)


def gamma_base(gamma_chunk, chunk=CHUNK):
    """The per-env-step discount whose CHUNK-th power is the quoted per-chunk discount."""
    return float(np.power(float(gamma_chunk), 1.0 / chunk))


# ==================================================================== the command line
# The help text is the ARGUMENT for a value and belongs beside the flag; the value itself lives
# in DEFAULTS above. They were in separate modules only because recurrent_ppo has two entry
# points and this has one -- the discipline is what matters, not the file boundary.

import argparse

D = DEFAULTS


def add_common_args(parser):
    # environment
    parser.add_argument("--obs", type=str, default=D["obs"], choices=["keypoint", "image"],
                        help="Which Q to learn. Two arms, and only two: `keypoint` (9 block "
                             "keypoints + agent xy + a visibility mask) and `image` (ST's own "
                             "96x96 frames through its own encoder).")
    parser.add_argument("--num-envs", type=int, default=D["num_envs"], help="Parallel environments.")
    parser.add_argument("--max-episode-steps", type=int, default=D["max_episode_steps"],
                        help="Base-step budget. MUST be a multiple of the chunk, or the final "
                             "chunk bootstraps at the wrong discount.")
    parser.add_argument("--render-size", type=int, default=D["render_size"],
                        help="Render size; also the image obs resolution.")
    parser.add_argument("--keypoint-visible-rate", type=float, default=D["keypoint_visible_rate"])
    parser.add_argument("--occlusion", type=str, default=D["occlusion"], choices=["iid", "persistent"])
    parser.add_argument("--occlusion-persistence", type=float, default=D["occlusion_persistence"])
    parser.add_argument("--reward", type=str, default=D["reward"],
                        choices=["sparse", "dense", "shaped", "delta"],
                        help="sparse is the POINT: Q* is then ~gamma^(chunks to solve), so any "
                             "pre-contact signal is earned by the TD backup rather than handed "
                             "over by a shaping term. The others are diagnostic arms and must "
                             "never be reported as the headline Q.")
    parser.add_argument("--tau-ladder", type=float, nargs="+", default=D["tau_ladder"],
                        help="Success thresholds to learn heads for. 0.95 is PushT's own and "
                             "the deliverable; the lower rungs exist because NO demonstration "
                             "ever reaches 0.95 (measured max 0.9018), so a single 0.95 head "
                             "seeded with demos has zero positive reward and learns Q = 0.")
    parser.add_argument("--agent-start-range", type=float, nargs=2, default=D["agent_start_range"])
    parser.add_argument("--block-start-range", type=float, nargs=2, default=D["block_start_range"],
                        help="Wider than the demo starts on purpose: a verifier must be accurate "
                             "on states a policy VISITS, and demo trajectories reach y=486.")
    parser.add_argument("--agent-near-block-prob", type=float, default=D["agent_near_block_prob"])
    parser.add_argument("--agent-block-gap", type=float, nargs=2, default=D["agent_block_gap"])
    parser.add_argument("--block-near-goal-prob", type=float, default=D["block_near_goal_prob"],
                        help="The ONLY mechanism producing a tau=0.95 terminal, since the "
                             "demonstrations never do.")
    parser.add_argument("--block-near-goal-prob-final", type=float,
                        default=D["block_near_goal_prob_final"],
                        help="Where the near-goal PROBABILITY anneals to. Annealing the offset "
                             "alone left the top rung with one terminal in 50k steps.")
    parser.add_argument("--block-goal-offset", type=float, nargs=2, default=D["block_goal_offset"],
                        help="Near-goal reset spread (px, rad) at the START of the anneal. "
                             "recurrent_ppo's [30, 0.25] sits at coverage 0.48 and NEVER crosses "
                             "0.95; [3, 0.02] sits at 0.94.")
    parser.add_argument("--block-goal-offset-final", type=float, nargs=2,
                        default=D["block_goal_offset_final"],
                        help="Where the anneal ends, so the final Q is on the real distribution.")
    parser.add_argument("--curriculum-anneal-frac", type=float, default=D["curriculum_anneal_frac"])
    parser.add_argument("--max-reset-coverage", type=float, default=D["max_reset_coverage"],
                        help="Reject resets at or above this coverage: an episode that begins "
                             "solved pays the sparse reward for doing nothing. Measured without "
                             "it, a do-nothing policy returned 0.349.")
    parser.add_argument("--dummy-vec-env", action="store_true", default=D["dummy_vec_env"])
    parser.add_argument("--seed", type=int, default=D["seed"])
    # agent
    parser.add_argument("--gamma", type=float, default=D["gamma"],
                        help="PER CHUNK. The env discount is its 8th root. All n candidates "
                             "share a state and differ by at most one chunk, so the quantity to "
                             "resolve is (1-gamma): 0.95 gives a 5%% per-chunk contrast with Q "
                             "~0.14 at episode start. Higher and candidates fall inside the "
                             "regression noise floor; lower and early states are all ~0.")
    parser.add_argument("--total-timesteps", type=int, default=D["total_timesteps"],
                        help="In CHUNK steps; one chunk is 8 base env steps.")
    parser.add_argument("--buffer-size", type=int, default=D["buffer_size"],
                        help="Chunk transitions. The image arm stores obs AND next_obs (SB3's "
                             "DictReplayBuffer refuses optimize_memory_usage) at 54.0 KiB each, "
                             "so 300k is 16.6 GB. Default is per-arm.")
    parser.add_argument("--batch-size", type=int, default=D["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=D["learning_rate"])
    parser.add_argument("--learning-starts", type=int, default=D["learning_starts"])
    parser.add_argument("--train-freq", type=int, default=D["train_freq"])
    parser.add_argument("--gradient-steps", type=int, default=D["gradient_steps"],
                        help="NOTE SB3 semantics: train_freq=(1,'step') collects num_envs "
                             "transitions per round, so gradient_steps=1 at 16 envs is an "
                             "update-to-data ratio of 1/16, not 1.")
    parser.add_argument("--tau", type=float, default=D["tau"])
    parser.add_argument("--n-critics", type=int, default=D["n_critics"])
    parser.add_argument("--ent-coef", type=str, default=D["ent_coef"])
    parser.add_argument("--target-entropy", type=float, default=D["target_entropy"],
                        help="SB3 defaults to -dim(A) = -16, calibrated on dense-reward "
                             "locomotion. A 16-d chunk has ~2 effective degrees of freedom and "
                             "-16 pins alpha so high the soft MDP stops resembling the MDP.")
    parser.add_argument("--net-arch", type=str, default=D["net_arch"])
    parser.add_argument("--share-features-extractor", action="store_true",
                        default=D["share_features_extractor"],
                        help="Image arm: otherwise SB3 builds THREE ResNets (actor, critic, "
                             "target) and pays three forward passes per gradient step.")
    # the behaviour mixture -- the off-distribution coverage BON needs
    parser.add_argument("--mix", type=float, nargs=4,
                        default=[D["mix_actor"], D["mix_uniform"], D["mix_demo"], D["mix_smooth"]],
                        metavar=("ACTOR", "UNIFORM", "DEMO", "SMOOTH"),
                        help="Behaviour mixture weights. Q is asked to rank DIFFUSION-POLICY "
                             "candidates, which the actor never proposes, so a buffer fed only "
                             "by the actor is sharp exactly where it is not needed. UNIFORM is "
                             "small on purpose: in absolute coordinates it jumps the target "
                             "~171px per step against the demos' 4px, and it scatters the block "
                             "before the sparse reward can ever be found.")
    parser.add_argument("--smooth-scale", type=float, default=D["smooth_scale"],
                        help="Per-step std (px) of the smooth random-walk exploration arm.")
    parser.add_argument("--split-file", type=str, default=D["split_file"],
                        help="Demo seeding uses only this manifest's TRAIN episodes.")
    parser.add_argument("--demo-seed-frac", type=float, default=D["demo_seed_frac"],
                        help="Fraction of the ~24k demo chunk transitions preloaded.")
    # bookkeeping
    parser.add_argument("--wandb", action="store_true", default=D["wandb"])
    parser.add_argument("--wandb-entity", type=str, default=D["wandb_entity"])
    parser.add_argument("--wandb-project", type=str, default=D["wandb_project"])
    parser.add_argument("--wandb-group", type=str, default=D["wandb_group"])
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=D["wandb_tags"])
    parser.add_argument("--log-dir", type=str, default=D["log_dir"])
    parser.add_argument("--log-interval", type=int, default=D["log_interval"])
    parser.add_argument("--save-freq", type=int, default=D["save_freq"])
    parser.add_argument("--eval-freq", type=int, default=D["eval_freq"])
    parser.add_argument("--n-eval-episodes", type=int, default=D["n_eval_episodes"])
    parser.add_argument("--bon-probe-freq", type=int, default=D["bon_probe_freq"] or D["eval_freq"],
                        help="How often to measure whether Q distinguishes candidates at all. "
                             "This is the replacement metric -- the heuristic's own blind rate "
                             "is 28-35%%, and bon/q_spread_zero_frac must beat it.")
    parser.add_argument("--checkpoint", type=str, default=D["checkpoint"])
    parser.add_argument("--device", type=str, default=D["device"])
    return parser


def parse_train_args(argv=None):
    p = argparse.ArgumentParser(description="Train SAC on chunked PushT to learn a BON verifier.")
    return add_common_args(p).parse_args(argv)
