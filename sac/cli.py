"""The command-line surface. Help text lives here; values come from config.DEFAULTS.

Same discipline as recurrent_ppo/cli.py, for the same reason: the help text is the ARGUMENT
for a value and belongs beside the flag, but a default that appears in more than one file
drifts between them and nothing detects it.
"""

import argparse

from sac.config import DEFAULTS as D


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


def add_play_args(parser):
    """play.py flags default to None so the runner can tell `not given` from `overridden`."""
    parser.add_argument("--obs", type=str, default=None, choices=["keypoint", "image"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--use-last-checkpoint", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--render-size", type=int, default=None)
    return parser


def parse_train_args(argv=None):
    p = argparse.ArgumentParser(description="Train SAC on chunked PushT to learn a BON verifier.")
    return add_common_args(p).parse_args(argv)
