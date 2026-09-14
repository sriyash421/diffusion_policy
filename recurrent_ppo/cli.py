"""The command-line surface both architectures share.

Every flag below describes the ENVIRONMENT, the reward, the PPO objective or the bookkeeping --
none of it depends on whether the policy is an LSTM or a frame stack. The two entry points add
only their own architecture flags on top, so a reward or curriculum change lands in both arms at
once and the comparison between them stays honest.

The help text is the argument FOR a value and belongs beside the flag, so it lives here; the
values themselves come from config.DEFAULTS.
"""

import argparse

from recurrent_ppo.config import DEFAULTS as D


def add_common_args(parser):
    """Everything that is not the architecture."""
    # environment
    parser.add_argument("--obs", type=str, default=D["obs"], choices=["keypoint", "state", "image"], help="Observation type.")
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
    parser.add_argument("--n-eval-episodes", type=int, default=D["n_eval_episodes"], help="Episodes per evaluation.")
    parser.add_argument("--eval-curriculum", type=str, default=D["eval_curriculum"], choices=["match", "off"],
                        help="match: eval/ uses the SAME start distribution as training, so it measures what was "
                             "actually trained. off: eval/ uses the real uniform distribution. Either way the "
                             "other one is logged too, as eval_real/ or eval_train/ -- the matched number says "
                             "whether the policy improved, the real one says whether it can do the task.")
    parser.add_argument("--checkpoint", type=str, default=D["checkpoint"], help="Continue training from a checkpoint, in its own run directory.")
    parser.add_argument("--device", type=str, default=D["device"], help="Torch device.")
    return parser


def add_lstm_args(parser):
    """sb3-contrib's LSTM knobs."""
    parser.add_argument("--lstm-hidden-size", type=int, default=D["lstm_hidden_size"], help="LSTM hidden size.")
    parser.add_argument("--n-lstm-layers", type=int, default=D["n_lstm_layers"], help="Number of LSTM layers.")
    parser.add_argument("--shared-lstm", action="store_true", default=D["shared_lstm"], help="Share one LSTM between actor and critic.")
    return parser


def add_play_args(parser):
    """The evaluation surface. Shared for the same reason the training one is: a
    checkpoint is evaluated by the arm that trained it, and the two must measure the
    same task the same way or the numbers are not comparable.
    """
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument("--arm", type=str, default="clean", choices=["clean", "corrupt"],
                        help="Which arm's logs to search when no --checkpoint is given. This selects a DIRECTORY; "
                             "it does not corrupt anything -- see --corrupt-obs-eval.")
    parser.add_argument("--obs", type=str, default=None, choices=["keypoint", "state", "image"],
                        help="Observation type. Default: whatever the checkpoint's run recorded.")
    parser.add_argument("--use-last-checkpoint", action="store_true", default=False,
                        help="When no checkpoint is given, use the last periodic save rather than the final model.")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of environments to run in parallel.")
    parser.add_argument("--n-episodes", type=int, default=50, help="Number of episodes to evaluate.")
    parser.add_argument("--max-episode-steps", type=int, default=None, help="Override the run's episode length.")
    parser.add_argument("--render-size", type=int, default=None, help="Override the run's render size.")
    parser.add_argument("--keypoint-visible-rate", type=float, default=None, help="Override the run's keypoint visibility.")
    parser.add_argument("--action-mode", type=str, default=None, choices=["delta", "absolute"], help="Override the run's action mode.")
    parser.add_argument("--delta-scale", type=float, default=None, help="Override the run's delta scale.")
    parser.add_argument("--reward", type=str, default=None, choices=["dense", "sparse", "shaped", "delta"], help="Override the run's reward mode.")
    parser.add_argument("--occlusion", type=str, default=None, choices=["iid", "persistent"], help="Override the run's occlusion mode.")
    parser.add_argument("--occlusion-persistence", type=float, default=None, help="Override the run's occlusion persistence.")
    parser.add_argument("--agent-near-block-prob", type=float, default=None, help="Override the run's near-block start fraction.")
    parser.add_argument("--agent-block-gap", type=float, nargs=2, default=None, help="Override the run's near-block gap.")
    parser.add_argument("--block-near-goal-prob", type=float, default=None, help="Override the run's near-goal start fraction.")
    parser.add_argument("--block-goal-offset", type=float, nargs=2, default=None, help="Override the run's near-goal offset.")
    parser.add_argument("--agent-start-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="Override the run's agent start range.")
    parser.add_argument("--block-start-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="Override the run's block start range.")
    parser.add_argument("--seed", type=int, default=100_000, help="Seed used for the environment.")
    parser.add_argument("--stochastic", action="store_true", default=False, help="Sample actions instead of taking the mean.")
    parser.add_argument("--corrupt-obs-eval", action="store_true", default=False,
                        help="Corrupt observations during THIS rollout. Off by default, so every reported number is "
                             "a clean-observation number, exactly as the in-training evaluation reports.")
    parser.add_argument("--video", action="store_true", default=False,
                        help="Record a video. Forces a single in-process environment, since the frames come from it.")
    parser.add_argument("--video-length", type=int, default=300, help="Length of the recorded video (in steps).")
    parser.add_argument("--device", type=str, default="auto", help="Torch device.")
    return parser
