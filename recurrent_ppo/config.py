"""Every parameter in one place.

Before this, a default could live in three: the `PushTGymEnv.__init__` signature, train.py's
argparse, and play.py's fallback table for runs that predate a key. They had already drifted --
play.py believed the action mode was `absolute` and the block start range `[100, 400]` when the
env had long since moved to `delta` and `[60, 490]`. Nothing detects that, because each file is
self-consistent; the disagreement only shows up as an evaluation quietly measuring a different
task from the one that was trained.

So: values live here, and the other modules read them. train.py's argparse keeps its help text,
because the text is the argument FOR a value and belongs beside the flag, but every `default=`
now points at `DEFAULTS`.

Three kinds of thing live here, and the distinction is load-bearing:

  ARENA      facts about PushT, not choices. Measured off the env, not invented.
  DEFAULTS   the choices, i.e. what a flag is if you do not pass it.

A run that predates a key is NOT covered here. play.py falls back to today's default and says
so, loudly and per key, because a silent guess is how an evaluation ends up measuring a
different task from the one that was trained. What each past run actually used is recorded in
recurrent_ppo_runs_sep11.md, which is generated from the run directories rather than kept by
hand -- a hand-kept table of the same facts is exactly what drifted before.
"""

import numpy as np

# ------------------------------------------------------------------ ARENA: facts about PushT
WS = 512.0                       # the arena is 512x512 (PushTEnv.window_size)
AGENT_RADIUS = 15.0              # add_circle((256, 400), 15)
WALL_INNER = 7.0                 # walls are segments at 5 and 506 with radius 2
SUCCESS_THRESHOLD = 0.95         # PushTEnv._setup; `done` is coverage STRICTLY above this
CONTROL_HZ = 10                  # metadata['video.frames_per_second']
GOAL_POSE = np.array([256.0, 256.0, np.pi / 4])      # constant for every episode

# The agent is a KINEMATIC body, so pymunk's walls do not stop it -- it passes straight through,
# and they only ever constrain the block. This is therefore the ONLY bound on the agent, which is
# why it has to be the region the agent can usefully occupy rather than the image bounds: clipping
# to [0, WS] let two 2M-step runs park at (0, 512), outside the arena, unable to reach the block
# and uncharged for it.
AGENT_BOUNDS = (WALL_INNER + AGENT_RADIUS, WS - WALL_INNER - AGENT_RADIUS)     # (22, 490)

# rejection-sampling budgets for reset
SPAWN_TRIES = 20                 # redraws when the block's arms would spawn through a wall
NEAR_TRIES = 40                  # redraws for an agent start at the requested gap

OBS_TYPES = ("keypoint", "image")
ACTION_MODES = ("delta", "absolute")
REWARD_MODES = ("dense", "sparse", "shaped", "delta")
OCCLUSION_MODES = ("iid", "persistent")
# What the shaping potential measures, all reusing diffusion_policy/env/pusht/feedback_util:
#   t_goal  -(mean per-keypoint distance of the achieved T from the goal T). Captures position
#           AND rotation, is 0 iff the block is at the goal pose, and keeps giving gradient
#           after contact -- which is where the arm-only potential stalls.
#   arm_t   -(t_goal + arm-to-T), the repo's own verifier value (pusht_verifier.value_arm_t).
#   arm     -(arm-to-T) alone. The only term that varies BEFORE contact, which is the whole
#           reason feedback_util keeps it; t_goal on its own is flat until the block moves.
SHAPING_POTENTIALS = ("t_goal", "arm_t", "arm")

DEMO_ZARR = "data/pusht_cchi_v7_replay.zarr"
# used only when the demonstrations are not on disk; --delta-scale auto measures the real one
DELTA_SCALE_FALLBACK = 32.0

# ------------------------------------------------------------------ DEFAULTS: the choices
DEFAULTS = {
    # environment
    "obs": "keypoint",
    "num_envs": 16,
    "max_episode_steps": 300,
    "render_size": 96,
    "keypoint_visible_rate": 1.0,
    "occlusion": "iid",
    "occlusion_persistence": 20.0,
    "reward": "dense",
    "shaping_coef": 10.0,
    "progress_coef": 30.0,
    "success_bonus": 10.0,
    "block_zero_coverage": True,
    "shaping_potential": "t_goal",
    "action_mode": "delta",
    "delta_scale": "auto",
    "delta_percentile": 99.0,
    "demo_zarr": DEMO_ZARR,
    "agent_start_range": [50.0, 450.0],
    "block_start_range": [60.0, 490.0],
    "agent_near_block_prob": 0.0,
    "agent_block_gap": [20.0, 80.0],
    "block_near_goal_prob": 0.0,
    "block_goal_offset": [30.0, 0.25],
    "dummy_vec_env": False,
    "seed": 0,
    # observation corruption
    "corrupt_obs": False,
    "corrupt_t_max": 200,
    # agent
    "total_timesteps": 2_000_000,
    "n_steps": 128,
    "batch_size": 256,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.001,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "log_std_init": -1.0,
    "net_arch": "128,128",
    "lstm_hidden_size": 128,
    "n_lstm_layers": 1,
    "shared_lstm": False,
    "norm_reward": True,
    "target_kl": None,
    "lr_schedule": "constant",
    # bookkeeping
    "wandb": False,
    "wandb_entity": "l2sml",
    "wandb_project": "recurrent_ppo",
    "wandb_group": None,
    "wandb_tags": [],
    "log_dir": None,
    "log_interval": 10_000,
    "save_freq": 100_000,
    "eval_freq": 0,
    "video_freq": 24,
    "video_length": 300,
    "video_episodes": 1,
    "video_stride": 5,
    "n_eval_episodes": 20,
    "eval_curriculum": "match",
    "checkpoint": None,
    "device": "auto",
}

# The PushTGymEnv kwargs an arm takes. keypoint_visible_rate / occlusion belong to
# PushTKeypointsEnv only, so the image arm must not be handed them -- expressed once, here,
# rather than at every call site.
ENV_KEYS = ("max_episode_steps", "render_size", "action_mode", "delta_scale", "agent_start_range",
            "block_start_range", "agent_near_block_prob", "agent_block_gap",
            "block_near_goal_prob", "block_goal_offset")
KEYPOINT_ONLY_KEYS = ("keypoint_visible_rate", "occlusion", "occlusion_persistence")

# What changes what a checkpoint IS, as opposed to how a run is driven. Resuming with any of
# these altered would continue one experiment under another's name.
IDENTITY_KEYS = (
    "obs", "corrupt_obs", "corrupt_t_max", "reward", "shaping_coef", "shaping_potential", "progress_coef", "success_bonus",
    "block_zero_coverage",
    "max_episode_steps", "render_size", "keypoint_visible_rate",
    "occlusion", "occlusion_persistence", "action_mode", "delta_scale",
    "agent_start_range", "block_start_range",
    "agent_near_block_prob", "agent_block_gap",
    "block_near_goal_prob", "block_goal_offset",
    "lstm_hidden_size", "n_lstm_layers", "shared_lstm", "net_arch",
)

# The training hyperparameters a checkpoint carries: RecurrentPPO.load rebuilds the agent from
# them, so a CLI value given on a resume is ignored rather than applied.
TRAINING_HPARAMS = ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma",
                    "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm")
