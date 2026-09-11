"""Play a checkpoint of a recurrent PPO agent on PushT.

    python recurrent_ppo/play.py --n-episodes 50
    python recurrent_ppo/play.py --arm corrupt --corrupt-obs-eval
    python recurrent_ppo/play.py --checkpoint logs/.../model.zip --video

How the environment was shaped is read from the run's own `params/args.yaml`, not retyped:
evaluating a 300-step task with a 500-step checkpoint is silent on the keypoint arm, and the
whole point of an evaluation is that it measures the thing that was trained. CLI flags still
override, and say so when they do.

The rollout threads the LSTM state by hand: `predict` needs the previous hidden state and an
episode_start mask, and a vectorised env auto-resets, so `dones` from step t is the mask for
step t+1. Getting this wrong looks like a working script with a memoryless policy.
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Play a checkpoint of a RecurrentPPO agent on PushT.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--arm", type=str, default="clean", choices=["clean", "corrupt"],
                    help="Which arm's logs to search when no --checkpoint is given. This selects a DIRECTORY; "
                         "it does not corrupt anything -- see --corrupt-obs-eval.")
parser.add_argument("--obs", type=str, default=None, choices=["keypoint", "image"],
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
args_cli = parser.parse_args()

"""Rest everything follows."""

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize

from recurrent_ppo.corrupt_policy import corrupting_extractor, set_corruption
from recurrent_ppo.config import DEFAULTS
from recurrent_ppo.pusht_gym import build_vec_env, env_kwargs_from
from recurrent_ppo.run_io import get_checkpoint_path, load_args, vecnormalize_path_for

# how the env was shaped. A key is overridable only if this script exposes a flag for it;
# getattr's default keeps the two from having to be kept in step by hand.
RESOLVE_KEYS = ("obs", "max_episode_steps", "render_size", "keypoint_visible_rate",
                "occlusion", "occlusion_persistence", "reward", "shaping_coef",
                "shaping_potential", "progress_coef", "success_bonus", "block_zero_coverage",
                "action_mode", "delta_scale", "agent_start_range", "block_start_range",
                "agent_near_block_prob", "agent_block_gap", "block_near_goal_prob",
                "block_goal_offset", "gamma")
# A run that predates a key gets today's default, announced per key. See
# recurrent_ppo_runs_sep11.md for what past runs actually used.
FALLBACKS = DEFAULTS


def resolve_config(run_dir):
    """The run's recorded environment configuration, with any CLI flag overriding it out loud."""
    saved = load_args(run_dir) or {}
    if not saved:
        print(f"[WARN] No params/args.yaml in {run_dir}; falling back to defaults, which may not "
              "be what this checkpoint trained on.")
    missing = [k for k in RESOLVE_KEYS if k not in saved]
    if missing:
        print(f"[WARN] This run predates {len(missing)} key(s); assuming today's default for each. "
              f"See recurrent_ppo_runs_sep11.md for what it actually used.")
        for k in missing:
            print(f"         {k} = {FALLBACKS[k]!r}")
    cfg = {}
    for key in RESOLVE_KEYS:
        override = getattr(args_cli, key, None)
        recorded = saved.get(key, FALLBACKS[key])
        if override is not None and override != recorded:
            print(f"[INFO] Overriding {key}: run recorded {recorded!r}, using {override!r}")
        cfg[key] = recorded if override is None else override
    return cfg


def main():
    """Play with a sb3-contrib recurrent agent."""
    if args_cli.checkpoint is None:
        log_root_path = os.path.abspath(os.path.join("logs", "recurrent_ppo", f"{args_cli.obs or 'keypoint'}_{args_cli.arm}"))
        checkpoint_path = get_checkpoint_path(log_root_path, args_cli.use_last_checkpoint)
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(os.path.abspath(checkpoint_path))

    cfg = resolve_config(log_dir)
    env_kwargs = env_kwargs_from(cfg, cfg["obs"])

    # video frames come off one env in this process, so it forces a single in-process env
    num_envs = args_cli.num_envs
    if args_cli.video:
        num_envs = 1
        env_kwargs["render_mode"] = "rgb_array"
        print("[INFO] --video forces a single in-process environment.")
    env = build_vec_env(
        obs_type=cfg["obs"],
        n_envs=num_envs,
        seed=args_cli.seed,
        use_subproc=not args_cli.video,
        **env_kwargs,
    )

    # normalize environment (if needed)
    vec_norm_path = vecnormalize_path_for(checkpoint_path)
    if vec_norm_path.exists():
        print(f"Loading saved normalization: {vec_norm_path}")
        env = VecNormalize.load(str(vec_norm_path), env)
        #  do not update them at test time
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False

    # create agent from sb3-contrib
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent = RecurrentPPO.load(checkpoint_path, env, print_system_info=True, device=args_cli.device)
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
    # (n_layers, n_envs, hidden) hidden state, and the mask that zeroes it on a new episode
    lstm_states = None
    episode_starts = np.ones((num_envs,), dtype=bool)
    while any(d < b for d, b in zip(done_count, budget)):
        actions, lstm_states = agent.predict(
            obs,
            state=lstm_states,
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


if __name__ == "__main__":
    main()
