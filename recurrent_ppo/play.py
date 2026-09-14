"""Play a checkpoint of a recurrent PPO agent on PushT.

    python -m recurrent_ppo.play --n-episodes 50
    python -m recurrent_ppo.play --arm corrupt --corrupt-obs-eval
    python -m recurrent_ppo.play --checkpoint logs/.../model.zip --video

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

def build_parser():
    """The CLI, built in a function so that importing this module does not parse
    sys.argv. The argparse-before-heavy-imports layout this file inherited needs no
    module-level parse: there is no simulator that must start before the imports.
    """
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
    parser.add_argument("--reward", type=str, default=None, choices=["dense", "sparse"], help="Override the run's reward mode.")
    parser.add_argument("--occlusion", type=str, default=None, choices=["iid", "persistent"], help="Override the run's occlusion mode.")
    parser.add_argument("--occlusion-persistence", type=float, default=None, help="Override the run's occlusion persistence.")
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
    parser.add_argument("--report-values", action="store_true", default=False,
                        help="Also report the value and Q heads: their explained variance against the "
                             "realised discounted return, de-normalised into real reward units.")
    parser.add_argument("--allow-missing-args", action="store_true", default=False,
                        help="Evaluate a run that has no params/args.yaml, accepting the current defaults "
                             "for how its environment was shaped. Off by default: that is a silent way to "
                             "measure a different task than the one trained.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device.")
    return parser

import numpy as np
import torch as th
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecNormalize

from recurrent_ppo.corrupt_policy import aug_for, corrupting_extractor, set_corruption
from recurrent_ppo.pusht_gym import build_vec_env, env_defaults, env_kwargs_from
from recurrent_ppo.run_io import get_checkpoint_path, load_args, vecnormalize_path_for

# how the env was shaped, and this script's flag for overriding each
ENV_KEYS = ("obs", "max_episode_steps", "render_size", "keypoint_visible_rate", "action_mode",
            "delta_scale", "agent_start_range", "block_start_range", "reward", "occlusion",
            "occlusion_persistence")
# what a run recorded before these keys existed -- taken from PushTGymEnv itself, never
# restated here, so they cannot drift away from the env the way they had
FALLBACKS = env_defaults()


def resolve_config(run_dir, args_cli):
    """The run's recorded environment configuration, with any CLI flag overriding it out loud."""
    saved = load_args(run_dir) or {}
    if not saved and not args_cli.allow_missing_args:
        # a warning was not enough: falling back silently evaluates the checkpoint in whatever
        # MDP the defaults happen to describe, which is the one failure an evaluation cannot
        # survive. Opt in explicitly, or pass the env flags yourself.
        raise SystemExit(
            f"[ERROR] No params/args.yaml in {run_dir}, so how this checkpoint's environment "
            "was shaped is unknown. Evaluating it against the current defaults would measure "
            "a different task. Pass --allow-missing-args to accept the defaults anyway, or "
            "give the env flags explicitly.")
    if not saved:
        print(f"[WARN] No params/args.yaml in {run_dir}; using defaults, which may not be what "
              "this checkpoint trained on.")
    cfg = {}
    for key in ENV_KEYS:
        override = getattr(args_cli, key)
        recorded = saved.get(key, FALLBACKS[key])
        if override is not None and override != recorded:
            print(f"[INFO] Overriding {key}: run recorded {recorded!r}, using {override!r}")
        cfg[key] = recorded if override is None else override
    return cfg


def _zero_states(policy, n_envs):
    """The (h, c) pair a fresh LSTM starts from, for actor and critic alike."""
    shape = (policy.lstm_actor.num_layers, n_envs, policy.lstm_actor.hidden_size)
    z = lambda: (th.zeros(shape, device=policy.device), th.zeros(shape, device=policy.device))
    return RNNStates(z(), z())


def _reward_scale(env):
    """What multiplies a head's output to put it back in real reward units.

    VecNormalize divides the reward by sqrt(ret_rms.var + eps) during training, so V and Q are
    fitted to returns in those units and mean nothing until the factor is put back. Reporting
    a value function without it is how a critic looks broken when it is merely rescaled.
    """
    rms = getattr(env, "ret_rms", None)
    if rms is None:
        return 1.0
    return float(np.sqrt(rms.var + getattr(env, "epsilon", 1e-8)))


def _returns_to_go(rewards, dones, gamma):
    """Realised discounted return from each step to the end of its episode.

    rewards/dones are (T, n_envs). Walked backwards per env and reset at a done, because a
    VecEnv auto-resets: the step after a done belongs to a different episode entirely.
    """
    out = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1], dtype=rewards.dtype)
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running * (1.0 - dones[t])
        out[t] = running
    return out


def main():
    """Play with a sb3-contrib recurrent agent."""
    args_cli = build_parser().parse_args()
    if args_cli.checkpoint is None:
        log_root_path = os.path.abspath(os.path.join("logs", "recurrent_ppo", f"{args_cli.obs or 'keypoint'}_{args_cli.arm}"))
        checkpoint_path = get_checkpoint_path(log_root_path, args_cli.use_last_checkpoint)
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(os.path.abspath(checkpoint_path))

    cfg = resolve_config(log_dir, args_cli)
    env_kwargs = env_kwargs_from(cfg, cfg["obs"])
    # the corruption regime is the run's IDENTITY, never a CLI choice here: the observation the
    # policy was built for carries the draw, so the env must emit exactly what training did
    saved = load_args(log_dir) or {}
    aug = aug_for(cfg["obs"], saved.get("corrupt_obs", False), render_size=cfg["render_size"],
                  t_max=saved.get("corrupt_t_max", 200))

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
        aug=aug,
        **env_kwargs,
    )

    # normalize environment (if needed)
    vec_norm_path = vecnormalize_path_for(checkpoint_path)
    if vec_norm_path.exists():
        print(f"Loading saved normalization: {vec_norm_path}")
        env = VecNormalize.load(str(vec_norm_path), env)
        # frozen at test time: neither the statistics nor the reward scaling
        env.training = False
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
    v_trace, q_trace, r_trace, d_trace = [], [], [], []

    policy = agent.policy
    policy.set_training_mode(False)
    low = th.as_tensor(env.action_space.low, device=policy.device)
    high = th.as_tensor(env.action_space.high, device=policy.device)
    has_q = hasattr(policy, "q_net")

    obs = env.reset()
    # the actor's and the critic's hidden states, threaded by hand. `predict` would return only
    # the actor's, and V and Q need the critic's -- so one forward drives both.
    lstm_states = _zero_states(policy, num_envs)
    episode_starts = np.ones((num_envs,), dtype=bool)
    while any(d < b for d, b in zip(done_count, budget)):
        obs_t, _ = policy.obs_to_tensor(obs)
        starts_t = th.as_tensor(episode_starts, dtype=th.float32, device=policy.device)
        with th.no_grad():
            # the state ENTERING this step, which is the one Q must be conditioned on
            entering_vf = lstm_states.vf
            actions_t, values_t, _, lstm_states = policy.forward(
                obs_t, lstm_states, starts_t, deterministic=not args_cli.stochastic)
            # the env is stepped with the clipped action, so that is what Q is asked about
            clipped_t = th.clamp(actions_t, low, high)
            q_t = (policy.q_values(obs_t, clipped_t, entering_vf, starts_t)
                   if has_q and args_cli.report_values else None)
        actions = clipped_t.cpu().numpy()

        # BEFORE the step: a VecEnv auto-resets a finished env inside step(), so a frame taken
        # afterwards would already show the next episode
        if args_cli.video and len(frames) < args_cli.video_length:
            frames.append(env.render("rgb_array"))

        obs, rewards, dones, infos = env.step(actions)
        if args_cli.report_values:
            v_trace.append(values_t.flatten().cpu().numpy())
            q_trace.append(q_t.cpu().numpy() if q_t is not None else np.zeros(num_envs))
            r_trace.append(np.asarray(rewards, dtype=np.float64))
            d_trace.append(np.asarray(dones, dtype=np.float64))
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

    if args_cli.report_values:
        gamma = float(saved.get("gamma", 0.99))
        scale = _reward_scale(env)
        v = np.asarray(v_trace, dtype=np.float64) * scale
        q = np.asarray(q_trace, dtype=np.float64) * scale
        g = _returns_to_go(np.asarray(r_trace), np.asarray(d_trace), gamma)
        # the last episode in each env is still running, so its return-to-go is truncated and
        # would look like a value-function error rather than a missing tail
        dones_arr = np.asarray(d_trace)
        keep = np.zeros_like(g, dtype=bool)
        for i in range(g.shape[1]):
            ends = np.flatnonzero(dones_arr[:, i])
            if len(ends):
                keep[: ends[-1] + 1, i] = True
        print(f"\n--- value and Q heads (gamma={gamma}, de-normalised x{scale:.4g}) ---")
        print(f"Steps scored: {int(keep.sum())} of {keep.size} (complete episodes only)")
        print(f"V: mean {v[keep].mean():.3f}   explained variance vs realised return "
              f"{explained_variance(v[keep], g[keep]):.3f}")
        if has_q:
            print(f"Q: mean {q[keep].mean():.3f}   explained variance vs realised return "
                  f"{explained_variance(q[keep], g[keep]):.3f}")
            print(f"Q - V (advantage estimate): mean {(q - v)[keep].mean():+.3f}")
        else:
            print("Q: this checkpoint has no q_net.")
        print(f"Realised discounted return: mean {g[keep].mean():.3f}")

    if args_cli.video:
        import imageio

        video_path = os.path.join(log_dir, "videos", "play.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, frames, fps=10)
        print(f"Saved video to: {video_path}")

    env.close()


if __name__ == "__main__":
    main()
