"""Score every checkpoint of a run on the FIXED evaluation set.

    python recurrent_ppo/scripts/eval_checkpoints.py logs/ppo/keypoint_clean/<run>
    python recurrent_ppo/scripts/eval_checkpoints.py <run> --arch lstm --n-episodes 50

Two reasons this exists rather than only the in-training callback:

  * A run trained before the fixed set landed still has all its checkpoints, so its paired curve
    can be recovered without re-running it.
  * The in-training evaluation is 20 episodes, chosen to be cheap enough to run every 200k steps.
    Offline there is no such constraint, so the same curve can be measured to whatever precision
    the question needs -- at a success rate near 0.07, 20 episodes carry a standard error of
    0.057 and 200 carry 0.018.

Every checkpoint sees the SAME episodes, in the same order, because the env is re-seeded before
each one -- which is the whole point. Both scores are reported: the episode return the reward
mode defines, and max coverage, which is what pusht_image_runner reports and therefore the
number comparable to the diffusion-policy arms.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from recurrent_ppo.arch import ARCHS
from recurrent_ppo.corrupt_policy import aug_for
from recurrent_ppo.pusht_gym import VecAugmentationDraw, build_vec_env, env_kwargs_from
from recurrent_ppo.runner import resolve_config


def checkpoints(run_dir):
    """The run's checkpoints in training order, final model last."""
    numbered = []
    for name in os.listdir(run_dir):
        if name.startswith("model_") and name.endswith("_steps.zip"):
            numbered.append((int(name[len("model_"):-len("_steps.zip")]), name))
    out = [name for _, name in sorted(numbered)]
    for name in ("best_model.zip", "model.zip"):
        if os.path.exists(os.path.join(run_dir, name)):
            out.append(name)
    return out


def score(agent, env, n_episodes, num_envs, seed, deterministic):
    """Mean return, success rate and mean max-coverage over a fixed set of episodes."""
    env.seed(seed)                       # the fixed set: the same episodes for every checkpoint
    budget = [(n_episodes + i) // num_envs for i in range(num_envs)]
    done_count = [0] * num_envs
    returns, successes, coverage = [], [], []
    obs = env.reset()
    states = None
    episode_starts = np.ones((num_envs,), dtype=bool)
    while any(d < b for d, b in zip(done_count, budget)):
        actions, states = agent.predict(obs, state=states, episode_start=episode_starts,
                                        deterministic=deterministic)
        obs, _, dones, infos = env.step(actions)
        episode_starts = dones
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done and done_count[i] < budget[i]:
                done_count[i] += 1
                returns.append(float(info["episode"]["r"]))
                successes.append(bool(info["is_success"]))
                coverage.append(float(info["max_reward"]))
    return np.mean(returns), np.mean(successes), np.mean(coverage)


def main():
    parser = argparse.ArgumentParser(description="Score a run's checkpoints on the fixed eval set.")
    parser.add_argument("run_dir", type=str, help="A run directory containing model_*_steps.zip.")
    parser.add_argument("--arch", type=str, default=None, choices=sorted(ARCHS),
                        help="Default: read from the run's params/args.yaml.")
    parser.add_argument("--n-episodes", type=int, default=50, help="Episodes per checkpoint.")
    parser.add_argument("--num-envs", type=int, default=8, help="Parallel envs.")
    parser.add_argument("--seed", type=int, default=10_000,
                        help="The fixed set. Matches the in-training eval (--seed + 10000) and "
                             "ST's test_start_seed.")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions. Worth running alongside the default: a policy held "
                             "up by its exploration noise scores very differently under the two.")
    parser.add_argument("--every", type=int, default=1, help="Take every Nth numbered checkpoint.")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    # resolve_config() reads overrides by attribute name; our --seed selects the
    # EVAL SET, not the env config, so it must not be picked up as an override
    eval_seed, args.seed = args.seed, None

    run_dir = os.path.abspath(args.run_dir)
    arch = ARCHS[args.arch] if args.arch else ARCHS[_recorded_arch(run_dir)]
    cfg = resolve_config(run_dir, args, arch)
    print(f"[INFO] arch={arch.name} obs={cfg['obs']} reward={cfg['reward']} "
          f"episodes={args.n_episodes} seed={eval_seed} "
          f"actions={'sampled' if args.stochastic else 'deterministic'}")

    env = arch.wrap(build_vec_env(obs_type=cfg["obs"], n_envs=args.num_envs, seed=eval_seed,
                                  use_subproc=False, **env_kwargs_from(cfg, cfg["obs"])), cfg)
    # a corrupted run's policy reads its noise back out of the observation, so this env has to
    # emit it -- and outside arch.wrap, or VecFrameStack would stack the draw with the frames
    aug = aug_for(cfg["obs"], cfg.get("corrupt_obs", False), render_size=cfg["render_size"],
                  t_max=cfg.get("corrupt_t_max", 200), n_stack=arch.video_n_stack(cfg),
                  snr=cfg.get("corrupt_snr"))
    if aug:
        env = VecAugmentationDraw(env, seed=eval_seed, **aug)
    names = checkpoints(run_dir)
    names = [n for i, n in enumerate(names) if not n.startswith("model_") or i % args.every == 0]
    print(f"\n{'checkpoint':26s} {'return':>9s} {'success':>9s} {'coverage':>9s}")
    for name in names:
        # the banner is per-checkpoint noise here, and identical every time
        agent = arch.load(os.path.join(run_dir, name), env, dict(cfg, device=args.device),
                          print_system_info=False)
        ret, suc, cov = score(agent, env, args.n_episodes, args.num_envs, eval_seed,
                              not args.stochastic)
        print(f"{name:26s} {ret:9.2f} {suc:9.3f} {cov:9.3f}", flush=True)
    env.close()


def _recorded_arch(run_dir):
    from recurrent_ppo.run_io import load_args
    return (load_args(run_dir) or {}).get("arch", "lstm")


if __name__ == "__main__":
    main()
