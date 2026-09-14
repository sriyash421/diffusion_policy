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

from recurrent_ppo.arch import LstmArch
from recurrent_ppo.cli import add_play_args
from recurrent_ppo.runner import play


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
    parser = argparse.ArgumentParser(description="Play a checkpoint of a RecurrentPPO agent on PushT.")
    add_play_args(parser)
    play(parser.parse_args(), LstmArch())


if __name__ == "__main__":
    main()
