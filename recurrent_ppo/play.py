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


def main():
    parser = argparse.ArgumentParser(description="Play a checkpoint of a RecurrentPPO agent on PushT.")
    add_play_args(parser)
    play(parser.parse_args(), LstmArch())


if __name__ == "__main__":
    main()
