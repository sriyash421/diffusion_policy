"""Train a recurrent PPO policy, value function and Q head on PushT.

Two observation arms, `--obs keypoint` (block T keypoints + agent xy, plus their visibility
mask) and `--obs image`, and two noise regimes: clean by default, and `--corrupt-obs` for ST's
flat DDPM corruption of the encoded observation.

    # phase 1 -- clean
    python -m recurrent_ppo.train --obs keypoint
    # phase 2 -- noised observations
    python -m recurrent_ppo.train --obs keypoint --corrupt-obs

Everything env-side lives in pusht_gym.py, the corruption in corrupt_policy.py, the flags in
cli.py, the loop in runner.py, and the LSTM itself in arch.LstmArch. The feed-forward
frame-stacking counterpart is recurrent_ppo/ppo/train.py, which shares all of those.

Evaluation is always CLEAN, in both arms, so every reported number is a clean-observation
number and the arms differ only in how they trained.
"""

import argparse

from recurrent_ppo.arch import LstmArch
from recurrent_ppo.cli import add_common_args, add_lstm_args
from recurrent_ppo.runner import train


def main():
    parser = argparse.ArgumentParser(description="Train a RecurrentPPO agent on PushT.")
    add_lstm_args(add_common_args(parser))
    train(parser.parse_args(), LstmArch())


if __name__ == "__main__":
    main()
