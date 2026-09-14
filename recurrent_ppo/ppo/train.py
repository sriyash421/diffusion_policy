"""Train a frame-stacking PPO policy, value function and Q head on PushT.

The feed-forward counterpart to recurrent_ppo/train.py, and sb3-contrib's own recommendation:
their RecurrentPPO docs advise starting with "simple frame-stacking as a simpler, faster and
usually competitive alternative", citing their PPO vs RecurrentPPO report on masked-velocity
environments and the Procgen paper's appendix Fig 11.

    # plain PPO, no stacking at all -- the baseline this project never had
    python recurrent_ppo/ppo/train.py --obs keypoint --n-stack 1
    # the usual frame-stacking setting
    python recurrent_ppo/ppo/train.py --obs keypoint --n-stack 4

Only the architecture differs from the LSTM arm: the environment, the reward, the curricula, the
callbacks, the evaluation protocol and the seed offsets are the same objects, not copies of them,
so a difference in the numbers is a difference in the architecture. Logs go to logs/ppo/ rather
than logs/recurrent_ppo/, and the run carries an `arch-stack` W&B tag.

Note what stacking can do HERE: it reconstructs state hidden from a single observation, and on
the clean arm nothing is hidden -- corruption off, occlusion off, and PushT's space damping is 0
so the block carries no momentum. The comparison only becomes representational once
--keypoint-visible-rate or --corrupt-obs is on.
"""

import argparse

from recurrent_ppo.arch import StackArch
from recurrent_ppo.cli import add_common_args
from recurrent_ppo.config import DEFAULTS as D
from recurrent_ppo.runner import train


def main():
    parser = argparse.ArgumentParser(description="Train a frame-stacking PPO agent on PushT.")
    add_common_args(parser)
    parser.add_argument("--n-stack", type=int, default=D["n_stack"],
                        help="Observations concatenated into the policy's input, oldest first. 1 is plain "
                             "PPO with no stacking. The stack is zeroed at every episode boundary, so the "
                             "first n_stack-1 steps of an episode are padded rather than carrying the "
                             "previous one -- the same per-episode reset the LSTM state gets.")
    train(parser.parse_args(), StackArch())


if __name__ == "__main__":
    main()
