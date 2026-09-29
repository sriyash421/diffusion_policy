"""Play a checkpoint of a frame-stacking PPO agent on PushT.

    python recurrent_ppo/ppo/play.py --n-episodes 50
    python recurrent_ppo/ppo/play.py --checkpoint logs/ppo/.../model.zip --video

The same evaluator as the LSTM arm's, on the same protocol. `--n-stack` is read from the run's
own params/args.yaml: a policy trained on 4 stacked frames and evaluated on 1 would load without
complaint and score like a broken policy.
"""

import argparse

from recurrent_ppo.arch import StackArch
from recurrent_ppo.cli import add_play_args
from recurrent_ppo.runner import play


def main():
    parser = argparse.ArgumentParser(description="Play a checkpoint of a frame-stacking PPO agent on PushT.")
    add_play_args(parser)
    parser.add_argument("--n-stack", type=int, default=None, help="Override the run's frame stack depth.")
    play(parser.parse_args(), StackArch())


if __name__ == "__main__":
    main()
