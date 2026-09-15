"""Train SAC on chunked PushT, to learn a Q that can rank best-of-N candidates.

    python sac/train.py --obs keypoint
    python sac/train.py --obs image --num-envs 8 --buffer-size 300000
    python sac/train.py --obs keypoint --tau-ladder 0.95      # the single-head version
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sac.cli import parse_train_args      # noqa: E402
from sac.runner import train              # noqa: E402

if __name__ == "__main__":
    train(parse_train_args())
