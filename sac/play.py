"""Evaluate a trained SAC checkpoint on the fixed episode set.

    python sac/play.py --n-episodes 50
    python sac/play.py --checkpoint logs/sac/keypoint/<run>/model.zip
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sac.cli import add_play_args         # noqa: E402
from sac.runner import play               # noqa: E402

if __name__ == "__main__":
    play(add_play_args(argparse.ArgumentParser(description="Evaluate a SAC checkpoint.")).parse_args())
