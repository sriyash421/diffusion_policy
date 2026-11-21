#!/usr/bin/env python
"""Contact sheet of recorded pi0 demos (``record_pi0_demos.py``): per demo the first frame, the
mid frame and the last frame, labelled with demo index, seed, steps and the strict-filter result
(``strict_kept.json``: "strict ok" or the first rule failed). Strict failures come first.

    python demo_sheet.py --task stack --source pi0_demos     # -> <source dir>/demo_sheet.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_n_sweep import STEP2  # noqa: E402

TW, TH = 160, 120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--source", default="pi0_demos")
    ap.add_argument("--cols", type=int, default=6)
    a = ap.parse_args()
    d = f"{STEP2}/{a.task}/{a.source}"
    man = [json.loads(l) for l in open(f"{d}/manifest.jsonl") if l.strip()]
    sk = json.load(open(f"{d}/strict_kept.json")) if os.path.exists(f"{d}/strict_kept.json") else None
    fail = (sk or {}).get("first_fail", {})
    man.sort(key=lambda r: (str(r["demo"]) not in fail, r["demo"]))
    tiles = []
    for r in man:
        fr = imageio.mimread(f"{d}/demo_videos/{r['video']}", memtest=False)
        strip = np.concatenate([cv2.resize(fr[i], (TW, TH), interpolation=cv2.INTER_AREA)
                                for i in (0, len(fr) // 2, -1)], 1)
        bad = fail.get(str(r["demo"]))
        label = f"#{r['demo']} seed {r['seed']} {r['steps']} st | " + (
            "strict ?" if sk is None else (f"FAIL {bad}" if bad else "strict ok"))
        cap = np.full((18, strip.shape[1], 3), 255, np.uint8)
        cv2.putText(cap, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        tile = np.concatenate([strip, cap], 0)
        tiles.append(cv2.copyMakeBorder(tile, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                        value=(213, 94, 0) if bad else (255, 255, 255)))
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + a.cols - 1) // a.cols
    sheet = np.full((rows * h, a.cols * w, 3), 255, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, a.cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    out = f"{d}/demo_sheet.png"
    cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(out, len(tiles), "demos")


if __name__ == "__main__":
    main()
