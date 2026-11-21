#!/usr/bin/env python
"""Contact sheet of an eval arm's episodes: first frame (the official video draws the Gemini plan
on it) and last frame per episode, failures first, labelled with seed and outcome.

    python scripts/simpler/veritas_pi0/plan_sheet.py <arm dir> [<arm dir> ...] --k 8 --out sheet.png

With several arm dirs, rows are matched by seed so plans of different models/prompts line up.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np


def episodes(arm):
    rows = {}
    for l in open(os.path.join(arm, "episodes.jsonl")):
        if l.strip():
            r = json.loads(l)
            rows.setdefault(r["seed"], r)
    return rows


def frames(arm, ep):
    m = glob.glob(os.path.join(arm, "videos", f"video_{ep}.mp4")) + \
        glob.glob(os.path.join(arm, "videos", f"video_{ep}_success.mp4"))
    if not m:
        return None
    c = cv2.VideoCapture(m[0])
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for f in (0, n - 1):
        c.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = c.read()
        out.append(cv2.resize(im, (320, 240)) if ok else np.zeros((240, 320, 3), np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--k", type=int, default=8, help="episodes per sheet")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [episodes(a) for a in args.arms]
    seeds = sorted(set.intersection(*(set(r) for r in rows)))
    # failures of the first arm first, so the sheet shows what goes wrong
    seeds = sorted(seeds, key=lambda s: (bool(rows[0][s]["success"]), s))[:args.k]
    tiles = []
    for s in seeds:
        line = []
        for arm, r in zip(args.arms, rows):
            fr = frames(arm, r[s]["episode"]) or [np.zeros((240, 320, 3), np.uint8)] * 2
            ok = bool(r[s]["success"])
            lab = f"{os.path.basename(arm.rstrip('/'))} seed {s}: {'SUCCESS' if ok else 'FAIL'}"
            cv2.putText(fr[0], lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3)
            cv2.putText(fr[0], lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            cv2.putText(fr[1], "last frame", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            line += fr
        tiles.append(np.concatenate(line, axis=1))
    cv2.imwrite(args.out, np.concatenate(tiles, axis=0))
    print(f"wrote {args.out} ({len(seeds)} seeds x {len(args.arms)} arms)")


if __name__ == "__main__":
    main()
