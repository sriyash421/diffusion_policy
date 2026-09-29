"""Render demo-trajectory videos with action + feedback overlays.

For 10 seeded **train** and 10 seeded **val** episodes (the same recreatable split the
offline PushT diffusion-search policy trains on), render the expert demo trajectory from
the dataset with the recorded **action** (agent target) and the **feedback** signal
(goal-vs-achieved T keypoint displacement) drawn on each frame. This is the "info needed
to render a video of the actions and feedback along the demo trajectory".

We deliberately visualize the demo data itself (no policy/checkpoint needed). Coordinates
in agent_pos/block_pos/action are in the 512-px sim space; the recorded image is 96 px, so
overlays are scaled by render_size/512.

Usage:
  python demo_video_pusht.py -o data/demo_videos
  python demo_video_pusht.py -o data/demo_videos --n 10 --splits train,val
"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import numpy as np
import cv2
import imageio.v2 as iio

from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.env.pusht.feedback_util import (
    keypoints_at_pose, compute_feedback_from_pose, GOAL_POSE, N_KEYPOINTS)

SIM_SIZE = 512.0
GREEN = (26, 175, 122)
BLUE = (42, 120, 214)
RED = (214, 60, 42)
INK = (11, 11, 11)


def _episode_slice(replay_buffer, episode_idx):
    ends = np.asarray(replay_buffer.episode_ends[:])
    start = 0 if episode_idx == 0 else int(ends[episode_idx - 1])
    end = int(ends[episode_idx])
    return start, end


def _to_img_coords(xy, render_size):
    return (np.asarray(xy) * (render_size / SIM_SIZE)).astype(np.int32)


def render_episode(replay_buffer, episode_idx, render_size=96, scale=4):
    """Return a list of RGB uint8 frames for one demo episode with overlays."""
    s, e = _episode_slice(replay_buffer, episode_idx)
    imgs = np.asarray(replay_buffer['img'][s:e])          # (T, 96, 96, 3), 0-255 float
    block_pos = np.asarray(replay_buffer['block_pos'][s:e])  # (T, 3)
    action = np.asarray(replay_buffer['action'][s:e])     # (T, 2)
    feedback = compute_feedback_from_pose(block_pos)      # (T, 16)
    goal_kp = keypoints_at_pose(GOAL_POSE)                # (N, 2) sim coords

    frames = []
    T = len(imgs)
    for t in range(T):
        frame = np.clip(imgs[t], 0, 255).astype(np.uint8).copy()
        # upscale for legible overlays
        big = cv2.resize(frame, (render_size * scale, render_size * scale),
                         interpolation=cv2.INTER_NEAREST)
        sc = (render_size * scale) / SIM_SIZE

        achieved_kp = keypoints_at_pose(block_pos[t])     # (N, 2)
        for k in range(N_KEYPOINTS):
            gp = (goal_kp[k] * sc).astype(np.int32)
            ap = (achieved_kp[k] * sc).astype(np.int32)
            cv2.circle(big, tuple(gp), 3, BLUE, -1)       # goal keypoints
            cv2.circle(big, tuple(ap), 3, GREEN, -1)      # achieved keypoints
            cv2.line(big, tuple(ap), tuple(gp), RED, 1)   # feedback vector

        # action (agent target) as a cross
        act = (action[t] * sc).astype(np.int32)
        cv2.drawMarker(big, tuple(act), INK, cv2.MARKER_CROSS, 12, 2)

        tdist = float(np.linalg.norm(
            feedback[t].reshape(N_KEYPOINTS, 2), axis=-1).mean())
        cv2.putText(big, f't={t} Tdist={tdist:5.1f}', (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
        frames.append(big)
    return frames


@click.command()
@click.option('-o', '--output_dir', required=True)
@click.option('--zarr-path', default='data/pusht_cchi_v7_replay.zarr')
@click.option('--seed', default=42)
@click.option('--n-test-episodes', default=50)
@click.option('--n-val-episodes', default=10)
@click.option('--train-ratio', default=0.2)
@click.option('--n', 'n_videos', default=10, help='episodes per split')
@click.option('--splits', default='train,val')
@click.option('--fps', default=10)
def main(output_dir, zarr_path, seed, n_test_episodes, n_val_episodes, train_ratio,
         n_videos, splits, fps):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    # instantiate with the SAME 3-way split the training config uses, so the selected
    # episode indices are recreatable.
    dataset = PushTImageDataset(
        zarr_path=zarr_path, horizon=16, pad_before=1, pad_after=7, seed=seed,
        n_test_episodes=n_test_episodes, n_val_episodes=n_val_episodes,
        split='train', train_ratio=train_ratio, return_sequences=False)
    rb = dataset.replay_buffer

    for split in splits.split(','):
        split = split.strip()
        idxs = dataset.get_video_episode_idxs(split, n=n_videos)
        print(f'{split}: episodes {list(map(int, idxs))}')
        for episode_idx in idxs:
            frames = render_episode(rb, int(episode_idx))
            out_path = os.path.join(output_dir, f'{split}_ep{int(episode_idx):04d}.mp4')
            writer = iio.get_writer(out_path, fps=fps)
            for f in frames:
                writer.append_data(f)
            writer.close()
            print('wrote', out_path)


if __name__ == '__main__':
    main()
