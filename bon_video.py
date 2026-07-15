"""Best-of-N contact sheet: one video, resets as rows, samples as columns.

Rolls the policy out ``--n-samples`` times from each of ``--n-resets`` held-out test
reset states and tiles every rollout into a single mp4. Each cell is bordered green
if that rollout succeeded (max reward >= 1.0) and grey otherwise, and is overlaid with
the live per-step reward, so a row shows how much the diffusion sampler varies from a
fixed start.

Usage:
python bon_video.py -c results/<run>/checkpoints/latest.ckpt -o results/<run>/bon
python bon_video.py -c ... -o ... --n-resets 5 --n-samples 8
python bon_video.py -c ... -o ... --reset-idxs 47,12,2,22,6   # resets that succeed more
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import dill
import hydra
import numpy as np
import torch
import tqdm

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.pusht_image_dataset import (
    get_split_masks, get_episode_init_states)
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.env.pusht.pusht_feedback import PushTFeedbackWrapper
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper, VideoRecorder)
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv

SUCCESS_REWARD = 1.0
BLUE = (42, 120, 214)     # best-of-n pick: the max-reward sample for that reset
GREEN = (26, 175, 122)    # success
GREY = (190, 190, 188)    # neither
# 96 + 2*8 = 112 per cell, so any grid of 96px cells stays a multiple of 16 and h264
# does not silently pad the frame out to macroblock alignment.
BORDER = 8


def build_envs(n_envs, n_obs_steps, n_action_steps, max_steps, fps, crf, render_size=96):
    steps_per_render = max(10 // fps, 1)
    def env_fn():
        return MultiStepWrapper(
            VideoRecordingWrapper(
                PushTFeedbackWrapper(
                    PushTImageEnv(legacy=False, render_size=render_size)
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=fps, codec='h264', input_pix_fmt='rgb24', crf=crf,
                    thread_type='FRAME', thread_count=1),
                file_path=None,
                steps_per_render=steps_per_render
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps
        )
    return AsyncVectorEnv([env_fn] * n_envs)


def add_border(clip, color, width=BORDER):
    """Frame a (T,H,W,3) clip in a solid colour, keeping every cell the same size."""
    T, H, W, _ = clip.shape
    out = np.empty((T, H + 2 * width, W + 2 * width, 3), dtype=np.uint8)
    out[:] = np.asarray(color, dtype=np.uint8)
    out[:, width:width + H, width:width + W] = clip
    return out


def upscale(clip, scale):
    """Nearest-neighbour upscale so the overlaid text stays legible."""
    if scale == 1:
        return clip
    import cv2
    H, W = clip.shape[1:3]
    return np.stack([
        cv2.resize(f, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
        for f in clip])


def overlay_reward(clip, rewards):
    """Burn the live per-step reward and the running max into each frame.

    The running max is what scores the rollout (and picks the best-of-n sample), and
    it can exceed the final reward when a rollout reaches the goal and then pushes the
    block back out -- so showing only the live reward makes the best-of-n pick look
    wrong.
    """
    import cv2
    out = clip.copy()
    running_max = np.maximum.accumulate(rewards)
    for t, frame in enumerate(out):
        # rewards is padded/truncated to the clip length by the caller
        for i, text in enumerate(('r   %.2f' % rewards[t],
                                  'max %.2f' % running_max[t])):
            org = (4, 14 + i * 15)
            # white halo first so the text reads on the light PushT background
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (11, 11, 11), 1, cv2.LINE_AA)
    return out


def pad_to(clip, n_frames):
    """Hold the last frame so every cell in the grid runs the same length."""
    if len(clip) >= n_frames:
        return clip[:n_frames]
    return np.concatenate([clip, np.repeat(clip[-1:], n_frames - len(clip), axis=0)],
                          axis=0)


def tile(clips, n_rows, n_cols):
    """Lay equal-length clips out row-major."""
    rows = [np.concatenate(clips[r * n_cols:(r + 1) * n_cols], axis=2)
            for r in range(n_rows)]
    return np.concatenate(rows, axis=1)


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--n-resets', default=5)
@click.option('--n-samples', default=8)
@click.option('--reset-idxs', default=None, help='comma-separated test-reset indices')
@click.option('--max-steps', default=300)
@click.option('--fps', default=10)
@click.option('--crf', default=22)
@click.option('--scale', default=2, help='upscale each 96px cell so text is legible')
@click.option('--seed', default=0)
def main(checkpoint, output_dir, device, n_resets, n_samples, reset_idxs,
         max_steps, fps, crf, scale, seed):
    import imageio.v2 as iio
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    media_dir = pathlib.Path(output_dir).joinpath('cells')
    media_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.to(torch.device(device))
    policy.eval()

    ds_cfg = cfg.task.dataset
    replay_buffer = ReplayBuffer.copy_from_path(
        ds_cfg.zarr_path, keys=['agent_pos', 'block_pos'])
    _, test_mask = get_split_masks(
        n_episodes=replay_buffer.n_episodes,
        n_test_episodes=ds_cfg.n_test_episodes,
        seed=ds_cfg.seed,
        n_train_episodes=ds_cfg.get('n_train_episodes', None))
    all_states = get_episode_init_states(replay_buffer, test_mask)
    all_eps = np.nonzero(test_mask)[0]

    if reset_idxs is not None:
        idxs = [int(x) for x in reset_idxs.split(',')]
    else:
        idxs = list(range(n_resets))
    n_resets = len(idxs)
    states = all_states[idxs]
    eps = all_eps[idxs]
    print(f'resets {idxs} (episodes {list(eps)}) x {n_samples} samples '
          f'= {n_resets * n_samples} rollouts')

    # one env per cell, so every rollout runs in a single pass
    n_envs = n_resets * n_samples
    env = build_envs(n_envs, cfg.n_obs_steps, cfg.n_action_steps, max_steps, fps, crf)
    try:
        paths = [str(media_dir.joinpath('r%02d_s%02d.mp4' % (idxs[r], s)))
                 for r in range(n_resets) for s in range(n_samples)]
        cell_states = [states[r] for r in range(n_resets) for _ in range(n_samples)]

        def make_init_fn(state, path):
            state = np.asarray(state, dtype=np.float64)
            def _fn(env):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = path
                # via .unwrapped: wrappers forward attribute reads but not writes
                env.unwrapped.reset_to_state = state
            return _fn

        env.call_each('run_dill_function', args_list=[
            (dill.dumps(make_init_fn(s, p)),) for s, p in zip(cell_states, paths)])

        obs = env.reset()
        policy.reset()
        done = False
        pbar = tqdm.tqdm(total=max_steps, desc='rollout')
        while not done:
            obs_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to(device=device))
            with torch.no_grad():
                action = policy.predict_action(obs_dict)['action'].detach().cpu().numpy()
            obs, reward, done, info = env.step(action)
            done = np.all(done)
            pbar.update(action.shape[1])
        pbar.close()

        video_paths = env.render()
        step_rewards = [np.asarray(r) for r in env.call('get_attr', 'reward')]
        rewards = np.array([r.max() for r in step_rewards])
    finally:
        env.close()

    success = rewards >= SUCCESS_REWARD
    print('successes: %d/%d rollouts' % (success.sum(), len(success)))

    # the best-of-n pick per reset: the max-reward sample among that row's samples
    by_reset = rewards.reshape(n_resets, n_samples)
    best = np.zeros_like(rewards, dtype=bool)
    best.reshape(n_resets, n_samples)[np.arange(n_resets), by_reset.argmax(axis=1)] = True

    raw = list()
    for p in video_paths:
        reader = iio.get_reader(p)
        raw.append(np.stack([f for f in reader]))
        reader.close()
    n_frames = max(len(c) for c in raw)

    clips = list()
    for clip, sr, ok, is_best in zip(raw, step_rewards, success, best):
        # one video frame per env step, so frame t and reward t line up
        assert len(clip) == len(sr), f'{len(clip)} frames vs {len(sr)} rewards'
        clip = pad_to(clip, n_frames)
        sr = np.concatenate([sr, np.repeat(sr[-1:], n_frames - len(sr))])
        clip = overlay_reward(upscale(clip, scale), sr)
        # blue wins over green: a row's best sample is often also its only success
        color = BLUE if is_best else (GREEN if ok else GREY)
        clips.append(add_border(clip, color))

    grid = tile(clips, n_resets, n_samples)
    out_path = os.path.join(output_dir, 'bon_grid_%dx%d.mp4' % (n_resets, n_samples))
    writer = iio.get_writer(out_path, fps=fps)
    for frame in grid:
        writer.append_data(frame)
    writer.close()

    print('grid %s (rows=resets, cols=samples), frame %s' % (
        (n_resets, n_samples), grid.shape[1:]))
    print('legend: B = best-of-n (max reward), G = success, . = neither')
    for r in range(n_resets):
        sl = slice(r * n_samples, (r + 1) * n_samples)
        row_s, row_b = success[sl], best[sl]
        marks = ''.join('B' if b else ('G' if g else '.')
                        for g, b in zip(row_s, row_b))
        print('  reset %2d (ep %3d): %d/%d success | best r=%.2f  %s' % (
            idxs[r], eps[r], row_s.sum(), n_samples, by_reset[r].max(), marks))
    print('wrote', out_path)


if __name__ == '__main__':
    main()
