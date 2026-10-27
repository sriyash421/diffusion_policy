"""Training-only calibration windows and metadata checks for Veritas scores."""
import numpy as np


def training_window_starts(episode_ends, train_episodes, planned_episodes, horizon):
    if horizon < 1:
        raise ValueError('horizon must be positive')
    ends = np.asarray(episode_ends, dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    train = sorted(set(map(int, train_episodes)))
    missing = set(train) - set(planned_episodes)
    if missing:
        raise ValueError(f'Missing training plans: {sorted(missing)}')
    windows = []
    for ep in train:
        if ep < 0 or ep >= len(ends):
            raise ValueError(f'Invalid training episode: {ep}')
        # Include the last complete H-action chunk without crossing an episode boundary.
        windows.append(np.arange(starts[ep], ends[ep] - horizon + 1, dtype=np.int64))
    return np.concatenate(windows) if windows else np.empty(0, dtype=np.int64)


def validate_calibration(meta, manifest, horizon, action_pos_scale, camera_file):
    from pathlib import Path
    expected = dict(calibration_split='train',
                    calibration_episodes=sorted(set(map(int, manifest['train']))),
                    horizon=int(horizon), action_pos_scale=action_pos_scale,
                    plan_done_mode='zero')
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f'Stale Veritas calibration: {key}={meta.get(key)!r}, '
                             f'expected {value!r}. Rebuild veritas_targets for this split/config.')
    if Path(meta['camera_file']).resolve() != Path(camera_file).resolve():
        raise ValueError('Veritas calibration camera differs from the training camera')
    if meta.get('zarr_path') != manifest['zarr_path']:
        raise ValueError('Veritas calibration dataset differs from the training dataset')
    mean, std = meta.get('value_mean'), meta.get('value_std')
    if mean is None or std is None or not np.isfinite(mean) or not (
            np.isfinite(std) and std > 0):
        raise ValueError('Veritas calibration requires finite mean and positive std')
