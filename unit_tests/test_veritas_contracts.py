"""Regression checks for evaluation overrides, noise identity, and split calibration."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion_policy.common.veritas_calibration import training_window_starts, validate_calibration
from diffusion_policy.policy.search_policy_robomonkey import (
    SearchPolicyRoboMonkey, SearchPolicyRoboMonkeyDiffusion)
from diffusion_policy.policy.veritas_verifier import VeritasPixelVerifier
from test_veritas_simpler import _camera


def test_eval_overrides_preserve_normalization_and_use_execution_horizon(tmp_path, monkeypatch):
    # Load the actual helper without importing the simulator/GPU evaluation entrypoint.
    script = Path(__file__).resolve().parents[1] / 'scripts/simpler/eval_search/eval_search.py'
    tree = ast.parse(script.read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_build_veritas_verifier')
    namespace = dict(os=os, json=json)
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(script), 'exec'), namespace)
    build = namespace['_build_veritas_verifier']
    for key in tuple(os.environ):
        if key.startswith('VERITAS_'):
            monkeypatch.delenv(key)
    _, path = _camera(tmp_path)
    old = VeritasPixelVerifier(str(path), horizon=8, normalize='fixed',
                              value_mean=-25, value_std=7, health_check=False)
    new = build(old, 4)
    assert (new.horizon, new.normalize, new.value_mean, new.value_std) == (4, 'fixed', -25, 7)
    monkeypatch.setenv('VERITAS_HORIZON', '2')
    monkeypatch.setenv('VERITAS_ACTION_POS_SCALE', '[0.5, 0.6, 0.7]')
    override = build(old, 4)
    assert override.horizon == 2
    assert override.action_pos_scale == (0.5, 0.6, 0.7)
    assert override.value_std == 7
    monkeypatch.setenv('VERITAS_CAMERA_FILE', str(path))
    monkeypatch.delenv('VERITAS_HORIZON')
    assert build(n_action_steps=4).horizon == 4


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('policy_class', [SearchPolicyRoboMonkey, SearchPolicyRoboMonkeyDiffusion])
def test_st_zero_noise_is_identity_without_consuming_rng(training, policy_class):
    features = torch.randn(2, 1, 40)
    state = torch.get_rng_state().clone()
    def unexpected_noise(*args):
        raise AssertionError('Clean mode must not draw noise')
    policy = SimpleNamespace(obs_noise_level=0., corrupt_obs=True,
                             corrupt_obs_eval=True, training=training,
                             _corrupt_at=unexpected_noise)
    assert policy_class.corrupt_obs_features(policy, features) is features
    assert torch.equal(state, torch.get_rng_state())


def test_calibration_excludes_heldout_and_includes_last_complete_chunk():
    ends = np.array([5, 10, 14])
    starts = training_window_starts(ends, [0, 2], [0, 1, 2], 4)
    assert starts.tolist() == [0, 1, 10]
    with pytest.raises(ValueError, match='Missing training plans'):
        training_window_starts(ends, [0, 2], [0], 4)


def test_calibration_rejects_stale_statistics(tmp_path):
    manifest = dict(train=[2, 0], zarr_path='dataset.zarr')
    camera = str(tmp_path / 'camera.json')
    meta = dict(calibration_split='train', calibration_episodes=[0, 2], horizon=4,
                action_pos_scale=None, plan_done_mode='zero', camera_file=camera,
                zarr_path='dataset.zarr', value_mean=-10., value_std=2.)
    validate_calibration(meta, manifest, 4, None, camera)
    for changes in ({'horizon': 8}, {'calibration_episodes': [0, 1, 2]},
                    {'calibration_split': None}, {'plan_done_mode': 'source_object_legacy'}):
        with pytest.raises(ValueError, match='Stale Veritas calibration'):
            validate_calibration(dict(meta, **changes), manifest, 4, None, camera)
