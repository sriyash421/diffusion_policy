"""Pin the two LSTM BC expert arms to each other and to the 176/0/30 manifest.

    pytest unit_tests/test_lstm_expert_configs.py

train_pusht_lstm_bc_keypoint_expert does NOT inherit the image expert -- it inherits the
keypoint master and restates the expert scalars. That direction was chosen because the
keypoint observation swap re-POINTS `policy` at a sibling node (merging a dict onto it would
keep the parent's keys), and duplicating that machinery to gain scalar inheritance trades a
safe duplication for a dangerous one. The price is that the four expert scalars exist twice,
and this file is what keeps the copies equal. See test_bc_expert_configs.py for the UNet
pair, which makes the opposite trade for the opposite reason.
"""
import os
import sys

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGE_CFG = 'train_pusht_lstm_bc_expert'
KEYPOINT_CFG = 'train_pusht_lstm_bc_keypoint_expert'

# The 30 held-out episodes and the budget that produces the 10k checkpoint grid.
N_DEMOS = 176
N_TEST = 30
N_VAL = 0

# Restated in both files, so asserted equal in both. These are exactly the keys
# train_pusht_lstm_bc_keypoint_expert.yaml copies.
EXPERT_SCALARS = ('max_gradient_steps', 'checkpoint_every')


def _compose(name):
    from hydra import compose, initialize_config_dir
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    cfg_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'diffusion_policy', 'config')
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        return compose(config_name=name)


@pytest.fixture(scope='module')
def cfgs():
    return _compose(IMAGE_CFG), _compose(KEYPOINT_CFG)


@pytest.mark.parametrize('key', EXPERT_SCALARS)
def test_expert_scalars_match_across_arms(cfgs, key):
    """The keypoint copy of each expert scalar must equal the image arm's."""
    image, keypoint = cfgs
    a, b = image.training[key], keypoint.training[key]
    assert a == b, (
        f'training.{key} differs: image={a!r} keypoint={b!r}. The keypoint expert restates '
        f'these rather than inheriting them; update '
        f'diffusion_policy/config/{KEYPOINT_CFG}.yaml to match.')


@pytest.mark.parametrize('name', [IMAGE_CFG, KEYPOINT_CFG])
def test_split_and_budget(name):
    """Both arms train on all 176 demos, hold out the same 30, share the 10k eval grid."""
    cfg = _compose(name)
    assert cfg.n_demos == N_DEMOS
    assert cfg.task.dataset.split_file.endswith(f'pusht_seed42_train{N_DEMOS}.json')
    assert cfg.task.dataset.n_test_episodes == N_TEST
    assert cfg.task.dataset.n_val_episodes == N_VAL
    assert cfg.training.max_gradient_steps == 200000
    assert cfg.training.checkpoint_every == 10000
    # Offline eval is the record; an in-training rollout would be a second, different
    # measurement of the same quantity through a different code path.
    assert cfg.training.get('rollout_every_steps') is None
    # NOT stretched to the 200k budget: 600 warmup + 15400 decay floors the lr at 16k, the
    # same schedule the 30- and 60-demo LSTM arms ran under.
    assert cfg.training.lr_scheduler_kwargs.decay_steps == 15400


@pytest.mark.parametrize('name', [IMAGE_CFG, KEYPOINT_CFG])
def test_no_val_split_means_no_val_readout(name):
    """n_val_episodes 0 is load-bearing, and the workspace must be the thing that copes.

    TrainMLPImageWorkspace guards the validation block on `has_val = len(val_dataset) > 0`
    because iterating a zero-length accelerate dataloader raises. If that guard is ever
    removed, these arms die at the first val step rather than silently losing a metric.
    """
    cfg = _compose(name)
    assert cfg.task.dataset.n_val_episodes == 0
    assert cfg._target_.endswith('TrainMLPImageWorkspace')
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'diffusion_policy', 'workspace', 'train_mlp_image_workspace.py')
    body = open(src).read()
    assert 'has_val = len(val_dataset) > 0' in body
    assert 'do_val = do_val and has_val' in body


def test_observation_axis_is_the_only_difference(cfgs):
    """The one axis the two arms are allowed to differ in, asserted both ways."""
    image, keypoint = cfgs
    assert set(image.shape_meta.obs) == {'image'}
    assert set(keypoint.shape_meta.obs) == {'keypoint', 'agent_pos'}
    assert keypoint.policy.obs_encoder._target_.endswith('FlattenObsEncoder')
    assert image.policy._target_.endswith('LSTMBCPolicy')
    assert keypoint.policy._target_.endswith('LSTMBCPolicy')


def test_run_names_are_distinct_and_unstamped(cfgs):
    """176 is unique among the manifests, so both arms own their bare `demos-176` spelling."""
    image, keypoint = cfgs
    assert image.run_name == 'lstmbc_enc-resnet18_demos-176_seed-42'
    assert keypoint.run_name == 'lstmbckp_demos-176_seed-42'
