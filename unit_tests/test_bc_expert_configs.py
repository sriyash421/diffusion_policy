"""Pin the two BC expert arms to each other: same UNet, same sampler, same everything but obs.

    pytest unit_tests/test_bc_expert_configs.py

These two configs are the candidate PushT experts -- train_pusht_unet_bc_expert (image) and
train_pusht_unet_bc_keypoint_expert (keypoint) -- and a success-rate comparison between them
is only an OBSERVATION ablation if nothing else differs. Keeping that true needs a test
rather than a comment, because the keypoint arm cannot inherit the image arm's `policy`
block and therefore carries its own copy of the architecture:

  * PushTSearchMixin._check_obs_contract refuses to construct when shape_meta declares any
    low_dim key, and the keypoint observation is entirely low_dim; and
  * DiffusionUnetImagePolicy stores leftover **kwargs and forwards them to
    `noise_scheduler.step(**self.kwargs)`, so the image arm's verifier and crop keys would
    raise a TypeError inside the sampler rather than being ignored.

So the keypoint config re-points `policy` at a sibling node, which replaces the parent's
outright. That is what ARCH_KEYS below re-ties, and what test_keypoint_policy_drops_search_keys
proves actually happened.
"""
import os
import sys

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGE_CFG = 'train_pusht_unet_bc_expert'
KEYPOINT_CFG = 'train_pusht_unet_bc_keypoint_expert'

# Every key whose value defines the MODEL or the SAMPLER. Not the observation: obs_encoder
# and shape_meta are exactly what the two arms are allowed to differ in.
ARCH_KEYS = ('diffusion_step_embed_dim', 'down_dims', 'kernel_size', 'n_groups',
             'cond_predict_scale', 'num_inference_steps', 'horizon', 'n_obs_steps',
             'n_action_steps', 'obs_as_global_cond')

# The 30 held-out episodes and the budget that produces the 10k checkpoint grid.
N_DEMOS = 176
N_TEST = 30
N_VAL = 0


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


@pytest.mark.parametrize('key', ARCH_KEYS)
def test_arch_matches_across_arms(cfgs, key):
    """The keypoint copy of the architecture must equal the image arm's definition."""
    image, keypoint = cfgs
    a = OmegaConf.to_container(image.policy[key], resolve=True) \
        if OmegaConf.is_config(image.policy[key]) else image.policy[key]
    b = OmegaConf.to_container(keypoint.policy[key], resolve=True) \
        if OmegaConf.is_config(keypoint.policy[key]) else keypoint.policy[key]
    assert a == b, (
        f'policy.{key} differs: image={a!r} keypoint={b!r}. The keypoint config carries a '
        f'copy of the image arm\'s architecture because it cannot inherit that block; '
        f'update diffusion_policy/config/{KEYPOINT_CFG}.yaml to match.')


def test_noise_scheduler_matches_across_arms(cfgs):
    """Same reverse process, key for key -- including _target_."""
    image, keypoint = cfgs
    a = OmegaConf.to_container(image.policy.noise_scheduler, resolve=True)
    b = OmegaConf.to_container(keypoint.policy.noise_scheduler, resolve=True)
    assert a == b, f'noise_scheduler differs:\n image={a}\n keypoint={b}'


def test_keypoint_policy_drops_search_keys(cfgs):
    """`policy: ${policy_keypoint}` must REPLACE the parent node, not merge into it.

    If it merged, these keys would survive and be forwarded into DDIMScheduler.step().
    """
    _, keypoint = cfgs
    leaked = sorted(k for k in keypoint.policy
                    if k.startswith('verifier_') or k in
                    ('search_context', 'selection', 'crop_shape', 'random_crop'))
    assert not leaked, (
        f'{leaked} survived into the keypoint policy block. DiffusionUnetImagePolicy '
        f'forwards unknown kwargs to noise_scheduler.step(), so these would raise at '
        f'sampling time.')
    assert keypoint.policy._target_.endswith('DiffusionUnetImagePolicy')


def test_keypoint_observation_is_keypoints_only(cfgs):
    """The one axis the two arms are allowed to differ in, asserted in both directions."""
    image, keypoint = cfgs
    assert set(image.shape_meta.obs) == {'image'}
    assert set(keypoint.shape_meta.obs) == {'keypoint', 'agent_pos'}
    assert keypoint.policy.obs_encoder._target_.endswith('FlattenObsEncoder')


@pytest.mark.parametrize('name', [IMAGE_CFG, KEYPOINT_CFG])
def test_split_and_budget(name):
    """Both arms train on all 176 demos, hold out the same 30, and share the 10k eval grid."""
    cfg = _compose(name)
    assert cfg.n_demos == N_DEMOS
    assert cfg.task.dataset.split_file.endswith(f'pusht_seed42_train{N_DEMOS}.json')
    assert cfg.task.dataset.n_test_episodes == N_TEST
    assert cfg.task.dataset.n_val_episodes == N_VAL
    assert cfg.training.max_gradient_steps == 200000
    assert cfg.training.checkpoint_every == 10000
    # Offline eval is the record; an in-training rollout would be a second, different
    # measurement of the same quantity.
    assert cfg.training.get('rollout_every_steps') is None
    # The lr schedule is deliberately NOT stretched to the 200k budget.
    assert cfg.training.lr_scheduler_kwargs.decay_steps == 77000


@pytest.mark.parametrize('name', [IMAGE_CFG, KEYPOINT_CFG])
def test_manifest_geometry_on_disk(name):
    """The manifest the configs point at really is 176 train / 0 val / 30 test."""
    import json
    import pathlib
    cfg = _compose(name)
    root = pathlib.Path(__file__).resolve().parents[1]
    manifest = json.loads((root / cfg.task.dataset.split_file).read_text())
    assert len(manifest['train']) == N_DEMOS
    assert len(manifest['val']) == N_VAL
    assert len(manifest['test']) == N_TEST
    # every episode accounted for exactly once
    all_idx = manifest['train'] + manifest['val'] + manifest['test']
    assert len(set(all_idx)) == len(all_idx) == manifest['n_episodes']


def test_keypoint_arm_declares_no_verifier():
    """verifier_tag null is what lets check_verifier_value pass on a policy that scores nothing."""
    cfg = _compose(KEYPOINT_CFG)
    assert cfg.verifier_tag is None
    assert 'verifier_value' not in cfg.policy


@pytest.mark.parametrize('name', [IMAGE_CFG, KEYPOINT_CFG])
def test_unfiltered_arms_own_the_bare_suffix(name):
    """transition_file null + empty split_suffix is the spelling check_transition_filter_labels wants.

    A moving-transition-filtered run at the same n_demos MUST set split_suffix or it would
    resume into these directories.
    """
    from diffusion_policy.dataset.pusht_image_dataset import check_transition_filter_labels
    cfg = _compose(name)
    assert cfg.get('transition_file', None) is None
    assert not str(cfg.get('split_suffix', '') or '')
    check_transition_filter_labels(cfg)


def test_run_names_are_distinct():
    image, keypoint = _compose(IMAGE_CFG), _compose(KEYPOINT_CFG)
    assert image.run_name != keypoint.run_name
    # task_name is also a hydra.run.dir component, so the two trees cannot collide
    assert image.task_name != keypoint.task_name
