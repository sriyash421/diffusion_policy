"""Smoke test for the learned-Q verifier -- the parts a CPU unit test cannot reach.

`unit_tests/test_q_context.py` pins the normalizer's arithmetic on a stub. Four things it
cannot check, because each needs a real policy, the real SAC weights and a GPU:

  1. `_build_verifier` returns a `PushTQVerifier` and forks NO sim pool. A PushTVerifier
     lazily forks 32 workers on its FIRST rollout, inside `search_candidates` -- so the
     child count has to be compared across that call, not merely at construction.
  2. The running-normalizer buffers are registered AND ride in `state_dict()`. If they did
     not, an arm would resume and evaluate with statistics it never trained under, and
     nothing would report it.
  3. The context is live. A verifier that silently returned a constant would train perfectly
     happily on a dead input; the loss would not say so, and 100k steps would be wasted. So
     this asserts real variance, that the context is NOT the raw score, and that it respects
     the clamp.
  4. The statistics freeze when the training loop is not driving. They must not drift with
     evaluation order.

    PYTHONPATH=$PWD python scripts/q_verifier_smoke.py

Needs a GPU and the real SAC checkpoint. Run it on a compute node -- the login node is one
shared ~10GB cgroup and loading the zarr there will be killed.
"""
import multiprocessing as mp
import pathlib
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data.dataloader import default_collate

from diffusion_policy.common.pytorch_util import dict_apply

# the configs use ${eval:...}; train.py registers this resolver, and a standalone entry
# point has to do the same or every interpolation raises UnsupportedInterpolationType.
OmegaConf.register_new_resolver('eval', eval, replace=True)

CFG = 'train_pusht_diffusion_search'
OVERRIDES = [
    'n_candidates=4',
    'task.dataset.split_file=diffusion_policy/config/splits/pusht_blockquad_bottomleft_train137.json',
    'n_demos=137', 'split_suffix=_split-blq',
    'task.dataset.n_test_episodes=50', 'task.dataset.n_val_episodes=19',
    'verifier_tag=q_sac_all',
]

# initialize_config_dir with an ABSOLUTE path: hydra.initialize resolves config_path
# relative to the CALLING FILE, which is scripts/ here, not the repo root.
REPO = pathlib.Path(__file__).resolve().parents[1]
with hydra.initialize_config_dir(config_dir=str(REPO / 'diffusion_policy' / 'config'),
                                 version_base=None):
    cfg = hydra.compose(config_name=CFG, overrides=OVERRIDES)
OmegaConf.resolve(cfg)

device = 'cuda:0'
policy = hydra.utils.instantiate(cfg.policy).to(device).eval()

# buffers registered only for a learned-value arm
assert hasattr(policy, 'q_norm_mean'), 'no running-value buffers on a q arm'
assert 'q_norm_mean' in policy.state_dict(), 'buffers must ride in the state_dict'
print('[ok] buffers registered and checkpointed')

dataset = hydra.utils.instantiate(cfg.task.dataset)
policy.set_normalizer(dataset.get_normalizer())
policy = policy.to(device)

before = len(mp.active_children())
verifier = policy.verifier
from sac.score import PushTQVerifier
assert isinstance(verifier, PushTQVerifier), type(verifier)
print(f'[ok] verifier is {type(verifier).__name__}')

batch = default_collate([dataset[i] for i in range(8)])
obs = dict_apply(batch['obs'], lambda x: x.to(device))

policy.set_value_norm_training(True)
with torch.no_grad():
    actions, values, scores = policy.search_candidates(
        obs, verifier=verifier, n_actions=4, return_scores=True)

assert len(mp.active_children()) == before, \
    f'a sim pool was forked: {before} -> {len(mp.active_children())} children'
print(f'[ok] no sim subprocess forked (children {before})')

v = values.float().cpu().numpy()
s = scores.float().cpu().numpy()
print(f'    raw Q    : mean {s.mean():.4f}  within-decision std {s.std(axis=1).mean():.6f}')
print(f'    context  : mean {v.mean():+.4f}  std {v.std():.4f}  '
      f'range [{v.min():+.3f}, {v.max():+.3f}]')
assert v.std() > 1e-3, 'the context is constant -- a dead input'
assert not np.allclose(v, s), 'the context is the RAW Q; the normalizer did not run'
assert np.abs(v).max() <= 5.0 + 1e-5, 'context escaped the +-5 clamp'
print('[ok] context is a live, clamped z-score, distinct from the raw score')

# ranking is on the RAW Q, per decision
for i in range(v.shape[0]):
    assert np.array_equal(np.argsort(s[i]), np.argsort(s[i])), 'scores must be raw'
assert bool(policy.q_norm_inited), 'statistics never seeded'
print(f'[ok] statistics seeded: mean {float(policy.q_norm_mean):.4f} '
      f'var {float(policy.q_norm_var):.3e}')

# frozen by default
policy.set_value_norm_training(False)
m0 = policy.q_norm_mean.clone()
with torch.no_grad():
    policy.search_candidates(obs, verifier=verifier, n_actions=4, return_scores=True)
assert torch.equal(policy.q_norm_mean, m0), 'statistics moved while frozen'
print('[ok] statistics frozen when not training')
print('\nALL CHECKS PASSED')
