"""Prove the search transformer degenerates to a single-sample policy at k=1.

Width 1 is an OVERRIDE (`n_candidates=1`), not a separate config: encoder, backbone,
normalizer, trainer and eval harness are then byte-identical to the k=16 arm and the only
difference is the search itself. k=1 gives ``max_context_actions = 0``, so every candidate
is drawn with an empty context:

  * training  = plain denoising on the expert action, no candidates, no verifier;
  * n=1 eval  = a single ordinary sample;
  * n>1 eval  = best-of-n over i.i.d. samples, i.e. width 1 handed the same test-time
    search budget as the search arms. That is the comparison that isolates *learned*
    search context from *test-time* sampling.

This asserts all three actually hold, because a silently-degenerate baseline (identical
candidates, or a verifier pool spawned per rollout) would look like a valid experiment.

NOTE ON NAMING: this arm is ST k=1, NOT "BC". In this repo BC means the diffusion UNet
(`train_pusht_unet_bc`) and nothing else.

    python scripts/st_k1_smoke.py
"""
import sys
import pathlib
import time

import torch

ROOT = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT)

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

OmegaConf.register_new_resolver('eval', eval, replace=True)

torch.manual_seed(0)
with initialize_config_dir(config_dir=str(pathlib.Path(ROOT, 'diffusion_policy/config')),
                           version_base=None):
    cfg = compose(config_name='train_pusht_diffusion_search',
                  overrides=['n_candidates=1', 'training.device=cpu'])

t0 = time.time()
policy = hydra.utils.instantiate(cfg.policy)
print(f'policy built: {type(policy).__name__} ({time.time()-t0:.1f}s)')
print('  max_actions         =', policy.max_actions)
print('  max_context_actions =', getattr(policy.model, 'max_context_actions', 'n/a'))
print('  params:', round(sum(p.numel() for p in policy.parameters()) / 1e6, 2), 'M')
assert policy.verifier._vec is None, 'verifier pool spawned at construction'

from diffusion_policy.model.common.normalizer import LinearNormalizer

B, T, To = 2, cfg.horizon, cfg.n_obs_steps
fit = {'action': torch.rand(64, 2) * 100, 'image': torch.rand(64, 3, 96, 96),
       'agent_pos': torch.rand(64, 2) * 100, 'feedback': torch.rand(64, 16) * 10}
nrm = LinearNormalizer()
nrm.fit(fit)
policy.set_normalizer(nrm)

batch = {'action': torch.rand(B, T, 2) * 100,
         'obs': {'image': torch.rand(B, To, 3, 96, 96),
                 'agent_pos': torch.rand(B, To, 2) * 100,
                 'feedback': torch.rand(B, To, 16) * 10}}

loss = policy.compute_loss(batch)
loss.backward()
gnorm = sum(float(p.grad.norm() ** 2)
            for p in policy.parameters() if p.grad is not None) ** 0.5
print(f'  compute_loss = {float(loss):.4f} | grad norm = {gnorm:.4f}')
assert torch.isfinite(loss) and gnorm > 0, 'no gradient reached the parameters'
assert policy.verifier._vec is None, 'training spawned the verifier pool'

policy.eval()
with torch.no_grad():
    out = policy.predict_action(batch['obs'])
print('  predict_action ->', {k: tuple(v.shape) for k, v in out.items()})
assert policy.verifier._vec is None, 'ST k=1 rollout spawned the verifier pool'
print('OK  trains and acts as plain ST k=1; verifier never spawned')

# best-of-n: candidates must be genuinely different draws, or "ST k=1 + search budget" is
# a no-op and every n would score identically
with torch.no_grad():
    acts, _, scores = policy.predict_n_actions(
        batch['obs'], verifier=policy.verifier, n_actions=4, return_scores=True)
flat = acts.reshape(acts.shape[0], acts.shape[1], -1)
spread = float(torch.cdist(flat[0], flat[0]).max())
print(f'  predict_n_actions(4) -> actions {tuple(acts.shape)} scores {tuple(scores.shape)}')
print(f'  max pairwise candidate distance (sample 0): {spread:.3f}')
assert acts.shape[1] == 4, f'expected 4 candidates, got {acts.shape[1]}'
assert spread > 1e-3, 'candidates are identical -- best-of-n would be a no-op'
print('OK  best-of-n draws distinct i.i.d. candidates at max_actions=1')

print('\nSTAGE OK: max_actions=1 is a faithful ST k=1 baseline')
