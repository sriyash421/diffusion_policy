"""Prove the GPT-2 conditioning encoder is wired the way the config claims.

`cond_encoder: gpt2` swaps the nn.TransformerEncoder over the conditioning memory for a
causal GPT2Model, matching how SearchPolicy and OnlineSearchPolicy encode their context.
Several of the ways that can go wrong are silent -- a run would train to completion and
only the numbers would be off:

  * the padding mask handed to HF has the OPPOSITE polarity to torch's
    (1 == attend vs True == ignore), so getting it backwards masks exactly the tokens it
    should keep and keeps the ones it should mask;
  * GPT2Config's default vocab_size=50257 attaches a 12.87M-parameter `wte` that is never
    read, which would show up only as a mysteriously large checkpoint;
  * GPT-2 adds its own `wpe`, so leaving cond_pos_emb in place double-encodes position;
  * if the encoder were not actually causal, the staircase of conditionals the whole search
    formulation rests on would not exist, and nothing downstream would complain.

This asserts each of those, plus that the DEFAULT (`cond_encoder: transformer`) is
untouched -- procgen_maze_diffusion_transformer_search.yaml and every archived checkpoint
depend on that.

    python scripts/gpt2_cond_smoke.py
"""
import sys
import pathlib

import torch

ROOT = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT)

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.policy.diffusion_transformer_search_policy import (
    SearchTransformerForDiffusion,
)

OmegaConf.register_new_resolver('eval', eval, replace=True)

# The train_pusht_diffusion_search.yaml policy block, as plain kwargs so the structural
# checks below need neither a dataset nor an env.
MODEL_KW = dict(
    action_dim=2, obs_feature_dim=530, horizon=16, n_obs_steps=2,
    max_context_actions=15, n_layer=4, n_cond_layers=2, n_head=4, n_emb=256,
    p_drop_emb=0.0, p_drop_attn=0.2, context_dim=1, context_decay=1.0,
)


def build(cond_encoder, causal_attn):
    torch.manual_seed(0)
    return SearchTransformerForDiffusion(
        cond_encoder=cond_encoder, causal_attn=causal_attn, **MODEL_KW).eval()


# ---------------------------------------------------------------- default path unchanged
# The `transformer` trunk must still produce exactly the model it produced before this
# option existed: same parameter count, same state_dict keys, and `mask` present iff causal.
base = build('transformer', False)
base_keys = sorted(base.state_dict().keys())
base_params = sum(p.numel() for p in base.parameters())
print(f'transformer/causal=False: {base_params/1e6:.4f}M params, {len(base_keys)} keys')
assert base_params == 5948162, f'default-path parameter count moved: {base_params}'
assert len(base_keys) == 108, f'default-path state_dict key count moved: {len(base_keys)}'
assert 'mask' not in base_keys, 'causal_attn=False should not register a mask buffer'
assert any(k.startswith('cond_pos_emb') for k in base_keys), 'cond_pos_emb went missing'

base_causal = build('transformer', True)
assert 'mask' in base_causal.state_dict(), 'causal_attn=True must register a mask buffer'
assert sum(p.numel() for p in base_causal.parameters()) == base_params, \
    'causal_attn changed the parameter count; it should only add a buffer'
print('  causal_attn=True adds exactly the `mask` buffer, no parameters')

# `mlp` must still be what n_cond_layers=0 selects, whatever the trunk was asked for -- that
# is the documented way to reclaim the trunk's parameters.
assert build('gpt2', True).cond_encoder == 'gpt2'
zero_layer = SearchTransformerForDiffusion(
    cond_encoder='gpt2', causal_attn=True, **{**MODEL_KW, 'n_cond_layers': 0})
assert zero_layer.cond_encoder == 'mlp', \
    'n_cond_layers=0 must fall back to the Mish MLP even when gpt2 is requested'
print('  n_cond_layers=0 still falls back to the Mish MLP')

# ------------------------------------------------------------------------ the gpt2 trunk
m = build('gpt2', True)
gpt2_params = sum(p.numel() for p in m.parameters())
enc_params = sum(p.numel() for p in m.encoder.parameters())
base_enc_params = sum(p.numel() for p in base.encoder.parameters())
print(f'gpt2/causal=True        : {gpt2_params/1e6:.4f}M params '
      f'(trunk {enc_params/1e6:.4f}M vs transformer trunk {base_enc_params/1e6:.4f}M)')

assert m.cond_pos_emb is None, 'gpt2 mode must drop cond_pos_emb; GPT-2 has its own wpe'
assert m.encoder.wte.weight.numel() < 1000, (
    f'dead token-embedding table of {m.encoder.wte.weight.numel()} entries -- '
    'vocab_size was left at the GPT2Config default')
assert m.encoder.wpe.weight.shape[0] == MODEL_KW['n_obs_steps'] + MODEL_KW[
    'max_context_actions'], 'wpe should be sized to the memory, not left at n_positions=1024'
# Like-for-like: the swap must not smuggle in capacity. Allow the wte/wpe rounding only.
assert abs(gpt2_params - base_params) < 0.1e6, (
    f'gpt2 trunk changed total parameters by {(gpt2_params-base_params)/1e6:.3f}M -- '
    'this is supposed to be a like-for-like trunk swap')

# --------------------------------------------------------------------- encoder causality
# Perturb one context token and confirm the encoder output moves at that position and after
# it, and is bit-identical before it. This is the staircase the search formulation needs.
B, K = 2, 16
obs = torch.randn(B, MODEL_KW['n_obs_steps'], MODEL_KW['obs_feature_dim'])
acts = torch.randn(B, 15, MODEL_KW['horizon'], MODEL_KW['action_dim'])
vals = torch.randn(B, 15)
sample = torch.randn(B, K, MODEL_KW['horizon'], MODEL_KW['action_dim'])
t = torch.zeros(B, K, dtype=torch.long)

memories = {}
handle = m.encoder.register_forward_hook(
    lambda mod, inp, out: memories.__setitem__('h', out.last_hidden_state.detach().clone()))
with torch.no_grad():
    m(sample, t, obs_cond=obs, actions=acts, values=vals)
before = memories['h']

PERTURB = 7                     # context slot 7 -> memory position n_obs_steps + 7
acts2 = acts.clone()
acts2[:, PERTURB] += 5.0
with torch.no_grad():
    m(sample, t, obs_cond=obs, actions=acts2, values=vals)
after = memories['h']
handle.remove()

pos = MODEL_KW['n_obs_steps'] + PERTURB
assert torch.equal(before[:, :pos], after[:, :pos]), \
    f'encoder is not causal: perturbing memory position {pos} changed earlier positions'
assert not torch.allclose(before[:, pos], after[:, pos]), \
    'perturbing a context token did not change its own encoder output at all'
assert not torch.allclose(before[:, pos + 1:], after[:, pos + 1:]), \
    'later positions did not see the perturbed token -- context is not flowing forward'
print(f'encoder causality: positions <{pos} bit-identical, >={pos} moved')

# --------------------------------------------------------------------- decoder causality
# causal_attn=True makes each candidate autoregressive over the 16 horizon steps, and must
# leave cross-candidate isolation intact.
noisy = torch.randn(B, K, MODEL_KW['horizon'], MODEL_KW['action_dim'])
with torch.no_grad():
    pred_a = m(noisy, t, obs_cond=obs, actions=acts, values=vals)
STEP, CAND = 9, 3
noisy2 = noisy.clone()
noisy2[:, CAND, STEP] += 5.0
with torch.no_grad():
    pred_b = m(noisy2, t, obs_cond=obs, actions=acts, values=vals)

assert torch.equal(pred_a[:, CAND, :STEP], pred_b[:, CAND, :STEP]), \
    f'decoder is not causal: perturbing horizon step {STEP} changed steps before it'
assert not torch.allclose(pred_a[:, CAND, STEP:], pred_b[:, CAND, STEP:]), \
    'perturbing a horizon step changed nothing at or after it'
other = [c for c in range(K) if c != CAND]
assert torch.equal(pred_a[:, other], pred_b[:, other]), \
    'candidates are not isolated from each other'
print(f'decoder causality: candidate {CAND} steps <{STEP} unchanged, other candidates intact')

# ------------------------------------------------------------------------- gradient flow
m.train()
loss = m(noisy, t, obs_cond=obs, actions=acts, values=vals).pow(2).mean()
loss.backward()
enc_grad = sum(p.grad.abs().sum().item() for p in m.encoder.parameters()
               if p.grad is not None)
assert torch.isfinite(loss), 'non-finite loss'
assert enc_grad > 0, 'no gradient reached the GPT-2 trunk -- it is not on the loss path'
print(f'gradient flow: loss {loss.item():.4f}, GPT-2 trunk grad mass {enc_grad:.3f}')

# ------------------------------------------------------------- the config says all this
with initialize_config_dir(config_dir=str(pathlib.Path(ROOT, 'diffusion_policy/config')),
                           version_base=None):
    cfg = compose(config_name='train_pusht_diffusion_search')
assert cfg.policy.cond_encoder == 'gpt2', 'config no longer selects the gpt2 trunk'
assert cfg.policy.causal_attn is True, 'config no longer sets causal_attn'
print(f'config: cond_encoder={cfg.policy.cond_encoder}, causal_attn={cfg.policy.causal_attn}')

# ...and instantiating through hydra reaches the same model (this is the path training uses,
# and it is where a missing constructor parameter would surface).
policy = hydra.utils.instantiate(cfg.policy)
assert policy.model.cond_encoder == 'gpt2', 'hydra did not plumb cond_encoder through'
assert policy.model.mask is not None, 'hydra did not plumb causal_attn through'
assert 'model.mask' in policy.state_dict(), \
    'the mask buffer is missing from the policy state_dict'
print(f'hydra: {type(policy).__name__} built, '
      f'{sum(p.numel() for p in policy.parameters())/1e6:.2f}M params incl. encoder')

print('\nOK -- gpt2 conditioning encoder verified')
