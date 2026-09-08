"""Capture the search transformer's CROSS-attention weights.

WHY A HOOK. `nn.TransformerDecoderLayer` calls its `multihead_attn` with
`need_weights=False`, so the attention matrix is computed inside the fused kernel and
thrown away -- there is nothing to read off the module afterwards. `capture_cross_attention`
temporarily wraps each layer's `multihead_attn.forward` to force
`need_weights=True, average_attn_weights=False`, records the result, and restores the
original on exit. Nothing is mutated permanently and no weights change, so a captured
rollout is numerically the rollout that would have happened anyway -- with one caveat below.

WHAT THE AXES MEAN, for SearchTransformerForDiffusion (see its forward()):

    memory = [ obs_tokens (n_obs_steps) | action_value_tokens (max_context_actions) ]
    queries = the horizon trajectory tokens

so a captured map is (heads, H_queries, S_memory) per layer, and "how much does this policy
use the search context" is the attention mass on the action_value block versus the obs block.

    max_context_actions == K - 1.  ST k=16 -> 15 context tokens; ST k=1 -> ZERO, its memory
    is obs only. The UNet BC arm has no attention at all. So a context-usage comparison
    exists only within the k>1 arms.

THE SLOT AXIS IS FOLDED INTO THE BATCH. forward() reshapes (B, K, S, E) memory to
(B*K, S, E) when obs_cond carries a per-slot view, so a captured tensor's batch dimension is
B*K with slot varying fastest. `split_slots` undoes that; it must be told K, because B*K
cannot be factored from the tensor alone.

CAVEAT: forcing need_weights=True takes torch off the fused fast path and onto the math
kernel. Outputs agree to float tolerance but are not guaranteed bit-identical, so capture is
for analysis, not for reproducing a scored rollout.
"""
import contextlib

import torch
import torch.nn as nn


def _decoder_layers(model):
    """The TransformerDecoderLayers of a SearchTransformerForDiffusion, in depth order."""
    dec = getattr(model, 'decoder', None)
    if dec is None or not hasattr(dec, 'layers'):
        raise TypeError(
            f'{type(model).__name__} has no `decoder.layers`; cross-attention capture is '
            f'only defined for the transformer search policy. The UNet BC arm '
            f'(ConditionalUnet1D, FiLM conditioning) has no attention to capture.')
    return list(dec.layers)


@contextlib.contextmanager
def capture_cross_attention(policy):
    """Yield a dict {layer_index: [tensor(B, heads, H, S), ...]} filled during the block.

    One entry per forward pass of that layer, so a full DDIM rollout leaves
    `num_inference_steps` tensors per layer, in denoising order.
    """
    model = getattr(policy, 'model', policy)
    layers = _decoder_layers(model)
    captured = {i: [] for i in range(len(layers))}
    originals = []

    def make_wrapper(idx, mha):
        orig = mha.forward

        def wrapper(*args, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False      # keep the head axis
            out, w = orig(*args, **kwargs)
            if w is not None:
                captured[idx].append(w.detach().to('cpu'))
            return out, w
        return orig, wrapper

    try:
        for i, layer in enumerate(layers):
            mha = layer.multihead_attn
            orig, wrapper = make_wrapper(i, mha)
            originals.append((mha, orig))
            mha.forward = wrapper
        yield captured
    finally:
        for mha, orig in originals:
            mha.forward = orig


def split_slots(attn, n_slots):
    """(B*K, heads, H, S) -> (B, K, heads, H, S).

    forward() folds the candidate slot into the batch when obs_cond is per-slot, and the
    fold is `reshape(B*K, ...)` with slot varying fastest. B*K cannot be factored from the
    tensor alone, so `n_slots` must be supplied by the caller.
    """
    bk = attn.shape[0]
    if bk % n_slots != 0:
        raise ValueError(f'batch {bk} is not divisible by n_slots {n_slots}; the capture '
                         f'did not come from a per-slot decode')
    return attn.reshape(bk // n_slots, n_slots, *attn.shape[1:])


def context_mass(attn, n_obs_tokens):
    """Fraction of each query's cross-attention mass landing on the CONTEXT tokens.

    Rows of the attention matrix already sum to 1 over the memory, so this is just the sum
    over the action_value block. Returns the same shape as `attn` minus its last axis.
    """
    if attn.shape[-1] <= n_obs_tokens:
        return torch.zeros(attn.shape[:-1], dtype=attn.dtype)
    return attn[..., n_obs_tokens:].sum(-1)
