"""One GPT-2 trunk builder, shared by every policy that runs a causal transformer over a
flat token stream (the Gaussian search policies, and the diffusion policy's
`cond_encoder: gpt2` conditioning encoder).
"""


def build_gpt2_trunk(n_emb, n_head, n_layer, n_positions,
                     p_drop_emb=0.0, p_drop_attn=0.1):
    """A causal GPT2Model over `inputs_embeds` -- no token ids anywhere.

    `transformers` is imported HERE rather than at module scope because it is a NEW
    dependency: conda_environment.yaml did not declare it until the gpt2 cond_encoder
    existed, so any env predating that still lacks it. A top-level import would break every
    arm on such an env -- including the ones that never build this trunk -- whereas a lazy
    one fails only where it is actually used, with the traceback pointing here.

    vocab_size=1 is load-bearing, not tidiness. GPT2Config defaults to 50257, and since
    nothing here ever passes token ids, that would be a `wte` embedding table that is never
    read: 12.87M parameters at n_emb 256, i.e. 79% of the whole trunk (16.29M -> 3.42M with
    vocab_size=1). n_positions is likewise sized to the caller's sequence, not left at 1024.
    """
    from transformers import GPT2Config, GPT2Model

    return GPT2Model(GPT2Config(
        vocab_size=1,
        n_positions=n_positions,
        n_embd=n_emb,
        n_layer=n_layer,
        n_head=n_head,
        # Mapped from the caller's own dropout knobs rather than a hardcoded 0.1: where the
        # trunk is the variable under test, regularization has to be held fixed where the
        # nn.TransformerEncoder had it (`dropout=p_drop_attn` on the layer, `p_drop_emb` on
        # the embedding dropout).
        resid_pdrop=p_drop_attn,
        attn_pdrop=p_drop_attn,
        embd_pdrop=p_drop_emb,
    ))
