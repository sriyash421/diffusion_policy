"""How a search policy's executed action is picked out of its scored candidates.

Lives outside the policy because selection is a pure function of the score tensor: the
same rule serves the Gaussian and diffusion families, and the env runner and the workspace
both need it. `final_pass` is NOT here -- it is not a selection but an extra conditioned
generation, so it stays on the policy (see DiffusionTransformerSearchPolicy).
"""
import torch

# How the executed action is chosen. 'final_pass' is listed because it is a legal value of
# `policy.selection`, but it is handled by the policy, not by select_candidate below.
SELECTION_MODES = ('argmax', 'softmax', 'final_pass')


def select_candidate(actions, scores, mode='argmax', temperature=1.0, generator=None):
    """Pick one chunk per batch row.

    actions (B, n, H, Da), scores (B, n) -> (B, H, Da).

    ``generator``: a torch.Generator the softmax draw is taken from. Pass one -- the
    policy owns a dedicated generator for exactly this. Selection and the diffusion
    sampler otherwise share the global RNG stream, and since DDIM at eta=0 draws only the
    initial `torch.randn` per generation, softmax's one extra draw shifts every subsequent
    noise vector. That made 'argmax' and 'softmax' rollouts diverge at n=1, where they
    take provably the same action -- an apparent selection effect that was only reseeding.
    """
    assert mode in ('argmax', 'softmax'), \
        f"select_candidate handles 'argmax'/'softmax'; got {mode!r} " \
        f"('final_pass' is generation, not selection -- the policy owns it)"
    B = actions.shape[0]
    arange = torch.arange(B, device=actions.device)
    if mode == 'argmax':
        pick = scores.argmax(dim=1)
    else:
        # z-score across the n candidates, then sample. unbiased=False so a single
        # candidate (n=1) gives std 0 rather than NaN; the eps then leaves z==0, i.e. a
        # uniform draw over one option -- the same action argmax would have taken, so the
        # n=1 column is comparable by construction rather than by luck.
        mu = scores.mean(dim=1, keepdim=True)
        sd = scores.std(dim=1, unbiased=False, keepdim=True)
        z = (scores - mu) / (sd + 1e-6)
        probs = torch.softmax(z / temperature, dim=1)
        # Inverse-CDF rather than Categorical.sample(): torch.distributions takes no
        # generator, so it can only draw from the global stream. Drawn on CPU so one
        # generator serves every device.
        u = torch.rand(B, 1, generator=generator).to(probs.device, probs.dtype)
        pick = (probs.cumsum(dim=1) < u).sum(dim=1).clamp_(max=probs.shape[1] - 1)
    return actions[arange, pick]
