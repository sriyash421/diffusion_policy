"""How a search policy's executed action is picked out of its scored candidates.

Lives outside the policy because selection is a pure function of the score tensor: the
same rule serves the Gaussian and diffusion families, and the env runner and the workspace
both need it. `final_pass` is NOT here -- it is not a selection but an extra conditioned
generation, so it stays on the policy (see DiffusionTransformerSearchPolicy).
"""
import torch

# How the executed action is chosen. 'final_pass' is listed because it is a legal value of
# `policy.selection`, but it is handled by the policy, not by select_candidate below.
SELECTION_MODES = ('argmax', 'softmax', 'index', 'final_pass')


def select_candidate(actions, scores, mode='argmax', temperature=1.0, generator=None,
                     index=None):
    """Pick one chunk per batch row.

    actions (B, n, H, Da), scores (B, n) -> (B, H, Da).

    ``generator``: a torch.Generator the softmax draw is taken from. Pass one -- the
    policy owns a dedicated generator for exactly this. Selection and the diffusion
    sampler otherwise share the global RNG stream, and since DDIM at eta=0 draws only the
    initial `torch.randn` per generation, softmax's one extra draw shifts every subsequent
    noise vector. That made 'argmax' and 'softmax' rollouts diverge at n=1, where they
    take provably the same action -- an apparent selection effect that was only reseeding.
    """
    assert mode in ('argmax', 'softmax', 'index'), \
        f"select_candidate handles 'argmax'/'softmax'/'index'; got {mode!r} " \
        f"('final_pass' is generation, not selection -- the policy owns it)"
    B, n = actions.shape[0], actions.shape[1]
    arange = torch.arange(B, device=actions.device)
    if mode == 'index':
        # A FIXED candidate by generation order, ignoring the scores entirely -- the
        # verifier takes no part. `index` is 1-BASED, the way "the 8th candidate" is said
        # out loud; negatives count from the end (-1 == last). This is the read-out that
        # asks what one slot of the search is worth on its own, with no selection on top.
        assert index is not None, "selection 'index' needs policy.selection_index"
        i = int(index)
        assert i != 0, 'selection_index is 1-based; 0 is not a candidate'
        i = i - 1 if i > 0 else n + i
        assert 0 <= i < n, \
            f'selection_index {index} is out of range at n={n}; the candidate does not ' \
            f'exist, so a number here would silently be some other slot'
        return actions[:, i]
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
