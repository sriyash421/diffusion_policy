"""How a search policy's executed action is picked out of its scored candidates.

Lives outside the policy because selection is a pure function of the score tensor: the
same rule serves the Gaussian and diffusion families, and the env runner and the workspace
both need it. `final_pass` is NOT here -- it is not a selection but an extra conditioned
generation, so it stays on the policy (see DiffusionTransformerSearchPolicy).
"""
import torch

# How the executed action is chosen. 'final_pass' is listed because it is a legal value of
# `policy.selection`, but it is handled by the policy, not by select_candidate below.
SELECTION_MODES = ('argmax', 'softmax', 'final_pass', 'index')


def resolve_index(index, n):
    """A (possibly negative) candidate index, resolved against the n ACTUALLY generated.

    Negative counts from the end, so -1 is the last (most deeply conditioned) candidate.
    Clamped rather than wrapped: a request for the 8th-from-last at n=4 takes candidate 0,
    instead of silently landing on a different slot than the one the arm is named after.
    """
    idx = int(index)
    if idx < 0:
        idx = n + idx
    return max(0, min(n - 1, idx))


def select_candidate(actions, scores, mode='argmax', temperature=1.0,
                     window=None, index=-1):
    """Pick one chunk per batch row.

    actions (B, n, H, Da), scores (B, n) -> ``(action (B, H, Da), pick (B,))``.

    ``pick`` is returned in FULL-candidate coordinates (i.e. indexes `scores`' n axis) so a
    caller logging it beside the scores does not have to re-derive it -- a re-derivation
    cannot reproduce a 'softmax' draw at all, and would silently disagree with what ran.

    ``window`` narrows the pool 'argmax'/'softmax' rank over to the LAST W candidates
    (None == all of them). It is a selection knob only: all n are still generated and still
    scored, so it costs nothing extra and `scores` stays the full (B, n). It exists because
    candidate order is meaningful here -- candidate k is conditioned on candidates 0..k-1,
    so the trailing ones are the deeply-conditioned ones, and "argmax over the last 8 of
    16" is a different question from "argmax over all 16".

    ``index`` is read only under mode 'index'.
    """
    assert mode in ('argmax', 'softmax', 'index'), \
        f"select_candidate handles 'argmax'/'softmax'/'index'; got {mode!r} " \
        f"('final_pass' is generation, not selection -- the policy owns it)"
    B, n = actions.shape[0], actions.shape[1]
    arange = torch.arange(B, device=actions.device)

    if mode == 'index':
        # A FIXED slot, ignoring the scores entirely. The control for the two below: it
        # isolates how much of a best-of-n gain comes from the ranking rather than from
        # candidate k simply being conditioned on candidates 0..k-1.
        idx = resolve_index(index, n)
        pick = torch.full((B,), idx, dtype=torch.long, device=actions.device)
        return actions[:, idx], pick

    # `lo` is the offset of the ranking window in the full candidate axis, so `pick` comes
    # back in full-candidate coordinates -- callers pairing it with `scores` (always all n)
    # would otherwise be off by `lo`.
    lo = 0 if window is None else max(0, n - int(window))
    pool = scores[:, lo:]                                       # (B, n-lo)
    if mode == 'argmax':
        pick = pool.argmax(dim=1) + lo
    else:
        # z-score across the pooled candidates, then sample. unbiased=False so a single
        # candidate (n=1) gives std 0 rather than NaN; the eps then leaves z==0, i.e. a
        # uniform draw over one option -- the same action argmax would have taken, so the
        # n=1 column is comparable by construction rather than by luck. Standardizing over
        # the POOL, not over all n, keeps T in units of "sd of the candidates being chosen
        # among" in both the windowed and unwindowed cases.
        mu = pool.mean(dim=1, keepdim=True)
        sd = pool.std(dim=1, unbiased=False, keepdim=True)
        z = (pool - mu) / (sd + 1e-6)
        pick = torch.distributions.Categorical(logits=z / temperature).sample() + lo
    return actions[arange, pick], pick
