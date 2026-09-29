"""PushT Gaussian search: maze's Gaussian search transformer on PushT's verifier.

Mirrors SearchPolicy exactly -- fused obs token, one token per candidate, Gaussian heads,
NLL loss, and both the `mask_obs` / `concat_obs` structural variants -- and takes its
verifier, search-context modes and candidate scoring from PushTSearchMixin, the same hooks
the diffusion arm uses. The two PushT arms therefore differ only in how one candidate is
produced (one rsample vs a denoising loop).
"""
from diffusion_policy.policy.pusht_search_mixin import (  # noqa: F401  (re-exported)
    PushTSearchMixin, SEARCH_CONTEXTS,
)
from diffusion_policy.policy.search_policy import SearchPolicy


class PushTGaussianSearchPolicy(PushTSearchMixin, SearchPolicy):
    """PushTSearchMixin comes FIRST so its hooks win over SearchPolicy's maze ones."""
