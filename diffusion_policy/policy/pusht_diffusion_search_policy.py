"""PushT diffusion search: the diffusion denoiser plus PushT's verifier hooks."""
from diffusion_policy.policy.diffusion_transformer_search_policy import (
    DiffusionTransformerSearchPolicy,
)
from diffusion_policy.policy.pusht_search_mixin import (  # noqa: F401  (re-exported)
    PushTSearchMixin, SEARCH_CONTEXTS,
)


class PushTDiffusionSearchPolicy(PushTSearchMixin, DiffusionTransformerSearchPolicy):
    """Everything is inherited: the denoiser and search loop from the base, the verifier
    hooks from the mixin. PushTSearchMixin comes FIRST so its hooks win over the base's."""
