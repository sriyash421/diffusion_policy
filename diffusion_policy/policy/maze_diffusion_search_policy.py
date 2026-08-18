"""The maze arm of the diffusion search transformer.

Supplies only the verifier; everything else -- network, search procedure, loss -- comes
from DiffusionTransformerSearchPolicy. Symmetric with PushTDiffusionSearchPolicy, and the
reason the base's `_build_verifier` is abstract: neither task's verifier is a default.
"""
from diffusion_policy.policy.diffusion_transformer_search_policy import (
    DiffusionTransformerSearchPolicy,
)


class MazeDiffusionSearchPolicy(DiffusionTransformerSearchPolicy):
    def _build_verifier(self, **kwargs):
        """`l2s` is imported here so the base module stays importable without it."""
        from l2s.verifier import MazeVerifier
        return MazeVerifier(
            maze_path=kwargs.get('maze_path', None),
            device=kwargs.get('device', 'cpu'),
            noise=kwargs.get('verifier_noise', 0.0),
        )
