"""Shared pieces for the four evaluations: the eval set, the policy, the two verifiers.

THE EVAL SET IS ST'S, NOT A SEEDED ONE. `recurrent_ppo`'s "fixed eval set" is seeded resets of
`PushTGymEnv` -- fixed in ST's *style* but not ST's *set*. What ST and BC are actually scored on
is the 50 held-out DEMO episodes named by the committed split manifest, whose indices are
identical across every `pusht_seed42*` manifest. Reusing `eval_search_pusht.get_split_states`
means the manifest, its checksum validation, and the `splits.json` cross-check all come along --
so an eval can never silently score a checkpoint against a partition it was not trained under.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np                                                          # noqa: E402

from diffusion_policy.common.replay_buffer import ReplayBuffer              # noqa: E402
from diffusion_policy.env.pusht.pusht_verifier import PushTVerifier         # noqa: E402
from eval_search_pusht import get_split_states, load_policy                 # noqa: E402
from sac.score import PushTQVerifier                                        # noqa: E402


def load_arm(checkpoint, device="cuda:0", split="test"):
    """(policy, cfg, init_states, episode_idxs) for an ST or BC checkpoint, on ITS OWN split."""
    policy, cfg = load_policy(checkpoint, device)
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))
    states, idxs = get_split_states(cfg, split, run_dir)
    print(f"[INFO] {split}: {len(idxs)} held-out demo episodes, indices {list(idxs[:8])}...")
    return policy, cfg, states, idxs


def replay_buffer(cfg):
    return ReplayBuffer.copy_from_path(
        cfg.task.dataset.zarr_path, keys=["img", "state", "action", "agent_pos", "block_pos", "keypoint"])


def build_verifier(spec, device="cuda:0", n_envs=32):
    """`spec` is a VALUE_FNS key (the sim verifier) or `q:<checkpoint>` (the learned one).

    One entry point for both, so every script takes `--verifier` and the comparison between a
    learned Q and the heuristic it replaces is a change of flag rather than a change of code.
    """
    if str(spec).startswith("q:"):
        return PushTQVerifier(spec[2:], device=device), "q"
    return PushTVerifier(n_envs=n_envs, legacy=False, use_async=True, value_fn=spec), spec


def verifier_label(spec):
    return "Q (learned)" if str(spec).startswith("q:") else f"{spec} (sim heuristic)"


def wilson(k, n, z=1.96):
    """95% CI for a rate, matching the repo's own `stats_util.wilson_interval` convention."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)
