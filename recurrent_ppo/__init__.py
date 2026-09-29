"""Recurrent PPO on PushT: a policy, a value function, and a Q head, under observation corruption."""

import os

import torch as th


def cap_torch_threads(env_var="PUSHT_TORCH_THREADS"):
    """Keep torch off the last core, because cv2 makes the full count pathological.

    `diffusion_policy.env.pusht` imports cv2, which initialises an OpenMP runtime. When that
    happens BEFORE torch is imported -- which is what a module-level `from ...pusht_gym import`
    does -- torch's default thread count, equal to the core count, becomes catastrophic for
    small tensors. Measured on a 24-core box, 50 gradient steps on a 2x256 MLP:

        batch  16:  1-16 threads 0.017s   |   24 threads 4.696s
        batch 512:  16 threads   0.035s   |   24 threads 7.854s

    A CLIFF at exactly the core count, not a gradient, and purely an artefact of import order:
    the same code with torch imported first runs at 0.02s on 24 threads. `cv2.setNumThreads(0)`,
    the usual mitigation, does nothing -- it is not cv2's own pool. Capping below the core count
    fixes it either way and costs nothing, since 16 threads was the fastest measured point and
    these batches never justified 24-way parallelism.

    Both packages here train MLPs of that size on CPU, so both call this. Set the env var to
    override; 0 leaves torch's default alone.
    """
    n = int(os.environ.get(env_var, min(8, max(1, (os.cpu_count() or 2) - 1))))
    if n > 0:
        th.set_num_threads(n)
    return n


cap_torch_threads()
