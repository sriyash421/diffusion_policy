"""SAC on PushT, to learn a Q that can rank best-of-N candidates.

The PushT best-of-N verifier is currently a hand-written distance heuristic evaluated in a
sim. It ignores `agent_pos` by construction, so before the arm touches the block every
candidate scores identically and `argmax` is a tie-break -- on 28-35% of decisions overall and
68-73% of approach decisions. This package learns `Q*(s, chunk)` instead, whose pre-contact
signal is produced by the Bellman backup rather than by a hand-written approach term.
"""

import os

import torch as th

# ------------------------------------------------------------------ CPU thread cap
# `diffusion_policy.env.pusht` imports cv2, which initialises an OpenMP runtime. When that
# happens BEFORE torch is imported, torch's default thread count -- equal to the core count --
# becomes pathological for small tensors. Measured on this 24-core box, 50 gradient steps on a
# 2x256 MLP:
#
#     batch  16:  1-16 threads 0.017s   |   24 threads 4.696s
#     batch 512:  16 threads   0.035s   |   24 threads 7.854s
#
# It is a CLIFF at exactly the core count, not a gradient, and it is purely an artefact of
# import order: the same code with torch imported first runs at 0.02s on 24 threads. Capping
# below the core count fixes it in either order, and costs nothing -- these batches are far too
# small for 24-way parallelism to pay anyway (16 threads is the fastest measured point).
#
# Set SAC_TORCH_THREADS to override; 0 leaves torch's default alone.
_threads = int(os.environ.get("SAC_TORCH_THREADS", min(8, max(1, (os.cpu_count() or 2) - 1))))
if _threads > 0:
    th.set_num_threads(_threads)
