# PushT offline search — code walkthrough

A read-through of the offline conditional-diffusion search policy on PushT images
(branch `harine/pushT`), plus two conceptual notes that came out of reading it: what the
diffusion objective actually trains, and what the obs encoder / feature dim really are.

Companion to [README_pusht.md](README_pusht.md), which covers *how to run* things. This
file covers *what the code does*.

---

## Part 1 — The offline search stack

### The shape of the idea

The policy is a diffusion transformer that denoises an action chunk conditioned on

* **(a)** the observation, and
* **(b)** a list of action chunks it already tried, each tagged with how well that chunk did.

Training teaches it: *given these k previous attempts and their outcomes, output the expert
action*. At eval it draws candidates one at a time, each seeing all the previous ones, and
executes the highest-scoring one.

"Offline" means the search happens inside the loss on dataset windows — no environment in
the training loop except a throwaway physics sim used as the scorer.

### 1. Entry point

[train_pusht_diffusion_search.yaml](diffusion_policy/config/train_pusht_diffusion_search.yaml)
→ [`TrainMLPImageWorkspace`](diffusion_policy/workspace/train_mlp_image_workspace.py)
(misnamed; it is the generic offline trainer). Task is
[pusht_image_search.yaml](diffusion_policy/config/task/pusht_image_search.yaml).

The identity keys at the top of the train config (`exp_name`, `trainer`, `search_context`,
`corrupt_obs`, `n_demos`, `training.seed`) determine both the policy's behaviour and the
output directory, so an ablation arm is readable off the path and a relaunch resumes rather
than forking into a new timestamped dir.

### 2. Data

[`PushTImageDataset`](diffusion_policy/dataset/pusht_image_dataset.py) emits windows of
`horizon=16`:

```
obs = {image (3,96,96), agent_pos (2), feedback (16)}
action = (16, 2)
```

`feedback` is the load-bearing key —
[`compute_feedback_from_pose`](diffusion_policy/env/pusht/feedback_util.py#L51) gives the 8
T-keypoint displacements between goal pose and achieved pose (16 numbers). It is an
**invertible** function of the block pose
([`block_pose_from_feedback`](diffusion_policy/env/pusht/feedback_util.py#L67)), which is why
`block_pos` is deliberately kept out of `shape_meta`: anything needing the block pose (the
verifier, resetting a sim) reconstructs it from a *declared* obs key, so no privileged
side-channel rides along in the obs dict.

Splits are read from a pinned manifest ([config/splits/](diffusion_policy/config/splits/)),
not derived — the `n_*_episodes` config keys are *validated* against it and a disagreement
raises. That is the fix for the 29 → 25 silent training-budget drift documented in
[README_pusht.md](README_pusht.md).

### 3. The network

[`SearchTransformerForDiffusion`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L36)
— an encoder/decoder transformer.

**Memory (what to condition on)** is `n_obs_steps` obs tokens ++ `max_context_actions = 15`
context tokens. Each context token is
[`action_value_emb`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L71)
applied to `[flattened candidate chunk (16x2) || feedback (context_dim)]` — one token per
previous candidate, carrying both *what it did* and *how it went*.

**Target** is the noisy trajectory, `K_decode` candidates at once, flattened to
`K * horizon` tokens.

Three masks do the real work:

* [`_build_tgt_mask`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L190) —
  candidates cannot see each other's tokens (block-diagonal); `causal_attn: False` in this
  config, so it is full attention within a chunk.
* [`_build_memory_masks`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L223)
  — **this is the trick.** With `context_lengths=None` and `K_decode > 1` it sets
  `context_lengths = arange(K)`, so decode-slot *k* may attend only to the first *k* context
  tokens. Slot 0 sees nothing (pure BC), slot 15 sees all 15.
* [`_build_encoder_mask`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L206)
  — inside the conditioning encoder, obs tokens see only obs; context token *i* sees obs plus
  context <= *i*.

Net effect: one forward pass trains **every prefix length of the search chain
simultaneously**. That is what makes the offline loss affordable relative to the 16
sequential samples eval performs.

### 4. The verifier

[`PushTVerifier`](diffusion_policy/env/pusht/pusht_verifier.py). PushT has no analytic value
function, so it brute-force simulates: reset a real `PushTEnv` to the state reconstructed
from `(agent_pos, feedback)`, step the candidate's action chunk, and score

```
value = -mean_kp ||goal_kp - achieved_kp||        (0 at the goal, negative otherwise)
```

A persistent pool of `verifier_n_envs=32` envs steps in parallel via `AsyncVectorEnv`,
batched in chunks with the last partial chunk padded. Two details that are load-bearing:

* **forkserver context**
  ([line 119](diffusion_policy/env/pusht/pusht_verifier.py#L119)) — the pool is built lazily
  on the first rollout, by which time CUDA is initialized in the parent, and fork-after-CUDA
  can deadlock.
* **`render_action=False`**
  ([line 61](diffusion_policy/env/pusht/pusht_verifier.py#L61)) — rendered subgoal frames go
  into the policy's own ResNet, so they must match `PushTImageEnv._get_obs` exactly or the
  encoder sees OOD images with a red action cross painted on.

`rollout()` returns `(value, state=[agent_pos || feedback], [image])` — all from the same
simulation, so the richer context modes cost no extra sim steps.

### 5. Generating the context

[`search_candidates`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L803) is
a plain loop:

```python
for _ in range(n):
    a = predict_action(obs, actions=so_far, values=so_far_feedback)['action_pred']
    v, score, sg = _score_candidates(verifier, obs, a)
    append
```

Each `predict_action` is an 8-step DDIM chain
([`conditional_sample`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L724)).
The obs is encoded **once** (`obs_features`) and reused across all candidates — they all
condition on the same observation.

[`_score_candidates`](diffusion_policy/policy/pusht_diffusion_search_policy.py#L171) in the
PushT subclass is where the ablation lives. `_verifier_inputs` slices carefully: obs at step
`To-1` (not `[:, -1]`, which at train time would be the state ~14 control steps in the
*future*), and action window `[To-1 : To-1+Ta]` — the window that actually gets executed.
Then, by `search_context`:

| mode | what the next candidate sees |
| --- | --- |
| `value` | the scalar, rescaled onto the fitted feedback scale |
| `state` | the reached `[agent_pos, feedback]`, normalizer-scaled |
| `state_value` | both |
| `subgoal` | the *rendered* reached frame, pushed through the policy's own obs encoder |
| `subgoal_value` | that embedding ++ scalar |

Ranking always uses the **raw** scalar, in every mode — only the conditioning changes.
`_normalize_value` recomputes the scalar from scale-normalized feedback and deliberately
drops the normalizer's offset, so "0 exactly at the goal" survives.

Two unit-space invariants worth internalizing:

* Context actions stay in **raw pixel units** everywhere the verifier or env touches them,
  and are normalized only at the model boundary in
  [`_normalize_context_actions`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L638).
* `generate_search_context` uses `no_grad` rather than `inference_mode`, precisely so the
  buffered tensors can be replayed on a later step by the outer/inner trainer (inference
  tensors carry a permanent restriction against autograd-recorded ops).

### 6. The loss

[`compute_loss`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L1008) →
`_compute_loss`:

1. encode obs once (grad-tracked);
2. `generate_search_context` → 15 candidates + their feedback, under `no_grad` (the verifier
   runs 15xB sims here — at batch 32 that is **480 sims per gradient step**, which is why the
   batch size is pinned);
3. tile the GT expert action 16x, add independent noise/timestep per copy;
4. one model call with `context_lengths=None` → slot *k* sees *k* candidates;
5. plain epsilon MSE against the noise.

So the regression target is the **expert action at every prefix length**. There is no reward,
ratio, or advantage anywhere — the search only shapes the conditioning. Everything generated
is discarded after the update.

The [`_crop_scope`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L581)
wrapper is subtle and worth flagging: one random 76x76 crop offset is drawn **per sample**,
derived from `(seed, global_step)`, and reused for that sample's obs window *and* every
subgoal image generated from it. `CropRandomizer` left to itself samples per *image*, which
would translate the obs and its own subgoal relative to each other at train time while eval
center-crops both — a registration mismatch the model would otherwise have to learn around.

### 7. The training loop

[`run()`](diffusion_policy/workspace/train_mlp_image_workspace.py#L233) is an ordinary epoch
loop: one batch → `set_crop_step(seed, global_step)` → `compute_loss` → backward → clip →
step → EMA step. The search is *inside* the loss, so there is no outer/inner structure here
at all — maximally on-policy and maximally expensive.

(`train_search_outer_inner_workspace.py` is the amortized variant: generate context once per
pool of windows, reuse it for `inner_epochs` passes, and monitor staleness with
`predict_epsilon` — the epsilon-space MSE between a frozen collector snapshot and the live
policy at matched inputs, which is exactly a per-denoising-step KL because the two reverse
kernels share the scheduler's variance.)

The distinctive metric is
[`_search_action_nrmse`](diffusion_policy/workspace/train_mlp_image_workspace.py#L172): for
held-out windows, generate all 16 candidates and report

* `nrmse_first` — candidate 0, empty context, i.e. the no-search baseline
* `nrmse_min` — best candidate
* `nrmse_avg` — control for "did the distribution move, or are we just sampling more?"

**`first - min` is the actual search gain.** The `_make_nrmse_loader` seeded-subset above it
exists because the plain `shuffle: False` loader truncated at 4 batches drew all 128 windows
from the split's *first* episode — every `val_*`/`test_*` number was a single-episode
statistic.

### 8. Eval

[`PushTSearchImageRunner`](diffusion_policy/env_runner/pusht_search_image_runner.py) resets
envs onto *recorded* val/test initial states (`legacy=False` is required for the reset to
round-trip exactly) and calls
[`predict_action_best`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L918)
— `predict_n_actions` (which slides a rolling 15-wide context window if `n > max_actions`)
then `scores.argmax`. `n_search_actions=1` bypasses the argmax entirely, giving the honest
no-search baseline.

[eval_search_pusht.py](eval_search_pusht.py) is the offline scorer: sweeps *n*, reports
success (coverage >= 95%) with Wilson intervals, and in `--watch` mode tails
`checkpoints/step_*.ckpt` maintaining `bon_search/success_curves.jsonl` and `best.json`.
Selection is on **val** success (30 episodes, ~5.5pp SE), with mean val reward as tiebreak
because a BC policy at n=1 scores 0% at every checkpoint and binary success cannot order
those.

### Where to look if you are changing something

* [`_build_memory_masks`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L223)
  is where the prefix-training trick lives.
* [`_score_candidates`](diffusion_policy/policy/pusht_diffusion_search_policy.py#L171) is the
  only place the ablation arms differ.

---

## Part 2 — What the diffusion objective actually trains

A common informal framing: *"we train pi(x_t, t) to go to x_0, and for the model to learn the
path we need to make multiple gradient updates."* The first half is right; the second half
conflates two different "multiples".

### What the objective is

Each training sample draws one clean trajectory `x_0`, **one** noise vector `eps`, and
**one** timestep `t`, forms

```
x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps
```

and regresses `|| eps_theta(x_t, t) - eps ||^2`. That is
[`_compute_loss`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L1024)
verbatim — `timesteps = randint(0, 100)`, `add_noise`, MSE against `noise`.

The minimizer is not "a path". It is a single function whose value at every point is a
conditional expectation:

```
eps_theta*(x_t, t) = E[eps | x_t]
  <=>  xhat_0 = (x_t - sqrt(1 - abar_t) * eps_theta) / sqrt(abar_t) = E[x_0 | x_t]
```

and, up to a constant, that is the **score**:

```
grad log p_t(x_t) = -eps_theta(x_t, t) / sqrt(1 - abar_t)
```

So one set of weights encodes a whole *vector field* — a denoising direction at every
`(x, t)`, for all 100 noise levels at once. There is no "path" stored anywhere in theta.

### The path is walked with forward passes, not gradient updates

The path from noise to data is produced at **sampling** time by integrating that field:
[`conditional_sample`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L724)
runs 8 DDIM steps, each one a forward pass through frozen weights. Zero gradients involved.
Training and sampling are two independent loops that people describe with the same word
"steps":

| | count here | what it costs | what it does |
| --- | --- | --- | --- |
| gradient updates | 20,000 | training | fits the field `eps_theta` |
| denoising steps | 8 per sample | inference | traverses the path through a *fixed* field |

You could take a billion gradient updates and still need those 8 forward passes to draw one
sample.

### So why do you need many gradient updates?

Not to learn a path — to **cover the domain**. Each update is a one-sample Monte Carlo
estimate of an expectation over `(x_0, eps, t)`. One step tells you the correct direction at
one point of a 100-level field; you need many draws before every noise level and every region
of trajectory space has been visited enough times.

This is exactly the EMA argument in [README_pusht.md](README_pusht.md): t ~ 90 is coarse
structure, t ~ 5 is fine detail, consecutive gradients estimate *different parts of the same
function*, so the live iterate rattles and averaging cancels the jitter.

This setup is unusually generous on coverage per step: `compute_loss` tiles the target 16x
and draws **independent** noise and timestep for each copy, so a batch of 32 yields 512
`(x_t, t)` triples per update instead of 32 — roughly 100k draws per timestep bucket over a
20k-step run.

### The part that is genuinely non-obvious

Given the trained field, why not call `eps_theta(x_T, T)` once and jump straight to
`xhat_0`?

Because `xhat_0` is a **mean**, and at high `t` the posterior `p(x_0 | x_t)` is wide.
`E[x_0 | x_T]` is roughly the average of the whole action distribution. On PushT, where "go
around the T clockwise" and "go around counter-clockwise" are both valid, the average is a
chunk that does neither — it drives straight into the block. Classic mode averaging.

Iterating fixes this: step to `xhat_0`, then partially re-noise back to level `t-1`.
Conditioning on `x_{t-1}` shrinks the posterior, so the *next* mean is an average over a
narrower set of modes. By low `t` the posterior is nearly a point mass and its mean is a
legitimate sample. **The multi-step chain is how you commit to a mode gradually instead of
averaging over all of them.**

That is the real reason the path exists — not that the model cannot be trained enough, but
that the quantity it is trained to output is only a valid sample when the conditioning is
tight.

### Where this bites in this code

Both multiples compound inside a single training step of the offline search policy.
[`generate_search_context`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L956)
runs 15 candidates x 8 DDIM steps = 120 forward passes, plus 15 verifier sims, **per batch
element**. At batch 32 that is 3,840 forward passes and 480 physics rollouts to buy a single
gradient step — which is why `batch_size` is pinned at 32 and why
`train_search_outer_inner_workspace.py` exists to amortize it.

---

## Part 3 — The obs encoder and `obs_feature_dim`

### The encoder

[`MultiImageObsEncoder`](diffusion_policy/model/vision/multi_image_obs_encoder.py#L60),
injected by the config's `policy.obs_encoder` block. It is not a learned fusion module — it
is a **concatenator with one CNN in it**. `_forward` walks the obs keys in `shape_meta` and,
per type:

* **rgb** (`image`) → `key_transform_map[key]` then `key_model_map[key]`. The transform is
  `Sequential(resize, crop, extra_randomizations, imagenet_norm)`; here that is
  `Identity -> CropRandomizer(76x76) -> Identity -> Identity`. The model is a ResNet-18 with
  `fc = Identity` ([model_getter.py](diffusion_policy/model/vision/model_getter.py)) and every
  `BatchNorm2d` swapped for `GroupNorm(C/16, C)` by `use_group_norm`.
* **low_dim** (`agent_pos`, `feedback`) → passed straight through, **unchanged**. No
  embedding, no MLP.

Then `torch.cat(features, dim=-1)`. That is the whole thing.

Key ordering matters if you ever index into the vector: `rgb_keys` and `low_dim_keys` are both
`sorted()` ([lines 192-193](diffusion_policy/model/vision/multi_image_obs_encoder.py#L192)),
and rgb goes first.

### The feature dim

`obs_feature_dim` is read once in the policy constructor:

```python
obs_feature_dim = obs_encoder.output_shape()[0]
```

[`output_shape`](diffusion_policy/model/vision/multi_image_obs_encoder.py#L314) does not
compute anything analytically — it builds a dummy zeros obs dict from `shape_meta` and runs a
real forward. For this PushT config:

| piece | width |
| --- | --- |
| ResNet-18 on the 76x76 crop | **512** |
| `agent_pos` | **2** |
| `feedback` | **16** |
| **`obs_feature_dim`** | **530** |

Layout: `[512 resnet || 2 agent_pos || 16 feedback]`.

The 512 is invariant to the crop size — ResNet's `AdaptiveAvgPool2d(1)` collapses whatever
spatial map 76x76 produces down to one vector per channel, so `crop_shape: [76,76]` vs
`[96,96]` changes receptive-field coverage, not width.

All three parts arrive **already normalized**:
[`_encode_obs_features`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L687)
calls `self.normalizer.normalize(obs_dict)` *before* the encoder, so the image is in [-1, 1]
via `get_image_range_normalizer` and the low-dim keys are on their fitted limits scale. That
is why `imagenet_norm: False` is mandatory in the config — ImageNet stats assume [0, 1], and
stacking them on [-1, 1] input gave the encoder roughly [-6.5, +2.25].

### Where 530 shows up

1. [`obs_emb = nn.Linear(obs_feature_dim, n_emb)`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L70)
   — 530 → 256, the projection into the transformer's memory tokens. (The online policy calls
   its version `obs_projection`; here it lives inside the transformer.)
2. [`_context_dim`](diffusion_policy/policy/pusht_diffusion_search_policy.py#L76) in the
   `subgoal` modes — the subgoal image goes through the *same* encoder, so its embedding is
   also 530 (531 for `subgoal_value`). That is the point of `_encode_subgoal`: the "here is
   what the world looks like afterwards" token lands in exactly the same feature space as the
   obs tokens, reusing the fitted normalizer and obs encoder and adding zero parameters.

So `context_dim` per `search_context` mode:

| mode | `context_dim` |
| --- | --- |
| `value` | 1 |
| `state` | 18 |
| `state_value` | 19 |
| `subgoal` | 530 |
| `subgoal_value` | 531 |

A 530x spread in how wide `action_value_emb`'s input is across the ablation arms.

### One thing to be aware of

The encoder's own `feature_dim` constructor arg (a distinct thing from `obs_feature_dim`)
would append an MLP projector — but it is **built lazily inside `_forward`**
([line 303](diffusion_policy/model/vision/multi_image_obs_encoder.py#L303)). If it is ever
set, those `nn.Linear` layers get created after the optimizer already captured
`parameters()`, on whatever device the first input happens to be, and will not appear in the
checkpoint's `state_dict` for a fresh model. The PushT configs leave it `null`, so it never
fires — but do not reach for it without moving the construction into `__init__`.
