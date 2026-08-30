"""Feedback-based verifier for the offline PushT diffusion-search policy.

The maze diffusion-search policy scores each candidate action with an external
``l2s.verifier.MazeVerifier`` (analytic, maze-only). PushT has no such analytic map from
(state, action-sequence) to outcome, so this verifier **simulates** the candidate in a
deterministic PushT sim and scores the result with the *same feedback signal the online
search policy uses*: ``feedback_util.compute_feedback_from_pose`` (goal-vs-achieved T
keypoint displacement).

``get_value(obs_dict, action) -> (B,)`` returns a scalar, in pixels, at the state reached
after executing the candidate action sequence from the obs state. Higher == better, and
the value is <= 0 always. WHICH scalar is chosen by ``value_fn``, one of ``VALUE_FNS``:

* ``'armTn'`` -- ``-(d_T->goal/13.6 + d_arm->T/52.1)``, both terms normalized by their
  within-step candidate spread; see ``value_arm_t_norm``. **DEPRECATED as of 2026-08-28**:
  ``pusht_base.yaml`` no longer defaults to it and no new arm should choose it. It is NOT
  retired the way ``armT`` is -- it stays fully supported as a ``verifier_tag`` and a
  ``--verifier-value``, because ~14 run directories, the eval watcher list and
  ``scripts/build_slot_norm_doc.py`` all need it loadable, and the arms already trained
  under it must keep being evaluated on exactly what they were trained on. Pin it
  explicitly (``verifier_tag=armTn``) wherever it is still wanted; inheriting the default
  now gives ``t_goal``.
* ``'armT'`` -- ``-(d_T->goal + d_arm->T)``, the raw unnormalized sum. RETIRED: the arm
  term outvotes the task term ~4:1 at every step, which measurably destroys best-of-n
  (see the T_GOAL_SPREAD comment). Unreachable from any config and from the eval CLI
  (``eval_search_pusht.py --verifier-value``); the function is kept ONLY so the existing
  ``bon_search_ver-armT/`` curves stay readable and reproducible. Do not run it again.
* ``'t_goal'`` -- ``-d_T->goal`` alone. THE CURRENT DEFAULT again as of 2026-08-28, and
  the pre-2026-08-19 value before that, so checkpoints from either era are evaluated on
  exactly what they were trained on. ``DEFAULT_VALUE_FN`` has always been this; the config
  default has now been brought back into line with it.
* ``'d_t_goal'`` -- ``-d_T->goal/13.6``, the same term on armTn's normalized footing.
  RANKS IDENTICALLY TO ``t_goal`` (a positive constant divisor is monotone); only the
  magnitude differs, which is what the recorded scores and the training context read. See
  ``value_d_t_goal``.

and one value that is NOT in ``VALUE_FNS`` because it is not a function of one candidate:

* ``'armTd'`` -- ``armTn`` with the two hard-coded spreads replaced by the spread measured
  ACROSS THE N CANDIDATES of the control step being decided. It lives in
  ``CROSS_CANDIDATE_VALUES`` and is applied by the policy at the ``(B, n)`` stack, not
  here; see ``value_arm_t_dyn``. THREE INVARIANTS ABOVE DO NOT HOLD FOR IT: it is not a
  pure function of a single candidate, it is **not <= 0** (it sums to ~0 across the
  candidates by construction), and it is **not comparable across control steps** -- only
  the ranking within one step is meaningful. It is EVAL-ONLY and never a ``verifier_tag``;
  see ``check_verifier_value``.

where the two terms (both ``feedback_util`` functions, both independently callable):

* ``d_T->goal`` = ``t_goal_distance`` = mean per-keypoint distance of the achieved T from
  the goal T. Task progress: 0 iff the block ends on the goal.
* ``d_arm->T`` = ``arm_to_t_distance`` = distance from the arm to the centre of the
  achieved T. Approach.

WHY THE APPROACH TERM EXISTS: before the arm touches the block no candidate chunk can move
the T, so ``d_T->goal`` is IDENTICAL across every candidate (measured: 0.0000 px spread
over 8 candidates) and ranking on ``t_goal`` alone is a coin flip -- on exactly the
approach steps where search should be helping. The approach term is the only one that
varies there, so it is what breaks the tie; once contact happens the block term moves and
dominates. Under ``armT`` the value is 0 only at "T on the goal, arm at the T's centre".

The two are DIFFERENT EXPERIMENTS: a candidate ranking, a training context and a success
curve produced under one are not comparable to the other. The choice rides in the run
identity (``verifier_tag`` -> run_name, wandb name and tags) so a directory cannot lie
about which one produced it.

``rollout(obs_dict, action, render=False) -> (value (B,), state (B, STATE_DIM)[, image])``
returns that same scalar *plus* the sim state reached at the end of the candidate chunk --
the "subgoal" the candidate lands on -- laid out as ``[agent_pos (2), feedback (16)]``, and
with ``render=True`` also the rendered subgoal observation ``(B, 3, 96, 96)``. All of it
comes from the same simulation, so the extra signals cost no extra sim steps (``render``
adds one render per chunk, not per step).

The reached state also backs the search-context modes of ``PushTDiffusionSearchPolicy``:
``value`` rescales the scalar from it, and ``subgoal``/``subgoal_value`` pair it with the
rendered frame and embed the whole observation through the policy's own obs encoder.
"""
from typing import Dict

import numpy as np
import torch

import gym
import dill

from diffusion_policy.env.pusht.pusht_env import PushTEnv
from diffusion_policy.env.pusht.feedback_util import (
    compute_feedback_from_pose, block_pose_from_feedback, keypoints_at_pose,
    t_goal_distance, arm_to_t_distance,
    T_VERTS, GOAL_POSE, GOAL_KEYPOINTS, N_KEYPOINTS)
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv, force_close
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv


class _DillEnv(gym.Wrapper):
    """Adds run_dill_function/get_attr so the vector env can set per-env reset states."""

    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        return dill.loads(dill_fn)(self)

    def render_obs(self):
        """The current state rendered as an (H, W, 3) uint8 frame.

        Must be a real method, not inherited via gym.Wrapper.__getattr__: the vector env
        worker resolves remote calls with ``getattr(env, name)`` on the wrapper.
        """
        return self.env._render_frame(mode='rgb_array')


def _make_verifier_env(legacy, render_size):
    """Picklable thunk building a PushTEnv for the verifier pool.

    ``render_action=False`` is REQUIRED, not cosmetic: PushTEnv defaults it to True, which
    draws a red action cross into every rendered frame. The subgoal frames are fed to the
    policy's obs encoder, so they must match ``PushTImageEnv._get_obs`` -- the pipeline that
    produced the dataset's ``img`` array and the eval-time obs -- or the encoder sees
    out-of-distribution images. Stepping never renders; only ``render_obs`` does.
    """
    def _fn():
        return _DillEnv(PushTEnv(
            legacy=legacy, render_action=False, render_size=render_size))
    return _fn


def value_t_goal(agent_pos, feedback):
    """``-(T-to-goal)`` -- THE PRE-2026-08-19 VALUE. (n,) from (n, 2), (n, 16).

    Kept callable, and kept the DEFAULT, so a checkpoint trained against it is evaluated on
    exactly what it was trained on: a saved cfg from before the cutover carries no
    ``verifier_value`` key at all, so "key absent -> this function" reproduces those runs
    with no flag and no chance of a silent mismatch.

    ``agent_pos`` is accepted and ignored -- that is the defining property, and the reason
    this value is flat across candidates until the arm touches the block. Do not make it
    the default for anything trained after the cutover.
    """
    return -t_goal_distance(feedback)


def value_d_t_goal(agent_pos, feedback):
    """``-(T-to-goal)/T_GOAL_SPREAD`` -- ``t_goal`` on armTn's normalized footing. (n,).

    RANKS IDENTICALLY TO ``t_goal``. Dividing by a positive constant is monotone, so
    ``argmax`` and ``softmax``-over-z pick the same candidate under either. What changes is
    the MAGNITUDE: the scalar lands in the same spread-normalized units as each armTn term
    instead of raw pixels, which is what makes a d_t_goal run's recorded candidate scores
    and its training context comparable with the armTn family's.

    Because the ranking is unchanged, a checkpoint's ``t_goal`` success curve IS its
    ``d_t_goal`` curve for any policy whose executed action depends on the scalar only
    through the ranking. Re-measuring one is a consistency check, not a new number.

    ``agent_pos`` is accepted and ignored, exactly as in ``value_t_goal`` -- this value
    inherits that flatness across candidates until the arm touches the block.
    """
    return -t_goal_distance(feedback) / T_GOAL_SPREAD


def value_arm_t(agent_pos, feedback):
    """``-(T-to-goal + arm-to-T)`` -- the current value. (n,) from (n, 2), (n, 16).

    RETIRED -- see the module docstring. Equal-weight sum: both terms are Euclidean
    distances in the SAME 512-px env frame, so the sum is unit-consistent and takes no
    weight constant. That reasoning is what made it look safe, and it is wrong: unit
    consistency says nothing about the WITHIN-STEP spreads argmax actually compares, which
    differ ~4x. Use ``value_arm_t_norm``.
    """
    return -(t_goal_distance(feedback) + arm_to_t_distance(agent_pos, feedback))


# Within-step spread of each term ACROSS CANDIDATES, in pixels: the std over the 8
# candidates reached from one state, averaged over 16 real dataset states
# (scratchpad/measure_spread.py). These -- not the marginal ranges -- are the scales that
# matter, because argmax compares candidates WITHIN a control step and is blind to any
# constant shared by all of them.
#
# The marginal distributions are already balanced (mean 80.2 vs 91.2 px over 25650 dataset
# states, ratio 1.14), which is why normalizing by overall range would be a near no-op. The
# within-step spreads are NOT: 13.6 vs 52.1, a 3.8x edge to the arm term, rising to 4.5x
# before contact where sd(T->goal) collapses toward 0 (measured 0.000 at 267px away -- one
# action chunk cannot reach the block, let alone move it). Summing the two raw is therefore
# a ~4:1 vote for the arm term at every step, which is what made `armT` rank on "park the
# arm nearest the T centre" and cost the UNet BC arm its best-of-n gain (0.460 -> 0.060 at
# n=8, step 10000).
T_GOAL_SPREAD = 13.6
ARM_T_SPREAD = 52.1

# Mean of the armTn value over the same 25650 dataset states: 80.2/13.6 + 91.2/52.1 = 7.65.
# Spread-normalized units are the right thing for RANKING but are not O(1) -- the value runs
# to about -32 -- so the CONTEXT copy is divided by this to average ~-1, which is the range
# action_value_emb's other input (the normalized state / subgoal embedding) lives in.
# Dividing by a positive constant cannot reorder candidates, so ranking is untouched.
ARM_TN_CONTEXT_SCALE = 7.65


def value_arm_t_norm(agent_pos, feedback):
    """``-(T-to-goal/13.6 + arm-to-T/52.1)`` -- both terms normalized. (n,) from (n,2),(n,16).

    Each term is divided by its own within-step candidate spread, so both are measured in
    "sigmas of what this choice can actually change" and a 1-sigma gain in either counts the
    same. That is the property the raw sum lacked.

    It keeps the tie-breaking that motivated the arm term -- before contact the T term has
    zero spread, so ANY positive weight on the arm term decides the ranking -- while
    stopping the arm term from outvoting real task progress once the block starts moving.
    """
    return -(t_goal_distance(feedback) / T_GOAL_SPREAD
             + arm_to_t_distance(agent_pos, feedback) / ARM_T_SPREAD)


# Denominator floor for `armTd`, in PIXELS. NOT a numerical guard -- in float64 the
# exactly-degenerate case gives a numerator of exactly 0, so no epsilon is needed for that.
# It is the statement that candidate spreads below this are sim/float noise rather than
# signal. float32 state at 80-300 px has an ulp around 1e-5, so 1e-3 sits ~100x above the
# noise floor and far below any block displacement a real contact produces. Any value in
# [1e-4, 1e-1] behaves identically; it is a floor, not a tuning knob.
#
# It exists because armTd has a failure mode the FIXED divisors do not. Exact degeneracy is
# safe either way, but NEAR-degeneracy is not: if one candidate barely grazes the block and
# sd(d_T->goal) is 1e-4 px, armTn still ignores it (1e-4/13.6 ~ 0, the arm term decides)
# while armTd would inflate that physically meaningless displacement to a +/-1.5 z-score and
# put it on equal footing with the arm term. This floor is what stops that.
ARM_TD_EPS_PX = 1e-3


def value_terms_from_state(state):
    """``(..., STATE_DIM)`` -> ``(..., 2)`` raw-pixel ``[d_T->goal, d_arm->T]``, in torch.

    The torch mirror of ``t_goal_distance`` / ``arm_to_t_distance``, taking the layout
    ``rollout`` already returns (``[agent_pos (2), feedback (16)]``). EXACT, not a fit:
    ``achieved_kp = GOAL_KEYPOINTS - feedback`` is an identity (feedback_util:70), so no
    pose is recovered and nothing is approximated.

    Deriving the terms from ``state`` rather than plumbing them out of ``rollout`` is
    deliberate: ``state`` is already sliced to ``[:m]`` there, so the tail-chunk padding
    (which duplicates the last row up to ``n_envs``) can never leak into a statistic taken
    over these -- which matters the moment a caller aggregates ACROSS candidates, as
    ``value_arm_t_dyn`` does.
    """
    agent_pos = state[..., :PushTVerifier.AGENT_DIM]                  # (..., 2)
    feedback = state[..., PushTVerifier.AGENT_DIM:]                   # (..., 2*N_KEYPOINTS)
    kp = feedback.reshape(*feedback.shape[:-1], N_KEYPOINTS, 2)
    goal_kp = torch.as_tensor(GOAL_KEYPOINTS).to(kp)                  # (N_KEYPOINTS, 2)
    centre = (goal_kp - kp).mean(dim=-2)                              # (..., 2) achieved centroid
    t_goal = kp.norm(dim=-1).mean(dim=-1)                             # (...,) px
    arm_t = (agent_pos - centre).norm(dim=-1)                         # (...,) px
    return torch.stack([t_goal, arm_t], dim=-1)                       # (..., 2)


def value_arm_t_dyn(terms, eps: float = ARM_TD_EPS_PX):
    """``-(z(d_T->goal) + z(d_arm->T))``, z taken ACROSS THE N CANDIDATES of one step.

    ``terms`` (B, n, K) raw-pixel per-candidate terms, candidate axis = 1 -> (B, n).

    Same idiom as ``selection_util.select_candidate``'s softmax z-score (mean-subtracted,
    ``unbiased=False``) -- but in FLOAT64, which is load-bearing rather than tidy.

    MEASURED: in float32, ``torch.std(unbiased=False)`` of n BIT-IDENTICAL values is 0 for
    n in (3, 5) but **7.63e-6** for n in (7, 8, 16, 32, 64) -- the mean of identical values
    is off by one ulp under pairwise accumulation, and the biased std comes out at exactly
    one ulp (at magnitude 80 px). That is a score manufactured out of pure rounding, on
    exactly the pre-contact steps where sd(d_T->goal) is genuinely 0 and the arm term is
    supposed to decide, and it hits every evaluated n except 1, 2 and 4 -- and only for
    values that are not exactly representable, so a round-number smoke test would pass.

    How bad it is depends on eps, which is why the two defences are described together:
    at eps=1e-6 the residual becomes |z| = 0.884, i.e. FULL magnitude, competing on equal
    footing with the real arm term; at the ARM_TD_EPS_PX floor of 1e-3 it is |z| = 7.6e-3,
    a ~1% contamination. So the pixel floor already absorbs most of it -- but float64 makes
    the residual exactly 0.0 at every n, for free. Keep both, and do not couple them: a
    future change to eps must not silently re-open this. scripts/armtd_smoke.py asserts the
    float32/float64 gap still exists, so the claim cannot rot.

    Mean subtraction is separately motivated and is NOT what fixes the above: it is
    ranking-neutral for argmax in exact arithmetic (a per-step constant shared by all
    candidates cannot reorder them, and `selection: softmax` re-standardizes anyway). It is
    here so the score stays O(1) instead of ~d/eps, which keeps full float64 precision on
    the term that actually varies, stops `eps` from setting the overall scale, and makes the
    logged --store-scores values readable (a degenerate step is exactly 0, and
    ``score.sum(dim=1) ~ 0`` is a checkable invariant).

    ``unbiased=False`` so n=1 gives sd 0 AND numerator 0 -> the score is exactly 0 for the
    single candidate, i.e. the n=1 column is unchanged from every other verifier value.

    SCALE INVARIANCE -- the whole point of the rule, and exact only ABOVE the floor. With
    eps = 0, scaling a term by any c > 0 leaves z untouched; with the floor, z(c*x) picks up
    a relative deviation of ~eps/sd. Measured: sd(d_arm) = 36 px gives max|dscore| = 4e-5
    even at c = 1000, and the ARGMAX is unchanged. Below the floor the invariance breaks
    completely, which is deliberate -- that is what the floor is for.
    """
    x = terms.to(torch.float64)
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, unbiased=False, keepdim=True)
    z = (x - mu) / (sd + eps)
    return (-z.sum(dim=-1)).to(terms.dtype)                           # (B, n)


# Every value the verifier can score with, keyed by the string that also names the run
# (`verifier_tag` in the configs), so the directory name and the scoring rule cannot drift
# apart. Signature: (agent_pos (n, 2), feedback (n, 16)) -> (n,), <= 0, higher is better.
VALUE_FNS = {
    't_goal': value_t_goal,
    'd_t_goal': value_d_t_goal,
    'armT': value_arm_t,
    'armTn': value_arm_t_norm,
}

# Pre-cutover default, deliberately: see value_t_goal. Training configs set the key
# explicitly and `check_verifier_value` (called by every PushT training workspace) asserts
# they do, so a NEW run cannot land here by omission -- only an OLD checkpoint, which is
# exactly who should.
DEFAULT_VALUE_FN = 't_goal'

# Values that need the WHOLE candidate set and so cannot be a VALUE_FNS entry: those are
# called with (agent_pos (B,2), feedback (B,16)) where B is the parallel-EPISODE batch, one
# candidate at a time -- SearchProcedureMixin.search_candidates generates candidates in a
# Python loop and builds the candidate axis outside with torch.cat. Signature here is
# therefore different: (terms (B, n, K)) -> (B, n), applied by the policy at the stack.
CROSS_CANDIDATE_VALUES = {
    'armTd': value_arm_t_dyn,
}

# The per-candidate scalar a cross-candidate value falls back to wherever fusion does NOT
# run: the search-context copy (_normalize_value), a direct search_candidates call, the
# sim's own per-chunk scoring. armTn, because it is armTd's closest sibling AND the current
# production value -- so the fallback is never something arbitrary.
BASE_VALUE_FN = {
    'armTd': 'armTn',
}

# Every string `verifier_value` accepts, per-candidate and cross-candidate alike.
VERIFIER_VALUES = tuple(VALUE_FNS) + tuple(CROSS_CANDIDATE_VALUES)


def base_value_fn(value: str) -> str:
    """The VALUE_FNS key backing `value` -- itself, unless it is cross-candidate."""
    return BASE_VALUE_FN.get(value, value)


def check_verifier_value(cfg):
    """Fail fast if the run's verifier tag and the policy's actual scoring rule disagree.

    ``PushTVerifier`` defaults to the PRE-2026-08-19 value (``t_goal``), deliberately, so
    that an old checkpoint -- whose saved cfg has no ``verifier_value`` key at all --
    reproduces itself with no flag. The cost of that direction is that a NEW config which
    forgets the key would train against the old value while its directory and wandb name
    say ``ver-armTn``. This is the guard that makes the cheap default safe: the tag names
    the run, so it must BE the scoring rule, not sit alongside one.

    Lives HERE, next to ``VALUE_FNS`` and ``DEFAULT_VALUE_FN``, rather than in any one
    workspace: it guards an invariant of this module, and every PushT training workspace
    needs it. It previously sat in ``TrainMLPImageWorkspace``, which meant the UNet BC arm
    -- routed through ``TrainDiffusionUnetImageWorkspace`` -- silently skipped it and could
    train under ``t_goal`` while its run dir claimed otherwise.

    Only checked when the config declares ``verifier_tag`` -- the maze configs do not, and
    are left alone.
    """
    tag = cfg.get('verifier_tag', None)
    if tag is None:
        return
    if tag in CROSS_CANDIDATE_VALUES:
        # Reject BEFORE the not-in-VALUE_FNS branch below, which would otherwise report
        # "not a known verifier value" -- true of VALUE_FNS but misleading, since the real
        # reason is that this value has no training semantics at all.
        raise ValueError(
            f'verifier_tag={tag!r} is a cross-candidate SELECTION rule: it is defined only '
            f'over the n candidates of one control step, so it has no per-candidate scalar '
            f'to train against and no meaning as a run identity. It also cannot feed the '
            f'search context, which is causal (candidate k is conditioned on candidates '
            f'< k, so a statistic over all n does not exist yet when k is scored). Train '
            f'under {BASE_VALUE_FN[tag]!r} and evaluate with '
            f'`eval_search_pusht.py --verifier-value {tag}`.')
    if tag not in VALUE_FNS:
        raise ValueError(
            f'verifier_tag={tag!r} is not a known verifier value; expected one of '
            f'{sorted(VALUE_FNS)} (pusht_verifier.VALUE_FNS).')
    declared = (cfg.get('policy', None) or {}).get('verifier_value', None)
    if declared is None:
        raise ValueError(
            f'config declares verifier_tag={tag!r} but policy.verifier_value is unset, so '
            f'the verifier would fall back to {DEFAULT_VALUE_FN!r} while the run directory '
            f'and wandb name say ver-{tag}. Set `verifier_value: ${{verifier_tag}}` in the '
            f'policy block.')
    if declared != tag:
        raise ValueError(
            f'config declares verifier_tag={tag!r} but policy.verifier_value={declared!r}. '
            f'The run directory is named from the tag, so this would file results scored '
            f'one way under a name that claims the other.')


class PushTVerifier:
    # layout of the rollout state returned by `rollout`: [agent_pos (2), feedback (16)].
    # agent_pos/feedback are exactly the two low_dim policy obs keys, so the state is
    # normalizable with the policy's own normalizer (see PushTDiffusionSearchPolicy).
    AGENT_DIM = 2
    STATE_DIM = AGENT_DIM + 2 * N_KEYPOINTS   # 18

    def __init__(self, n_envs: int = 32, legacy: bool = False, use_async: bool = True,
                 verifier_steps: int = None, render_size: int = 96,
                 value_fn: str = DEFAULT_VALUE_FN, **kwargs):
        """
        Args:
            n_envs: size of the persistent vectorized PushT sim pool; candidates are
                scored in chunks of this size (the envs step in parallel).
            legacy: must be False so ``reset_to_state`` round-trips a recorded state
                exactly (see runner/eval_bon notes); kept configurable for parity.
            use_async: run the pool with AsyncVectorEnv (parallel worker processes).
                Set False for a single-process SyncVectorEnv (debugging).
            verifier_steps: optional cap on how many action steps to simulate
                (None = all of what is passed in). The caller already slices the action
                down to the executed window, so this should normally stay None -- a cap
                shorter than that window silently truncates the evaluated chunk.
            render_size: edge length of the subgoal frames from ``rollout(render=True)``.
                Must match the policy's image shape_meta (96) or the encoder rejects them.
            value_fn: which value to score with, one of ``VERIFIER_VALUES`` (see the
                module docstring). Defaults to the pre-cutover ``'t_goal'``; every current
                config passes ``'armTn'`` explicitly. Runs under different values are not
                comparable. A cross-candidate value is accepted here and the SIM reports
                its base value; the policy applies the real rule at the (B, n) stack.
        """
        self.value_fn = value_fn        # validated by the property setter below
        self.n_envs = n_envs
        self.legacy = legacy
        self.use_async = use_async
        self.verifier_steps = verifier_steps
        self.render_size = render_size
        # Lazily built vectorized pool, reused across get_value calls: the workers are
        # only forked on the first rollout (i.e. inside the first compute_loss), which
        # is why _get_vec uses a forkserver context -- by then the parent has CUDA
        # initialized, and forking a CUDA-initialized process can deadlock.
        self._vec = None

    @property
    def value_fn(self) -> str:
        """Which value this verifier was asked for -- possibly cross-candidate."""
        return self._value_fn

    @value_fn.setter
    def value_fn(self, value: str):
        """Validating setter, because eval_search_pusht.py's --verifier-value override
        assigns this attribute DIRECTLY on an already-built verifier. Before this was a
        property that path bypassed __init__'s assert entirely, so a bad string sailed
        through and only surfaced as a KeyError deep inside `rollout`.
        """
        assert value in VERIFIER_VALUES, \
            f'value_fn must be one of {sorted(VERIFIER_VALUES)}, got {value!r}'
        self._value_fn = value
        # What the SIM scores each chunk with. Identical to value_fn for every
        # per-candidate value; for a cross-candidate one it is the base scalar, because
        # the real rule needs all n candidates and is applied by the policy
        # (SearchProcedureMixin.predict_n_actions), not here.
        self._score_key = base_value_fn(value)

    def _get_vec(self):
        if self._vec is None:
            env_fns = [_make_verifier_env(self.legacy, self.render_size)
                       for _ in range(self.n_envs)]
            if self.use_async:
                # forkserver: workers are forked from a clean server process, NOT from the
                # (CUDA-initialized) training process -- fork-after-CUDA can deadlock. The
                # server imports torch once, so this is far cheaper than 'spawn'.
                self._vec = AsyncVectorEnv(env_fns, context='forkserver')
            else:
                self._vec = SyncVectorEnv(env_fns)
        return self._vec

    def _set_reset_states(self, vec, states):
        def _make_setter(state):
            state = np.asarray(state, dtype=np.float64)
            def _fn(env):
                env.unwrapped.reset_to_state = state
            return _fn
        vec.call_each('run_dill_function',
                      args_list=[(dill.dumps(_make_setter(s)),) for s in states])

    @staticmethod
    def _reset_states_from_obs(obs_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """(B, 5) env reset state [agent_x, agent_y, block_x, block_y, angle].

        Uses the last obs step. CALLER CONTRACT: the obs window handed in must END at
        the state the candidate action is taken from -- normally a single step, sliced
        by ``PushTDiffusionSearchPolicy._verifier_inputs``. Do NOT pass a raw training
        batch: those carry the full ``horizon`` (16) steps, so ``[:, -1]`` would be the
        state ~14 control steps ahead of the one the policy conditioned on.

        The block pose is reconstructed from ``feedback`` (exactly -- feedback is an
        invertible function of the pose, see ``block_pose_from_feedback``). This is the
        single path for both training and eval, so the two produce identical resets;
        ``feedback`` is a declared obs key, so no privileged reset carrier is needed.
        """
        agent_last = obs_dict['agent_pos'][:, -1].detach().cpu().numpy().astype(np.float64)
        feedback_last = obs_dict['feedback'][:, -1].detach().cpu().numpy()
        block_last = block_pose_from_feedback(feedback_last)  # (B, 3)
        return np.concatenate([agent_last, block_last], axis=-1)  # (B, 5)

    @torch.no_grad()
    def get_value(self, obs_dict: Dict[str, torch.Tensor],
                  action: torch.Tensor) -> torch.Tensor:
        """Scalar score of a candidate action sequence per batch element.

        Thin wrapper over ``rollout`` keeping the original scalar-only interface (the
        interface the maze ``MazeVerifier`` also exposes).
        """
        return self.rollout(obs_dict, action)[0]

    @torch.no_grad()
    def rollout(self, obs_dict: Dict[str, torch.Tensor],
                action: torch.Tensor, render: bool = False):
        """Simulate a candidate action sequence per batch element.

        Each candidate is simulated in a deterministic PushT sim (reset to the obs state)
        by stepping its action sequence; the value is ``VALUE_FNS[self._score_key]`` at the
        reached state (see the module docstring), and the state is the sim state
        reached at the end of the chunk. The pool of ``n_envs`` sims steps in parallel, so
        a batch of candidates is simulated in ceil(B / n_envs) parallel rounds. The env's
        own obs already carries agent pos (obs[:, :2]) and the block pose (obs[:, 2:5]),
        so both value terms and the low-dim outputs need no render.

        Args:
            obs_dict: must contain ``agent_pos`` (B, To, 2) and ``feedback`` (B, To, 16),
                from which the block pose is reconstructed. Same at train and eval time.
            action: (B, horizon, 2) UNNORMALIZED agent-target positions (pixel coords).
            render: also return the subgoal frame. Costs ONE render per chunk (not per
                step), taken after the last step; leave False when unused.
        Returns:
            ``(value, state)``, or ``(value, state, image)`` when ``render``:
            ``value`` (B,); ``state`` (B, STATE_DIM) laid out as
            ``[agent_pos (2), feedback (16)]``; ``image`` (B, 3, render_size,
            render_size) float32 in [0, 1], channel-first -- byte-for-byte the obs
            ``PushTImageEnv`` would emit at that state. All on ``action``'s device/dtype.
        """
        states = self._reset_states_from_obs(obs_dict)          # (B, 5)
        actions = action.detach().cpu().numpy().astype(np.float64)  # (B, H, 2)
        B, H, _ = actions.shape
        n_steps = H if self.verifier_steps is None else min(H, self.verifier_steps)

        vec = self._get_vec()
        values = np.empty(B, dtype=np.float32)
        end_states = np.empty((B, self.STATE_DIM), dtype=np.float32)
        images = np.empty(
            (B, 3, self.render_size, self.render_size), dtype=np.float32) \
            if render else None
        for start in range(0, B, self.n_envs):
            end = min(start + self.n_envs, B)
            m = end - start
            # pad the final partial chunk up to n_envs; padded results are ignored
            chunk_states = states[start:end]
            chunk_actions = actions[start:end]
            if m < self.n_envs:
                pad = self.n_envs - m
                chunk_states = np.concatenate(
                    [chunk_states, np.repeat(chunk_states[-1:], pad, axis=0)], axis=0)
                chunk_actions = np.concatenate(
                    [chunk_actions, np.repeat(chunk_actions[-1:], pad, axis=0)], axis=0)

            self._set_reset_states(vec, list(chunk_states))
            obs = vec.reset()                                   # (n_envs, 5)
            for t in range(n_steps):
                obs, _, _, _ = vec.step(chunk_actions[:, t])    # obs (n_envs, 5)
            obs = np.asarray(obs)
            agent_pos = obs[:, :2]                              # (n_envs, 2)
            block_pose = obs[:, 2:5]                            # (n_envs, 3): x, y, angle
            feedback = compute_feedback_from_pose(block_pose)   # (n_envs, 16)
            # agent_pos and feedback are both already in hand, so no value in VALUE_FNS
            # costs an extra sim step -- the choice is free at this point.
            values[start:end] = VALUE_FNS[self._score_key](
                agent_pos, feedback)[:m].astype(np.float32)
            end_states[start:end] = np.concatenate(
                [agent_pos[:m], feedback[:m]], axis=-1).astype(np.float32)

            if render:
                # one render per chunk, at the state the action chunk landed on. Same
                # transform as PushTImageEnv._get_obs: HWC uint8 -> CHW float32 in [0,1].
                frames = np.asarray(vec.call('render_obs'), dtype=np.float32)  # (n,H,W,3)
                images[start:end] = np.moveaxis(frames[:m] / 255.0, -1, 1)

        out = (
            torch.as_tensor(values, dtype=action.dtype, device=action.device),
            torch.as_tensor(end_states, dtype=action.dtype, device=action.device),
        )
        if render:
            out = out + (
                torch.as_tensor(images, dtype=action.dtype, device=action.device),)
        return out

    def close(self):
        # force_close, not _vec.close(): a plain close() first tries to drain whatever call
        # is in flight, with no timeout, so a worker that died mid-reply hangs teardown
        # forever. That happened -- a training run sat wedged inside this call for 13.5h
        # holding a GPU, with SLURM still reporting it RUNNING and the original traceback
        # swallowed by the stuck unwind. See force_close for the full account.
        if self._vec is not None:
            vec, self._vec = self._vec, None
            force_close(vec)
