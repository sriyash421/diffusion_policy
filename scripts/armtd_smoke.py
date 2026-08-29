"""Smoke test for `armTd` -- the cross-candidate (dynamically standardized) verifier value.

`armTd` is the only verifier value that is not a pure function of one candidate: it divides
each distance term by the spread measured ACROSS the n candidates of the control step being
decided, rather than by the two constants `armTn` uses. That makes several things checkable
here that no other value needs, and two of them would otherwise fail silently:

  1. The float64 cast is load-bearing, not tidiness. In float32 `torch.std(unbiased=False)`
     of n BIT-IDENTICAL values is 0 for n in (3, 5) but ONE ULP for n >= 7, which the
     division turns into |z| = 0.88 -- a full-magnitude score manufactured out of rounding,
     on exactly the pre-contact steps where the arm term is supposed to decide. It hits
     every evaluated n except 1, 2 and 4, and only for values that are not exactly
     representable, so a round-number test would pass. Check 1 pins this down.
  2. n=1 must be a no-op: score exactly 0, and the executed action bitwise identical to
     what every other value produces, since no selection happens with one candidate.
  3. Fusion must not perturb the sampler's noise stream -- same seed must give the same
     candidates under armTn and armTd, with only the SCORES differing.
  4. The guards must fire: armTd on a policy that consumes search context, armTd as a
     training `verifier_tag`, and a bad value_fn assigned to a built verifier.

Checks 1 and 5a are pure functions (no sim, no GPU, instant). The rest build a tiny policy
and a pool of PushT sims, so run this on a compute node -- the login node's shared ~10GB
cgroup will not tolerate a ResNet18 plus the sim pool.

  python scripts/armtd_smoke.py
"""
if __name__ == "__main__":
    # `python scripts/armtd_smoke.py` puts scripts/ on sys.path, not the repo root, so
    # `import diffusion_policy` fails however the CWD is set. Same preamble as the siblings.
    import sys, os, pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import numpy as np
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

import hydra
from hydra import compose, initialize_config_dir

from diffusion_policy.env.pusht.pusht_verifier import (
    value_arm_t_dyn, value_terms_from_state, check_verifier_value,
    PushTVerifier, ARM_TD_EPS_PX, CROSS_CANDIDATE_VALUES, base_value_fn)
from scripts.selection_smoke import _tiny, _fake_normalizer, CFG_DIR

# The n's the real sweep evaluates, plus 7 -- the smallest n where float32's pairwise
# accumulation starts leaving a residual on identical inputs.
N_GRID = (1, 2, 4, 7, 8, 16, 32, 64)

# A d_T->goal that is NOT exactly representable in float32. Using a round number here
# (267.5, say) makes check 1 pass under float32 too and defeats the whole point.
DEGENERATE_DT = 80.123456


def _terms(dT, dArm):
    """(n,) , (n,) -> (1, n, 2) in the layout value_arm_t_dyn expects."""
    return torch.stack([torch.as_tensor(dT, dtype=torch.float32),
                        torch.as_tensor(dArm, dtype=torch.float32)], dim=-1).unsqueeze(0)


def check_fusion_pure(failures):
    """Checks 1 + 5a: the fusion rule on its own, no sim."""
    for n in N_GRID:
        # pre-contact: every candidate returns the IDENTICAL d_T (nothing moved the block),
        # so the arm term must be the only thing that decides.
        dT = np.full(n, DEGENERATE_DT, dtype=np.float32)
        dArm = np.arange(n, dtype=np.float32)[::-1].copy() + 5.0   # argmin at the LAST slot
        score = value_arm_t_dyn(_terms(dT, dArm))
        if n == 1:
            if not torch.equal(score, torch.zeros_like(score)):
                failures.append(f'n=1 fused score is {score.tolist()}, must be exactly 0')
            continue
        want = int(dArm.argmin())
        got = int(score.argmax(dim=1))
        if got != want:
            failures.append(
                f'degenerate d_T at n={n}: argmax slot {got}, expected {want} (the arm '
                f'term must decide when the task term has zero spread)')
        if got == 0 and want != 0:
            failures.append(f'n={n}: argmax collapsed to slot 0 -- the arm term was lost')

    # the float32 counter-example, asserted to FAIL, so this test documents why the cast in
    # value_arm_t_dyn exists and breaks if someone "optimizes" it away.
    n = 16
    d32 = torch.full((1, n), DEGENERATE_DT, dtype=torch.float32)
    mu, sd = d32.mean(1, keepdim=True), d32.std(1, unbiased=False, keepdim=True)
    z32 = ((d32 - mu) / (sd + ARM_TD_EPS_PX)).abs().max().item()
    d64 = d32.double()
    mu6, sd6 = d64.mean(1, keepdim=True), d64.std(1, unbiased=False, keepdim=True)
    z64 = ((d64 - mu6) / (sd6 + ARM_TD_EPS_PX)).abs().max().item()
    if not (sd.item() > 0.0 and sd6.item() == 0.0):
        failures.append(
            f'the float32 residual this guards against is gone (f32 sd={sd.item():.3e}, '
            f'f64 sd={sd6.item():.3e}); re-derive value_arm_t_dyn\'s docstring before '
            f'relaxing the cast')
    print(f'  [ok] degenerate d_T at n={n}: float32 |z|={z32:.3e} vs float64 |z|={z64:.3e}')

    # invariances that define the rule
    torch.manual_seed(0)
    t = torch.stack([torch.rand(1, 8) * 30, torch.rand(1, 8) * 200], dim=-1)
    base = value_arm_t_dyn(t)
    shifted = value_arm_t_dyn(t + torch.tensor([13.0, 0.0]))
    if not torch.allclose(base, shifted, atol=1e-5):
        failures.append('adding a constant to d_T changed the score; z must be shift-invariant')
    for c in (1.5, 7.3, 1000.0):
        scaled = value_arm_t_dyn(t * torch.tensor([1.0, c]))
        # ARGMAX invariance is the contract. The scores themselves move by ~eps/sd, since
        # the pixel floor deliberately breaks exact scale invariance below itself.
        if not torch.equal(base.argmax(1), scaled.argmax(1)):
            failures.append(f'scaling d_arm by {c} changed the argmax; armTd must be '
                            f'scale-invariant above the floor (this is what armTn cannot do)')
    if base.sum(dim=1).abs().max().item() > 1e-4:
        failures.append(f'fused scores do not sum to ~0 across candidates: {base.sum().item()}')
    print('  [ok] shift/scale invariance and zero-sum')


def check_terms_from_state(failures):
    """value_terms_from_state must mirror the numpy distance fns it replaces."""
    from diffusion_policy.env.pusht.feedback_util import t_goal_distance, arm_to_t_distance
    torch.manual_seed(1)
    B = 50   # deliberately NOT a multiple of verifier_n_envs=32
    state = torch.rand(B, PushTVerifier.STATE_DIM) * 200
    got = value_terms_from_state(state)
    ap = state[:, :PushTVerifier.AGENT_DIM].numpy()
    fb = state[:, PushTVerifier.AGENT_DIM:].numpy()
    want = np.stack([t_goal_distance(fb), arm_to_t_distance(ap, fb)], axis=-1)
    err = np.abs(got.numpy() - want).max()
    if err > 1e-3:
        failures.append(f'value_terms_from_state disagrees with the numpy fns by {err:.3e} px')
    print(f'  [ok] torch/numpy term mirror agrees to {err:.2e} px at B={B}')


def check_guards(failures):
    """Check 4: every guard fires, and none of them fires on the allowed combination."""
    def raises(fn, kind):
        try:
            fn()
        except kind:
            return True
        except Exception as e:                       # noqa: BLE001 - want the actual type
            failures.append(f'expected {kind.__name__}, got {type(e).__name__}: {e}')
            return False
        return False

    if not raises(lambda: check_verifier_value(
            {'verifier_tag': 'armTd', 'policy': {'verifier_value': 'armTd'}}), ValueError):
        failures.append('armTd was accepted as a training verifier_tag')
    v = PushTVerifier(value_fn='armTd')
    if v._score_key != base_value_fn('armTd'):
        failures.append(f'armTd verifier scores the sim with {v._score_key!r}, '
                        f'expected {base_value_fn("armTd")!r}')
    if not raises(lambda: setattr(v, 'value_fn', 'nonsense'), AssertionError):
        failures.append('a bogus value_fn was accepted by the setter')
    # the allowed combination must NOT raise
    for ok_tag in ('t_goal', 'armTn'):
        check_verifier_value({'verifier_tag': ok_tag, 'policy': {'verifier_value': ok_tag}})
    print('  [ok] training-tag, setter and per-candidate-value guards')


def _build(overrides, n_envs=4):
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        cfg = compose(config_name='train_pusht_unet_bc', overrides=overrides)
    cfg = _tiny_unet(cfg, n_envs)
    policy = hydra.utils.instantiate(cfg.policy)
    _fake_normalizer(policy, cfg.policy.shape_meta)
    policy.eval()
    return policy, cfg


def _tiny_unet(cfg, n_envs):
    """_tiny() shrinks transformer keys the UNet config does not have."""
    cfg.policy.num_inference_steps = 2
    cfg.policy.verifier_n_envs = n_envs
    cfg.policy.down_dims = [64, 128]
    cfg.policy.diffusion_step_embed_dim = 64
    return cfg


def _obs(cfg, B=2):
    To = cfg.policy.n_obs_steps
    obs = {}
    for key, attr in cfg.policy.shape_meta['obs'].items():
        shape = tuple(attr['shape'])
        if attr.get('type', 'low_dim') == 'rgb':
            obs[key] = torch.rand(B, To, *shape)
        else:
            obs[key] = torch.rand(B, To, *shape) * 400 + 50
    return obs


def _swap_value(policy, value):
    """Repoint an ALREADY-BUILT policy at another verifier value.

    Mirrors eval_search_pusht.py's --verifier-value override exactly, and that is the whole
    point: these checks compare ranking rules on ONE set of weights, the way the real
    re-rank does. Rebuilding the policy per value instead would give each a different random
    init, and every "the action changed" assertion below would fire on that rather than on
    anything to do with the verifier.
    """
    policy._search_kwargs['verifier_value'] = value
    built = policy.__dict__.get('_verifier')
    if built is not None:
        built.value_fn = value


def check_policy(failures):
    """Checks 2, 3, 5b: n=1 is a no-op, the RNG is untouched, and armTd actually bites."""
    policy, cfg = _build(['verifier_tag=armTn', 'policy.verifier_value=armTn'])
    try:
        obs = _obs(cfg)

        # ---- n=1: no selection happens, so every value must execute the SAME action
        ref, ref_value = None, None
        for value in ('t_goal', 'armTn', 'armTd'):
            _swap_value(policy, value)
            torch.manual_seed(7)
            out1 = policy.predict_action_best(obs, n_actions=1)
            if value == 'armTd':
                sc = out1.get('scores')
                if sc is not None and not torch.equal(sc, torch.zeros_like(sc)):
                    failures.append(f'armTd n=1 scores are {sc.tolist()}, must be exactly 0')
            if ref is None:
                ref, ref_value = out1['action_pred'], value
            elif not torch.equal(ref, out1['action_pred']):
                d = (ref - out1['action_pred']).abs().max().item()
                failures.append(
                    f'n=1 action under {value!r} differs from {ref_value!r} by {d:.3e}; no '
                    f'selection happens at n=1, so it must be bitwise equal')
        print('  [ok] n=1 is a no-op across all three values')

        # ---- n=8: fusion must change the SCORES and nothing else
        got = {}
        for value in ('armTn', 'armTd'):
            _swap_value(policy, value)
            torch.manual_seed(23)
            a, _, sc = policy.predict_n_actions(
                obs, verifier=policy.verifier, n_actions=8, return_scores=True)
            got[value] = (a.clone(), sc.clone())
    finally:
        policy.close()

    if not torch.equal(got['armTn'][0], got['armTd'][0]):
        d = (got['armTn'][0] - got['armTd'][0]).abs().max().item()
        failures.append(f'armTd changed the CANDIDATES by {d:.3e}, not just their scores -- '
                        f'fusion must not touch the sampler noise stream')
    sc = got['armTd'][1]
    if sc.shape != got['armTn'][1].shape:
        failures.append(f'armTd scores have shape {tuple(sc.shape)}, expected '
                        f'{tuple(got["armTn"][1].shape)}')
    if not torch.isfinite(sc).all():
        failures.append('armTd produced non-finite scores')
    if sc.mean(dim=1).abs().max().item() > 1e-3:
        failures.append(f'armTd scores are not zero-mean per row: {sc.mean(dim=1).tolist()}')
    if torch.equal(got['armTn'][1], sc):
        failures.append('armTd scores are identical to armTn -- fusion never ran')
    print(f'  [ok] candidates identical, scores differ; armTd argmax '
          f'{got["armTd"][1].argmax(1).tolist()} vs armTn {got["armTn"][1].argmax(1).tolist()}')


def check_passthrough(failures):
    """The refactor must be TRANSPARENT for every per-candidate value.

    `_score_candidates` grew a 4th return value and `predict_n_actions` grew a shared exit
    with a fusion step. Neither may perturb a value that is not cross-candidate: for
    t_goal / armTn the scores coming out of predict_n_actions must be EXACTLY what the
    verifier reports for those same candidates, with no fusion applied.

    This is what distinguishes "the refactor changed the ranking" from "the eval drifted
    across nodes" -- a question re-running the sweep cannot settle, because the eval is
    documented as not bit-exact across batch widths / hardware.
    """
    for value in ('t_goal', 'armTn'):
        policy, cfg = _build([f'verifier_tag=armTn', f'policy.verifier_value={value}'])
        try:
            if policy._fuses_scores():
                failures.append(f'{value!r} reports _fuses_scores()=True; only a '
                                f'cross-candidate value may fuse')
            obs = _obs(cfg)
            torch.manual_seed(31)
            actions, _, scores = policy.predict_n_actions(
                obs, verifier=policy.verifier, n_actions=4, return_scores=True)
            # re-score the SAME candidates straight through the verifier; the sim is
            # deterministic given (state, action), so this must reproduce them exactly.
            direct = torch.stack([
                policy._score_candidates(policy.verifier, obs, actions[:, k])[1]
                for k in range(actions.shape[1])], dim=1)
            if not torch.equal(scores, direct):
                d = (scores - direct).abs().max().item()
                failures.append(
                    f'{value!r}: predict_n_actions scores differ from a direct verifier '
                    f're-score by {d:.3e} -- the refactor is NOT transparent for a '
                    f'per-candidate value')
        finally:
            policy.close()
    print('  [ok] t_goal/armTn scores pass through the refactor untouched')


def check_context_policy_rejects(failures):
    """The ST arm must refuse armTd rather than silently mis-conditioning itself."""
    try:
        with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
            cfg = compose(config_name='train_pusht_diffusion_search',
                          overrides=['verifier_tag=armTn', 'policy.verifier_value=armTd'])
        cfg = _tiny(cfg)
        policy = hydra.utils.instantiate(cfg.policy)
        _fake_normalizer(policy, cfg.policy.shape_meta)
        policy.verifier          # in case a subclass builds it lazily instead
        failures.append('PushTDiffusionSearchPolicy accepted armTd; its search context is '
                        'causal and cannot carry a statistic over all n candidates')
    except Exception as e:                                   # noqa: BLE001
        # This policy builds its verifier in __init__, so hydra.instantiate wraps the
        # ValueError in an InstantiationException. Walk the __cause__ chain rather than
        # matching on the wrapper, which would make the test hydra-version-dependent.
        chain, cur = [], e
        while cur is not None:
            chain.append(cur)
            cur = cur.__cause__
        cause = next((c for c in chain if isinstance(c, ValueError)), None)
        if cause is None or 'causal' not in str(cause):
            failures.append(f'ST policy raised, but not the causal-context ValueError: '
                            f'{type(e).__name__}: {e}')
        else:
            print('  [ok] ST policy refuses armTd (causal search context, at construction)')


def main():
    failures = []
    print('fusion rule (pure):')
    check_fusion_pure(failures)
    check_terms_from_state(failures)
    print('guards:')
    check_guards(failures)
    check_context_policy_rejects(failures)
    if os.environ.get('ARMTD_SMOKE_PURE_ONLY'):
        print('\nARMTD_SMOKE_PURE_ONLY set -- skipping the policy checks (need a compute node)')
    else:
        print('policy:')
        check_passthrough(failures)
        check_policy(failures)

    if failures:
        print('\nFAILURES:')
        for f in failures:
            print(' -', f)
        raise SystemExit(1)
    print('\nall armTd smoke checks passed')


if __name__ == '__main__':
    main()
