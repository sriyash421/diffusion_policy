"""Preflight for the image-only ablation: prove the key deletion is threaded end to end.

The ablation is expressed as two hydra key deletions and NOTHING else:

    ~task.shape_meta.obs.agent_pos  ~task.shape_meta.obs.feedback

That is cheap to write and easy to get silently wrong, in two directions at once:

  1. TOO LITTLE -- the keys are still reaching the encoder, so the "image-only" arm is
     secretly the control and the whole experiment is a null by construction.
  2. TOO MUCH -- the keys stop reaching the VERIFIER, which resets a pymunk sim from
     `agent_pos` + `feedback` (PushTSearchMixin._verifier_inputs). Then the search is
     scoring garbage and a gap gets attributed to the policy's observation when it was
     really the verifier's. `_normalize_value` also reads normalizer['feedback'].

Neither failure raises on its own -- both train happily and produce plausible curves. So
this asserts both, on the train path AND on the eval path.

THE EVAL PATH IS THE POINT. eval_search_pusht.load_policy rebuilds the policy from
`payload['cfg']` -- the checkpoint's own config -- so the reduced shape_meta is supposed to
ride inside the checkpoint with no eval-side flag to remember. Meanwhile build_envs
constructs PushTFeedbackWrapper(PushTImageEnv(...)) and the runner does
`np_obs_dict = dict(obs)`, never consulting shape_meta, so the obs dict at eval still
carries all three keys. This script does that round trip for real: saves a payload, reloads
it through load_policy, and re-checks both invariants on the reloaded policy.

NOTE ON WIDTH. `n_candidates` lives in train_pusht_diffusion_search.yaml and defaults to
16, and ..._single.yaml INHERITS that -- the config name selects the trainer, not the
width. So pass the width explicitly, exactly as scripts/run_nopos_30demo.sh does; the k=16
path is minutes slower here because each of the 15 context candidates costs 8 DDIM steps
plus a sim rollout on CPU.

    # ST k=1
    python scripts/nopos_smoke.py --config-name train_pusht_diffusion_search_single \
        -o n_candidates=1
    # ST k=16
    python scripts/nopos_smoke.py --config-name train_pusht_diffusion_search \
        -o n_candidates=16
    # UNet BC
    python scripts/nopos_smoke.py --config-name train_pusht_unet_bc


RETIRED 2026-08-29. The two hydra deletions this checks now target keys that no longer
exist: image-only stopped being an ablation and became the task, so
config/task/pusht_image_search_imgonly.yaml declares `image` alone and the policy
asserts no low_dim obs key. Kept as the record of a finished experiment; it raises
rather than reproducing anything.

"""
import argparse
import pathlib
import sys
import tempfile
import time

import torch

ROOT = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT)

import dill                                                    # noqa: E402
import hydra                                                   # noqa: E402
from hydra import compose, initialize_config_dir              # noqa: E402
from omegaconf import OmegaConf                                # noqa: E402

OmegaConf.register_new_resolver('eval', eval, replace=True)

from diffusion_policy.model.common.normalizer import LinearNormalizer   # noqa: E402

# The deletions under test. Kept as one list so this script and
# scripts/run_nopos_30demo.sh cannot drift on what "no pos" means.
NOPOS = ['~task.shape_meta.obs.agent_pos', '~task.shape_meta.obs.feedback']

# resnet18 global-avg-pool. 530 = this + agent_pos(2) + feedback(16).
IMAGE_ONLY_DIM = 512

failures = []
_t0 = time.time()


def stage(msg):
    """Progress marker. Flushed, because the two policy builds take ~2 min each on CPU and
    a silent log is indistinguishable from a hang."""
    print(f'[{time.time() - _t0:6.1f}s] {msg}', flush=True)


def check(ok, msg):
    print(('  [ok] ' if ok else '  [FAIL] ') + msg, flush=True)
    if not ok:
        failures.append(msg)


def fake_normalizer():
    """Params for ALL THREE obs keys, exactly as PushTImageDataset.get_normalizer fits them.

    The dataset does not read shape_meta -- it hardcodes the three keys -- so the normalizer
    inside a nopos checkpoint still holds `agent_pos` and `feedback`. That is load-bearing:
    the ST path normalizes the WHOLE obs dict, and LinearNormalizer._normalize_impl does
    `self.params_dict[key]` for every key present, so a normalizer missing them would
    KeyError at the first eval step. Fitting all three here reproduces that.
    """
    nrm = LinearNormalizer()
    nrm.fit({'action': torch.rand(64, 2) * 100,
             'agent_pos': torch.rand(64, 2) * 400,
             'feedback': torch.rand(64, 16) * 50})
    from diffusion_policy.common.normalize_util import get_image_range_normalizer
    nrm['image'] = get_image_range_normalizer()
    return nrm


def obs_batch(B, To, dev='cpu'):
    """What the dataset and the env wrapper BOTH emit: all three keys, always."""
    return {'image': torch.rand(B, To, 3, 96, 96, device=dev),
            'agent_pos': torch.rand(B, To, 2, device=dev) * 400,
            'feedback': torch.rand(B, To, 16, device=dev) * 50}


def check_policy(policy, where):
    enc = policy.obs_encoder
    check(list(enc.low_dim_keys) == [],
          f'{where}: obs_encoder.low_dim_keys is empty (got {list(enc.low_dim_keys)})')
    dim = int(enc.output_shape()[0])
    check(dim == IMAGE_ONLY_DIM,
          f'{where}: obs_encoder.output_shape() == ({IMAGE_ONLY_DIM},) (got ({dim},))')
    check(policy.normalizer.params_dict.get('feedback') is not None,
          f'{where}: normalizer still holds feedback (the verifier scalar is rescaled from it)')


def check_search(policy, obs, where):
    """One best-of-n readout. Only succeeds if the verifier got agent_pos + feedback."""
    policy.eval()
    with torch.no_grad():
        out = policy.predict_action_best(obs, n_actions=2)
    act = out['action']
    check(torch.isfinite(act).all(), f'{where}: predict_action_best returned finite actions')
    check(act.shape[0] == obs['image'].shape[0],
          f'{where}: best-of-n ran over the batch (shape {tuple(act.shape)})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-name', default='train_pusht_diffusion_search_single')
    ap.add_argument('-o', '--override', action='append', default=[],
                    help='extra hydra override, repeatable (e.g. -o n_candidates=1)')
    # CPU is not a practical default at k=16: the context search is 15 candidates x 8 DDIM
    # steps plus a sim rollout each, unbatched, which runs for tens of minutes. Run this on
    # a ckpt-partition GPU -- scripts/slurm/nopos_smoke.sbatch.
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device(args.device)
    overrides = NOPOS + [
        f'training.device={args.device}',
        # A 32-process AsyncVectorEnv is the production setting and is far too heavy for a
        # preflight; 2 sync envs exercise the same reset-from-obs path.
        'policy.verifier_n_envs=2',
        'policy.verifier_use_async=False',
    ] + args.override
    print(f'== {args.config_name}  {" ".join(args.override) or "(no extra overrides)"}')

    with initialize_config_dir(config_dir=str(pathlib.Path(ROOT, 'diffusion_policy/config')),
                               version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    check(list(cfg.shape_meta.obs.keys()) == ['image'],
          f'resolved shape_meta.obs is image-only (got {list(cfg.shape_meta.obs.keys())})')
    check(cfg.verifier_tag == 't_goal', f'verifier_tag is t_goal (got {cfg.verifier_tag})')

    stage('building policy...')
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(fake_normalizer())
    policy.to(dev)
    print(f'  built {type(policy).__name__}: '
          f'{sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M params, '
          f'max_actions={getattr(policy, "max_actions", "n/a")}', flush=True)
    check_policy(policy, 'fresh')

    # --- train path: the batch still carries all three keys, as the dataset emits them ---
    B, To = 2, cfg.n_obs_steps
    batch = {'action': torch.rand(B, cfg.horizon, 2, device=dev) * 400,
             'obs': obs_batch(B, To, dev)}
    stage('train path: compute_loss...')
    policy.train()
    loss = policy.compute_loss(batch)
    loss.backward()
    gnorm = sum(float(p.grad.norm() ** 2)
                for p in policy.parameters() if p.grad is not None) ** 0.5
    check(torch.isfinite(loss) and gnorm > 0,
          f'compute_loss ran with the extra keys present (loss {float(loss):.4f}, '
          f'grad norm {gnorm:.4f})')

    # --- eval path, in-process ---
    stage('eval path (in-process): predict_action_best...')
    check_search(policy, obs_batch(B, To, dev), 'fresh')

    # --- eval path, THROUGH A CHECKPOINT: the claim that needs no eval-side flag ---
    stage('eval path (through a checkpoint): save + load_policy...')
    import eval_search_pusht
    with tempfile.TemporaryDirectory() as td:
        ckpt = pathlib.Path(td, 'step_0000000.ckpt')
        # Exactly the shape BaseWorkspace.save_checkpoint writes: the run's own cfg plus
        # BOTH model copies. load_policy reads `payload['cfg']` and rebuilds from it.
        #
        # `ema_model` IS REQUIRED, not belt-and-braces. All three configs set
        # `training.use_ema: True`, and load_policy then returns `workspace.ema_model`, not
        # `workspace.model`. A payload without it does not raise: load_payload simply has
        # nothing to load into the freshly-constructed EMA copy, so eval silently proceeds
        # on an UNTRAINED policy whose normalizer is an empty ParameterDict. The first
        # symptom is `AttributeError: 'ParameterDict' object has no attribute 'image'`,
        # nowhere near the cause. (A real checkpoint carries model, ema_model and optimizer,
        # each with all four normalizer entries -- verified against step_0010000.ckpt of
        # outer_inner/value_k16_corrupt-False_demos-30_seed-42.)
        sd = policy.state_dict()
        with ckpt.open('wb') as f:
            torch.save({'cfg': cfg,
                        'state_dicts': {'model': sd, 'ema_model': sd},
                        'pickles': {}}, f, pickle_module=dill)
        reloaded, rcfg = eval_search_pusht.load_policy(str(ckpt), device=args.device)
        check(list(rcfg.shape_meta.obs.keys()) == ['image'],
              'checkpoint cfg round-trips image-only shape_meta')
        # Prove the reload actually RESTORED weights rather than handing back a fresh
        # policy. Without this the encoder checks below would pass on an untrained model --
        # they only read shapes, which a freshly-constructed policy also has.
        ref = next(iter(policy.obs_encoder.state_dict().values())).cpu()
        got = next(iter(reloaded.obs_encoder.state_dict().values())).cpu()
        check(torch.equal(ref, got),
              'reloaded: obs_encoder weights match the saved policy (not a fresh model)')
        check_policy(reloaded, 'reloaded')
        check_search(reloaded, obs_batch(B, To, dev), 'reloaded')

    for p in (policy, reloaded):
        if hasattr(p, 'close'):
            p.close()

    print()
    if failures:
        print(f'FAILED ({len(failures)}):')
        for f in failures:
            print('  -', f)
        sys.exit(1)
    print('all checks passed')


if __name__ == '__main__':
    main()
