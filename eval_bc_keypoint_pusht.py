"""Success rate at every checkpoint for the PushT KEYPOINT BC arm.

``eval_search_pusht.py`` cannot do this job: it constructs a ``PushTImageEnv`` directly and
threads a verifier through a best-of-n search loop, so it is image-only by construction and
has no notion of a keypoint observation. Rather than teach that measurement script -- which
published image results depend on -- a second observation mode, this script evaluates the
keypoint arm the one way that cannot drift from training:

    IT REUSES THE RUN'S OWN ``task.env_runner``.

The checkpoint payload carries the config the run was trained under, so instantiating
``cfg.task.env_runner`` yields the exact ``PushTSearchKeypointsRunner`` that produced the
in-training numbers -- same 30 held-out episodes, same recorded initial states, same 300-step
budget, same success predicate (episode max coverage >= 1.0). There is no second definition
of the rollout here to fall out of step with the first.

This arm has NO VERIFIER and NO SEARCH, so there is no n to sweep: every control step
executes a single sample. The output is therefore one success rate per checkpoint, not a
best-of-n curve, and it is deliberately NOT written into ``bon_search/`` where the search
arms' curves live.

Single checkpoint:
  python eval_bc_keypoint_pusht.py -c <run>/checkpoints/step_0010000.ckpt
Watcher (evaluates each step_*.ckpt as training writes it):
  python eval_bc_keypoint_pusht.py --watch --run-dir <train output_dir> --idle-exit-sec 7200

Results land in <run-dir>/bc_eval/success_rates.jsonl, one row per checkpoint. Nothing in
this script nominates a best checkpoint -- it records measurements and the step is chosen by
hand from the curve.
"""
import sys

if __name__ == '__main__':
    # Line-buffered so a SLURM log shows progress instead of arriving in 8 KB blocks. ONLY
    # when run as a script -- re-opening the fd on IMPORT breaks pytest's capture machinery.
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import contextlib
import fcntl
import json
import pathlib
import re
import time

import click
import hydra
import numpy as np
import torch

from diffusion_policy.common.stats_util import wilson_interval

# FULL FP32, for the reason eval_search_pusht.py documents at length: cuDNN's TF32 path
# picks different kernels for different batch widths, which makes an episode's actions
# depend on --n-envs, and a 300-step contact sim amplifies that into different coverage.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CKPT_RE = re.compile(r'step_(\d+)\.ckpt$')


def step_from_ckpt(path):
    m = CKPT_RE.search(str(path))
    return int(m.group(1)) if m else None


def load_policy(checkpoint, device):
    """Rebuild the EMA policy and its config from a checkpoint payload.

    Deliberately the same shape as eval_search_pusht.load_policy, minus the sampler
    overrides that only make sense for the search arms. EMA weights are what training rolls
    out, validates and ships, so they are what gets scored here too.
    """
    import dill
    from diffusion_policy.workspace.base_workspace import BaseWorkspace

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.get('use_ema', False) and getattr(workspace, 'ema_model', None) is not None:
        policy = workspace.ema_model
    policy.to(torch.device(device))
    policy.eval()
    return policy, cfg


def eval_checkpoint(checkpoint, device, output_dir, n_envs=None, max_steps=None, seed=None):
    """Roll the checkpoint out on its own held-out split and return the success rates.

    Returns a dict keyed by split prefix ('test_', and 'val_' when the manifest has one), so
    a manifest with a non-empty val split is reported rather than silently dropped.
    """
    policy, cfg = load_policy(checkpoint, device)

    # MUST EXIST BEFORE THE RUNNER STARTS. The runner writes rollout videos to
    # `<output_dir>/media` with `mkdir(parents=False)`, so it creates `media` but NOT the
    # directory above it -- during training that one is the hydra run dir, which already
    # exists. Here it is a fresh path, and without this every render worker dies with
    # FileNotFoundError before a single episode runs.
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner_cfg = cfg.task.env_runner
    overrides = dict()
    if n_envs is not None:
        overrides['n_envs'] = int(n_envs)
    if max_steps is not None:
        overrides['max_steps'] = int(max_steps)
    # No search on this arm; make that explicit rather than inheriting whatever the config
    # happened to carry.
    overrides['n_search_actions'] = 1
    env_runner = hydra.utils.instantiate(runner_cfg, output_dir=str(output_dir), **overrides)

    # Seed before the rollout, exactly as the training loop does: the env is deterministic
    # given a reset state, but conditional_sample draws from the global RNG, so without this
    # the number is not reproducible from the checkpoint.
    if seed is None:
        seed = int(cfg.training.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        log = env_runner.run(policy)
    finally:
        # The runner holds a pool of worker SUBPROCESSES that garbage collection will not
        # reap. In --watch this leaks one pool per checkpoint and eventually exhausts the
        # process table. `close()` is the runner's own contract (it force_closes rather than
        # draining, so one dead worker cannot wedge teardown), and it is what
        # BaseWorkspace._close_worker_pools calls. Best-effort, so a teardown failure never
        # masks a real eval error.
        close = getattr(env_runner, 'close', None)
        if close is not None:
            with contextlib.suppress(Exception):
                close()

    out = {'checkpoint': str(checkpoint), 'seed': int(seed), 'splits': {}}
    # DISCOVER the split prefixes from the log rather than assuming them. The runner spells
    # them 'val/' and 'test/' WITH A SLASH (pusht_search_keypoints_runner.splits), and an
    # assumed 'test_' silently matched nothing and produced an empty result that looked like
    # a successful eval. Reading them back off the log cannot drift from the runner, and a
    # manifest with no val split simply yields one prefix instead of two.
    prefixes = sorted(k[:-len('success_rate')] for k in log if k.endswith('success_rate'))
    for prefix in prefixes:
        # Recover the episode count from the per-episode entries the runner logs, so the
        # Wilson interval is computed against the real n rather than an assumed one.
        n_ep = sum(1 for k in log if k.startswith(prefix + 'sim_max_reward_ep'))
        rate = float(log[prefix + 'success_rate'])
        k_succ = int(round(rate * n_ep)) if n_ep else 0
        lo, hi = wilson_interval(k_succ, n_ep)
        out['splits'][prefix.rstrip('/_')] = {
            'success_rate': rate,
            'n_episodes': n_ep,
            'n_success': k_succ,
            'ci95': [lo, hi],
            'mean_score': float(log.get(prefix + 'mean_score', float('nan'))),
        }
    if not out['splits']:
        # LOUD, not an empty row. A rollout that scored nothing means the runner's log keys
        # changed shape; recording {} would put a row on the curve that reads as "evaluated"
        # and carries no measurement.
        raise RuntimeError(
            f'{checkpoint}: the rollout produced no success_rate key. Log keys were: '
            f'{sorted(log)[:20]}')
    return out


@contextlib.contextmanager
def _locked(path):
    """Exclusive lock around the read-modify-write of the shared results file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.lock', 'w') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def results_path(out_root):
    return pathlib.Path(out_root) / 'success_rates.jsonl'


def read_rows(out_root):
    path = results_path(out_root)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_row(out_root, step, result):
    """Append one checkpoint's measurement, replacing any earlier row for the same step."""
    path = results_path(out_root)
    row = {'step': int(step), **result}
    with _locked(path):
        rows = [r for r in read_rows(out_root) if r.get('step') != int(step)]
        rows.append(row)
        rows.sort(key=lambda r: r.get('step', 0))
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return row


def _describe(row):
    parts = []
    for split, s in sorted(row.get('splits', {}).items()):
        parts.append(f"{split}: {s['success_rate']:.3f} "
                     f"({s['n_success']}/{s['n_episodes']}, "
                     f"95% CI {s['ci95'][0]:.3f}-{s['ci95'][1]:.3f})")
    return '  '.join(parts) if parts else 'no split reported'


@click.command()
@click.option('-c', '--checkpoint', default=None, help='single checkpoint to evaluate')
@click.option('--watch', is_flag=True, help='poll run-dir/checkpoints and eval each new step_*.ckpt')
@click.option('--run-dir', default=None, help='training output dir (contains checkpoints/)')
@click.option('-o', '--output-dir', default=None,
              help='where to write success_rates.jsonl (default <run-dir>/bc_eval)')
@click.option('-d', '--device', default='cuda:0')
@click.option('--n-envs', default=None, type=int,
              help='parallel envs; default is the training config\'s value')
@click.option('--max-steps', default=None, type=int,
              help='episode length; default is the training config\'s value (300)')
@click.option('--seed', default=None, type=int,
              help='default: the run\'s own training.seed, so a rerun reproduces')
@click.option('--poll-sec', default=60.0)
@click.option('--idle-exit-sec', default=None, type=float,
              help='exit after this long with no new checkpoint (watch mode)')
@click.option('--redo', is_flag=True, help='re-evaluate steps already present in the results file')
def main(checkpoint, watch, run_dir, output_dir, device, n_envs, max_steps, seed,
         poll_sec, idle_exit_sec, redo):
    if watch == bool(checkpoint):
        raise click.UsageError('pass either -c/--checkpoint or --watch (with --run-dir)')

    if not watch:
        ckpt = pathlib.Path(checkpoint)
        out_root = pathlib.Path(output_dir) if output_dir else ckpt.parent.parent / 'bc_eval'
        step = step_from_ckpt(ckpt)
        result = eval_checkpoint(ckpt, device, out_root / 'render', n_envs=n_envs,
                                 max_steps=max_steps, seed=seed)
        if step is not None:
            row = append_row(out_root, step, result)
            print(f'step {step}: {_describe(row)}')
            print(f'wrote {results_path(out_root)}')
        else:
            print(f'{ckpt.name}: {_describe(result)}')
            print('(no step_*.ckpt in the name, so nothing was recorded)')
        return

    if not run_dir:
        raise click.UsageError('--watch requires --run-dir')
    run_dir = pathlib.Path(run_dir)
    ckpt_dir = run_dir / 'checkpoints'
    out_root = pathlib.Path(output_dir) if output_dir else run_dir / 'bc_eval'

    # Resume-safe: this watcher runs on the preemptible ckpt partition, so an in-memory-only
    # `seen` would re-evaluate every checkpoint from scratch after each preemption.
    seen = set() if redo else {r['step'] for r in read_rows(out_root) if 'step' in r}
    if seen:
        print(f'resuming: {len(seen)} checkpoint(s) already evaluated, skipping those')
    print(f'watching {ckpt_dir} for step_*.ckpt (poll {poll_sec}s)')

    failed = {}
    last_progress = time.time()
    while True:
        ckpts = sorted(ckpt_dir.glob('step_*.ckpt')) if ckpt_dir.is_dir() else []
        for ckpt in ckpts:
            step = step_from_ckpt(ckpt)
            if step is None or step in seen:
                continue
            print(f'== evaluating {ckpt.name} (step {step}) ==')
            try:
                result = eval_checkpoint(ckpt, device, out_root / f'step_{step:07d}',
                                         n_envs=n_envs, max_steps=max_steps, seed=seed)
            except Exception as e:
                # NOT marked seen: the checkpoint may simply have been half-written (the
                # trainer saves on a background thread), or this may be a transient CUDA OOM.
                failed[step] = failed.get(step, 0) + 1
                print(f'eval failed for {ckpt.name} (attempt {failed[step]}): {e}')
                if failed[step] >= 3:
                    print(f'  giving up on step {step} after 3 attempts')
                    seen.add(step)
                continue
            row = append_row(out_root, step, result)
            seen.add(step)
            last_progress = time.time()
            print(f'step {step}: {_describe(row)}')

        if idle_exit_sec is not None and (time.time() - last_progress) > idle_exit_sec:
            print(f'no new checkpoint for {idle_exit_sec}s, exiting')
            break
        time.sleep(poll_sec)


if __name__ == '__main__':
    main()
