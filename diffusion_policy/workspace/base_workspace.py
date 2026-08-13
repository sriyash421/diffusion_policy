from typing import Optional
import os
import pathlib
import datetime
import hydra
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import dill
import torch
import threading


def clone_policy(policy):
    """Deep-copy a search policy WITHOUT duplicating its verifier's subprocess pool.

    ``policy.verifier`` owns a pool of worker processes (PushTVerifier holds an
    ``AsyncVectorEnv``), which ``copy.deepcopy`` would either choke on or silently fork a
    second copy of -- doubling the sim processes on every clone. The verifier carries no
    policy state (it is handed the obs and the candidate chunk on each call), so every
    clone shares the original one. Only the live model's ``close()`` is ever called, so the
    shared pool is torn down exactly once.

    Lives here rather than in one workspace because both the offline and the outer/inner
    trainers need it to build their EMA copy.
    """
    verifier = getattr(policy, 'verifier', None)
    if verifier is None:
        return copy.deepcopy(policy)
    policy.verifier = None
    try:
        clone = copy.deepcopy(policy)
    finally:
        policy.verifier = verifier
    clone.verifier = verifier
    return clone


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=True):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        def _atomic_save():
            """Write to a sibling .tmp then rename into place.

            The rename is atomic on POSIX, so a reader never observes a partial file. Two
            consumers depend on this: the eval watcher globs `step_*.ckpt` on existence
            alone and would otherwise torch.load a half-written file, and `latest.ckpt` --
            which `training.resume` reads -- was a truncate-then-write that a preemption
            mid-save could leave truncated.
            """
            tmp = path.with_suffix(path.suffix + '.tmp')
            with tmp.open('wb') as f:
                torch.save(payload, f, pickle_module=dill)
            os.replace(tmp, path)

        if use_thread:
            # A previous save may still be writing (possibly to this same path, e.g. two
            # `latest.ckpt` saves within one epoch). Joining first serializes the writes so
            # they cannot interleave and truncate each other.
            self.join_saving_thread()
            self._saving_thread = threading.Thread(target=_atomic_save)
            self._saving_thread.start()
        else:
            _atomic_save()
        return str(path.absolute())

    def join_saving_thread(self):
        """Block until any in-flight threaded checkpoint save has finished.

        Must be called before the process exits, otherwise the last checkpoint can be
        left truncated on disk.
        """
        thread = getattr(self, '_saving_thread', None)
        if thread is not None:
            thread.join()
            self._saving_thread = None


    def write_manifest(self):
        """Record run identity in <output_dir>/run.json, appending one entry per launch.

        Nothing else on disk ties a run directory to the code and job that produced it:
        the git sha existed only incidentally inside wandb's metadata, and there was no
        back-pointer from a SLURM job id to a run dir at all -- which is why the reporting
        scripts had to hand-maintain arm->path->jobid tables that drifted apart.

        Appending rather than overwriting means requeues and resumes stay visible, so a
        run that restarted three times is distinguishable from one that ran straight
        through.
        """
        import json
        import subprocess

        def _git(*args):
            try:
                return subprocess.run(
                    ['git', *args], cwd=os.path.dirname(os.path.abspath(__file__)),
                    capture_output=True, text=True, timeout=10).stdout.strip()
            except Exception:
                return None

        path = pathlib.Path(self.output_dir).joinpath('run.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = self.cfg
        launch = {
            'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'hostname': os.environ.get('SLURMD_NODENAME') or os.uname().nodename,
            'git_sha': _git('rev-parse', 'HEAD'),
            'git_dirty': bool(_git('status', '--porcelain')),
            'global_step': int(getattr(self, 'global_step', 0)),
        }
        payload = {}
        if path.is_file():
            try:
                payload = json.loads(path.read_text())
            except Exception:
                payload = {}
        payload.update({
            'name': cfg.get('name'),
            'exp_name': cfg.get('exp_name'),
            'trainer': cfg.get('trainer'),
            'task_name': cfg.get('task_name'),
            'search_context': cfg.get('search_context'),
            'corrupt_obs': cfg.get('corrupt_obs'),
            'seed': cfg.training.get('seed'),
            'zarr_path': cfg.task.dataset.get('zarr_path'),
            'train_ratio': cfg.task.dataset.get('train_ratio'),
            'target': cfg.get('_target_'),
        })
        payload.setdefault('launches', []).append(launch)
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, path)
        return str(path)

    def write_splits(self, dataset):
        """Record the exact episodes this run trains/validates/tests on.

        Nothing on disk used to say which episodes a checkpoint had been trained on: the
        three splits were derived at runtime, independently, in the dataset, the env runner
        and the eval script, from five config keys. Changing any one of them silently
        repartitioned the data -- which is how a `n_val_episodes` 10 -> 30 change quietly
        cut the training budget from 29 to 25 episodes.

        On resume this ALSO acts as a guard: if the run directory already has a splits.json
        whose checksum differs from what the current config resolves to, the data changed
        underneath an in-flight experiment and we refuse rather than continue. That guard is
        the main reason this method exists.

        No-op for datasets that do not implement `get_split_indices` (the maze/robomimic
        ones), so it is safe to call unconditionally.
        """
        import json

        get_indices = getattr(dataset, 'get_split_indices', None)
        if get_indices is None:
            return None
        splits = get_indices()

        path = pathlib.Path(self.output_dir).joinpath('splits.json')
        if path.is_file():
            try:
                previous = json.loads(path.read_text())
            except Exception:
                previous = None
            if previous is not None and previous.get('checksum') != splits['checksum']:
                raise RuntimeError(
                    f'{path} records checksum {previous.get("checksum")} but this config '
                    f'resolves to {splits["checksum"]}. The train/val/test partition '
                    f'changed underneath an existing run -- resuming would train on '
                    f'different data than the checkpoints in this directory were built '
                    f'from, and every metric would silently mix the two. Restore the '
                    f'original split settings, or start a new run directory.\n'
                    f'  before: '
                    f'{ {k: len(previous.get(k, [])) for k in ("train", "val", "test")} }\n'
                    f'  now:    '
                    f'{ {k: len(splits[k]) for k in ("train", "val", "test")} }')

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(splits, indent=2) + '\n')
        os.replace(tmp, path)
        return str(path)

    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys)
        return payload

    def assert_ema_payload(self, payload, path):
        """Refuse to resume a pre-EMA checkpoint into a run with ``use_ema: True``.

        ``load_payload`` only restores keys the PAYLOAD actually has, so resuming a
        checkpoint written before EMA existed leaves ``ema_model`` at its RANDOM
        initialization while ``model`` receives the trained weights. Since the EMA copy is
        what gets rolled out, validated and shipped in the checkpoint, every metric until
        the average converges (~200 steps at decay 0.995) would be silently wrong -- and the
        checkpoints written in that window ship a partly-random policy.

        Called from every workspace's resume block. It lives here rather than in one
        workspace because each of the three overrides ``run()`` and so has its own resume
        path; the guard existed only in TrainMLPImageWorkspace, which meant the search arms
        (TrainSearchOuterInnerWorkspace, the default search trainer since 5294c31) and the
        diffusion-UNet arm ran without it.

        No-op when the workspace has no ``ema_model`` (``use_ema: False``).
        """
        if getattr(self, 'ema_model', None) is None:
            return
        if 'ema_model' in payload.get('state_dicts', {}):
            return
        raise RuntimeError(
            f'{path} predates EMA (no ema_model in the payload) but training.use_ema is '
            f'True. Resuming would evaluate and ship a randomly-initialized EMA copy. '
            f'Start a new run directory, or set use_ema: False to continue this one.')

    def _close_worker_pools(self, env_runner=None):
        """Release the subprocess pools held by the policy's verifier and the env runner.

        Both the search verifier and the env runner hold pools of worker SUBPROCESSES that
        garbage collection will not reap, so they must be closed explicitly -- register this
        on an ExitStack so it runs on normal exit and on exceptions alike.

        Best-effort: this runs during teardown (including after an exception), so a failure
        to close must not mask the original error.
        """
        owners = [getattr(self, 'model', None)]
        accelerator = getattr(self, 'accelerator', None)
        if accelerator is not None and owners[0] is not None:
            owners[0] = accelerator.unwrap_model(owners[0])
        if env_runner is not None:
            owners.append(env_runner)
        for owner in owners:
            if owner is None:
                continue
            close = getattr(owner, 'close', None)
            if close is None:
                continue
            try:
                close()
            except Exception as e:
                print(f'warning: failed to close {type(owner).__name__}: {e}')


    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
