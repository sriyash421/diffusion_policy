from typing import Optional
import os
import pathlib
import hydra
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import dill
import torch
import threading


def clone_policy(policy):
    """Deep-copy a policy WITHOUT duplicating its verifier.

    ``policy.verifier`` owns heavyweight external state that must not be duplicated --
    ``DQCVerifier`` holds a live flax/JAX agent plus its loaded checkpoint, and the maze /
    PushT verifiers hold pools of worker subprocesses. ``copy.deepcopy`` would either
    choke on those or silently create a second copy on every clone. The verifier carries
    no policy state (it is handed the obs and the candidate chunk on each call), so every
    clone shares the original one, and only the live model's ``close()`` is ever called.

    Lives here rather than in one workspace because the outer/inner trainer needs it for
    both its EMA copy and its frozen collector snapshot.
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


def set_initial_lr(optimizer):
    """Backfill ``initial_lr`` in every param group, if it is not already there.

    A torch LRScheduler constructed with ``last_epoch >= 0`` (which is how every workspace
    here resumes a step-indexed schedule) REQUIRES ``initial_lr`` in each param group, and
    normally gets it from the optimizer state restored out of the checkpoint. But a
    checkpoint written by a run launched with ``training.resume: False`` contains no
    optimizer state at all, so relaunching that same output dir with ``resume: True``
    used to die with::

        KeyError: param 'initial_lr' is not specified in param_groups[0]
                  when resuming an optimizer

    Seeding it from the group's configured ``lr`` is exactly what the scheduler's own
    ``last_epoch == -1`` path does, so a resumed schedule lands on the same LR it would
    have had. Kept as a backfill (``setdefault``) so a genuinely restored optimizer state
    always wins.
    """
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])


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
        if use_thread:
            # Join the previous save before starting another. Without this, two saves
            # close together (e.g. latest.ckpt then step_*.ckpt on the same step) run
            # concurrently and race on the same file, and the handle was never joined at
            # all -- so a run could exit with its last checkpoint truncated.
            self.join_saving_thread()
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        return str(path.absolute())

    def join_saving_thread(self):
        """Block until any in-flight threaded checkpoint save has finished.

        Call at the end of ``run()``: the process exiting while a save is still writing
        leaves that checkpoint truncated on disk.
        """
        if self._saving_thread is not None:
            self._saving_thread.join()
            self._saving_thread = None

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
