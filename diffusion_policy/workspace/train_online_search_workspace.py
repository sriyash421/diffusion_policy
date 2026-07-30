if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import contextlib
import os
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
import pathlib
import copy
import wandb
import tqdm

from diffusion_policy.workspace.train_mlp_image_workspace import TrainMLPImageWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.pusht_image_dataset import get_episode_init_states
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.common.sampler import get_collate_fn

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainOnlineSearchWorkspace(TrainMLPImageWorkspace):
    """Outer/inner trainer: roll out on-policy context once for a large pool, then take
    many gradient updates on mini-batches, subsampling which rollouts form the context.
    """

    def _subsample_context(self, policy, pool_rollouts, sel, rng):
        n_trajs = policy.n_trajs
        lo = self.cfg.training.get('min_context_trajs', 0)
        hi = self.cfg.training.get('max_context_trajs', n_trajs)
        rows = []
        for p in sel:
            rlist = pool_rollouts[p]
            # clamp lo as well as hi: an episode that terminated early can yield fewer
            # rollouts than min_context_trajs, and rng.integers raises when low > high.
            hi_p = min(hi, len(rlist))
            lo_p = min(lo, hi_p)
            k = int(rng.integers(lo_p, hi_p + 1)) if len(rlist) else 0
            if k > 0:
                chosen = rng.choice(len(rlist), size=k, replace=False)
                rows.append(torch.cat([rlist[c] for c in chosen], dim=0))
            else:
                # match the dtype/device pad_context_rows allocates with, so the
                # empty-row branch is consistent with the populated one
                rows.append(torch.zeros(0, policy.hidden_dim,
                                        dtype=policy.dtype, device=policy.device))
        return policy.pad_context_rows(rows)

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        checkpoint_loaded = False
        if cfg.training.resume:
            ckpt = self.get_checkpoint_path()
            if ckpt.is_file():
                print(f"Resuming from checkpoint {ckpt}")
                self.load_checkpoint(path=ckpt)
                checkpoint_loaded = True

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) and dataset.return_sequences

        # Same rule as the parent workspace: a resumed checkpoint's normalizer is the one
        # the resumed weights were trained under. Recomputing it here would silently
        # change the input/output scaling mid-run.
        if checkpoint_loaded and len(self.model.normalizer.params_dict) > 0:
            print("Checkpoint loaded with normalizer - preserving existing statistics")
        else:
            self.model.set_normalizer(dataset.get_normalizer())

        init_states_all = torch.as_tensor(
            get_episode_init_states(dataset.replay_buffer, dataset.episode_mask),
            dtype=torch.float32)
        n_traj = len(dataset)
        collate = get_collate_fn()

        env_runner: BaseImageRunner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir)

        outer_bs = min(cfg.training.outer_batch_size, n_traj)
        inner_bs = cfg.training.inner_batch_size
        num_inner = cfg.training.num_inner_steps
        num_outer = cfg.training.num_outer_steps
        rng = np.random.default_rng(cfg.training.seed)

        # must precede the scheduler, which is sized from num_outer * num_inner
        if cfg.training.debug:
            num_outer, num_inner, outer_bs, inner_bs = 2, 3, 4, 2

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=num_outer * num_inner,
            last_epoch=self.global_step - 1)

        self.model, self.optimizer, lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, lr_scheduler)
        device = self.accelerator.device
        optimizer_to(self.optimizer, device)

        if self.accelerator.is_main_process:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging)

        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with contextlib.ExitStack() as stack:
            json_logger = stack.enter_context(JsonLogger(log_path))
            # the env runner and the policy's context vec-env each hold a pool of worker
            # SUBPROCESSES that garbage collection will not reap; release them on normal
            # exit and on exceptions alike.
            stack.callback(self._close_worker_pools, env_runner)
            # resume continues the outer loop where it stopped rather than restarting it,
            # which would run the full num_outer again on every restart.
            for _ in range(self.epoch, num_outer):
                policy = self.accelerator.unwrap_model(self.model)

                policy.eval()
                pool = rng.choice(n_traj, size=outer_bs, replace=outer_bs > n_traj)
                pool_states = init_states_all[pool].to(device)
                pool_rollouts = policy.generate_context_rollouts(
                    pool_states, policy.n_trajs)

                self.model.train()
                inner_losses = list()
                with tqdm.tqdm(range(num_inner), desc="update_step",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as pbar:
                    for _ in pbar:
                        sel = rng.choice(len(pool), size=inner_bs,
                                         replace=inner_bs > len(pool))
                        batch = collate([dataset[int(pool[p])] for p in sel])
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        cf, cm = self._subsample_context(policy, pool_rollouts, sel, rng)

                        raw_loss = policy.compute_loss(
                            batch, context_features=cf, context_mask=cm)
                        self.optimizer.zero_grad()
                        self.accelerator.backward(raw_loss)
                        if cfg.training.get('gradient_clip_norm', None) is not None:
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(), cfg.training.gradient_clip_norm)
                        self.optimizer.step()
                        lr_scheduler.step()

                        loss_cpu = raw_loss.item()
                        inner_losses.append(loss_cpu)
                        pbar.set_postfix(loss=loss_cpu, refresh=False)
                        step_log = {
                            'train_loss': loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0],
                        }
                        if self.accelerator.is_main_process:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                        self.global_step += 1

                step_log = {
                    'train_loss': float(np.mean(inner_losses)),
                    'global_step': self.global_step,
                    'epoch': self.epoch,
                }
                policy.eval()
                if (self.epoch % cfg.training.rollout_every) == 0:
                    step_log.update(env_runner.run(policy))
                if (self.epoch % cfg.training.checkpoint_every) == 0 \
                        and self.accelerator.is_main_process:
                    # swap in the unwrapped module so the checkpoint holds plain module
                    # keys, then restore the prepared one. Re-running accelerator.prepare
                    # here would build a NEW DDP wrapper (on rank 0 only) that no longer
                    # matches the optimizer prepared at the top of run().
                    model_ddp = self.model
                    self.model = policy
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    self.model = model_ddp

                if self.accelerator.is_main_process:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                self.epoch += 1
                # the loop ends in eval() (rollout/checkpoint); restore train mode so the
                # next outer iteration does not train with dropout silently disabled
                self.model.train()

        self.join_saving_thread()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainOnlineSearchWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
