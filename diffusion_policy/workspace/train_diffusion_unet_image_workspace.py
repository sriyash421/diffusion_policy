if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import pickle
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import (
    DiffusionUnetImagePolicy)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch', 'last_checkpoint_step']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = (
            hydra.utils.instantiate(cfg.policy))

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.last_checkpoint_step = 0  # Track last checkpoint step

        # accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            # mixed_precision=cfg.training.mixed_precision,
            kwargs_handlers=[ddp_kwargs]
        )
        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        checkpoint_loaded = False
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                checkpoint_loaded = True

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # The RESUMED normalizer wins, as in TrainMLPImageWorkspace. This used to refit from
        # the dataset unconditionally, after load_checkpoint had already restored the saved
        # statistics -- so a resume silently re-derived them. That is a no-op while the data
        # is fixed (get_normalizer(mode='limits') is deterministic over a manifest-pinned
        # train set, so the refit reproduces the stored values), and a silent corruption the
        # moment the split or the zarr changes underneath a `resume: True` run: the action
        # and obs scales would shift mid-run with nothing reporting it, while every
        # checkpoint before the change had been trained against the old ones.
        if checkpoint_loaded and len(self.model.normalizer.params_dict) > 0:
            print("Checkpoint loaded with normalizer - preserving existing normalizer statistics")
            normalizer = self.model.normalizer
        else:
            print("Computing normalizer from dataset")
            normalizer = dataset.get_normalizer()
            self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            # the EMA copy is what gets rolled out, validated and shipped in the checkpoint;
            # without this it would carry empty normalizer stats
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        # Forward training.lr_scheduler_kwargs, as TrainMLPImageWorkspace does. Without
        # this the custom `decay_then_constant` schedule silently falls back to its
        # defaults (decay_steps=10000, min_lr_ratio=0.1) no matter what the config says --
        # no error, just the wrong curve. A 100k run configured to decay over 77k would
        # instead hit its floor by 13k and sit there for the remaining 87k.
        lr_scheduler_kwargs = dict(cfg.training.get('lr_scheduler_kwargs', {}) or {})
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1,
            **lr_scheduler_kwargs
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        if self.accelerator.is_main_process:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                },
                allow_val_change=True
            )

        # configure checkpoint
        # accelerator prepare
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = self.accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        device = self.accelerator.device
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        # when set, rollouts are triggered on gradient steps instead of epochs
        rollout_every_steps = cfg.training.get('rollout_every_steps', None)

        # Hard stop in GRADIENT STEPS, as in TrainMLPImageWorkspace. This workspace used to
        # have no step bound at all, so `num_epochs` was the only thing ending a run -- and
        # since an epoch is `ceil(n_windows / batch_size)` steps, the run length had to be
        # recomputed by hand from the demo budget and the batch size every time either
        # moved. With this set, `num_epochs` is only a safety bound.
        max_gradient_steps = cfg.training.get('max_gradient_steps', None)
        # Enforced MID-EPOCH (see the break below), not just at epoch boundaries: an
        # epoch-granular check overshoots by up to a full epoch and, whenever the last epoch
        # begins below the cap, never fires at all.
        stop_training = False

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.accelerator.unwrap_model(self.model).compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        self.accelerator.backward(loss)
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            # Honors training.gradient_clip_norm when set, as
                            # TrainMLPImageWorkspace does. Previously this workspace had no
                            # clip call at all, so the key was silently dead here -- a
                            # config could ask for clipping and simply not get it. Left
                            # UNSET in the pusht UNet configs, so behaviour is unchanged.
                            if cfg.training.get('gradient_clip_norm', None) is not None:
                                self.accelerator.clip_grad_norm_(
                                    self.model.parameters(), cfg.training.gradient_clip_norm)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        if cfg.training.use_ema:
                            ema.step(self.accelerator.unwrap_model(self.model))
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            if self.accelerator.is_main_process:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            self.global_step += 1

                            # checkpoint based on gradient steps - check right after step increment
                            next_checkpoint_step = self.last_checkpoint_step + cfg.training.checkpoint_every
                            if self.global_step >= next_checkpoint_step and self.accelerator.is_main_process:
                                print(f"Saving checkpoint at step {self.global_step} (target was {next_checkpoint_step})")
                                model_ddp = self.model
                                self.model = self.accelerator.unwrap_model(self.model)
                                if cfg.checkpoint.save_last_ckpt:
                                    self.save_checkpoint()
                                if cfg.checkpoint.save_last_snapshot:
                                    self.save_snapshot()

                                # Save a per-step checkpoint. These are ~4.6GB each, so
                                # save_step_ckpt=False keeps only the rolling latest.ckpt.
                                if cfg.checkpoint.get('save_step_ckpt', True):
                                    step_ckpt_path = os.path.join(self.output_dir, 'checkpoints', f'step_{self.global_step:07d}.ckpt')
                                    os.makedirs(os.path.dirname(step_ckpt_path), exist_ok=True)
                                    self.save_checkpoint(path=step_ckpt_path)
                                self.model = model_ddp

                                # Update last checkpoint step
                                self.last_checkpoint_step = self.global_step

                            # rollout eval on a gradient-step schedule, so evals land
                            # on exact step multiples rather than epoch boundaries
                            if (rollout_every_steps is not None) \
                                and (self.global_step % rollout_every_steps == 0):
                                eval_policy = self.accelerator.unwrap_model(self.model)
                                if cfg.training.use_ema:
                                    eval_policy = self.ema_model
                                eval_policy.eval()
                                eval_log = dict(env_runner.run(eval_policy))
                                if self.accelerator.is_main_process:
                                    wandb_run.log(eval_log, step=self.global_step)
                                    json_logger.log({**eval_log,
                                        'global_step': self.global_step,
                                        'epoch': self.epoch})
                                self.model.train()

                        # Stop on the exact step. Placed AFTER the checkpoint and rollout
                        # blocks above so the final step is still saved, and it leaves the
                        # end-of-epoch block below to run once on the truncated epoch -- so
                        # a capped run still ends with a val_loss and a topk checkpoint.
                        if max_gradient_steps is not None \
                            and self.global_step >= max_gradient_steps:
                            stop_training = True
                            break

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss
                policy = self.accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout (epoch schedule; null when rollout_every_steps is used)
                if (cfg.training.rollout_every is not None) \
                    and (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.accelerator.unwrap_model(self.model).compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']

                        # Score only the EXECUTED window, matching TrainMLPImageWorkspace.
                        # This used to compare the full `horizon` (16) prediction against the
                        # full ground truth, while the search workspace sliced both sides to
                        # [n_obs_steps-1 : n_obs_steps+n_action_steps-1] -- the 8 steps
                        # `predict_action` actually returns for execution. The other 8 are
                        # discarded at the next re-plan and are the least constrained part of
                        # the prediction, so including them inflated this number relative to
                        # the search arms and made the two workspaces' `train_action_mse_*`
                        # series not comparable. A config difference would have been visible;
                        # this was only in the logging code.
                        To = cfg.n_obs_steps
                        Ta = cfg.n_action_steps
                        asl = slice(To - 1, To + Ta - 1)
                        gt_action = batch['action'][:, asl]

                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred'][:, asl]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                if (self.epoch % cfg.training.checkpoint_every) == 0 and self.accelerator.is_main_process:
                    model_ddp = self.model
                    self.model = self.accelerator.unwrap_model(self.model)
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    self.model = model_ddp

                # Restore train mode on the ONLINE model. With use_ema=False `policy` is
                # the online model, so the eval() above would otherwise leave the whole
                # run with dropout/BN disabled from epoch 1 onward. The EMA copy is
                # deliberately left in eval().
                self.accelerator.unwrap_model(self.model).train()

                if self.accelerator.is_main_process:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                # This increment accounts for the epoch's LAST batch, which takes a gradient
                # step but is deliberately not counted inside the loop (`if not
                # is_last_batch`). On a capped stop the epoch was cut short and that batch
                # never ran, so counting it would leave global_step one past the cap.
                if not stop_training:
                    self.global_step += 1
                self.epoch += 1

                # The batch loop already ran its end-of-epoch validation, sampling and topk
                # checkpoint above, so the run ends fully evaluated rather than mid-epoch.
                if stop_training:
                    print(f'Reached max_gradient_steps={max_gradient_steps}, stopping.')
                    break
        # trackers only exist if the Accelerator was built with log_with=; this repo
        # logs to wandb directly, so end_training() would raise on an empty tracker list
        self.join_saving_thread()
        if getattr(self.accelerator, 'trackers', None):
            self.accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
