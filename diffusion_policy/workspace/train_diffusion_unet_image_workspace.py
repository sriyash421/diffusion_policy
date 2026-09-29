if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import math
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

from diffusion_policy.workspace.base_workspace import BaseWorkspace, set_initial_lr
from diffusion_policy.policy.diffusion_unet_image_policy import (
    DiffusionUnetImagePolicy)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import (
    get_scheduler, get_cosine_then_constant_schedule)

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
        # NOTE: the optimizer is ALWAYS persisted. This used to be dropped when
        # cfg.training.resume was False, but `resume` is a property of the LAUNCH, not of
        # the checkpoint: relaunching that same output dir with resume=True then found no
        # optimizer state and died with KeyError: 'initial_lr' while building the
        # scheduler. A checkpoint should always be resumable; see set_initial_lr for the
        # backfill that lets already-written optimizer-less checkpoints load too.

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
            else:
                # hydra.run.dir is timestamped in every config here, so a plain re-submit
                # mints a NEW output dir and looks for latest.ckpt somewhere that cannot
                # contain it -- silently starting over despite resume: True. Say so.
                print(f"training.resume=True but no checkpoint found at "
                      f"{lastest_ckpt_path} -- STARTING FROM SCRATCH. To continue a "
                      f"previous run, relaunch with hydra.run.dir=<that run's output dir>.")

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # A resumed checkpoint's normalizer is the one its weights were trained under.
        # Re-fitting it here (which this workspace did unconditionally, AFTER
        # load_checkpoint) silently changes the input/output scaling mid-run -- and the
        # statistics do move, because the dataset's split is derived at load time and
        # discover_zarr_files() picks up whatever is in dataset_dir today. Matches the
        # preserve-on-resume rule train_mlp_image_workspace already follows.
        if checkpoint_loaded and len(self.model.normalizer.params_dict) > 0:
            print("Checkpoint loaded with normalizer - preserving existing normalizer statistics")
        else:
            print("Computing normalizer from dataset")
            normalizer = dataset.get_normalizer()
            self.model.set_normalizer(normalizer)
            if cfg.training.use_ema:
                self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        # last_epoch=global_step-1 resumes the (step-indexed) schedule exactly.
        # Building at last_epoch >= 0 requires initial_lr in every param group. It is
        # normally restored with the optimizer state; backfill it so a checkpoint written
        # before optimizer state was persisted still resumes instead of raising KeyError.
        set_initial_lr(self.optimizer)
        if cfg.training.lr_scheduler == 'cosine_then_constant':
            # Cosine decay to a small floor by lr_decay_steps, then constant.
            # Decouples LR shape from epoch count, so extending a run (raising
            # max_gradient_steps) keeps the post-decay LR flat.
            lr_scheduler = get_cosine_then_constant_schedule(
                optimizer=self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                num_decay_steps=cfg.training.lr_decay_steps,
                min_lr_ratio=cfg.training.get('lr_min_ratio', 0.1),
                last_epoch=self.global_step-1
            )
        else:
            lr_scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                # Optimizer steps, not batches. The epoch's final partial accumulation
                # window is always flushed, so there are ceil(len/accum) steps per epoch;
                # floor division here left the schedule short by up to one step per epoch
                # and a cosine schedule then never reached its floor.
                num_training_steps=math.ceil(
                    len(train_dataloader) / cfg.training.gradient_accumulate_every
                    ) * cfg.training.num_epochs,
                # pytorch assumes stepping LRScheduler every epoch
                # however huggingface diffusers steps it every batch
                last_epoch=self.global_step-1
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

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        # Optional hard cap on total gradient steps. When set, training stops as
        # soon as global_step reaches it (otherwise the loop runs num_epochs).
        # With resume=True + a fixed output dir, raising this and re-running
        # continues a finished run from latest.ckpt.
        max_grad_steps = cfg.training.get('max_gradient_steps', None)
        should_stop = False
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                if max_grad_steps is not None and self.global_step >= max_grad_steps:
                    break
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                n_batches = len(train_dataloader)
                accumulate_every = cfg.training.gradient_accumulate_every
                max_train_steps = cfg.training.max_train_steps
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.accelerator.unwrap_model(self.model).compute_loss(batch)
                        loss = raw_loss / accumulate_every
                        self.accelerator.backward(loss)

                        # Accumulation windows are counted in BATCHES; keying them on
                        # global_step (which counts optimizer steps) meant the counter
                        # could not delimit its own windows, and each epoch's final
                        # partial window was dropped with its gradients leaking into the
                        # next epoch. Inert while every config sets accumulate=1.
                        is_last_batch = (batch_idx == (n_batches - 1)) or (
                            max_train_steps is not None
                            and batch_idx >= (max_train_steps - 1))
                        is_optimizer_step = \
                            ((batch_idx + 1) % accumulate_every == 0) or is_last_batch
                        if is_optimizer_step:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            # EMA tracks OPTIMIZER steps, not batches -- otherwise its
                            # decay schedule advances faster than the weights do.
                            if cfg.training.use_ema:
                                ema.step(self.accelerator.unwrap_model(self.model))
                            # global_step is exactly the number of optimizer steps taken.
                            # That is what checkpoint_every, max_gradient_steps and the
                            # step_*.ckpt names (parsed by the eval/BoN scripts) mean.
                            self.global_step += 1

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        # The last batch's slot is left for the end-of-epoch summary log
                        # at the same global_step, so wandb steps stay strictly increasing.
                        if is_optimizer_step and not is_last_batch:
                            if self.accelerator.is_main_process:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)

                        if is_optimizer_step:
                            # checkpoint on the gradient-step schedule
                            next_checkpoint_step = self.last_checkpoint_step + cfg.training.checkpoint_every
                            if self.global_step >= next_checkpoint_step and self.accelerator.is_main_process:
                                print(f"Saving checkpoint at step {self.global_step} (target was {next_checkpoint_step})")
                                # Anchor BEFORE writing, so the value inside the checkpoint
                                # is the step it was taken at; setting it after persists
                                # the previous anchor and a resumed run fires an extra
                                # checkpoint on its first step.
                                self.last_checkpoint_step = self.global_step
                                model_ddp = self.model
                                self.model = self.accelerator.unwrap_model(self.model)
                                if cfg.checkpoint.save_last_ckpt:
                                    self.save_checkpoint()
                                if cfg.checkpoint.save_last_snapshot:
                                    self.save_snapshot()

                                # Save checkpoint for this step
                                step_ckpt_path = os.path.join(self.output_dir, 'checkpoints', f'step_{self.global_step:07d}.ckpt')
                                os.makedirs(os.path.dirname(step_ckpt_path), exist_ok=True)
                                self.save_checkpoint(path=step_ckpt_path)
                                self.model = model_ddp

                            # Hard stop at the gradient-step cap (if set), after
                            # this step's checkpoint has been written.
                            if max_grad_steps is not None \
                                    and self.global_step >= max_grad_steps:
                                should_stop = True
                                break

                        if max_train_steps is not None \
                            and batch_idx >= (max_train_steps - 1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss
                policy = self.accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
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
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
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
                if self.accelerator.is_main_process:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                # NOTE: global_step is deliberately NOT incremented here. It counts
                # optimizer steps and is advanced only in the training loop; incrementing
                # it per epoch as well drifted it +1/epoch away from the true step count,
                # and from the step_*.ckpt names the eval/BoN scripts parse.
                self.epoch += 1
                if should_stop:
                    break

        # Persist a final latest.ckpt synchronously so a step-capped run is
        # resumable from its exact end (the periodic cadence may not land on the
        # final step). Join any in-flight async save first to avoid racing the
        # same file.
        if self.accelerator.is_main_process:
            self.join_saving_thread()
            model_ddp = self.model
            self.model = self.accelerator.unwrap_model(self.model)
            self.save_checkpoint(use_thread=False)
            self.model = model_ddp
        # end_training() can raise AttributeError('trackers') on some accelerate
        # versions because this workspace logs to wandb manually rather than via
        # accelerate's tracker integration. Training + checkpoints are already
        # complete by here, so guard the teardown to avoid a non-zero exit that
        # mislabels a finished run as FAILED.
        try:
            self.accelerator.end_training()
        except Exception as e:
            print(f"[train] accelerator.end_training() teardown skipped: {e!r}")

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
