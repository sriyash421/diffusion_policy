if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import contextlib
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

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import (
    DiffusionUnetImagePolicy)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import (
    dict_apply, optimizer_to, trainable_parameters)
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.env.pusht.pusht_verifier import check_verifier_value

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch', 'last_checkpoint_step']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        # The UNet BC arm (train_pusht_unet_bc.yaml) trains through THIS workspace, not
        # TrainMLPImageWorkspace, so without this call a `verifier_value` typo or omission
        # would train against the default t_goal while the run dir said ver-armTn. No-op
        # for the non-PushT configs, which declare no `verifier_tag`.
        check_verifier_value(cfg)

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

        # FREEZE BEFORE the optimizer is built, so the frozen parameters are excluded from it
        # and the trainable_parameters() print below reports the real count. The handler in
        # run()'s epoch loop applies the same two lines, but it runs strictly later -- so on
        # its own it would leave the startup line reading "(0 frozen)" on a genuinely frozen
        # run, which is exactly the signal a freeze ablation is read off. That loop copy is
        # kept: it re-asserts .eval() each epoch, which is needed because self.model.train()
        # is called mid-epoch and nn.Module.train() recurses into children.
        if cfg.training.get('freeze_encoder', False):
            self.model.obs_encoder.eval()
            self.model.obs_encoder.requires_grad_(False)

        # configure training state
        # Frozen parameters are excluded, not merely skipped by AdamW: the SD VAE obs
        # backbone is 34.2M frozen parameters and there is no reason for them to sit in the
        # optimizer's state_dict or under its weight_decay.
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer,
            params=trainable_parameters(self.model, type(self).__name__))

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
                payload = self.load_checkpoint(path=lastest_ckpt_path)
                checkpoint_loaded = True
                # Refuse a pre-EMA payload resumed into a use_ema run. This workspace ships
                # the EMA copy and rolls it out, so without this a resume from a pre-EMA
                # checkpoint would evaluate a randomly-initialized average. See
                # BaseWorkspace.assert_ema_payload.
                self.assert_ema_payload(payload, lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        # Record the exact episodes this run trains/validates/tests on, and refuse to resume
        # if they differ from what the run directory already recorded. TrainMLPImageWorkspace
        # and TrainSearchOuterInnerWorkspace both do this; this workspace did not, so its runs
        # carried no splits.json at all.
        #
        # That is not only a provenance gap: eval_search_pusht.get_split_states cross-checks
        # its resolved indices against <run_dir>/splits.json, but only `if recorded.is_file()`
        # -- so a missing file silently SKIPS the check, and every UNet checkpoint was scored
        # with no guard against being evaluated on a different partition than it trained on.
        #
        # Must follow the instantiate above: the guard needs the resolved partition, not the
        # config keys. write_splits is a no-op for datasets without get_split_indices.
        if self.accelerator.is_main_process:
            self.write_manifest()
            self.write_splits(dataset)

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
            # optimizer steps, not batches, and ceil-then-multiply rather than
            # multiply-then-floor: the epoch's final partial accumulation window is still
            # flushed, so there are ceil(len/accum) steps per epoch. Floor-dividing left the
            # schedule short by up to one step per epoch. Inert while lr_scheduler is
            # decay_then_constant (which ignores num_training_steps) -- it matters the moment
            # anything switches back to plain `cosine`, whose horizon this is.
            num_training_steps=math.ceil(
                len(train_dataloader) / cfg.training.gradient_accumulate_every
                ) * cfg.training.num_epochs,
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
            # EMAModel exposes no state_dict, so save_checkpoint never persists this counter
            # (it only serializes attributes having state_dict AND load_state_dict). Left at
            # 0 on resume, get_decay returns 0.0 for step <= 0 and the update degenerates to
            # ema_param.copy_(param) -- silently destroying the restored average on the first
            # post-resume steps and shipping the live weights as the EMA copy. Both other
            # workspaces already do this; this one did not.
            ema.optimization_step = self.global_step

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
        with contextlib.ExitStack() as stack:
            json_logger = stack.enter_context(JsonLogger(log_path))
            # The env runner holds a pool of worker SUBPROCESSES which garbage collection
            # will not reap. Register its shutdown so it is released on normal exit and on
            # exceptions alike, as the other two workspaces already do.
            stack.callback(self._close_worker_pools, env_runner)
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                # `.get`, because the PushT configs no longer declare the key: their obs
                # backbone (SDVAEEncoder) freezes itself unconditionally. Kept for
                # train_diffusion_unet_{image,real}_pretrained_workspace, which set it True.
                # unwrap first: accelerate's DDP wrapper does not forward attribute access,
                # so `self.model.obs_encoder` would raise once prepare() wraps the model.
                if cfg.training.get('freeze_encoder', False):
                    self.accelerator.unwrap_model(self.model).obs_encoder.eval()
                    self.accelerator.unwrap_model(self.model).obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # The crop offsets are a pure function of (seed, global_step), so
                        # they must be told which step this is -- without this call they stay
                        # frozen at step 0 and every batch gets the identical crop, which is
                        # worse than no augmentation. `hasattr` because this workspace also
                        # serves the upstream UNet configs, whose policies have no crop scope.
                        if hasattr(self.accelerator.unwrap_model(self.model), 'set_crop_step'):
                            self.accelerator.unwrap_model(self.model).set_crop_step(
                                cfg.training.seed, self.global_step)
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        # Refreshed every batch, NOT frozen at epoch 0. The old
                        # `if train_sampling_batch is None` guard pinned the FIRST batch of
                        # the FIRST epoch and reused that same GPU tensor for every
                        # sample_every evaluation for the rest of the run -- so
                        # train_action_mse_error measured fit on 32 windows the model re-saw
                        # every epoch, not anything that moved with training. Matching
                        # TrainMLPImageWorkspace here is what actually makes the two
                        # workspaces' series comparable; b487ec0 aligned only the slicing.
                        train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.accelerator.unwrap_model(self.model).compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        self.accelerator.backward(loss)
                        # Accumulation windows are delimited by the BATCH INDEX WITHIN THE
                        # EPOCH, which is the only thing that can delimit them. global_step
                        # is a running counter that deliberately skips each epoch's last
                        # batch (see `if not is_last_batch` below), so windows keyed on it
                        # drift out of phase with the epoch by one every epoch. The epoch's
                        # final partial window is flushed here, so a partial window's
                        # gradients never leak into the next epoch. Identical to the old
                        # condition at gradient_accumulate_every: 1, which every PushT
                        # config sets.
                        if ((batch_idx + 1) % cfg.training.gradient_accumulate_every == 0) \
                            or (batch_idx == len(train_dataloader) - 1):
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
                                # Update the cadence anchor BEFORE writing, so the value that
                                # lands in the checkpoint is the step this checkpoint was
                                # taken at. Setting it afterwards persists the PREVIOUS
                                # anchor, so a resumed run believes it is overdue and fires
                                # an extra checkpoint on its very first step. (Fixed in
                                # TrainSearchOuterInnerWorkspace._save; back-ported here.)
                                self.last_checkpoint_step = self.global_step
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

                            # rollout eval on a gradient-step schedule, so evals land
                            # on exact step multiples rather than epoch boundaries
                            if (rollout_every_steps is not None) \
                                and (self.global_step % rollout_every_steps == 0):
                                eval_policy = self.accelerator.unwrap_model(self.model)
                                if cfg.training.use_ema:
                                    eval_policy = self.ema_model
                                eval_policy.eval()
                                # Seed before the rollout so its success rate is reproducible
                                # from the checkpoint: the env is deterministic given a reset
                                # state, but conditional_sample draws from the global RNG.
                                torch.manual_seed(cfg.training.seed)
                                np.random.seed(cfg.training.seed)
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
                        # a capped run still ends with a val_loss and a final rollout.
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
                    # seeded for the same reason as the step-cadence rollout above
                    torch.manual_seed(cfg.training.seed)
                    np.random.seed(cfg.training.seed)
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

                # The batch loop already ran its end-of-epoch validation and sampling
                # above, so the run ends fully evaluated rather than mid-epoch.
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
