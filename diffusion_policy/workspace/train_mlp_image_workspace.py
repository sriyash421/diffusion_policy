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
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from diffusion_policy.workspace.base_workspace import BaseWorkspace, set_initial_lr
from diffusion_policy.policy.mlp_image_policy import MLPImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler

from diffusion_policy.common.sampler import get_collate_fn

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _is_search_policy(policy) -> bool:
    """Whether this policy exposes the best-of-n search interface.

    One explicit capability check, in place of the ad-hoc `hasattr(policy, 'verifier')`
    probes used elsewhere in this file. Search policies add `search_candidates` /
    `predict_n_actions` on top of the standard `predict_action` contract; everything else
    is driven through `predict_action` alone.
    """
    return hasattr(policy, 'search_candidates') and hasattr(policy, 'verifier')


class TrainMLPImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch', 'last_checkpoint_step']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: MLPImagePolicy = (
            hydra.utils.instantiate(cfg.policy))

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
            kwargs_handlers=[ddp_kwargs],
        )
        # NOTE: the optimizer is ALWAYS persisted. This used to be dropped when
        # cfg.training.resume was False, but `resume` is a property of the LAUNCH, not of
        # the checkpoint: relaunching that same output dir with resume=True then found no
        # optimizer state and died with KeyError: 'initial_lr' while building the
        # scheduler. A checkpoint should always be resumable; see set_initial_lr for the
        # backfill that lets already-written optimizer-less checkpoints load too.

    def _close_worker_pools(self, env_runner):
        """Release any subprocess pools held by the policy's verifier and the env runner.

        Best-effort: this runs during teardown (including after an exception), so a
        failure to close must not mask the original error.
        """
        for owner in (self.accelerator.unwrap_model(self.model), env_runner):
            close = getattr(owner, 'close', None)
            if close is None:
                continue
            try:
                close()
            except Exception as e:
                print(f'warning: failed to close {type(owner).__name__}: {e}')

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

        # Pretrained weights are the INITIALIZATION for a fresh finetune run only. If we
        # just resumed, the resumed weights are strictly newer -- loading the pretrained
        # ones on top would discard all progress while KEEPING the resumed optimizer
        # state and global_step, so the run continues as though nothing happened. Every
        # search config sets resume: True, so this fired on every restart.
        if cfg.checkpoint.get('pretrained_ckpt_path', None) is not None \
                and not checkpoint_loaded:
            pretrained_ckpt_path = pathlib.Path(
                hydra.utils.to_absolute_path(cfg.checkpoint.pretrained_ckpt_path))
            if pretrained_ckpt_path.is_file():
                print(f"Loading pretrained weights from {pretrained_ckpt_path}")
                # include_keys=[] because it filters payload['pickles'], NOT state_dicts:
                # ['model'] never selected the weights (those come from state_dicts and
                # are governed by exclude_keys), it only let the pretrained run's
                # global_step / epoch / last_checkpoint_step leak into this fresh run.
                self.load_checkpoint(path=pretrained_ckpt_path,
                    exclude_keys=['optimizer'],
                    include_keys=[]
                )
            else:
                # print(f"Pretrained checkpoint path {pretrained_ckpt_path} not found - skipping")
                raise FileNotFoundError(
                    f"Pretrained checkpoint path {pretrained_ckpt_path} not found"
                )

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        # Use weighted sampler if available (for multi-dataset with sampling ratios)
        dataloader_kwargs = dict(cfg.dataloader)
        if hasattr(dataset, 'weighted_sampler') and dataset.weighted_sampler is not None:
            dataloader_kwargs['sampler'] = dataset.weighted_sampler
            dataloader_kwargs.pop('shuffle', None)  # Remove shuffle when using custom sampler

        collate_fn = get_collate_fn() if dataset.return_sequences else None
        train_dataloader = DataLoader(dataset, collate_fn=collate_fn, **dataloader_kwargs)

        # Only recompute normalizer if checkpoint wasn't loaded or normalizer is empty
        if checkpoint_loaded and len(self.model.normalizer.params_dict) > 0:
            print("Checkpoint loaded with normalizer - preserving existing normalizer statistics")
            normalizer = self.model.normalizer
        else:
            print("Computing normalizer from dataset")
            normalizer = dataset.get_normalizer()
            self.model.set_normalizer(normalizer)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, **cfg.val_dataloader)

        # configure lr scheduler
        # By default the LR decays over the whole run (steps_per_epoch * num_epochs).
        # Set training.lr_decay_steps to decouple the decay horizon from the run
        # length: e.g. with the `polynomial` scheduler the LR decays to lr_end over
        # lr_decay_steps and then HOLDS constant at lr_end for the remaining steps.
        lr_decay_steps = cfg.training.get('lr_decay_steps', None)
        if lr_decay_steps is None:
            # Optimizer steps, not batches. The epoch's final partial accumulation window
            # is always flushed, so there are ceil(len/accum) steps per epoch -- floor
            # division here left the schedule short by up to one step per epoch, and a
            # cosine schedule then never reached its floor.
            lr_decay_steps = math.ceil(
                len(train_dataloader) / cfg.training.gradient_accumulate_every
                ) * cfg.training.num_epochs
        _lsk = cfg.training.get('lr_scheduler_kwargs', None)
        lr_scheduler_kwargs = (OmegaConf.to_container(_lsk, resolve=True)
                               if _lsk is not None else {})
        # Building at last_epoch >= 0 requires initial_lr in every param group. It is
        # normally restored with the optimizer state; backfill it so a checkpoint written
        # before optimizer state was persisted still resumes instead of raising KeyError.
        set_initial_lr(self.optimizer)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=lr_decay_steps,
            last_epoch=self.global_step-1,
            **lr_scheduler_kwargs
        )

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
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None
        val_sampling_batch = None

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
                if cfg.training.get('max_gradient_steps', None) is not None:
                    if self.global_step >= cfg.training.max_gradient_steps:
                        break
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.accelerator.unwrap_model(self.model).obs_encoder.eval()
                    self.accelerator.unwrap_model(self.model).obs_encoder.requires_grad_(False)

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

                        # Accumulation windows are counted in BATCHES. Keying them on
                        # global_step (which counts optimizer steps) meant a counter could
                        # not delimit its own windows: the window shifted by one every
                        # epoch, the very first batch stepped on a single micro-batch, and
                        # each epoch's final partial window was dropped with its gradients
                        # leaking into the next epoch (zero_grad only ran inside the
                        # branch). Inert while every config sets accumulate=1, but silently
                        # wrong the moment one does not.
                        is_last_batch = (batch_idx == (n_batches - 1)) or (
                            max_train_steps is not None
                            and batch_idx >= (max_train_steps - 1))
                        is_optimizer_step = \
                            ((batch_idx + 1) % accumulate_every == 0) or is_last_batch
                        if is_optimizer_step:
                            if cfg.training.get('gradient_clip_norm', None) is not None:
                                self.accelerator.clip_grad_norm_(self.model.parameters(), cfg.training.gradient_clip_norm)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            # global_step is exactly the number of optimizer steps taken.
                            # That is what checkpoint_every, max_gradient_steps and the
                            # step_*.ckpt names (which the eval/BoN scripts parse) mean.
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

                        # checkpoint on the gradient-step schedule
                        if is_optimizer_step:
                            next_checkpoint_step = self.last_checkpoint_step + cfg.training.checkpoint_every
                            if self.global_step >= next_checkpoint_step and self.accelerator.is_main_process:
                                print(f"Saving checkpoint at step {self.global_step} (target was {next_checkpoint_step})")
                                # Anchor BEFORE writing, so the value that lands inside the
                                # checkpoint is the step it was taken at; setting it after
                                # persists the previous anchor and a resumed run believes
                                # it is overdue and fires an extra checkpoint immediately.
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

                        if max_train_steps is not None \
                            and batch_idx >= (max_train_steps - 1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss
                policy = self.accelerator.unwrap_model(self.model)
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
                                if val_sampling_batch is None:
                                    val_sampling_batch = batch      # first val batch -> val action-MSE sampling
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

                # run sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        # Copy, never mutate: train_sampling_batch is held on GPU for the
                        # whole run, so an attention_mask inserted into its obs dict below
                        # would stick there forever and be fed to every later epoch.
                        obs_dict = dict(batch['obs'])
                        gt_action = batch['action'][
                            :, policy.n_obs_steps-1:policy.n_obs_steps+policy.n_action_steps-1
                        ]
                        if batch.get('attention_mask', None) is not None:
                            obs_dict['attention_mask'] = batch['attention_mask']
                        
                        if hasattr(policy, 'verifier'):
                            # Include the expert action so ground-truth verifiers (e.g. MSEVerifier) can read it.
                            result = policy.search_candidates({**obs_dict, 'action': batch['action']}, verifier=policy.verifier, n_actions=policy.max_actions)
                            pred_action, values = result
                            pred_action = pred_action[:, :, policy.n_obs_steps-1:policy.n_obs_steps+policy.n_action_steps-1]
                            # UNNORMALIZED: raw action units (mixed m / rad / gripper[0,1]).
                            mse = (pred_action - gt_action.unsqueeze(1)).pow(2).mean(dim=(-1, -2)) # B, max_actions
                            min_mse, _ = torch.min(mse, dim=-1) # B
                            avg_mse = mse.mean(dim=-1) # B
                            # NORMALIZED: per-dim [-1,1] space (the space the loss is computed in).
                            na = policy.normalizer['action']
                            mse_n = (na.normalize(pred_action) - na.normalize(gt_action).unsqueeze(1)).pow(2).mean(dim=(-1, -2))
                            min_n, _ = torch.min(mse_n, dim=-1) # B
                            avg_n = mse_n.mean(dim=-1) # B
                            step_log['train_action_mse_error_min'] = min_mse.mean().item()
                            step_log['train_action_mse_error_avg'] = avg_mse.mean().item()
                            step_log['train_action_mse_error_min_unnormalized'] = min_mse.mean().item()
                            step_log['train_action_mse_error_avg_unnormalized'] = avg_mse.mean().item()
                            step_log['train_action_mse_error_min_normalized'] = min_n.mean().item()
                            step_log['train_action_mse_error_avg_normalized'] = avg_n.mean().item()
                            # RMSE = sqrt(per-sample MSE), then mean over batch.
                            step_log['train_action_rmse_error_min_normalized'] = min_n.sqrt().mean().item()
                            step_log['train_action_rmse_error_avg_normalized'] = avg_n.sqrt().mean().item()
                            step_log['train_action_rmse_error_min_unnormalized'] = min_mse.sqrt().mean().item()
                            step_log['train_action_rmse_error_avg_unnormalized'] = avg_mse.sqrt().mean().item()
                            step_log['train_action_value'] = values.mean().item()
                        else:
                            result = policy.predict_action(obs_dict)
                            pred_action = result['action']
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # run sampling on a VALIDATION batch -> val action MSE (mirrors the
                # train_action_mse_error above, but on held-out data). This is the
                # action-space error (sampled action vs expert), complementing
                # val_loss which is only the denoising/epsilon MSE.
                if (self.epoch % cfg.training.sample_every) == 0 and val_sampling_batch is not None:
                    with torch.no_grad():
                        batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        # copy, never mutate -- see the train sampling block above
                        obs_dict = dict(batch['obs'])
                        gt_action = batch['action'][
                            :, policy.n_obs_steps-1:policy.n_obs_steps+policy.n_action_steps-1
                        ]
                        if batch.get('attention_mask', None) is not None:
                            obs_dict['attention_mask'] = batch['attention_mask']

                        if hasattr(policy, 'verifier'):
                            result = policy.search_candidates({**obs_dict, 'action': batch['action']}, verifier=policy.verifier, n_actions=policy.max_actions)
                            pred_action, values = result
                            pred_action = pred_action[:, :, policy.n_obs_steps-1:policy.n_obs_steps+policy.n_action_steps-1]
                            # UNNORMALIZED: raw action units (mixed m / rad / gripper[0,1]).
                            mse = (pred_action - gt_action.unsqueeze(1)).pow(2).mean(dim=(-1, -2)) # B, max_actions
                            min_mse, _ = torch.min(mse, dim=-1) # B
                            avg_mse = mse.mean(dim=-1) # B
                            # NORMALIZED: per-dim [-1,1] space (the space the loss is computed in).
                            na = policy.normalizer['action']
                            mse_n = (na.normalize(pred_action) - na.normalize(gt_action).unsqueeze(1)).pow(2).mean(dim=(-1, -2))
                            min_n, _ = torch.min(mse_n, dim=-1) # B
                            avg_n = mse_n.mean(dim=-1) # B
                            step_log['val_action_mse_error_min'] = min_mse.mean().item()
                            step_log['val_action_mse_error_avg'] = avg_mse.mean().item()
                            step_log['val_action_mse_error_min_unnormalized'] = min_mse.mean().item()
                            step_log['val_action_mse_error_avg_unnormalized'] = avg_mse.mean().item()
                            step_log['val_action_mse_error_min_normalized'] = min_n.mean().item()
                            step_log['val_action_mse_error_avg_normalized'] = avg_n.mean().item()
                            # RMSE = sqrt(per-sample MSE), then mean over batch.
                            step_log['val_action_rmse_error_min_normalized'] = min_n.sqrt().mean().item()
                            step_log['val_action_rmse_error_avg_normalized'] = avg_n.sqrt().mean().item()
                            step_log['val_action_rmse_error_min_unnormalized'] = min_mse.sqrt().mean().item()
                            step_log['val_action_rmse_error_avg_unnormalized'] = avg_mse.sqrt().mean().item()
                            step_log['val_action_value'] = values.mean().item()
                        else:
                            result = policy.predict_action(obs_dict)
                            pred_action = result['action']
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            step_log['val_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # ========= eval end for this epoch ==========
                # policy.eval() above disables dropout for the rollout/val/sample blocks;
                # without this the model would train in eval mode for every epoch after
                # the first (i.e. with p_drop_attn silently zeroed).
                policy.train()

                if self.accelerator.is_main_process:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                # NOTE: global_step is deliberately NOT incremented here. It counts
                # optimizer steps and is advanced only in the training loop; incrementing
                # it per epoch as well drifted it +1/epoch away from the true step count,
                # and from the step_*.ckpt names the eval/BoN scripts parse.
                self.epoch += 1
        # A threaded checkpoint save may still be in flight; without this the process can
        # exit and leave the final checkpoint truncated.
        self.join_saving_thread()
        # Guard: end_training() iterates self.accelerator.trackers, which is
        # only set inside init_trackers(); we never call it (no log_with=),
        # so skip cleanly on accelerate versions that don't lazy-init it.
        if hasattr(self.accelerator, 'trackers'):
            self.accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainMLPImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main() 
