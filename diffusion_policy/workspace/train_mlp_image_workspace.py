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
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.mlp_image_policy import MLPImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler

from diffusion_policy.common.sampler import get_collate_fn

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _subgoal_panel(subgoals, n_rows=2):
    """Tile per-candidate subgoal frames into one wandb.Image, or None.

    ``subgoals`` is the dict returned by a search policy's ``return_subgoals=True``:
    ``image`` (B, n, 3, H, W) in [0, 1] and ``value`` (B, n). Each row is one batch
    element, each column one candidate IN GENERATION ORDER -- so reading left to right
    shows whether conditioning on previous candidates is actually steering the search.
    The caption carries each candidate's verifier value with the executed (argmax) one
    starred, which is the quickest way to see search working (or not) at a glance.
    """
    if not subgoals or subgoals.get('image', None) is None:
        return None
    from torchvision.utils import make_grid

    images = subgoals['image'][:n_rows].float().cpu()          # b, n, 3, H, W
    values = subgoals['value'][:n_rows].float().cpu()          # b, n
    b, n = values.shape
    grid = make_grid(images.reshape(b * n, *images.shape[2:]).clamp(0, 1), nrow=n)
    best = values.argmax(dim=1)
    caption = ' | '.join(
        'row{}: {}'.format(r, ' '.join(
            f'{v:.1f}' + ('*' if c == best[r] else '')
            for c, v in enumerate(values[r].tolist())))
        for r in range(b))
    return wandb.Image(grid, caption=f'subgoals (cols = candidate order); {caption}')


def _is_search_policy(policy) -> bool:
    """Whether this policy exposes the best-of-n search interface.

    One explicit capability check, used everywhere instead of ad-hoc `hasattr(policy,
    'verifier')` / `getattr(policy, 'supports_return_scores')` probes. Search policies add
    `search_candidates` / `predict_action_best` on top of the standard `predict_action`
    contract; everything else is driven through `predict_action` alone.
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
        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def _close_worker_pools(self, env_runner):
        """Release the subprocess pools held by the policy's verifier and the env runner.

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

    def _search_action_nrmse(self, policy, dataloader, device, max_batches=None):
        """Search-quality metrics over a split (search policies only).

        For each window, generate ``max_actions`` candidates and compare each against the
        GT expert action over the action-step window, in the normalizer's normalized
        action space. Returns a dict (or None for non-search policies, i.e. no
        ``.verifier``):

          nrmse_min    best candidate -- how good search *can* get
          nrmse_avg    mean candidate -- the control: did the distribution itself move,
                       or are we just drawing more samples from an unchanged one?
          nrmse_first  candidate 0, generated with an EMPTY search context -- i.e. the
                       no-search baseline. (first - min) is the actual best-of-n gain,
                       which nrmse_min alone cannot show.
          action_value       mean verifier score over candidates.
          action_value_best  score of the candidate search would actually execute
                             (argmax) -- the value-space analogue of nrmse_min.
          action_value_first candidate 0's score, i.e. the no-search baseline.
                             (best - first) is what says whether search is helping.
        """
        if not _is_search_policy(policy):
            return None
        To, Ta = policy.n_obs_steps, policy.n_action_steps
        sl = slice(To - 1, To - 1 + Ta)
        action_normalizer = policy.normalizer['action']
        mins, avgs, firsts, vals = list(), list(), list(), list()
        vals_best, vals_first = list(), list()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                # copy, never mutate: batch['obs'] may alias a long-lived batch
                obs_dict = dict(batch['obs'])
                if batch.get('attention_mask', None) is not None:
                    obs_dict['attention_mask'] = batch['attention_mask']
                # ask for scores explicitly: in the wider search_context modes `values`
                # is the rollout state, not a rankable scalar.
                pred_action, _, scores = policy.search_candidates(
                    obs_dict, verifier=policy.verifier,
                    n_actions=policy.max_actions, return_scores=True)  # (B,K,H,Da), _, (B,K)
                npred = action_normalizer.normalize(pred_action[:, :, sl])   # B, K, Ta, Da
                ngt = action_normalizer.normalize(batch['action'][:, sl]).unsqueeze(1)
                rmse = (npred - ngt).pow(2).mean(dim=(-1, -2)).sqrt()        # B, K
                mins.append(rmse.min(dim=1).values)                          # B
                avgs.append(rmse.mean(dim=1))                                # B
                firsts.append(rmse[:, 0])                                    # B
                vals.append(scores.mean(dim=1))                              # B
                vals_best.append(scores.max(dim=1).values)                   # B
                vals_first.append(scores[:, 0])                              # B
                if max_batches is not None and batch_idx >= max_batches - 1:
                    break
        if not mins:
            return None
        return {
            'nrmse_min': torch.cat(mins).mean().item(),
            'nrmse_avg': torch.cat(avgs).mean().item(),
            'nrmse_first': torch.cat(firsts).mean().item(),
            'action_value': torch.cat(vals).mean().item(),
            'action_value_best': torch.cat(vals_best).mean().item(),
            'action_value_first': torch.cat(vals_first).mean().item(),
        }

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

        # Pretrained weights are the INITIALIZATION for a fresh finetune run only. If we
        # just resumed, the resumed weights are strictly newer -- loading the pretrained
        # ones on top would throw away all progress while keeping the resumed optimizer
        # state and global_step, which is silently catastrophic on any restart.
        if cfg.checkpoint.get('pretrained_ckpt_path', None) is not None and not checkpoint_loaded:
            pretrained_ckpt_path = pathlib.Path(
                hydra.utils.to_absolute_path(cfg.checkpoint.pretrained_ckpt_path))
            if pretrained_ckpt_path.is_file():
                print(f"Loading pretrained weights from {pretrained_ckpt_path}")
                # include_keys=[] so no pickled workspace state (global_step, epoch,
                # last_checkpoint_step) leaks in from the pretrained run; this is a fresh
                # run that merely starts from those weights.
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

        # optional held-out test dataloader (only for a real 3-way split, where the val
        # and test pools are distinct; with the legacy 2-way split they are the same set,
        # so we skip it to avoid duplicated metrics).
        test_dataloader = None
        if hasattr(dataset, 'get_test_dataset') and \
                getattr(dataset, 'val_pool', None) is not getattr(dataset, 'test_pool', None):
            test_dataset = dataset.get_test_dataset()
            test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, **cfg.val_dataloader)

        # configure lr scheduler
        # optional extra kwargs for custom schedules (e.g. decay_then_constant:
        # {decay_steps, min_lr_ratio}); empty for the standard diffusers schedules.
        lr_scheduler_kwargs = dict(cfg.training.get('lr_scheduler_kwargs', {}) or {})
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            # optimizer steps, not batches: the epoch's final partial accumulation window
            # is flushed, so there are ceil(len/accum) steps per epoch (floor-dividing
            # here left the schedule short of its floor by up to one step per epoch).
            num_training_steps=math.ceil(
                len(train_dataloader) / cfg.training.gradient_accumulate_every
                ) * cfg.training.num_epochs,
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
            # The search verifier and the env runner each hold a pool of worker
            # SUBPROCESSES, which garbage collection will not reap. Register their
            # shutdown so they are released on normal exit and on exceptions alike.
            stack.callback(self._close_worker_pools, env_runner)
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

                        # Accumulation windows are counted in BATCHES, not in global_step
                        # (which counts optimizer steps and so cannot delimit its own
                        # windows). The epoch's final window is always flushed, so a
                        # partial window's gradients never leak into the next epoch.
                        max_train_steps = cfg.training.max_train_steps
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
                            # global_step is exactly the number of optimizer steps taken,
                            # which is what checkpoint_every and step_*.ckpt names mean.
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
                        # (same global_step), so wandb steps stay strictly increasing.
                        if is_optimizer_step and not is_last_batch:
                            if self.accelerator.is_main_process:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)

                        # checkpoint on the gradient-step schedule
                        if is_optimizer_step:
                            next_checkpoint_step = self.last_checkpoint_step + cfg.training.checkpoint_every
                            if self.global_step >= next_checkpoint_step and self.accelerator.is_main_process:
                                print(f"Saving checkpoint at step {self.global_step} (target was {next_checkpoint_step})")
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

                                # Update last checkpoint step
                                self.last_checkpoint_step = self.global_step

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

                    # best-of-candidates normalized RMSE on val (and test, if 3-way).
                    # Search policies only; bounded by nrmse_max_batches (verifier is
                    # physics-simulated, so this is not free).
                    if _is_search_policy(policy):
                        nrmse_max_batches = cfg.training.get('nrmse_max_batches', None)
                        for prefix, loader in (('val', val_dataloader),
                                               ('test', test_dataloader)):
                            if loader is None:
                                continue
                            metrics = self._search_action_nrmse(
                                policy, loader, device, nrmse_max_batches)
                            if metrics is None:
                                continue
                            for key, value in metrics.items():
                                step_log[f'{prefix}_{key}'] = value

                # run sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        # copy, never mutate: train_sampling_batch is held on GPU for the
                        # whole run, so anything inserted into its obs dict sticks forever
                        obs_dict = dict(batch['obs'])
                        gt_action = batch['action'][
                            :, policy.n_obs_steps-1:policy.n_obs_steps+policy.n_action_steps-1
                        ]
                        if batch.get('attention_mask', None) is not None:
                            obs_dict['attention_mask'] = batch['attention_mask']

                        if _is_search_policy(policy):
                            # search policies whose context feedback is wider than the
                            # scalar (e.g. a rollout state) also return the scalar
                            # ranking score; log that, not the mean over the context.
                            # also ask for the subgoals: this block already runs a full
                            # search, so reusing it is far cheaper than a second one.
                            pred_action, _, scores, subgoals = policy.search_candidates(
                                obs_dict, verifier=policy.verifier,
                                n_actions=policy.max_actions,
                                return_scores=True, return_subgoals=True)
                            pred_action = pred_action[:, :, policy.n_obs_steps-1:policy.n_obs_steps+policy.n_action_steps-1]
                            mse = (pred_action - gt_action.unsqueeze(1)).pow(2).mean(dim=(-1, -2)) # B, max_actions
                            min_mse, _ = torch.min(mse, dim=-1) # B
                            avg_mse = mse.mean(dim=-1) # B
                            step_log['train_action_mse_error_min'] = min_mse.mean().item()
                            step_log['train_action_mse_error_avg'] = avg_mse.mean().item()
                            # candidate 0 == empty search context == the no-search
                            # baseline; (first - min) is the best-of-n gain.
                            step_log['train_action_mse_error_first'] = mse[:, 0].mean().item()
                            step_log['train_action_value'] = scores.mean().item()
                            step_log['train_action_value_best'] = scores.max(dim=1).values.mean().item()
                            step_log['train_action_value_first'] = scores[:, 0].mean().item()
                            panel = _subgoal_panel(subgoals)
                            if panel is not None:
                                step_log['train_subgoals'] = panel
                            del subgoals
                        else:
                            pred_action = policy.predict_action(obs_dict)['action']
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            step_log['train_action_mse_error'] = mse.item()
                        # release RAM
                        del batch
                        del obs_dict
                        del gt_action
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
                # NOTE: global_step is NOT incremented here. It counts optimizer steps and
                # is advanced only in the training loop; incrementing it per epoch as well
                # made it drift from the true step count (and from the step_*.ckpt names
                # that the eval script parses) by one per epoch.
                self.epoch += 1
        # the last checkpoint may still be writing on the background save thread; without
        # this the process can exit and leave it truncated
        self.join_saving_thread()
        # Guard: end_training() iterates self.accelerator.trackers, which is only
        # populated by init_trackers(); we never call it (no log_with=), so an empty list
        # here means there is nothing to end.
        if getattr(self.accelerator, 'trackers', None):
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
