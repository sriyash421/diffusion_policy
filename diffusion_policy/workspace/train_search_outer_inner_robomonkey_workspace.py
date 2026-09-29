if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import contextlib
import copy
import math
import os
import pathlib
import random

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.workspace.base_workspace import clone_policy, set_initial_lr
from diffusion_policy.workspace.train_mlp_image_workspace import (
    TrainMLPImageWorkspace, _is_search_policy)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _cat_batches(batches):
    """Concatenate a list of same-structure batch dicts along the batch dim.

    Batches come off the standard DataLoader, so every leaf is a tensor with a leading
    batch dim and identical trailing shape (``return_sequences`` -- the ragged path -- is
    refused in ``run()``). Nested one level deep, which is the batch contract everywhere
    in this repo: ``{'obs': {key: tensor}, 'action': tensor, ...}``.
    """
    out = dict()
    for key in batches[0]:
        if isinstance(batches[0][key], dict):
            out[key] = {
                sub: torch.cat([b[key][sub] for b in batches], dim=0)
                for sub in batches[0][key]
            }
        else:
            out[key] = torch.cat([b[key] for b in batches], dim=0)
    return out


def _index_batch(batch, idx):
    """Row-select a (possibly nested) batch dict with a LongTensor of indices."""
    return {
        key: ({sub: val[idx] for sub, val in value.items()}
              if isinstance(value, dict) else value[idx])
        for key, value in batch.items()
    }


def _batch_len(batch):
    value = batch['action']
    return value.shape[0]


def _dataloader_kwargs(cfg_node):
    """Config -> DataLoader kwargs, with the worker-dependent options made consistent.

    ``persistent_workers`` and ``prefetch_factor`` are only meaningful with worker
    processes; torch raises outright if either is set while ``num_workers == 0``. Every
    config here carries ``persistent_workers: true``, so any override to 0 workers (a
    debug run, a low-CPU partition) became a crash raised AFTER the dataset had already
    been loaded -- minutes in, for a configuration error. Drop them instead.
    """
    kwargs = dict(cfg_node)
    if int(kwargs.get('num_workers', 0)) == 0:
        kwargs.pop('persistent_workers', None)
        kwargs.pop('prefetch_factor', None)
    return kwargs


class TrainSearchOuterInnerRoboMonkeyWorkspace(TrainMLPImageWorkspace):
    """Outer/inner trainer for the conditional-diffusion search policy.

    The offline parent (``TrainMLPImageWorkspace``) regenerates the search context inside
    ``compute_loss`` on EVERY gradient step: ``max_actions - 1`` candidates per batch
    element, each a full DDIM sample plus a verifier call, all discarded after one update.
    On the eggplant DQC arm that is the dominant term in step time. This workspace
    generates that context once per outer step for a pool of ``outer_batch_size`` windows
    and reuses it for ``inner_epochs`` passes, which divides the search cost per update by
    exactly ``inner_epochs``.

    The price is staleness: the buffered context comes from weights up to ``num_inner``
    updates old. That is measured rather than assumed -- a frozen snapshot of the collector
    policy is kept for the whole inner loop, and ``train_drift_mse_eps`` reports how far
    the live policy has moved from it (see ``_drift_mse``).

    Note the loss target is always the GT expert action, so unlike a policy-gradient method
    there is no importance weight to correct; staleness enters only through the
    conditioning context.
    """

    # last_{rollout,val,sample}_step make the gradient-step eval cadences resume-safe:
    # without them a resumed run re-fires every eval block immediately.
    include_keys = [
        'global_step', 'epoch', 'last_checkpoint_step',
        'last_rollout_step', 'last_val_step', 'last_sample_step',
    ]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        self.last_rollout_step = 0
        self.last_val_step = 0
        self.last_sample_step = 0

        # Built here, BEFORE any load_checkpoint: load_payload assigns into
        # self.__dict__[key] for every saved state_dict, so a use_ema checkpoint cannot be
        # resumed into a workspace that has no ema_model attribute. (Which also means
        # use_ema cannot be flipped on resume.)
        self.ema_model = None
        if cfg.training.get('use_ema', False):
            self.ema_model = clone_policy(self.model)

    # ------------------------------------------------------------------ drift metrics

    @staticmethod
    def _drift_mse(policy, collector, batch, aux):
        """Epsilon-space MSE between the live policy and the snapshot that collected the
        buffered context, at matched ``(noisy_trajectory, timestep, obs, context)``.

        For fixed inputs the two snapshots' reverse transition kernels are Gaussians
        sharing the scheduler's variance, so their KL is proportional to this quantity --
        the tractable stand-in for the policy KL PPO would monitor. Rising values across
        the inner loop mean the buffered context is going stale.

        Both sides run in eval mode: with ``p_drop_attn`` active the difference would be
        dominated by independent dropout draws rather than by any real drift.
        """
        was_training = policy.training
        policy.eval()
        try:
            eps_new = policy.predict_epsilon(
                batch, aux['actions'], aux['values'],
                aux['noisy_trajectory'], aux['timesteps'])
            eps_old = collector.predict_epsilon(
                batch, aux['actions'], aux['values'],
                aux['noisy_trajectory'], aux['timesteps'])
        finally:
            if was_training:
                policy.train()
        return F.mse_loss(eps_new, eps_old).item()

    # ------------------------------------------------------------------ search metrics

    def _search_action_nrmse(self, policy, dataloader, device, max_batches=None):
        """Search-quality metrics over a split (search policies only).

        For each window, generate ``max_actions`` candidates and compare each against the
        expert action over the EXECUTED action window, in the normalizer's normalized
        action space -- the same slice and space the parent workspace's
        ``*_action_rmse_error_*_normalized`` metrics use, so the two arms' wandb panels
        line up.

          rmse_min    best candidate -- how good search *can* get
          rmse_avg    mean candidate -- the control: did the distribution itself move, or
                      are we just drawing more samples from an unchanged one?
          rmse_first  candidate 0, generated with an EMPTY search context, i.e. the
                      no-search baseline. (first - min) is the actual best-of-n gain,
                      which rmse_min alone cannot show.
          action_value{,_best,_first}  the same three readings in verifier-value space.
        """
        if not _is_search_policy(policy):
            return None
        To, Ta = policy.n_obs_steps, policy.n_action_steps
        sl = slice(To - 1, To - 1 + Ta)
        action_normalizer = policy.normalizer['action']
        mins, avgs, firsts = list(), list(), list()
        vals, vals_best, vals_first = list(), list(), list()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                # copy, never mutate: obs may alias a long-lived batch
                obs_dict = dict(batch['obs'])
                if batch.get('attention_mask', None) is not None:
                    obs_dict['attention_mask'] = batch['attention_mask']
                pred_action, scores = policy.search_candidates(
                    obs_dict, verifier=policy.verifier,
                    n_actions=policy.max_actions)          # (B,K,H,Da), (B,K)
                npred = action_normalizer.normalize(pred_action[:, :, sl])
                ngt = action_normalizer.normalize(batch['action'][:, sl]).unsqueeze(1)
                rmse = (npred - ngt).pow(2).mean(dim=(-1, -2)).sqrt()   # (B, K)
                mins.append(rmse.min(dim=1).values)
                avgs.append(rmse.mean(dim=1))
                firsts.append(rmse[:, 0])
                vals.append(scores.mean(dim=1))
                vals_best.append(scores.max(dim=1).values)
                vals_first.append(scores[:, 0])
                if max_batches is not None and batch_idx >= max_batches - 1:
                    break
        if not mins:
            return None
        return {
            'action_rmse_error_min_normalized': torch.cat(mins).mean().item(),
            'action_rmse_error_avg_normalized': torch.cat(avgs).mean().item(),
            'action_rmse_error_first_normalized': torch.cat(firsts).mean().item(),
            'action_value': torch.cat(vals).mean().item(),
            'action_value_best': torch.cat(vals_best).mean().item(),
            'action_value_first': torch.cat(vals_first).mean().item(),
        }

    # ------------------------------------------------------------------ context buffer

    def _fill_pool(self, dataloader, data_iter, outer_bs):
        """Pull consecutive batches off the standard DataLoader until the pool is full.

        Using the configured DataLoader (workers, pin_memory, persistent_workers) rather
        than indexing the dataset synchronously matters here: the eggplant obs carries a
        3x224x224 image per window, and single-threaded fetching would hand back the wall
        clock this workspace exists to save.

        The pool is kept on the CPU and sliced onto the device per inner step, so
        outer_batch_size can be raised without risking a device OOM that only appears
        several outer steps in.
        """
        chunks, n = list(), 0
        while n < outer_bs:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            chunks.append(batch)
            n += _batch_len(batch)
        pool = _cat_batches(chunks)
        if n > outer_bs:
            keep = torch.arange(outer_bs)
            pool = _index_batch(pool, keep)
        return pool, data_iter

    def _fill_context_buffer(self, policy, pool, device, chunk_size):
        """Generate the search context for every window in ``pool``, once.

        Generated in chunks of the configured dataloader batch size so the verifier sees
        the same batch width it does during offline training. Runs under ``eval()``: the
        offline path generates its context with dropout active (a known train/eval context
        mismatch), and a buffer reused across many updates should at least be drawn from
        the deployment-mode distribution.
        """
        actions, values = list(), list()
        was_training = policy.training
        policy.eval()
        try:
            n = _batch_len(pool)
            for start in tqdm.trange(0, n, chunk_size, desc="context buffer",
                    leave=False, mininterval=self.cfg.training.tqdm_interval_sec):
                idx = torch.arange(start, min(start + chunk_size, n))
                chunk = dict_apply(_index_batch(pool, idx),
                                   lambda x: x.to(device, non_blocking=True))
                chunk_actions, chunk_values = policy.generate_search_context(
                    chunk['obs'], action=chunk['action'])
                actions.append(chunk_actions)
                values.append(chunk_values)
        finally:
            if was_training:
                policy.train()
        return torch.cat(actions, dim=0), torch.cat(values, dim=0)

    # ------------------------------------------------------------------ main loop

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # ---- resume ---------------------------------------------------------------
        # Must precede the lr scheduler: rebuilding LambdaLR at last_epoch != -1 requires
        # 'initial_lr' in the optimizer's param_groups, which exists only because the
        # restored optimizer state carries it over from the original construction.
        checkpoint_loaded = False
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                checkpoint_loaded = True
            else:
                # Hydra mints a NEW timestamped output dir per launch, so a plain
                # re-submit looks for latest.ckpt somewhere that cannot contain it and
                # starts over. Silent restarts are expensive here; say so loudly.
                print(f"training.resume=True but no checkpoint found at "
                      f"{lastest_ckpt_path} -- STARTING FROM SCRATCH. To continue a "
                      f"previous run, relaunch with hydra.run.dir=<that run's output dir>.")

        # ---- data -----------------------------------------------------------------
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        if getattr(dataset, 'return_sequences', False):
            raise NotImplementedError(
                "TrainSearchOuterInnerRoboMonkeyWorkspace pools several batches into one tensor per "
                "outer step, which needs fixed-length windows. This dataset has "
                "return_sequences=True (ragged, re-padded by the collate fn). Use "
                "TrainMLPImageWorkspace for it.")

        dataloader_kwargs = _dataloader_kwargs(cfg.dataloader)
        if getattr(dataset, 'weighted_sampler', None) is not None:
            dataloader_kwargs['sampler'] = dataset.weighted_sampler
            dataloader_kwargs.pop('shuffle', None)
        train_dataloader = DataLoader(dataset, **dataloader_kwargs)

        # Same rule as the parent workspace: a resumed checkpoint's normalizer is the one
        # the resumed weights were trained under. Recomputing it would silently change the
        # input/output scaling mid-run.
        if checkpoint_loaded and len(self.model.normalizer.params_dict) > 0:
            print("Checkpoint loaded with normalizer - preserving existing normalizer statistics")
        else:
            print("Computing normalizer from dataset")
            normalizer = dataset.get_normalizer()
            self.model.set_normalizer(normalizer)
            if self.ema_model is not None:
                self.ema_model.set_normalizer(normalizer)

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **_dataloader_kwargs(cfg.val_dataloader))

        # ---- loop geometry, all derived from max_gradient_steps -------------------
        max_steps = cfg.training.max_gradient_steps
        outer_bs = min(cfg.training.outer_batch_size, len(dataset))
        inner_bs = cfg.training.inner_batch_size
        inner_epochs = cfg.training.inner_epochs
        drift_every = cfg.training.drift_every
        rollout_every_steps = cfg.training.rollout_every_steps
        val_every_steps = cfg.training.val_every_steps
        sample_every_steps = cfg.training.sample_every_steps
        buffer_chunk = cfg.dataloader.batch_size

        # Must precede the scheduler, which is sized from max_steps.
        if cfg.training.debug:
            max_steps = 6
            outer_bs, inner_bs, inner_epochs = 8, 2, 1
            drift_every = 1
            rollout_every_steps = val_every_steps = sample_every_steps = 1
            cfg.training.checkpoint_every = 1
            cfg.training.max_val_steps = 2
            buffer_chunk = 4

        # updates per outer step. There is deliberately no num_outer constant -- the outer
        # loop runs until the gradient-step budget is spent.
        num_inner = inner_epochs * math.ceil(outer_bs / inner_bs)
        print(f"outer/inner: {num_inner} updates per outer step "
              f"({inner_epochs} epochs x ceil({outer_bs}/{inner_bs})), "
              f"search cost per update divided by {inner_epochs}x")

        lr_scheduler_kwargs = dict(cfg.training.get('lr_scheduler_kwargs', {}) or {})
        # Decay horizon, decoupled from the run length so the LR schedule can be matched
        # to a comparison run whose budget differs (same rule as TrainMLPImageWorkspace).
        # e.g. polynomial with lr_decay_steps=10000 decays to lr_end by 10k and then HOLDS,
        # regardless of max_gradient_steps. Defaults to the full budget.
        lr_decay_steps = cfg.training.get('lr_decay_steps', None) or max_steps
        # Building at last_epoch >= 0 requires initial_lr in every param group. It is
        # normally restored with the optimizer state; backfill it so a checkpoint written
        # before optimizer state was persisted still resumes instead of raising KeyError.
        set_initial_lr(self.optimizer)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=lr_decay_steps,
            last_epoch=self.global_step - 1,
            **lr_scheduler_kwargs)

        env_runner: BaseImageRunner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        if self.accelerator.is_main_process:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging)
            wandb.config.update({"output_dir": self.output_dir}, allow_val_change=True)

        self.model, self.optimizer, lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, lr_scheduler)
        device = self.accelerator.device
        optimizer_to(self.optimizer, device)

        # ---- EMA ------------------------------------------------------------------
        ema = None
        if cfg.training.get('use_ema', False):
            self.ema_model.to(device)
            decay = cfg.training.ema_decay
            # min_value == max_value clamps get_decay's warmup curve flat, so the decay is
            # the same constant at every step: independent of run length, and identical
            # across restarts. Nothing schedule-shaped is left to restore.
            ema = EMAModel(model=self.ema_model, update_after_step=0,
                           inv_gamma=1.0, power=0.75,
                           min_value=decay, max_value=decay)
            # EMAModel exposes no state_dict, so save_checkpoint never persists this
            # counter (it only serializes attributes having state_dict AND
            # load_state_dict). Left at 0 on resume, get_decay returns 0.0 and the update
            # degenerates to ema_param.copy_(param) -- silently destroying the restored
            # average on the first post-resume step.
            ema.optimization_step = self.global_step

        train_sampling_batch = None
        data_iter = iter(train_dataloader)
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with contextlib.ExitStack() as stack:
            json_logger = stack.enter_context(JsonLogger(log_path))
            # a verifier or env runner may hold worker SUBPROCESSES that garbage
            # collection will not reap; release them on normal exit and on error alike.
            stack.callback(self._close_worker_pools, env_runner)

            # One collector instance for the whole run, refreshed in place each outer
            # step. Deep-copying the policy per outer step would churn the allocator a few
            # thousand times over a long run for no benefit.
            collector = clone_policy(self.accelerator.unwrap_model(self.model))
            collector.eval()
            collector.requires_grad_(False)
            can_drift = hasattr(collector, 'predict_epsilon')

            while self.global_step < max_steps:
                policy = self.accelerator.unwrap_model(self.model)

                # ======== outer step: freeze a collector, fill the context buffer =====
                # Snapshot the weights that are about to generate the buffer. They are the
                # live weights right now, so this costs nothing beyond the copy -- the
                # buffer is generated by the live policy and the collector preserves the
                # state that did it, for the drift metric to compare against.
                collector.load_state_dict(policy.state_dict())

                pool, data_iter = self._fill_pool(train_dataloader, data_iter, outer_bs)
                buf_actions, buf_values = self._fill_context_buffer(
                    policy, pool, device, buffer_chunk)

                # ======== inner: reuse that buffer for inner_epochs passes ============
                self.model.train()
                inner_losses = list()
                inner_idx = 0
                pool_n = _batch_len(pool)
                for epoch_idx in range(inner_epochs):
                    # Randomness derived from POSITION (outer index, inner epoch), never
                    # from call count: a resumed run then draws the order this outer step
                    # would have drawn anyway, instead of replaying outer steps 0, 1, 2...
                    order = np.random.default_rng(
                        [cfg.training.seed, self.epoch, epoch_idx]).permutation(pool_n)
                    for start in range(0, pool_n, inner_bs):
                        if self.global_step >= max_steps:
                            break
                        sel = torch.as_tensor(
                            order[start:start + inner_bs].copy(), dtype=torch.long)
                        batch = dict_apply(_index_batch(pool, sel),
                                           lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        sel_dev = sel.to(buf_actions.device)

                        want_aux = can_drift and (inner_idx % drift_every) == 0
                        result = policy.compute_loss(
                            batch,
                            actions=buf_actions[sel_dev],
                            values=buf_values[sel_dev],
                            return_aux=want_aux)
                        raw_loss, aux = result if want_aux else (result, None)

                        self.optimizer.zero_grad()
                        self.accelerator.backward(raw_loss)
                        if cfg.training.get('gradient_clip_norm', None) is not None:
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(), cfg.training.gradient_clip_norm)
                        self.optimizer.step()
                        lr_scheduler.step()
                        if ema is not None:
                            ema.step(policy)
                        self.global_step += 1

                        loss_cpu = raw_loss.item()
                        inner_losses.append(loss_cpu)
                        step_log = {
                            'train_loss': loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0],
                        }
                        if aux is not None:
                            step_log['train_drift_mse_eps'] = self._drift_mse(
                                policy, collector, batch, aux)
                            # position WITHIN the inner loop, so drift growth across a
                            # single buffer is readable -- that is what says whether
                            # inner_epochs is set too high.
                            step_log['train_drift_inner_step'] = inner_idx
                        if self.accelerator.is_main_process:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)

                        next_ckpt = self.last_checkpoint_step + cfg.training.checkpoint_every
                        if self.global_step >= next_ckpt and self.accelerator.is_main_process:
                            self._save(cfg)

                        inner_idx += 1
                    if self.global_step >= max_steps:
                        break

                # ======== end of outer step: evaluation ==============================
                step_log = {
                    'train_loss': float(np.mean(inner_losses)) if inner_losses else float('nan'),
                    'global_step': self.global_step,
                    'epoch': self.epoch,
                }
                # EMA weights are what gets evaluated and checkpointed when enabled; the
                # live weights are only ever the optimization target.
                eval_policy = self.ema_model if self.ema_model is not None else policy
                eval_policy.eval()

                if self.global_step - self.last_rollout_step >= rollout_every_steps:
                    # Re-seed so a given checkpoint's rollout is reproducible rather than
                    # depending on whatever RNG state the preceding updates left behind.
                    torch.manual_seed(cfg.training.seed)
                    np.random.seed(cfg.training.seed)
                    random.seed(cfg.training.seed)
                    step_log.update(env_runner.run(eval_policy))
                    self.last_rollout_step = self.global_step

                if self.global_step - self.last_val_step >= val_every_steps:
                    with torch.no_grad():
                        val_losses = list()
                        for batch_idx, batch in enumerate(val_dataloader):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            val_losses.append(eval_policy.compute_loss(batch).item())
                            if cfg.training.max_val_steps is not None \
                                    and batch_idx >= cfg.training.max_val_steps - 1:
                                break
                        if val_losses:
                            step_log['val_loss'] = float(np.mean(val_losses))
                    metrics = self._search_action_nrmse(
                        eval_policy, val_dataloader, device,
                        cfg.training.get('nrmse_max_batches', None))
                    if metrics is not None:
                        for key, value in metrics.items():
                            step_log[f'val_{key}'] = value
                    self.last_val_step = self.global_step

                if self.global_step - self.last_sample_step >= sample_every_steps \
                        and train_sampling_batch is not None:
                    step_log.update(self._sample_log(eval_policy, train_sampling_batch, device))
                    self.last_sample_step = self.global_step

                if self.accelerator.is_main_process:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                self.epoch += 1
                # the eval blocks above leave the model in eval(); restore train mode so
                # the next outer step does not train with dropout silently disabled
                self.model.train()

        # a threaded save may still be in flight; without this the process can exit and
        # leave the last checkpoint truncated
        self.join_saving_thread()
        if getattr(self.accelerator, 'trackers', None):
            self.accelerator.end_training()

    # ------------------------------------------------------------------ helpers

    def _save(self, cfg):
        """Checkpoint on the gradient-step schedule, with plain (unwrapped) module keys."""
        print(f"Saving checkpoint at step {self.global_step}")
        # Update the cadence anchor BEFORE writing, so the value that lands in the
        # checkpoint is the step this checkpoint was taken at. Setting it afterwards
        # persists the PREVIOUS anchor, so a resumed run believes it is overdue and fires
        # an extra checkpoint on its very first step.
        self.last_checkpoint_step = self.global_step
        model_ddp = self.model
        self.model = self.accelerator.unwrap_model(self.model)
        if cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint()
        if cfg.checkpoint.save_last_snapshot:
            self.save_snapshot()
        step_ckpt_path = os.path.join(
            self.output_dir, 'checkpoints', f'step_{self.global_step:07d}.ckpt')
        os.makedirs(os.path.dirname(step_ckpt_path), exist_ok=True)
        self.save_checkpoint(path=step_ckpt_path)
        self.model = model_ddp

    def _sample_log(self, eval_policy, batch, device):
        """Search-quality readout on a training batch, in the parent workspace's names."""
        log = dict()
        with torch.no_grad():
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            obs_dict = dict(batch['obs'])
            if batch.get('attention_mask', None) is not None:
                obs_dict['attention_mask'] = batch['attention_mask']
            To, Ta = eval_policy.n_obs_steps, eval_policy.n_action_steps
            gt_action = batch['action'][:, To - 1:To - 1 + Ta]

            if not _is_search_policy(eval_policy):
                pred = eval_policy.predict_action(obs_dict)['action']
                log['train_action_mse_error'] = F.mse_loss(pred, gt_action).item()
                return log

            pred_action, scores = eval_policy.search_candidates(
                obs_dict, verifier=eval_policy.verifier,
                n_actions=eval_policy.max_actions)
            pred_action = pred_action[:, :, To - 1:To - 1 + Ta]
            na = eval_policy.normalizer['action']
            rmse_n = (na.normalize(pred_action)
                      - na.normalize(gt_action).unsqueeze(1)).pow(2).mean(dim=(-1, -2)).sqrt()
            log['train_action_rmse_error_min_normalized'] = rmse_n.min(dim=-1).values.mean().item()
            log['train_action_rmse_error_avg_normalized'] = rmse_n.mean(dim=-1).mean().item()
            # candidate 0 is generated with an EMPTY search context, i.e. the no-search
            # baseline; (first - min) is the actual best-of-n gain.
            log['train_action_rmse_error_first_normalized'] = rmse_n[:, 0].mean().item()
            log['train_action_value'] = scores.mean().item()
            log['train_action_value_best'] = scores.max(dim=1).values.mean().item()
            log['train_action_value_first'] = scores[:, 0].mean().item()
        return log


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainSearchOuterInnerRoboMonkeyWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
