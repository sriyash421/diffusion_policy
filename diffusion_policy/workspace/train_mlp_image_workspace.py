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

from diffusion_policy.workspace.base_workspace import BaseWorkspace, clone_policy
from diffusion_policy.policy.mlp_image_policy import MLPImagePolicy
from diffusion_policy.model.diffusion.ema_model import EMAModel
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


# The arm label for each (search_context, selection) pair. This is what the run DIRECTORY
# is named, so it is asserted against cfg.arm below rather than left as a comment: the run
# dir is how every downstream report identifies the arm, and a config whose label disagreed
# with its mechanism would mislabel a whole column of SUCCESS_RATES.md with no error.
_ARM_LABELS = {
    ('value', 'argmax'): 'value',
    ('subgoal', 'argmax'): 'subgoal-chosen4value',
    ('subgoal_value', 'argmax'): 'subgoal-value',
    ('subgoal', 'final_pass'): 'subgoal-only',
}


# Policy families whose arm is named after the FAMILY rather than the search-context
# ablation, because the family is what they vary against train_pusht_diffusion_search: a
# Gaussian candidate is one rsample where the diffusion arm runs a denoising loop, and that
# -- not the context -- is the comparison. Their context ablations still have to be
# nameable, so anything other than the default (value, argmax) carries the context in the
# label too; see _expected_arm.
_FAMILY_ARMS = {'PushTGaussianSearchPolicy': 'gaussian'}


def _expected_arm(cfg, key):
    """The arm label cfg SHOULD declare, or None if (search_context, selection) is unnamed.

    For the context-ablation families this is just _ARM_LABELS[key]. For a family in
    _FAMILY_ARMS it is the family name at the default context, and family-context otherwise
    -- so two Gaussian runs that differ only in search_context still get distinct run
    directories rather than silently sharing one.
    """
    context_label = _ARM_LABELS.get(key)
    target = str((cfg.get('policy', None) or {}).get('_target_', '') or '')
    family = _FAMILY_ARMS.get(target.rsplit('.', 1)[-1])
    if family is None:
        return context_label
    if context_label is None:
        return None
    return family if key == ('value', 'argmax') else f'{family}-{context_label}'


def _check_arm_label(cfg):
    """Fail fast if cfg.arm disagrees with (search_context, selection).

    Only checked when the config declares an `arm` -- the maze configs and the pre-arm
    PushT configs do not, and are left alone.
    """
    arm = cfg.get('arm', None)
    if arm is None:
        return
    key = (cfg.get('search_context', 'value') or 'value',
           cfg.get('selection', 'argmax') or 'argmax')
    expected = _expected_arm(cfg, key)
    if expected is None:
        raise ValueError(
            f'no arm label defined for search_context/selection {key}; add it to '
            f'_ARM_LABELS (and pick the run-directory name deliberately) before training '
            f'a combination nothing can name.')
    if arm != expected:
        raise ValueError(
            f'config declares arm={arm!r} but search_context/selection {key} is the '
            f'{expected!r} arm. The run directory is named from `arm`, so this would file '
            f'the results under the wrong ablation.')


def _is_search_policy(policy) -> bool:
    """Whether this policy exposes the best-of-n search interface.

    One explicit capability check, used everywhere instead of ad-hoc `hasattr(policy,
    'verifier')` / `getattr(policy, 'supports_return_scores')` probes. Search policies add
    `search_candidates` / `predict_action_best` on top of the standard `predict_action`
    contract; everything else is driven through `predict_action` alone.
    """
    return hasattr(policy, 'search_candidates') and hasattr(policy, 'verifier')


class TrainMLPImageWorkspace(BaseWorkspace):
    # last_{rollout,val,sample}_step make the gradient-step eval cadences resume-safe:
    # without them a resumed run re-fires every eval block immediately.
    include_keys = [
        'global_step', 'epoch', 'last_checkpoint_step',
        'last_rollout_step', 'last_val_step', 'last_sample_step',
    ]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        _check_arm_label(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: MLPImagePolicy = (
            hydra.utils.instantiate(cfg.policy))

        # Built here, BEFORE any load_checkpoint: load_payload assigns into
        # self.__dict__[key] for every saved state_dict, so a use_ema checkpoint cannot be
        # resumed into a workspace that has no ema_model attribute. (Which also means
        # use_ema cannot be flipped on resume.)
        self.ema_model = None
        if cfg.training.get('use_ema', False):
            self.ema_model = clone_policy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.last_checkpoint_step = 0  # Track last checkpoint step
        # gradient-step eval cadences (see run()); epochs are not a stable unit here
        # because an epoch's length changes with the dataset size and train budget.
        self.last_rollout_step = 0
        self.last_val_step = 0
        self.last_sample_step = 0

        # accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
        )
        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def _make_nrmse_loader(self, split_dataset, cfg, collate_fn, seed=0):
        """Loader for the search-quality metrics: a FIXED, episode-spanning window subset.

        The plain val/test loader has `shuffle: False` and the sampler emits in episode
        order, so truncating it at `nrmse_max_batches` (4 x 32 = 128 windows) drew every
        window from the split's FIRST episode -- 2% of the test split. Everything named
        `val_*`/`test_*`, including the `first - min` search-gain statistic the whole
        "does search help" claim rests on, was a single-episode number.

        A seeded Subset fixes both halves of that: it spans the split, and because the
        indices are drawn once with a fixed seed it is the SAME windows at every
        evaluation, so the metric is comparable across training steps. A reshuffling
        loader would span the split but inject fresh sampling noise into a curve that is
        read over time. Cost is unchanged -- still `nrmse_max_batches` batches.
        """
        n_windows = len(split_dataset)
        want = int(cfg.training.get('nrmse_max_batches', 4) or 4) * int(cfg.val_dataloader.batch_size)
        want = min(want, n_windows)
        idx = np.random.default_rng(seed).choice(n_windows, size=want, replace=False)
        subset = torch.utils.data.Subset(split_dataset, sorted(idx.tolist()))
        kwargs = dict(cfg.val_dataloader)
        kwargs['shuffle'] = False          # the subset already carries the spread
        kwargs['persistent_workers'] = False
        return DataLoader(subset, collate_fn=collate_fn, **kwargs)

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

        Under ``selection: final_pass`` two more are added, because none of the above
        describes the action that arm actually executes -- it deploys a FURTHER sample
        conditioned on all K candidates, not any of them:

          nrmse_final        that extra sample vs the expert action.
          action_value_final its verifier score. Simulated for MONITORING ONLY; the
                             deployed policy never scores it. ``final - best`` is the
                             arm's whole question: does the model's own synthesis beat the
                             oracle argmax it is trying to replace?
        """
        if not _is_search_policy(policy):
            return None
        To, Ta = policy.n_obs_steps, policy.n_action_steps
        sl = slice(To - 1, To - 1 + Ta)
        action_normalizer = policy.normalizer['action']
        mins, avgs, firsts, vals = list(), list(), list(), list()
        vals_best, vals_first = list(), list()
        finals, vals_final = list(), list()
        final_pass = getattr(policy, 'selection', 'argmax') == 'final_pass'
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                # copy, never mutate: batch['obs'] may alias a long-lived batch
                obs_dict = dict(batch['obs'])
                if batch.get('attention_mask', None) is not None:
                    obs_dict['attention_mask'] = batch['attention_mask']
                # ask for scores explicitly: in the wider search_context modes `values`
                # is the rollout state, not a rankable scalar. `values` IS what the extra
                # final pass has to be conditioned on, so keep it too.
                pred_action, values, scores = policy.search_candidates(
                    obs_dict, verifier=policy.verifier,
                    n_actions=policy.max_actions, return_scores=True)  # (B,K,H,Da), ctx, (B,K)
                npred = action_normalizer.normalize(pred_action[:, :, sl])   # B, K, Ta, Da
                ngt = action_normalizer.normalize(batch['action'][:, sl]).unsqueeze(1)
                rmse = (npred - ngt).pow(2).mean(dim=(-1, -2)).sqrt()        # B, K
                mins.append(rmse.min(dim=1).values)                          # B
                avgs.append(rmse.mean(dim=1))                                # B
                firsts.append(rmse[:, 0])                                    # B
                vals.append(scores.mean(dim=1))                              # B
                vals_best.append(scores.max(dim=1).values)                   # B
                vals_first.append(scores[:, 0])                              # B
                if final_pass:
                    # exactly what predict_action_best deploys, plus one sim to score it
                    keep = policy.max_actions - 1
                    final = policy.predict_action(
                        obs_dict, actions=pred_action[:, -keep:],
                        values=values[:, -keep:])['action_pred']             # B, H, Da
                    nfinal = action_normalizer.normalize(final[:, sl])       # B, Ta, Da
                    finals.append(
                        (nfinal - ngt[:, 0]).pow(2).mean(dim=(-1, -2)).sqrt())
                    vals_final.append(
                        policy._score_candidates(policy.verifier, obs_dict, final)[1])
                if max_batches is not None and batch_idx >= max_batches - 1:
                    break
        if not mins:
            return None
        out = {
            'nrmse_min': torch.cat(mins).mean().item(),
            'nrmse_avg': torch.cat(avgs).mean().item(),
            'nrmse_first': torch.cat(firsts).mean().item(),
            'action_value': torch.cat(vals).mean().item(),
            'action_value_best': torch.cat(vals_best).mean().item(),
            'action_value_first': torch.cat(vals_first).mean().item(),
        }
        if finals:
            out['nrmse_final'] = torch.cat(finals).mean().item()
            out['action_value_final'] = torch.cat(vals_final).mean().item()
        return out

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
                # Refuse a pre-EMA payload resumed into a use_ema run -- see
                # BaseWorkspace.assert_ema_payload. Shared with the outer/inner and
                # diffusion-UNet workspaces, which each override run() and so need the
                # same guard on their own resume paths.
                self.assert_ema_payload(payload, lastest_ckpt_path)

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

        # record run identity (git sha, slurm job id, arm, seed) next to the results
        if self.accelerator.is_main_process:
            self.write_manifest()

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        # Record the exact episodes this run trains/validates/tests on, and refuse to
        # resume if they differ from what the run directory already recorded. Cannot fold
        # into write_manifest above -- that runs before the dataset exists.
        if self.accelerator.is_main_process:
            self.write_splits(dataset)

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
        if self.ema_model is not None:
            # the EMA copy is what gets evaluated and checkpointed; without this it would
            # carry empty normalizer stats
            self.ema_model.set_normalizer(normalizer)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, **cfg.val_dataloader)
        # separate from val_dataloader: the loss loop wants every window in order, the
        # search metrics want a fixed spread-out subset (see _make_nrmse_loader)
        nrmse_val_loader = self._make_nrmse_loader(val_dataset, cfg, collate_fn, seed=0)

        # optional held-out test dataloader (only for a real 3-way split, where the val
        # and test pools are distinct; with the legacy 2-way split they are the same set,
        # so we skip it to avoid duplicated metrics).
        test_dataloader = None
        nrmse_test_loader = None
        if hasattr(dataset, 'get_test_dataset') and \
                getattr(dataset, 'val_pool', None) is not getattr(dataset, 'test_pool', None):
            test_dataset = dataset.get_test_dataset()
            test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, **cfg.val_dataloader)
            nrmse_test_loader = self._make_nrmse_loader(test_dataset, cfg, collate_fn, seed=1)

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

        # configure best-checkpoint retention (optional: only when checkpoint.topk is set)
        topk_manager = None
        if cfg.checkpoint.get('topk', None) is not None:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk)

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
            # load_state_dict). Left at 0 on resume, get_decay returns 0.0 for step <= 0
            # and the update degenerates to ema_param.copy_(param) -- silently destroying
            # the restored average on the first post-resume step.
            ema.optimization_step = self.global_step

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
            # the gradient-step cadences take precedence over the epoch ones when set, so
            # they must be overridden too or a debug run fires no eval block at all
            for key in ('rollout_every_steps', 'val_every_steps', 'sample_every_steps'):
                if cfg.training.get(key, None) is not None:
                    cfg.training[key] = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with contextlib.ExitStack() as stack:
            json_logger = stack.enter_context(JsonLogger(log_path))
            # The search verifier and the env runner each hold a pool of worker
            # SUBPROCESSES, which garbage collection will not reap. Register their
            # shutdown so they are released on normal exit and on exceptions alike.
            stack.callback(self._close_worker_pools, env_runner)
            # Hard stop in GRADIENT STEPS. Enforced both here (so a run resumed already past
            # the cap exits before training) and MID-EPOCH in the batch loop below, which is
            # what actually makes it exact.
            max_gradient_steps = cfg.training.get('max_gradient_steps', None)
            stop_training = False
            for local_epoch_idx in range(cfg.training.num_epochs):
                if max_gradient_steps is not None \
                    and self.global_step >= max_gradient_steps:
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
                        # Refreshed every batch, NOT frozen at epoch 0. The old
                        # `if train_sampling_batch is None` guard pinned the FIRST batch of
                        # the FIRST epoch and reused that same GPU tensor for every
                        # sample_every evaluation for the rest of the run -- so
                        # train_action_* measured fit on 32 windows the model re-saw every
                        # epoch, not anything that moved with training.
                        train_sampling_batch = batch

                        # compute loss
                        model = self.accelerator.unwrap_model(self.model)
                        # Crop offsets are a pure function of (seed, global_step): the same
                        # offset covers this sample's obs window AND every subgoal image the
                        # search generates from it, and a resumed run reproduces an
                        # uninterrupted one because there is no RNG state to restore.
                        if hasattr(model, 'set_crop_step'):
                            model.set_crop_step(cfg.training.seed, self.global_step)
                        raw_loss = model.compute_loss(batch)
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
                            if ema is not None:
                                # per OPTIMIZER step, not per batch
                                ema.step(self.accelerator.unwrap_model(self.model))
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
                                # Anchor BEFORE writing, so the value that lands in the
                                # checkpoint is the step it was taken at. Setting it
                                # afterwards persists the PREVIOUS anchor, so a resumed run
                                # believes it is overdue and fires an extra checkpoint on
                                # its very first step.
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

                        # Stop on the EXACT step. The epoch-level check above only fires
                        # between epochs, so on its own it overshoots by up to a full epoch
                        # -- and whenever the final epoch begins below the cap it never
                        # fires at all, leaving num_epochs as the real bound. (At 382
                        # steps/epoch x 786 epochs the last epoch started at 299,870, so a
                        # 300,000 cap did nothing and the run ended at 300,252.)
                        #
                        # Placed after the checkpoint block so the final step is saved, and
                        # it leaves the end-of-epoch block below to run once on the
                        # truncated epoch -- so a capped run still ends with a val_loss and
                        # a topk checkpoint.
                        if max_gradient_steps is not None \
                            and self.global_step >= max_gradient_steps:
                            stop_training = True
                            break

                        if max_train_steps is not None \
                            and batch_idx >= (max_train_steps - 1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss
                policy = self.accelerator.unwrap_model(self.model)
                # EMA weights are what gets rolled out, validated, sampled and checkpointed
                # by topk when enabled; the live weights are only the optimization target.
                eval_policy = self.ema_model if self.ema_model is not None else policy
                eval_policy.eval()
                policy.eval()

                # Eval cadences in GRADIENT STEPS. Epochs are not a stable unit: an epoch's
                # length depends on the dataset size and the train budget, so the same
                # `val_every: 10` fired every ~960 steps at a 25-episode budget and every
                # ~4030 at 100 episodes. rollout_every_steps is additionally a multiple of
                # checkpoint_every, so every in-training success number belongs to weights
                # that were actually saved and can be re-verified against an eval curve.
                # The epoch-based keys remain as the fallback for configs predating this.
                rollout_every_steps = cfg.training.get('rollout_every_steps', None)
                if rollout_every_steps is None:      # legacy epoch-based configs
                    do_rollout = (self.epoch % cfg.training.rollout_every) == 0
                else:
                    do_rollout = (self.global_step - self.last_rollout_step) >= rollout_every_steps
                val_every_steps = cfg.training.get('val_every_steps', None)
                if val_every_steps is None:
                    do_val = (self.epoch % cfg.training.val_every) == 0
                else:
                    do_val = (self.global_step - self.last_val_step) >= val_every_steps
                sample_every_steps = cfg.training.get('sample_every_steps', None)
                if sample_every_steps is None:
                    do_sample = (self.epoch % cfg.training.sample_every) == 0
                else:
                    do_sample = (self.global_step - self.last_sample_step) >= sample_every_steps

                # run rollout
                if do_rollout:
                    self.last_rollout_step = self.global_step
                    # Seed before the rollout so its success rate is reproducible from the
                    # checkpoint. The env itself is deterministic given a reset state, but
                    # conditional_sample draws from the global RNG, so without this the
                    # number depended on whatever state the preceding training left behind.
                    torch.manual_seed(cfg.training.seed)
                    np.random.seed(cfg.training.seed)
                    runner_log = env_runner.run(eval_policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if do_val:
                    self.last_val_step = self.global_step
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = eval_policy.compute_loss(batch)
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
                    if _is_search_policy(eval_policy):
                        nrmse_max_batches = cfg.training.get('nrmse_max_batches', None)
                        for prefix, loader in (('val', nrmse_val_loader),
                                               ('test', nrmse_test_loader)):
                            if loader is None:
                                continue
                            metrics = self._search_action_nrmse(
                                eval_policy, loader, device, nrmse_max_batches)
                            if metrics is None:
                                continue
                            for key, value in metrics.items():
                                step_log[f'{prefix}_{key}'] = value

                # run sampling on a training batch
                if do_sample:
                    self.last_sample_step = self.global_step
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        # copy, never mutate: train_sampling_batch is held on GPU for the
                        # whole run, so anything inserted into its obs dict sticks forever
                        obs_dict = dict(batch['obs'])
                        gt_action = batch['action'][
                            :, eval_policy.n_obs_steps-1:eval_policy.n_obs_steps+eval_policy.n_action_steps-1
                        ]
                        if batch.get('attention_mask', None) is not None:
                            obs_dict['attention_mask'] = batch['attention_mask']

                        if _is_search_policy(eval_policy):
                            # search policies whose context feedback is wider than the
                            # scalar (e.g. a rollout state) also return the scalar
                            # ranking score; log that, not the mean over the context.
                            # also ask for the subgoals: this block already runs a full
                            # search, so reusing it is far cheaper than a second one.
                            all_action, values, scores, subgoals = eval_policy.search_candidates(
                                obs_dict, verifier=eval_policy.verifier,
                                n_actions=eval_policy.max_actions,
                                return_scores=True, return_subgoals=True)
                            asl = slice(eval_policy.n_obs_steps-1,
                                        eval_policy.n_obs_steps+eval_policy.n_action_steps-1)
                            pred_action = all_action[:, :, asl]
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
                            if getattr(eval_policy, 'selection', 'argmax') == 'final_pass':
                                # the sample this arm deploys is none of the candidates
                                # above -- draw it and log it alongside them
                                keep = eval_policy.max_actions - 1
                                final = eval_policy.predict_action(
                                    obs_dict, actions=all_action[:, -keep:],
                                    values=values[:, -keep:])['action_pred']
                                fscore = eval_policy._score_candidates(
                                    eval_policy.verifier, obs_dict, final)[1]
                                step_log['train_action_mse_error_final'] = (
                                    final[:, asl] - gt_action).pow(2).mean().item()
                                step_log['train_action_value_final'] = fscore.mean().item()
                                del final
                            del all_action, values
                            panel = _subgoal_panel(subgoals)
                            if panel is not None:
                                step_log['train_subgoals'] = panel
                            del subgoals
                        else:
                            pred_action = eval_policy.predict_action(obs_dict)['action']
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

                # keep the k best checkpoints by the monitored metric. This is the safety
                # net that was missing: a previous run's val_loss bottomed at ~3k steps and
                # then rose 3x while training continued to 100k, and with no topk manager
                # the good weights were simply lost. get_ckpt_path returns None when the
                # metric is absent this epoch (cadences differ), so this is a no-op then.
                if topk_manager is not None and self.accelerator.is_main_process:
                    metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        model_ddp = self.model
                        self.model = self.accelerator.unwrap_model(self.model)
                        try:
                            self.save_checkpoint(path=topk_ckpt_path)
                        finally:
                            self.model = model_ddp

                if self.accelerator.is_main_process:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                # NOTE: global_step is NOT incremented here. It counts optimizer steps and
                # is advanced only in the training loop; incrementing it per epoch as well
                # made it drift from the true step count (and from the step_*.ckpt names
                # that the eval script parses) by one per epoch.
                self.epoch += 1

                # The end-of-epoch validation, sampling and topk checkpoint above have
                # already run, so the run ends fully evaluated rather than mid-epoch.
                if stop_training:
                    print(f'Reached max_gradient_steps={max_gradient_steps}, stopping.')
                    break
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
