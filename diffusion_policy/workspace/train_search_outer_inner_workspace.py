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

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.common.sampler import get_collate_fn
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.workspace.base_workspace import clone_policy as _clone_policy
from diffusion_policy.workspace.train_mlp_image_workspace import (
    TrainMLPImageWorkspace, _is_search_policy, _subgoal_panel)

OmegaConf.register_new_resolver("eval", eval, replace=True)



def _val_slot_weighting(policy) -> bool:
    """Whether val_loss is computed WITH the per-slot weighting.

    Default False (uniform). val_loss is the cross-arm comparison signal, and a weighting
    that changes over training -- a slot_weights curriculum -- would make it move for
    reasons that have nothing to do with fit, so a curve could not be compared with itself.
    The legacy `slot_weight_decay` scalar resolves to 'trained', preserving the semantics
    every run under that key already had.

    There is a second reason the default matters: ema_model is a deepcopy and the workspace
    sets _slot_weight_step only on the LIVE model, so a curriculum evaluated on the EMA copy
    would sit at the schedule's starting profile forever -- silently uniform, but uniform by
    accident rather than by choice.
    """
    spec = getattr(policy, 'slot_weight_spec', None)
    return bool(spec and spec.get('val') == 'trained')


# What an inner gradient step needs from a dataset window BESIDES the encoded features.
# `action` is the denoising target; agent_pos and feedback are the verifier's obs keys
# (pusht_search_mixin._VERIFIER_OBS_KEYS), carried so the obs dict handed to compute_loss
# is well-formed even though nothing on the buffered path reads it. `image` is pointedly
# NOT here -- see _fill_context_buffer.
_WINDOW_OBS_KEYS = ('agent_pos', 'feedback')


def _can_cache_obs(policy) -> bool:
    """Whether the encoded observation may be reused across gradient steps.

    ONLY when the obs backbone has no trainable parameter. With a trainable encoder the
    cache would feed the model activations from an earlier step AND cut the encoder out of
    the gradient entirely -- it would look like a speedup and silently stop training the
    backbone. Checked here rather than assumed from the config, because `freeze` is a
    property of the module (SDVAEEncoder freezes itself), not of a flag anyone can read.
    """
    enc = getattr(policy, 'obs_encoder', None)
    if enc is None or not hasattr(policy, 'train_crops'):
        return False
    return not any(p.requires_grad for p in enc.parameters())


class TrainSearchOuterInnerWorkspace(TrainMLPImageWorkspace):
    """Outer/inner trainer for the conditional-diffusion search policy.

    The offline parent (``TrainMLPImageWorkspace``) regenerates the search context inside
    ``compute_loss`` on EVERY gradient step: ``max_actions - 1`` candidates per batch
    element, each an 8-step DDIM sample plus a physics-simulated verifier rollout, all
    discarded after one update. This workspace generates that context once per outer step
    for a pool of ``outer_batch_size`` windows and reuses it for ``inner_epochs`` passes,
    which divides the search cost per update by exactly ``inner_epochs``.

    The price is staleness: the buffered context comes from weights that are up to
    ``num_inner`` updates old. That is measured rather than assumed -- a frozen snapshot of
    the collector policy is kept for the whole inner loop, and ``train_drift_mse_eps``
    reports how far the live policy has moved from it (see ``_drift_mse``).

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
        if cfg.training.use_ema:
            self.ema_model = _clone_policy(self.model)

        # SUBGOAL CONTEXTS WERE MIS-REGISTERED ON THIS TRAINER, and it was not detectable
        # from the loss. Fixed for a frozen encoder by caching the pool's encode (which pins
        # its crop); still live if the encoder is trainable, hence the guard below. _fill_context_buffer runs under policy.eval(), so a buffered subgoal
        # image is CENTER-cropped, while the observation re-encoded in the inner step is
        # RANDOMLY cropped -- the exact registration mismatch the per-sample shared crop
        # offset exists to prevent. The offline trainer
        # (train_pusht_diffusion_search_single) generates its context inside the same forward
        # pass and shares one offset, so it does not have this.
        #
        # A warning rather than an error: which way to resolve it is the open question about
        # whether the context should be generated in eval or train mode at all, and that is
        # not a decision to make silently at construction. Under `search_context: value` --
        # the default, and every arm currently trained -- the buffer holds scalars and there
        # is no image to mis-register, so this never fires there.
        if (str(cfg.get('search_context', 'value') or 'value').startswith('subgoal')
                and not _can_cache_obs(self.model)):
            print(f'WARNING: {type(self).__name__} with search_context='
                  f'{cfg.search_context!r} and a TRAINABLE obs encoder: the buffered subgoal '
                  f'images and the observation re-encoded in the inner step carry different '
                  f'crops, so the two are not spatially registered. A frozen encoder fixes '
                  f'this -- the pool is encoded once and the crop is pinned with it (see '
                  f'_fill_context_buffer). Otherwise use trainer: offline '
                  f'(train_pusht_diffusion_search_single) for a subgoal arm.')

    # ------------------------------------------------------------------ drift metrics

    @staticmethod
    def _drift_mse(policy, collector, batch, aux):
        """Epsilon-space MSE between the live policy and the snapshot that collected the
        buffered context, at matched ``(noisy_trajectory, timestep, obs, context)``.

        For fixed inputs the two snapshots' reverse transition kernels are Gaussians
        sharing the scheduler's variance, so their KL is exactly proportional to this
        quantity -- the tractable stand-in for the policy KL PPO would monitor. Rising
        values across the inner loop mean the buffered context is going stale.

        Both sides run in eval mode: with ``p_drop_attn=0.2`` active the difference would
        be dominated by independent dropout draws rather than by any real drift.

        AT k=1 THIS IS NOT A STALENESS METRIC. There is no buffered context to go stale
        (``aux['actions']`` is None), so what it measures is plain weight movement between
        the collector snapshot and the live policy. The number is real, but do not read it
        as context drift -- at width 1 there is nothing to be stale.
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

    @staticmethod
    def _drift_action_mse(policy, collector, batch, aux, seed):
        """Same drift, read out in ACTION space: sample a chunk from each snapshot under
        the SAME noise draw and compare.

        Interpretable (PushT pixel units) but costs two full sampling chains, so this is
        logged only on the sampling cadence while ``_drift_mse`` runs throughout.
        """
        was_training = policy.training
        policy.eval()
        preds = list()
        try:
            for snapshot in (policy, collector):
                device = next(snapshot.parameters()).device
                generator = torch.Generator(device=device).manual_seed(seed)
                with torch.no_grad():
                    nsample = snapshot.conditional_sample(
                        obs_cond=snapshot._encode_obs_features(batch['obs']),
                        actions=snapshot._normalize_context_actions(aux['actions']),
                        values=aux['values'],
                        generator=generator)
                preds.append(snapshot.normalizer['action'].unnormalize(nsample))
        finally:
            if was_training:
                policy.train()
        return F.mse_loss(preds[0], preds[1]).item()

    # ------------------------------------------------------------------ context buffer

    def _fill_context_buffer(self, policy, dataset, collate, pool, device, chunk_size):
        """Generate the search context AND the encoded observation for every window, once.

        Returns ``(actions, values, obs_features, window)``. Fetched and generated in chunks
        of the configured dataloader batch size so the verifier's env pool sees the same
        batch width it does during offline training.

        ``window`` carries the non-image remainder of each pool window -- the loss target
        and the verifier's obs keys -- so an inner step can be assembled without touching
        the dataset again. ``None`` when the features are not cached, which is exactly when
        the inner loop must re-read (and re-encode) the images anyway.

        THE ENCODED OBSERVATION IS KEPT TOO, when the backbone is frozen. Caching the raw
        image windows would cost GBs; caching their features is ``outer_batch_size x
        n_obs_steps x D`` floats -- 256 x 2 x 324 x 4 B = 663 kB -- and it removes the
        ``inner_epochs`` re-encodes of every pool window, which on a frozen 34.2M-parameter
        VAE is the dominant per-update cost once the search is amortized. It also PINS THE
        CROP for the pool, which is the fix for the buffered-subgoal mis-registration: the
        observation and the subgoals predicted from it now carry the same offset by
        construction. Cost: a window is cropped the same way for all of its inner passes
        instead of getting a fresh crop each time. See ``_can_cache_obs`` for why the freeze
        is load-bearing.

        WIDTH 1: there is no context to buffer -- ``generate_search_context`` would ask for
        ``max_actions - 1 = 0`` candidates and return ``(None, None)``, which ``torch.cat``
        cannot take -- so the search is skipped and the verifier pool is never spawned. The
        obs encode still happens, so k=1 gets the cache like every other width.

        This trainer buys nothing at k=1 (there is no search cost to amortize) and its pooled
        ordering is strictly worse than full-dataset epochs. That is a deliberate trade: one
        code path for every k, so the width-1 arm cannot drift from the k=16 arm in anything
        but the width itself.
        """
        cache = _can_cache_obs(policy)
        feats = list()
        window = {'action': list(), 'obs': {k: list() for k in _WINDOW_OBS_KEYS}}
        actions, values = list(), list()
        want_context = policy.max_actions != 1
        was_training = policy.training
        policy.eval()
        try:
            # train_crops(), not eval's centre crop: these features and these subgoals are
            # what the inner updates will actually train on, so they must carry the crop a
            # training step would have drawn. Dropout stays off -- that is the reason for
            # eval() here and it is untouched.
            with policy.train_crops() if cache else contextlib.nullcontext():
                for start in tqdm.trange(0, len(pool), chunk_size, desc="context buffer",
                        leave=False, mininterval=self.cfg.training.tqdm_interval_sec):
                    idxs = pool[start:start + chunk_size]
                    batch = collate([dataset[int(i)] for i in idxs])
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    # Advance the crop step PER CHUNK. The offsets are a pure function of
                    # (seed, step), so without this every chunk of the pool would draw the
                    # identical 32 offsets and the pool would carry only chunk_size distinct
                    # crops instead of outer_batch_size.
                    if hasattr(policy, 'set_crop_step'):
                        policy.set_crop_step(self.cfg.training.seed,
                                             self.global_step + start)
                    # One crop scope per chunk, so the obs encode and every subgoal generated
                    # from it share an offset -- the same guarantee the offline trainer gets
                    # for free by generating the context inside one forward pass.
                    # no_grad: these features are buffered (detached, moved to CPU) and
                    # the search below builds no graph either, so the only thing autograd
                    # could do here is record a graph for the whole pool and throw it
                    # away. Free on the frozen SD-VAE, real memory on a trainable backbone.
                    with torch.no_grad(), policy._crop_scope():
                        chunk_feats = policy._encode_obs_features(batch['obs'])
                        if want_context:
                            chunk_actions, chunk_values = policy.generate_search_context(
                                batch['obs'], obs_features=chunk_feats)
                            actions.append(chunk_actions)
                            values.append(chunk_values)
                    if cache:
                        feats.append(chunk_feats.detach().cpu())
                        # The rest of what an inner step needs from the window, kept
                        # alongside the features so the inner loop can skip the dataset
                        # entirely. `action` is the loss target; agent_pos/feedback are the
                        # verifier's keys, carried so the obs dict stays well-formed for
                        # the k=1 path (which calls generate_search_context with an empty
                        # candidate budget). The IMAGE is deliberately absent -- it is what
                        # `feats` already stands for, and 256 windows of it is the ~450 MB
                        # this exists to stop copying. At 256 windows the three together
                        # are ~330 kB.
                        window['action'].append(batch['action'].detach().cpu())
                        for k in _WINDOW_OBS_KEYS:
                            window['obs'][k].append(batch['obs'][k].detach().cpu())
        finally:
            if was_training:
                policy.train()
        obs_feats = torch.cat(feats, dim=0) if cache else None
        buf_window = None
        if cache:
            buf_window = {
                'action': torch.cat(window['action'], dim=0),
                'obs': {k: torch.cat(v, dim=0) for k, v in window['obs'].items()},
            }
        if not want_context:
            return None, None, obs_feats, buf_window
        return (torch.cat(actions, dim=0), torch.cat(values, dim=0),
                obs_feats, buf_window)

    # ------------------------------------------------------------------ main loop

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # ---- resume ---------------------------------------------------------------
        # Must precede the lr scheduler: rebuilding LambdaLR at last_epoch != -1 requires
        # 'initial_lr' in the optimizer's param_groups, which only exists because the
        # restored optimizer state carries it over from the original construction.
        checkpoint_loaded = False
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                payload = self.load_checkpoint(path=lastest_ckpt_path)
                checkpoint_loaded = True
                # Refuse a pre-EMA payload resumed into a use_ema run. This class overrides
                # run(), so the parent's resume block never executes here -- the guard was
                # simply absent on this path, which matters more than anywhere else now
                # that outer/inner is the DEFAULT search trainer (5294c31) and every
                # argmax arm runs through it. See BaseWorkspace.assert_ema_payload.
                self.assert_ema_payload(payload, lastest_ckpt_path)
            else:
                # Hydra mints a NEW timestamped output dir per launch, so a plain re-submit
                # looks for latest.ckpt somewhere that cannot contain it and starts over.
                # Silent restarts are expensive here; say so loudly.
                print(f"training.resume=True but no checkpoint found at "
                      f"{lastest_ckpt_path} -- STARTING FROM SCRATCH. To continue a "
                      f"previous run, relaunch with hydra.run.dir=<that run's output dir>.")

        # ---- data -----------------------------------------------------------------
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        collate = get_collate_fn() if dataset.return_sequences else default_collate

        # Record the exact episodes this run trains/validates/tests on, and refuse to
        # resume if they differ from what the run directory already recorded. The parent
        # workspace does this inside its own run(); this class overrides run(), so without
        # repeating it here an outer/inner run writes no splits.json and gets no guard --
        # which matters because the demo budget is selected by a COMMAND-LINE override
        # (n_demos + split_file), so a mistyped relaunch would otherwise resume a directory
        # whose checkpoints were built from different episodes. Must follow the instantiate
        # above: the guard needs the resolved partition, not the config keys.
        if self.accelerator.is_main_process:
            self.write_manifest()
            self.write_splits(dataset)

        if checkpoint_loaded and len(self.model.normalizer.params_dict) > 0:
            print("Checkpoint loaded with normalizer - preserving existing normalizer statistics")
        else:
            print("Computing normalizer from dataset")
            normalizer = dataset.get_normalizer()
            self.model.set_normalizer(normalizer)
            if self.ema_model is not None:
                self.ema_model.set_normalizer(normalizer)

        val_dataset = dataset.get_validation_dataset()
        val_collate = get_collate_fn() if dataset.return_sequences else None
        val_dataloader = DataLoader(val_dataset, collate_fn=val_collate, **cfg.val_dataloader)
        test_dataloader = None
        if hasattr(dataset, 'get_test_dataset') and \
                getattr(dataset, 'val_pool', None) is not getattr(dataset, 'test_pool', None):
            test_dataloader = DataLoader(
                dataset.get_test_dataset(), collate_fn=val_collate, **cfg.val_dataloader)

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
        # Extra checkpoints off the `checkpoint_every` cadence, for the early steps where a
        # 10k grid has no resolution. Absolute step numbers, so a resumed run past them does
        # not re-fire (the membership test is on the exact post-increment step).
        extra_ckpt_steps = set(
            int(s) for s in (cfg.training.get('checkpoint_steps', None) or []))

        if cfg.training.debug:
            max_steps = 6
            outer_bs, inner_bs, inner_epochs = 8, 2, 1
            drift_every = 1
            rollout_every_steps = val_every_steps = sample_every_steps = 1
            cfg.training.checkpoint_every = 1
            buffer_chunk = 4

        # updates per outer step; there is deliberately no num_outer constant -- the outer
        # loop runs until the gradient-step budget is spent.
        num_inner = inner_epochs * math.ceil(outer_bs / inner_bs)

        lr_scheduler_kwargs = dict(cfg.training.get('lr_scheduler_kwargs', {}) or {})
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=max_steps,
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
        if cfg.training.use_ema:
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

        train_sampling_batch = None
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with contextlib.ExitStack() as stack:
            json_logger = stack.enter_context(JsonLogger(log_path))
            # the verifier's sim pool and the env runner's env pool are subprocesses that
            # garbage collection will not reap; release both on normal exit and on error.
            stack.callback(self._close_worker_pools, env_runner)

            # One collector instance for the whole run, refreshed in place each outer step.
            # Deep-copying the ResNet18 + transformer per outer step would churn ~100MB of
            # allocations a few thousand times over a 100k-step run for no benefit.
            collector = _clone_policy(self.accelerator.unwrap_model(self.model))
            collector.eval()
            collector.requires_grad_(False)

            while self.global_step < max_steps:
                policy = self.accelerator.unwrap_model(self.model)

                # ======== outer step: freeze a collector, fill the context buffer =====
                # Snapshot the weights that are about to generate the buffer. They are the
                # live weights right now, so this costs nothing beyond the copy -- the
                # buffer is generated by the live policy and the collector preserves the
                # state that did it, for the drift metric to compare against.
                collector.load_state_dict(policy.state_dict())

                # Randomness derived from POSITION (the outer index), never from call
                # count: a resumed run then draws the same pool this outer step would have
                # drawn anyway, instead of replaying the pools of outer steps 0, 1, 2...
                pool_rng = np.random.default_rng([cfg.training.seed, self.epoch])
                pool = pool_rng.choice(
                    len(dataset), size=outer_bs, replace=outer_bs > len(dataset))
                buf_actions, buf_values, buf_feats, buf_window = \
                    self._fill_context_buffer(
                        policy, dataset, collate, pool, device, buffer_chunk)

                # ======== inner: reuse that buffer for inner_epochs passes ============
                self.model.train()
                inner_losses = list()
                inner_idx = 0
                for epoch_idx in range(inner_epochs):
                    order = np.random.default_rng(
                        [cfg.training.seed, self.epoch, epoch_idx]).permutation(len(pool))
                    for start in range(0, len(order), inner_bs):
                        if self.global_step >= max_steps:
                            break
                        sel = order[start:start + inner_bs]
                        want_aux = (inner_idx % drift_every) == 0

                        # THE WINDOW IS ONLY FETCHED WHEN SOMETHING READS IT. With the
                        # features buffered, compute_loss touches nothing but
                        # batch['action'] -- the two branches that read batch['obs'] are
                        # both guarded by `actions is None` / `obs_features is None`, and
                        # both are supplied here. So the dataset fetch, which is a
                        # synchronous single-threaded decode of 32 x horizon images on the
                        # training thread (~113 MB of memcpy), was pure overhead on every
                        # step that is not a drift step.
                        #
                        # Drift steps DO need it: _drift_mse -> predict_epsilon re-encodes
                        # batch['obs'] on both snapshots by design. So does the sampling
                        # batch captured once below. Everything else takes the buffer.
                        need_window = (buf_window is None or want_aux
                                       or train_sampling_batch is None)
                        if need_window:
                            batch = collate([dataset[int(pool[p])] for p in sel])
                            batch = dict_apply(
                                batch, lambda x: x.to(device, non_blocking=True))
                            if train_sampling_batch is None:
                                train_sampling_batch = batch
                        else:
                            sel_cpu = torch.as_tensor(sel, dtype=torch.long)
                            batch = {
                                'action': buf_window['action'][sel_cpu].to(
                                    device, non_blocking=True),
                                'obs': {k: v[sel_cpu].to(device, non_blocking=True)
                                        for k, v in buf_window['obs'].items()},
                            }
                        # device from the batch, not the buffer: at k=1 the buffer is None.
                        sel_t = torch.as_tensor(sel, dtype=torch.long, device=device)
                        # Crop offsets are a pure function of (seed, global_step). This
                        # workspace overrides run(), so without setting it here the step
                        # would stay 0 and every update would see the SAME crop -- no
                        # augmentation at all.
                        #
                        # Only meaningful when the obs is re-encoded here, i.e. when the
                        # pool's features were NOT cached. With the cache the crop is
                        # whatever the buffer fill drew, which is the point: the observation
                        # and the subgoals predicted from it then share one offset, instead
                        # of the buffer holding centre-cropped subgoals against a randomly
                        # cropped obs.
                        if buf_feats is None and hasattr(policy, 'set_crop_step'):
                            policy.set_crop_step(cfg.training.seed, self.global_step)
                        # beside set_crop_step so the crop offset and the
                        # slot-weight curriculum share one notion of 'now'
                        if hasattr(policy, 'set_slot_weight_step'):
                            policy.set_slot_weight_step(self.global_step)
                        result = policy.compute_loss(
                            batch,
                            actions=None if buf_actions is None else buf_actions[sel_t],
                            values=None if buf_values is None else buf_values[sel_t],
                            obs_features=(None if buf_feats is None
                                          else buf_feats[sel_t.cpu()].to(device)),
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
                        if self.accelerator.is_main_process:
                            if self.global_step >= next_ckpt:
                                self._save(cfg)
                            elif self.global_step in extra_ckpt_steps:
                                # off-cadence checkpoint; does NOT move the cadence anchor
                                self._save(cfg, advance_anchor=False)

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
                    # Seed before the rollout so its success rate is reproducible from the
                    # checkpoint. The env is deterministic given a reset state, but
                    # conditional_sample draws from the global RNG, so without this the
                    # number depended on whatever state the preceding training left behind.
                    torch.manual_seed(cfg.training.seed)
                    np.random.seed(cfg.training.seed)
                    step_log.update(env_runner.run(eval_policy))
                    self.last_rollout_step = self.global_step

                if self.global_step - self.last_val_step >= val_every_steps:
                    with torch.no_grad():
                        val_losses = list()
                        for batch_idx, batch in enumerate(val_dataloader):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            val_losses.append(eval_policy.compute_loss(
                                batch,
                                slot_weighting=_val_slot_weighting(eval_policy)).item())
                            if cfg.training.max_val_steps is not None \
                                    and batch_idx >= cfg.training.max_val_steps - 1:
                                break
                        if val_losses:
                            step_log['val_loss'] = float(np.mean(val_losses))
                    if _is_search_policy(eval_policy):
                        nrmse_max_batches = cfg.training.get('nrmse_max_batches', None)
                        for prefix, loader in (('val', val_dataloader), ('test', test_dataloader)):
                            if loader is None:
                                continue
                            metrics = self._search_action_nrmse(
                                eval_policy, loader, device, nrmse_max_batches)
                            if metrics is None:
                                continue
                            for key, value in metrics.items():
                                step_log[f'{prefix}_{key}'] = value
                    self.last_val_step = self.global_step

                if self.global_step - self.last_sample_step >= sample_every_steps \
                        and train_sampling_batch is not None:
                    step_log.update(self._sample_log(
                        eval_policy, policy, collector, train_sampling_batch, device))
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

    def _save(self, cfg, advance_anchor=True):
        """Checkpoint on the gradient-step schedule, with plain (unwrapped) module keys.

        ``advance_anchor=False`` writes the checkpoint WITHOUT moving the cadence anchor.
        That is what off-cadence saves (``training.checkpoint_steps``) use: anchoring on
        them would reschedule the regular grid off them, so a save at 1000 with
        ``checkpoint_every: 10000`` would put the next cadence checkpoint at 11000 instead
        of 10000 and every subsequent one 1000 late -- silently producing a grid that no
        eval sweep expects.
        """
        print(f"Saving checkpoint at step {self.global_step}")
        # Update the cadence anchor BEFORE writing, so the value that lands in the
        # checkpoint is the step this checkpoint was taken at. Setting it afterwards
        # persists the PREVIOUS anchor, so a resumed run believes it is overdue and fires an
        # extra checkpoint on its very first step. (The parent workspace and the UNet one
        # both had it after the write; both now match this.)
        if advance_anchor:
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

    def _sample_log(self, eval_policy, policy, collector, batch, device):
        """Search-quality readout on a training batch, plus the action-space drift."""
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

            pred_action, _, scores, subgoals = eval_policy.search_candidates(
                obs_dict, verifier=eval_policy.verifier,
                n_actions=eval_policy.max_actions,
                return_scores=True, return_subgoals=True)
            pred_action = pred_action[:, :, To - 1:To - 1 + Ta]
            mse = (pred_action - gt_action.unsqueeze(1)).pow(2).mean(dim=(-1, -2))
            log['train_action_mse_error_min'] = mse.min(dim=-1).values.mean().item()
            log['train_action_mse_error_avg'] = mse.mean(dim=-1).mean().item()
            # candidate 0 is generated with an EMPTY search context, i.e. the no-search
            # baseline; (first - min) is the actual best-of-n gain.
            log['train_action_mse_error_first'] = mse[:, 0].mean().item()
            log['train_action_value'] = scores.mean().item()
            log['train_action_value_best'] = scores.max(dim=1).values.mean().item()
            log['train_action_value_first'] = scores[:, 0].mean().item()
            panel = _subgoal_panel(subgoals)
            if panel is not None:
                log['train_subgoals'] = panel

        # action-space drift uses the LIVE policy (the collector's counterpart), not the
        # EMA copy, so it answers the same question as train_drift_mse_eps.
        actions, values = policy.generate_search_context(batch['obs'])
        log['train_drift_action_mse'] = self._drift_action_mse(
            policy, collector, batch,
            {'actions': actions, 'values': values}, seed=self.global_step)
        return log


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainSearchOuterInnerWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
