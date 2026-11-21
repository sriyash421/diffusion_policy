"""Multi-episode DAgger trainer for the PushT DT (asteroid's train_asteroid.py loop, on the demos).

Stage n = 0..n_stages-1 (pi_kn): the weights continue from pi_k(n-1); AdamW and asteroid's
schedule (3% linear warmup, cosine to 0) restart. Each of `iters_per_stage` DAgger iterations:
the CURRENT weights roll out n attempts in a row, `repeats` rollouts from every training start in
parallel (attempt j sees attempts < j; `context_obs: state`), then `epochs` passes over those
rollouts -- each paired with its start's demo -- every use drawing a
context width w ~ U{0..n} -- the first w attempts, then the demo's NLL. The uniform width stands
in for asteroid training step k on all steps' data merged. stages/pi_k<n>.ckpt per stage; a
restart resumes after the last completed stage.
"""
import contextlib
import copy
import os
import pathlib
import time

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf

from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, trainable_parameters
from diffusion_policy.common.sampler import get_collate_fn
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.workspace.train_dt_rounds_workspace import TrainDTRoundsWorkspace, asteroid_schedule

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDTDaggerWorkspace(TrainDTRoundsWorkspace):

    def _collect(self, env, policy, n, states, plans, device, max_steps):
        """n attempts in a row from every start by the current weights -> per-start context."""
        from eval_dt_pusht import rollout_attempt
        ctx, cov = [None] * len(states), []
        policy.reset()
        for _ in range(n):
            res = rollout_attempt(env, policy, states, plans, device, max_steps)
            ctx = [policy.append_context(c, r) for c, r in zip(ctx, policy.context_records())]
            cov.append(float(np.mean(res['max_coverage'])))
        policy.reset()
        return ctx, cov

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        D = cfg.dagger
        device = torch.device(cfg.training.device)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) and dataset.return_sequences
        normalizer = dataset.get_normalizer()
        collate = get_collate_fn()
        n_traj = len(dataset)
        n_stages, iters, epochs, bs = D.n_stages, D.iters_per_stage, D.epochs, cfg.dataloader.batch_size
        if cfg.training.debug:
            n_stages, iters, epochs = 3, 2, 1
        rows = np.repeat(np.arange(n_traj), D.repeats)       # rollout row -> training episode
        n_rows = len(rows)
        per_epoch = -(-n_rows // bs)
        n_updates = iters * epochs * per_epoch
        max_steps = cfg.policy.max_attempt_steps

        out = pathlib.Path(self.output_dir) / 'stages'
        out.mkdir(parents=True, exist_ok=True)
        start = 0
        while (out / f'pi_k{start}.ckpt').is_file():
            start += 1
        self.model.set_normalizer(normalizer)
        if start:
            print(f'Resuming after stage {start - 1}')
            self.load_checkpoint(path=out / f'pi_k{start - 1}.ckpt')
        self.model.to(device)
        self.global_step = start * n_updates
        states, plans = self.model._init_states_and_plans(self._first_frames(dataset))
        states, plans = states[rows], [plans[i] for i in rows]

        from diffusion_policy.env.pusht.pusht_feedback import build_parallel
        env = build_parallel(n_envs=n_rows, use_async=True, n_obs_steps=1, n_action_steps=1,
                             max_episode_steps=max_steps, legacy=False, tracker=True)
        wandb_run = wandb.init(dir=str(self.output_dir),
                               config=OmegaConf.to_container(cfg, resolve=True), **cfg.logging)
        rng = np.random.default_rng(cfg.training.seed + start)
        torch.manual_seed(cfg.training.seed + start)
        with contextlib.ExitStack() as stack:
            json_logger = stack.enter_context(JsonLogger(os.path.join(self.output_dir, 'logs.json.txt')))
            stack.callback(env.close)
            for n in range(start, n_stages):
                self.optimizer = torch.optim.AdamW(
                    trainable_parameters(self.model, type(self).__name__), lr=D.lr)
                sched = asteroid_schedule(self.optimizer, n_updates, D.warmup_ratio)
                with tqdm.tqdm(total=n_updates, desc=f'pi_k{n}',
                               mininterval=cfg.training.tqdm_interval_sec) as pbar:
                    for it in range(iters):
                        ctx = [None] * n_rows
                        if n > 0:
                            self.model.eval()
                            t0 = time.time()
                            ctx, cov = self._collect(env, self.model, n, states, plans, device, max_steps)
                            log = {'stage': n, 'iteration': it, 'global_step': self.global_step,
                                   'collect/seconds': time.time() - t0,
                                   **{f'collect/attempt{j}_mean_max_coverage': c for j, c in enumerate(cov)}}
                            wandb_run.log(log, step=self.global_step)
                            json_logger.log(log)
                        self.model.train()
                        for _ in range(epochs):
                            order = rng.permutation(n_rows)
                            for u in range(per_epoch):
                                sel = order[u * bs:(u + 1) * bs]
                                batch = collate([dataset[int(rows[r])] for r in sel])
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                context = [self.model.truncate_context(ctx[r], int(rng.integers(0, n + 1)))
                                           for r in sel]
                                loss = self.model.compute_loss(batch, context=context)
                                self.optimizer.zero_grad()
                                loss.backward()
                                self.optimizer.step()
                                sched.step()
                                log = {'train_loss': loss.item(), 'lr': sched.get_last_lr()[0],
                                       'stage': n, 'global_step': self.global_step}
                                wandb_run.log(log, step=self.global_step)
                                json_logger.log(log)
                                self.global_step += 1
                                pbar.update(1)
                                pbar.set_postfix(loss=log['train_loss'], refresh=False)
                self.model.eval()
                self.epoch = n
                self.save_checkpoint(path=out / f'pi_k{n}.ckpt', use_thread=False)
