"""Context-rounds trainer for the PushT DT (asteroid procgen schedule, WEIRDLabUW/asteroid@95eee5e).

Round n = 0..n_rounds-1: a FRESH policy (re-initialised, fresh AdamW, 3% linear warmup + cosine
to 0 over the round) imitates the demos given exactly the n trajectories that the policies of
rounds 0..n-1 made from the same start (`context_obs: state`). Each iteration draws
`starts_per_iter` training episodes with replacement and steps through them in batches. At the
end of the round the policy rolls out once from every training start, conditioned on that start's
context, and the rollout is appended to it. rounds/round_<n>.ckpt and rounds/ctx_round_<n>.pt
are written per round; a restart resumes after the last round that has both.
"""
import contextlib
import copy
import os
import pathlib

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
from diffusion_policy.workspace.train_mlp_image_workspace import TrainMLPImageWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def asteroid_schedule(optimizer, total_steps, warmup_ratio):
    """asteroid's get_optimizer_scheduler: linear warmup from ~0, then cosine to 0."""
    warmup_steps = int(total_steps * warmup_ratio)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_steps])


class TrainDTRoundsWorkspace(TrainMLPImageWorkspace):

    def _first_frames(self, dataset):
        """(E, 1, ...) first-frame obs of every training episode (no images kept beyond frame 0)."""
        items = [dataset[i]['obs'] for i in range(len(dataset))]
        return {k: torch.stack([torch.as_tensor(o[k][:1]) for o in items]) for k in items[0]}

    def _rollout_round(self, env, policy, ctx, states, plans, device, max_steps):
        from eval_dt_pusht import rollout_attempt
        policy.prefill(ctx, device)
        res = rollout_attempt(env, policy, states, plans, device, max_steps)
        return policy.context_records(), res

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        R = cfg.rounds
        device = torch.device(cfg.training.device)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) and dataset.return_sequences
        normalizer = dataset.get_normalizer()
        collate = get_collate_fn()
        n_traj = len(dataset)
        n_rounds, iters, per_iter = R.n_rounds, R.iters_per_round, R.starts_per_iter
        bs = cfg.dataloader.batch_size
        if cfg.training.debug:
            n_rounds, iters, per_iter, bs = 2, 2, 4, 2
        n_updates = iters * (per_iter // bs)
        max_steps = cfg.policy.max_attempt_steps

        out = pathlib.Path(self.output_dir) / 'rounds'
        out.mkdir(parents=True, exist_ok=True)
        ctx, start = [None] * n_traj, 0
        for n in range(n_rounds):
            if (out / f'round_{n}.ckpt').is_file() and (out / f'ctx_round_{n}.pt').is_file():
                ctx, start = torch.load(out / f'ctx_round_{n}.pt'), n + 1
        if start:
            print(f'Resuming after round {start - 1}')
        self.global_step = start * n_updates

        self.model.set_normalizer(normalizer)
        self.model.to(device)
        states, plans = self.model._init_states_and_plans(self._first_frames(dataset))

        from diffusion_policy.env.pusht.pusht_feedback import build_parallel
        env = build_parallel(n_envs=n_traj, use_async=True, n_obs_steps=1, n_action_steps=1,
                             max_episode_steps=max_steps, legacy=False, tracker=True)
        wandb_run = wandb.init(dir=str(self.output_dir),
                               config=OmegaConf.to_container(cfg, resolve=True), **cfg.logging)
        rng = np.random.default_rng(cfg.training.seed + start)
        with contextlib.ExitStack() as stack:
            json_logger = stack.enter_context(JsonLogger(os.path.join(self.output_dir, 'logs.json.txt')))
            stack.callback(env.close)
            for n in range(start, n_rounds):
                torch.manual_seed(cfg.training.seed + n)
                self.model = hydra.utils.instantiate(cfg.policy)
                self.model.set_normalizer(normalizer)
                self.model.to(device)
                self.optimizer = torch.optim.AdamW(
                    trainable_parameters(self.model, type(self).__name__), lr=R.lr)
                sched = asteroid_schedule(self.optimizer, n_updates, R.warmup_ratio)

                self.model.train()
                with tqdm.tqdm(total=n_updates, desc=f'round {n}',
                               mininterval=cfg.training.tqdm_interval_sec) as pbar:
                    for _ in range(iters):
                        pool = rng.choice(n_traj, size=per_iter, replace=True)
                        for u in range(per_iter // bs):
                            sel = pool[u * bs:(u + 1) * bs]
                            batch = collate([dataset[int(i)] for i in sel])
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss = self.model.compute_loss(batch, context=[ctx[i] for i in sel])
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            sched.step()
                            log = {'train_loss': loss.item(), 'lr': sched.get_last_lr()[0],
                                   'round': n, 'global_step': self.global_step}
                            wandb_run.log(log, step=self.global_step)
                            json_logger.log(log)
                            self.global_step += 1
                            pbar.update(1)
                            pbar.set_postfix(loss=log['train_loss'], refresh=False)

                self.model.eval()
                recs, res = self._rollout_round(env, self.model, ctx, states, plans, device, max_steps)
                ctx = [self.model.append_context(c, r) for c, r in zip(ctx, recs)]
                self.epoch = n
                self.save_checkpoint(path=out / f'round_{n}.ckpt', use_thread=False)
                torch.save(ctx, out / f'ctx_round_{n}.pt')
                log = {'round': n, 'global_step': self.global_step,
                       'round_rollout/mean_max_coverage': float(np.mean(res['max_coverage'])),
                       'round_rollout/success_rate': float(np.mean(res['success']))}
                print(f"round {n}: rollout mean max coverage {log['round_rollout/mean_max_coverage']:.3f}")
                wandb_run.log(log, step=self.global_step)
                json_logger.log(log)
