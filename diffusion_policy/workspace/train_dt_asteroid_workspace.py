"""In-context DAgger for the PushT DT, as the maze asteroid.py (train_asteroid, steps-per-iter schedule).

  D = {(no context, demo) for each training start}
  for k = 0 .. n_stages-1 (pi_kk; weights carried over):
      if k > 0: the current weights make k attempts in a row from every training start,
                `chains_per_start` times; D += {(those k attempts, that start's demo)}
      fresh AdamW (constant lr, or warmup + cosine to 0 over the stage); repeat `iters_per_stage`:
          draw a batch of `batch` items from D (cycling permutations), take `steps_per_iter` gradient
          steps on it (micro-batches, token-weighted, grad clip); every `eval_every` iterations
          evaluate the current weights on train and test starts: k own attempts, attempt k+1
          scored (attempts 1..k+1 logged).
Loss: demo NLL after the item's context. Optional training-only noise on the demo's image features
(`obs_noise`: flat t, or a ramp over the context width). `collect: false` + `reset_optimizer: false` is the
no-context control: D stays the demos and one AdamW runs on, so pi_k<k> = (k+1) x iters x steps updates. stages/pi_k<k>.ckpt + items_k<k>.pt per
stage; a restart resumes after the last stage with both, or mid-stage from stages/partial.pt
(written at every eval).
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
from diffusion_policy.workspace.train_dt_dagger_workspace import TrainDTDaggerWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def obs_noise_t(A, n, n_stages):
    """DDPM timestep of the feature noise for an item with n context attempts; None = clean."""
    if A.obs_noise == 'flat':
        return int(A.obs_noise_t)
    if A.obs_noise == 'ramp':
        t0, t1 = A.obs_noise_ramp
        return int(round(t0 + (t1 - t0) * n / max(1, n_stages - 1)))
    assert A.obs_noise == 'none', A.obs_noise
    return None


class TrainDTAsteroidWorkspace(TrainDTDaggerWorkspace):

    def _eval_point(self, env, policy, k, states, plans, n_train, device, max_steps):
        """Current weights from train + test starts: k own attempts, then attempt k+1.
        Returns {split: [mean max coverage per attempt]} and the scored attempt's success."""
        from eval_dt_pusht import rollout_attempt
        was = policy.training
        policy.eval(); policy.reset()
        cov, succ = {'train': [], 'test': []}, {}
        for _ in range(k + 1):
            res = rollout_attempt(env, policy, states, plans, device, max_steps)
            for sp, sl in (('train', slice(0, n_train)), ('test', slice(n_train, None))):
                cov[sp].append(round(float(np.mean(res['max_coverage'][sl])), 4))
                succ[sp] = round(float(np.mean(res['success'][sl])), 4)
        policy.reset(); policy.train(was)
        return cov, succ

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        A = cfg.asteroid
        device = torch.device(cfg.training.device)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) and dataset.return_sequences
        normalizer = dataset.get_normalizer()
        collate = get_collate_fn()
        n_traj = len(dataset)
        n_stages, iters, steps = A.n_stages, A.iters_per_stage, A.steps_per_iter
        if cfg.training.debug:
            n_stages, iters, steps = 3, 4, 2
        max_steps = cfg.policy.max_attempt_steps
        noise_t = [obs_noise_t(A, n, A.n_stages) for n in range(A.n_stages)]

        out = pathlib.Path(self.output_dir) / 'stages'
        out.mkdir(parents=True, exist_ok=True)
        self.model.set_normalizer(normalizer)
        items, start = [(i, None) for i in range(n_traj)], 0
        while (out / f'pi_k{start}.ckpt').is_file() and (out / f'items_k{start}.pt').is_file():
            start += 1
        if start:
            print(f'Resuming after stage {start - 1}')
            self.load_checkpoint(path=out / f'pi_k{start - 1}.ckpt')
            items = torch.load(out / f'items_k{start - 1}.pt')
        partial = torch.load(out / 'partial.pt') if (out / 'partial.pt').is_file() else None
        if partial is not None and partial['k'] != start:
            partial = None
        if partial is not None:
            print(f"Resuming stage {start} at iteration {partial['it']}")
            self.model.load_state_dict(partial['model'])
            items = partial['items']
        self.model.to(device)
        self.global_step = start * iters * steps if partial is None else partial['global_step']

        # collection rows: every training start, chains_per_start times
        rows = np.repeat(np.arange(n_traj), A.chains_per_start)
        states, plans = self.model._init_states_and_plans(self._first_frames(dataset))
        c_states, c_plans = states[rows], [plans[i] for i in rows]
        # eval rows: train starts then test starts, eval_seeds each
        from diffusion_policy.env.pusht.veritas.tracker import WAYPOINT_PLAN_DIRS, load_split_plans
        from eval_dt_pusht import FEEDBACK_VALUE
        from eval_search_pusht import get_split_states
        e_states, e_plans = [], []
        for split in ('train', 'test'):
            st, idxs = get_split_states(cfg, split, run_dir=pathlib.Path(self.output_dir))
            pl = load_split_plans(WAYPOINT_PLAN_DIRS[FEEDBACK_VALUE], idxs, split, FEEDBACK_VALUE)
            e_states += [s for s in st for _ in range(A.eval_seeds)]
            e_plans += [p for p in pl for _ in range(A.eval_seeds)]
            if split == 'train':
                n_train = len(e_states)

        from diffusion_policy.env.pusht.pusht_feedback import build_parallel
        mk = lambda n: build_parallel(n_envs=n, use_async=True, n_obs_steps=1, n_action_steps=1,
                                      max_episode_steps=max_steps, legacy=False, tracker=True)
        wandb_run = wandb.init(dir=str(self.output_dir),
                               config=OmegaConf.to_container(cfg, resolve=True), **cfg.logging)
        rng = np.random.default_rng(cfg.training.seed + start)
        torch.manual_seed(cfg.training.seed + start)
        if partial is not None:
            rng.bit_generator.state = partial['np_rng']
            torch.set_rng_state(partial['torch_rng'])
        with contextlib.ExitStack() as stack:
            json_logger = stack.enter_context(JsonLogger(os.path.join(self.output_dir, 'logs.json.txt')))
            c_env = mk(len(rows)); stack.callback(c_env.close)
            e_env = mk(len(e_states)); stack.callback(e_env.close)

            def log(d):
                wandb_run.log(d, step=self.global_step)
                json_logger.log(d)

            def eval_log(k, it, u):
                t0 = time.time()
                cov, succ = self._eval_point(e_env, self.model, k, e_states, e_plans, n_train, device, max_steps)
                log({'stage': k, 'iteration': it, 'update': u, 'global_step': self.global_step,
                     'eval/seconds': time.time() - t0,
                     **{f'eval/{sp}_coverage': cov[sp][-1] for sp in cov},
                     **{f'eval/{sp}_success': succ[sp] for sp in succ},
                     **{f'eval/{sp}_coverage_attempt{j + 1}': v for sp in cov for j, v in enumerate(cov[sp])}})

            for k in range(start, n_stages):
                resume = partial if k == start else None
                if k > 0 and resume is None and A.collect:
                    t0 = time.time()
                    self.model.eval()
                    ctx, cov = self._collect(c_env, self.model, k, c_states, c_plans, device, max_steps)
                    items += [(int(rows[r]), ctx[r]) for r in range(len(rows))]
                    log({'stage': k, 'global_step': self.global_step, 'collect/seconds': time.time() - t0,
                         'collect/items': len(items),
                         **{f'collect/attempt{j}_mean_max_coverage': c for j, c in enumerate(cov)}})
                self.model.train()
                # reset_optimizer=false: one AdamW for the whole run; after a restart it is the one saved in
                # pi_k<start-1>.ckpt (load_checkpoint) or partial.pt
                if A.reset_optimizer or k == 0:
                    self.optimizer = torch.optim.AdamW(
                        trainable_parameters(self.model, type(self).__name__), lr=A.lr,
                        betas=tuple(A.betas), eps=A.eps, weight_decay=A.weight_decay)
                else:
                    assert A.lr_schedule == 'constant', 'a per-stage schedule needs reset_optimizer'
                    # the checkpoint's state was loaded before the model moved to `device`: re-cast it
                    self.optimizer.load_state_dict(self.optimizer.state_dict())
                params = [p for g in self.optimizer.param_groups for p in g['params']]
                sched = None
                if A.lr_schedule == 'cosine':
                    from diffusion_policy.model.common.lr_scheduler import get_scheduler
                    sched = get_scheduler('cosine', self.optimizer, num_warmup_steps=A.warmup,
                                          num_training_steps=iters * steps)
                else:
                    assert A.lr_schedule == 'constant', A.lr_schedule
                order, it0 = [], 0
                if resume is not None:
                    self.optimizer.load_state_dict(resume['optimizer'])
                    if sched is not None:
                        sched.load_state_dict(resume['sched'])
                    order, it0 = resume['order'], resume['it']
                else:
                    eval_log(k, 0, 0)

                def save_partial(it):
                    torch.save({'k': k, 'it': it, 'global_step': self.global_step, 'items': items,
                                'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                                'sched': None if sched is None else sched.state_dict(), 'order': order,
                                'np_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state()},
                               out / 'partial.tmp')
                    os.replace(out / 'partial.tmp', out / 'partial.pt')

                with tqdm.tqdm(total=iters * steps, initial=it0 * steps, desc=f'pi_k{k}',
                               mininterval=cfg.training.tqdm_interval_sec) as pbar:
                    for it in range(it0, iters):
                        while len(order) < A.batch:
                            order += list(rng.permutation(len(items)))
                        sel, order = order[:A.batch], order[A.batch:]
                        # the same batch for every step of this iteration: collate and move once
                        micro = []
                        for j in range(0, len(sel), A.micro):
                            part = [items[i] for i in sel[j:j + A.micro]]
                            ctx = [c for _, c in part]
                            t = [noise_t[0 if c is None else len(c['att_lens'])] for c in ctx]
                            b = dict_apply(collate([dataset[e] for e, _ in part]), lambda x: x.to(device))
                            micro.append((b, ctx, None if t[0] is None else torch.tensor(t, device=device)))
                        tokens = sum(float(b['attention_mask'].sum()) for b, _, _ in micro)
                        for _ in range(steps):
                            self.optimizer.zero_grad()
                            tot = 0.
                            for b, ctx, t in micro:
                                w = float(b['attention_mask'].sum()) / tokens
                                loss = self.model.compute_loss(b, context=ctx, feat_noise_t=t)
                                (loss * w).backward()
                                tot += loss.item() * w
                            gn = torch.nn.utils.clip_grad_norm_(params, A.grad_clip)
                            self.optimizer.step()
                            lr = self.optimizer.param_groups[0]['lr']
                            if sched is not None:
                                sched.step()
                            log({'train_loss': tot, 'grad_norm': float(gn), 'lr': lr, 'stage': k,
                                 'global_step': self.global_step})
                            self.global_step += 1
                            pbar.update(1)
                            pbar.set_postfix(loss=tot, refresh=False)
                        if (it + 1) % A.eval_every == 0 or it + 1 == iters:
                            eval_log(k, it + 1, (it + 1) * steps)
                            if it + 1 < iters:
                                save_partial(it + 1)
                self.model.eval()
                self.epoch = k
                self.save_checkpoint(path=out / f'pi_k{k}.ckpt', use_thread=False)
                torch.save(items, out / f'items_k{k}.pt')
                (out / 'partial.pt').unlink(missing_ok=True)
