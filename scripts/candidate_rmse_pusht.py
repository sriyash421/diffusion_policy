"""Per-candidate RMSE to the expert and verifier value, in generation order, on dataset windows.

The PushT counterpart of the maze report's "Metrics on generated candidates" (chunked): at a
demonstration window the expert's next action chunk is the training target, so each of a
64-candidate stream (each candidate conditioned on the previous ones and their values, with the
rolling context window the rollout search uses: `predict_n_actions`) can be scored against it. RMSE is the
workspaces' `nrmse`: normalized action space, over the n_action_steps the policy executes.
N = 4/8/16/64 are prefixes of the one stream. Rollout states drift off the demos after the
first decision, so closed-loop candidates have no expert to compare to; windows do.

`--split val` reads the val of `--split-file` (the run's valpool50 manifest: same train and
test). Writes <out>/candidates.npz: rmse, value (W, n), episode, window.

    python scripts/candidate_rmse_pusht.py -c <run>/checkpoints/step_0010000.ckpt --split test \
        --out <dir> [--split-file <manifest>] [--windows 256] [--n 64]
"""
import pathlib
import sys

import click
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import hydra  # noqa: E402
from torch.utils.data.dataloader import default_collate  # noqa: E402
from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy.common.sampler import get_collate_fn  # noqa: E402
from diffusion_policy.workspace.train_search_outer_inner_workspace import (  # noqa: E402
    TrainSearchOuterInnerWorkspace)
import eval_search_pusht as ev  # noqa: E402


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('--split', type=click.Choice(['val', 'test']), required=True)
@click.option('--split-file', default=None)
@click.option('--out', required=True)
@click.option('--windows', type=int, default=256)
@click.option('--n', 'n_cand', type=int, default=64)
@click.option('--batch', type=int, default=32)
@click.option('--seed', type=int, default=0)
def main(checkpoint, split, split_file, out, windows, n_cand, batch, seed):
    device = 'cuda:0'
    policy, cfg = ev.load_policy(checkpoint, device)
    ev.override_terminal_scoring(policy, True)   # as the report's rollouts score
    if split_file:
        cfg.task.dataset.split_file = split_file
        cfg.task.dataset.n_val_episodes = None   # the manifest's own count
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if split_file:   # the swap is only valid while train and test are untouched
        ev.get_split_states(cfg, 'test', split_file=split_file)
    part = dataset.get_validation_dataset() if split == 'val' else dataset.get_test_dataset()
    store = TrainSearchOuterInnerWorkspace._build_waypoint_store(None, cfg, dataset)
    collate = get_collate_fn() if dataset.return_sequences else default_collate
    idx = np.sort(np.random.default_rng(seed).choice(len(part), size=min(windows, len(part)), replace=False))
    To, Ta = policy.n_obs_steps, policy.n_action_steps
    sl = slice(To - 1, To - 1 + Ta)
    norm = policy.normalizer['action']
    torch.manual_seed(seed)
    rmse, value, episode = [], [], []
    with torch.no_grad():
        for b in range(0, len(idx), batch):
            ids = idx[b:b + batch]
            if store is not None:
                policy.set_waypoint_trackers([store.for_dataset_index(part, int(i)) for i in ids])
            bt = dict_apply(collate([part[int(i)] for i in ids]), lambda x: x.to(device))
            obs = dict(bt['obs'])
            if bt.get('attention_mask', None) is not None:
                obs['attention_mask'] = bt['attention_mask']
            act, _, scores = policy.predict_n_actions(obs, verifier=policy.verifier, n_actions=n_cand,
                                                      return_scores=True)
            d = norm.normalize(act[:, :, sl]) - norm.normalize(bt['action'][:, sl]).unsqueeze(1)
            rmse.append(d.pow(2).mean(dim=(-1, -2)).sqrt().cpu().numpy())
            value.append(scores.float().cpu().numpy())
            episode += [int(np.searchsorted(dataset.replay_buffer.episode_ends[:], part.sampler.indices[int(i)][0],
                                            side='right')) for i in ids]
            print(f'{b + len(ids)}/{len(idx)} windows', flush=True)
    close = getattr(policy, 'close', None)
    if close:
        close()
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'candidates.npz', rmse=np.concatenate(rmse), value=np.concatenate(value),
             episode=np.array(episode), window=idx, checkpoint=checkpoint, split=split,
             split_file=str(split_file))
    print('wrote', out / 'candidates.npz')


if __name__ == '__main__':
    main()
