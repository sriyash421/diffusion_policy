"""Env runner for the offline PushT diffusion-search policy.

Differs from ``PushTImageRunner`` in two ways:
1. It rolls out on BOTH the held-out **val** and **test** splits (3-way seeded split),
   with ``val/`` and ``test/`` prefixes.
2. It drives the search policy via ``policy.predict_action_best(obs_dict)`` (argmax over
   the verifier-scored candidates), since the search policy's ``predict_action`` has a
   different signature (``obs_dict, verifier, n_actions``) than the standard one.

Metrics per split: ``mean_score`` (mean max reward), ``success_rate`` (fraction with max
coverage >= threshold, i.e. max reward >= 1.0), and ``T_distance`` (final-step mean
per-keypoint distance of the achieved T from the goal T, read from ``obs['feedback']``).

The best-of-N-search *success curve* (n=1..64) is a separate, test-only artifact produced
by ``eval_search_pusht.py`` -- not here.
"""
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.env.pusht.pusht_feedback import PushTFeedbackWrapper
from diffusion_policy.env.pusht.feedback_util import N_KEYPOINTS
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper, VideoRecorder)
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.pusht_image_dataset import (
    get_split_masks, get_split_masks_3way, get_episode_init_states,
    load_split_manifest, masks_from_manifest)
from diffusion_policy.env_runner.pusht_image_runner import PushTImageRunner


def _feedback_to_tdistance(feedback):
    """(..., 16) goal-vs-achieved keypoint displacement -> (...) mean per-keypoint dist."""
    disp = np.asarray(feedback).reshape(*feedback.shape[:-1], N_KEYPOINTS, 2)
    return np.linalg.norm(disp, axis=-1).mean(axis=-1)


class PushTSearchImageRunner(PushTImageRunner):
    def __init__(self,
            output_dir,
            zarr_path,
            seed=42,
            n_test_episodes=50,
            n_val_episodes=10,
            n_train_episodes=None,
            n_vis=5,
            legacy=False,
            max_steps=300,
            n_obs_steps=2,
            n_action_steps=8,
            n_search_actions=None,
            fps=10,
            crf=22,
            render_size=96,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            split_file=None,
        ):
        # NOTE: intentionally does NOT call PushTImageRunner.__init__ (that one builds a
        # test-only set); we reimplement init to cover val+test. BaseImageRunner.__init__
        # only stores output_dir.
        from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
        BaseImageRunner.__init__(self, output_dir)

        # number of candidates per decision at eval time (best is executed).
        self.n_search_actions = n_search_actions

        replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['agent_pos', 'block_pos'])
        if split_file is not None:
            # Same manifest the dataset reads, so the rollout splits cannot drift from the
            # training splits. Previously both re-derived from the seed independently, which
            # only agreed by convention.
            manifest = load_split_manifest(
                split_file,
                episode_ends=replay_buffer.episode_ends[:],
                expected_counts={
                    'test': n_test_episodes,
                    'val': n_val_episodes if n_val_episodes else None,
                    'train': n_train_episodes,
                })
            _, val_mask, test_mask = masks_from_manifest(
                manifest, replay_buffer.n_episodes)
        elif n_val_episodes > 0:
            _, val_mask, test_mask = get_split_masks_3way(
                n_episodes=replay_buffer.n_episodes,
                n_test_episodes=n_test_episodes,
                n_val_episodes=n_val_episodes,
                seed=seed,
                n_train_episodes=n_train_episodes)
        else:
            _, test_mask = get_split_masks(
                n_episodes=replay_buffer.n_episodes,
                n_test_episodes=n_test_episodes,
                seed=seed,
                n_train_episodes=n_train_episodes)
            val_mask = np.zeros_like(test_mask)

        splits = [('val/', val_mask), ('test/', test_mask)]

        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTFeedbackWrapper(
                        PushTImageEnv(legacy=legacy, render_size=render_size)
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps, codec='h264', input_pix_fmt='rgb24', crf=crf,
                        thread_type='FRAME', thread_count=1),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        for prefix, mask in splits:
            reset_states = get_episode_init_states(replay_buffer, mask)
            episode_idxs = np.nonzero(mask)[0]
            for i in range(len(reset_states)):
                state = reset_states[i]
                episode_idx = int(episode_idxs[i])
                enable_render = i < n_vis

                def init_fn(env, state=state, episode_idx=episode_idx,
                            enable_render=enable_render):
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        env.env.file_path = str(filename)
                    assert isinstance(env.unwrapped, PushTImageEnv)
                    env.unwrapped.reset_to_state = np.asarray(state)
                    assert isinstance(env, MultiStepWrapper)
                    env.seed(episode_idx)

                env_seeds.append(episode_idx)
                env_prefixs.append(prefix)
                env_init_fn_dills.append(dill.dumps(init_fn))

        n_total = len(env_init_fn_dills)
        if n_envs is None:
            n_envs = n_total
        env_fns = [env_fn] * n_envs

        self.env = AsyncVectorEnv(env_fns)
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.n_vis = n_vis
        self.output_dir = output_dir
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    def close(self):
        """Shut down the rollout worker pool (one subprocess per env, held for the whole
        run so rollouts don't pay respawn cost). Called by the training workspace."""
        env = getattr(self, 'env', None)
        if env is not None:
            env.close()
            self.env = None

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_tdistance = [np.nan] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)

            env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            policy.reset()
            last_obs = obs

            pbar = tqdm.tqdm(total=self.max_steps,
                desc=f"Eval PushTSearchRunner {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                obs_dict = dict_apply(dict(obs),
                    lambda x: torch.from_numpy(x).to(device=device))
                with torch.no_grad():
                    if (self.n_search_actions == 1
                            and getattr(policy, 'selection', 'argmax') == 'argmax'):
                        # BC rollout: one sample, executed. Going through
                        # predict_action_best would argmax over a single candidate -- the
                        # same action -- but would still physics-simulate it in the
                        # verifier, so the BC baseline would pay a search cost it does not
                        # use and would spawn a worker pool it never needs.
                        #
                        # Only under 'argmax'. Under 'final_pass' n=1 is NOT the same
                        # action: it is one scored candidate plus the extra conditioned
                        # sample that actually gets executed, so shortcutting it here would
                        # silently evaluate BC and label it as the search arm.
                        action_dict = policy.predict_action(obs_dict)
                    else:
                        action_dict = policy.predict_action_best(
                            obs_dict, n_actions=self.n_search_actions)
                action = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())['action']
                obs, reward, done, info = env.step(action)
                last_obs = obs
                done = np.all(done)
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            # final-step feedback -> T-distance for the active envs in this chunk
            final_feedback = np.asarray(last_obs['feedback'])[this_local_slice, -1]  # (m, 16)
            all_tdistance[this_global_slice] = list(_feedback_to_tdistance(final_feedback))

        _ = env.reset()

        # per-split aggregation
        max_rewards = collections.defaultdict(list)
        successes = collections.defaultdict(list)
        tdistances = collections.defaultdict(list)
        log_data = dict()
        for i in range(n_inits):
            episode_idx = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = float(np.max(all_rewards[i]))
            max_rewards[prefix].append(max_reward)
            successes[prefix].append(1.0 if max_reward >= 1.0 else 0.0)
            tdistances[prefix].append(all_tdistance[i])
            log_data[prefix + f'sim_max_reward_ep{episode_idx}'] = max_reward

        # tile the first n_vis rollouts (per split) side-by-side into one video
        vis_by_prefix = collections.defaultdict(list)
        for i in range(n_inits):
            if all_video_paths[i] is not None:
                vis_by_prefix[self.env_prefixs[i]].append(all_video_paths[i])
        for prefix, paths in vis_by_prefix.items():
            paths = paths[:self.n_vis]
            if paths:
                combined_path = self._tile_videos(paths)
                if combined_path is not None:
                    log_data[prefix + f'sim_video_first{len(paths)}'] = wandb.Video(
                        combined_path, fps=self.fps)

        for prefix in max_rewards:
            log_data[prefix + 'mean_score'] = float(np.mean(max_rewards[prefix]))
            log_data[prefix + 'success_rate'] = float(np.mean(successes[prefix]))
            log_data[prefix + 'T_distance'] = float(np.mean(tdistances[prefix]))

        return log_data
