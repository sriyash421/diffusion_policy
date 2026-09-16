"""Rollout on the manifest's val+test episodes, with the KEYPOINT observation.

THE SAME EPISODES AS THE IMAGE ARMS, WHICH IS THE WHOLE POINT. `PushTKeypointsRunner` -- the
existing lowdim runner -- evaluates on procedurally generated env seeds
(`test_start_seed=100000`), which is a different set of initial states from the held-out
demonstration episodes every image arm is scored on. A keypoint number measured there could not
be put beside a UNet BC number. So this runner is `PushTSearchImageRunner`'s structure with
`PushTKeypointsEnv` in it: the same manifest, the same `get_episode_init_states`, the same
`env.seed(episode_idx)`, the same `val/` + `test/` prefixes and the same 300 steps. Episode for
episode, it is the same benchmark.

WHY IT CANNOT SIMPLY SUBCLASS PushTSearchImageRunner'S `run`. Four incompatibilities, all
because that runner assumes a Dict observation from an image env:

  1. `PushTKeypointsEnv`'s observation is a flat 40-d Box -- 9 block keypoints (18) + agent xy
     (2), then the same 20 again as a visibility mask -- so `dict_apply(dict(obs), ...)` raises
     on an ndarray. This runner slices the value half and splits it into the {keypoint,
     agent_pos} dict the dataset emits, so the policy sees at eval exactly what it was
     trained on.
  2. `PushTFeedbackWrapper` does `dict(self.env.observation_space.spaces)`, and a Box has no
     `.spaces`. It cannot wrap this env, so `obs['feedback']` does not exist.
  3. Hence the final-step distances come from `info['block_pose']` / `info['pos_agent']`
     (PushTEnv._get_info) instead, put through the SAME `feedback_util` functions the image
     runner uses. One definition of each metric, so the two arms' numbers are comparable.
     NOTE the two keypoint sets are different and must never be cross-wired: `feedback` is the
     8 canonical T vertices from feedback_util, while the observation's 9 points are
     farthest-point samples from PymunkKeypointManager.
  4. `PushTSearchImageRunner.run` asserts `isinstance(env.unwrapped, PushTImageEnv)`.

`_tile_videos` is inherited (hence the PushTImageRunner base), because that is pure video
plumbing with nothing image-observation-specific in it.
"""
import collections
import math
import pathlib

import dill
import numpy as np
import torch
import tqdm
import wandb
import wandb.sdk.data_types.video as wv

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.pusht_image_dataset import (
    get_episode_init_states, load_split_manifest, masks_from_manifest)
from diffusion_policy.env.pusht.feedback_util import (
    arm_to_t_distance, compute_feedback_from_pose, t_goal_distance)
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env_runner.pusht_image_runner import PushTImageRunner
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv, force_close
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecorder, VideoRecordingWrapper)
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# 9 block keypoints; the value half of the observation is 9*2 + agent xy
N_BLOCK_KEYPOINTS = 9
OBS_VALUE_DIM = N_BLOCK_KEYPOINTS * 2 + 2


class PushTSearchKeypointsRunner(PushTImageRunner):
    def __init__(self,
            output_dir,
            zarr_path,
            seed=42,
            n_test_episodes=50,
            n_val_episodes=30,
            n_train_episodes=None,
            n_vis=5,
            legacy=False,
            max_steps=300,
            n_obs_steps=8,
            n_action_steps=8,
            n_search_actions=1,
            fps=10,
            crf=22,
            render_size=96,
            keypoint_visible_rate=1.0,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            split_file=None,
        ):
        # Deliberately NOT PushTImageRunner.__init__ (that builds an image env on a test-only
        # 2-way split); BaseImageRunner.__init__ only stores output_dir.
        BaseImageRunner.__init__(self, output_dir)

        # Accepted for config symmetry with the image arms, whose task block interpolates it.
        # This arm has no verifier, so there is nothing to search over and nothing that could
        # rank candidates -- a value > 1 would silently be ignored.
        assert n_search_actions in (None, 1), (
            f'n_search_actions={n_search_actions}: the LSTM BC arm has no verifier, so '
            f'best-of-n is not defined for it. Pass 1.')

        replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['agent_pos', 'block_pos'])
        # REQUIRED, matching the dataset. Two independent derivations of "which episodes are
        # val" agreeing by convention is what the manifest exists to replace.
        if split_file is None:
            raise ValueError(
                'PushTSearchKeypointsRunner requires split_file: it must roll out on the '
                'same manifest the dataset trained on. Pass ${task.dataset.split_file}.')
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
        splits = [('val/', val_mask), ('test/', test_mask)]

        # The SAME keypoint definition that generated the zarr's `keypoint` array. If these
        # ever diverge, train and eval observations are different quantities and every success
        # rate here is meaningless -- which is why test_keypoint_obs_matches_zarr exists.
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsEnv(
                        legacy=legacy,
                        render_size=render_size,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        **kp_kwargs),
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
                    assert isinstance(env.unwrapped, PushTKeypointsEnv)
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
        """force_close, not env.close(): a plain close() drains the in-flight call with no
        timeout, so one dead worker wedges teardown permanently."""
        env = getattr(self, 'env', None)
        if env is not None:
            self.env = None
            force_close(env)

    @staticmethod
    def _obs_to_dict(obs):
        """The 40-d Box -> the {keypoint, agent_pos} dict the dataset emits.

        The value half is `kps.flatten()` then `agent_pos` (PushTKeypointsEnv._get_obs), so
        the first 18 entries reshape row-major back to (9, 2). The second half is the
        visibility mask, dropped: this arm never occludes, so it is a constant and the policy
        has no input for it.
        """
        value = obs[..., :OBS_VALUE_DIM]
        keypoint = value[..., :N_BLOCK_KEYPOINTS * 2].reshape(
            *value.shape[:-1], N_BLOCK_KEYPOINTS, 2)
        return {
            'keypoint': keypoint.astype(np.float32),
            'agent_pos': value[..., N_BLOCK_KEYPOINTS * 2:].astype(np.float32),
        }

    @staticmethod
    def _final_states(infos, local_slice):
        """(agent_pos, feedback) at the last step, from PushTEnv._get_info.

        `infos` is a tuple of per-env dicts whose values MultiStepWrapper has stacked over the
        last n_obs_steps, so `[-1]` is the final step. `feedback` is reconstructed from the
        block pose with the same function the dataset uses, so the distances below are the
        image arm's metrics and not a second definition of them.
        """
        infos = list(infos)[local_slice]
        agent = np.stack([np.asarray(i['pos_agent'])[-1] for i in infos])       # (n, 2)
        pose = np.stack([np.asarray(i['block_pose'])[-1] for i in infos])       # (n, 3)
        return agent, compute_feedback_from_pose(pose.astype(np.float32))       # (n, 16)

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_t_goal_dist = [np.nan] * n_inits
        all_arm_t_dist = [np.nan] * n_inits

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
            # zeroes the LSTM hidden state -- the episode boundary for a recurrent policy
            policy.reset()
            last_info = None

            pbar = tqdm.tqdm(total=self.max_steps,
                desc=f"Eval PushTSearchKeypointsRunner {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                obs_dict = dict_apply(self._obs_to_dict(obs),
                    lambda x: torch.from_numpy(x).to(device=device))
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                action = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())['action']
                obs, reward, done, info = env.step(action)
                last_info = info
                done = np.all(done)
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            final_agent, final_feedback = self._final_states(last_info, this_local_slice)
            all_t_goal_dist[this_global_slice] = list(t_goal_distance(final_feedback))
            all_arm_t_dist[this_global_slice] = list(
                arm_to_t_distance(final_agent, final_feedback))

        _ = env.reset()

        max_rewards = collections.defaultdict(list)
        successes = collections.defaultdict(list)
        t_goal_dists = collections.defaultdict(list)
        arm_t_dists = collections.defaultdict(list)
        log_data = dict()
        for i in range(n_inits):
            episode_idx = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = float(np.max(all_rewards[i]))
            max_rewards[prefix].append(max_reward)
            successes[prefix].append(1.0 if max_reward >= 1.0 else 0.0)
            t_goal_dists[prefix].append(all_t_goal_dist[i])
            arm_t_dists[prefix].append(all_arm_t_dist[i])
            log_data[prefix + f'sim_max_reward_ep{episode_idx}'] = max_reward

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

        # Same metric names as the image runner, so the two arms overlay on one panel.
        for prefix in max_rewards:
            log_data[prefix + 'mean_score'] = float(np.mean(max_rewards[prefix]))
            log_data[prefix + 'success_rate'] = float(np.mean(successes[prefix]))
            t_goal = float(np.mean(t_goal_dists[prefix]))
            arm_t = float(np.mean(arm_t_dists[prefix]))
            log_data[prefix + 'T_goal_distance'] = t_goal
            log_data[prefix + 'arm_T_distance'] = arm_t
            log_data[prefix + 'verifier_distance'] = t_goal + arm_t

        return log_data
