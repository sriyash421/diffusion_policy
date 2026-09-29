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
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.pusht_image_dataset import (
    get_split_masks, get_episode_init_states)
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class PushTImageRunner(BaseImageRunner):
    """Rolls out the policy from the initial states of the held-out test episodes.

    Every rollout resets to a recorded test-episode initial state rather than to a
    random seed, so the eval distribution is exactly the held-out split. seed and
    n_test_episodes must match the dataset's so the split is identical.
    """
    def __init__(self,
            output_dir,
            zarr_path,
            seed=42,
            n_test_episodes=50,
            n_train_episodes=None,
            n_vis=4,
            legacy=False,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22,
            render_size=96,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        # recover the same test episodes the dataset holds out, then take the
        # initial [agent_x, agent_y, block_x, block_y, block_angle] of each.
        replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['agent_pos', 'block_pos'])
        _, test_mask = get_split_masks(
            n_episodes=replay_buffer.n_episodes,
            n_test_episodes=n_test_episodes,
            seed=seed,
            n_train_episodes=n_train_episodes)
        reset_states = get_episode_init_states(replay_buffer, test_mask)
        episode_idxs = np.nonzero(test_mask)[0]
        n_test = len(reset_states)

        if n_envs is None:
            n_envs = n_test

        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    # feedback is computed live from the pymunk body, matching the
                    # signal the dataset derives from block_pos
                    PushTFeedbackWrapper(
                        PushTImageEnv(
                            # legacy=True sets block position before angle, and rotating
                            # about the CoM then moves the position ~90px off the state we
                            # asked for. legacy=False round-trips a recorded state exactly.
                            legacy=legacy,
                            render_size=render_size
                        )
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # test: one rollout per held-out test episode, reset to its initial state
        for i in range(n_test):
            state = reset_states[i]
            episode_idx = int(episode_idxs[i])
            enable_render = i < n_vis

            def init_fn(env, state=state, episode_idx=episode_idx,
                        enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # reset to this test episode's recorded initial state.
                # must go through .unwrapped: gym wrappers forward attribute reads but
                # not writes, so setting this on a wrapper would silently be ignored.
                assert isinstance(env.unwrapped, PushTImageEnv)
                env.unwrapped.reset_to_state = np.asarray(state)

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(episode_idx)

            env_seeds.append(episode_idx)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
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
    
    def _tile_videos(self, paths):
        """Tile per-episode rollout videos side-by-side into a single mp4.

        Rollouts can end at different lengths, so shorter clips are padded with
        their final frame to keep the tiles in sync.
        """
        import imageio.v2 as iio
        try:
            clips = list()
            for p in paths:
                reader = iio.get_reader(p)
                clips.append(np.stack([frame for frame in reader]))
                reader.close()
            n_frames = max(len(c) for c in clips)
            padded = list()
            for c in clips:
                if len(c) < n_frames:
                    tail = np.repeat(c[-1:], n_frames - len(c), axis=0)
                    c = np.concatenate([c, tail], axis=0)
                padded.append(c)
            grid = np.concatenate(padded, axis=2)  # along width

            out_path = str(pathlib.Path(self.output_dir).joinpath(
                'media', wv.util.generate_id() + '_first%d.mp4' % len(paths)))
            writer = iio.get_writer(out_path, fps=self.fps)
            for frame in grid:
                writer.append_data(frame)
            writer.close()
            return out_path
        except Exception as e:
            # a broken video must not take down a training run
            print(f'[PushTImageRunner] failed to tile rollout videos: {e}')
            return None

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtImageRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            episode_idx = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_ep{episode_idx}'] = max_reward

        # tile the first n_vis rollouts side-by-side into one video
        vis_paths = [p for p in all_video_paths[:self.n_vis] if p is not None]
        if len(vis_paths) > 0:
            combined_path = self._tile_videos(vis_paths)
            if combined_path is not None:
                log_data[f'test/sim_video_first{len(vis_paths)}'] = wandb.Video(
                    combined_path, fps=self.fps)

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
