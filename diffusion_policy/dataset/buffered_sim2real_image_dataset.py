from typing import Dict, List, Optional, Tuple
import os
import glob
import threading
import queue
import math
import random
import time
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.model.common.normalizer import (
	LinearNormalizer, SingleFieldLinearNormalizer
)
from diffusion_policy.dataset.sim2real_image_dataset import (
	discover_zarr_files, Sim2RealImageDataset
)


def _detect_distributed_rank_world() -> Tuple[int, int]:
	"""
	Detect rank/world_size from torch.distributed or environment variables
	commonly set by accelerate/torchrun. Fallback to (0,1).
	"""
	try:
		import torch.distributed as dist
		if dist.is_available() and dist.is_initialized():
			return dist.get_rank(), dist.get_world_size()
	except Exception:
		pass
	# Try env vars (Accelerate, torchrun)
	for rk_key, ws_key in [("RANK", "WORLD_SIZE"), ("LOCAL_RANK", "LOCAL_WORLD_SIZE")]:
		if rk_key in os.environ and ws_key in os.environ:
			try:
				return int(os.environ[rk_key]), int(os.environ[ws_key])
			except Exception:
				continue
	return 0, 1


class EpisodeRecord:
	"""
	Container for a single episode's data.
	Stores arrays per key to enable window sampling without re-materialization.
	"""
	__slots__ = [
		"obs",             # dict[str, np.ndarray], shapes: T,*  (RGB uint8 T,H,W,C ; lowdim float32 T,D)
		"action",          # np.ndarray, shape: T,Da
		"length",          # int
		"is_val"           # bool (validation split flag)
	]

	def __init__(self, obs: Dict[str, np.ndarray], action: np.ndarray, is_val: bool):
		assert len(action.shape) == 2
		lengths = [v.shape[0] for v in obs.values()] + [action.shape[0]]
		length = lengths[0]
		for l in lengths:
			assert l == length
		self.obs = obs
		self.action = action
		self.length = length
		self.is_val = is_val


class EpisodeRingBuffer:
	"""
	Thread-safe ring buffer storing episodes with FIFO eviction.
	Supports bounded capacity by number of episodes or total steps.
	"""
	def __init__(self, max_episodes: Optional[int] = None, max_steps: Optional[int] = None):
		assert (max_episodes is not None) or (max_steps is not None)
		self.max_episodes = max_episodes
		self.max_steps = max_steps
		self._episodes: List[EpisodeRecord] = []
		self._total_steps: int = 0
		self._lock = threading.Lock()

	def _over_capacity(self) -> bool:
		if (self.max_episodes is not None) and (len(self._episodes) > self.max_episodes):
			return True
		if (self.max_steps is not None) and (self._total_steps > self.max_steps):
			return True
		return False

	def add_episode(self, episode: EpisodeRecord):
		with self._lock:
			self._episodes.append(episode)
			self._total_steps += episode.length
			# Evict oldest until within capacity
			while self._over_capacity() and len(self._episodes) > 0:
				old = self._episodes.pop(0)
				self._total_steps -= old.length

	def snapshot_indices(self, split: str) -> Tuple[int, List[int]]:
		"""
		Return current count and a list of indices for either 'train' or 'val'.
		Caller should not assume stability across calls.
		"""
		with self._lock:
			if split == 'train':
				idxs = [i for i, e in enumerate(self._episodes) if not e.is_val]
			else:
				idxs = [i for i, e in enumerate(self._episodes) if e.is_val]
			return len(self._episodes), idxs

	def sample_window(self,
			split: str,
			sequence_length: int,
			pad_before: int,
			pad_after: int,
			rng: random.Random) -> Optional[Tuple[EpisodeRecord, int, int, int, int]]:
		"""
		Randomly pick an episode from the requested split and a valid start index,
		and return (episode, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx).
		Returns None if buffer is empty for that split.
		"""
		with self._lock:
			candidates = [e for e in self._episodes if (e.is_val if split == 'val' else not e.is_val)]
			if not candidates:
				return None
			episode = rng.choice(candidates)
			L = episode.length
			# Follow SequenceSampler.create_indices logic per episode
			pad_before = min(max(pad_before, 0), sequence_length - 1)
			pad_after = min(max(pad_after, 0), sequence_length - 1)
			min_start = -pad_before
			max_start = L - sequence_length + pad_after
			
			# Fix: Check if episode is long enough for sequence_length
			if max_start < min_start:
				# Episode too short, use first valid position
				start_idx = 0
				buffer_start_idx = 0
				buffer_end_idx = min(sequence_length, L)
				start_offset = 0
				end_offset = max(0, sequence_length - L)
				sample_start_idx = 0
				sample_end_idx = sequence_length - end_offset
			else:
				start_idx = rng.randint(min_start, max_start)
				buffer_start_idx = max(start_idx, 0)
				buffer_end_idx = min(start_idx + sequence_length, L)
				start_offset = buffer_start_idx - start_idx
				end_offset = (start_idx + sequence_length) - buffer_end_idx
				sample_start_idx = 0 + start_offset
				sample_end_idx = sequence_length - end_offset
			return episode, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx

	def get_stats(self) -> Dict[str, int]:
		"""Get current buffer statistics for debugging."""
		with self._lock:
			train_episodes = sum(1 for e in self._episodes if not e.is_val)
			val_episodes = sum(1 for e in self._episodes if e.is_val)
			return {
				'total_episodes': len(self._episodes),
				'train_episodes': train_episodes,
				'val_episodes': val_episodes,
				'total_steps': self._total_steps,
				'max_episodes': self.max_episodes,
				'max_steps': self.max_steps
			}


class EpisodeReaderPool:
	"""
	Background reader that iterates files, reads episodes, and appends to the ring buffer.
	"""
	def __init__(self,
			file_paths: List[str],
			buffer: EpisodeRingBuffer,
			rgb_keys: List[str],
			lowdim_keys: List[str],
			val_ratio: float,
			seed: int,
			num_workers: int = 2,
			shuffle_files: bool = True,
			repeat_forever: bool = True):
		self.file_paths = list(file_paths)
		self.buffer = buffer
		self.rgb_keys = list(rgb_keys)
		self.lowdim_keys = list(lowdim_keys)
		self.val_ratio = float(val_ratio)
		self.seed = int(seed)
		self.num_workers = int(max(1, num_workers))
		self.shuffle_files = bool(shuffle_files)
		self.repeat_forever = bool(repeat_forever)
		self._threads: List[threading.Thread] = []
		self._stop_event = threading.Event()
		self._file_queue: "queue.Queue[str]" = queue.Queue()

		# Seed RNG for file order
		random.seed(self.seed)
		self._refill_file_queue()

	def _refill_file_queue(self):
		files = list(self.file_paths)
		if self.shuffle_files:
			random.shuffle(files)
		for f in files:
			self._file_queue.put(f)

	def start(self):
		for _ in range(self.num_workers):
			th = threading.Thread(target=self._worker, daemon=True)
			th.start()
			self._threads.append(th)

	def stop(self):
		self._stop_event.set()
		# Drain queue to unblock threads
		try:
			while not self._file_queue.empty():
				self._file_queue.get_nowait()
		except Exception:
			pass
		for th in self._threads:
			th.join(timeout=1.0)

	def _worker(self):
		local_rng = random.Random(self.seed ^ (hash(threading.get_ident()) & 0xFFFF))
		while not self._stop_event.is_set():
			try:
				file_path = self._file_queue.get(timeout=0.5)
			except queue.Empty:
				if self.repeat_forever:
					self._refill_file_queue()
					continue
				else:
					break
			try:
				self._read_file(file_path, local_rng)
			except Exception as e:
				print(f"EpisodeReaderPool error reading {file_path}: {e}")
			finally:
				# Re-enqueue for continuous streaming if desired
				if self.repeat_forever:
					self._file_queue.put(file_path)

	def _read_file(self, dataset_path: str, rng: random.Random):
		z = zarr.open(dataset_path, mode='r')
		obs_group = z['data']['obs']
		action_arr = z['data']['actions']
		episode_ends = z['meta']['episode_ends'][:]

		# Determine validation mask per-episode using stable RNG seeded by file
		file_seed = int(hash(os.path.abspath(dataset_path)) & 0xFFFFFFFF)
		file_rng = np.random.default_rng(seed=file_seed ^ self.seed)
		val_mask = get_val_mask(n_episodes=len(episode_ends), val_ratio=self.val_ratio, seed=int(file_rng.integers(0, 2**31-1)))

		start = 0
		for epi_idx, end in enumerate(episode_ends):
			length = int(end - start)
			if length <= 0:
				start = end
				continue
			obs: Dict[str, np.ndarray] = dict()
			# RGB keys stored T,H,W,C uint8
			for k in self.rgb_keys:
				arr = obs_group[k]
				# Slice and copy to compact array
				obs[k] = arr[start:end][:]
			# Lowdim keys T,D float32
			for k in self.lowdim_keys:
				arr = obs_group[k]
				obs[k] = arr[start:end][:]
			actions = action_arr[start:end][:]
			is_val = bool(val_mask[epi_idx])
			rec = EpisodeRecord(obs=obs, action=actions, is_val=is_val)
			self.buffer.add_episode(rec)
			start = end


class BufferedSim2RealImageDataset(BaseImageDataset):
	"""
	Episode-buffered dataset with background readers and on-demand sampling.
	Matches Sim2RealImageDataset batch structure exactly.
	"""
	def __init__(self,
			shape_meta: dict,
			dataset_dir: Optional[str] = None,
			file_paths: Optional[List[str]] = None,
			horizon: int = 1,
			pad_before: int = 0,
			pad_after: int = 0,
			n_obs_steps: Optional[int] = None,
			n_latency_steps: int = 0,
			seed: int = 42,
			val_ratio: float = 0.02,
			use_cache: bool = False,
			use_disk: bool = False,
			# buffering/sampling
			buffer_max_episodes: Optional[int] = 2000,
			buffer_max_steps: Optional[int] = None,
			num_reader_workers: int = 2,
			num_sampler_workers: int = 0,
			shuffle_files: bool = True,
			repeat_forever: bool = True,
			pin_memory_stage: bool = True,
			distributed_shard: str = 'auto',
			samples_per_epoch: Optional[int] = None,
            batches_per_epoch: Optional[int] = None,
			debug_buffer_log_interval: float = 5.0,
			enable_debug_logging: bool = False):
		super().__init__()

		self.shape_meta = shape_meta
		self.horizon = int(horizon)
		self.pad_before = int(pad_before)
		self.pad_after = int(pad_after)
		self.n_obs_steps = n_obs_steps
		self.n_latency_steps = int(n_latency_steps)
		self.seed = int(seed)
		self.val_ratio = float(val_ratio)
		self.use_cache = bool(use_cache)
		self.use_disk = bool(use_disk)
		self.pin_memory_stage = bool(pin_memory_stage)

		# Determine keys
		self.rgb_keys = [k for k, v in shape_meta['obs'].items() if v.get('type', 'low_dim') == 'rgb']
		self.lowdim_keys = [k for k, v in shape_meta['obs'].items() if v.get('type', 'low_dim') == 'low_dim']
		if shape_meta.get('auxiliary_obs', None) is not None:
			self.lowdim_keys.extend([k for k in shape_meta['auxiliary_obs'].keys()])

		# Discover files
		if (dataset_dir is None) and (not file_paths):
			raise ValueError("Either dataset_dir or file_paths must be provided")
		if dataset_dir is not None:
			all_paths = discover_zarr_files(dataset_dir)
		else:
			all_paths = list(file_paths)

		# FIXED: Don't shard files for distributed training - each GPU should see same data
		# This ensures proper gradient synchronization across GPUs
		rank, world_size = _detect_distributed_rank_world()
		if distributed_shard == 'auto' and world_size > 1:
			# Use all files on all ranks to ensure same training data
			self.file_paths = sorted(all_paths)
			print(f"Rank {rank}/{world_size}: Using all {len(self.file_paths)} files for consistent distributed training")
		else:
			self.file_paths = sorted(all_paths)

		# Initialize episode buffer
		self.buffer = EpisodeRingBuffer(max_episodes=buffer_max_episodes, max_steps=buffer_max_steps)

		# Start reader pool
		self.reader = EpisodeReaderPool(
			file_paths=self.file_paths,
			buffer=self.buffer,
			rgb_keys=self.rgb_keys,
			lowdim_keys=self.lowdim_keys,
			val_ratio=self.val_ratio,
			seed=self.seed,
			num_workers=num_reader_workers,
			shuffle_files=shuffle_files,
			repeat_forever=repeat_forever)
		self.reader.start()

		# Sampler queue (optional producers); keep simple: sample on demand in __getitem__
		self.num_sampler_workers = int(max(0, num_sampler_workers))

		# Pre-compute normalizer via streaming pass (consistent with existing code)
		self._cached_normalizer = self._compute_normalizer_via_streaming(
			file_paths=self.file_paths,
			shape_meta=shape_meta,
			horizon=horizon,
			pad_before=pad_before,
			pad_after=pad_after,
			n_obs_steps=n_obs_steps,
			n_latency_steps=n_latency_steps,
			seed=seed,
			val_ratio=val_ratio,
			use_cache=use_cache,
			use_disk=use_disk)

		# FIXED: Calculate epoch size based on total files, not sharded files
		# This ensures consistent epoch length across all GPUs
		total_files = len(all_paths)  # Use original file count, not sharded
		if samples_per_epoch is not None:
			self._samples_per_epoch = int(samples_per_epoch)
		else:
			# Estimate: assume ~1000 samples per file on average
			self._samples_per_epoch = max(10000, 1000 * total_files)
		print(f"Epoch size set to {self._samples_per_epoch} samples across {total_files} files")

		# Start debug logging thread only if enabled
		if enable_debug_logging:
			self._debug_log_interval = debug_buffer_log_interval
			self._debug_stop_event = threading.Event()
			self._debug_thread = threading.Thread(target=self._debug_logger, daemon=True)
			self._debug_thread.start()

	def _compute_normalizer_via_streaming(self, file_paths: List[str], **dataset_kwargs) -> LinearNormalizer:
		# Borrow approach from StreamingMultiDataset: compute normalizer combining per-file stats
		dataset_stats: List[LinearNormalizer] = []
		dataset_sizes: List[int] = []
		for fp in file_paths[:2]:
			tmp_kwargs = dict(dataset_kwargs)
			tmp_kwargs['dataset_path'] = fp
			tmp_ds = Sim2RealImageDataset(**tmp_kwargs)
			file_len = len(tmp_ds)
			dataset_sizes.append(file_len)
			dataset_stats.append(tmp_ds.get_normalizer())
			del tmp_ds
			# be friendly to memory
			import gc
			gc.collect()
		if len(dataset_stats) == 1:
			return dataset_stats[0]
		# Weighted combine
		total_size = sum(dataset_sizes)
		weights = [s / total_size for s in dataset_sizes]
		combined = LinearNormalizer()
		first = dataset_stats[0]
		for key in first.params_dict.keys():
			means: List[torch.Tensor] = []
			stds: List[torch.Tensor] = []
			for norm in dataset_stats:
				fld = norm[key]
				params = fld.params_dict
				means.append(params['offset'])
				stds.append(1.0 / params['scale'])
			means_t = torch.stack(means)
			stds_t = torch.stack(stds)
			w_t = torch.tensor(weights, dtype=torch.float32)
			combined_mean = (means_t * w_t.unsqueeze(-1)).sum(dim=0)
			combined_std = (stds_t * w_t.unsqueeze(-1)).sum(dim=0)
			first_stats = first[key].params_dict['input_stats']
			combined[key] = SingleFieldLinearNormalizer.create_manual(
				scale=1.0 / combined_std,
				offset=combined_mean,
				input_stats_dict=dict(first_stats)
			)
		return combined

	def _materialize_window(self,
			episode: EpisodeRecord,
			buffer_start_idx: int,
			buffer_end_idx: int,
			sample_start_idx: int,
			sample_end_idx: int) -> Dict[str, torch.Tensor]:
		seq_len = self.horizon + self.n_latency_steps
		# Build obs tensors
		obs_out: Dict[str, torch.Tensor] = {}
		T_slice = slice(self.n_obs_steps) if self.n_obs_steps is not None else slice(None)
		for key in self.rgb_keys:
			arr = episode.obs[key]
			seg = arr[buffer_start_idx:buffer_end_idx]
			# Pad to seq_len by edge values
			if (sample_start_idx > 0) or (sample_end_idx < seq_len):
				pad = np.zeros((seq_len,) + seg.shape[1:], dtype=seg.dtype)
				if sample_start_idx > 0:
					pad[:sample_start_idx] = seg[0]
				if sample_end_idx < seq_len:
					pad[sample_end_idx:] = seg[-1]
				pad[sample_start_idx:sample_end_idx] = seg
				seg = pad
			# Convert to CHW float32 [0,1]
			# Input seg is T,H,W,C uint8
			seg = np.moveaxis(seg, -1, 1).astype(np.float32) / 255.0
			t = torch.from_numpy(seg[T_slice])
			obs_out[key] = t
		for key in self.lowdim_keys:
			arr = episode.obs[key]
			seg = arr[buffer_start_idx:buffer_end_idx]
			if (sample_start_idx > 0) or (sample_end_idx < seq_len):
				pad = np.zeros((seq_len,) + seg.shape[1:], dtype=seg.dtype)
				if sample_start_idx > 0:
					pad[:sample_start_idx] = seg[0]
				if sample_end_idx < seq_len:
					pad[sample_end_idx:] = seg[-1]
				pad[sample_start_idx:sample_end_idx] = seg
				seg = pad
			t = torch.from_numpy(seg[T_slice].astype(np.float32))
			obs_out[key] = t
		# Actions: length horizon (+ latency handled below)
		act_seg = episode.action[buffer_start_idx:buffer_end_idx]
		if (sample_start_idx > 0) or (sample_end_idx < seq_len):
			pad = np.zeros((seq_len,) + act_seg.shape[1:], dtype=act_seg.dtype)
			if sample_start_idx > 0:
				pad[:sample_start_idx] = act_seg[0]
			if sample_end_idx < seq_len:
				pad[sample_end_idx:] = act_seg[-1]
			pad[sample_start_idx:sample_end_idx] = act_seg
			act_seg = pad
		act_seg = act_seg.astype(np.float32)
		if self.n_latency_steps > 0:
			act_seg = act_seg[self.n_latency_steps:]
		act_t = torch.from_numpy(act_seg)
		# if self.pin_memory_stage:
		# 	obs_out = {k: v.pin_memory() if isinstance(v, torch.Tensor) else v for k, v in obs_out.items()}
		# 	act_t = act_t.pin_memory()
		return {
			'obs': obs_out,
			'action': act_t
		}

	def __len__(self) -> int:
		return self._samples_per_epoch

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
		# Sample on demand from train split
		threadpool_limits(1)
		rng = random.Random(self.seed ^ idx)
		seq_len = self.horizon + self.n_latency_steps
		for _ in range(100):  # retry loop while buffer warms up
			res = self.buffer.sample_window(split='train', sequence_length=seq_len,
				pad_before=self.pad_before, pad_after=self.pad_after, rng=rng)
			if res is not None:
				episode, bs, be, ss, se = res
				return self._materialize_window(episode, bs, be, ss, se)
			time.sleep(0.01)
		raise RuntimeError("BufferedSim2RealImageDataset buffer is empty or not ready.")

	def get_validation_dataset(self) -> 'BufferedSim2RealImageDataset':
		# Return a shallow wrapper that samples from 'val' split in __getitem__
		parent = self
		class _ValView(BufferedSim2RealImageDataset):
			def __init__(self, parent: BufferedSim2RealImageDataset):
				# create a very thin wrapper; do not start new readers
				self.__dict__ = parent.__dict__.copy()
				self._is_val_view = True
			def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
				threadpool_limits(1)
				rng = random.Random(self.seed ^ (idx + 123456))
				seq_len = self.horizon + self.n_latency_steps
				for _ in range(100):
					res = self.buffer.sample_window(split='val', sequence_length=seq_len,
						pad_before=self.pad_before, pad_after=self.pad_after, rng=rng)
					if res is not None:
						episode, bs, be, ss, se = res
						return parent._materialize_window(episode, bs, be, ss, se)
					time.sleep(0.01)
				raise RuntimeError("Validation buffer is empty or not ready.")
			def __len__(self) -> int:
				return max(1, self._samples_per_epoch // 20)
		return _ValView(parent)

	def get_normalizer(self, **kwargs) -> LinearNormalizer:
		return self._cached_normalizer

	def get_all_actions(self) -> torch.Tensor:
		# Concatenate actions from current buffer snapshot
		with self.buffer._lock:
			acts: List[torch.Tensor] = []
			for epi in self.buffer._episodes:
				acts.append(torch.from_numpy(epi.action.astype(np.float32)))
			if not acts:
				return torch.zeros((0,))
			return torch.cat(acts, dim=0)

	def _debug_logger(self):
		"""Periodically log buffer statistics for debugging."""
		last_log_time = time.time()
		while not self._debug_stop_event.is_set():
			time.sleep(1.0)  # Check every second
			current_time = time.time()
			if current_time - last_log_time >= self._debug_log_interval:
				stats = self.buffer.get_stats()
				print(f"[BufferedSim2RealImageDataset] Buffer stats: "
					  f"total_episodes={stats['total_episodes']}, "
					  f"train_episodes={stats['train_episodes']}, "
					  f"val_episodes={stats['val_episodes']}, "
					  f"total_steps={stats['total_steps']}, "
					  f"max_episodes={stats['max_episodes']}")
				last_log_time = current_time

	def __del__(self):
		try:
			if hasattr(self, 'reader'):
				self.reader.stop()
			if hasattr(self, '_debug_stop_event'):
				self._debug_stop_event.set()
		except Exception:
			pass
