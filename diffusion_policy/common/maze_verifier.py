from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


def get_shortest_path_table(maze, device):
    free = [(int(x), int(y)) for y, x in np.argwhere(maze)]
    pos_to_idx = {pos: i for i, pos in enumerate(free)}
    h, w = maze.shape
    unreachable = float(h * w + 1)
    dists = torch.full(
        (len(free), len(free)),
        unreachable,
        dtype=torch.float32,
        device=device,
    )
    for i, start in enumerate(free):
        q = deque([start])
        seen = {start: 0}
        while q:
            x, y = q.popleft()
            cur = seen[(x, y)]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = x + dx
                ny = y + dy
                nxt = (nx, ny)
                if nxt in seen:
                    continue
                if not (0 <= nx < w and 0 <= ny < h and maze[ny, nx]):
                    continue
                seen[nxt] = cur + 1
                q.append(nxt)
        for pos, dist in seen.items():
            dists[i, pos_to_idx[pos]] = float(dist)

    grid_to_idx = torch.full((h, w), -1, dtype=torch.long, device=device)
    for i, (x, y) in enumerate(free):
        grid_to_idx[y, x] = i
    return dists, grid_to_idx, unreachable


# harine/openloop: batched torch versions of MazeEnv's dynamics, shared by the search policies'
# training-time candidate generation and the open-loop eval harness so the two cannot drift.

def rollout_step_torch(free, pos, delta):
    """One `MazeEnv.rollout_delta` for a batch: walk the rounded line, stop at the first wall."""
    bsz, h, w = free.shape
    batch_idx = torch.arange(bsz, device=free.device)
    pos = pos.float()
    delta = delta.float()
    a = torch.round(pos).long()
    b = torch.round(pos + delta).long()
    d = b - a
    n = d.abs().amax(dim=-1)
    cur = a.clone()
    moving = n > 0
    max_n = int(n.max().item())

    for step in range(1, max_n + 1):
        active = moving & (n >= step)

        denom = n.clamp_min(1).float().unsqueeze(-1)
        nxt = torch.round(a.float() + d.float() * (step / denom)).long()
        changed = (nxt != cur).any(dim=-1)
        active_changed = active & changed

        x = nxt[:, 0]
        y = nxt[:, 1]
        in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x_safe = x.clamp(0, w - 1)
        y_safe = y.clamp(0, h - 1)
        passable = free[batch_idx, y_safe, x_safe]
        valid = in_bounds & passable

        should_move = active_changed & valid
        cur = torch.where(should_move[:, None], nxt, cur)
        moving = moving & ~(active_changed & ~valid)

    return cur.float()


def snap_cardinal_torch(action):
    """Nearest unit cardinal move: (sign(major axis), 0) / (0, sign) or (0, 0) if |a|_inf < 0.5.

    Makes one policy decision equal one BFS step, like the expert's actions; applied identically
    in the policies' training-time context rollouts and in the eval harness.
    """
    ax = action[..., 0]
    ay = action[..., 1]
    use_x = ax.abs() >= ay.abs()
    mag = torch.maximum(ax.abs(), ay.abs())
    move = (mag >= 0.5).float()
    sx = torch.sign(ax) * use_x.float() * move
    sy = torch.sign(ay) * (~use_x).float() * move
    return torch.stack([sx, sy], dim=-1)


def rollout_trajectory_torch(free, start, actions, goal, stop_radius=1.0, snap=False):
    """Open-loop rollout of (B, T, 2) deltas that freezes a row once it reaches the goal.

    Mirrors `eval_bc.eval_episode_batch`: success is checked after each step with
    ||pos - goal|| <= stop_radius, and a finished row stops moving. Returns the positions
    (B, T+1, 2) and `done_step` (B,), the number of steps taken to succeed, or -1.
    """
    pos = start.float()
    goal = goal.float()
    if snap:
        actions = snap_cardinal_torch(actions)
    bsz, T, _ = actions.shape
    done = torch.zeros(bsz, dtype=torch.bool, device=free.device)
    done_step = torch.full((bsz,), -1, dtype=torch.long, device=free.device)
    traj = [pos]
    for t in range(T):
        nxt = rollout_step_torch(free, pos, actions[:, t])
        pos = torch.where(done[:, None], pos, nxt)
        traj.append(pos)
        reached = ~done & ((pos - goal).norm(dim=-1) <= stop_radius)
        done_step = torch.where(reached, torch.full_like(done_step, t + 1), done_step)
        done = done | reached
    return torch.stack(traj, dim=1), done_step


def realized_actions(traj):
    """Per-step displacements actually taken, (B, T, 2); zero once a row has stopped."""
    return traj[:, 1:] - traj[:, :-1]


def build_images_torch(free, pos, goal):
    """`maze.build_image(...) / 255` for a batch: (B, 3, H, W) with wall, goal, agent planes."""
    bsz, h, w = free.shape
    batch_idx = torch.arange(bsz, device=free.device)
    img = torch.zeros((bsz, 3, h, w), dtype=torch.float32, device=free.device)
    img[:, 0] = (~free).float()
    g = torch.round(goal).long()
    p = torch.round(pos).long()
    img[batch_idx, 1, g[:, 1].clamp(0, h - 1), g[:, 0].clamp(0, w - 1)] = 1.0
    img[batch_idx, 2, p[:, 1].clamp(0, h - 1), p[:, 0].clamp(0, w - 1)] = 1.0
    return img


class MazeVerifier:
    def __init__(self, env=None, maze_path=None, device="cpu", success_radius=0.5, noise=0.0,
                 noise_mode="multiplicative", noise_generator=None):
        self.env = env
        self.device = torch.device(device)
        self.success_radius = success_radius
        self.noise = noise
        # verifier-noise study (2026-10-21): "additive" adds N(0, noise^2) in BFS steps to every value;
        # the default keeps the original multiplicative value * (1 + noise * eps).
        if noise_mode not in ("multiplicative", "additive"):
            raise ValueError(f"unknown noise_mode: {noise_mode}")
        self.noise_mode = noise_mode
        # optional torch.Generator for the noise only, so the caller's sampling stream is untouched
        # (noisy and clean evals then draw identical candidates and differ only in what the verifier reports)
        self.noise_generator = noise_generator
        self._cache = {}
        self._torch_distance_cache = None
        if env is not None and env.maze is not None:
            self._ensure_table(env.maze)
        elif maze_path is not None:
            self._ensure_table(np.load(maze_path))

    def _add_noise(self, value):
        if self.noise <= 0:
            return value
        g = self.noise_generator
        eps = (torch.randn(value.shape, generator=g, device=g.device, dtype=value.dtype).to(value.device)
               if g is not None else torch.randn_like(value))
        if self.noise_mode == "additive":
            return value + self.noise * eps
        return value * (1 + self.noise * eps)

    def _active_maze(self):
        if self.env is None or self.env.maze is None:
            if not self._cache:
                raise ValueError("No active maze available; pass maze=... or use get_value with image obs")
            key = next(iter(self._cache))
            return self._cache[key]["maze"]
        return self.env.maze

    def _ensure_table(self, maze):
        key = maze.tobytes()
        if key not in self._cache:
            dists, grid_to_idx, unreachable = get_shortest_path_table(maze, self.device)
            self._cache[key] = {
                "maze": maze,
                "dists": dists,
                "grid_to_idx": grid_to_idx,
                "unreachable": unreachable,
            }
        return self._cache[key]

    def goal_distance(self, pos, goal, maze=None):
        maze = self._active_maze() if maze is None else maze
        table = self._ensure_table(maze)
        pos = pos.to(self.device)
        goal = goal.to(self.device)
        if goal.shape[0] == 1:
            goal = goal.expand(pos.shape[0], -1)

        pos_cell = torch.round(pos).long()
        goal_cell = torch.round(goal).long()
        px = pos_cell[:, 0]
        py = pos_cell[:, 1]
        gx = goal_cell[:, 0]
        gy = goal_cell[:, 1]

        h, w = maze.shape
        pos_ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        goal_ok = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)

        pos_idx = torch.full((pos.shape[0],), -1, dtype=torch.long, device=self.device)
        goal_idx = torch.full((pos.shape[0],), -1, dtype=torch.long, device=self.device)
        pos_idx[pos_ok] = table["grid_to_idx"][py[pos_ok], px[pos_ok]]
        goal_idx[goal_ok] = table["grid_to_idx"][gy[goal_ok], gx[goal_ok]]

        valid = (pos_idx >= 0) & (goal_idx >= 0)
        dist = torch.full(
            (pos.shape[0],),
            table["unreachable"],
            dtype=torch.float32,
            device=self.device,
        )
        dist[valid] = table["dists"][pos_idx[valid], goal_idx[valid]]
        return dist

    def trajectory_goal_distance(self, traj, goal, maze=None):
        if goal.shape[0] == 1:
            goal = goal.expand(traj.shape[0], -1)
        b, t, _ = traj.shape
        tiled_goal = goal[:, None, :].expand(b, t, 2).reshape(-1, 2)
        return self.goal_distance(traj.reshape(-1, 2), tiled_goal, maze=maze).reshape(b, t)

    def score(self, traj, goal, maze=None):
        x = torch.as_tensor(traj, dtype=torch.float32, device=self.device)
        g = torch.as_tensor(goal, dtype=torch.float32, device=self.device)
        if x.ndim == 2:
            x = x[None]
        if g.ndim == 1:
            g = g[None]

        goal_dists = self.trajectory_goal_distance(x, g, maze=maze)
        score = -goal_dists[:, -1]
        score = self._add_noise(score)
        return score

    def _maze_from_obs(self, obs_dict):
        image = obs_dict.get("image")
        if image is None:
            return None
        # image is B,T,C,H,W and channel 0 is wall/free: wall=1, free=0 after dataset scaling.
        wall = image[:, -1, 0].detach().cpu().numpy()
        return (wall < 0.5).astype(np.uint8)

    def _free_from_obs_torch(self, obs_dict):
        image = obs_dict.get("image")
        if image is None:
            return None
        # image is B,T,C,H,W and channel 0 is wall/free: wall=1, free=0 after dataset scaling.
        wall = image[:, -1, 0].to(self.device, non_blocking=True)
        return wall < 0.5

    def _rollout_actions_torch(self, free, start, action):
        pos = start.to(self.device, non_blocking=True).float()
        action = action.to(self.device, non_blocking=True).float()
        for delta in action.unbind(dim=1):
            pos = rollout_step_torch(free, pos, delta)
        return pos

    def get_value_and_realized(self, obs_dict, action, stop_radius=1.0, snap=False):
        """harine/openloop: value of a full open-loop chunk plus the trajectory it realizes.

        Like `get_value` on the image path, but the rollout stops once the agent is within
        `stop_radius` of the goal (the eval success rule), so the returned context describes
        what the env would actually execute. Returns (value, realized, traj, done_step).
        """
        pos = obs_dict["agent_pos"][:, -1]
        goal = obs_dict["goal_pos"][:, -1]
        if action.ndim == 4:
            action = action[:, 0]
        image = obs_dict["image"]
        free = self._free_from_obs_torch(obs_dict)
        traj, done_step = rollout_trajectory_torch(free, pos, action, goal, stop_radius, snap=snap)
        dist, unreachable = self._get_goal_distance_field_torch(image, free, goal)
        value = -self._sample_goal_distance_torch(free, dist, traj[:, -1], unreachable)
        value = self._add_noise(value)
        return value, realized_actions(traj), traj, done_step

    def trajectory_distances(self, obs_dict, traj):
        """harine/openloop: BFS distance to goal at every point of `traj` (B, T+1)."""
        goal = obs_dict["goal_pos"][:, -1]
        free = self._free_from_obs_torch(obs_dict)
        dist, unreachable = self._get_goal_distance_field_torch(obs_dict["image"], free, goal)
        b, t, _ = traj.shape
        flat = traj.reshape(-1, 2)
        free_rep = free.repeat_interleave(t, dim=0)
        dist_rep = dist.repeat_interleave(t, dim=0)
        return self._sample_goal_distance_torch(free_rep, dist_rep, flat, unreachable).reshape(b, t)

    def _tensor_cache_key(self, tensor):
        try:
            storage_ptr = tensor.untyped_storage().data_ptr()
        except AttributeError:
            storage_ptr = tensor.storage().data_ptr()
        return (
            storage_ptr,
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.device),
        )

    def _goal_distance_field_torch(self, free, goal):
        bsz, h, w = free.shape
        unreachable = float(h * w + 1)
        batch_idx = torch.arange(bsz, device=self.device)
        dist = torch.full(
            (bsz, h, w),
            unreachable,
            dtype=torch.float32,
            device=self.device,
        )

        goal_cell = torch.round(goal.to(self.device, non_blocking=True)).long()
        gx = goal_cell[:, 0]
        gy = goal_cell[:, 1]
        goal_ok = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        gx_safe = gx.clamp(0, w - 1)
        gy_safe = gy.clamp(0, h - 1)
        goal_ok = goal_ok & free[batch_idx, gy_safe, gx_safe]
        dist[batch_idx[goal_ok], gy_safe[goal_ok], gx_safe[goal_ok]] = 0.0

        for _ in range(h * w - 1):
            up = F.pad(dist[:, :-1, :], (0, 0, 1, 0), value=unreachable)
            down = F.pad(dist[:, 1:, :], (0, 0, 0, 1), value=unreachable)
            left = F.pad(dist[:, :, :-1], (1, 0, 0, 0), value=unreachable)
            right = F.pad(dist[:, :, 1:], (0, 1, 0, 0), value=unreachable)
            neighbor = torch.minimum(
                torch.minimum(up, down),
                torch.minimum(left, right),
            )
            new_dist = torch.minimum(dist, neighbor + 1.0)
            dist = torch.where(free, new_dist, torch.full_like(dist, unreachable))

        return dist, unreachable

    def _get_goal_distance_field_torch(self, image, free, goal):
        key = (
            self._tensor_cache_key(image),
            self._tensor_cache_key(goal),
        )
        cache = self._torch_distance_cache
        if cache is not None and cache["key"] == key:
            return cache["dist"], cache["unreachable"]

        dist, unreachable = self._goal_distance_field_torch(free, goal)
        self._torch_distance_cache = {
            "key": key,
            "dist": dist,
            "unreachable": unreachable,
            "image": image,
            "goal": goal,
        }
        return dist, unreachable

    def _sample_goal_distance_torch(self, free, dist, pos, unreachable):
        bsz, h, w = free.shape
        batch_idx = torch.arange(bsz, device=self.device)
        pos_cell = torch.round(pos).long()
        px = pos_cell[:, 0]
        py = pos_cell[:, 1]
        pos_ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        px_safe = px.clamp(0, w - 1)
        py_safe = py.clamp(0, h - 1)
        pos_ok = pos_ok & free[batch_idx, py_safe, px_safe]

        out = torch.full((bsz,), unreachable, dtype=torch.float32, device=self.device)
        out[pos_ok] = dist[batch_idx[pos_ok], py_safe[pos_ok], px_safe[pos_ok]]
        return out

    def _goal_distance_torch(self, free, pos, goal):
        dist, unreachable = self._goal_distance_field_torch(free, goal)
        return self._sample_goal_distance_torch(free, dist, pos, unreachable)

    def _get_value_image_torch(self, obs_dict, pos, goal, action):
        image = obs_dict.get("image")
        if image is None:
            return None
        free = self._free_from_obs_torch(obs_dict)
        final_pos = self._rollout_actions_torch(free, pos, action)
        dist, unreachable = self._get_goal_distance_field_torch(image, free, goal)
        value = -self._sample_goal_distance_torch(free, dist, final_pos, unreachable)
        value = self._add_noise(value)
        return value

    def _rollout_delta_np(self, maze, pos, delta):
        h, w = maze.shape
        a = np.rint(pos).astype(np.int32)
        b = np.rint(pos + delta).astype(np.int32)
        d = b - a
        n = int(np.abs(d).max())
        cur = a.copy()
        if n == 0:
            return cur.astype(np.float32)
        for i in range(1, n + 1):
            nxt = np.rint(a + d * (i / n)).astype(np.int32)
            if np.all(nxt == cur):
                continue
            x, y = nxt
            if not (0 <= x < w and 0 <= y < h and maze[y, x]):
                break
            cur = nxt
        return cur.astype(np.float32)

    def _rollout_actions_np(self, maze, start, action):
        pos = np.asarray(start, dtype=np.float32)
        traj = [pos.copy()]
        for delta in action:
            pos = self._rollout_delta_np(maze, pos, delta)
            traj.append(pos.copy())
        return np.asarray(traj, dtype=np.float32)

    def get_value(self, obs_dict, action):
        if "agent_pos" in obs_dict:
            pos = obs_dict["agent_pos"][:, -1]
            goal = obs_dict["goal_pos"][:, -1]
        else:
            obs = obs_dict["obs"]
            pos = obs[:, -1, :2]
            goal = obs[:, -1, 2:4]

        if action.ndim == 4:
            action = action[:, 0]

        if self.env is not None and action.shape[0] == 1:
            traj_np = self.env.rollout_actions(
                pos[0].detach().cpu().numpy(),
                action[0].detach().cpu().numpy(),
            )
            traj = torch.as_tensor(traj_np[None], dtype=torch.float32, device=self.device)
            return self.score(traj, goal)

        value = self._get_value_image_torch(obs_dict, pos, goal, action)
        if value is not None:
            return value

        mazes = self._maze_from_obs(obs_dict)
        if mazes is not None:
            values = []
            pos_np = pos.detach().cpu().numpy()
            action_np = action.detach().cpu().numpy()
            goal_np = goal.detach().cpu().numpy()
            for i, maze in enumerate(mazes):
                traj = self._rollout_actions_np(maze, pos_np[i], action_np[i])
                value = self.score(traj[None], goal_np[i:i + 1], maze=maze)
                values.append(value[0])
            return torch.stack(values)

        traj = torch.cat([pos[:, None], pos[:, None] + action.cumsum(dim=1)], dim=1)
        return self.score(traj, goal)
