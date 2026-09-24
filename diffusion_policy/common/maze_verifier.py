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


class MazeVerifier:
    def __init__(self, env=None, maze_path=None, device="cpu", success_radius=0.5, noise=0.0):
        self.env = env
        self.device = torch.device(device)
        self.success_radius = success_radius
        self.noise = noise
        self._cache = {}
        self._torch_distance_cache = None
        if env is not None and env.maze is not None:
            self._ensure_table(env.maze)
        elif maze_path is not None:
            self._ensure_table(np.load(maze_path))

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
        if self.noise > 0:
            score = score * (1 + self.noise * torch.randn_like(score))
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
        bsz, h, w = free.shape
        batch_idx = torch.arange(bsz, device=self.device)
        pos = start.to(self.device, non_blocking=True).float()
        action = action.to(self.device, non_blocking=True).float()

        for delta in action.unbind(dim=1):
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

            pos = cur.float()

        return pos

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
        if self.noise > 0:
            value = value * (1 + self.noise * torch.randn_like(value))
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
