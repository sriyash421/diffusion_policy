"""Ground-truth verifier for the L2S maze (`scripts/l2s_maze`).

Rolls an action chunk through the exact `ProcgenMazeEnv.step` dynamics and scores
    value = -||final_pos - goal||_2
Positions are grid (row, col); `maze[i, j] == 1` is wall.

Chunk alignment: with n_obs_steps = To, `chunk[To-1]` is the action taken at the current
obs, so the verifier rolls `chunk[start_offset:]` (start_offset = To-1) from
`agent_pos[:, -1]`.
"""
import numpy as np
import torch


class MazeVerifier:
    def __init__(self, maze_path=None, maze=None, device='cpu', noise=0.0,
                 step_scale=0.25, action_clip=1.0, start_offset=1):
        if maze is None:
            if maze_path is None:
                raise ValueError('MazeVerifier needs maze or maze_path')
            maze = np.load(maze_path)
        self.maze = np.asarray(maze, dtype=np.uint8)
        self.device = torch.device(device)
        self.noise = noise
        self.step_scale = step_scale
        self.action_clip = action_clip
        self.start_offset = start_offset
        self._free_cache = {}

    def _free(self, device):
        """(H, W) bool free map on `device`, cached."""
        key = str(device)
        if key not in self._free_cache:
            self._free_cache[key] = torch.as_tensor(self.maze == 0, device=device)
        return self._free_cache[key]

    def _is_free(self, free, pos):
        """pos (B, 2) float -> (B,) bool: round(pos) in bounds and not a wall."""
        h, w = free.shape
        ij = torch.round(pos).long()
        i, j = ij[:, 0], ij[:, 1]
        inb = (i >= 0) & (i < h) & (j >= 0) & (j < w)
        return inb & free[i.clamp(0, h - 1), j.clamp(0, w - 1)]

    def rollout(self, start, actions):
        """start (B, 2), actions (B, T, 2) -> positions (B, T+1, 2), float64.

        Per step: clip, scale, then full move -> row-only -> col-only -> stay,
        the same order as ProcgenMazeEnv.step.
        """
        start = torch.as_tensor(start)
        actions = torch.as_tensor(actions)
        device = actions.device
        free = self._free(device)
        pos = start.to(device=device, dtype=torch.float64)
        delta = self.step_scale * actions.to(torch.float64).clamp(
            -self.action_clip, self.action_clip)
        zero = torch.zeros_like(pos[:, :1])
        traj = [pos]
        for t in range(delta.shape[1]):
            d = delta[:, t]
            full = pos + d
            row = pos + torch.cat([d[:, :1], zero], dim=1)
            col = pos + torch.cat([zero, d[:, 1:]], dim=1)
            nxt = torch.where(
                self._is_free(free, full)[:, None], full,
                torch.where(
                    self._is_free(free, row)[:, None], row,
                    torch.where(self._is_free(free, col)[:, None], col, pos)))
            pos = nxt
            traj.append(pos)
        return torch.stack(traj, dim=1)

    def get_value(self, obs_dict, action):
        """obs_dict raw (unnormalized) with agent_pos/goal_pos (B, To, 2); action (B, H, 2)
        in env units -> (B,) float32 value, on action's device."""
        if action.ndim == 4:
            action = action[:, 0]
        device = action.device
        start = obs_dict['agent_pos'][:, -1].to(device)
        goal = obs_dict['goal_pos'][:, -1].to(device=device, dtype=torch.float64)
        final = self.rollout(start, action[:, self.start_offset:])[:, -1]
        value = -torch.linalg.norm(final - goal, dim=-1)
        if self.noise > 0:
            value = value * (1 + self.noise * torch.randn_like(value))
        return value.to(torch.float32)
