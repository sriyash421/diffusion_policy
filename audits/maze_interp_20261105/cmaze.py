"""Continuous-position version of the 11x11 maze (MAZE_SETTING=interp30c).

The agent is a point with a real-valued position; cell (x, y) is the square [x-.5, x+.5] x [y-.5, y+.5] and the point is in free space
iff its rounded cell is in bounds and free. An action is a real-valued delta (dx, dy), clipped to length <= MAX_STEP (1 cell),
executed in K sub-steps; a sub-step that would enter a wall is replaced by its x-only or y-only part (sliding along the wall), or
dropped. A sub-step that changes both cell coordinates also needs one of the two side cells free (no squeezing between two walls
that meet at a corner). Observations: the image marks the agent's ROUNDED cell (build_images_torch), agent_pos is real-valued;
the BFS feedback is the BFS distance of the rounded cell, dx, dy = goal - pos (continuous). Success = within 1 of the goal centre.

Demos (smoothed A*, make_cont_demos.py): start = start-cell centre + U(-JITTER, JITTER)^2; the A* cell path, string-pulled (from the
current point jump to the farthest later cell centre whose straight segment keeps CLEAR from every wall), each straight segment
split into ceil(length) equal steps (<= 1); the last point is the goal-cell centre. Every demo is replayed through cont_step_torch.
"""
import math
import numpy as np
import torch

MAX_STEP = 1.0; K = 8; JITTER = 0.3; CLEAR = 0.1   # demo start jitter (< .5 - CLEAR) and demo wall clearance


def _cell_free(free, cx, cy):
    h, w = free.shape[-2:]; b = torch.arange(free.shape[0], device=free.device)
    ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
    return ok & free[b, cy.clamp(0, h - 1), cx.clamp(0, w - 1)]


def can_enter(free, p, q):
    """(B,) bool: a sub-step from p to q (less than one cell long) is allowed."""
    cp = torch.round(p).long(); cq = torch.round(q).long()
    ok = _cell_free(free, cq[:, 0], cq[:, 1])
    both = (cp != cq).all(-1)
    side = _cell_free(free, cq[:, 0], cp[:, 1]) | _cell_free(free, cp[:, 0], cq[:, 1])
    return ok & (~both | side)


def clip_action(a):
    n = a.norm(dim=-1, keepdim=True)
    return a * (MAX_STEP / n.clamp_min(MAX_STEP))


def cont_step_torch(free, pos, a):
    """New positions (B, 2) after one clipped action; free (B, H, W) bool."""
    p = pos.float(); d = clip_action(a.float()) / K
    ex = torch.tensor([1., 0.], device=p.device); ey = torch.tensor([0., 1.], device=p.device)
    for _ in range(K):
        full, xo, yo = p + d, p + d * ex, p + d * ey
        p = torch.where(can_enter(free, p, full)[:, None], full,
                        torch.where(can_enter(free, p, xo)[:, None], xo, torch.where(can_enter(free, p, yo)[:, None], yo, p)))
    return p


def segment_ok(free1, a, b, ds=0.02, clear=CLEAR):
    """The straight segment a -> b (numpy (2,)) stays in free space (dense sub-steps under can_enter) with `clear` clearance: the
    square of half-size `clear` around every sample point lies in free cells (so demo segments never graze a wall corner)."""
    n = max(1, int(math.ceil(np.linalg.norm(b - a) / ds)))
    t = torch.linspace(0, 1, n + 1, dtype=torch.float64)[:, None]
    pts = (torch.as_tensor(a)[None] * (1 - t) + torch.as_tensor(b)[None] * t).float()
    fr = free1.expand(n, -1, -1)
    if clear > 0:
        fa = free1.expand(n + 1, -1, -1)
        for ox, oy in ((clear, clear), (clear, -clear), (-clear, clear), (-clear, -clear)):
            c = torch.round(pts + torch.tensor([ox, oy])).long()
            if not bool(_cell_free(fa, c[:, 0], c[:, 1]).all()):
                return False
    return bool(can_enter(fr, pts[:-1], pts[1:]).all()) and bool(_cell_free(free1, torch.round(pts[:1]).long()[:, 0], torch.round(pts[:1]).long()[:, 1]).all())


def smooth_path(free1, start_pt, cells):
    """String-pulled polyline from start_pt through the A* cell centres `cells` (n, 2) to the last one."""
    pts = [np.asarray(start_pt, np.float64)]; i = 0; cells = np.asarray(cells, np.float64)
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i and not segment_ok(free1, pts[-1], cells[j]):
            j -= 1
        if j == i:   # cannot happen: the next A* cell centre is always visible (jitter < .5)
            raise RuntimeError('stuck while string-pulling')
        pts.append(cells[j]); i = j
    return np.stack(pts)


def resample(poly):
    """Split each straight piece into ceil(length) equal steps (each <= 1); returns the points (T+1, 2)."""
    out = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) - 1e-9)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.stack(out)
