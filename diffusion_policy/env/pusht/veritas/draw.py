"""Veritas waypoint overlay, ported from veritas/src/agent/utils/frame_render.py.

Only ``draw_waypoints_rgb`` and ``_catmull_rom_spline`` come over (the rest of that module
pulls in transforms3d); drawing is unchanged so PushT overlays match Veritas's logs. Its
marker sizes are tuned for a 640x480 frame, so callers draw on an upscaled obs.
"""

import cv2
import numpy as np


def _catmull_rom_spline(points, num_interp=20):
    """Catmull-Rom spline through *points* (Nx2).  Returns (Mx2) int32."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.astype(np.int32)
    if n == 2:
        ts = np.linspace(0, 1, num_interp)
        interp = pts[0][None] + ts[:, None] * (pts[1] - pts[0])[None]
        return interp.astype(np.int32)
    padded = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    result = []
    for i in range(n - 1):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        for t in np.linspace(0, 1, num_interp, endpoint=(i == n - 2)):
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
                        + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                        + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
                        + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                        + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            result.append((x, y))
    return np.array(result, dtype=np.int32)


def draw_waypoints_rgb(frame_rgb, waypoints_dbg, draw_curve=False):
    """Draw waypoints on an RGB frame as numbered pin markers.

    Each waypoint is rendered as a filled circle with a white ring and a
    rounded-rectangle badge showing the waypoint index.

    When *draw_curve* is True a smooth Catmull-Rom spline is drawn through
    the waypoints in order (semi-transparent, behind the markers).

    Args:
        waypoints_dbg: list of dicts with ``uv`` (pixel coords in *frame_rgb*) and ``idx``.
    """
    if not waypoints_dbg:
        return frame_rgb
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    overlay = frame_bgr.copy()
    h, w = frame_bgr.shape[:2]

    # Google-style blue palette (BGR)
    pin_blue = (210, 150, 50)
    badge_blue = (210, 150, 50)
    curve_color = (210, 170, 90)

    markers = []
    for wp_dbg in waypoints_dbg:
        goal = wp_dbg.get("goal_uv") or wp_dbg.get("uv")
        idx_wp = wp_dbg.get("idx", -1)
        if goal and len(goal) == 2:
            u = max(0, min(w - 1, int(round(goal[0]))))
            v = max(0, min(h - 1, int(round(goal[1]))))
            markers.append((u, v, idx_wp))

    # Smooth curve first, behind the markers: white halo, then coloured line.
    if draw_curve and len(markers) >= 2:
        ctrl_pts = np.array([(u, v) for u, v, _ in markers], dtype=np.float64)
        spline_pts = _catmull_rom_spline(ctrl_pts, num_interp=30)
        cv2.polylines(overlay, [spline_pts], isClosed=False,
                      color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
        cv2.polylines(overlay, [spline_pts], isClosed=False,
                      color=curve_color, thickness=1, lineType=cv2.LINE_AA)

    dot_r = 6
    ring_r = dot_r + 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.40
    font_thick = 1
    badge_pad_x, badge_pad_y = 5, 3

    for u, v, idx_wp in markers:
        cv2.circle(overlay, (u, v), ring_r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, (u, v), dot_r, pin_blue, -1, cv2.LINE_AA)

        label = str(idx_wp)
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
        bw = tw + badge_pad_x * 2
        bh = th + badge_pad_y * 2 + baseline
        bh = max(bh, bw)  # keep badge roughly square for single digits

        # Badge to the upper-right of the dot, kept on screen.
        bx = u + ring_r + 1
        by = v - bh - 1
        if bx + bw > w:
            bx = u - ring_r - 1 - bw
        by = max(0, min(h - bh, by))

        r = min(bh, bw) // 2
        cv2.rectangle(overlay, (bx + r, by), (bx + bw - r, by + bh), badge_blue, -1)
        cv2.rectangle(overlay, (bx, by + r), (bx + bw, by + bh - r), badge_blue, -1)
        for cx, cy in [
            (bx + r, by + r), (bx + bw - r, by + r),
            (bx + r, by + bh - r), (bx + bw - r, by + bh - r),
        ]:
            cv2.circle(overlay, (cx, cy), r, badge_blue, -1, cv2.LINE_AA)

        tx = bx + (bw - tw) // 2
        ty = by + (bh + th) // 2
        cv2.putText(overlay, label, (tx, ty), font, font_scale,
                    (255, 255, 255), font_thick, cv2.LINE_AA)

    # Slight transparency so the scene shows through.
    cv2.addWeighted(overlay, 0.88, frame_bgr, 0.12, 0, frame_bgr)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
