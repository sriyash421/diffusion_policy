"""Env-free feedback signal for PushT (pure numpy, no gym import).

The feedback is a per-step keypoint-displacement vector between the goal-T and the
achieved-T: transform the T's canonical keypoints by the achieved block pose and by
the goal pose, and return ``goal_kp - achieved_kp`` (flattened). It is ``0`` iff the
block pose equals the goal pose.

Kept import-light so the dataset can produce the exact same signal from the stored
``block_pos`` that the live env wrapper produces from its pymunk body.

It also owns the scalar distances derived from that signal, so the verifier, the env
runner's logged metrics and the offline analysis scripts all read one definition:
``t_goal_distance`` (T-to-goal, the task-progress term) and ``arm_to_t_distance``
(arm-to-T, the approach term). The verifier value is the negated sum of the two.
"""
import numpy as np

# Canonical T keypoints in block-local coordinates, matching PushTEnv.add_tee
# (scale=30, length=4): the top bar (4 verts) + the stem (4 verts).
_SCALE = 30
_LENGTH = 4
T_VERTS = np.array([
    (-_LENGTH * _SCALE / 2, _SCALE), (_LENGTH * _SCALE / 2, _SCALE),
    (_LENGTH * _SCALE / 2, 0.0), (-_LENGTH * _SCALE / 2, 0.0),   # top bar
    (-_SCALE / 2, _SCALE), (-_SCALE / 2, _LENGTH * _SCALE),
    (_SCALE / 2, _LENGTH * _SCALE), (_SCALE / 2, _SCALE),        # stem
], dtype=np.float32)  # (8, 2)

# Fixed goal pose of the T (matches PushTEnv._setup: goal_pose = [256, 256, pi/4]).
GOAL_POSE = np.array([256.0, 256.0, np.pi / 4], dtype=np.float32)
N_KEYPOINTS = T_VERTS.shape[0]      # 8
FEEDBACK_DIM = 2 * N_KEYPOINTS      # 16


def keypoints_at_pose(pose):
    """Transform the canonical T keypoints by a pose.

    Args:
        pose: (..., 3) array of ``[x, y, theta]``.
    Returns:
        (..., N_KEYPOINTS, 2) world-frame keypoints, ``world = R(theta) @ v + [x, y]``.
    """
    pose = np.asarray(pose, dtype=np.float32)
    x = pose[..., 0]
    y = pose[..., 1]
    theta = pose[..., 2]
    cos = np.cos(theta)[..., None]   # (..., 1)
    sin = np.sin(theta)[..., None]
    vx = T_VERTS[:, 0]               # (N,)
    vy = T_VERTS[:, 1]
    kx = cos * vx - sin * vy + x[..., None]   # (..., N)
    ky = sin * vx + cos * vy + y[..., None]
    return np.stack([kx, ky], axis=-1)         # (..., N, 2)


# The goal T's keypoints. Constant, so it is hoisted rather than recomputed at every call
# site; every function below is defined relative to it.
GOAL_KEYPOINTS = keypoints_at_pose(GOAL_POSE)   # (N_KEYPOINTS, 2)


def keypoints_from_feedback(feedback):
    """(..., FEEDBACK_DIM) -> (..., N_KEYPOINTS, 2) achieved T keypoints.

    ``feedback = goal_kp - achieved_kp`` by definition, so ``achieved_kp = goal_kp -
    feedback`` -- an exact identity, no pose fit needed (unlike ``block_pose_from_feedback``,
    which recovers the *pose* and needs a Kabsch solve). Everything that only needs the
    achieved geometry should come through here.
    """
    feedback = np.asarray(feedback, dtype=np.float32)
    return GOAL_KEYPOINTS - feedback.reshape(*feedback.shape[:-1], N_KEYPOINTS, 2)


def t_goal_distance(feedback):
    """Mean per-keypoint distance of the achieved T from the goal T, in pixels.

    Args:
        feedback: (..., FEEDBACK_DIM) goal-vs-achieved keypoint displacement.
    Returns:
        (...) mean over keypoints of the per-keypoint EUCLIDEAN distance (not squared).
        ``0`` iff the block is at the goal pose. This is the task-progress half of the
        verifier value (see pusht_verifier).
    """
    feedback = np.asarray(feedback, dtype=np.float32)
    disp = feedback.reshape(*feedback.shape[:-1], N_KEYPOINTS, 2)
    return np.linalg.norm(disp, axis=-1).mean(axis=-1)


def t_center_from_feedback(feedback):
    """(..., FEEDBACK_DIM) -> (..., 2) centre of the achieved T, in pixels.

    The centre is the CENTROID of the achieved keypoints, not the pymunk body origin: the
    body origin sits at the bar/stem junction (block-local (0, 0)) while the vertex
    centroid is at block-local (0, 45), nearer the middle of the shape. Defining it off
    ``feedback`` -- rather than off the block pose -- is what lets the torch-side rescale
    in PushTSearchMixin._normalize_value mirror this exactly from the same inputs.
    """
    return keypoints_from_feedback(feedback).mean(axis=-2)


def arm_to_t_distance(agent_pos, feedback):
    """Distance from the arm (end-effector) to the centre of the achieved T, in pixels.

    Args:
        agent_pos: (..., 2) arm position in the 512-px env frame.
        feedback: (..., FEEDBACK_DIM) goal-vs-achieved keypoint displacement.
    Returns:
        (...) Euclidean distance. This is the approach half of the verifier value: it is
        the only term that varies while the arm has not yet reached the block, which is
        the whole reason it exists (see pusht_verifier).
    """
    agent_pos = np.asarray(agent_pos, dtype=np.float32)
    return np.linalg.norm(agent_pos - t_center_from_feedback(feedback), axis=-1)


def compute_feedback_from_pose(block_pose3, goal_pose=GOAL_POSE):
    """Per-step feedback = flattened ``goal_kp - achieved_kp``.

    Args:
        block_pose3: (..., 3) achieved block pose ``[x, y, theta]``.
        goal_pose: (3,) goal pose; defaults to the fixed PushT goal.
    Returns:
        (..., FEEDBACK_DIM) float32; all-zeros iff ``block_pose3 == goal_pose``.
    """
    block_pose3 = np.asarray(block_pose3, dtype=np.float32)
    goal_kp = keypoints_at_pose(goal_pose)          # (N, 2)
    achieved_kp = keypoints_at_pose(block_pose3)    # (..., N, 2)
    disp = goal_kp - achieved_kp                    # (..., N, 2)
    return disp.reshape(*block_pose3.shape[:-1], FEEDBACK_DIM).astype(np.float32)


def block_pose_from_feedback(feedback):
    """Recover block pose [x, y, theta] from the feedback signal.

    feedback = goal_kp - achieved_kp, so achieved_kp = goal_kp - feedback; the block pose
    is then the rigid transform mapping the canonical T keypoints (T_VERTS) onto
    achieved_kp, recovered per-sample with a 2D Kabsch fit. Exact inverse of
    ``compute_feedback_from_pose``. This is how anything needing the block pose (the
    verifier's sim resets, the online policy's context init states) gets it from the
    declared ``feedback`` obs key, so no privileged reset state has to ride in the obs.

    Args:
        feedback: (..., 16) goal-vs-achieved keypoint displacement.
    Returns:
        (..., 3) block pose [x, y, theta].
    """
    feedback = np.asarray(feedback, dtype=np.float64)
    lead = feedback.shape[:-1]
    goal_kp = keypoints_at_pose(GOAL_POSE).astype(np.float64)          # (N, 2)
    achieved = goal_kp - feedback.reshape(*lead, N_KEYPOINTS, 2)       # (..., N, 2)

    P = T_VERTS.astype(np.float64)
    Pc = P - P.mean(axis=0)                                            # (N, 2)
    ach_mean = achieved.mean(axis=-2, keepdims=True)                   # (..., 1, 2)
    Qc = achieved - ach_mean                                           # (..., N, 2)

    # H = Pc^T Qc  (..., 2, 2); Q_i ~= R P_i
    H = np.einsum('ki,...kj->...ij', Pc, Qc)
    U, _, Vt = np.linalg.svd(H)
    V = np.swapaxes(Vt, -1, -2)
    Ut = np.swapaxes(U, -1, -2)
    d = np.sign(np.linalg.det(np.matmul(V, Ut)))
    D = np.zeros(H.shape, dtype=np.float64)
    D[..., 0, 0] = 1.0
    D[..., 1, 1] = d
    R = np.matmul(np.matmul(V, D), Ut)                                 # (..., 2, 2)

    theta = np.arctan2(R[..., 1, 0], R[..., 0, 0])                     # (...)
    pos = ach_mean[..., 0, :] - np.einsum('...ij,j->...i', R, P.mean(axis=0))
    return np.concatenate([pos, theta[..., None]], axis=-1)           # (..., 3)
