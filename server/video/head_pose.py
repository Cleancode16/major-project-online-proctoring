"""Head pose from MediaPipe landmarks via solvePnP (flow.txt §A.5)."""

import cv2
import numpy as np

# nose tip, chin, image-left eye outer corner, image-right eye outer corner,
# image-left mouth corner, image-right mouth corner
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# v1 model: y axis points UP, z towards the camera (generic face model).
MODEL_3D_LEGACY = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1),
], dtype=np.float64)


def _camera(w, h):
    return np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)


def wrap180(a):
    """Map an angle (deg) into [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def head_pose_legacy(landmarks, w, h):
    """v1 extraction, kept verbatim for the legacy profile."""
    image_2d = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LANDMARK_IDS],
                        dtype=np.float64)
    _, rvec, _ = cv2.solvePnP(MODEL_3D_LEGACY, image_2d, _camera(w, h), np.zeros((4, 1)),
                              flags=cv2.SOLVEPNP_ITERATIVE)
    r, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    yaw = np.degrees(np.arctan2(-r[2, 0], sy))
    roll = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return float(yaw), float(pitch), float(roll)


# v2 model: same points in the OpenCV camera frame (x right, y down, z away from camera),
# so a face looking straight at the camera has R ~ identity.
MODEL_3D = MODEL_3D_LEGACY * np.array([1.0, -1.0, -1.0])


def euler_from_matrix(r):
    """(yaw, pitch, roll) in degrees for R = Rz(roll) Ry(yaw') Rx(pitch), with yaw = -yaw'
    so that yaw > 0 means the face turned towards image right (flow.txt §A.5)."""
    sy = np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    yaw = -np.degrees(np.arctan2(-r[2, 0], sy))
    roll = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return float(yaw), float(pitch), float(roll)


def head_pose_fixed(landmarks, w, h):
    image_2d = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LANDMARK_IDS], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, image_2d, _camera(w, h), np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    r, _ = cv2.Rodrigues(rvec)
    return euler_from_matrix(r)


# ---------------------------------------------------------------- HP2: person-specific template (flow.txt §A.5)
def rigid_points(landmarks, w, h, ids):
    """Camera-frame 3D points (X = x*w, Y = y*h, Z = z*w) of the rigid landmark subset."""
    return np.array([(landmarks[i].x * w, landmarks[i].y * h, landmarks[i].z * w) for i in ids], dtype=np.float64)


def normalize_points(p):
    """Centre on the centroid and scale to unit RMS radius (removes translation and distance to camera)."""
    c = p - p.mean(axis=0)
    return c / max(float(np.sqrt((c ** 2).sum(axis=1).mean())), 1e-9)


def build_template(point_sets):
    """Per-coordinate median of the normalised calibration point sets."""
    return np.median(np.stack([normalize_points(p) for p in point_sets]), axis=0)


def relative_rotation(template, points):
    """Kabsch / Umeyama: rotation R minimising sum ||R t_i - q_i||^2 (reflection-corrected)."""
    q = normalize_points(points)
    u, _, vt = np.linalg.svd(template.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def head_pose_template(template, landmarks, w, h, ids):
    """(yaw, pitch, roll) of the head RELATIVE to the calibration template, same convention as head_pose_fixed."""
    return euler_from_matrix(relative_rotation(template, rigid_points(landmarks, w, h, ids)))


def head_pose(landmarks, w, h, mode="fixed"):
    """Return (yaw, pitch, roll) in degrees, or None if PnP failed."""
    if mode == "legacy":
        return head_pose_legacy(landmarks, w, h)
    return head_pose_fixed(landmarks, w, h)
