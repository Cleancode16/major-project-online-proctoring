"""
V1: head-pose Euler extraction.
  (a) synthetic: rotate the 3D model by known angles, project it with the same camera model,
      run the extractor on the projected points, compare recovered angles (legacy vs fixed);
  (b) direction: a face turned towards image right must give yaw > 0, chin down pitch > 0;
  (c) real: frontal frames of the sample clip must give |pitch| < 15 deg with the fixed model.

Run:  .venv\\Scripts\\python.exe -m tests.test_v1_head_pose
"""

import itertools
import sys
from types import SimpleNamespace

import cv2
import numpy as np

from tests import paths
from video import head_pose as hp
from video import models

W, H = 1280, 720


def rot(yaw_img_right, pitch_down, roll_cw):
    """Rotation for the documented convention (yaw>0 = towards image right)."""
    y, p, r = np.radians([-yaw_img_right, pitch_down, roll_cw])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def fake_landmarks(yaw, pitch, roll):
    r = rot(yaw, pitch, roll)
    rvec, _ = cv2.Rodrigues(r)
    pts, _ = cv2.projectPoints(hp.MODEL_3D, rvec, np.array([0.0, 0.0, 600.0]), hp._camera(W, H), np.zeros(4))
    pts = pts.reshape(-1, 2)
    lm = [SimpleNamespace(x=0.0, y=0.0) for _ in range(478)]
    for i, idx in enumerate(hp.LANDMARK_IDS):
        lm[idx] = SimpleNamespace(x=pts[i, 0] / W, y=pts[i, 1] / H)
    return lm, pts


def synthetic():
    grid = list(itertools.product([-30, -15, 0, 15, 30], [-20, -10, 0, 10, 20], [-15, 0, 15]))
    err = {"fixed": [], "legacy": []}
    for y, p, r in grid:
        lm, _ = fake_landmarks(y, p, r)
        fy, fp, fr = hp.head_pose(lm, W, H, "fixed")
        err["fixed"].append(max(abs(fy - y), abs(fp - p), abs(fr - r)))
        ly, lp, lr = hp.head_pose(lm, W, H, "legacy")
        err["legacy"].append((ly, lp, lr))
    max_err = max(err["fixed"])
    legacy_pitch = [v[1] for v in err["legacy"]]
    print(f"(a) synthetic grid {len(grid)} poses: fixed max abs error = {max_err:.3f} deg")
    print(f"    legacy pitch range on same poses: {min(legacy_pitch):.1f} .. {max(legacy_pitch):.1f} deg "
          f"(true pitch range -20..20)")
    lm0, _ = fake_landmarks(0, 0, 0)
    print(f"    frontal face: fixed={tuple(round(v, 2) for v in hp.head_pose(lm0, W, H, 'fixed'))} "
          f"legacy={tuple(round(v, 1) for v in hp.head_pose(lm0, W, H, 'legacy'))}")
    return max_err < 1.0


def direction():
    _, pts = fake_landmarks(25, 0, 0)
    nose_x, eyes_mid_x = pts[0, 0], (pts[2, 0] + pts[3, 0]) / 2
    # The nose tip protrudes towards the camera, so when the head nods down it moves down
    # relative to the (recessed) eye and mouth corners; nodding up moves it up.
    def nose_drop(p):
        _, q = fake_landmarks(0, p, 0)
        return q[0, 1] - q[2:, 1].mean()
    down, level, up = nose_drop(20), nose_drop(0), nose_drop(-20)
    ok = nose_x > eyes_mid_x and down > level > up
    print(f"(b) yaw=+25: nose x {nose_x:.0f} > eye-mid x {eyes_mid_x:.0f} (turned to image right) -> "
          f"{nose_x > eyes_mid_x}; nose offset below eye/mouth corners: pitch +20 -> {down:.1f}px, "
          f"0 -> {level:.1f}px, -20 -> {up:.1f}px (pitch>0 = chin down) -> {down > level > up}")
    return ok


def real_frames():
    mesh = models.face_mesh(max_faces=1, static=True)
    cap = cv2.VideoCapture(str(paths.SAMPLE_VIDEO))
    out = []
    for idx in range(0, 240, 20):  # frames 0-239: candidate faces the screen, no phone yet
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        res = mesh.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            continue
        lm = res.multi_face_landmarks[0].landmark
        h, w = f.shape[:2]
        out.append((idx, hp.head_pose(lm, w, h, "fixed"), hp.head_pose(lm, w, h, "legacy")))
    cap.release()
    fixed_p = [v[1][1] for v in out]
    legacy_p = [v[2][1] for v in out]
    print(f"(c) real frontal frames n={len(out)}: fixed pitch {min(fixed_p):.1f}..{max(fixed_p):.1f}, "
          f"fixed yaw {min(v[1][0] for v in out):.1f}..{max(v[1][0] for v in out):.1f}; "
          f"legacy pitch {min(legacy_p):.1f}..{max(legacy_p):.1f}")
    return all(abs(p) < 15 for p in fixed_p)


def main():
    results = {"synthetic": synthetic(), "direction": direction(), "real": real_frames()}
    print("V1 HEAD POSE:", "PASS" if all(results.values()) else f"FAIL {results}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
