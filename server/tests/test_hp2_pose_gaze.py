"""
HP2 / GZ2 accuracy upgrade (flow.txt §A.5, §A.6 "Accuracy upgrade"). SYNTHETIC data, written for the user to run.

  (a) head pose: person-specific template (33 rigid landmarks, Kabsch) vs generic 6-point PnP,
      same pixel noise; max / mean angle error over a pose grid.
  (b) hysteresis: a turn that wobbles around the threshold -> 1 event with hysteresis vs several without.
  (c) baseline adaptation: slow posture drift (pitch 0 -> 12 deg over 10 min) gives no false positives,
      a sudden 25 deg nod and a slow deliberate drift to 20 deg are still flagged.
  (d) gaze: corner-line vertical ratio vs lid ratio for a downward look (the upper lid follows the eye);
      width-weighted average vs plain mean when the far eye is foreshortened and noisy.

Run (from server/):  ..\\.venv\\Scripts\\python.exe -m tests.test_hp2_pose_gaze
"""

import sys
from types import SimpleNamespace

import numpy as np

import config as C
from video import gaze
from video import head_pose as hp
from video.pipeline import VideoAnalyzer

W, H = 1280, 720
rng = np.random.default_rng(11)
FAILS = []


def check(name, ok, info=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        FAILS.append(name)


def rot(yaw, pitch, roll):
    """Rotation for the project convention (yaw > 0 = towards image right, pitch > 0 = down, roll > 0 = cw)."""
    y, p, r = np.radians([-yaw, pitch, roll])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


# a face-sized rigid point set (mm) for the template ids, plus the 6 PnP model points
IDS = C.get_config("v2")["head_pose"]["rigid_ids"]
FACE = np.c_[rng.uniform(-60, 60, len(IDS)), rng.uniform(-70, 40, len(IDS)), rng.uniform(-30, 20, len(IDS))]


def _project(points_mm, r, dist=600.0):
    """Same pinhole camera as head_pose._camera (focal = W): pixel x, y and a MediaPipe-like z (pixel scale)."""
    cam = points_mm @ r.T + np.array([0.0, 0.0, dist])
    f = float(W)
    u = f * cam[:, 0] / cam[:, 2] + W / 2
    v = f * cam[:, 1] / cam[:, 2] + H / 2
    z = (cam[:, 2] - dist) * f / dist
    return np.c_[u, v, z]


def landmarks_for(yaw, pitch, roll, noise_px):
    """Fake MediaPipe landmarks for a head with the given pose; the same pixel noise on every point."""
    r = rot(yaw, pitch, roll)
    lm = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    for idx, (x, y, z) in zip(IDS, _project(FACE, r) + rng.normal(0, noise_px, (len(IDS), 3))):
        lm[idx] = SimpleNamespace(x=x / W, y=y / H, z=z / W)
    # the generic PnP model points, projected with the same camera and noise
    for idx, (x, y, _) in zip(hp.LANDMARK_IDS, _project(hp.MODEL_3D, r) + rng.normal(0, noise_px, (6, 3))):
        lm[idx] = SimpleNamespace(x=x / W, y=y / H, z=0.0)
    return lm


def a_template_vs_pnp(noise_px=1.5):
    print("(a) head-pose error, template (33 pts) vs PnP (6 pts), noise", noise_px, "px")
    template = hp.build_template([hp.rigid_points(landmarks_for(0, 0, 0, noise_px), W, H, IDS) for _ in range(9)])
    base = np.median([hp.head_pose(landmarks_for(0, 0, 0, noise_px), W, H, "fixed") for _ in range(9)], axis=0)
    errs_t, errs_p = [], []
    for yaw in (-30, -15, 0, 15, 30):
        for pitch in (-20, -10, 0, 10, 20):
            for roll in (-10, 0, 10):
                lm = landmarks_for(yaw, pitch, roll, noise_px)
                t = hp.head_pose_template(template, lm, W, H, IDS)
                p = np.array(hp.head_pose(lm, W, H, "fixed")) - base
                truth = np.array([yaw, pitch, roll])
                errs_t.append(np.abs(np.array(t) - truth).max())
                errs_p.append(np.abs(p - truth).max())
    print(f"    template: mean {np.mean(errs_t):.2f} deg, max {np.max(errs_t):.2f} deg | "
          f"PnP: mean {np.mean(errs_p):.2f} deg, max {np.max(errs_p):.2f} deg")
    check("template more accurate than PnP", np.mean(errs_t) < np.mean(errs_p))
    check("template mean error < 2 deg", np.mean(errs_t) < 2.0)


def _run_pose(seq_pitch, hyst, adapt):
    cfg = C.get_config("v2", {"head_pose": {"hyst_deg": hyst, "adapt_tau_s": adapt}})
    va = VideoAnalyzer(np.zeros(512), cfg, 3.0)
    va.baseline = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    for i, p in enumerate(seq_pitch):
        s = {"t": i / 3, "phase": "analysis", "raw": {}}
        va._flag_pose(s, None, (0.0, float(p), 0.0))
        va.samples.append(s)
    flags = [s["flags"][C.HEAD_POSE_DEVIATION] for s in va.finalize()["samples"]]
    runs = sum(1 for i, f in enumerate(flags) if f and (i == 0 or not flags[i - 1]))
    return flags, runs


def b_hysteresis():
    print("(b) hysteresis: 20 s nod wobbling 8-13 deg around the 10 deg pitch threshold")
    seq = [0.0] * 9 + list(10.5 + 2.5 * np.sin(np.arange(60) * 1.3)) + [0.0] * 9
    _, runs_off = _run_pose(seq, 0.0, None)
    _, runs_on = _run_pose(seq, 3.0, None)
    print(f"    events without hysteresis {runs_off}, with hysteresis {runs_on}")
    check("hysteresis merges the wobble into one event", runs_on == 1 and runs_off > 1)


def c_adaptation():
    print("(c) slow posture drift vs real turns (3 fps)")
    n = 1800                                         # 10 min
    drift = list(np.linspace(0, 12, n))              # posture slowly slumps to 12 deg pitch (threshold 10)
    flags_off, _ = _run_pose(drift + [12.0] * 30, 3.0, None)
    flags_on, _ = _run_pose(drift + [12.0] * 30, 3.0, 120.0)
    print(f"    drift false-positive samples: without adaptation {sum(flags_off)}, with {sum(flags_on)}")
    check("adaptation removes drift false positives", sum(flags_off) > 0 and sum(flags_on) == 0)
    nod = [0.0] * 30 + [25.0] * 6 + [0.0] * 30
    flags, _ = _run_pose(nod, 3.0, 120.0)
    check("sudden 25 deg nod still flagged", sum(flags) >= 5)
    slow_cheat = list(np.linspace(0, 20, 900)) + [20.0] * 30
    flags, _ = _run_pose(slow_cheat, 3.0, 120.0)
    check("slow deliberate drift to 20 deg still flagged (cap 8 deg)", flags[-1])


def eyes(iris_dy, lid_follow, far_eye_noise=0.0, far_eye_width=60):
    """Two eyes; iris moved down by iris_dy px; the upper lid moves down by lid_follow * iris_dy."""
    lm = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]

    def put(i, x, y):
        lm[i] = SimpleNamespace(x=x / W, y=y / H, z=0.0)

    for k, ((a, b, iris, top, bot), x0) in enumerate(zip(gaze.EYES, (500, 640))):
        width = 60 if k == 0 else far_eye_width
        put(a, x0, 300); put(b, x0 + width, 300)
        put(top, x0 + width / 2, 290 + lid_follow * iris_dy); put(bot, x0 + width / 2, 310)
        cx = x0 + width / 2 + (rng.normal(0, far_eye_noise) if k == 1 else 0.0)
        for j in range(5):
            dx, dy = [(0, 0), (6, 0), (0, 6), (-6, 0), (0, -6)][j]
            put(iris + j, cx + dx, 300 + iris_dy + dy)
    return lm


def d_gaze():
    print("(d) gaze")
    gc = C.get_config("v2")["gaze"]
    base_l = gaze.eye_ratios(eyes(0, 0.8), W, H, vertical="lids")["v"]
    down_l = gaze.eye_ratios(eyes(6, 0.8), W, H, vertical="lids")["v"]
    base_c = gaze.eye_ratios(eyes(0, 0.8), W, H, vertical="corner_line", iris_mean=True)["v"]
    down_c = gaze.eye_ratios(eyes(6, 0.8), W, H, vertical="corner_line", iris_mean=True)["v"]
    deg_l = (down_l - base_l) * 50.0
    deg_c = (down_c - base_c) * gc["deg_per_ratio_v"]
    true_deg = np.degrees(np.arcsin(6 / 60 * 2.5))    # iris shift / eye width * (30 mm / 12 mm)
    print(f"    6 px downward iris shift (~{true_deg:.0f} deg): lid ratio -> {deg_l:.1f} deg, corner line -> {deg_c:.1f} deg")
    check("corner line closer to the true downward angle than lids", abs(deg_c - true_deg) < abs(deg_l - true_deg))
    e_mean, e_w = [], []
    for _ in range(200):
        lm = eyes(0, 0.0, far_eye_noise=4.0, far_eye_width=30)
        e_mean.append(abs(gaze.eye_ratios(lm, W, H, iris_mean=True, weight_by_width=False)["h"] - 0.5))
        e_w.append(abs(gaze.eye_ratios(lm, W, H, iris_mean=True, weight_by_width=True)["h"] - 0.5))
    print(f"    foreshortened noisy far eye: mean |h error| plain {np.mean(e_mean):.4f}, width-weighted {np.mean(e_w):.4f}")
    check("width weighting reduces the error", np.mean(e_w) < np.mean(e_mean))


def main():
    a_template_vs_pnp()
    b_hysteresis()
    c_adaptation()
    d_gaze()
    print("HP2/GZ2:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
