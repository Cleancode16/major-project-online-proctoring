"""
GZ3 L2CS-Net gaze (flow.txt §A.6 "L2CS-Net gaze").

  (a) decision rule: +-15 deg thresholds, hysteresis, direction labels, optional eye-in-head minimum
  (b) weights: file present, SHA-256 matches, strict load (all 324 tensors)
  (c) mirror symmetry on the real sample clip: flipping the frame must flip the horizontal angle and keep the
      vertical one (checks the axis / sign mapping h = -yaw, v = -pitch)
  (d) offline smoothing: centred moving average keeps a 1-sample spike below the threshold, a 3-sample look stays
  (e) timing per face at the live (224) and recorded (448) input sizes

Run (from server/):  ..\\.venv\\Scripts\\python.exe -m tests.test_gz3_l2cs
"""

import sys
import time

import cv2
import numpy as np

import config as C
from tests import paths
from video import face, gaze, l2cs, models
from video.temporal import centered_mean

FAILS = []


def check(name, ok, info=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        FAILS.append(name)


def a_decision():
    print("(a) decision rule")
    lc = C.get_config("v2")["gaze"]["l2cs"]
    dec = lambda h, v, prev=False, ry=None, lc_=lc: gaze.gaze_decision_deg(h, v, ry, None, lc_, prev_away=prev)  # noqa: E731
    check("10 deg right -> centre", not dec(10, 0)[0])
    check("16 deg towards image right -> away, Looking Left", dec(16, 0)[:2] == (True, "Looking Left"))
    check("-16 deg -> Looking Right", dec(-16, 0)[1] == "Looking Right")
    check("16 deg down -> Looking Down", dec(0, 16)[:2] == (True, "Looking Down"))
    check("-16 deg -> Looking Up", dec(0, -16)[1] == "Looking Up")
    check("hysteresis: 14 deg stays away after away", dec(14, 0, prev=True)[0] and not dec(14, 0)[0])
    strict = dict(lc, min_eye_deg=8.0)
    check("min_eye_deg: head turned 20, eyes follow -> not gaze", not dec(20, 0, ry=18, lc_=strict)[0])
    check("min_eye_deg: head 5, gaze 20 -> gaze", dec(20, 0, ry=5, lc_=strict)[0])


def b_weights():
    print("(b) weights")
    path = C.ROOT / C.get_config("v2")["gaze"]["l2cs"]["weights"]
    check("weights file present", path.exists(), str(path))
    if path.exists():
        check("SHA-256 matches", l2cs.sha256(path) == l2cs.WEIGHTS_SHA256)
        from safetensors.torch import load_file
        check("324 tensors", len(load_file(str(path))) == 324)


def _first_face(img, det):
    faces = face.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), det)
    return max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])["bbox"] if faces else None


def c_mirror():
    print("(c) mirror symmetry on the real clip")
    if not paths.SAMPLE_VIDEO.exists():
        check("sample clip present", False, str(paths.SAMPLE_VIDEO))
        return
    lc = C.get_config("v2")["gaze"]["l2cs"]
    model = models.l2cs(lc["weights"], lc["input_size"], lc.get("torch_threads"))
    det = models.face_detector(0.6)
    cap = cv2.VideoCapture(str(paths.SAMPLE_VIDEO))
    dh, dv = [], []
    for idx in range(0, 240, 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            continue
        g = []
        for img in (f, cv2.flip(f, 1)):
            box = _first_face(img, det)
            g.append(None if box is None else model.predict(img, box))
        if g[0] and g[1]:
            dh.append(abs(g[0]["h"] + g[1]["h"]))      # flipped: h -> -h
            dv.append(abs(g[0]["v"] - g[1]["v"]))      # flipped: v unchanged
            print(f"     frame {idx}: h {g[0]['h']:+.1f} / flipped {g[1]['h']:+.1f}, v {g[0]['v']:+.1f} / {g[1]['v']:+.1f}")
    check("frames measured", len(dh) >= 4, f"({len(dh)})")
    if dh:
        check("horizontal flips sign (mean |h + h_flipped| < 4 deg)", float(np.mean(dh)) < 4.0, f"{np.mean(dh):.2f}")
        check("vertical unchanged (mean |v - v_flipped| < 4 deg)", float(np.mean(dv)) < 4.0, f"{np.mean(dv):.2f}")


def d_smoothing():
    print("(d) offline centred moving average (window 3)")
    win = C.get_config("v2")["gaze"]["smooth_window"]
    spike = centered_mean([0, 0, 0, 30, 0, 0, 0], win)
    look = centered_mean([0, 0, 25, 25, 25, 0, 0], win)
    check("1-sample 30 deg spike -> 10 deg (< 15)", abs(spike[3] - 10.0) < 1e-9)
    check("3-sample 25 deg look keeps its centre (25)", abs(look[3] - 25.0) < 1e-9)


def e_timing():
    print("(e) CPU time per face")
    lc = C.get_config("v2")["gaze"]["l2cs"]
    img = np.full((480, 640, 3), 128, np.uint8)
    for size in (lc["live_input_size"], lc["input_size"]):
        m = models.l2cs(lc["weights"], size, lc.get("torch_threads"))
        ts = []
        for _ in range(10):
            t = time.perf_counter()
            m.predict(img, (200, 120, 200, 240))
            ts.append((time.perf_counter() - t) * 1000)
        print(f"     input {size}: median {np.median(ts):.0f} ms, max {max(ts):.0f} ms")
    check("timing measured", True)


def main():
    a_decision()
    b_weights()
    c_mirror()
    d_smoothing()
    e_timing()
    print("GZ3 L2CS:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
