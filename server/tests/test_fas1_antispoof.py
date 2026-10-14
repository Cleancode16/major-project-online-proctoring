"""
FAS1 face anti-spoofing (MiniFASNet, flow.txt §A.7b).

  (a) weights present, strict load of both models
  (b) crop box = upstream CropImage._get_new_box (centre, scale, shift into the frame)
  (c) real faces of the user's data score high: enrollment photos + sample clip frames (>= 0.5 on 90%)
  (d) a printed-photo style attack cannot be produced here: a flat, re-photographed copy of a frame (blurred,
      downscaled, JPEG q=20, moire-like stripes) is scored and printed for information only
  (e) decision rule: mean over window_s below real_thr -> FACE_SPOOF raw flag; FACE_SPOOF is critical, has
      an evidence group and a session limit

Run (from server/):  ..\\.venv\\Scripts\\python.exe -m tests.test_fas1_antispoof
"""

import glob
import sys

import cv2
import numpy as np

import config as C
from aggregation import evidence
from tests import paths
from video import antispoof, face, models

FAILS = []


def check(name, ok, info=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        FAILS.append(name)


def main():
    cfg = C.get_config("v2")
    print("(a) weights")
    fas = models.antispoof(cfg["antispoof"]["models_dir"])
    check("both models loaded", len(fas.models) == 2)

    print("(b) crop box")
    check("centred 2.7x box", antispoof.crop_box(640, 480, (270, 190, 100, 100), 2.7) == (185, 105, 455, 375))
    x1, y1, x2, y2 = antispoof.crop_box(640, 480, (0, 0, 100, 100), 4.0)
    check("box shifted into the frame at the corner", x1 == 0 and y1 == 0 and x2 <= 639 and y2 <= 479)

    print("(c) real faces")
    det = models.face_detector(0.6)

    def score(img):
        fs = face.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), det)
        return fas.real_score(img, max(fs, key=lambda f: f["bbox"][2] * f["bbox"][3])["bbox"]) if fs else None

    real = [score(cv2.imread(p)) for p in glob.glob(str(C.ENROLL_DIR / "*" / "face.jpg"))]
    frames = []
    if paths.SAMPLE_VIDEO.exists():
        cap = cv2.VideoCapture(str(paths.SAMPLE_VIDEO))
        for i in range(0, 370, 30):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if ok:
                frames.append(f)
                real.append(score(f))
    real = [r for r in real if r is not None]
    check("real faces measured", len(real) >= 5, f"(n={len(real)})")
    if real:
        frac = float(np.mean(np.array(real) >= cfg["antispoof"]["real_thr"]))
        check(">= 90% of real faces above real_thr", frac >= 0.9, f"({100 * frac:.0f}%, median {np.median(real):.2f})")

    print("(d) simulated re-photographed copy (information only, not a real attack)")
    if frames:
        f = frames[0]
        h, w = f.shape[:2]
        flat = cv2.resize(cv2.GaussianBlur(cv2.resize(f, (w // 4, h // 4)), (5, 5), 0), (w, h))
        flat = cv2.imdecode(cv2.imencode(".jpg", flat, [cv2.IMWRITE_JPEG_QUALITY, 20])[1], cv2.IMREAD_COLOR)
        stripes = (8 * np.sin(np.arange(h) * 1.3))[:, None, None]
        flat = np.clip(flat.astype(np.float32) + stripes, 0, 255).astype(np.uint8)
        print(f"     real frame {score(f):.2f} -> re-photographed copy {score(flat) or float('nan'):.2f}")

    print("(e) decision rule / integration")
    ac = cfg["antispoof"]
    seq = [0.9, 0.2, 0.9, 0.9]                      # one bad frame inside real ones
    n = 3
    means = [float(np.mean(seq[max(0, i - n + 1):i + 1])) for i in range(len(seq))]
    check("single low frame is averaged away", all(m >= ac["real_thr"] for m in means), str([round(m, 2) for m in means]))
    check("FACE_SPOOF is critical", cfg["aggregation"]["severity"][C.FACE_SPOOF] == "critical")
    check("spoof evidence group + limit", "spoof" in evidence.GROUPS and "spoof" in cfg["evidence"]["limits"])
    print("FAS1:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
