"""
Build SYNTHETIC test clips (derived from the real sample clip) + their label files in tests/data/synthetic.

  noface_clip.mp4   frames 104-193 (3.5-6.5 s): the top 65% of the frame (face and head) is blacked out,
                    the candidate's torso stays visible. FACE_NOT_VISIBLE positive there; all other labels
                    inherited from the real clip.

Run:  .venv\\Scripts\\python.exe -m tests.make_synthetic_video
"""

import copy
import json
import sys

import cv2

from tests import paths

OUT = paths.DATA / "synthetic"


def write_clip(name, transform):
    cap = cv2.VideoCapture(str(paths.SAMPLE_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)) // 2, int(cap.get(4)) // 2  # 960x540 keeps files small
    out_path = OUT / name
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (w, h))
        vw.write(transform(idx, f))
        idx += 1
    vw.release()
    cap.release()
    return out_path, idx


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads((paths.DATA / "sample_clip_labels.json").read_text(encoding="utf-8-sig"))

    def noface(i, f):
        if 104 <= i <= 193:
            f = f.copy()
            f[: int(f.shape[0] * 0.65)] = 0
        return f

    path, n = write_clip("noface_clip.mp4", noface)
    lab = copy.deepcopy(base)
    lab["_about"] = "SYNTHETIC: real sample clip at 960x540 with the head region blacked out in frames 104-193."
    lab["positive"]["FACE_NOT_VISIBLE"] = [[104, 193]]
    lab["ambiguous"]["FACE_NOT_VISIBLE"] = [[101, 103], [194, 196]]  # 1 sample of tolerance at each edge
    # pose/gaze/identity are undefined while the face is hidden; nothing else changes
    (OUT / "noface_clip.labels.json").write_text(json.dumps(lab, indent=1))
    print(f"wrote {path} ({n} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
