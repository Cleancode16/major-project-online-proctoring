"""
V6 face verification.
  (a) LFW calibration summary (tests/results/V6_face_calibration.json, made by calibrate_face_threshold).
  (b) real clip, correct reference: no IDENTITY_MISMATCH (v1 and v2).
  (c) real IMPOSTOR case: reference = the second person's face, cropped from frame 360 of the clip.
      The candidate on camera is then the wrong person -> IDENTITY_MISMATCH expected for frames 0-323.
      Frames 324-376 are AMBIGUOUS: the enrolled person is only half in view with his head rolled ~60 deg,
      so neither "verified" nor "mismatch" is a fair expectation; how they came out is printed for the record.
  (d) unrelated reference (an LFW photo of someone else) -> every analysed sample is a mismatch.

Run:  .venv\\Scripts\\python.exe -m tests.test_v6_face
"""

import json
import sys

import cv2
import numpy as np

import config as C
from tests import eval_video, paths
from video import face, models

INTRUDER_REF = paths.DATA / "intruder_ref.jpg"
OTHER_REF = paths.DATA / "lfw_other_ref.jpg"


def make_refs():
    if not INTRUDER_REF.exists():
        cap = cv2.VideoCapture(str(paths.SAMPLE_VIDEO))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 348)   # both faces are detected in this frame
        _, f = cap.read()
        cap.release()
        faces = face.detect_faces(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), models.face_detector(0.6))
        assert len(faces) == 2, f"expected 2 faces in frame 348, got {len(faces)}"
        fc = C.get_config("v2")["face"]
        cand, _, _ = face.reference_from_image(cv2.imread(str(paths.SAMPLE_REF)), fc)
        # the second person = the face LEAST similar to the candidate (not a position guess)
        x, y, w, h = min(faces, key=lambda d: face.compare(face.embed(f, d["bbox"], fc["crop_margin"]), cand, fc)[0])["bbox"]
        pad = int(0.6 * w)
        cv2.imwrite(str(INTRUDER_REF), f[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad])
    if not OTHER_REF.exists():
        from sklearn.datasets import fetch_lfw_pairs
        d = fetch_lfw_pairs(subset="test", color=True, resize=1.0, slice_=None, data_home=str(paths.DATA / "lfw"))
        img = d.pairs[0, 0]
        img = (img * 255 if img.max() <= 1 else img).astype(np.uint8)
        cv2.imwrite(str(OTHER_REF), cv2.cvtColor(cv2.resize(img, (500, 500)), cv2.COLOR_RGB2BGR))


def labels_with(identity_pos, identity_amb):
    lab = json.loads((paths.DATA / "sample_clip_labels.json").read_text(encoding="utf-8-sig"))
    lab["positive"]["IDENTITY_MISMATCH"] = identity_pos
    lab["ambiguous"]["IDENTITY_MISMATCH"] = identity_amb
    p = paths.RESULTS / "_tmp_labels.json"
    p.write_text(json.dumps(lab))
    return p


def run_case(name, ref, labels):
    out = {}
    for profile in ("legacy", "v2"):
        summary, res = eval_video.run(profile, None, paths.SAMPLE_VIDEO, labels, ref)
        sc = summary["scores"].get(C.IDENTITY_MISMATCH)
        sims = [s.get("identity_score") for s in res["samples"] if s.get("identity_score") is not None]
        out[profile] = sc
        print(f"({name}) [{profile}] IDENTITY_MISMATCH {sc}; score range {min(sims):.3f}..{max(sims):.3f}")
    return out


def main():
    make_refs()
    cal = json.loads((paths.RESULTS / "V6_face_calibration.json").read_text())
    v1, v2 = cal["v1_first_nomargin"], cal["v2_largest_m0.30"]
    print(f"(a) LFW test: v1 acc {v1['test_at_cos0.5_(v1_rule)']['acc']:.3f} FRR {v1['test_at_cos0.5_(v1_rule)']['FRR']:.3f}"
          f" | v2 acc {v2['test_at_train_threshold']['acc']:.3f} FRR {v2['test_at_train_threshold']['FRR']:.3f}")
    ok_a = v2["test_at_train_threshold"]["acc"] > v1["test_at_cos0.5_(v1_rule)"]["acc"]

    b = run_case("b correct ref", paths.SAMPLE_REF, labels_with([], []))
    c = run_case("c impostor ref", INTRUDER_REF, labels_with([[0, 323]], [[324, 376]]))
    d = run_case("d unrelated ref", OTHER_REF, labels_with([[0, 376]], []))
    ok_b = b["v2"]["FP"] == 0
    ok_c = c["v2"]["recall"] is not None and c["v2"]["recall"] >= 0.9 and c["v2"]["FP"] == 0
    ok_d = d["v2"]["recall"] == 1.0
    r = {"a": ok_a, "b": ok_b, "c": ok_c, "d": ok_d}
    print("V6 FACE:", "PASS" if all(r.values()) else f"FAIL {r}")
    return 0 if all(r.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
