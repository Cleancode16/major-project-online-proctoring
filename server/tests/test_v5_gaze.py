"""
V5 gaze.
  (a) synthetic landmarks: iris shifted towards image-left in BOTH eyes must give the same h in both
      eyes (< 0.5) and label "Looking Right" (candidate's right, unmirrored camera); legacy check too.
  (b) head compensation: head turned 20 deg with eyes counter-rotated -> not away; eyes 35 deg away -> away.
  (c) real mirror symmetry: horizontally flipped frames must give h' ~ 1 - h.
  (d) real false-positive rate: sample clip t < 8 s (candidate looks at the screen, no phone yet).

Run:  .venv\\Scripts\\python.exe -m tests.test_v5_gaze
"""

import sys
from types import SimpleNamespace

import cv2
import numpy as np

import config as C
from tests import paths
from video import face, gaze, models, pipeline

W, H = 1000, 1000


def synth_eyes(shift):
    """Two eyes 60 px wide; iris displaced by `shift` (fraction of eye width, + = image right)."""
    lm = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]

    def put(i, x, y):
        lm[i] = SimpleNamespace(x=x / W, y=y / H)

    for (a, b, iris, top, bot), x0 in zip(gaze.EYES, (400, 540)):
        put(a, x0, 400); put(b, x0 + 60, 400)
        put(top, x0 + 30, 390); put(bot, x0 + 30, 410)
        put(iris, x0 + 30 + shift * 60, 400)
    return lm


def synthetic():
    gc = C.get_config("v2")["gaze"]
    r = gaze.eye_ratios(synth_eyes(-0.2), W, H)
    per_eye = []
    for a, b, iris, _, _ in gaze.EYES:
        l = synth_eyes(-0.2)
        per_eye.append((l[iris].x - l[a].x) / (l[b].x - l[a].x))
    away, label, gh, _ = gaze.gaze_decision(r["h"] - 0.5, 0.0, 0.0, 0.0, gc)
    leg = gaze.gaze_legacy(synth_eyes(-0.2), W)
    leg_c = gaze.gaze_legacy(synth_eyes(0.0), W)
    print(f"(a) iris 0.2 towards image-left: v2 h={r['h']:.3f} (per-eye {per_eye[0]:.2f}/{per_eye[1]:.2f}) "
          f"-> {label} ({gh:+.1f} deg); legacy left={leg['left_ratio']:.2f} right={leg['right_ratio']:.2f} "
          f"-> {leg['label']}; legacy with centred iris -> {leg_c['label']} (avg {leg_c['h']:.2f})")
    ok_a = abs(r["h"] - 0.3) < 1e-6 and label == "Looking Right" and away

    # (b) head compensation
    turn = 20.0
    comp = gaze.gaze_decision(-turn / gc["deg_per_ratio_h"], 0, turn, 0, gc)
    eyes_away = gaze.gaze_decision(35 / gc["deg_per_ratio_h"], 0, 0, 0, gc)
    head_and_eyes = gaze.gaze_decision(0.0, 0, 30, 0, gc)
    down = gaze.gaze_decision(0, 0.6, 0, 0, gc)
    print(f"(b) head +20 / eyes counter-rotated -> away={comp[0]} ({comp[2]:+.1f} deg); eyes +35 -> "
          f"away={eyes_away[0]} {eyes_away[1]}; head +30 eyes fixed -> away={head_and_eyes[0]} "
          f"(left to HEAD_POSE); eyes down dv=0.6 -> away={down[0]} {down[1]}")
    ok_b = (not comp[0]) and eyes_away[0] and eyes_away[1] == "Looking Left" and not head_and_eyes[0] and down[0]
    return ok_a, ok_b


def mirror():
    mesh = models.face_mesh(max_faces=1, static=True)
    cap = cv2.VideoCapture(str(paths.SAMPLE_VIDEO))
    errs, legacy_errs = [], []
    for idx in range(0, 240, 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        _, f = cap.read()
        vals = []
        for img in (f, cv2.flip(f, 1)):
            res = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            lm = res.multi_face_landmarks[0].landmark
            vals.append((gaze.eye_ratios(lm, img.shape[1], img.shape[0])["h"], gaze.gaze_legacy(lm, img.shape[1])["h"]))
        errs.append(abs(vals[0][0] - (1 - vals[1][0])))
        legacy_errs.append(abs(vals[0][1] - (1 - vals[1][1])))
    cap.release()
    print(f"(c) mirror symmetry |h - (1 - h_flipped)| over {len(errs)} real frames: v2 max={max(errs):.3f}, "
          f"legacy max={max(legacy_errs):.3f}")
    return max(errs) < 0.05


def clip_fp():
    out = {}
    for profile in ("legacy", "v2"):
        cfg = C.get_config(profile)
        ref, _, _ = face.reference_from_image(cv2.imread(str(paths.SAMPLE_REF)), cfg["face"])
        res = pipeline.analyze_video(paths.SAMPLE_VIDEO, ref, cfg)
        win = [s for s in res["samples"] if s["t"] < 8.0 and (profile == "legacy" or s["phase"] == "analysis")]
        flagged = sum(1 for s in win if s.get("flags", {}).get(C.GAZE_AWAY))
        out[profile] = (flagged, len(win))
        hs = [s["gaze"]["h"] for s in win if s.get("gaze") and s["gaze"]["h"] is not None]
        gh = [s["gaze"].get("gaze_h_deg") for s in win if s.get("gaze") and s["gaze"].get("gaze_h_deg") is not None]
        extra = f", combined gaze_h {min(gh):+.1f}..{max(gh):+.1f} deg" if gh else ""
        print(f"(d) [{profile}] t<8 s, candidate looking at screen: GAZE_AWAY {flagged}/{len(win)} samples "
              f"({flagged / max(1, len(win)) * 100:.0f}%); h range {min(hs):.3f}..{max(hs):.3f}{extra}")
        if profile == "v2":
            print(f"    v2 baseline: {res['baseline']}")
    return out["v2"][0] == 0


def main():
    ok_a, ok_b = synthetic()
    results = {"a_labels": ok_a, "b_compensation": ok_b, "c_mirror": mirror(), "d_clip_fp": clip_fp()}
    print("V5 GAZE:", "PASS" if all(results.values()) else f"FAIL {results}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
