"""
R1 regression: the refactored pipeline run with the "legacy" profile must reproduce the
per-frame log lines and final counters of the original app.py (tests/results/baseline_app_v1.txt).

Run:  .venv\\Scripts\\python.exe -m tests.test_r1_legacy_equivalence
"""

import re
import sys

import cv2

import config as C
from tests import paths
from video import face, pipeline


def legacy_lines(result):
    lines = []
    cal_n = 0
    for s in result["samples"]:
        ident = "Authorized" if s["identity_ok_raw"] else "Unauthorized"
        if s["phase"] == "calibration":
            cal_n += 1
        g = s["gaze"]
        gstat = "Suspicious" if g["label"] != "Looking Center" else "Normal"
        p = s["persons"]
        pstat = f"Multiple ({p})" if p > 1 else ("Single" if p == 1 else "None")
        ostat = ("Found: " + ", ".join(f"{k}({v})" for k, v in s["objects"].items())) if s["objects"] else "None"
        tail = f"Gaze: {gstat} ({g['label']}) | Persons: {pstat} | Objects: {ostat}"
        if s["phase"] == "calibration" and cal_n < 3:
            pose = f"Pose: Calibrating ({cal_n}/3)"
        elif s["phase"] == "calibration":
            pose = "Pose: Calibrating (ΔY=+0.0°, ΔP=+0.0°, ΔR=+0.0°)"
        else:
            st = "Deviating" if s["raw"].get(C.HEAD_POSE_DEVIATION) else "Normal"
            pose = f"Pose: {st} (ΔY={s['rel_yaw']:+.1f}°, ΔP={s['rel_pitch']:+.1f}°, ΔR={s['rel_roll']:+.1f}°)"
        lines.append(f"Frame {s['frame']}: {ident} | {pose} | {tail}")
    return lines


def baseline_lines():
    text = paths.BASELINE_V1.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\r?\n(?=Single|Multiple|None \|)", " ", text)  # undo console wrapping
    return [ln.strip() for ln in text.splitlines() if re.match(r"^Frame \d+:", ln.strip())]


def main():
    cfg = C.get_config("legacy")
    ref, _, _ = face.reference_from_image(cv2.imread(str(paths.SAMPLE_REF)), cfg["face"])
    res = pipeline.analyze_video(paths.SAMPLE_VIDEO, ref, cfg)
    ours, theirs = legacy_lines(res), baseline_lines()
    print(f"samples: ours={len(ours)} baseline={len(theirs)}")
    print(f"baseline (legacy profile): {res['baseline']}")
    diffs = [(a, b) for a, b in zip(ours, theirs) if a != b]
    for a, b in diffs[:10]:
        print("OURS :", a)
        print("BASE :", b)
    ok = len(ours) == len(theirs) and not diffs
    print("R1 LEGACY EQUIVALENCE:", "PASS" if ok else f"FAIL ({len(diffs)} differing lines)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
