"""
Score the video pipeline against frame-level labels (sample-level TP/FP/FN per violation type).

A sample is POSITIVE for a type if its frame lies in a labelled positive range, AMBIGUOUS (not scored)
if it lies in an ambiguous range, NEGATIVE otherwise (for types listed in negative_everywhere_else).
For v2, which samples count per type is decided by aggregation.denominators.counts_toward (pose/gaze
skip calibration, everything else counts from t=0); the legacy profile scores every sample it kept.

Usage:
  python -m tests.eval_video --profile v2 [--weights yolov8s.pt] [--video ... --labels ... --ref ...]
"""

import argparse
import json
import time

import cv2
import numpy as np

import config as C
from aggregation.denominators import counts_toward
from tests import paths
from video import face, pipeline


def frame_label(frame, labels, vtype):
    def inside(ranges):
        return any(a <= frame <= b for a, b in ranges)
    if inside(labels.get("positive", {}).get(vtype, [])):
        return True
    if inside(labels.get("ambiguous", {}).get(vtype, [])):
        return None
    if vtype in labels.get("negative_everywhere_else", []):
        return False
    return None


def score(result, labels, profile):
    out = {}
    samples = result["samples"]  # v2: membership per type decided by counts_toward()
    for vtype in C.VIDEO_TYPES:
        tp = fp = fn = tn = 0
        for s in samples:
            if vtype not in s.get("flags", {}):
                continue
            if profile != "legacy" and not counts_toward(s, vtype):
                continue
            truth = frame_label(s["frame"], labels, vtype)
            if truth is None:
                continue
            pred = s["flags"][vtype]
            tp += pred and truth
            fp += pred and not truth
            fn += (not pred) and truth
            tn += (not pred) and (not truth)
        if tp + fp + fn + tn == 0:
            continue
        out[vtype] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                      "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                      "recall": round(tp / (tp + fn), 3) if tp + fn else None}
    return out


def run(profile="v2", overrides=None, video=None, labels=None, ref=None):
    cfg = C.get_config(profile, overrides)
    video = video or paths.SAMPLE_VIDEO
    labels = json.loads((labels or (paths.DATA / "sample_clip_labels.json")).read_text(encoding="utf-8-sig"))
    ref_emb, _, _ = face.reference_from_image(cv2.imread(str(ref or paths.SAMPLE_REF)), cfg["face"])
    t0 = time.time()
    res = pipeline.analyze_video(video, ref_emb, cfg)
    elapsed = time.time() - t0
    sc = score(res, labels, profile)
    steady = [s["ms"] for s in res["samples"][3:] if "ms" in s]  # skip warm-up samples
    timing = {k: round(float(np.median([m[k] for m in steady])), 1) for k in steady[0]} if steady else {}
    return {"profile": profile, "overrides": overrides or {}, "n_samples": len(res["samples"]),
            "wall_s_incl_model_load": round(elapsed, 1), "median_ms_per_sample_after_warmup": timing,
            "baseline": res["baseline"], "scores": sc}, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="v2")
    ap.add_argument("--weights")
    ap.add_argument("--video")
    ap.add_argument("--labels")
    ap.add_argument("--ref")
    ap.add_argument("--out")
    a = ap.parse_args()
    ov = {"objects": {"weights": a.weights}} if a.weights else None
    from pathlib import Path
    summary, _ = run(a.profile, ov, Path(a.video) if a.video else None,
                     Path(a.labels) if a.labels else None, Path(a.ref) if a.ref else None)
    print(json.dumps(summary, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
