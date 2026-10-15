"""
Which samples count towards each violation type (flow.txt §A.10, §C).

One rule: a sample counts towards a type (numerator AND denominator) exactly when that type was
measurable on it. Only pose and gaze need the calibration baseline, so only they skip calibration samples.
"""

import config as C

FACE_DEPENDENT = {C.IDENTITY_MISMATCH, C.HEAD_POSE_DEVIATION, C.GAZE_AWAY, C.FACE_SPOOF}
NEEDS_BASELINE = {C.HEAD_POSE_DEVIATION, C.GAZE_AWAY}
SCENE_TYPES = {C.MULTIPLE_PERSONS, C.PROHIBITED_OBJECT}


def observed(sample, vtype):
    """PERF2: False if this sample reused an earlier YOLO result for `vtype` (it covers the time, but is not an
    observation: it never adds raw detections or supporting observations)."""
    return not (vtype in SCENE_TYPES and sample.get("scene_held"))


def counts_toward(sample, vtype, include_calibration=False):
    """True if this video sample is part of the analysed population for `vtype`."""
    if vtype in NEEDS_BASELINE and sample.get("phase") == "calibration" and not include_calibration:
        return False
    if vtype not in FACE_DEPENDENT:
        return True
    if not sample.get("face_visible", True):
        return False
    if vtype == C.HEAD_POSE_DEVIATION:
        return sample.get("pose_ok", True) and "rel_yaw" in sample
    if vtype == C.GAZE_AWAY:
        return C.GAZE_AWAY in sample.get("raw", {})
    if vtype == C.FACE_SPOOF:                       # FAS1: samples where the anti-spoof model scored the face
        return sample.get("spoof_score") is not None
    return sample.get("identity_ok") is not None
