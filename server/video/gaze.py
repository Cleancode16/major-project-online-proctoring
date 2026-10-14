"""Eye gaze from MediaPipe iris landmarks (flow.txt §A.6) and the L2CS-Net decision rule (GZ3, video/l2cs.py)."""

import numpy as np

LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_IRIS = 133, 33, 468
RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_IRIS = 362, 263, 473

# (image-left corner, image-right corner, iris, upper lid, lower lid) per eye, same orientation
EYES = [(33, 133, 468, 159, 145), (362, 263, 473, 386, 374)]


def _pt(landmarks, i, w, h):
    return np.array([landmarks[i].x * w, landmarks[i].y * h])


def _iris(landmarks, iris_i, w, h, use_mean):
    """Iris centre: the centre landmark, or (GZ2) the mean of the centre + 4 ring landmarks."""
    if not use_mean or len(landmarks) < iris_i + 5:
        return _pt(landmarks, iris_i, w, h)
    return np.mean([_pt(landmarks, i, w, h) for i in range(iris_i, iris_i + 5)], axis=0)


def eye_ratios(landmarks, w, h, blink_open_ratio=0.12, vertical="lids", iris_mean=False, weight_by_width=False):
    """
    Horizontal / vertical iris ratios over both eyes. valid=False on blink/closed eyes.
      vertical="lids"        v = position between upper (0) and lower (1) lid (V5)
      vertical="corner_line" v = signed distance below the eye-corner line / eye width (GZ2, skull-fixed)
      weight_by_width        average the eyes weighted by their visible width (GZ2)
    """
    hs, vs, opens, widths = [], [], [], []
    for a_i, b_i, iris_i, top_i, bot_i in EYES:
        a, b = _pt(landmarks, a_i, w, h), _pt(landmarks, b_i, w, h)
        iris = _iris(landmarks, iris_i, w, h, iris_mean)
        top, bot = _pt(landmarks, top_i, w, h), _pt(landmarks, bot_i, w, h)
        axis = b - a
        length = float(np.linalg.norm(axis))
        if length < 1e-6:
            return {"h": None, "v": None, "open": 0.0, "valid": False}
        n = np.array([-axis[1], axis[0]]) / length          # unit normal pointing image-down
        opening = float(np.dot(bot - top, n))
        hs.append(float(np.dot(iris - a, axis) / length ** 2))
        if vertical == "corner_line":
            vs.append(float(np.dot(iris - (a + b) / 2.0, n) / length))
        else:
            vs.append(float(np.dot(iris - top, n) / opening) if opening > 1e-6 else 0.5)
        opens.append(opening / length)
        widths.append(length)
    wts = np.array(widths) if weight_by_width else np.ones(len(widths))
    open_ratio = float(np.mean(opens))
    return {"h": float(np.average(hs, weights=wts)), "v": float(np.average(vs, weights=wts)), "open": open_ratio,
            "valid": open_ratio >= blink_open_ratio}


def gaze_decision(dh, dv, rel_yaw, rel_pitch, gc, prev_away=False):
    """
    Combine eye offsets (vs calibration) with relative head pose. Returns (away, label, gaze_h, gaze_v).
    Hysteresis (GZ2): if the previous sample was away, the thresholds are lowered by gaze.hyst_deg.
    """
    hyst = gc.get("hyst_deg", 0.0) if prev_away else 0.0
    eye_yaw, eye_pitch = dh * gc["deg_per_ratio_h"], dv * gc["deg_per_ratio_v"]
    gaze_h, gaze_v = (rel_yaw or 0.0) + eye_yaw, (rel_pitch or 0.0) + eye_pitch
    away_h = abs(gaze_h) > gc["h_thr_deg"] - hyst and abs(eye_yaw) >= gc["min_eye_deg"]
    away_v = abs(gaze_v) > gc["v_thr_deg"] - hyst and abs(eye_pitch) >= gc["min_eye_deg"]
    return away_h or away_v, _label(away_h, away_v, gaze_h, gaze_v, gc.get("camera_mirrored")), gaze_h, gaze_v


def gaze_decision_deg(gaze_h, gaze_v, rel_yaw, rel_pitch, lc, mirrored=False, prev_away=False):
    """
    GZ3 (L2CS-Net): gaze_h / gaze_v are already full gaze angles (head + eyes) in degrees relative to the
    calibrated forward gaze (> 0 = towards image right / down). Away when beyond +-h_thr / v_thr (hysteresis
    lowers them by hyst_deg while away). Optional min_eye_deg: eye-in-head part (gaze - head pose) must reach it.
    """
    hyst = lc.get("hyst_deg", 0.0) if prev_away else 0.0
    min_eye = lc.get("min_eye_deg", 0.0)
    eye_h, eye_v = gaze_h - (rel_yaw or 0.0), gaze_v - (rel_pitch or 0.0)
    away_h = abs(gaze_h) > lc["h_thr_deg"] - hyst and abs(eye_h) >= min_eye
    away_v = abs(gaze_v) > lc["v_thr_deg"] - hyst and abs(eye_v) >= min_eye
    return away_h or away_v, _label(away_h, away_v, gaze_h, gaze_v, mirrored), gaze_h, gaze_v


def _label(away_h, away_v, gaze_h, gaze_v, mirrored):
    """Direction from the candidate's view (camera not mirrored: image right = candidate's left)."""
    if away_h and (not away_v or abs(gaze_h) >= abs(gaze_v)):
        right_of_image = gaze_h > 0
        if mirrored:
            right_of_image = not right_of_image
        return "Looking Left" if right_of_image else "Looking Right"
    if away_v:
        return "Looking Down" if gaze_v > 0 else "Looking Up"
    return "Looking Center"


def gaze_legacy(landmarks, w, low=0.35, high=0.65):
    """v1 ratios and labels, kept verbatim for the legacy profile."""
    def ratio(inner, outer, iris):
        ix, nx, ox = landmarks[iris].x * w, landmarks[inner].x * w, landmarks[outer].x * w
        width = abs(nx - ox)
        return (ix - ox) / width if width > 0 else 0.5

    left = ratio(LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_IRIS)
    right = ratio(RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_IRIS)
    avg = (left + right) / 2
    if avg < low:
        label = "Looking Right"
    elif avg > high:
        label = "Looking Left"
    else:
        label = "Looking Center"
    return {"left_ratio": left, "right_ratio": right, "h": avg, "v": None,
            "label": label, "valid": True, "away_raw": label != "Looking Center"}
