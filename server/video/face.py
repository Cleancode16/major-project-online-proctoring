"""Face detection, subject selection, embedding and identity verification (flow.txt §A.7)."""

import cv2
import numpy as np

from video import models


def detect_faces(rgb, detector, clamp=True, min_face_px=0):
    """All MediaPipe detections as [{"bbox": (x, y, w, h), "score": float}] in detector order."""
    h, w = rgb.shape[:2]
    result = detector.process(rgb)
    faces = []
    for det in result.detections or []:
        d = det.location_data.relative_bounding_box
        x, y, bw, bh = int(d.xmin * w), int(d.ymin * h), int(d.width * w), int(d.height * h)
        if clamp:
            x2, y2 = min(w, x + bw), min(h, y + bh)
            x, y = max(0, x), max(0, y)
            bw, bh = x2 - x, y2 - y
        if bw < min_face_px or bw <= 0 or bh <= 0:
            continue
        faces.append({"bbox": (x, y, bw, bh), "score": float(det.score[0])})
    return faces


def select_subject(faces, frame_shape, mode="largest"):
    """Pick the face to verify: the first detection (v1) or the largest, ties broken by centrality."""
    if not faces:
        return None
    if mode == "first":
        return faces[0]
    h, w = frame_shape[:2]

    def key(face):
        x, y, bw, bh = face["bbox"]
        off = np.hypot((x + bw / 2) / w - 0.5, (y + bh / 2) / h - 0.5)
        return (bw * bh, -off)

    return max(faces, key=key)


def crop_face(frame_bgr, bbox, margin=0.0):
    """160x160 RGB crop around bbox, with `margin` (fraction of the box) added per side."""
    x, y, bw, bh = bbox
    if margin > 0:
        h, w = frame_bgr.shape[:2]
        mx, my = int(bw * margin), int(bh * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
        crop = frame_bgr[y0:y1, x0:x1]
    else:
        crop = frame_bgr[y:y + bh, x:x + bw]
    if crop.size == 0:
        return None
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (160, 160))


def embed(frame_bgr, bbox, margin=0.0):
    crop = crop_face(frame_bgr, bbox, margin)
    if crop is None:
        return None
    return models.embed_faces([crop])[0]


def l2n(v):
    v = np.asarray(v, np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


def compare(emb, ref, face_cfg):
    """Return (score, is_match). score is cosine similarity (v2) or Euclidean distance (legacy)."""
    if face_cfg["metric"] == "euclidean":
        d = float(np.linalg.norm(np.asarray(emb) - np.asarray(ref)))
        return d, d < face_cfg["euclid_threshold"]
    s = float(np.dot(l2n(emb), l2n(ref)))
    return s, s >= face_cfg["cos_threshold"]


def reference_from_image(img_bgr, face_cfg, detector=None):
    """Reference embedding from an enrollment image. Raises ValueError if no usable face."""
    detector = detector or models.face_detector(face_cfg["det_confidence"])
    faces = detect_faces(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), detector,
                         clamp=face_cfg["clamp_bbox"], min_face_px=face_cfg["min_face_px"])
    subject = select_subject(faces, img_bgr.shape, face_cfg["select"])
    if subject is None:
        raise ValueError("No face detected in reference image")
    emb = embed(img_bgr, subject["bbox"], face_cfg["crop_margin"])
    if emb is None:
        raise ValueError("Could not crop the face in the reference image")
    return emb, subject, len(faces)
