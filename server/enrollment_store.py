"""Local storage of enrollment references (face + voice). Nothing here is ever sent over the network."""

import json
import re
import time

import cv2
import numpy as np

import config as C


def safe_id(candidate_id):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", candidate_id.strip()) or "candidate"


def exists(candidate_id):
    """True if an enrollment folder with a meta file exists (does not create anything)."""
    return (C.ENROLL_DIR / safe_id(candidate_id) / "meta.json").exists()


def _dir(candidate_id):
    d = C.ENROLL_DIR / safe_id(candidate_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(candidate_id, face_img=None, face_embedding=None, voiceprint=None, meta=None):
    d = _dir(candidate_id)
    if face_img is not None:
        cv2.imwrite(str(d / "face.jpg"), face_img)
    if face_embedding is not None:
        np.save(d / "face_embedding.npy", np.asarray(face_embedding, np.float32))
    if voiceprint is not None:
        np.save(d / "voiceprint.npy", np.asarray(voiceprint, np.float32))
    old = load_meta(candidate_id)
    old.update(meta or {})
    old["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (d / "meta.json").write_text(json.dumps(old, indent=2, default=str))
    return d


def load_meta(candidate_id):
    p = _dir(candidate_id) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load(candidate_id):
    d = _dir(candidate_id)
    out = {"dir": d, "meta": load_meta(candidate_id)}
    for key, name in (("face_embedding", "face_embedding.npy"), ("voiceprint", "voiceprint.npy")):
        out[key] = np.load(d / name) if (d / name).exists() else None
    out["face_img"] = cv2.imread(str(d / "face.jpg")) if (d / "face.jpg").exists() else None
    return out


def list_candidates():
    return sorted(p.name for p in C.ENROLL_DIR.glob("*") if p.is_dir()) if C.ENROLL_DIR.exists() else []
