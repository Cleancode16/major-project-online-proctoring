"""
V6: choose the face-verification cosine threshold and crop margin from data (flow.txt §A.7).

Data: LFW pairs (public, scikit-learn fetch_lfw_pairs, funneled colour 250x250), downloaded to
tests/data/lfw. Threshold chosen on "train" (2200 pairs), reported on the disjoint "test" (1000 pairs).
Variants: v1 crop (first face, no clamp, no margin) and v2 crops (largest face, clamped, margin m).

Run:  .venv\\Scripts\\python.exe -m tests.calibrate_face_threshold
"""

import json
import sys
import time

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_auc_score

from tests import paths
from video import face, models

LFW_HOME = paths.DATA / "lfw"
VARIANTS = {
    "v1_first_nomargin": {"select": "first", "clamp": False, "margin": 0.0},
    "v2_largest_m0.00": {"select": "largest", "clamp": True, "margin": 0.0},
    "v2_largest_m0.15": {"select": "largest", "clamp": True, "margin": 0.15},
    "v2_largest_m0.30": {"select": "largest", "clamp": True, "margin": 0.30},
}


def load(subset):
    d = fetch_lfw_pairs(subset=subset, color=True, resize=1.0, slice_=None, data_home=str(LFW_HOME))
    pairs = d.pairs
    if pairs.max() <= 1.0:
        pairs = pairs * 255.0
    return pairs.astype(np.uint8), d.target.astype(int)


def embed_all(images_rgb, variant, detector):
    crops, n_fallback = [], 0
    for img in images_rgb:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        faces = face.detect_faces(img, detector, clamp=variant["clamp"])
        subj = face.select_subject(faces, img.shape, variant["select"])
        if subj is None:
            n_fallback += 1
            h, w = img.shape[:2]
            subj = {"bbox": (w // 4, h // 4, w // 2, h // 2)}
        crop = face.crop_face(bgr, subj["bbox"], variant["margin"])
        if crop is None:
            n_fallback += 1
            crop = cv2.resize(img, (160, 160))
        crops.append(crop)
    embs = np.concatenate([models.embed_faces(crops[i:i + 64]) for i in range(0, len(crops), 64)])
    return embs, n_fallback


def sims_for(pairs, variant, detector, cache_name):
    cache = LFW_HOME / f"emb_{cache_name}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["sims"], int(z["fallback"])
    flat = pairs.reshape(-1, *pairs.shape[2:])
    embs, fb = embed_all(flat, variant, detector)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    sims = np.sum(embs[0::2] * embs[1::2], axis=1)
    np.savez(cache, sims=sims, fallback=fb)
    return sims, fb


def best_threshold(sims, y):
    cands = np.unique(np.round(sims, 4))
    accs = [np.mean((sims >= t) == y) for t in cands]
    i = int(np.argmax(accs))
    return float(cands[i]), float(accs[i])


def rates(sims, y, thr):
    pred = sims >= thr
    tp = int(np.sum(pred & (y == 1))); fn = int(np.sum(~pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0))); tn = int(np.sum(~pred & (y == 0)))
    return {"acc": (tp + tn) / len(y), "FRR": fn / max(1, tp + fn), "FAR": fp / max(1, fp + tn),
            "TP": tp, "FN": fn, "FP": fp, "TN": tn}


def main():
    t0 = time.time()
    data = {s: load(s) for s in ("train", "test")}
    print(f"LFW loaded: train {data['train'][0].shape}, test {data['test'][0].shape} ({time.time() - t0:.0f}s)")
    detector = models.face_detector(0.6)
    out = {}
    for name, var in VARIANTS.items():
        s_tr, fb_tr = sims_for(data["train"][0], var, detector, f"{name}_train")
        s_te, fb_te = sims_for(data["test"][0], var, detector, f"{name}_test")
        y_tr, y_te = data["train"][1], data["test"][1]
        thr, tr_acc = best_threshold(s_tr, y_tr)
        neg = np.sort(s_tr[y_tr == 0])
        thr_far1 = float(neg[int(np.ceil(0.99 * len(neg))) - 1])
        out[name] = {
            "train_best_threshold": thr, "train_acc": tr_acc,
            "test_at_train_threshold": rates(s_te, y_te, thr),
            "test_at_cos0.5_(v1_rule)": rates(s_te, y_te, 0.5),
            "threshold_far1pct_train": thr_far1,
            "test_at_far1pct_threshold": rates(s_te, y_te, thr_far1),
            "test_auc": float(roc_auc_score(y_te, s_te)),
            "no_face_fallbacks": fb_tr + fb_te,
            "same_mean": float(s_te[y_te == 1].mean()), "diff_mean": float(s_te[y_te == 0].mean()),
        }
        r = out[name]
        print(f"{name:20s} thr*={thr:.3f} test acc={r['test_at_train_threshold']['acc']:.3f} "
              f"FRR={r['test_at_train_threshold']['FRR']:.3f} FAR={r['test_at_train_threshold']['FAR']:.3f} "
              f"| v1 rule cos>=0.5 acc={r['test_at_cos0.5_(v1_rule)']['acc']:.3f} "
              f"FRR={r['test_at_cos0.5_(v1_rule)']['FRR']:.3f} FAR={r['test_at_cos0.5_(v1_rule)']['FAR']:.3f} "
              f"| AUC={r['test_auc']:.4f} fallbacks={r['no_face_fallbacks']} ({time.time() - t0:.0f}s)")
    (paths.RESULTS / "V6_face_calibration.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
