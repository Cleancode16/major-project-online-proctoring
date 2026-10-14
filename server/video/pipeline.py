"""
VideoAnalyzer: sampling, calibration and per-sample violation flags (flow.txt §A.2-§A.10).

Usage (offline):  result = analyze_video(path, ref_embedding, cfg)
Usage (live):     va = VideoAnalyzer(ref, cfg, rate_hz); va.process(frame, t) ...; va.finalize()
"""

import math
import time
from collections import deque

import cv2
import numpy as np

import config as C
from video import detectors, face, gaze, head_pose, models
from video.temporal import MajorityVote, RollingMedian, centered_mean, centered_median, filter_runs, samples_for


def iter_sampled_frames(path, sampling):
    """Yield (frame_bgr, t_seconds, frame_index) according to the sampling policy (§A.3)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx, next_t = 0, 0.0
    step_t = 1.0 / sampling["fps"]
    try:
        while True:
            if not cap.grab():
                break
            t = idx / native
            if sampling["mode"] == "every_n":
                take = idx % sampling["every_n"] == 0
            else:
                take = t + 1e-6 >= next_t
            if take:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if sampling["mode"] != "every_n":
                    next_t += step_t
                yield frame, t, idx
            idx += 1
    finally:
        cap.release()


def _iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def promote_weak(strong, weak, reach=1):
    """OB2: a weak detection within +-reach samples of a confident one counts as detected."""
    n = len(strong)
    return [bool(strong[i] or (weak[i] and any(strong[j] for j in range(max(0, i - reach), min(n, i + reach + 1))
                                               if j != i)))
            for i in range(n)]


def windows(cfg, rate):
    """
    FPS10: every sample-count window derived from seconds at `rate`, so detection behaves the same at 3 or 10 fps
    (and on a live stream whose real rate is lower than configured). A block without the *_s key (legacy profile)
    keeps its old sample count.
    """
    hp, gz, fc, pc, oc = cfg["head_pose"], cfg["gaze"], cfg["face"], cfg["persons"], cfg["objects"]

    def sec(block, key, count):
        return samples_for(block[key], rate) if block.get(key) else count

    w = {"pose_smooth": sec(hp, "smooth_s", hp["smooth_window"]),
         "gaze_smooth": sec(gz, "smooth_s", gz["smooth_window"]),
         "id_vote": sec(fc, "vote_s", fc["smooth_window"]),
         "static_n": sec(pc, "static_s", pc.get("static_samples", 9)),
         "person_gap": (int(round(pc["persist_gap_s"] * rate)) if pc.get("persist_gap_s") is not None
                        else pc.get("persist_gap", 0)),
         "weak_reach": max(1, int(round(oc["weak_reach_s"] * rate))) if oc.get("weak_reach_s") else 1}
    if oc.get("persist_window_s"):
        n = samples_for(oc["persist_window_s"], rate)
        k = min(n, max(oc["persist_k"], int(math.ceil(oc.get("persist_frac", 0.0) * n - 1e-9))))
    else:
        n, k = oc["persist_n"], oc["persist_k"]
    w["obj_n"], w["obj_k"] = n, k
    return w


def annotated_snapshot(frame, s, cfg, width=640):
    """Evidence image: objects (label + confidence), faces (count), head-pose angles, timestamp. Returns JPEG bytes."""
    h, w = frame.shape[:2]
    k = width / w
    img = cv2.resize(frame, (width, int(h * k)))

    def label(text, x, y, color):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(th + 4, y)
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + 2), color, -1)
        cv2.putText(img, text, (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for b in s.get("object_boxes", []):
        x1, y1, x2, y2 = [int(v * k) for v in b["xyxy"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 220), 2)
        label(f"{b['label']} {b['conf']:.2f}", x1, y1, (0, 0, 220))
    for x, y, bw, bh in s.get("face_boxes", []):
        cv2.rectangle(img, (int(x * k), int(y * k)), (int((x + bw) * k), int((y + bh) * k)), (0, 170, 255), 2)
    lines = [f"t = {s['t']:.2f} s", f"faces: {s.get('n_faces', 0)}  persons: {s.get('persons', 0)}"]
    if "rel_yaw" in s:
        hp = cfg["head_pose"]
        lines.append(f"dYaw {s['rel_yaw']:+.0f}  dPitch {s['rel_pitch']:+.0f}  dRoll {s['rel_roll']:+.0f} deg "
                     f"(limits {hp['yaw_thr']:.0f}/{hp['pitch_thr']:.0f}/{hp['roll_thr']:.0f})")
    for i, text in enumerate(lines):
        label(text, 6, img.shape[0] - 8 - 20 * (len(lines) - 1 - i), (40, 40, 40))
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    return {"fps": fps, "frames": n, "duration_s": n / fps if fps else 0.0, "width": w, "height": h}


def sample_rate(sampling, native_fps):
    return native_fps / sampling["every_n"] if sampling["mode"] == "every_n" else sampling["fps"]


class VideoAnalyzer:
    def __init__(self, ref_embedding, cfg, rate_hz, live=False):
        self.cfg = cfg
        self.ref = np.asarray(ref_embedding)
        self.rate = rate_hz
        self.live = live
        self.t_first = None
        self.w = windows(cfg, rate_hz)        # FPS10: sample windows from seconds
        fc = cfg["face"]
        self.detector = models.face_detector(fc["det_confidence"])
        self.class_names = detectors.verify_class_ids(cfg["objects"])
        self.mesh = models.face_mesh(max_faces=fc["mesh_max_faces"])
        self.samples = []
        self.cal = {"yaw": [], "pitch": [], "roll": [], "gaze_h": [], "gaze_v": [], "t": []}
        self.cal_t0 = None
        self.cal_seen = 0
        self.cal_any = []
        self.baseline = None
        self.processed = 0
        self.n_thumbs = 0
        self.id_vote = MajorityVote(self.w["id_vote"])
        hp, gz = cfg["head_pose"], cfg["gaze"]
        self.sm = {k: RollingMedian(self.w["pose_smooth"]) for k in ("yaw", "pitch", "roll")}
        self.sm_gaze = {k: RollingMedian(self.w["gaze_smooth"]) for k in ("h", "v")}
        # HP2 / GZ2 (flow.txt §A.5, §A.6 "Accuracy upgrade")
        self.use_template = hp.get("method") == "template"
        self.cal_pts, self.cal_any_pts = [], []
        self.template = None
        self.pose_offset = np.zeros(3)        # slow baseline adaptation (deg)
        self.gaze_offset = np.zeros(2)        # slow baseline adaptation (ratio units)
        self.live_pose_prev = False           # hysteresis state for the live (causal) flags
        self.live_gaze_prev = False
        self.face_tracks = []                 # MP2 static-face tracking
        # GZ3: L2CS-Net gaze (falls back to gaze.fallback if torch / the weights are unavailable)
        self.gaze_method, self.gaze_fallback, self.l2cs, self.ema_gaze = gz["method"], None, None, None
        self.late_gaze = []                   # DET2: forward gaze collected after calibration if needed
        if self.gaze_method == "l2cs":
            lc = gz["l2cs"]
            try:
                size = lc.get("live_input_size", lc["input_size"]) if live else lc["input_size"]
                self.l2cs = models.l2cs(lc["weights"], size, lc.get("torch_threads"))
            except Exception as exc:
                self.gaze_method = gz.get("fallback", "v2")
                self.gaze_fallback = f"L2CS-Net gaze model not available ({exc}); used the {self.gaze_method} gaze method."
                print(f">>> {self.gaze_fallback}")

    # ------------------------------------------------------------------ calibration
    def _calibrating(self):
        return self.baseline is None

    def _calibration_add(self, t, pose, gz, identity_ok, pts=None):
        cc = self.cfg["calibration"]
        if self.cal_t0 is None:
            self.cal_t0 = t
        self.cal_seen += 1
        accepted = pose is not None
        if accepted and cc["require_identity"] and not identity_ok:
            accepted = False
        if accepted and (abs(pose[0]) > cc["max_abs_yaw"] or abs(pose[1]) > cc["max_abs_pitch"]):
            accepted = False
        if accepted:
            for k, v in zip(("yaw", "pitch", "roll"), pose):
                self.cal[k].append(v)
            if gz is not None and gz.get("valid") and self._gaze_cal_ok(gz):
                self.cal["gaze_h"].append(gz["h"])
                if gz.get("v") is not None:
                    self.cal["gaze_v"].append(gz["v"])
            self.cal["t"].append(t)
            if pts is not None:
                self.cal_pts.append(pts)

        if pose is not None:
            self.cal_any.append((pose, gz))
            if pts is not None:
                self.cal_any_pts.append(pts)

        n = len(self.cal["yaw"])
        fallback = False
        if cc["mode"] == "count":
            done = n >= cc["count"]
        else:
            elapsed = t - self.cal_t0
            done = (elapsed >= cc["duration_s"] and n >= cc["min_samples"]) or \
                   (elapsed >= cc["max_duration_s"] and n >= 1)
            if not done and elapsed >= cc["max_duration_s"] and self.cal_any:
                done = fallback = True
        if done:
            agg = np.median if cc["aggregate"] == "median" else np.mean
            src = self.cal
            if fallback:
                src = {"yaw": [p[0] for p, _ in self.cal_any], "pitch": [p[1] for p, _ in self.cal_any],
                       "roll": [p[2] for p, _ in self.cal_any],
                       "gaze_h": [g["h"] for _, g in self.cal_any if g and g.get("valid") and self._gaze_cal_ok(g)],
                       "gaze_v": [g["v"] for _, g in self.cal_any
                                  if g and g.get("valid") and g.get("v") is not None and self._gaze_cal_ok(g)]}
            self.baseline = {k: float(agg(src[k])) for k in ("yaw", "pitch", "roll")}
            n_min = self.cfg["gaze"].get("l2cs", {}).get("cal_min_samples", 1) if self.gaze_method == "l2cs" else 1
            ok_g = len(src["gaze_h"]) >= n_min and len(src["gaze_v"]) >= n_min
            self.baseline["gaze_h"] = float(agg(src["gaze_h"])) if ok_g else None
            self.baseline["gaze_v"] = float(agg(src["gaze_v"])) if ok_g else None
            self.baseline["gaze_samples"] = len(src["gaze_h"])
            self.baseline["n_samples"] = len(src["yaw"])
            self.baseline["n_seen"] = self.cal_seen
            self.baseline["t_end"] = t
            self.baseline["fallback"] = fallback
            # HP2: person-specific landmark template from the same calibration samples
            point_sets = self.cal_any_pts if fallback else self.cal_pts
            if self.use_template and point_sets:
                self.template = head_pose.build_template(point_sets)
            self.baseline["pose_method"] = "template" if self.template is not None else "pnp"
            self.baseline["template_samples"] = len(point_sets) if self.template is not None else 0
        return accepted

    def set_rate(self, rate):
        """FPS10: new sample rate (live: measured) -> recompute the windows and resize the causal smoothers."""
        self.rate = float(rate)
        self.w = windows(self.cfg, self.rate)
        self.id_vote.resize(self.w["id_vote"])
        for sm in self.sm.values():
            sm.resize(self.w["pose_smooth"])
        for sm in self.sm_gaze.values():
            sm.resize(self.w["gaze_smooth"])

    def _track_live_rate(self, t):
        """Live streams run below the configured rate when the CPU is the limit: follow the measured rate."""
        if self.t_first is None:
            self.t_first = t
        if self.processed >= 5 and t > self.t_first:
            measured = self.processed / (t - self.t_first)
            if abs(measured - self.rate) > 0.15 * self.rate:
                self.set_rate(measured)

    def _gaze_cal_ok(self, gz):
        """DET2: L2CS calibration gaze must be plausibly 'at the screen' (absolute angle gate)."""
        if self.gaze_method != "l2cs":
            return True
        lc = self.cfg["gaze"]["l2cs"]
        return (gz.get("h") is not None and gz.get("v") is not None
                and abs(gz["h"]) <= lc.get("cal_max_abs_h", 1e9) and abs(gz["v"]) <= lc.get("cal_max_abs_v", 1e9))

    # ------------------------------------------------------------------ per frame
    def process(self, frame, t):
        """Analyse one sampled frame. Returns the sample dict, or None if the legacy profile skips it."""
        cfg, fc = self.cfg, self.cfg["face"]
        t_start = time.perf_counter()
        if self.live:
            self._track_live_rate(t)
        legacy_skip = not cfg["face_not_visible"]["enabled"]
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face.detect_faces(rgb, self.detector, clamp=fc["clamp_bbox"], min_face_px=fc["min_face_px"])
        self._mark_static_faces(faces)
        s = {"t": round(float(t), 3), "phase": "calibration" if self._calibrating() else "analysis",
             "face_visible": bool(faces), "n_faces": len(faces), "raw": {},
             "brightness": round(float(cv2.resize(rgb, (64, 36)).mean()), 1)}       # DET2 low-light check

        identity_ok, pose, gz = None, None, None
        subject, match = self._select_subject(frame, faces)
        if subject is not None:
            subject["static"] = False          # the candidate's own face is never treated as a photo (MP2)
        s["static_faces"] = sum(1 for f in faces if f.get("static"))
        if subject is not None:
            if match is None:
                if legacy_skip:
                    return None
                s["face_visible"] = False
            else:
                score, identity_ok = match
                s["identity_score"] = score
                s["identity_ok_raw"] = identity_ok
                s["bbox"] = subject["bbox"]
        elif legacy_skip:
            return None

        t_face = time.perf_counter()
        landmarks = None
        if s["face_visible"]:
            mesh = self.mesh.process(rgb)
            if mesh.multi_face_landmarks:
                landmarks = self._pick_mesh(mesh.multi_face_landmarks, subject["bbox"], w, h)
            elif legacy_skip:
                return None

        pts, rel_template = None, None
        if landmarks is not None:
            pose = head_pose.head_pose(landmarks, w, h, cfg["head_pose"]["euler"])
            gz = self._gaze(landmarks, w, h, frame, subject["bbox"])
            if pose is not None:
                s.update(yaw=pose[0], pitch=pose[1], roll=pose[2])
            s["gaze"] = {k: gz.get(k) for k in ("h", "v", "label", "valid", "method", "yaw_deg", "pitch_deg")
                         if k in gz}
            if self.use_template:
                pts = head_pose.rigid_points(landmarks, w, h, cfg["head_pose"]["rigid_ids"])
                if self.template is not None:
                    rel_template = head_pose.euler_from_matrix(head_pose.relative_rotation(self.template, pts))
        s["pose_ok"] = pose is not None or rel_template is not None
        s["gaze_ok"] = bool(gz and gz.get("valid"))

        # calibration (needs the raw values before baseline exists)
        if self._calibrating():
            s["calibration_accepted"] = self._calibration_add(t, pose, gz, bool(identity_ok), pts)

        t_mesh = time.perf_counter()
        self._flag_identity(s, identity_ok)
        self._flag_pose(s, pose, rel_template)
        self._flag_gaze(s, gz, pose)
        self._flag_scene(s, frame, faces)
        t_end = time.perf_counter()
        s["ms"] = {"face": round((t_face - t_start) * 1000, 1), "mesh_pose_gaze": round((t_mesh - t_face) * 1000, 1),
                   "yolo": round((t_end - t_mesh) * 1000, 1), "total": round((t_end - t_start) * 1000, 1)}

        # evidence: keep an annotated JPEG only for samples with a raw violation (bounded memory, flow.txt §C.5)
        max_thumbs = cfg.get("aggregation", {}).get("max_thumbnails", 0)
        if any(s["raw"].values()) and self.n_thumbs < max_thumbs:
            s["_thumb"] = annotated_snapshot(frame, s, cfg)
            self.n_thumbs += 1

        self.processed += 1
        self.samples.append(s)
        return s

    def _select_subject(self, frame, faces):
        """Returns (subject face, (score, is_match)) or (subject, None) if the crop failed."""
        fc = self.cfg["face"]
        if not faces:
            return None, None
        if fc["select"] == "best_match" and len(faces) > 1:
            cands = sorted(faces, key=lambda f: -f["bbox"][2] * f["bbox"][3])[:3]
            crops = [face.crop_face(frame, f["bbox"], fc["crop_margin"]) for f in cands]
            ok = [(f, c) for f, c in zip(cands, crops) if c is not None]
            if not ok:
                return cands[0], None
            embs = models.embed_faces([c for _, c in ok])
            scored = [(face.compare(e, self.ref, fc), f) for e, (f, _) in zip(embs, ok)]
            best = max(scored, key=lambda x: x[0][0])
            if best[0][1]:
                return best[1], best[0]
            largest = face.select_subject([f for _, f in scored], frame.shape, "largest")
            return largest, next(m for m, f in scored if f is largest)
        mode = "largest" if fc["select"] == "best_match" else fc["select"]
        subject = face.select_subject(faces, frame.shape, mode)
        emb = face.embed(frame, subject["bbox"], fc["crop_margin"])
        return subject, (None if emb is None else face.compare(emb, self.ref, fc))

    def _pick_mesh(self, meshes, bbox, w, h):
        if len(meshes) == 1 and not self.cfg["face"]["mesh_must_match"]:
            return meshes[0].landmark
        x, y, bw, bh = bbox
        cx, cy = x + bw / 2, y + bh / 2

        def centre(m):
            return np.mean([p.x for p in m.landmark]) * w, np.mean([p.y for p in m.landmark]) * h

        best = min(meshes, key=lambda m: (centre(m)[0] - cx) ** 2 + (centre(m)[1] - cy) ** 2)
        if self.cfg["face"]["mesh_must_match"]:
            mx, my = centre(best)
            if not (x <= mx <= x + bw and y <= my <= y + bh):
                return None
        return best.landmark

    def _gaze(self, landmarks, w, h, frame=None, bbox=None):
        gc = self.cfg["gaze"]
        if self.gaze_method == "legacy":
            return gaze.gaze_legacy(landmarks, w, gc["legacy_low"], gc["legacy_high"])
        eyes = gaze.eye_ratios(landmarks, w, h, gc["blink_open_ratio"], vertical=gc.get("vertical", "lids"),
                               iris_mean=gc.get("iris_mean", False), weight_by_width=gc.get("weight_by_width", False))
        if self.gaze_method != "l2cs":
            return eyes
        # GZ3: the landmarks still decide blinks / closed eyes; L2CS-Net gives the gaze angles
        out = {"h": None, "v": None, "open": eyes["open"], "valid": False, "method": "l2cs"}
        if not eyes["valid"] or bbox is None:
            return out
        r = self.l2cs.predict(frame, bbox, gc["l2cs"].get("crop_margin", 0.0))
        if r is not None:
            out.update(r, valid=True)
        return out

    # ------------------------------------------------------------------ raw flags
    def _flag_identity(self, s, identity_ok):
        if identity_ok is None:
            s["identity_ok"] = None
            return
        s["identity_ok"] = self.id_vote(identity_ok)
        s["raw"][C.IDENTITY_MISMATCH] = not s["identity_ok"]

    def _flag_pose(self, s, pose, rel_template=None):
        hp = self.cfg["head_pose"]
        if self.baseline is None or s["phase"] == "calibration":
            return
        if rel_template is not None:            # HP2: relative to the candidate's own template
            rel = np.array(rel_template, dtype=np.float64)
        elif pose is not None:                  # PnP relative to the calibration baseline
            rel = [pose[i] - self.baseline[k] for i, k in enumerate(("yaw", "pitch", "roll"))]
            if hp["euler"] != "legacy":
                rel = [head_pose.wrap180(v) for v in rel]
            rel = np.array(rel, dtype=np.float64)
        else:
            return
        s.update(raw_rel_yaw=float(rel[0]), raw_rel_pitch=float(rel[1]), raw_rel_roll=float(rel[2]))
        rel_c = rel - self.pose_offset
        self._adapt_pose(rel)
        s.update(rel_yaw=float(rel_c[0]), rel_pitch=float(rel_c[1]), rel_roll=float(rel_c[2]))
        # live status uses a causal median; finalize() recomputes with a zero-lag centred median
        live = [self.sm[k](v) for k, v in zip(("yaw", "pitch", "roll"), rel_c)]
        self.live_pose_prev = self._pose_deviates(*live, prev=self.live_pose_prev)
        s["raw"][C.HEAD_POSE_DEVIATION] = self.live_pose_prev

    def _pose_thresholds(self):
        hp = self.cfg["head_pose"]
        return np.array([hp["yaw_thr"], hp["pitch_thr"], hp["roll_thr"]], dtype=np.float64)

    def _adapt_pose(self, rel):
        """HP2 slow baseline adaptation: only from clearly normal samples, capped (flow.txt §A.5)."""
        hp = self.cfg["head_pose"]
        tau = hp.get("adapt_tau_s")
        if not tau:
            return
        c = rel - self.pose_offset
        if np.all(np.abs(c) < self._pose_thresholds() / 2):
            self.pose_offset = np.clip(self.pose_offset + c * min(1.0, (1.0 / self.rate) / tau),
                                       -hp["adapt_max_deg"], hp["adapt_max_deg"])

    def _pose_deviates(self, ry, rp, rr, prev=False):
        """Deviation with hysteresis: once deviating, every axis must drop below threshold - hyst_deg to end it."""
        hyst = self.cfg["head_pose"].get("hyst_deg", 0.0) if prev else 0.0
        thr = self._pose_thresholds() - hyst
        return bool(abs(ry) > thr[0] or abs(rp) > thr[1] or abs(rr) > thr[2])

    def _recompute_pose_offline(self):
        hp = self.cfg["head_pose"]
        win = self.w["pose_smooth"]
        if win <= 1 and not hp.get("hyst_deg"):
            return
        sm = {k: centered_median([s.get(k) for s in self.samples], win)
              for k in ("rel_yaw", "rel_pitch", "rel_roll")}
        prev = False
        for i, s in enumerate(self.samples):
            if C.HEAD_POSE_DEVIATION in s["raw"] and not np.isnan(sm["rel_yaw"][i]):
                prev = self._pose_deviates(sm["rel_yaw"][i], sm["rel_pitch"][i], sm["rel_roll"][i], prev=prev)
                s["raw"][C.HEAD_POSE_DEVIATION] = prev
            else:
                prev = False   # no measurement: the hysteresis state does not carry over a gap

    def _flag_gaze(self, s, gz, pose):
        if gz is None or not gz["valid"]:
            return
        gc = self.cfg["gaze"]
        if self.gaze_method == "l2cs":
            self._flag_gaze_l2cs(s, gz)
            return
        if self.gaze_method == "legacy":
            s["raw"][C.GAZE_AWAY] = gz["away_raw"]
            return
        if self.baseline is None or s["phase"] == "calibration":
            return
        h0 = self.baseline.get("gaze_h") if self.baseline.get("gaze_h") is not None else 0.5
        v0 = self.baseline.get("gaze_v") if self.baseline.get("gaze_v") is not None else 0.5
        d = np.array([gz["h"] - h0, gz["v"] - v0], dtype=np.float64)
        s["raw_gaze_dh"], s["raw_gaze_dv"] = float(d[0]), float(d[1])
        dc = d - self.gaze_offset
        self._adapt_gaze(d)
        s["gaze_dh"], s["gaze_dv"] = float(dc[0]), float(dc[1])
        away, label, gh, gv = gaze.gaze_decision(self.sm_gaze["h"](s["gaze_dh"]), self.sm_gaze["v"](s["gaze_dv"]),
                                                 s.get("rel_yaw"), s.get("rel_pitch"), gc, prev_away=self.live_gaze_prev)
        self.live_gaze_prev = away
        s["gaze"].update(label=label, gaze_h_deg=gh, gaze_v_deg=gv)
        s["raw"][C.GAZE_AWAY] = away

    def _flag_gaze_l2cs(self, s, gz):
        """GZ3: degrees vs the calibrated forward gaze, drift correction, EMA, +-threshold with hysteresis."""
        gc = self.cfg["gaze"]
        lc = gc["l2cs"]
        if self.baseline is None or s["phase"] == "calibration":
            return
        if self.baseline.get("gaze_h") is None:
            # DET2: no usable forward gaze from calibration -> collect it from frontal analysis samples first
            hp = self.cfg["head_pose"]
            frontal = abs(s.get("rel_yaw", 99)) < hp["yaw_thr"] / 2 and abs(s.get("rel_pitch", 99)) < hp["pitch_thr"] / 2
            if frontal and self._gaze_cal_ok(gz):
                self.late_gaze.append((gz["h"], gz["v"]))
            if len(self.late_gaze) < max(lc.get("cal_min_samples", 3), 5):
                return
            hv = np.median(np.array(self.late_gaze), axis=0)
            self.baseline.update(gaze_h=float(hv[0]), gaze_v=float(hv[1]), gaze_late=True, gaze_samples=len(self.late_gaze))
        h0, v0 = self.baseline["gaze_h"], self.baseline["gaze_v"]
        d = np.array([gz["h"] - h0, gz["v"] - v0], dtype=np.float64)
        s["raw_gaze_dh"], s["raw_gaze_dv"] = float(d[0]), float(d[1])
        dc = d - self.gaze_offset
        tau = lc.get("adapt_tau_s")
        if tau and abs(dc[0]) < lc["h_thr_deg"] / 2 and abs(dc[1]) < lc["v_thr_deg"] / 2:
            self.gaze_offset = np.clip(self.gaze_offset + dc * min(1.0, (1.0 / self.rate) / tau),
                                       -lc["adapt_max_deg"], lc["adapt_max_deg"])
        s["gaze_dh"], s["gaze_dv"] = float(dc[0]), float(dc[1])
        if lc.get("smoothing", "ema") == "ema":
            a = lc["ema_alpha"]
            self.ema_gaze = dc if self.ema_gaze is None else a * dc + (1 - a) * self.ema_gaze
            sh, sv = self.ema_gaze
        else:
            sh, sv = self.sm_gaze["h"](s["gaze_dh"]), self.sm_gaze["v"](s["gaze_dv"])
        away, label, gh, gv = gaze.gaze_decision_deg(float(sh), float(sv), s.get("rel_yaw"), s.get("rel_pitch"), lc,
                                                     gc.get("camera_mirrored"), prev_away=self.live_gaze_prev)
        self.live_gaze_prev = away
        s["gaze"].update(label=label, gaze_h_deg=round(gh, 2), gaze_v_deg=round(gv, 2))
        s["raw"][C.GAZE_AWAY] = away

    def _adapt_gaze(self, d):
        """GZ2 slow baseline adaptation of the iris offsets: only clearly centred samples, capped (flow.txt §A.6)."""
        gc = self.cfg["gaze"]
        tau = gc.get("adapt_tau_s")
        if not tau:
            return
        c = d - self.gaze_offset
        if abs(c[0]) * gc["deg_per_ratio_h"] < gc["h_thr_deg"] / 2 and abs(c[1]) * gc["deg_per_ratio_v"] < gc["v_thr_deg"] / 2:
            self.gaze_offset = np.clip(self.gaze_offset + c * min(1.0, (1.0 / self.rate) / tau),
                                       -gc["adapt_max_ratio"], gc["adapt_max_ratio"])

    def _recompute_gaze_offline(self):
        gc = self.cfg["gaze"]
        if self.gaze_method == "l2cs":
            self._recompute_gaze_l2cs_offline()
            return
        if self.gaze_method == "legacy" or (self.w["gaze_smooth"] <= 1 and not gc.get("hyst_deg")):
            return
        win = self.w["gaze_smooth"]
        dh = centered_median([s.get("gaze_dh") for s in self.samples], win)
        dv = centered_median([s.get("gaze_dv") for s in self.samples], win)
        ry = centered_median([s.get("rel_yaw") for s in self.samples], self.w["pose_smooth"])
        rp = centered_median([s.get("rel_pitch") for s in self.samples], self.w["pose_smooth"])
        prev = False
        for i, s in enumerate(self.samples):
            if C.GAZE_AWAY not in s["raw"] or np.isnan(dh[i]):
                prev = False
                continue
            y = None if np.isnan(ry[i]) else ry[i]
            p = None if np.isnan(rp[i]) else rp[i]
            away, label, gh, gv = gaze.gaze_decision(dh[i], dv[i], y, p, gc, prev_away=prev)
            prev = away
            s["raw"][C.GAZE_AWAY] = away
            s["gaze"].update(label=label, gaze_h_deg=float(gh), gaze_v_deg=float(gv))

    def _recompute_gaze_l2cs_offline(self):
        """GZ3 offline: centred moving average (zero lag) instead of the causal EMA, same decision rule."""
        gc = self.cfg["gaze"]
        lc = gc["l2cs"]
        win = self.w["gaze_smooth"]
        dh = centered_mean([s.get("gaze_dh") for s in self.samples], win)
        dv = centered_mean([s.get("gaze_dv") for s in self.samples], win)
        ry = centered_median([s.get("rel_yaw") for s in self.samples], self.w["pose_smooth"])
        rp = centered_median([s.get("rel_pitch") for s in self.samples], self.w["pose_smooth"])
        prev = False
        for i, s in enumerate(self.samples):
            if C.GAZE_AWAY not in s["raw"] or np.isnan(dh[i]) or np.isnan(dv[i]):
                prev = False
                continue
            y = None if np.isnan(ry[i]) else ry[i]
            p = None if np.isnan(rp[i]) else rp[i]
            away, label, gh, gv = gaze.gaze_decision_deg(dh[i], dv[i], y, p, lc, gc.get("camera_mirrored"), prev_away=prev)
            prev = away
            s["raw"][C.GAZE_AWAY] = away
            s["gaze"].update(label=label, gaze_h_deg=round(float(gh), 2), gaze_v_deg=round(float(gv), 2))

    def _mark_static_faces(self, faces):
        """MP2: track faces across samples; f["static"] = no micro-motion over the last static_samples (photo / poster)."""
        pc = self.cfg["persons"]
        if not pc.get("static_reject"):
            return
        n, frac = self.w["static_n"], pc["static_frac"]
        tracks, used = [], set()
        for f in faces:
            best, best_iou = None, 0.3
            for i, tr in enumerate(self.face_tracks):
                if i in used:
                    continue
                iou = _iou_xywh(tr["last"], f["bbox"])
                if iou >= best_iou:
                    best, best_iou = i, iou
            if best is None:
                tr = {"hist": deque(maxlen=n)}
            else:
                used.add(best)
                tr = self.face_tracks[best]
            x, y, w, h = f["bbox"]
            tr["hist"].append((x + w / 2.0, y + h / 2.0, float(w)))
            tr["last"] = f["bbox"]
            hist = np.array(tr["hist"])
            width = max(float(hist[:, 2].mean()), 1.0)
            f["static"] = bool(len(hist) >= n and hist[:, :2].std(axis=0).max() < frac * width
                               and hist[:, 2].std() < frac * width)
            tracks.append(tr)
        self.face_tracks = tracks       # faces not seen in this sample drop their track

    def _flag_scene(self, s, frame, faces):
        oc, pc = self.cfg["objects"], self.cfg["persons"]
        boxes = detectors.run_yolo(frame, oc)
        if oc.get("roi_pass") and oc["method"] != "legacy" and s.get("bbox") is not None:
            boxes = detectors.merge_boxes(boxes, detectors.run_yolo_roi(frame, s["bbox"], oc))   # OB2
        face_arg = faces if (pc.get("static_reject") or pc.get("yolo_conf_low")) else len(faces)
        persons, yolo_n = detectors.count_persons(boxes, face_arg, frame.shape, pc)
        if oc["method"] == "legacy":
            objects = detectors.objects_legacy(boxes, oc)
        else:
            objects, s["object_boxes"], s["object_weak"] = detectors.objects_v2(boxes, oc, s.get("bbox"))
        s["persons"], s["persons_yolo"] = persons, yolo_n
        s["objects"] = objects
        s["face_boxes"] = [list(f["bbox"]) for f in faces]
        s["raw"][C.MULTIPLE_PERSONS] = persons > 1
        s["raw"][C.PROHIBITED_OBJECT] = bool(objects)
        if self.cfg["face_not_visible"]["enabled"]:
            s["raw"][C.FACE_NOT_VISIBLE] = not s["face_visible"]

    # ------------------------------------------------------------------ finalize
    def finalize(self):
        """Apply offline persistence to the raw flags; return samples + calibration info."""
        cfg = self.cfg
        self.w = windows(cfg, self.rate)      # FPS10: final rate (live: achieved rate set by the session)
        self._recompute_pose_offline()
        self._recompute_gaze_offline()
        persist = {
            C.IDENTITY_MISMATCH: (samples_for(cfg["face"]["persist_s"], self.rate), 0),
            C.HEAD_POSE_DEVIATION: (samples_for(cfg["head_pose"]["persist_s"], self.rate), 0),
            C.GAZE_AWAY: (samples_for(cfg["gaze"]["persist_s"], self.rate), 0),
            C.MULTIPLE_PERSONS: (samples_for(cfg["persons"]["persist_s"], self.rate), self.w["person_gap"]),
            C.PROHIBITED_OBJECT: (self.w["obj_k"], self.w["obj_n"] - self.w["obj_k"]),
            C.FACE_NOT_VISIBLE: (samples_for(cfg["face_not_visible"]["persist_s"], self.rate), 0),
        }
        for vtype, (min_len, max_gap) in persist.items():
            raw = [s["raw"].get(vtype, False) for s in self.samples]
            if vtype == C.PROHIBITED_OBJECT and cfg["objects"].get("weak_conf"):
                raw = promote_weak(raw, [s.get("object_weak", False) for s in self.samples], self.w["weak_reach"])  # OB2
            final = filter_runs(raw, min_len, max_gap)
            for s, f in zip(self.samples, final):
                s.setdefault("flags", {})[vtype] = bool(f)
        return {
            "samples": self.samples,
            "rate_hz": self.rate,
            "baseline": self.baseline,
            "calibration": {"accepted": len(self.cal["yaw"]), "seen": self.cal_seen,
                            "config": cfg["calibration"]},
            "gaze_method": self.gaze_method,
            "gaze_fallback": self.gaze_fallback,
        }


def analyze_video(path, ref_embedding, cfg, progress=None):
    info = video_info(path)
    rate = sample_rate(cfg["sampling"], info["fps"])
    va = VideoAnalyzer(ref_embedding, cfg, rate)
    max_samples = cfg["sampling"]["max_samples"]
    for frame, t, idx in iter_sampled_frames(path, cfg["sampling"]):
        s = va.process(frame, t)
        if s is not None:
            s["frame"] = idx
        if progress:
            progress(t, info["duration_s"], s)
        if max_samples and va.processed >= max_samples:
            break
    out = va.finalize()
    out["video"] = info
    return out
