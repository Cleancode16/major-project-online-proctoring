"""
Single source of every threshold, weight and switch used by the pipeline.

Each value is documented in flow.txt PART II (section given in the comment).
Two profiles exist:
  "v2"     - current defaults (all accuracy fixes on)
  "legacy" - reproduces the original app.py (v1) behaviour, used for
             regression checks and before/after accuracy measurements.
"""

import copy
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ENROLL_DIR = DATA_DIR / "enrollment"      # local only, never uploaded
REPORTS_DIR = DATA_DIR / "reports"
MODELS_DIR = ROOT / "models"

# Keep the YAMNet download inside the project so later runs are offline.
os.environ.setdefault("TFHUB_CACHE_DIR", str(MODELS_DIR / "tfhub"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Violation type names (shared by video, audio and aggregation).
IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
HEAD_POSE_DEVIATION = "HEAD_POSE_DEVIATION"
GAZE_AWAY = "GAZE_AWAY"
MULTIPLE_PERSONS = "MULTIPLE_PERSONS"
PROHIBITED_OBJECT = "PROHIBITED_OBJECT"
FACE_NOT_VISIBLE = "FACE_NOT_VISIBLE"
SPEECH_DETECTED = "SPEECH_DETECTED"
VOICE_MISMATCH = "VOICE_MISMATCH"
MULTIPLE_SPEAKERS = "MULTIPLE_SPEAKERS"
WHISPERING = "WHISPERING"
SUSPICIOUS_SOUND = "SUSPICIOUS_SOUND"

VIDEO_TYPES = [IDENTITY_MISMATCH, HEAD_POSE_DEVIATION, GAZE_AWAY,
               MULTIPLE_PERSONS, PROHIBITED_OBJECT, FACE_NOT_VISIBLE]
AUDIO_TYPES = [SPEECH_DETECTED, VOICE_MISMATCH, MULTIPLE_SPEAKERS,
               WHISPERING, SUSPICIOUS_SOUND]
ALL_TYPES = VIDEO_TYPES + AUDIO_TYPES


V2 = {
    # §A.3 sampling
    "sampling": {
        "mode": "fps",          # "fps" = time based, "every_n" = frame index based
        "fps": 3.0,             # samples per second of video
        "every_n": 10,          # only for mode "every_n"
        "max_samples": None,    # None = whole video
    },
    # §A.4 calibration
    "calibration": {
        "mode": "duration",     # "duration" (seconds) or "count" (samples)
        "duration_s": 3.0,
        "count": 3,
        "min_samples": 5,       # need at least this many accepted samples
        "max_duration_s": 10.0, # give up extending the window after this
        "aggregate": "median",  # "median" or "mean"
        "require_identity": True,
        "max_abs_yaw": 25.0,    # frontal-enough gate (absolute, deg)
        "max_abs_pitch": 25.0,
    },
    # §A.5 head pose
    "head_pose": {
        "euler": "fixed",       # "fixed" (camera-convention model) or "legacy"
        "yaw_thr": 15.0,
        "pitch_thr": 10.0,
        "roll_thr": 10.0,
        "smooth_window": 3,     # median over last N samples (1 = off)
        "persist_s": 1.0,       # deviation must last this long to count
        # HP2 (flow.txt §A.5 "Accuracy upgrade")
        "method": "template",   # "template" (person-specific 3D landmark alignment) or "pnp" (generic 6-point model)
        "rigid_ids": [10, 109, 67, 103, 54, 338, 297, 332, 284, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 98, 327,
                      33, 133, 362, 263, 234, 454, 127, 356, 93, 323, 116, 345],
        "hyst_deg": 3.0,        # deviation ends only below threshold - hyst
        "adapt_tau_s": 120.0,   # slow baseline adaptation time constant (None = off)
        "adapt_max_deg": 8.0,   # adaptation never moves the baseline further than this
    },
    # §A.6 gaze
    "gaze": {
        "method": "v2",         # "v2" or "legacy"
        "legacy_low": 0.35,
        "legacy_high": 0.65,
        "deg_per_ratio_h": 140.0,  # iris-ratio offset -> eye yaw (eyeball geometry, §A.6)
        "deg_per_ratio_v": 140.0,  # GZ2: vertical ratio vs the eye-corner line (use 50 with vertical="lids")
        "h_thr_deg": 22.0,         # combined head+eye angle that counts as away
        "v_thr_deg": 25.0,
        "min_eye_deg": 8.0,        # eye itself must contribute this much
        "blink_open_ratio": 0.12,  # eye opening / eye width below = blink, skip
        "smooth_window": 3,
        "persist_s": 1.0,
        "camera_mirrored": False,  # labels are from the candidate's view
        # GZ2 (flow.txt §A.6 "Accuracy upgrade")
        "vertical": "corner_line",  # "corner_line" (skull-fixed) or "lids" (V5)
        "iris_mean": True,          # iris centre = mean of the 5 iris landmarks
        "weight_by_width": True,    # weight the eyes by visible width
        "hyst_deg": 3.0,
        "adapt_tau_s": 120.0,
        "adapt_max_ratio": 0.05,
    },
    # §A.7 face verification
    "face": {
        "det_confidence": 0.6,
        "select": "best_match",    # "best_match" (most similar of several faces), "largest" or "first"
        "clamp_bbox": True,
        "crop_margin": 0.30,       # fraction of bbox added on each side (LFW-calibrated, §A.7)
        "metric": "cosine",        # "cosine" or "euclidean" (legacy)
        "cos_threshold": 0.415,    # LFW train max-accuracy threshold for margin 0.30 (§A.7)
        "euclid_threshold": 1.0,
        "smooth_window": 5,        # majority vote over last N verified samples
        "persist_s": 2.0,
        "mesh_max_faces": 3,
        "mesh_must_match": True,   # mesh centre must lie inside the subject box (else no pose/gaze)
        "min_face_px": 40,         # ignore faces smaller than this (width)
    },
    # §A.8 persons
    "persons": {
        "method": "fused",         # "fused" (YOLO + MediaPipe faces) or "yolo"
        "yolo_conf": 0.5,
        "dedupe_iou": 0.6,
        "min_area_frac": 0.01,
        "persist_s": 1.0,
        # MP2 (flow.txt §A.8 "Accuracy upgrade")
        "persist_gap": 1,          # bridge one missed sample inside a visit
        "yolo_conf_low": 0.35,     # lower-confidence person boxes count only with a face inside
        "static_reject": True,     # ignore faces that do not move (photos / posters / screens)
        "static_samples": 9,       # track length examined (3 s at 3 fps)
        "static_frac": 0.01,       # max std of centre / width, as a fraction of the face width
    },
    # §A.9 objects
    "objects": {
        "method": "v2",            # "v2" (per-class conf) or "legacy" (one shared conf)
        "weights": "yolov8s.pt",   # V8: s vs n measured, see flow.txt §A.9
        # display names; 62 = COCO "tv" (monitors) = extra screen (EV1). Earphones are not a COCO class:
        # add them here + in conf/model_names together with custom weights (flow.txt §C.5).
        "classes": {67: "Cell Phone", 73: "Book", 63: "Laptop", 62: "Extra Screen"},
        "conf": {67: 0.35, 73: 0.40, 63: 0.50, 62: 0.50},
        "model_names": {0: "person", 67: "cell phone", 73: "book", 63: "laptop", 62: "tv"},  # verified at start-up
        "legacy_conf": 0.4,
        "persist_k": 2,            # detected in >= k of the last n samples
        "persist_n": 3,
        "imgsz": 640,
        # OB2 (flow.txt §A.9b)
        "roi_pass": True,          # second YOLO pass on the hands / desk region around the face
        "roi_max_frac": 0.6,       # skip the ROI pass if the crop would cover more of the frame
        "weak_conf": 0.20,         # weak detections, promoted only next to a confident one
        "ignore_inside_face": 0.8, # ignore object boxes lying this much inside the face box
    },
    # §A.10 denominators / FACE_NOT_VISIBLE
    "face_not_visible": {
        "enabled": True,
        "persist_s": 2.0,
    },
    # §B.1 audio sources
    "audio": {
        "sr": 16000,
        "enabled": True,
        "silent_rms_dbfs": -80.0,  # below this the source is treated as muted / blocked
    },
    # §B.2 VAD
    "vad": {
        "engine": "silero",        # "silero" or "webrtc"
        "start_threshold": 0.5,
        "end_threshold": 0.35,
        "min_silence_ms": 300,
        "min_speech_ms": 250,
        "speech_pad_ms": 100,
        "webrtc_mode": 2,
        "window_speech_frac": 0.5,
        # VAD2 (flow.txt §B.2 "Accuracy upgrade")
        "highpass_hz": 80.0,       # None = off
        "prob_median": 3,          # frames (32 ms each)
        "min_seg_snr_db": 6.0,     # segment RMS vs noise floor
        "validate_below_s": 1.0,   # shorter segments need YAMNet speech confirmation
        "yamnet_confirm": 0.30,
        # VAD3 (flow.txt §B.2 "Fix")
        "merge_gap_s": 0.8,        # segments closer than this form one utterance (speaker checks + events)
    },
    # §B.3 voice enrollment
    "enroll": {
        "record_s": 25.0,
        "min_voiced_s": 8.0,
        "min_snr_db": 10.0,
        "warn_snr_db": 15.0,
        "max_clip_frac": 0.001,
        "min_level_dbfs": -45.0,
        "warn_level_dbfs": -38.0,
        "asr_enabled": False,
        "asr_min_match": 0.6,
        "asr_model_dir": str(MODELS_DIR / "vosk-model-small-en-us-0.15"),
        "chunk_s": 3.0,
        "min_consistency": 0.75,
        "max_attempts": 3,
    },
    # §B.4 YAMNet categories
    "yamnet": {
        "thresholds": {"speech": 0.50, "whisper": 0.30, "phone": 0.30, "music": 0.40, "tv_radio": 0.30,
                       "typing": 0.30},
        "suspicious": ["phone", "music", "tv_radio"],
        "persist_k": 2, "persist_n": 3,
        "suspicious_gap_s": 2.0,       # join SUSPICIOUS_SOUND runs across pauses (ring cadence)
        "voiced_frac_whisper": 0.10,   # periodicity cue (§B.4); VAD3: was 0.15
        "voicing_corr": 0.4,           # on 70-1000 Hz band-passed audio (VAD3); was 0.5 unfiltered
        "voicing_band_hz": [70.0, 1000.0],
        "whisper_min_speech_frac": 0.8,
        "whisper_persist": 3,          # consecutive windows (~1.5 s)
    },
    # §B.5 speaker verification (thresholds from Mini LibriSpeech, held-out FRR 0.0% / FAR 1.9%)
    "speaker": {
        "chunk_s": 2.0,
        "verify_thr": 0.72,
        "diff_thr": 0.62,
        "smooth": 3,
        "persist_s": 2.0,
        "multi_window_s": 10.0,
        "min_chunk_s": 1.5,        # VAD3: no speaker decision on shorter speech
        "mix_margin": 0.05,        # VAD3: accepted/rejected mix needs this margin around verify_thr
    },
    # §B.6 audio policy
    "audio_policy": {
        "speech": "flag_after",        # "flag_after" | "flag_all" | "allow"
        "allow_speech_s": 5.0,
    },
    # §C aggregation
    "aggregation": {
        "include_calibration_in_numerator": False,
        "verdict": "v2",           # "v2" (score + overrides) or "legacy"
        # SC2 scoring: every violation adds points to a 0-100 score; it reaches its max_points at saturation_pct.
        # critical = heavy (persons, objects, identity, voice); head pose low; gaze lowest (user decision)
        "severity": {IDENTITY_MISMATCH: "critical", MULTIPLE_PERSONS: "critical", PROHIBITED_OBJECT: "critical",
                     VOICE_MISMATCH: "critical", MULTIPLE_SPEAKERS: "critical", WHISPERING: "critical",
                     FACE_NOT_VISIBLE: "medium", HEAD_POSE_DEVIATION: "low", SPEECH_DETECTED: "low",
                     SUSPICIOUS_SOUND: "low", GAZE_AWAY: "very low"},
        "max_points": {IDENTITY_MISMATCH: 40, MULTIPLE_PERSONS: 35, PROHIBITED_OBJECT: 35, VOICE_MISMATCH: 30,
                       MULTIPLE_SPEAKERS: 30, WHISPERING: 20, FACE_NOT_VISIBLE: 12, HEAD_POSE_DEVIATION: 10,
                       SPEECH_DETECTED: 6, SUSPICIOUS_SOUND: 6, GAZE_AWAY: 4},
        "critical_floor": 0.5,     # a critical violation that occurs at all scores at least this share of its max
        "saturation_pct": {IDENTITY_MISMATCH: 20, MULTIPLE_PERSONS: 10, PROHIBITED_OBJECT: 10, VOICE_MISMATCH: 10,
                           MULTIPLE_SPEAKERS: 10, WHISPERING: 10, FACE_NOT_VISIBLE: 20, HEAD_POSE_DEVIATION: 25,
                           SPEECH_DETECTED: 20, SUSPICIOUS_SOUND: 20, GAZE_AWAY: 30},
        # critical overrides -> INVALID: (percentage threshold, minimum single-event seconds), priority order
        "critical": [(IDENTITY_MISMATCH, 50.0, 10.0, "Unauthorized person detected"),
                     (MULTIPLE_PERSONS, 20.0, 3.0, "Multiple persons detected"),
                     (PROHIBITED_OBJECT, 15.0, 3.0, "Prohibited object detected"),
                     (VOICE_MISMATCH, 10.0, 5.0, "Voice does not match the enrolled candidate"),
                     (MULTIPLE_SPEAKERS, 10.0, 5.0, "Another person speaking")],
        "invalid_score": 50.0,     # SC2: score is now a sum of points (was a weighted mean, 60 / 20)
        "suspicious_score": 15.0,
        "audio_silent_is_suspicious": True,
        "max_thumbnails": 400,
    },
    # §C.5 / §C.6 evidence package, counters and upload (EV1)
    "evidence": {
        # LIM2: a group is exceeded (session violated) when its share of the analysed session is GREATER than
        # its limit ("percent": head pose % of samples, multiple faces % of samples, voice violations % of audio);
        # prohibited objects use "count" 0 = any detection flags the session
        "limits": {"head_pose": {"basis": "percent", "value": 25.0},
                   "multiple_faces": {"basis": "percent", "value": 5.0},
                   "voice": {"basis": "percent", "value": 5.0},
                   "object": {"basis": "count", "value": 0}},
        "clip_pad_s": 0.5,     # audio clip = event +- pad
        "clip_max_s": 30.0,
        "voice_merge_gap_s": 1.0,  # overlapping / adjacent voice violations -> one evidence item
    },
    "evidence_upload": {
        "enabled": True,
        # None = store via the local backend's evidence store (data/evidence_uploads, no network);
        # "http(s)://host:port/api/evidence/upload" = multipart POST to that server
        "url": None,
        "timeout_s": 30,
    },
}


LEGACY = copy.deepcopy(V2)
LEGACY["sampling"].update(mode="every_n", every_n=10, max_samples=50)
LEGACY["calibration"].update(mode="count", count=3, min_samples=3, aggregate="mean",
                             require_identity=False, max_abs_yaw=1e9, max_abs_pitch=1e9)
LEGACY["head_pose"].update(euler="legacy", smooth_window=1, persist_s=0.0, method="pnp", hyst_deg=0.0,
                           adapt_tau_s=None)
LEGACY["gaze"].update(method="legacy", smooth_window=1, persist_s=0.0, hyst_deg=0.0, adapt_tau_s=None)
LEGACY["face"].update(select="first", clamp_bbox=False, crop_margin=0.0, metric="euclidean",
                      smooth_window=1, persist_s=0.0, mesh_max_faces=1, mesh_must_match=False,
                      min_face_px=0)
LEGACY["persons"].update(method="yolo", persist_s=0.0, persist_gap=0, static_reject=False)
LEGACY["objects"].update(method="legacy", weights="yolov8n.pt", persist_k=1, persist_n=1, roi_pass=False,
                         weak_conf=None, ignore_inside_face=None,
                         classes={67: "Cell Phone", 73: "Book", 63: "Laptop"},
                         conf={67: 0.35, 73: 0.40, 63: 0.50})
LEGACY["face_not_visible"].update(enabled=False)
LEGACY["aggregation"].update(include_calibration_in_numerator=True, verdict="legacy")

PROFILES = {"v2": V2, "legacy": LEGACY}


def get_config(profile="v2", overrides=None):
    """Return a deep copy of a profile, optionally with nested overrides applied."""
    cfg = copy.deepcopy(PROFILES[profile])
    cfg["profile"] = profile
    for section, values in (overrides or {}).items():
        cfg.setdefault(section, {}).update(values)
    return cfg
