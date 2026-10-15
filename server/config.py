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
FACE_SPOOF = "FACE_SPOOF"                 # FAS1: photo / screen-replay presentation attack (MiniFASNet)
SPEECH_DETECTED = "SPEECH_DETECTED"
VOICE_MISMATCH = "VOICE_MISMATCH"
MULTIPLE_SPEAKERS = "MULTIPLE_SPEAKERS"
WHISPERING = "WHISPERING"
SUSPICIOUS_SOUND = "SUSPICIOUS_SOUND"

VIDEO_TYPES = [IDENTITY_MISMATCH, HEAD_POSE_DEVIATION, GAZE_AWAY,
               MULTIPLE_PERSONS, PROHIBITED_OBJECT, FACE_NOT_VISIBLE, FACE_SPOOF]
AUDIO_TYPES = [SPEECH_DETECTED, VOICE_MISMATCH, MULTIPLE_SPEAKERS,
               WHISPERING, SUSPICIOUS_SOUND]
ALL_TYPES = VIDEO_TYPES + AUDIO_TYPES


V2 = {
    # §A.3 sampling
    "sampling": {
        "mode": "fps",          # "fps" = time based, "every_n" = frame index based
        "fps": 10.0,            # FPS10: samples per second (user request; was 3). Live runs as fast as the CPU
                                # allows up to this (newest-frame policy); windows below are in seconds
        "every_n": 10,          # only for mode "every_n"
        "max_samples": None,    # None = whole video
    },
    # §A.4 calibration
    "calibration": {
        "mode": "duration",     # "duration" (seconds) or "count" (samples)
        "duration_s": 3.0,
        "count": 3,
        "min_samples": 5,       # need at least this many accepted samples
        "max_duration_s": 15.0, # give up extending the window after this (slow CPUs: room for min_valid_samples)
        "aggregate": "median",  # "median" or "mean"
        "require_identity": True,
        "max_abs_yaw": 25.0,    # frontal-enough gate (absolute, deg)
        "max_abs_pitch": 25.0,
    },
    # §A.5 head pose
    "head_pose": {
        "euler": "fixed",       # "fixed" (camera-convention model) or "legacy"
        # Fixed thresholds (deg). HP3: used only as the FALLBACK when adaptive thresholds are off or the
        # calibration quality is insufficient (calibration confidence "reduced").
        "yaw_thr": 15.0,
        "pitch_thr": 10.0,
        "roll_thr": 10.0,
        "smooth_window": 3,     # median over last N samples (1 = off); used only when smooth_s is None
        "smooth_s": 1.0,        # FPS10: median window in seconds (= 3 samples at 3 fps, 10 at 10 fps)
        "persist_s": 1.0,       # deviation must last this long to count
        # HP2 (flow.txt §A.5 "Accuracy upgrade")
        "method": "template",   # "template" (person-specific 3D landmark alignment) or "pnp" (generic 6-point model)
        "rigid_ids": [10, 109, 67, 103, 54, 338, 297, 332, 284, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 98, 327,
                      33, 133, 362, 263, 234, 454, 127, 356, 93, 323, 116, 345],
        # deviation ends only when every axis is below threshold - hyst (HP3: per axis; a number = all axes)
        "hyst_deg": {"yaw": 3.0, "pitch": 2.0, "roll": 2.0},
        "adapt_tau_s": 120.0,   # slow baseline adaptation time constant (None = off)
        "adapt_max_deg": 8.0,   # adaptation never moves the baseline further than this (total drift, each axis)
        # HP3 candidate-specific adaptive thresholds (video/pipeline.py, flow.txt §A.5). Calibration window and
        # frontal gate: "calibration" block; smoothing: smooth_s; persistence: persist_s; baseline rate: adapt_tau_s.
        # INITIAL VALUES FOR EVALUATION, not validated thresholds.
        #   threshold = clamp(base_deg + k * robust_variability, min_deg, max_deg)
        #   robust_variability = 1.4826 * MAD of the calibration angles (relative to the candidate's template)
        "adaptive": {
            "enabled": True,
            "base_deg": {"yaw": 12.0, "pitch": 8.0, "roll": 8.0},
            "k": {"yaw": 2.0, "pitch": 2.0, "roll": 2.0},
            "min_deg": {"yaw": 10.0, "pitch": 7.0, "roll": 7.0},
            "max_deg": {"yaw": 25.0, "pitch": 18.0, "roll": 18.0},
            # calibration quality: below min_valid_samples, above max_cal_variability_deg on any axis, or the
            # calibration fallback -> fixed yaw_thr/pitch_thr/roll_thr, confidence "reduced", no threshold adaptation
            "min_valid_samples": 8,
            "max_cal_variability_deg": 4.0,
            "cal_max_motion_deg_s": 30.0,  # calibration sample rejected when the head turns faster than this
            # observation quality
            "min_face_px": 60,             # smaller face box -> low quality (no calibration, no adaptation)
            "max_fit_residual": 0.08,      # template fit residual above this -> landmarks unreliable -> unmeasured
            # online adaptation (baseline and thresholds adapt separately; both freeze on the same conditions)
            "adapt_below_frac": 0.5,       # adapt only while every smoothed axis is below this share of its threshold
            "abrupt_deg_s": 40.0,          # faster head movement freezes adaptation
            "recovery_s": 3.0,             # adaptation stays frozen this long after any freeze condition
            "max_gap_s": 1.0,              # a longer gap without a measurement (face lost, dropped frames) freezes
            "adapt_max_step_deg": 0.05,    # per-update limit of the baseline change (each axis)
            "thr_adapt_tau_s": 300.0,      # threshold adaptation time constant (None = thresholds stay at calibration)
            "thr_max_shift_deg": {"yaw": 3.0, "pitch": 2.0, "roll": 2.0},  # max change vs the calibrated threshold
        },
    },
    # §A.6 gaze
    "gaze": {
        "method": "l2cs",       # GZ3: "l2cs" (L2CS-Net, §A.6 "L2CS-Net gaze"), "v2" (iris landmarks) or "legacy"
        "fallback": "v2",       # used when the L2CS model cannot be loaded (missing torch / weights)
        # GZ3 L2CS-Net (ResNet50, Gaze360 weights, CPU only). Angles in degrees vs the calibrated forward gaze;
        # the gaze already contains the head rotation, so no head pose is added.
        "l2cs": {
            "weights": "models/l2cs/l2cs_gaze360_resnet50.safetensors",
            "input_size": 448,      # recorded videos: upstream training / demo size (~0.2-0.25 s per face on CPU)
            "live_input_size": 224, # live WebRTC: ~65-80 ms per face so the 3 fps budget holds (Py-Feat uses 224)
            "crop_margin": 0.0,     # face-detector box as in the upstream pipeline
            "h_thr_deg": 15.0,      # starting threshold +-15 deg (spec); tune on real sessions
            "v_thr_deg": 15.0,
            "hyst_deg": 2.0,
            "smoothing": "ema",     # "ema" (live) / centred moving average (offline); "median" = rolling median
            "ema_alpha": 0.5,
            "min_eye_deg": 0.0,     # > 0: also require this much eye-in-head angle (gaze - head pose)
            "adapt_tau_s": 120.0,   # slow forward-gaze drift correction (only clearly centred samples)
            "adapt_max_deg": 5.0,
            "torch_threads": 4,
            # DET2: forward-gaze calibration gate. A session calibrated on 2 samples read -48 deg as "forward",
            # so every later look at the screen was "away" (98% GAZE_AWAY). Calibration gaze samples must lie
            # within these absolute angles; with fewer than cal_min_samples the baseline is collected from the
            # first frontal analysis samples instead (gaze not flagged until then).
            "cal_max_abs_h": 25.0,
            "cal_max_abs_v": 35.0,  # webcam above the screen: looking at the screen reads ~10-20 deg down
            "cal_min_samples": 3,     # measured fastest on the dev CPU (12 threads): 4 -> 205 ms vs 8 -> 239 ms at 448
        },
        "legacy_low": 0.35,
        "legacy_high": 0.65,
        "deg_per_ratio_h": 140.0,  # iris-ratio offset -> eye yaw (eyeball geometry, §A.6)
        "deg_per_ratio_v": 140.0,  # GZ2: vertical ratio vs the eye-corner line (use 50 with vertical="lids")
        "h_thr_deg": 22.0,         # combined head+eye angle that counts as away
        "v_thr_deg": 25.0,
        "min_eye_deg": 8.0,        # eye itself must contribute this much
        "blink_open_ratio": 0.12,  # eye opening / eye width below = blink, skip
        "smooth_window": 3,
        "smooth_s": 1.0,           # FPS10: smoothing window in seconds (overrides smooth_window)
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
        "smooth_window": 5,        # majority vote over last N verified samples (if vote_s is None)
        "vote_s": 1.6,             # FPS10: majority-vote window in seconds (= 5 samples at 3 fps)
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
        "persist_gap": 1,          # bridge one missed sample inside a visit (if persist_gap_s is None)
        "persist_gap_s": 0.33,     # FPS10: bridge misses up to this long (1 sample at 3 fps, 3 at 10 fps)
        "yolo_conf_low": 0.35,     # lower-confidence person boxes count only with a face inside
        "static_reject": True,     # ignore faces that do not move (photos / posters / screens)
        "static_samples": 9,       # track length examined (if static_s is None)
        "static_s": 3.0,           # FPS10: track length examined in seconds
        "static_frac": 0.01,       # max std of centre / width, as a fraction of the face width
        # DET2: a face-less YOLO person box overlapping a face-bearing person box by >= this share of its own
        # area is part of that person (arm / hand holding a phone), not a second person (seen in a live session)
        "attach_frac": 0.3,
    },
    # §A.9 objects
    "objects": {
        "method": "v2",            # "v2" (per-class conf) or "legacy" (one shared conf)
        "weights": "yolov8n.pt",   # V8: s vs n measured, see flow.txt §A.9 (n: live CPU budget, s: phone recall)
        # PERF2: YOLO (persons + objects) runs on every yolo_every_n-th sample; head pose / gaze run on every sample.
        # A skipped sample reuses the last YOLO result only while it is <= yolo_max_age_s old (otherwise YOLO runs
        # on it). Reused results are neither new positive nor negative evidence: persistence and the buffer's
        # support counts use the real YOLO observations only (pipeline.finalize, aggregation.denominators.observed).
        "yolo_every_n": 2,
        "yolo_max_age_s": 1.5,
        # display names; 62 = COCO "tv" (monitors) = extra screen (EV1). Earphones are not a COCO class:
        # add them here + in conf/model_names together with custom weights (flow.txt §C.5).
        "classes": {67: "Cell Phone", 73: "Book", 63: "Laptop", 62: "Extra Screen"},
        "conf": {67: 0.35, 73: 0.40, 63: 0.50, 62: 0.50},
        "model_names": {0: "person", 67: "cell phone", 73: "book", 63: "laptop", 62: "tv"},  # verified at start-up
        "legacy_conf": 0.4,
        "persist_k": 2,            # detected in >= k of the last n samples (if persist_window_s is None)
        "persist_n": 3,
        "persist_window_s": 1.0,   # FPS10: n = samples in this window, k = max(persist_k, ceil(persist_frac * n))
        "persist_frac": 0.4,       #   -> 2 of 3 at 3 fps (as before), 4 of 10 at 10 fps
        "imgsz": 640,
        # OB2 (flow.txt §A.9b)
        "roi_pass": True,          # second YOLO pass on the hands / desk region around the face
        "roi_max_frac": 0.6,       # skip the ROI pass if the crop would cover more of the frame
        "weak_conf": 0.20,         # weak detections, promoted only next to a confident one
        "weak_reach_s": 0.33,      # FPS10: "next to" = within this time (1 sample at 3 fps, 3 at 10 fps)
        "ignore_inside_face": 0.8, # ignore object boxes lying this much inside the face box ...
        "ignore_inside_face_max_conf": 0.5,  # DET2: ... unless confident (a phone held up in front of the face)
    },
    # DET2: low light (mean grey level of the frame, 0-255). Dark frames lose phones / partial persons in YOLO.
    "lighting": {
        "low_mean": 60.0,          # below = low light warning (live + report)
        "window_s": 3.0,           # live: mean over this many seconds
        "report_frac": 0.3,        # report warning when >= this share of samples were dark
    },
    # §A.10 denominators / FACE_NOT_VISIBLE
    "face_not_visible": {
        "enabled": True,
        "persist_s": 2.0,
    },
    # FAS1 face anti-spoofing (flow.txt §A.7b): MiniFASNetV2 + MiniFASNetV1SE on the candidate's face, CPU.
    # Decision = confidence threshold on the real-face score + temporal aggregation (mean over window_s, then the
    # violation must last persist_s). Real faces in the user's data: median 0.87-0.99, 1 of 56 live snapshots < 0.5.
    "antispoof": {
        "enabled": True,
        "models_dir": "models/antispoof",
        "real_thr": 0.5,           # mean real-face score below this = spoof
        "window_s": 1.0,           # temporal aggregation (mean of the per-frame scores)
        "persist_s": 2.0,          # spoof must last this long to count
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
        "mix_margin": 0.05,        # VAD3: accepted/rejected mix needs this margin around verify_thr (pairwise rule)
        # VOICE2 (flow.txt §B.5 "VOICE2", tests/calibrate_speaker_live.py: real speech through a simulated live
        # channel - Opus 24/32 kbps, noise 12-30 dB, gain): the candidate's voice is compared with the enrollment
        # print AND a session print (mean of this session's chunks with enrollment similarity >= adapt_thr), so the
        # live channel does not turn the candidate into a "different" voice. Held-out: FRR 13.1% -> 2.3%.
        "adapt_thr": 0.70,
        "adapt_min_chunks": 3,
        # MULTIPLE_SPEAKERS: old rule = any chunk pair < diff_thr within 10 s -> 41% false alarms on single-speaker
        # windows under the live channel. New rule = a run of >= other_run consecutive chunks below other_thr
        # (candidate similarity) next to candidate speech: 0% false alarms, 95.7% detection (held-out).
        "multi_rule": "foreign_run",   # "foreign_run" (VOICE2) or "pairwise" (old)
        "other_thr": 0.70,
        "other_run": 3,            # chunks overlap by 50% -> 3 chunks = ~4 s of the other voice
        "live_min_speech_frac": 0.7,   # live check only when >= 70% of the last chunk is speech
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
                     FACE_SPOOF: "critical",
                     FACE_NOT_VISIBLE: "medium", HEAD_POSE_DEVIATION: "low", SPEECH_DETECTED: "low",
                     SUSPICIOUS_SOUND: "low", GAZE_AWAY: "very low"},
        "max_points": {IDENTITY_MISMATCH: 40, FACE_SPOOF: 40, MULTIPLE_PERSONS: 35, PROHIBITED_OBJECT: 35,
                       VOICE_MISMATCH: 30,
                       MULTIPLE_SPEAKERS: 30, WHISPERING: 20, FACE_NOT_VISIBLE: 12, HEAD_POSE_DEVIATION: 10,
                       SPEECH_DETECTED: 6, SUSPICIOUS_SOUND: 6, GAZE_AWAY: 4},
        "critical_floor": 0.5,     # a critical violation that occurs at all scores at least this share of its max
        "saturation_pct": {IDENTITY_MISMATCH: 20, FACE_SPOOF: 10, MULTIPLE_PERSONS: 10, PROHIBITED_OBJECT: 10,
                           VOICE_MISMATCH: 10,
                           MULTIPLE_SPEAKERS: 10, WHISPERING: 10, FACE_NOT_VISIBLE: 20, HEAD_POSE_DEVIATION: 25,
                           SPEECH_DETECTED: 20, SUSPICIOUS_SOUND: 20, GAZE_AWAY: 30},
        # critical overrides -> INVALID: (percentage threshold, minimum single-event seconds), priority order
        "critical": [(IDENTITY_MISMATCH, 50.0, 10.0, "Unauthorized person detected"),
                     (FACE_SPOOF, 20.0, 5.0, "Face spoofing detected (photo or screen instead of a live face)"),
                     (MULTIPLE_PERSONS, 20.0, 3.0, "Multiple persons detected"),
                     (PROHIBITED_OBJECT, 15.0, 3.0, "Prohibited object detected"),
                     (VOICE_MISMATCH, 10.0, 5.0, "Voice does not match the enrolled candidate"),
                     (MULTIPLE_SPEAKERS, 10.0, 5.0, "Another person speaking")],
        "invalid_score": 50.0,     # SC2: score is now a sum of points (was a weighted mean, 60 / 20)
        "suspicious_score": 15.0,
        "audio_silent_is_suspicious": True,
        "max_thumbnails": 1200,    # FPS10: 3x the samples (was 400 at 3 fps); ~640 px JPEGs in memory
    },
    # BUF1 violation-level tolerance buffer (aggregation/buffer.py), applied after the detectors' own persistence.
    # merge_gap_s  flagged runs closer than this form one event
    # confirm_s    a merged event is confirmed (scored) only if it lasts >= this ...
    # min_support  ... and has >= this many flagged samples (video) / 0.48 s windows (audio)
    # grace_s      session buffer: the first grace_s s of confirmed time earn no points (0 for critical types)
    # review_unconfirmed  unconfirmed events of this type still send the session to review
    # correlated_with     time already confirmed for these types is not scored again for this one
    # STARTING POLICIES TO EVALUATE, NOT VALIDATED THRESHOLDS: tune on labelled recordings against an explicit
    # false-alert target, then check the frozen values on an independent test set.
    "buffer": {
        "enabled": True,
        "min_monitored_s": 5.0,        # less analysed video -> assessment INCONCLUSIVE (retest recommended)
        "min_video_usable_frac": 0.6,  # share of samples that were not too dark (lighting.low_mean)
        "policies": {
            GAZE_AWAY: {"merge_gap_s": 1.0, "confirm_s": 2.0, "min_support": 3, "grace_s": 5.0,
                        "review_unconfirmed": False},
            HEAD_POSE_DEVIATION: {"merge_gap_s": 1.0, "confirm_s": 1.5, "min_support": 2, "grace_s": 3.0,
                                  "review_unconfirmed": False},
            FACE_NOT_VISIBLE: {"merge_gap_s": 1.0, "confirm_s": 2.0, "min_support": 2, "grace_s": 3.0,
                               "review_unconfirmed": False},
            PROHIBITED_OBJECT: {"merge_gap_s": 1.0, "confirm_s": 1.0, "min_support": 2, "grace_s": 0.0,
                                "review_unconfirmed": True},
            MULTIPLE_PERSONS: {"merge_gap_s": 1.0, "confirm_s": 1.0, "min_support": 2, "grace_s": 0.0,
                               "review_unconfirmed": True},
            IDENTITY_MISMATCH: {"merge_gap_s": 1.0, "confirm_s": 2.0, "min_support": 3, "grace_s": 0.0,
                                "review_unconfirmed": True},
            FACE_SPOOF: {"merge_gap_s": 1.0, "confirm_s": 2.0, "min_support": 3, "grace_s": 0.0,
                         "review_unconfirmed": True},
            VOICE_MISMATCH: {"merge_gap_s": 1.0, "confirm_s": 2.0, "min_support": 3, "grace_s": 0.0,
                             "review_unconfirmed": True},
            MULTIPLE_SPEAKERS: {"merge_gap_s": 1.0, "confirm_s": 2.0, "min_support": 3, "grace_s": 0.0,
                                "review_unconfirmed": True},
            WHISPERING: {"merge_gap_s": 1.0, "confirm_s": 1.5, "min_support": 3, "grace_s": 0.0,
                         "review_unconfirmed": False},
            SUSPICIOUS_SOUND: {"merge_gap_s": 2.0, "confirm_s": 1.0, "min_support": 2, "grace_s": 2.0,
                               "review_unconfirmed": False},
            SPEECH_DETECTED: {"merge_gap_s": 0.8, "confirm_s": 0.5, "min_support": 1, "grace_s": 0.0,
                              "review_unconfirmed": False},
        },
    },
    # FUS1 evidence fusion (aggregation/fusion.py). Correlated time is de-duplicated inside an evidence family:
    # "priority_dedup" = members in priority order; time already confirmed for an earlier member is not scored again
    # for a later one (replaces the BUF1 correlated_with lists). Families without a rule are independent.
    "fusion": {
        "families": {
            "identity": {"label": "Identity / presentation attack", "rule": "priority_dedup",
                         "members": [IDENTITY_MISMATCH, FACE_SPOOF]},
            "other_person_video": {"label": "Another person in view", "members": [MULTIPLE_PERSONS]},
            "objects": {"label": "Prohibited objects", "members": [PROHIBITED_OBJECT]},
            "face_presence": {"label": "Face presence", "members": [FACE_NOT_VISIBLE]},
            "head_orientation": {"label": "Head and gaze orientation", "rule": "priority_dedup",
                                 "members": [HEAD_POSE_DEVIATION, GAZE_AWAY]},
            "voice": {"label": "Voices and speech", "rule": "priority_dedup",
                      "members": [VOICE_MISMATCH, MULTIPLE_SPEAKERS, WHISPERING, SPEECH_DETECTED]},
            "environment_sound": {"label": "Environment sounds", "members": [SUSPICIOUS_SOUND]},
        },
        # independent sensors describing the same situation: overlapping confirmed events raise review priority
        # (never the score)
        "corroboration_pairs": [[MULTIPLE_PERSONS, MULTIPLE_SPEAKERS], [MULTIPLE_PERSONS, VOICE_MISMATCH],
                                [IDENTITY_MISMATCH, VOICE_MISMATCH], [PROHIBITED_OBJECT, SUSPICIOUS_SOUND]],
        "relation_gap_s": 1.0,         # events this close count as overlapping for relations
        # observation quality: event points x weight; "low" also blocks the critical floor and the critical overrides
        "quality_weights": {"good": 1.0, "low": 0.8, "unknown": 1.0},
        "low_quality_frac": 0.5,       # video event is "low" when >= this share of its samples is dark / unreliable
        "min_speech_frac": 0.7,        # speaker events need this mean speech fraction (= speaker.live_min_speech_frac)
        # checks without which no assessment is possible (unmeasurable -> INCONCLUSIVE)
        "required_checks": [IDENTITY_MISMATCH, MULTIPLE_PERSONS, PROHIBITED_OBJECT],
        # review priority: high when confirmed events of these severities come from >= N independent families
        "priority_severities": ["critical", "medium"],
        "high_priority_families": 2,
        # audit records
        "max_filtered_per_type": 100,  # filtered raw runs kept per type (the counts are always complete)
        "gap_merge_s": 1.0,            # coverage gaps closer than this are one unavailable record
        "min_gap_s": 1.0,              # shorter coverage gaps are not listed
    },
    # Detector-level validation metrics shown next to each violation (never the accuracy of one candidate's report).
    # Only figures measured in this repository are listed; every other type reports "not measured".
    "validation_metrics": {
        VOICE_MISMATCH: {"dataset": "Mini LibriSpeech held-out + simulated live channel (tests/calibrate_speaker_live.py)",
                         "version": "VOICE2", "false_reject_rate": 0.023, "false_accept_rate": 0.019,
                         "note": "FAR from the clean-channel calibration (A5_speaker_calibration.json)"},
        MULTIPLE_SPEAKERS: {"dataset": "Mini LibriSpeech held-out + simulated live channel (tests/calibrate_speaker_live.py)",
                            "version": "VOICE2 foreign_run", "false_alarm_rate": 0.0, "detection_rate": 0.957},
    },
    # §C.5 / §C.6 evidence package, counters and upload (EV1)
    "evidence": {
        # LIM2: a group is exceeded (session violated) when its share of the analysed session is GREATER than
        # its limit ("percent": head pose % of samples, multiple faces % of samples, voice violations % of audio);
        # prohibited objects use "count" 0 = any detection flags the session
        "limits": {"head_pose": {"basis": "percent", "value": 25.0},
                   "multiple_faces": {"basis": "percent", "value": 5.0},
                   "voice": {"basis": "percent", "value": 5.0},
                   "object": {"basis": "count", "value": 0},
                   "spoof": {"basis": "count", "value": 0}},     # FAS1: any confirmed spoof event
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
LEGACY["head_pose"].update(euler="legacy", smooth_window=1, smooth_s=None, persist_s=0.0, method="pnp", hyst_deg=0.0,
                           adapt_tau_s=None)
LEGACY["head_pose"]["adaptive"] = dict(LEGACY["head_pose"]["adaptive"], enabled=False)
LEGACY["gaze"].update(method="legacy", smooth_window=1, smooth_s=None, persist_s=0.0, hyst_deg=0.0, adapt_tau_s=None)
LEGACY["face"].update(select="first", clamp_bbox=False, crop_margin=0.0, metric="euclidean",
                      smooth_window=1, vote_s=None, persist_s=0.0, mesh_max_faces=1, mesh_must_match=False,
                      min_face_px=0)
LEGACY["persons"].update(method="yolo", persist_s=0.0, persist_gap=0, persist_gap_s=None, static_reject=False)
LEGACY["objects"].update(method="legacy", weights="yolov8n.pt", yolo_every_n=1, persist_k=1, persist_n=1, persist_window_s=None, roi_pass=False,
                         weak_conf=None, ignore_inside_face=None,
                         classes={67: "Cell Phone", 73: "Book", 63: "Laptop"},
                         conf={67: 0.35, 73: 0.40, 63: 0.50})
LEGACY["face_not_visible"].update(enabled=False)
LEGACY["antispoof"].update(enabled=False)
LEGACY["aggregation"].update(include_calibration_in_numerator=True, verdict="legacy")
LEGACY["buffer"].update(enabled=False)

PROFILES = {"v2": V2, "legacy": LEGACY}


def get_config(profile="v2", overrides=None):
    """Return a deep copy of a profile, optionally with nested overrides applied."""
    cfg = copy.deepcopy(PROFILES[profile])
    cfg["profile"] = profile
    for section, values in (overrides or {}).items():
        cfg.setdefault(section, {}).update(values)
    return cfg
