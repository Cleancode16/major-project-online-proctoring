# Test summary (2026-10-14 18:23)

| Test | Module | Result | Seconds | Last line |
|---|---|---|---|---|
| R1 legacy equivalence | `tests.test_r1_legacy_equivalence` | PASS | 33.5 | R1 LEGACY EQUIVALENCE: PASS |
| V1 head pose | `tests.test_v1_head_pose` | PASS | 30.9 | V1 HEAD POSE: PASS |
| V2-V4 sampling/calibration/pose | `tests.test_v2_v4_sampling_calibration_pose` | PASS | 103.1 | V4: PASS |
| V5 gaze | `tests.test_v5_gaze` | PASS | 54.8 | V5 GAZE: PASS |
| V6 face verification | `tests.test_v6_face` | PASS | 82.3 | V6 FACE: PASS |
| V7/V8 persons/objects | `tests.test_v7_v8_scene` | PASS | 19.3 | V7/V8 SCENE: PASS |
| A1/A2 audio I/O + VAD | `tests.test_a1_a2_audio_vad` | PASS | 13.5 | A1: PASS / A2: PASS |
| A3 voice enrollment | `tests.test_a3_enrollment` | PASS | 27.2 | A3 ENROLLMENT: PASS (9 pass, 0 skipped) |
| A4-A6 audio monitor | `tests.test_a4_a6_audio_monitor` | PASS | 36.5 | A4-A6 AUDIO MONITOR: PASS |
| G1 aggregation | `tests.test_g1_aggregation` | PASS | 34.0 | G1 AGGREGATION: PASS |
| U1 UI | `tests.test_u1_ui` | PASS | 5.8 | U1 UI: PASS |
| T2 end-to-end | `tests.test_e2e` | PASS | 88.3 | T2 E2E: PASS |
| server.py regression | `tests.test_server_regression` | PASS | 34.0 | SERVER REGRESSION: PASS |

## Per-detector sample counts

| Data | Violation | Version | TP | FP | FN | TN |
|---|---|---|---|---|---|---|
| real clip | IDENTITY_MISMATCH | v1 (legacy) | 0 | 0 | 0 | 38 |
| real clip | HEAD_POSE_DEVIATION | v1 (legacy) | 2 | 0 | 0 | 34 |
| real clip | GAZE_AWAY | v1 (legacy) | 0 | 38 | 0 | 0 |
| real clip | MULTIPLE_PERSONS | v1 (legacy) | 4 | 0 | 0 | 33 |
| real clip | PROHIBITED_OBJECT | v1 (legacy) | 11 | 0 | 1 | 25 |
| real clip | FACE_NOT_VISIBLE | v1 (legacy) | 0 | 0 | 0 | 38 |
| real clip | IDENTITY_MISMATCH | v2 | 0 | 0 | 0 | 38 |
| real clip | HEAD_POSE_DEVIATION | v2 | 2 | 0 | 0 | 24 |
| real clip | GAZE_AWAY | v2 | 0 | 0 | 0 | 28 |
| real clip | MULTIPLE_PERSONS | v2 | 4 | 0 | 0 | 33 |
| real clip | PROHIBITED_OBJECT | v2 | 12 | 0 | 0 | 25 |
| real clip | FACE_NOT_VISIBLE | v2 | 0 | 0 | 0 | 38 |
| SYNTHETIC no-face clip | FACE_NOT_VISIBLE | v2 | 9 | 0 | 0 | 29 |

Audio window-level results (SYNTHETIC scenes) are in `log_test_a4_a6_audio_monitor.txt`; LFW / LibriSpeech calibration numbers in `V6_face_calibration.json` / `A5_speaker_calibration.json`.