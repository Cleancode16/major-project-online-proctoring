"""
Flask Backend Server for Online Proctoring System
Provides REST API and WebSocket for real-time proctoring
"""

import os
# Suppress TensorFlow / oneDNN / absl startup noise. Must run BEFORE any TF import.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('ABSL_LOG_MIN_LEVEL', '2')

print(">>> Server starting...")

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import mediapipe as mp
from keras_facenet import FaceNet
from ultralytics import YOLO
import csv
import json
import base64
import io
from PIL import Image
import time
import threading
from collections import deque
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    ort = None

print(">>> Imports done")

# ================= FLASK INITIALIZATION =================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Serialize frame processing across threads to keep MediaPipe stream timestamps monotonic.
FRAME_PROCESS_LOCK = threading.Lock()

# ================= AI MODELS INITIALIZATION =================
print(">>> Loading AI models...")
embedder = FaceNet()
print(">>> FaceNet loaded")

# MediaPipe Face Detection and Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

print(">>> MediaPipe models loaded")

yolo_model = YOLO('yolov8n.pt')
print(">>> YOLO model loaded")

# ================= STATIC 3D FACE MODEL =================
model_3d = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1)
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# Eye landmarks
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
LEFT_IRIS = 468

RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_IRIS = 473

# YOLO classes
PERSON_CLASS = 0
CELL_PHONE_CLASS = 67
BOOK_CLASS = 73
LAPTOP_CLASS = 63

PROHIBITED_CLASSES = {
    CELL_PHONE_CLASS: 'Cell Phone',
    BOOK_CLASS: 'Book',
    LAPTOP_CLASS: 'Laptop'
}

# Passive liveness scaffold (heuristic placeholder)
PASSIVE_LIVENESS_MIN_FACE_RATIO = 0.08
PASSIVE_LIVENESS_MIN_BRIGHTNESS = 50
PASSIVE_LIVENESS_MAX_BRIGHTNESS = 210
PASSIVE_LIVENESS_MIN_BLUR_VARIANCE = 60.0
PASSIVE_EAR_THRESHOLD = 0.21
PASSIVE_EAR_CLOSED_FRAMES_MIN = 2
PASSIVE_EAR_HISTORY_SIZE = 20
PASSIVE_BLINK_EVENT_WINDOW = 60
PASSIVE_BLINK_RECENT_FRAME_WINDOW = 20
PASSIVE_MOTION_HISTORY_SIZE = 20
PASSIVE_MOTION_STATIC_STD_THRESHOLD = 1.0
PASSIVE_MOTION_NATURAL_STD_THRESHOLD = 4.0
PASSIVE_MOTION_MAX_JUMP_THRESHOLD = 12.0

ANTI_SPOOF_MODEL_PATH = os.getenv('ANTI_SPOOF_MODEL_PATH', '').strip()
ANTI_SPOOF_OUTPUT_MODE = os.getenv('ANTI_SPOOF_OUTPUT_MODE', 'auto').strip().lower()

LIVENESS_PROFILE = os.getenv('LIVENESS_PROFILE', 'balanced').strip().lower()
LIVENESS_DEBUG_LOG_ENABLED = os.getenv('LIVENESS_DEBUG_LOG_ENABLED', 'false').strip().lower() in ('1', 'true', 'yes', 'on')
LIVENESS_DEBUG_LOG_FORMAT = os.getenv('LIVENESS_DEBUG_LOG_FORMAT', 'jsonl').strip().lower()
LIVENESS_DEBUG_LOG_PATH = os.getenv('LIVENESS_DEBUG_LOG_PATH', '').strip()

LIVENESS_PROFILE_CONFIGS = {
    'strict': {
        'quality': {
            'min_face_ratio': 0.10,
            'min_brightness': 65,
            'max_brightness': 190,
            'min_blur_variance': 90.0,
        },
        'blink': {
            'threshold': 0.23,
            'closed_frames_min': 2,
            'history_size': 20,
            'recent_frame_window': 18,
        },
        'motion': {
            'history_size': 20,
            'static_std_threshold': 0.8,
            'natural_std_threshold': 3.0,
            'max_jump_threshold': 10.0,
        },
        'weights': {
            'brightness': 0.18,
            'face_ratio': 0.12,
            'texture': 0.12,
            'blink': 0.18,
            'motion': 0.18,
            'anti_spoof': 0.22,
        }
    },
    'balanced': {
        'quality': {
            'min_face_ratio': PASSIVE_LIVENESS_MIN_FACE_RATIO,
            'min_brightness': PASSIVE_LIVENESS_MIN_BRIGHTNESS,
            'max_brightness': PASSIVE_LIVENESS_MAX_BRIGHTNESS,
            'min_blur_variance': PASSIVE_LIVENESS_MIN_BLUR_VARIANCE,
        },
        'blink': {
            'threshold': PASSIVE_EAR_THRESHOLD,
            'closed_frames_min': PASSIVE_EAR_CLOSED_FRAMES_MIN,
            'history_size': PASSIVE_EAR_HISTORY_SIZE,
            'recent_frame_window': PASSIVE_BLINK_RECENT_FRAME_WINDOW,
        },
        'motion': {
            'history_size': PASSIVE_MOTION_HISTORY_SIZE,
            'static_std_threshold': PASSIVE_MOTION_STATIC_STD_THRESHOLD,
            'natural_std_threshold': PASSIVE_MOTION_NATURAL_STD_THRESHOLD,
            'max_jump_threshold': PASSIVE_MOTION_MAX_JUMP_THRESHOLD,
        },
        'weights': {
            'brightness': 0.18,
            'face_ratio': 0.13,
            'texture': 0.13,
            'blink': 0.18,
            'motion': 0.18,
            'anti_spoof': 0.20,
        }
    },
    'lenient': {
        'quality': {
            'min_face_ratio': 0.05,
            'min_brightness': 40,
            'max_brightness': 225,
            'min_blur_variance': 40.0,
        },
        'blink': {
            'threshold': 0.19,
            'closed_frames_min': 2,
            'history_size': 20,
            'recent_frame_window': 24,
        },
        'motion': {
            'history_size': 20,
            'static_std_threshold': 1.5,
            'natural_std_threshold': 5.0,
            'max_jump_threshold': 15.0,
        },
        'weights': {
            'brightness': 0.16,
            'face_ratio': 0.14,
            'texture': 0.14,
            'blink': 0.18,
            'motion': 0.18,
            'anti_spoof': 0.20,
        }
    }
}

ACTIVE_LIVENESS_PROFILE = LIVENESS_PROFILE if LIVENESS_PROFILE in LIVENESS_PROFILE_CONFIGS else 'balanced'
ACTIVE_LIVENESS_CONFIG = LIVENESS_PROFILE_CONFIGS[ACTIVE_LIVENESS_PROFILE]

def get_default_liveness_log_path():
    """Resolve the default debug log path when logging is enabled."""
    if LIVENESS_DEBUG_LOG_PATH:
        return Path(LIVENESS_DEBUG_LOG_PATH)

    logs_dir = Path(__file__).resolve().parent / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    extension = 'csv' if LIVENESS_DEBUG_LOG_FORMAT == 'csv' else 'jsonl'
    return logs_dir / f'liveness_debug.{extension}'

LIVENESS_DEBUG_LOG_FILE = get_default_liveness_log_path() if LIVENESS_DEBUG_LOG_ENABLED else None

# ================= SESSION STATE =================
session_state = {
    'reference_embedding': None,
    'baseline_yaw': None,
    'baseline_pitch': None,
    'baseline_roll': None,
    'baseline_calibrated': False,
    'calibration_yaws': [],
    'calibration_pitches': [],
    'calibration_rolls': [],
    'frame_count': 0,
    'same_person_frames': 0,
    'different_person_frames': 0,
    'deviation_frames': 0,
    'gaze_deviation_frames': 0,
    'multiple_person_frames': 0,
    'prohibited_object_frames': 0,
    'prohibited_objects_detected': {},
    'passive_liveness_score': 0.0,
    'passive_liveness_label': 'suspicious',
    'passive_liveness_stats': {
        'live': 0,
        'suspicious': 0,
        'spoof': 0,
        'quality_insufficient': 0,
        'anti_spoof_model_live': 0,
        'anti_spoof_model_spoof': 0,
        'anti_spoof_model_missing': 0,
        'motion_static': 0,
        'motion_consistent': 0,
        'motion_unstable': 0,
        'motion_variable': 0,
        'avg_score': 0.0,
        'frames_evaluated': 0,
        'blink_count': 0,
        'blink_rate_per_min': 0.0,
        'blink_detected_recently': False,
        'motion_consistency_score': 0.0,
        'motion_label': 'unknown'
    },
    'ear_history': deque(maxlen=ACTIVE_LIVENESS_CONFIG['blink']['history_size']),
    'blink_event_window': deque(maxlen=ACTIVE_LIVENESS_CONFIG['blink']['history_size']),
    'closed_eye_consecutive_frames': 0,
    'blink_count': 0,
    'last_blink_frame': -999,
    'frame_timestamps': deque(maxlen=ACTIVE_LIVENESS_CONFIG['blink']['history_size']),
    'motion_history': deque(maxlen=ACTIVE_LIVENESS_CONFIG['motion']['history_size']),
    'anti_spoof_session': None,
    'anti_spoof_model_info': {
        'available': False,
        'path': None,
        'mode': 'none'
    },
    'is_active': False,
    'CALIBRATION_FRAMES': 3
}

# ================= HELPER FUNCTIONS =================

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        img_array = np.array(img)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None

def get_face_embedding(frame, bbox):
    """Generate face embedding from frame and bounding box"""
    x, y, w, h = bbox
    face_crop = frame[y:y+h, x:x+w]

    if face_crop.size == 0:
        return None

    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_crop = cv2.resize(face_crop, (160, 160))
    return embedder.embeddings([face_crop])[0]

def get_head_pose(frame, landmarks):
    """Calculate head pose angles"""
    h, w = frame.shape[:2]

    image_2d = np.array([
        (int(landmarks[i].x * w), int(landmarks[i].y * h))
        for i in LANDMARK_IDS
    ], dtype=np.float64)

    cam_matrix = np.array([
        [w, 0, w / 2],
        [0, w, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    _, rvec, _ = cv2.solvePnP(
        model_3d,
        image_2d,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    rot_matrix, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot_matrix[0, 0]**2 + rot_matrix[1, 0]**2)

    pitch = np.degrees(np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2]))
    yaw   = np.degrees(np.arctan2(-rot_matrix[2, 0], sy))
    roll  = np.degrees(np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0]))
    
    return yaw, pitch, roll

def get_eye_gaze(frame, landmarks):
    """Calculate eye gaze direction"""
    h, w = frame.shape[:2]
    
    # Left eye
    left_iris = landmarks[LEFT_IRIS]
    left_inner = landmarks[LEFT_EYE_INNER]
    left_outer = landmarks[LEFT_EYE_OUTER]
    
    left_iris_x = left_iris.x * w
    left_inner_x = left_inner.x * w
    left_outer_x = left_outer.x * w
    
    left_eye_width = abs(left_inner_x - left_outer_x)
    left_gaze_ratio = (left_iris_x - left_outer_x) / left_eye_width if left_eye_width > 0 else 0.5
    
    # Right eye
    right_iris = landmarks[RIGHT_IRIS]
    right_inner = landmarks[RIGHT_EYE_INNER]
    right_outer = landmarks[RIGHT_EYE_OUTER]
    
    right_iris_x = right_iris.x * w
    right_inner_x = right_inner.x * w
    right_outer_x = right_outer.x * w
    
    right_eye_width = abs(right_inner_x - right_outer_x)
    right_gaze_ratio = (right_iris_x - right_outer_x) / right_eye_width if right_eye_width > 0 else 0.5
    
    avg_gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2
    
    if avg_gaze_ratio < 0.35:
        gaze_direction = "Looking Right"
    elif avg_gaze_ratio > 0.65:
        gaze_direction = "Looking Left"
    else:
        gaze_direction = "Looking Center"
    
    return left_gaze_ratio, right_gaze_ratio, gaze_direction

def detect_persons_and_objects(frame):
    """Detect persons and prohibited objects using YOLO"""
    results = yolo_model(frame, verbose=False)
    
    person_count = 0
    prohibited_objects = {}
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls == PERSON_CLASS and conf > 0.5:
                person_count += 1
            
            if cls in PROHIBITED_CLASSES and conf > 0.4:
                obj_name = PROHIBITED_CLASSES[cls]
                if obj_name not in prohibited_objects:
                    prohibited_objects[obj_name] = 0
                prohibited_objects[obj_name] += 1
    
    return person_count, prohibited_objects

def point_dist(p1, p2):
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(np.array(p1, dtype=np.float64) - np.array(p2, dtype=np.float64)))

def calculate_ear(landmarks, eye_indices, frame_w, frame_h):
    """Compute Eye Aspect Ratio (EAR) for one eye from FaceMesh landmarks."""
    p1 = (landmarks[eye_indices[0]].x * frame_w, landmarks[eye_indices[0]].y * frame_h)
    p2 = (landmarks[eye_indices[1]].x * frame_w, landmarks[eye_indices[1]].y * frame_h)
    p3 = (landmarks[eye_indices[2]].x * frame_w, landmarks[eye_indices[2]].y * frame_h)
    p4 = (landmarks[eye_indices[3]].x * frame_w, landmarks[eye_indices[3]].y * frame_h)
    p5 = (landmarks[eye_indices[4]].x * frame_w, landmarks[eye_indices[4]].y * frame_h)
    p6 = (landmarks[eye_indices[5]].x * frame_w, landmarks[eye_indices[5]].y * frame_h)

    vertical_1 = point_dist(p2, p6)
    vertical_2 = point_dist(p3, p5)
    horizontal = point_dist(p1, p4)

    if horizontal <= 1e-6:
        return 0.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))

def update_blink_metrics(landmarks, frame_w, frame_h, frame_count):
    """Update blink state using EAR history and return current blink metrics."""
    blink_config = ACTIVE_LIVENESS_CONFIG['blink']
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]

    left_ear = calculate_ear(landmarks, left_eye_indices, frame_w, frame_h)
    right_ear = calculate_ear(landmarks, right_eye_indices, frame_w, frame_h)
    avg_ear = float((left_ear + right_ear) / 2.0)

    session_state['ear_history'].append(avg_ear)
    session_state['frame_timestamps'].append(time.time())

    blink_detected = False

    if avg_ear < blink_config['threshold']:
        session_state['closed_eye_consecutive_frames'] += 1
    else:
        if session_state['closed_eye_consecutive_frames'] >= blink_config['closed_frames_min']:
            session_state['blink_count'] += 1
            session_state['last_blink_frame'] = frame_count
            blink_detected = True
        session_state['closed_eye_consecutive_frames'] = 0

    session_state['blink_event_window'].append(1 if blink_detected else 0)

    timestamps = list(session_state['frame_timestamps'])
    fps_estimate = 0.0
    if len(timestamps) >= 2:
        elapsed = max(1e-6, timestamps[-1] - timestamps[0])
        fps_estimate = float((len(timestamps) - 1) / elapsed)

    window_blinks = int(sum(session_state['blink_event_window']))
    window_size = max(1, len(session_state['blink_event_window']))
    if fps_estimate > 0:
        blink_rate_per_min = float((window_blinks / window_size) * fps_estimate * 60.0)
    else:
        blink_rate_per_min = 0.0

    blink_detected_recently = (frame_count - session_state['last_blink_frame']) <= blink_config['recent_frame_window']

    avg_ear_window = float(np.mean(session_state['ear_history'])) if session_state['ear_history'] else avg_ear

    return {
        'left_ear': float(left_ear),
        'right_ear': float(right_ear),
        'avg_ear': float(avg_ear),
        'avg_ear_window': float(avg_ear_window),
        'blink_detected': blink_detected,
        'blink_detected_recently': blink_detected_recently,
        'blink_count': int(session_state['blink_count']),
        'blink_rate_per_min': float(blink_rate_per_min),
        'fps_estimate': float(fps_estimate)
    }

def evaluate_liveness_quality(frame, bbox):
    """Evaluate frame quality before spoof/liveness judgment."""
    quality_config = ACTIVE_LIVENESS_CONFIG['quality']
    x, y, w, h = bbox
    frame_h, frame_w = frame.shape[:2]

    if w <= 0 or h <= 0:
        return {
            'quality_ok': False,
            'quality_insufficient': True,
            'reason': 'Face bounding box unavailable',
            'guidance': 'Keep your face centered and clearly visible to the camera.',
            'signals': {
                'brightness': 0.0,
                'blur_variance': 0.0,
                'face_size_ratio': 0.0
            }
        }

    face_crop = frame[y:y+h, x:x+w]
    if face_crop.size == 0:
        return {
            'quality_ok': False,
            'quality_insufficient': True,
            'reason': 'Face crop unavailable',
            'guidance': 'Keep your face centered and clearly visible to the camera.',
            'signals': {
                'brightness': 0.0,
                'blur_variance': 0.0,
                'face_size_ratio': 0.0
            }
        }

    gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray_crop))
    blur_variance = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    face_size_ratio = float((w * h) / max(1, frame_w * frame_h))

    low_brightness = brightness < quality_config['min_brightness'] or brightness > quality_config['max_brightness']
    low_blur = blur_variance < quality_config['min_blur_variance']
    small_face = face_size_ratio < quality_config['min_face_ratio']

    quality_ok = not (low_brightness or low_blur or small_face)

    reasons = []
    if low_brightness:
        reasons.append('lighting')
    if low_blur:
        reasons.append('blur')
    if small_face:
        reasons.append('face size')

    if quality_ok:
        guidance = 'Face quality looks usable for liveness analysis.'
        reason_text = 'Quality sufficient'
    else:
        guidance = 'Face quality is too low. Improve lighting, hold the camera steady, and move your face closer to the frame.'
        reason_text = f"Low quality: {', '.join(reasons)}"

    return {
        'quality_ok': quality_ok,
        'quality_insufficient': not quality_ok,
        'reason': reason_text,
        'guidance': guidance,
        'signals': {
            'brightness': brightness,
            'blur_variance': blur_variance,
            'face_size_ratio': face_size_ratio
        }
    }

def _shape_to_int(value, default_value):
    """Convert ONNX dynamic shape dimension to int when possible."""
    try:
        return int(value)
    except Exception:
        return default_value


def run_heuristic_anti_spoof(face_crop, mode='heuristic-fallback'):
    """Fallback anti-spoof score using image quality cues when no ONNX model is available."""
    if face_crop is None or face_crop.size == 0:
        return {
            'available': True,
            'live_score': 0.5,
            'spoof_score': 0.5,
            'label': 'suspicious',
            'mode': mode
        }

    gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray_crop))
    blur_variance = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())

    brightness_score = 1.0 if 55.0 <= brightness <= 205.0 else 0.45
    texture_score = float(np.clip(blur_variance / 120.0, 0.0, 1.0))
    live_score = float(np.clip((0.55 * brightness_score) + (0.45 * texture_score), 0.0, 1.0))
    spoof_score = float(1.0 - live_score)
    label = 'live' if live_score >= 0.58 else 'spoof'

    return {
        'available': True,
        'live_score': live_score,
        'spoof_score': spoof_score,
        'label': label,
        'mode': mode
    }

def load_anti_spoof_session():
    """Load an optional ONNX anti-spoof model if configured."""
    if not ANTI_SPOOF_MODEL_PATH:
        session_state['anti_spoof_model_info'] = {
            'available': True,
            'path': 'built-in',
            'mode': 'heuristic-fallback'
        }
        return None

    if ort is None:
        print('>>> ONNX Runtime not installed; anti-spoof model disabled')
        session_state['anti_spoof_model_info'] = {
            'available': False,
            'path': ANTI_SPOOF_MODEL_PATH,
            'mode': 'onnxruntime-missing'
        }
        return None

    if not os.path.exists(ANTI_SPOOF_MODEL_PATH):
        print(f">>> Anti-spoof model not found: {ANTI_SPOOF_MODEL_PATH}")
        session_state['anti_spoof_model_info'] = {
            'available': False,
            'path': ANTI_SPOOF_MODEL_PATH,
            'mode': 'missing-file'
        }
        return None

    try:
        session = ort.InferenceSession(ANTI_SPOOF_MODEL_PATH, providers=['CPUExecutionProvider'])
        session_state['anti_spoof_model_info'] = {
            'available': True,
            'path': ANTI_SPOOF_MODEL_PATH,
            'mode': 'onnxruntime'
        }
        print(f">>> Anti-spoof model loaded: {ANTI_SPOOF_MODEL_PATH}")
        return session
    except Exception as exc:
        print(f">>> Failed to load anti-spoof model: {exc}")
        session_state['anti_spoof_model_info'] = {
            'available': False,
            'path': ANTI_SPOOF_MODEL_PATH,
            'mode': 'load-failed'
        }
        return None

def run_anti_spoof_inference(face_crop):
    """Run optional anti-spoof inference and return a normalized live score."""
    session = session_state.get('anti_spoof_session')
    if session is None:
        return run_heuristic_anti_spoof(face_crop)

    try:
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        input_shape = input_meta.shape

        channels_first = len(input_shape) >= 4 and _shape_to_int(input_shape[1], 3) in (1, 3)
        if channels_first:
            target_h = _shape_to_int(input_shape[2], 224)
            target_w = _shape_to_int(input_shape[3], 224)
        else:
            target_h = _shape_to_int(input_shape[1], 224)
            target_w = _shape_to_int(input_shape[2], 224)

        resized = cv2.resize(face_crop, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0

        if channels_first:
            model_input = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        else:
            model_input = normalized[np.newaxis, ...]

        outputs = session.run(None, {input_name: model_input})
        if not outputs:
            return run_heuristic_anti_spoof(face_crop, mode='heuristic-no-output')

        raw_output = np.array(outputs[0]).astype(np.float32).flatten()
        if raw_output.size == 0:
            return run_heuristic_anti_spoof(face_crop, mode='heuristic-empty-output')

        output_mode = ANTI_SPOOF_OUTPUT_MODE
        if raw_output.size == 1:
            raw_value = float(raw_output[0])
            live_score = float(1.0 / (1.0 + np.exp(-raw_value)))
            spoof_score = 1.0 - live_score
            if output_mode == 'spoof_probability':
                spoof_score = float(np.clip(raw_value, 0.0, 1.0))
                live_score = 1.0 - spoof_score
            label = 'live' if live_score >= spoof_score else 'spoof'
        else:
            probs = np.exp(raw_output - np.max(raw_output))
            probs = probs / np.sum(probs)
            spoof_index = 0
            live_index = int(np.argmax(probs))

            if output_mode == 'spoof_probability' and raw_output.size >= 2:
                spoof_index = min(1, raw_output.size - 1)
                spoof_score = float(probs[spoof_index])
                live_score = 1.0 - spoof_score
                label = 'live' if live_score >= spoof_score else 'spoof'
            elif output_mode == 'live_probability' and raw_output.size >= 2:
                live_index = min(1, raw_output.size - 1)
                live_score = float(probs[live_index])
                spoof_score = 1.0 - live_score
                label = 'live' if live_score >= spoof_score else 'spoof'
            else:
                live_score = float(np.max(probs))
                spoof_score = 1.0 - live_score
                label = 'live' if live_score >= 0.5 else 'spoof'

        return {
            'available': True,
            'live_score': float(np.clip(live_score, 0.0, 1.0)),
            'spoof_score': float(np.clip(spoof_score, 0.0, 1.0)),
            'label': label,
            'mode': 'onnxruntime'
        }
    except Exception as exc:
        print(f">>> Anti-spoof inference failed: {exc}")
        return run_heuristic_anti_spoof(face_crop, mode='heuristic-error')

def update_motion_consistency(frame_count, bbox, frame_w, frame_h, yaw, pitch, roll, landmarks, blink_metrics):
    """Track short-window motion consistency from face box and pose changes."""
    motion_config = ACTIVE_LIVENESS_CONFIG['motion']
    x, y, w, h = bbox
    center_x = float(x + (w / 2.0))
    center_y = float(y + (h / 2.0))

    landmark_points = np.array([
        [landmarks[idx].x * frame_w, landmarks[idx].y * frame_h]
        for idx in [1, 33, 61, 199, 263, 291]
    ], dtype=np.float64)
    landmark_center = np.mean(landmark_points, axis=0) if len(landmark_points) else np.array([center_x, center_y], dtype=np.float64)

    motion_sample = {
        'frame': frame_count,
        'center_x': center_x,
        'center_y': center_y,
        'yaw': float(yaw),
        'pitch': float(pitch),
        'roll': float(roll),
        'landmark_center_x': float(landmark_center[0]),
        'landmark_center_y': float(landmark_center[1]),
        'blink_recent': bool(blink_metrics.get('blink_detected_recently', False)) if blink_metrics else False
    }

    session_state['motion_history'].append(motion_sample)

    samples = list(session_state['motion_history'])
    if len(samples) < 3:
        return {
            'motion_consistency_score': 0.5,
            'motion_label': 'warming_up',
            'motion_std': 0.0,
            'motion_delta': 0.0,
            'motion_center_jitter': 0.0,
            'motion_pose_jitter': 0.0,
            'motion_history_size': len(samples)
        }

    center_x_values = np.array([sample['center_x'] for sample in samples], dtype=np.float64)
    center_y_values = np.array([sample['center_y'] for sample in samples], dtype=np.float64)
    yaw_values = np.array([sample['yaw'] for sample in samples], dtype=np.float64)
    pitch_values = np.array([sample['pitch'] for sample in samples], dtype=np.float64)
    roll_values = np.array([sample['roll'] for sample in samples], dtype=np.float64)

    center_jitter = float(np.sqrt(np.std(center_x_values) ** 2 + np.std(center_y_values) ** 2))
    pose_jitter = float(np.sqrt(np.std(yaw_values) ** 2 + np.std(pitch_values) ** 2 + np.std(roll_values) ** 2))

    center_delta = float(np.sqrt((center_x_values[-1] - center_x_values[-2]) ** 2 + (center_y_values[-1] - center_y_values[-2]) ** 2))
    pose_delta = float(np.sqrt((yaw_values[-1] - yaw_values[-2]) ** 2 + (pitch_values[-1] - pitch_values[-2]) ** 2 + (roll_values[-1] - roll_values[-2]) ** 2))
    motion_delta = float(np.sqrt(center_delta ** 2 + pose_delta ** 2))

    if motion_delta > motion_config['max_jump_threshold']:
        motion_label = 'unstable'
        motion_consistency_score = 0.2
    elif center_jitter < motion_config['static_std_threshold'] and pose_jitter < motion_config['static_std_threshold']:
        motion_label = 'static'
        motion_consistency_score = 0.25
    elif center_jitter <= motion_config['natural_std_threshold'] and pose_jitter <= motion_config['natural_std_threshold']:
        motion_label = 'consistent'
        motion_consistency_score = 0.9
    else:
        motion_label = 'variable'
        motion_consistency_score = 0.55

    if blink_metrics and blink_metrics.get('blink_detected_recently', False):
        motion_consistency_score = min(1.0, motion_consistency_score + 0.05)

    return {
        'motion_consistency_score': float(np.clip(motion_consistency_score, 0.0, 1.0)),
        'motion_label': motion_label,
        'motion_std': float(max(center_jitter, pose_jitter)),
        'motion_delta': float(motion_delta),
        'motion_center_jitter': float(center_jitter),
        'motion_pose_jitter': float(pose_jitter),
        'motion_history_size': len(samples)
    }

def compute_passive_liveness_placeholder(frame, bbox, blink_metrics=None, quality_gate=None, motion_metrics=None, anti_spoof_metrics=None):
    """Heuristic-only passive liveness scaffold (no external model)."""
    weights = ACTIVE_LIVENESS_CONFIG['weights']
    x, y, w, h = bbox
    frame_h, frame_w = frame.shape[:2]

    if w <= 0 or h <= 0:
        return {
            'score': 0.3,
            'label': 'suspicious',
            'mode': 'heuristic-placeholder',
            'signals': {
                'brightness': 0.0,
                'face_size_ratio': 0.0,
                'texture_variance': 0.0
            }
        }

    face_crop = frame[y:y+h, x:x+w]
    if face_crop.size == 0:
        return {
            'score': 0.3,
            'label': 'suspicious',
            'mode': 'heuristic-placeholder',
            'signals': {
                'brightness': 0.0,
                'face_size_ratio': 0.0,
                'texture_variance': 0.0
            }
        }

    gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray_crop))
    texture_variance = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    face_size_ratio = float((w * h) / max(1, frame_w * frame_h))

    quality_config = ACTIVE_LIVENESS_CONFIG['quality']
    brightness_score = 1.0 if quality_config['min_brightness'] <= brightness <= quality_config['max_brightness'] else 0.45
    face_ratio_score = 1.0 if face_size_ratio >= quality_config['min_face_ratio'] else 0.4
    texture_score = min(1.0, texture_variance / 120.0)

    blink_score = 0.45
    blink_rate = 0.0
    blink_detected_recently = False
    motion_consistency_score = 0.5
    motion_label = 'unknown'
    motion_std = 0.0
    motion_delta = 0.0
    motion_center_jitter = 0.0
    motion_pose_jitter = 0.0
    anti_spoof_live_score = 0.5
    anti_spoof_spoof_score = 0.5
    anti_spoof_label = 'unavailable'
    anti_spoof_mode = 'fallback'

    if blink_metrics is not None:
        blink_rate = float(blink_metrics.get('blink_rate_per_min', 0.0))
        blink_detected_recently = bool(blink_metrics.get('blink_detected_recently', False))

        if 6.0 <= blink_rate <= 35.0:
            blink_rate_score = 1.0
        elif 3.0 <= blink_rate < 6.0 or 35.0 < blink_rate <= 45.0:
            blink_rate_score = 0.7
        else:
            blink_rate_score = 0.4

        blink_recency_score = 1.0 if blink_detected_recently else 0.5
        blink_score = (0.6 * blink_rate_score) + (0.4 * blink_recency_score)

    if motion_metrics is not None:
        motion_consistency_score = float(motion_metrics.get('motion_consistency_score', 0.5))
        motion_label = motion_metrics.get('motion_label', 'unknown')
        motion_std = float(motion_metrics.get('motion_std', 0.0))
        motion_delta = float(motion_metrics.get('motion_delta', 0.0))
        motion_center_jitter = float(motion_metrics.get('motion_center_jitter', 0.0))
        motion_pose_jitter = float(motion_metrics.get('motion_pose_jitter', 0.0))

    if anti_spoof_metrics is not None:
        anti_spoof_live_score = float(anti_spoof_metrics.get('live_score', 0.5))
        anti_spoof_spoof_score = float(anti_spoof_metrics.get('spoof_score', 0.5))
        anti_spoof_label = anti_spoof_metrics.get('label', 'unavailable')
        anti_spoof_mode = anti_spoof_metrics.get('mode', 'fallback')

    score = (
        (weights['brightness'] * brightness_score) +
        (weights['face_ratio'] * face_ratio_score) +
        (weights['texture'] * texture_score) +
        (weights['blink'] * blink_score) +
        (weights['motion'] * motion_consistency_score) +
        (weights['anti_spoof'] * anti_spoof_live_score)
    )
    score = float(np.clip(score, 0.0, 1.0))

    quality_insufficient = False
    quality_reason = 'Quality sufficient'
    quality_guidance = 'Face quality looks usable for liveness analysis.'
    quality_signals = {
        'brightness': brightness,
        'blur_variance': texture_variance,
        'face_size_ratio': face_size_ratio
    }

    if quality_gate is not None:
        quality_insufficient = bool(quality_gate.get('quality_insufficient', False))
        quality_reason = quality_gate.get('reason', quality_reason)
        quality_guidance = quality_gate.get('guidance', quality_guidance)
        quality_signals = quality_gate.get('signals', quality_signals)

    if quality_insufficient:
        label = 'quality_insufficient'
    elif score >= 0.72:
        label = 'live'
    elif score >= 0.5:
        label = 'suspicious'
    else:
        label = 'spoof'

    return {
        'score': score,
        'label': label,
        'quality_insufficient': quality_insufficient,
        'quality_reason': quality_reason,
        'quality_guidance': quality_guidance,
        'motion_consistency_score': float(motion_consistency_score),
        'motion_label': motion_label,
        'motion_std': float(motion_std),
        'motion_delta': float(motion_delta),
        'motion_center_jitter': float(motion_center_jitter),
        'motion_pose_jitter': float(motion_pose_jitter),
        'anti_spoof_live_score': float(anti_spoof_live_score),
        'anti_spoof_spoof_score': float(anti_spoof_spoof_score),
        'anti_spoof_label': anti_spoof_label,
        'anti_spoof_mode': anti_spoof_mode,
        'mode': 'heuristic-placeholder',
        'signals': {
            'brightness': brightness,
            'face_size_ratio': face_size_ratio,
            'texture_variance': texture_variance,
            'blur_variance': quality_signals.get('blur_variance', texture_variance),
            'quality_ok': not quality_insufficient,
            'quality_reason': quality_reason,
            'quality_guidance': quality_guidance,
            'blink_rate_per_min': blink_rate,
            'blink_detected_recently': blink_detected_recently,
            'blink_score': float(blink_score),
            'motion_consistency_score': float(motion_consistency_score),
            'motion_label': motion_label,
            'motion_std': float(motion_std),
            'motion_delta': float(motion_delta),
            'motion_center_jitter': float(motion_center_jitter),
            'motion_pose_jitter': float(motion_pose_jitter),
            'anti_spoof_live_score': float(anti_spoof_live_score),
            'anti_spoof_spoof_score': float(anti_spoof_spoof_score),
            'anti_spoof_label': anti_spoof_label,
            'anti_spoof_mode': anti_spoof_mode
        }
    }

session_state['anti_spoof_session'] = load_anti_spoof_session()

def write_liveness_debug_entry(entry):
    """Write a passive liveness debug entry when logging is enabled."""
    if not LIVENESS_DEBUG_LOG_ENABLED or LIVENESS_DEBUG_LOG_FILE is None:
        return

    try:
        LIVENESS_DEBUG_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if LIVENESS_DEBUG_LOG_FORMAT == 'csv':
            flat_entry = {}
            for key, value in entry.items():
                if isinstance(value, dict):
                    flat_entry[key] = json.dumps(value, ensure_ascii=True)
                else:
                    flat_entry[key] = value

            file_exists = LIVENESS_DEBUG_LOG_FILE.exists()
            with LIVENESS_DEBUG_LOG_FILE.open('a', newline='', encoding='utf-8') as log_file:
                writer = csv.DictWriter(log_file, fieldnames=list(flat_entry.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(flat_entry)
        else:
            with LIVENESS_DEBUG_LOG_FILE.open('a', encoding='utf-8') as log_file:
                log_file.write(json.dumps(entry, ensure_ascii=True) + '\n')
    except Exception as exc:
        print(f">>> Failed to write liveness debug entry: {exc}")

# ================= API ENDPOINTS =================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'timestamp': time.time()
    })

@app.route('/api/set-reference', methods=['POST'])
def set_reference():
    """Set reference image for face verification"""
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        frame = base64_to_image(image_data)
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_detection.process(rgb)
        
        if not result.detections:
            return jsonify({'error': 'No face detected in reference image'}), 400
        
        d = result.detections[0].location_data.relative_bounding_box
        h, w = frame.shape[:2]
        bbox = (
            max(0, int(d.xmin * w)),
            max(0, int(d.ymin * h)),
            int(d.width * w),
            int(d.height * h)
        )
        
        embedding = get_face_embedding(frame, bbox)
        if embedding is None:
            return jsonify({'error': 'Failed to generate face embedding'}), 400
        
        session_state['reference_embedding'] = embedding
        
        # Reset session
        session_state['baseline_calibrated'] = False
        session_state['calibration_yaws'] = []
        session_state['calibration_pitches'] = []
        session_state['calibration_rolls'] = []
        session_state['frame_count'] = 0
        session_state['same_person_frames'] = 0
        session_state['different_person_frames'] = 0
        session_state['deviation_frames'] = 0
        session_state['gaze_deviation_frames'] = 0
        session_state['multiple_person_frames'] = 0
        session_state['prohibited_object_frames'] = 0
        session_state['prohibited_objects_detected'] = {}
        session_state['passive_liveness_score'] = 0.0
        session_state['passive_liveness_label'] = 'suspicious'
        session_state['passive_liveness_stats'] = {
            'live': 0,
            'suspicious': 0,
            'spoof': 0,
            'quality_insufficient': 0,
            'anti_spoof_model_live': 0,
            'anti_spoof_model_spoof': 0,
            'anti_spoof_model_missing': 0,
            'avg_score': 0.0,
            'frames_evaluated': 0,
            'blink_count': 0,
            'blink_rate_per_min': 0.0,
            'blink_detected_recently': False
        }
        session_state['ear_history'].clear()
        session_state['blink_event_window'].clear()
        session_state['closed_eye_consecutive_frames'] = 0
        session_state['blink_count'] = 0
        session_state['last_blink_frame'] = -999
        session_state['frame_timestamps'].clear()
        session_state['motion_history'].clear()
        
        return jsonify({
            'success': True,
            'message': 'Reference image set successfully'
        })
    
    except Exception as e:
        print(f"Error in set_reference: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-session', methods=['POST'])
def start_session():
    """Start proctoring session"""
    if session_state['reference_embedding'] is None:
        return jsonify({'error': 'Reference image not set'}), 400
    
    session_state['is_active'] = True
    session_state['ear_history'].clear()
    session_state['blink_event_window'].clear()
    session_state['closed_eye_consecutive_frames'] = 0
    session_state['blink_count'] = 0
    session_state['last_blink_frame'] = -999
    session_state['frame_timestamps'].clear()
    session_state['motion_history'].clear()
    return jsonify({'success': True, 'message': 'Session started'})

@app.route('/api/stop-session', methods=['POST'])
def stop_session():
    """Stop proctoring session"""
    session_state['is_active'] = False
    return jsonify({'success': True, 'message': 'Session stopped'})

@app.route('/api/get-stats', methods=['GET'])
def get_stats():
    """Get current session statistics"""
    total_frames = session_state['same_person_frames'] + session_state['different_person_frames']
    analyzed_frames = max(0, total_frames - session_state['CALIBRATION_FRAMES'])
    
    return jsonify({
        'total_frames': total_frames,
        'analyzed_frames': analyzed_frames,
        'calibrated': session_state['baseline_calibrated'],
        'baseline': {
            'yaw': session_state['baseline_yaw'],
            'pitch': session_state['baseline_pitch'],
            'roll': session_state['baseline_roll']
        } if session_state['baseline_calibrated'] else None,
        'stats': {
            'same_person': session_state['same_person_frames'],
            'different_person': session_state['different_person_frames'],
            'deviation': session_state['deviation_frames'],
            'gaze_deviation': session_state['gaze_deviation_frames'],
            'multiple_person': session_state['multiple_person_frames'],
            'prohibited_object': session_state['prohibited_object_frames'],
            'quality_insufficient': session_state['passive_liveness_stats'].get('quality_insufficient', 0),
            'motion_static': session_state['passive_liveness_stats'].get('motion_static', 0),
            'motion_consistent': session_state['passive_liveness_stats'].get('motion_consistent', 0),
            'motion_unstable': session_state['passive_liveness_stats'].get('motion_unstable', 0),
            'motion_variable': session_state['passive_liveness_stats'].get('motion_variable', 0)
        },
        'passive_liveness': {
            'score': session_state['passive_liveness_score'],
            'label': session_state['passive_liveness_label'],
            'stats': session_state['passive_liveness_stats'],
            'model': session_state['anti_spoof_model_info']
        },
        'tuning': {
            'profile': ACTIVE_LIVENESS_PROFILE,
            'profiles_available': list(LIVENESS_PROFILE_CONFIGS.keys()),
            'debug_logging_enabled': LIVENESS_DEBUG_LOG_ENABLED,
            'debug_log_format': LIVENESS_DEBUG_LOG_FORMAT,
            'debug_log_path': str(LIVENESS_DEBUG_LOG_FILE) if LIVENESS_DEBUG_LOG_FILE else None
        },
        'thresholds': {
            'passive_liveness_min_face_ratio': ACTIVE_LIVENESS_CONFIG['quality']['min_face_ratio'],
            'passive_liveness_min_brightness': ACTIVE_LIVENESS_CONFIG['quality']['min_brightness'],
            'passive_liveness_max_brightness': ACTIVE_LIVENESS_CONFIG['quality']['max_brightness'],
            'passive_liveness_min_blur_variance': ACTIVE_LIVENESS_CONFIG['quality']['min_blur_variance'],
            'passive_ear_threshold': ACTIVE_LIVENESS_CONFIG['blink']['threshold'],
            'passive_ear_closed_frames_min': ACTIVE_LIVENESS_CONFIG['blink']['closed_frames_min'],
            'passive_ear_history_size': ACTIVE_LIVENESS_CONFIG['blink']['history_size'],
            'passive_blink_recent_frame_window': ACTIVE_LIVENESS_CONFIG['blink']['recent_frame_window'],
            'passive_motion_history_size': ACTIVE_LIVENESS_CONFIG['motion']['history_size'],
            'passive_motion_static_std_threshold': ACTIVE_LIVENESS_CONFIG['motion']['static_std_threshold'],
            'passive_motion_natural_std_threshold': ACTIVE_LIVENESS_CONFIG['motion']['natural_std_threshold'],
            'passive_motion_max_jump_threshold': ACTIVE_LIVENESS_CONFIG['motion']['max_jump_threshold'],
            'anti_spoof_model_path': ANTI_SPOOF_MODEL_PATH or None,
            'anti_spoof_output_mode': ANTI_SPOOF_OUTPUT_MODE
        },
        'prohibited_objects': session_state['prohibited_objects_detected']
    })

# ================= WEBSOCKET EVENTS =================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'message': 'Connected to proctoring server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('process_frame')
def handle_frame(data):
    """Process a single frame from client"""
    if not FRAME_PROCESS_LOCK.acquire(blocking=False):
        # Drop overlapping frames instead of processing concurrently.
        return

    try:
        if not session_state['is_active']:
            emit('frame_result', {'error': 'Session not active'})
            return
        
        if session_state['reference_embedding'] is None:
            emit('frame_result', {'error': 'Reference not set'})
            return
        
        image_data = data.get('image')
        frame = base64_to_image(image_data)
        
        if frame is None:
            emit('frame_result', {'error': 'Invalid frame data'})
            return
        
        session_state['frame_count'] += 1
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det_result = face_detection.process(rgb)
        
        if not det_result.detections:
            emit('frame_result', {
                'no_face': True,
                'message': 'No face detected',
                'frame_count': session_state['frame_count']
            })
            return
        
        # Get bounding box
        d = det_result.detections[0].location_data.relative_bounding_box
        h, w = frame.shape[:2]
        bbox = (
            max(0, int(d.xmin * w)),
            max(0, int(d.ymin * h)),
            int(d.width * w),
            int(d.height * h)
        )
        face_crop = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        anti_spoof_metrics = run_anti_spoof_inference(face_crop) if face_crop.size > 0 else {
            'available': False,
            'live_score': 0.5,
            'spoof_score': 0.5,
            'label': 'missing-crop',
            'mode': 'fallback'
        }

        # Face verification
        embedding = get_face_embedding(frame, bbox)
        if embedding is None:
            emit('frame_result', {'error': 'Failed to generate embedding'})
            return
        
        distance = np.linalg.norm(embedding - session_state['reference_embedding'])
        
        if distance < 1.0:
            session_state['same_person_frames'] += 1
            identity = "Authorized"
        else:
            session_state['different_person_frames'] += 1
            identity = "Unauthorized"
        
        # Head pose
        mesh_result = face_mesh.process(rgb)
        if not mesh_result.multi_face_landmarks:
            emit('frame_result', {'error': 'No face mesh detected'})
            return
        
        landmarks = mesh_result.multi_face_landmarks[0].landmark

        # EAR-based temporal blink signal
        blink_metrics = update_blink_metrics(landmarks, w, h, session_state['frame_count'])

        yaw, pitch, roll = get_head_pose(frame, landmarks)

        # Quality gate before spoof/liveness judgment
        quality_gate = evaluate_liveness_quality(frame, bbox)

        # Motion consistency signal from short-window face movement
        motion_metrics = update_motion_consistency(
            session_state['frame_count'],
            bbox,
            w,
            h,
            yaw,
            pitch,
            roll,
            landmarks,
            blink_metrics
        )

        # Passive liveness scaffold (heuristic mode + blink + motion signals)
        passive_liveness = compute_passive_liveness_placeholder(
            frame,
            bbox,
            blink_metrics=blink_metrics,
            quality_gate=quality_gate,
            motion_metrics=motion_metrics,
            anti_spoof_metrics=anti_spoof_metrics
        )
        session_state['passive_liveness_score'] = passive_liveness['score']
        session_state['passive_liveness_label'] = passive_liveness['label']
        if passive_liveness['label'] not in session_state['passive_liveness_stats']:
            session_state['passive_liveness_stats'][passive_liveness['label']] = 0
        session_state['passive_liveness_stats'][passive_liveness['label']] += 1
        session_state['passive_liveness_stats']['frames_evaluated'] += 1
        session_state['passive_liveness_stats']['blink_count'] = session_state['blink_count']
        session_state['passive_liveness_stats']['blink_rate_per_min'] = blink_metrics['blink_rate_per_min']
        session_state['passive_liveness_stats']['blink_detected_recently'] = blink_metrics['blink_detected_recently']
        session_state['passive_liveness_stats']['motion_consistency_score'] = motion_metrics['motion_consistency_score']
        session_state['passive_liveness_stats']['motion_label'] = motion_metrics['motion_label']
        if anti_spoof_metrics.get('available', False):
            if anti_spoof_metrics.get('label') == 'live':
                session_state['passive_liveness_stats']['anti_spoof_model_live'] += 1
            elif anti_spoof_metrics.get('label') == 'spoof':
                session_state['passive_liveness_stats']['anti_spoof_model_spoof'] += 1
        else:
            session_state['passive_liveness_stats']['anti_spoof_model_missing'] += 1
        motion_stat_key = f"motion_{motion_metrics['motion_label']}"
        if motion_stat_key in session_state['passive_liveness_stats']:
            session_state['passive_liveness_stats'][motion_stat_key] += 1
        evaluated_frames = session_state['passive_liveness_stats']['frames_evaluated']
        running_avg = session_state['passive_liveness_stats']['avg_score']
        session_state['passive_liveness_stats']['avg_score'] = (
            ((running_avg * (evaluated_frames - 1)) + passive_liveness['score']) / evaluated_frames
        )
        
        # Baseline calibration
        if not session_state['baseline_calibrated']:
            session_state['calibration_yaws'].append(yaw)
            session_state['calibration_pitches'].append(pitch)
            session_state['calibration_rolls'].append(roll)
            
            if len(session_state['calibration_yaws']) >= session_state['CALIBRATION_FRAMES']:
                session_state['baseline_yaw'] = np.mean(session_state['calibration_yaws'])
                session_state['baseline_pitch'] = np.mean(session_state['calibration_pitches'])
                session_state['baseline_roll'] = np.mean(session_state['calibration_rolls'])
                session_state['baseline_calibrated'] = True
                
                emit('calibration_complete', {
                    'baseline': {
                        'yaw': float(session_state['baseline_yaw']),
                        'pitch': float(session_state['baseline_pitch']),
                        'roll': float(session_state['baseline_roll'])
                    }
                })
            
            pose_status = "Calibrating"
            relative_yaw = 0
            relative_pitch = 0
            relative_roll = 0
        else:
            relative_yaw = yaw - session_state['baseline_yaw']
            relative_pitch = pitch - session_state['baseline_pitch']
            relative_roll = roll - session_state['baseline_roll']

            deviating = abs(relative_yaw) > 15 or abs(relative_pitch) > 10 or abs(relative_roll) > 10
            pose_status = "Deviating" if deviating else "Normal"

            if deviating:
                session_state['deviation_frames'] += 1
        
        # Eye gaze
        left_gaze, right_gaze, gaze_direction = get_eye_gaze(frame, landmarks)
        gaze_suspicious = gaze_direction != "Looking Center"
        
        if gaze_suspicious and session_state['baseline_calibrated']:
            session_state['gaze_deviation_frames'] += 1
        
        # YOLO detection
        person_count, detected_objects = detect_persons_and_objects(frame)
        
        if person_count > 1 and session_state['baseline_calibrated']:
            session_state['multiple_person_frames'] += 1
        
        if detected_objects and session_state['baseline_calibrated']:
            session_state['prohibited_object_frames'] += 1
            for obj, count in detected_objects.items():
                if obj not in session_state['prohibited_objects_detected']:
                    session_state['prohibited_objects_detected'][obj] = 0
                session_state['prohibited_objects_detected'][obj] += count
        
        # Send result
        result = {
            'success': True,
            'frame_count': session_state['frame_count'],
            'identity': identity,
            'distance': float(distance),
            'pose': {
                'status': pose_status,
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': float(roll),
                'relative_yaw': float(relative_yaw),
                'relative_pitch': float(relative_pitch),
                'relative_roll': float(relative_roll)
            },
            'gaze': {
                'direction': gaze_direction,
                'suspicious': gaze_suspicious,
                'left_ratio': float(left_gaze),
                'right_ratio': float(right_gaze)
            },
            'detection': {
                'person_count': person_count,
                'objects': detected_objects
            },
            'liveness': {
                'passive_score': float(passive_liveness['score']),
                'passive_label': passive_liveness['label'],
                'quality_insufficient': bool(passive_liveness['quality_insufficient']),
                'quality_reason': passive_liveness['quality_reason'],
                'quality_guidance': passive_liveness['quality_guidance'],
                'motion_consistency_score': float(passive_liveness['motion_consistency_score']),
                'motion_label': passive_liveness['motion_label'],
                'motion_std': float(passive_liveness['motion_std']),
                'motion_delta': float(passive_liveness['motion_delta']),
                'motion_center_jitter': float(passive_liveness['motion_center_jitter']),
                'motion_pose_jitter': float(passive_liveness['motion_pose_jitter']),
                'anti_spoof_live_score': float(passive_liveness['anti_spoof_live_score']),
                'anti_spoof_spoof_score': float(passive_liveness['anti_spoof_spoof_score']),
                'anti_spoof_label': passive_liveness['anti_spoof_label'],
                'anti_spoof_mode': passive_liveness['anti_spoof_mode'],
                'mode': passive_liveness['mode'],
                'signals': {
                    'brightness': float(passive_liveness['signals']['brightness']),
                    'face_size_ratio': float(passive_liveness['signals']['face_size_ratio']),
                    'texture_variance': float(passive_liveness['signals']['texture_variance']),
                    'blur_variance': float(passive_liveness['signals']['blur_variance']),
                    'quality_ok': bool(passive_liveness['signals']['quality_ok']),
                    'quality_reason': passive_liveness['signals']['quality_reason'],
                    'quality_guidance': passive_liveness['signals']['quality_guidance'],
                    'blink_rate_per_min': float(passive_liveness['signals']['blink_rate_per_min']),
                    'blink_detected_recently': bool(passive_liveness['signals']['blink_detected_recently']),
                    'blink_score': float(passive_liveness['signals']['blink_score']),
                    'motion_consistency_score': float(passive_liveness['signals']['motion_consistency_score']),
                    'motion_label': passive_liveness['signals']['motion_label'],
                    'motion_std': float(passive_liveness['signals']['motion_std']),
                    'motion_delta': float(passive_liveness['signals']['motion_delta']),
                    'motion_center_jitter': float(passive_liveness['signals']['motion_center_jitter']),
                    'motion_pose_jitter': float(passive_liveness['signals']['motion_pose_jitter']),
                    'anti_spoof_live_score': float(passive_liveness['signals']['anti_spoof_live_score']),
                    'anti_spoof_spoof_score': float(passive_liveness['signals']['anti_spoof_spoof_score']),
                    'anti_spoof_label': passive_liveness['signals']['anti_spoof_label'],
                    'anti_spoof_mode': passive_liveness['signals']['anti_spoof_mode'],
                    'avg_ear': float(blink_metrics['avg_ear']),
                    'avg_ear_window': float(blink_metrics['avg_ear_window']),
                    'left_ear': float(blink_metrics['left_ear']),
                    'right_ear': float(blink_metrics['right_ear']),
                    'blink_count': int(blink_metrics['blink_count']),
                    'blink_detected': bool(blink_metrics['blink_detected']),
                    'fps_estimate': float(blink_metrics['fps_estimate'])
                }
            },
            'calibrated': session_state['baseline_calibrated'],
            'calibration_progress': len(session_state['calibration_yaws']) if not session_state['baseline_calibrated'] else session_state['CALIBRATION_FRAMES'],
            'tuning': {
                'profile': ACTIVE_LIVENESS_PROFILE,
                'debug_logging_enabled': LIVENESS_DEBUG_LOG_ENABLED
            }
        }

        write_liveness_debug_entry({
            'timestamp': time.time(),
            'frame_count': session_state['frame_count'],
            'profile': ACTIVE_LIVENESS_PROFILE,
            'identity': identity,
            'distance': float(distance),
            'pose_status': pose_status,
            'gaze_direction': gaze_direction,
            'passive_label': passive_liveness['label'],
            'passive_score': float(passive_liveness['score']),
            'quality_insufficient': bool(passive_liveness['quality_insufficient']),
            'quality_reason': passive_liveness['quality_reason'],
            'quality_guidance': passive_liveness['quality_guidance'],
            'motion_label': passive_liveness['motion_label'],
            'motion_consistency_score': float(passive_liveness['motion_consistency_score']),
            'motion_delta': float(passive_liveness['motion_delta']),
            'blink_rate_per_min': float(passive_liveness['signals']['blink_rate_per_min']),
            'blink_detected_recently': bool(passive_liveness['signals']['blink_detected_recently']),
            'anti_spoof_mode': passive_liveness['anti_spoof_mode'],
            'anti_spoof_label': passive_liveness['anti_spoof_label'],
            'anti_spoof_live_score': float(passive_liveness['anti_spoof_live_score']),
            'anti_spoof_spoof_score': float(passive_liveness['anti_spoof_spoof_score']),
            'thresholds': {
                'quality': ACTIVE_LIVENESS_CONFIG['quality'],
                'blink': ACTIVE_LIVENESS_CONFIG['blink'],
                'motion': ACTIVE_LIVENESS_CONFIG['motion'],
                'weights': ACTIVE_LIVENESS_CONFIG['weights']
            }
        })
        
        emit('frame_result', result)
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        emit('frame_result', {'error': str(e)})
    finally:
        FRAME_PROCESS_LOCK.release()

# ================= MAIN =================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Online Proctoring Server Ready!")
    print("="*50)
    print("Server running on http://localhost:5000")
    print("WebSocket ready for real-time proctoring")
    print("="*50 + "\n")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
