"""Lazy, cached model loaders so the CLI, the UI and the tests share one copy."""

import os
from functools import lru_cache

import numpy as np

import config  # noqa: F401  (sets TF env vars before TensorFlow is imported)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@lru_cache(maxsize=None)
def facenet():
    from keras_facenet import FaceNet
    return FaceNet()


@lru_cache(maxsize=None)
def _facenet_fn():
    import tensorflow as tf
    keras_model = facenet().model
    # Compiled graph: measured 37.5 ms per face vs 236 ms for an eager model(x) call and
    # ~93 ms for model.predict() (which also prints a progress bar every call).
    return tf.function(lambda x: keras_model(x, training=False),
                       input_signature=[tf.TensorSpec([None, 160, 160, 3], tf.float32)])


def embed_faces(crops_rgb):
    """Embed a list of 160x160 RGB face crops. Returns (n, 512), L2-normalised by the model."""
    if not crops_rgb:
        return np.zeros((0, 512), np.float32)
    model = facenet()
    x = np.float32([model._normalize(c) for c in crops_rgb])
    return _facenet_fn()(x).numpy()


def face_detector(confidence=0.6):
    import mediapipe as mp
    return mp.solutions.face_detection.FaceDetection(model_selection=1,
                                                     min_detection_confidence=confidence)


def face_mesh(max_faces=1, static=False):
    # Not cached: FaceMesh tracks between calls, so each analyzer owns one.
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=static,
                                           max_num_faces=max_faces,
                                           refine_landmarks=True,
                                           min_detection_confidence=0.6,
                                           min_tracking_confidence=0.6)


@lru_cache(maxsize=None)
def yolo(weights="yolov8n.pt"):
    from ultralytics import YOLO
    path = config.ROOT / weights
    return YOLO(str(path) if path.exists() else weights)
