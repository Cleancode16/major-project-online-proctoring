"""
Orchestration shared by the CLI (app.py) and the UI: reference -> video + audio analysis -> report.
Everything runs locally; nothing is uploaded.
"""

from pathlib import Path

import cv2
import numpy as np

import config as C
from aggregation.report import build_report, console_summary, save_report
from audio import io as aio
from audio import monitor
from video import face, pipeline


def reference_embedding(ref, cfg):
    """`ref` may be an embedding (np.ndarray of size 512), an image (BGR array) or an image path."""
    if isinstance(ref, np.ndarray) and ref.ndim == 1:
        return ref
    img = cv2.imread(str(ref)) if isinstance(ref, (str, Path)) else ref
    if img is None:
        raise ValueError(f"Cannot read reference image: {ref}")
    emb, _, n = face.reference_from_image(img, cfg["face"])
    return emb


def load_audio_source(audio_source, video_path):
    """audio_source: "video" (track of the video), "none", or a path to an audio file."""
    if audio_source in (None, "none"):
        return None
    if audio_source == "video":
        return aio.load_from_video(video_path)
    return aio.load_file(audio_source)


def analyze_files(video_path, ref, cfg, voiceprint=None, audio_source="video", candidate="candidate",
                  progress=None, out_dir=None, save=True, session_id=None):
    emb = reference_embedding(ref, cfg)
    video = pipeline.analyze_video(video_path, emb, cfg, progress=progress)
    audio = None
    if audio_source not in (None, "none") and cfg["audio"]["enabled"]:
        audio = monitor.analyze_audio(load_audio_source(audio_source, video_path), cfg, voiceprint)
    meta = {"candidate": candidate, "video_path": str(video_path), "audio_source": str(audio_source),
            "reference": "embedding" if isinstance(ref, np.ndarray) and ref.ndim == 1 else str(ref)[:200],
            "voiceprint": voiceprint is not None}
    if session_id:
        meta["session_id"] = session_id
    report = build_report(video, audio, cfg, meta)
    path = save_report(report, video, out_dir, audio=audio) if save else None
    return report, path, video, audio


__all__ = ["analyze_files", "console_summary", "reference_embedding", "C"]
