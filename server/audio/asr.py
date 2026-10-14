"""Optional offline ASR check that the passage was actually read (flow.txt §B.3 step 6)."""

import json
import re
from pathlib import Path

import numpy as np

from audio.io import SR


def _words(text):
    return re.findall(r"[a-z']+", text.lower())


def word_match(passage, transcript):
    """Fraction of distinct passage words that appear in the transcript (order-insensitive)."""
    p = set(_words(passage))
    t = set(_words(transcript))
    return len(p & t) / max(1, len(p))


def transcribe(x, model_dir):
    """Returns transcript text, or None if Vosk or its model is not available."""
    try:
        import vosk
    except ImportError:
        return None
    if not Path(model_dir).exists():
        return None
    vosk.SetLogLevel(-1)
    rec = vosk.KaldiRecognizer(vosk.Model(str(model_dir)), SR)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    for i in range(0, len(pcm), 8000):
        rec.AcceptWaveform(pcm[i:i + 8000])
    return json.loads(rec.FinalResult()).get("text", "")


def check(x, passage, ec):
    """{"status": "skipped"|"pass"|"fail", "match": float|None, "transcript": str|None}"""
    if not ec["asr_enabled"]:
        return {"status": "skipped", "reason": "disabled in config", "match": None, "transcript": None}
    text = transcribe(x, ec["asr_model_dir"])
    if text is None:
        return {"status": "skipped", "reason": "vosk or model not installed", "match": None, "transcript": None}
    m = word_match(passage, text)
    return {"status": "pass" if m >= ec["asr_min_match"] else "fail", "match": round(m, 3), "transcript": text}
