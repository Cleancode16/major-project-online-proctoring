"""YAMNet sound-event scores -> proctoring categories (flow.txt §B.4)."""

import csv
from functools import lru_cache

import numpy as np

import config  # noqa: F401  (sets TFHUB_CACHE_DIR before tensorflow_hub is imported)

WIN = 15360   # 0.96 s at 16 kHz
HOP = 7680    # 0.48 s
CTX = 15600   # samples one batch YAMNet frame really covers: 95 * 160 hop + 400 STFT window

CATEGORIES = {
    "speech": ["Speech", "Child speech, kid speaking", "Narration, monologue", "Conversation", "Babbling",
               "Speech synthesizer", "Hubbub, speech noise, speech babble"],
    "whisper": ["Whispering"],
    "phone": ["Telephone", "Telephone bell ringing", "Ringtone", "Telephone dialing, DTMF",
              "Dial tone", "Busy signal"],
    "music": ["Music", "Musical instrument", "Singing", "Song"],
    "tv_radio": ["Television", "Radio"],
    "typing": ["Typing", "Computer keyboard", "Typewriter"],
}


@lru_cache(maxsize=None)
def model():
    import tensorflow_hub as hub
    return hub.load("https://tfhub.dev/google/yamnet/1")


@lru_cache(maxsize=None)
def class_names():
    with open(model().class_map_path().numpy().decode(), newline="") as f:
        return [row["display_name"] for row in csv.DictReader(f)]


@lru_cache(maxsize=None)
def category_index():
    names = class_names()
    idx = {}
    for cat, labels in CATEGORIES.items():
        missing = [l for l in labels if l not in names]
        if missing:
            raise ValueError(f"YAMNet class map has no {missing}")
        idx[cat] = [names.index(l) for l in labels]
    return idx


def scores(x):
    """Per-frame (0.96 s / 0.48 s hop) YAMNet scores for a waveform. Returns (n_frames, 521)."""
    s, _, _ = model()(np.asarray(x, np.float32))
    return s.numpy()


def window_scores(x):
    """Scores of one window; pass CTX samples so the result equals the batch frame exactly."""
    return scores(x[:CTX])[0]


def categorize(frame_scores):
    """{category: max score over its classes} + top-3 AudioSet labels."""
    idx = category_index()
    cat = {c: float(frame_scores[i].max()) for c, i in idx.items()}
    names = class_names()
    top = np.argsort(frame_scores)[::-1][:3]
    return cat, [(names[i], round(float(frame_scores[i]), 3)) for i in top]
