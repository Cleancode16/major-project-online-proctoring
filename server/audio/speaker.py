"""Speaker embeddings (Resemblyzer d-vectors) and voiceprints (flow.txt §B.3 step 7, §B.5)."""

from functools import lru_cache

import numpy as np

from audio.io import SR


@lru_cache(maxsize=None)
def encoder():
    from resemblyzer import VoiceEncoder
    return VoiceEncoder("cpu", verbose=False)


def _prep(x):
    """Peak-normalise to a fixed level (Resemblyzer expects roughly normalised float audio)."""
    x = np.asarray(x, np.float32)
    peak = np.abs(x).max()
    return x / peak * 0.5 if peak > 1e-6 else x


def embed(x):
    """256-D unit-norm d-vector of one stretch of (voiced) audio, >= ~0.5 s."""
    e = encoder().embed_utterance(_prep(x))
    return e / (np.linalg.norm(e) + 1e-12)


def voiced_audio(x, segs, sr=SR):
    parts = [x[int(s * sr):int(e * sr)] for s, e in segs]
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def chunk(x, seconds, sr=SR, min_frac=0.6):
    n = int(seconds * sr)
    out = [x[i:i + n] for i in range(0, len(x), n)]
    return [c for c in out if len(c) >= min_frac * n]


def voiceprint(x, segs, chunk_s=3.0):
    """Returns (voiceprint, consistency, n_chunks). voiceprint = normalised mean of per-chunk d-vectors."""
    chunks = chunk(voiced_audio(x, segs), chunk_s)
    if not chunks:
        return None, 0.0, 0
    vecs = np.stack([embed(c) for c in chunks])
    vp = vecs.mean(axis=0)
    vp /= np.linalg.norm(vp) + 1e-12
    if len(vecs) > 1:
        sims = vecs @ vecs.T
        consistency = float(sims[np.triu_indices(len(vecs), 1)].mean())
    else:
        consistency = 1.0
    return vp, consistency, len(vecs)


def similarity(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))
