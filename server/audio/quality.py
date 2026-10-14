"""Recording quality: level, noise floor, SNR, clipping (flow.txt §B.3 step 5)."""

import numpy as np

from audio.io import SR, rms_dbfs


def voiced_mask(n, segs, sr=SR):
    m = np.zeros(n, bool)
    for s, e in segs:
        m[int(s * sr):int(e * sr)] = True
    return m


def measure(x, segs, sr=SR):
    x = np.asarray(x, np.float32)
    m = voiced_mask(len(x), segs, sr)
    level = rms_dbfs(x[m]) if m.any() else -240.0
    if np.sum(~m) >= sr:
        noise = rms_dbfs(x[~m])
    else:
        frames = x[: len(x) // 512 * 512].reshape(-1, 512)
        noise = float(np.percentile(20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12), 10)) \
            if len(frames) else -240.0
    return {"level_dbfs": round(level, 1), "noise_dbfs": round(noise, 1), "snr_db": round(level - noise, 1),
            "clip_frac": round(float(np.mean(np.abs(x) >= 0.99)), 5)}


def judge(q, ec):
    """Returns (ok, problems[list of user-facing messages], warnings[list])."""
    problems, warnings = [], []
    if q["clip_frac"] > ec["max_clip_frac"]:
        problems.append("Too loud: the recording is clipping. Move back from the microphone or lower the input volume.")
    if q["level_dbfs"] < ec["min_level_dbfs"]:
        problems.append("Too quiet: speak up or move closer to the microphone.")
    elif q["level_dbfs"] < ec["warn_level_dbfs"]:
        warnings.append("Voice level is a bit low.")
    if q["snr_db"] < ec["min_snr_db"]:
        problems.append("Too noisy: background noise is close to your voice level. Move somewhere quieter.")
    elif q["snr_db"] < ec["warn_snr_db"]:
        warnings.append("Some background noise detected.")
    return not problems, problems, warnings
