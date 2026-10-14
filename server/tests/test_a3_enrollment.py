"""
A3 voice enrollment on SYNTHETIC recordings (25 s window, passage read by TTS voice "David" from t=2 s,
low pink room noise). One positive and several negative cases, each must give the expected outcome
and the expected user-facing message.

Run:  .venv\\Scripts\\python.exe -m tests.test_a3_enrollment
"""

import sys

import numpy as np

import config as C
from audio import enrollment
from audio import io as aio
from audio.passages import PASSAGES
from tests import paths

AUD = paths.DATA / "audio"
SR = aio.SR
rng = np.random.default_rng(3)


def room(seconds, db=-55):
    x = rng.standard_normal(int(seconds * SR)).astype(np.float32)
    return x / np.sqrt(np.mean(x ** 2)) * 10 ** (db / 20)


def window(speech, at=2.0, total=25.0, noise_db=-55):
    x = room(total, noise_db)
    i = int(at * SR)
    n = min(len(speech), len(x) - i)
    x[i:i + n] += speech[:n]
    return x


def to_dbfs(x, db):
    return x / (np.sqrt(np.mean(x[np.abs(x) > 1e-3] ** 2)) + 1e-12) * 10 ** (db / 20)


def cases():
    d0 = aio.load_file(AUD / "tts" / "david_p0.wav")
    d1 = aio.load_file(AUD / "tts" / "david_p1.wav")
    z0 = aio.load_file(AUD / "tts" / "zira_p0.wav")
    d0n = to_dbfs(d0, -24)
    half = len(d0n) // 2
    mixed = np.concatenate([d0n[:half], to_dbfs(z0, -24)[half:len(z0)]])
    return [
        # name, audio, passage index, cfg overrides, expected ok, expected message keyword
        ("good_reading", window(d0n), 0, {}, True, None),
        ("silence_only", room(25.0), 0, {}, False, "speech was detected"),
        ("muted_mic", np.zeros(int(25 * SR), np.float32), 0, {}, False, "muted or blocked"),
        ("too_quiet", window(to_dbfs(d0, -50), noise_db=-75), 0, {}, False, "Too quiet"),
        ("clipped", window(np.clip(to_dbfs(d0, -4) * 3, -1, 1)), 0, {}, False, "Too loud"),
        ("noisy", window(d0n, noise_db=-27), 0, {}, False, "Too noisy"),
        ("two_voices", window(mixed), 0, {}, False, "not consistent"),
        ("wrong_passage_asr", window(to_dbfs(d1, -24)), 0, {"asr_enabled": True}, False, "does not match"),
        ("right_passage_asr", window(d0n), 0, {"asr_enabled": True}, True, None),
    ]


def main():
    results = []
    for name, x, p_idx, ov, exp_ok, keyword in cases():
        cfg = C.get_config("v2", {"enroll": ov} if ov else None)
        r = enrollment.evaluate_recording(x, PASSAGES[p_idx], cfg)
        msg = " | ".join(r["problems"]) or "-"
        ok = r["ok"] == exp_ok and (keyword is None or keyword in msg)
        if name.endswith("_asr") and r["asr"] and r["asr"]["status"] == "skipped":
            ok = None  # Vosk not installed: cannot test, report as skipped
        results.append(ok)
        m = r["metrics"]
        print(f"[{'PASS' if ok else ('SKIP' if ok is None else 'FAIL')}] {name:18s} ok={r['ok']!s:5s} "
              f"voiced={m.get('voiced_s', '-')} snr={m.get('snr_db', '-')} level={m.get('level_dbfs', '-')} "
              f"clip={m.get('clip_frac', '-')} consistency={m.get('consistency', '-')} "
              f"asr={r['asr'] and r['asr'].get('match')} -> {msg[:110]}")
    passed = all(v is not False for v in results)
    print("A3 ENROLLMENT:", "PASS" if passed else "FAIL", f"({sum(v is True for v in results)} pass, "
          f"{sum(v is None for v in results)} skipped)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
