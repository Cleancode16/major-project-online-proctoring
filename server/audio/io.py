"""Audio sources -> 16 kHz mono float32 (flow.txt §B.1)."""

import subprocess
import threading

import numpy as np

SR = 16000


def rms_dbfs(x):
    x = np.asarray(x, np.float64)
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)) if len(x) else -240.0


def source_status(x, silent_rms_dbfs=-80.0):
    """"unavailable" | "digital_silence" | "ok" (flow.txt §B.1)."""
    if x is None or len(x) == 0:
        return "unavailable"
    return "digital_silence" if rms_dbfs(x) < silent_rms_dbfs else "ok"


def load_from_video(path, sr=SR):
    """Decode a video's audio track with the bundled ffmpeg. Returns None if the file has no audio."""
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(sr), "-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode(errors="replace")
        if "does not contain any stream" in err or "matches no streams" in err or not proc.stdout:
            return None
        raise IOError(f"ffmpeg failed on {path}: {err[-300:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def load_file(path, sr=SR):
    """Read an audio file, downmix to mono and resample to `sr`."""
    import soundfile as sf
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        import resampy
        x = resampy.resample(x, file_sr, sr).astype(np.float32)
    return x


def save_wav(path, x, sr=SR):
    import soundfile as sf
    sf.write(str(path), np.asarray(x, np.float32), sr, subtype="PCM_16")


def record(seconds, sr=SR, device=None):
    """Blocking microphone recording (enrollment). Raises RuntimeError with a readable message if no mic."""
    import sounddevice as sd
    try:
        x = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", device=device)
        sd.wait()
    except Exception as exc:  # PortAudio errors when no input device / permission denied
        raise RuntimeError(f"Microphone not available: {exc}") from exc
    return x[:, 0].copy()


def has_microphone():
    try:
        import sounddevice as sd
        sd.query_devices(kind="input")
        return True
    except Exception:
        return False


class MicStream:
    """Non-blocking microphone capture into a growing buffer (live test). t = samples / sr."""

    def __init__(self, sr=SR, device=None, block=1600):
        import sounddevice as sd
        self.sr = sr
        self._chunks = []
        self._lock = threading.Lock()
        self.level = 0.0
        self._stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32", device=device,
                                      blocksize=block, callback=self._cb)

    def _cb(self, indata, frames, time_info, status):
        x = indata[:, 0].copy()
        with self._lock:
            self._chunks.append(x)
        self.level = float(np.sqrt(np.mean(x ** 2)) + 1e-12)

    def start(self):
        self._stream.start()
        return self

    def stop(self):
        self._stream.stop()
        self._stream.close()

    def read_all(self):
        """All audio captured so far (concatenated)."""
        with self._lock:
            return np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)

    def pop(self):
        """Audio captured since the last pop()."""
        with self._lock:
            chunks, self._chunks = self._chunks, []
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
