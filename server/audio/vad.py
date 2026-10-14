"""Voice activity detection: Silero (primary) / webrtcvad (fallback) + shared hysteresis (flow.txt §B.2)."""

import numpy as np

SR = 16000


class VAD:
    """Incremental VAD. feed() any chunk length; probabilities accumulate per frame; segments() any time."""

    def __init__(self, vcfg):
        self.cfg = vcfg
        self.engine = None
        self.probs = []
        self._pending = np.zeros(0, np.float32)
        # VAD2: streaming 2nd-order Butterworth high-pass (state carried across chunks -> live == offline)
        self._sos, self._zi = None, None
        if vcfg.get("highpass_hz"):
            from scipy.signal import butter, sosfilt_zi
            self._sos = butter(2, vcfg["highpass_hz"], btype="highpass", fs=SR, output="sos")
            self._zi = sosfilt_zi(self._sos) * 0.0
        if vcfg["engine"] == "silero":
            try:
                import torch
                from silero_vad import load_silero_vad
                torch.set_num_threads(1)
                self._torch = torch
                self._model = load_silero_vad()
                self._model.reset_states()
                self.frame = 512
                self.engine = "silero"
            except Exception as exc:  # pragma: no cover - fallback path
                print(f">>> Silero VAD unavailable ({exc}); falling back to webrtcvad")
        if self.engine is None:
            import webrtcvad
            self._webrtc = webrtcvad.Vad(vcfg["webrtc_mode"])
            self.frame = 480  # 30 ms
            self.engine = "webrtc"

    @property
    def frame_s(self):
        return self.frame / SR

    def feed(self, x):
        """Append audio; returns the probabilities of the frames completed by this chunk."""
        x = np.asarray(x, np.float32)
        if self._sos is not None and len(x):
            from scipy.signal import sosfilt
            y, self._zi = sosfilt(self._sos, x, zi=self._zi)
            x = y.astype(np.float32)
        buf = np.concatenate([self._pending, x])
        n = len(buf) // self.frame
        new = []
        for i in range(n):
            fr = buf[i * self.frame:(i + 1) * self.frame]
            if self.engine == "silero":
                p = float(self._model(self._torch.from_numpy(fr), SR).item())
            else:
                pcm = (np.clip(fr, -1, 1) * 32767).astype(np.int16).tobytes()
                p = 1.0 if self._webrtc.is_speech(pcm, SR) else 0.0
            new.append(p)
        self._pending = buf[n * self.frame:]
        self.probs.extend(new)
        return new

    def segments(self):
        probs = self.probs
        k = int(self.cfg.get("prob_median") or 1)
        if k > 1 and len(probs) >= k:        # VAD2: zero-lag median, removes one-frame spikes
            from scipy.ndimage import median_filter
            probs = median_filter(np.asarray(probs, np.float64), size=k, mode="nearest").tolist()
        return hysteresis_segments(probs, self.frame_s, self.cfg)


def hysteresis_segments(probs, frame_s, vcfg):
    """[(start_s, end_s)] from per-frame speech probabilities using start/end thresholds + min durations."""
    segs, start, below = [], None, 0
    min_sil = max(1, int(round(vcfg["min_silence_ms"] / 1000 / frame_s)))
    for i, p in enumerate(probs):
        if start is None:
            if p >= vcfg["start_threshold"]:
                start, below = i, 0
        else:
            if p >= vcfg["end_threshold"]:
                below = 0
            else:
                below += 1
                if below >= min_sil:
                    segs.append((start, i - below + 1))
                    start = None
    if start is not None:
        segs.append((start, len(probs) - below))
    pad = vcfg["speech_pad_ms"] / 1000
    total = len(probs) * frame_s
    out = []
    for a, b in segs:
        s, e = a * frame_s, b * frame_s
        if e - s < vcfg["min_speech_ms"] / 1000:
            continue
        s, e = max(0.0, s - pad), min(total, e + pad)
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def speech_seconds(segs):
    return float(sum(e - s for s, e in segs))


def overlap(segs, t0, t1):
    """Seconds of [t0, t1) covered by segments."""
    return float(sum(max(0.0, min(e, t1) - max(s, t0)) for s, e in segs))
