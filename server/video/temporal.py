"""Temporal smoothing / persistence helpers shared by the video and audio monitors."""

import math
from collections import deque

import numpy as np


def samples_for(seconds, rate_hz):
    """Number of consecutive samples covering `seconds` at `rate_hz` (at least 1)."""
    return max(1, int(math.ceil(seconds * rate_hz - 1e-9)))


class RollingMedian:
    """Median of the last `n` values (n=1 passes values through)."""

    def __init__(self, n):
        self.buf = deque(maxlen=max(1, int(n)))

    def __call__(self, x):
        self.buf.append(x)
        return float(np.median(self.buf))

    def resize(self, n):
        """FPS10: change the window (live sample rate re-estimated), keeping the newest values."""
        n = max(1, int(n))
        if n != self.buf.maxlen:
            self.buf = deque(self.buf, maxlen=n)


def centered_median(values, window):
    """Zero-lag (centred) median over a sequence that may contain None/NaN gaps (offline use)."""
    x = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    if window <= 1 or len(x) == 0:
        return x
    half = int(window) // 2
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        if np.isnan(x[i]):
            continue
        out[i] = np.nanmedian(x[max(0, i - half):i + half + 1])
    return out


def centered_mean(values, window):
    """Zero-lag (centred) moving average over a sequence with None/NaN gaps (offline use, GZ3)."""
    x = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    if window <= 1 or len(x) == 0:
        return x
    half = int(window) // 2
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        if np.isnan(x[i]):
            continue
        out[i] = np.nanmean(x[max(0, i - half):i + half + 1])
    return out


class MajorityVote:
    """True when more than half of the last `n` booleans are True."""

    def __init__(self, n):
        self.buf = deque(maxlen=max(1, int(n)))

    def __call__(self, x):
        self.buf.append(bool(x))
        return sum(self.buf) * 2 > len(self.buf)

    def resize(self, n):
        """FPS10: change the window (live sample rate re-estimated), keeping the newest values."""
        n = max(1, int(n))
        if n != self.buf.maxlen:
            self.buf = deque(self.buf, maxlen=n)


class LivePersistence:
    """Causal debounce for live status: True once the last `k` raw values were all True."""

    def __init__(self, k):
        self.buf = deque(maxlen=max(1, int(k)))

    def __call__(self, x):
        self.buf.append(bool(x))
        return len(self.buf) == self.buf.maxlen and all(self.buf)

    def resize(self, n):
        """FPS10: change the window (live sample rate re-estimated), keeping the newest values."""
        n = max(1, int(n))
        if n != self.buf.maxlen:
            self.buf = deque(self.buf, maxlen=n)


def filter_runs(flags, min_len=1, max_gap=0):
    """
    Offline persistence filter over a whole boolean sequence (used for final counts).
      1. gaps of <= max_gap False samples between True samples are filled
         (so "k of n" style flicker does not split one violation in two);
      2. runs shorter than min_len samples are dropped.
    Gap filling only bridges between True samples, it never extends a run's ends.
    """
    x = [bool(f) for f in flags]
    n = len(x)
    if max_gap > 0:
        last_true = None
        for i, v in enumerate(x):
            if v:
                if last_true is not None and 0 < i - last_true - 1 <= max_gap:
                    for j in range(last_true + 1, i):
                        x[j] = True
                last_true = i
    if min_len > 1:
        i = 0
        while i < n:
            if x[i]:
                j = i
                while j < n and x[j]:
                    j += 1
                if j - i < min_len:
                    for t in range(i, j):
                        x[t] = False
                i = j
            else:
                i += 1
    return x
