"""
PERF1: loading the audio models must not change PyTorch's process-wide thread count.

silero_vad/model.py calls torch.set_num_threads(1) at import; before PERF1 that made YOLO and L2CS run on one
thread for the whole live session (video ~2 s per sample instead of ~0.3 s, so persons / objects / gaze were
sampled every 2-3 s and short violations were missed).

Run (from server/):  ..\\.venv\\Scripts\\python.exe -m tests.test_perf1_threads
"""

import sys

import numpy as np
import torch

import config as C
from audio.monitor import AudioAnalyzer


def main():
    before = torch.get_num_threads()
    aa = AudioAnalyzer(C.get_config("v2"), np.zeros(256, np.float32))
    aa.feed(np.zeros(16000, np.float32))
    after = torch.get_num_threads()
    ok = after == before
    print(f"  [{'ok' if ok else 'FAIL'}] torch threads before {before}, after audio models {after}")
    print("PERF1 THREADS:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
