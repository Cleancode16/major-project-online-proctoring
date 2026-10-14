"""
A1 audio I/O + A2 VAD.
  A1: decode the real sample clip's audio track (duration/sr/level); 1 s microphone smoke test
      (audio is NOT saved); wav round trip.
  A2: VAD on the SYNTHETIC scenes, scored per 10 ms frame against the label segments with a 0.3 s
      tolerance band at every label boundary (TTS files contain short leading/trailing silences).
      Reports speech recall / precision for speech scenes and false speech seconds for non-speech scenes,
      for Silero and webrtcvad.

Run:  .venv\\Scripts\\python.exe -m tests.test_a1_a2_audio_vad
"""

import json
import sys

import numpy as np

import config as C
from audio import io as aio
from audio import vad as V
from tests import paths

AUD = paths.DATA / "audio"
STEP = 0.01
TOL = 0.3


def a1():
    x = aio.load_from_video(paths.SAMPLE_VIDEO)
    dur = len(x) / aio.SR
    rms_db = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
    print(f"A1 video audio: {len(x)} samples = {dur:.2f} s @16 kHz, RMS {rms_db:.1f} dBFS, peak {np.abs(x).max():.3f}"
          f" -> source status: {aio.source_status(x)}")
    # The real clip's audio track decodes but is digital silence (recorded with the mic muted).
    ok = abs(dur - 12.67) < 0.25 and aio.source_status(x) == "digital_silence"
    try:
        m = aio.MicStream().start()
        import time
        time.sleep(1.0)
        m.stop()
        mic = m.read_all()
        print(f"A1 mic smoke test: {len(mic) / aio.SR:.2f} s captured, RMS {20 * np.log10(np.sqrt(np.mean(mic ** 2)) + 1e-12):.1f} dBFS "
              f"-> source status: {aio.source_status(mic)} (not saved)")
        ok &= len(mic) > 0.8 * aio.SR
    except Exception as exc:
        print(f"A1 mic smoke test: FAILED to open microphone: {exc}")
        ok = False
    tmp = paths.RESULTS / "_roundtrip.wav"
    aio.save_wav(tmp, x)
    y = aio.load_file(tmp)
    tmp.unlink()
    print(f"A1 wav round trip max abs diff {np.abs(x[:len(y)] - y).max():.5f} (16-bit quantisation)")
    return ok and np.abs(x[:len(y)] - y).max() < 1e-3, x


def frame_truth(dur, segs):
    t = np.arange(0, dur, STEP)
    truth = np.zeros(len(t), bool)
    care = np.ones(len(t), bool)
    for s, e in segs:
        truth |= (t >= s) & (t < e)
        care &= ~((np.abs(t - s) < TOL) | (np.abs(t - e) < TOL))
    return t, truth, care


def a2():
    scenes = json.loads((AUD / "scenes.json").read_text())
    res = {}
    for engine in ("silero", "webrtc"):
        vcfg = dict(C.get_config("v2")["vad"], engine=engine)
        tp = fp = fn = 0
        utt_hit = utt_n = 0
        false_speech = {}
        for name, sc in scenes.items():
            x = aio.load_file(AUD / f"{name}.wav")
            v = V.VAD(vcfg)
            v.feed(x)
            segs = v.segments()
            t, truth, care = frame_truth(sc["duration_s"], sc["labels"].get("SPEECH", []))
            pred = np.zeros(len(t), bool)
            for s, e in segs:
                pred |= (t >= s) & (t < e)
            tp += int(np.sum(pred & truth & care)); fp += int(np.sum(pred & ~truth & care))
            fn += int(np.sum(~pred & truth & care))
            for s, e in sc["labels"].get("SPEECH", []):
                utt_n += 1
                utt_hit += V.overlap(segs, s, e) >= 0.5 * (e - s)
            outside = pred & ~truth & care
            false_speech[name] = round(float(np.sum(outside) * STEP), 2)
        prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
        res[engine] = (utt_hit / max(1, utt_n), false_speech)
        print(f"A2 [{engine}] utterances detected {utt_hit}/{utt_n}; false speech seconds outside labels "
              f"(0.3 s tolerance) per scene: {false_speech}; informational frame-level precision {prec:.3f} "
              f"recall {rec:.3f}")
    utt_recall, fs = res["silero"]
    non_speech = ("silence", "music", "phone_ring", "typing")
    return utt_recall >= 0.95 and all(fs[n] <= 0.5 for n in non_speech)


def a2_real(x):
    v = V.VAD(C.get_config("v2")["vad"])
    v.feed(x)
    segs = v.segments()
    print(f"A2 real clip audio (no labels, informational): speech segments {[(round(a, 2), round(b, 2)) for a, b in segs]}")


def main():
    ok1, x = a1()
    ok2 = a2()
    a2_real(x)
    print("A1:", "PASS" if ok1 else "FAIL", "| A2:", "PASS" if ok2 else "FAIL")
    return 0 if ok1 and ok2 else 1


if __name__ == "__main__":
    sys.exit(main())
