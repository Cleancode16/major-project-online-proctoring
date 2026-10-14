"""
VAD2 / MP2 / OB2 accuracy upgrades (flow.txt §B.2, §A.8, §A.9b). Written for the user to run.

  (a) VAD, SYNTHETIC scene: room noise + 50 Hz hum + 6 coughs / knocks + 2 real utterances (TTS "David").
      Old config (no high-pass, no smoothing, no validation) vs new: false speech segments and utterances found.
  (b) Multiple persons: static-face (photo) rejection over a track, face-confirmed low-confidence person box,
      k-of-n persistence bridging one missed sample.
  (c) Prohibited objects: ROI geometry + mapping, per-class merge, weak-detection promotion, face-box filter.
  (d) optional --real: v2 on the labelled real clip (phone + 2nd person), prints TP/FP/FN and ms per sample.

Run (from server/):  ..\\.venv\\Scripts\\python.exe -m tests.test_vad2_mp2_ob2 [--real]
"""

import sys

import numpy as np

import config as C
from audio import io as aio
from audio import monitor
from tests import paths
from video import detectors
from video.pipeline import VideoAnalyzer, promote_weak
from video.temporal import filter_runs

SR = 16000
rng = np.random.default_rng(5)
FAILS = []


def check(name, ok, info=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------------------------------- (a) VAD
def _scene():
    dur = 30.0
    t = np.arange(int(dur * SR)) / SR
    x = (rng.standard_normal(len(t)) * 10 ** (-55 / 20) + 10 ** (-35 / 20) * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    events = []
    for k, at in enumerate((2.0, 5.5, 9.0, 21.0, 24.5, 27.5)):          # coughs (odd) and knocks (even)
        n = int((0.25 if k % 2 == 0 else 0.05) * SR)
        burst = rng.standard_normal(n) * np.exp(-np.arange(n) / (n / 4))
        if k % 2 == 0:                                                   # cough: band-limited noise burst
            from scipy.signal import butter, sosfilt
            burst = sosfilt(butter(4, [300, 3000], btype="bandpass", fs=SR, output="sos"), burst)
        burst = burst / (np.abs(burst).max() + 1e-9) * 0.4
        i = int(at * SR)
        x[i:i + n] += burst.astype(np.float32)
        events.append((at, at + n / SR))
    speech = []
    for name, at in (("david_s0", 12.0), ("david_s1", 16.5)):
        s = aio.load_file(paths.DATA / "audio" / "tts" / f"{name}.wav")[: int(4.0 * SR)]
        s = s / (np.abs(s).max() + 1e-9) * 0.5
        i = int(at * SR)
        x[i:i + len(s)] += s
        speech.append((at, at + len(s) / SR))
    return x, events, speech


def a_vad():
    print("(a) VAD: coughs / knocks / hum vs real speech (SYNTHETIC)")
    x, noise_events, speech = _scene()
    old = C.get_config("v2", {"vad": {"highpass_hz": None, "prob_median": 1, "min_seg_snr_db": None,
                                      "validate_below_s": None}})
    new = C.get_config("v2")
    out = {}
    for name, cfg in (("old", old), ("new", new)):
        res = monitor.analyze_audio(x, cfg, None)
        segs = res["segments"]

        def hits(ranges):
            return sum(any(min(e, b) - max(s, a) > 0 for s, e in segs) for a, b in ranges)
        false_segs = sum(1 for s, e in segs if not any(min(e, b) - max(s, a) > 0 for a, b in speech))
        out[name] = (false_segs, hits(speech))
        print(f"    {name}: false speech segments {false_segs}, noise events detected as speech "
              f"{hits(noise_events)}/{len(noise_events)}, utterances found {hits(speech)}/{len(speech)}, "
              f"rejected {res.get('vad_rejected')}")
    check("new config: no fewer utterances found", out["new"][1] >= out["old"][1])
    check("new config: false speech segments not higher", out["new"][0] <= out["old"][0])


# ------------------------------------------------------------------------------------------- (b) persons
def b_persons():
    print("(b) multiple persons")
    cfg = C.get_config("v2")
    va = VideoAnalyzer(np.zeros(512), cfg, 3.0)
    poster_static = []
    for i in range(12):
        subject = {"bbox": (400 + int(rng.integers(-6, 7)), 200 + int(rng.integers(-6, 7)), 160, 160), "score": 0.9}
        poster = {"bbox": (1500, 100, 80, 80), "score": 0.8}           # a face in a photo: never moves
        faces = [subject, poster]
        va._mark_static_faces(faces)
        poster_static.append(poster["static"])
    check("photo face becomes static after 9 samples", poster_static[-1] and not poster_static[0],
          f"(static flags {['%d' % v for v in poster_static]})")
    shape = (1080, 1920, 3)
    pc = cfg["persons"]
    live = {"bbox": (1300, 300, 100, 100), "static": False}
    box_low = {"cls": 0, "conf": 0.42, "xyxy": [1200, 200, 1500, 1000]}
    with_face, _ = detectors.count_persons([box_low], [live], shape, pc)
    without_face, _ = detectors.count_persons([box_low], [], shape, pc)
    check("0.42-confidence person box counts only with a face inside", with_face == 1 and without_face == 0)
    poster_box = {"cls": 0, "conf": 0.6, "xyxy": [1450, 50, 1650, 400]}
    poster = {"bbox": (1500, 100, 80, 80), "static": True}
    n, _ = detectors.count_persons([poster_box], [poster], shape, pc)
    check("person box whose only face is a photo is ignored", n == 0)
    raw = [0, 1, 1, 0, 1, 1, 0, 0, 0]
    out = filter_runs(raw, 3, pc["persist_gap"])
    check("one missed sample does not split / cancel a visit", out[1:6] == [True] * 5, f"{[int(v) for v in out]}")


# ------------------------------------------------------------------------------------------- (c) objects
def c_objects():
    print("(c) prohibited objects")
    oc = C.get_config("v2")["objects"]
    roi = detectors.roi_for_face((800, 200, 200, 200), (1080, 1920, 3), oc)
    check("ROI covers hands / desk below the face", roi == (500, 100, 1300, 1000), f"{roi}")
    a = [{"cls": 67, "conf": 0.30, "xyxy": [600, 600, 700, 800]}]
    b = [{"cls": 67, "conf": 0.55, "xyxy": [605, 602, 702, 805]}, {"cls": 73, "conf": 0.6, "xyxy": [900, 700, 1100, 900]}]
    merged = detectors.merge_boxes(a, b)
    check("merge keeps the higher confidence per object", len(merged) == 2 and max(m["conf"] for m in merged if m["cls"] == 67) == 0.55)
    found, _, weak = detectors.objects_v2([{"cls": 67, "conf": 0.25, "xyxy": [0, 0, 10, 10]}], oc)
    check("0.25 phone = weak, not detected", not found and weak)
    found, _, _ = detectors.objects_v2([{"cls": 67, "conf": 0.9, "xyxy": [820, 230, 880, 300]}], oc, (800, 200, 200, 200))
    check("phone box inside the face box is ignored", not found)
    strong = [0, 0, 1, 1, 0, 0, 0, 0]
    weak = [0, 1, 0, 0, 1, 0, 1, 0]
    check("weak promoted only next to a strong detection",
          promote_weak(strong, weak) == [False, True, True, True, True, False, False, False])


def e_vad3():
    print("(e) VAD3: utterances, event gap merging, voice evidence merging")
    from aggregation.events import events_from_flags
    from aggregation.evidence import GROUPS, merge_voice_events
    segs = [(53.7, 55.2), (55.4, 58.6), (59.0, 60.4), (62.5, 64.3)]   # fragments from the user's live report
    utt = monitor.merge_close(segs, 0.8)
    check("phrase fragments joined into utterances", utt == [(53.7, 60.4), (62.5, 64.3)], f"{utt}")
    slots = [(i * 0.48, (i + 1) * 0.48) for i in range(10)]
    flags = [1, 1, 0, 1, 1, 0, 0, 0, 1, 0]
    ev_gap = events_from_flags(slots, flags, "X", "audio", gap_s=0.8)
    ev_no = events_from_flags(slots, flags, "X", "audio")
    check("events closer than 0.8 s merge (1 slot gap), farther apart stay separate",
          len(ev_gap) == 2 and len(ev_no) == 3, f"(with gap {len(ev_gap)}, default {len(ev_no)})")
    timeline = [{"type": C.MULTIPLE_SPEAKERS, "source": "audio", "start_s": 39.1, "end_s": 43.4, "details": {}},
                {"type": C.VOICE_MISMATCH, "source": "audio", "start_s": 40.1, "end_s": 43.4, "details": {}},
                {"type": C.WHISPERING, "source": "audio", "start_s": 42.5, "end_s": 43.4, "details": {}},
                {"type": C.VOICE_MISMATCH, "source": "audio", "start_s": 55.4, "end_s": 60.2, "details": {}}]
    merged = merge_voice_events(timeline, 1.0)
    check("overlapping voice violations -> one item listing all types",
          len(merged) == 2 and merged[0]["types"] == [C.VOICE_MISMATCH, C.MULTIPLE_SPEAKERS, C.WHISPERING])
    check("plain speech is not voice evidence", C.SPEECH_DETECTED not in GROUPS["voice"])


def d_real():
    print("(d) real clip (v2): phone + second person")
    from tests import eval_video
    summary, _ = eval_video.run("v2")
    for t in (C.PROHIBITED_OBJECT, C.MULTIPLE_PERSONS):
        print(f"    {t}: {summary['scores'].get(t)}")
    print(f"    ms per sample: {summary['median_ms_per_sample_after_warmup']}")


def main():
    a_vad()
    b_persons()
    c_objects()
    e_vad3()
    if "--real" in sys.argv:
        d_real()
    print("VAD2/MP2/OB2:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
