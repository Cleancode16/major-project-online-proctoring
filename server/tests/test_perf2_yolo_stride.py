"""
PERF2: YOLO on every yolo_every_n-th sample, head pose / gaze on every sample.
  (a) stride: YOLO runs on every 2nd sample; skipped samples reuse the last result (marked scene_held)
  (b) freshness: a result older than yolo_max_age_s is never reused (YOLO runs instead)
  (c) persistence: a single YOLO detection reused on the next sample does NOT pass the 2-observation rules;
      two real detections do; reused samples take the flag of their source; a skipped sample is no negative
  (d) buffer support / raw counts use real observations only
  (e) legacy profile (yolo_every_n 1) unchanged: YOLO on every sample
"""

import copy
import sys

import config as C
from aggregation import buffer
from video.pipeline import SCENE_TYPES, VideoAnalyzer

OK = True


def check(name, cond, extra=""):
    global OK
    OK &= bool(cond)
    print(f"  [{'ok' if cond else 'FAIL'}] {name} {extra}")


def analyzer(profile="v2", rate=10.0):
    """VideoAnalyzer without models: only the state _flag_scene / finalize need."""
    va = VideoAnalyzer.__new__(VideoAnalyzer)
    va.cfg = copy.deepcopy(C.get_config(profile))
    va.rate, va.samples, va.processed = rate, [], 0
    va.scene_last, va.scene_skips, va.scene_runs = None, 0, 0
    va.yolo_calls = []

    def fake_run(s, frame, faces):          # frame = (persons, objects) the "detector" sees
        va.yolo_calls.append(s["t"])
        persons, objects = frame
        s["persons"], s["persons_yolo"], s["objects"], s["object_weak"] = persons, persons, dict(objects), False
        s["object_boxes"] = [{"label": k, "conf": v, "xyxy": [0, 0, 1, 1]} for k, v in objects.items()]
        s["raw"][C.MULTIPLE_PERSONS] = persons > 1
        s["raw"][C.PROHIBITED_OBJECT] = bool(objects)
    va._run_scene = fake_run
    return va


def feed(va, frames, dt=0.1):
    for i, fr in enumerate(frames):
        s = {"t": round(i * dt, 3), "phase": "analysis", "face_visible": True, "raw": {}}
        va._flag_scene(s, fr, [])
        va.processed += 1
        va.samples.append(s)
    return va.samples


def finalize(va):
    """The scene part of VideoAnalyzer.finalize (the rest needs the models)."""
    for vt in SCENE_TYPES:
        for s, f in zip(va.samples, va._scene_persistence(vt)):
            s.setdefault("flags", {})[vt] = bool(f)
    return va.samples


def main():
    none, phone, two = (1, {}), (1, {"Cell Phone": 0.9}), (2, {})

    print("(a) stride")
    va = analyzer()
    ss = feed(va, [none] * 10)
    check("YOLO on every 2nd sample", va.yolo_calls == [0.0, 0.2, 0.4, 0.6, 0.8], va.yolo_calls)
    check("skipped samples marked and keep the scene fields",
          all(s.get("scene_held") and s["scene_src"] == i - 1 and s["persons"] == 1 and SCENE_TYPES[0] in s["raw"]
              for i, s in enumerate(ss) if i % 2))
    check("observation rate = rate / 2", abs(va.scene_rate() - 5.0) < 1e-9, va.scene_rate())

    print("(b) freshness")
    va = analyzer(rate=0.5)                 # 2 s between samples > yolo_max_age_s 1.5
    feed(va, [none] * 6, dt=2.0)
    check("stale result never reused", len(va.yolo_calls) == 6 and not any(s.get("scene_held") for s in va.samples))

    print("(c) persistence on real observations")
    va = analyzer(rate=3.0)                 # 3 fps: objects 2-of-3 / persons 1 s, at the 1.5 Hz observation rate
    seq = [none, none, phone, phone, none, none, none, none, two, two, none, none]
    ss = feed(va, seq, dt=1 / 3)
    out = finalize(va)
    obj = [s["flags"][C.PROHIBITED_OBJECT] for s in out]
    per = [s["flags"][C.MULTIPLE_PERSONS] for s in out]
    check("one YOLO phone detection (+ its reused sample) is not flagged", not any(obj), obj)
    check("one YOLO 2-person detection (+ its reused sample) is not flagged", not any(per), per)

    va = analyzer(rate=3.0)
    seq = [none, none, phone, phone, phone, phone, none, none, none, none]
    feed(va, seq, dt=1 / 3)
    obj = [s["flags"][C.PROHIBITED_OBJECT] for s in finalize(va)]
    check("two YOLO phone detections are flagged, reused samples follow their source",
          obj == [False, False, True, True, True, True, False, False, False, False], obj)

    va = analyzer(rate=10.0)                # skipped sample inside a run is no negative evidence
    seq = [phone, none] * 10                # the "none" frames are never seen by YOLO
    feed(va, seq)
    obj = [s["flags"][C.PROHIBITED_OBJECT] for s in finalize(va)]
    check("continuous run across skipped samples", all(obj), obj)

    print("(d) buffer support")
    video = {"samples": va.samples, "rate_hz": va.rate}
    sup = buffer._video_support(va.samples, C.PROHIBITED_OBJECT)
    check("raw detections = real observations only", sum(1 for _, r, _, _ in sup if r) == 10)
    check("supporting observations = real observations only", sum(1 for *_, o in sup if o) == 10)
    ev = [{"start_s": 0.0, "end_s": 2.0, "first_index": 0, "n": len(va.samples)}]
    pol = {"merge_gap_s": 1.0, "confirm_s": 1.0, "min_support": 2}
    merged, _, counts = buffer.buffer_video(video, C.PROHIBITED_OBJECT, ev, pol)
    check("event support counts 10 observations (not 20 samples)", merged[0]["support"] == 10, merged[0]["support"])
    check("valid observation time still covers every sample", counts["den"] == 20)

    print("(e) legacy profile")
    va = analyzer("legacy")
    feed(va, [none] * 6)
    check("YOLO on every sample", len(va.yolo_calls) == 6)
    check("weights", C.get_config("v2")["objects"]["weights"] == "yolov8n.pt"
          and C.get_config("v2")["objects"]["yolo_every_n"] == 2)

    print("PERF2:", "PASS" if OK else "FAIL")
    return OK


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
