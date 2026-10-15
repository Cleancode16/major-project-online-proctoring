"""
A4 YAMNet categories, A5 speaker verification / multiple speakers, A6 audio events, on the SYNTHETIC scenes.
Voiceprint = enrollment of TTS voice "David" reading passage 0 (same code path as the UI enrollment).

Window-level scoring: a window is positive for a type if its owned slot overlaps a labelled segment;
windows whose slot is within 0.5 s of a label boundary are not scored (TTS files have silent edges).
Also checks: live (incremental feed) == offline results, and event schema fields.

Run:  .venv\\Scripts\\python.exe -m tests.test_a4_a6_audio_monitor
"""

import json
import sys

import numpy as np

import config as C
from audio import enrollment, monitor
from audio import io as aio
from audio.passages import SYNTHETIC_PASSAGES as PASSAGES  # recordings were made from these
from tests import paths

AUD = paths.DATA / "audio"
TOL = 0.5

EXPECT = {  # scene -> {type: label key in scenes.json or "none"}
    "silence": {},
    "candidate_speech": {C.SPEECH_DETECTED: "SPEECH"},
    "other_speaker": {C.SPEECH_DETECTED: "SPEECH", C.VOICE_MISMATCH: "SPEECH"},
    "dialogue": {C.SPEECH_DETECTED: "SPEECH", C.MULTIPLE_SPEAKERS: "SPEECH", C.VOICE_MISMATCH: "OTHER"},
    "music": {C.SUSPICIOUS_SOUND: "SUSPICIOUS_SOUND"},
    "phone_ring": {C.SUSPICIOUS_SOUND: "SUSPICIOUS_SOUND"},
    "typing": {},
    "whisper": {C.SPEECH_DETECTED: "SPEECH", C.WHISPERING: "WHISPERING"},
}
TYPES = [C.SPEECH_DETECTED, C.VOICE_MISMATCH, C.MULTIPLE_SPEAKERS, C.WHISPERING, C.SUSPICIOUS_SOUND]


def label_segments(sc, key):
    lab = sc["labels"]
    if key == "OTHER":  # dialogue: only the other speaker's turns
        return [s for s, who in zip(lab["SPEECH"], lab["speakers"]) if who == "other"]
    return lab.get(key, [])


def score_scene(res, sc, expect, cfg):
    out = {}
    for vtype in TYPES:
        segs = label_segments(sc, expect[vtype]) if vtype in expect else []
        tp = fp = fn = 0
        for w in res["windows"]:
            if vtype not in w.get("flags", {}):
                continue
            a, b = w["slot"]
            truth = any(min(b, e) - max(a, s) > 0 for s, e in segs)
            near_edge = any(min(abs(a - s), abs(b - s), abs(a - e), abs(b - e)) < TOL for s, e in segs)
            if near_edge:
                continue
            if vtype == C.SPEECH_DETECTED and truth:
                # policy flag_after: the first allow_speech_s of speech is allowed by design
                pass
            pred = w["flags"][vtype]
            tp += pred and truth
            fp += pred and not truth
            fn += (not pred) and truth
        out[vtype] = (tp, fp, fn)
    return out


def main():
    cfg = C.get_config("v2", {"audio_policy": {"speech": "flag_all"}})
    enr = enrollment.evaluate_recording(aio.load_file(AUD / "enroll_david_p0.wav"), PASSAGES[0], cfg)
    assert enr["ok"], enr["problems"]
    vp = enr["voiceprint"]
    scenes = json.loads((AUD / "scenes.json").read_text())
    totals = {t: [0, 0, 0] for t in TYPES}
    ok = True
    for name, sc in scenes.items():
        x = aio.load_file(AUD / f"{name}.wav")
        res = monitor.analyze_audio(x, cfg, vp)
        s = score_scene(res, sc, EXPECT[name], cfg)
        for t, (tp, fp, fn) in s.items():
            totals[t][0] += tp; totals[t][1] += fp; totals[t][2] += fn
        sims = [c.get("sim_s") for c in res["chunks"] if c.get("sim_s") is not None]
        ev = [(e["type"], e["start_s"], e["end_s"]) for e in res["events"]]
        print(f"{name:17s} " + " ".join(f"{t.split('_')[0][:5]}:{v}" for t, v in s.items() if any(v))
              + (f" | chunk sims {min(sims):.2f}..{max(sims):.2f}" if sims else "") + f" | events {ev[:4]}{'...' if len(ev) > 4 else ''}")
    print("\nWindow-level totals over all scenes (TP, FP, FN):")
    for t, (tp, fp, fn) in totals.items():
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        print(f"  {t:18s} TP {tp:3d} FP {fp:3d} FN {fn:3d}  precision {p if p is None else round(p, 3)}  recall {r if r is None else round(r, 3)}")
        if t != C.WHISPERING:
            ok &= (p is None or p >= 0.9) and (r is None or r >= 0.8)

    # live == offline on the dialogue scene (feed in 0.1 s blocks)
    x = aio.load_file(AUD / "dialogue.wav")
    off = monitor.analyze_audio(x, cfg, vp)
    aa = monitor.AudioAnalyzer(cfg, vp)
    for i in range(0, len(x), 1600):
        aa.feed(x[i:i + 1600])
    status = aa.live_status()
    live = aa.finalize()
    same = [(e["type"], e["start_s"], e["end_s"]) for e in off["events"]] == \
           [(e["type"], e["start_s"], e["end_s"]) for e in live["events"]]
    max_diff = max(abs(a["cats"][c] - b["cats"][c]) for a, b in zip(off["windows"], live["windows"]) for c in a["cats"])
    print(f"\nlive vs offline (dialogue): same events={same}, max YAMNet category diff {max_diff:.4f}; "
          f"live_status keys {sorted(status)}")
    keys_ok = all({"type", "source", "start_s", "end_s", "duration_s"} <= set(e) for e in off["events"])
    print(f"event schema fields present: {keys_ok}")
    ok &= same and keys_ok
    print("A4-A6 AUDIO MONITOR:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
