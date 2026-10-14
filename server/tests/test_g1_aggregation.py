"""
G1 aggregation: hand-built inputs with hand-computed expected values, every verdict path,
the legacy verdict on the real clip, and the saved report (JSON keys + evidence files).

Run:  .venv\\Scripts\\python.exe -m tests.test_g1_aggregation
"""

import json
import sys

import config as C
from aggregation.aggregate import aggregate
from tests import paths

FAILS = []


def check(name, got, exp, tol=1e-6):
    ok = abs(got - exp) <= tol if isinstance(exp, (int, float)) and not isinstance(exp, bool) else got == exp
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: got {got!r} expected {exp!r}")
    if not ok:
        FAILS.append(name)


def video(n=30, cal=3, flags=None, face_hidden=(), rate=3.0):
    """n samples at `rate`; first `cal` are calibration; flags = {type: [sample indices]}."""
    flags = flags or {}
    samples = []
    for i in range(n):
        visible = i not in face_hidden
        s = {"t": round(i / rate, 3), "phase": "calibration" if i < cal else "analysis", "face_visible": visible,
             "identity_ok": True if visible else None, "pose_ok": visible, "raw": {}, "flags": {},
             "persons": 1, "objects": {}, "frame": i * 10}
        if visible and i >= cal:
            s.update(rel_yaw=0.0, rel_pitch=0.0, rel_roll=0.0)
            s["raw"][C.GAZE_AWAY] = False
        for t in C.VIDEO_TYPES:
            s["flags"][t] = i in flags.get(t, [])
        samples.append(s)
    return {"samples": samples, "rate_hz": rate, "baseline": {"yaw": 0, "pitch": 0, "roll": 0}, "calibration": {}}


def audio(dur=60.0, flags=None, status="ok", voiceprint=True):
    flags = flags or {}
    n = int((dur - 0.96) / 0.48) + 1
    windows = [{"slot": (0.0 if k == 0 else k * 0.48 + 0.24, k * 0.48 + 0.72),
                "flags": {t: k in flags.get(t, []) for t in C.AUDIO_TYPES}} for k in range(n)]
    events = []
    for t, ks in flags.items():
        if ks:
            events.append({"type": t, "source": "audio", "start_s": ks[0] * 0.48, "end_s": (ks[-1] + 1) * 0.48,
                           "duration_s": round(len(ks) * 0.48, 2)})
    return {"status": status, "duration_s": dur, "windows": windows if status == "ok" else [],
            "events": events if status == "ok" else [], "voiceprint": voiceprint}


def stats_math():
    print("(1) per-type statistics")
    cfg = C.get_config("v2")
    v = video(30, 3, {C.PROHIBITED_OBJECT: [10, 11, 12, 13], C.HEAD_POSE_DEVIATION: [20, 21, 22],
                      C.FACE_NOT_VISIBLE: [25, 26, 27]}, face_hidden=(25, 26, 27))
    a = audio(60.0, {C.SPEECH_DETECTED: list(range(10, 20)), C.SUSPICIOUS_SOUND: [50, 51, 52]})
    r = aggregate(v, a, cfg)
    pv = r["per_violation"]
    # objects: 4 of 30 samples (count from t=0) -> 13.333%, 4/3 s, 1 event
    check("objects numerator/denominator", (pv[C.PROHIBITED_OBJECT]["numerator"], pv[C.PROHIBITED_OBJECT]["denominator"]), (4, 30))
    check("objects %", pv[C.PROHIBITED_OBJECT]["percentage"], 13.33)
    check("objects duration_s", pv[C.PROHIBITED_OBJECT]["duration_s"], 1.33)
    check("objects events", pv[C.PROHIBITED_OBJECT]["count"], 1)
    # head pose: denominator = analysis samples with a visible face = 30 - 3 cal - 3 hidden = 24
    check("head pose denominator", pv[C.HEAD_POSE_DEVIATION]["denominator"], 24)
    check("head pose %", pv[C.HEAD_POSE_DEVIATION]["percentage"], 12.5)
    # identity: face-visible samples incl. calibration = 27
    check("identity denominator", pv[C.IDENTITY_MISMATCH]["denominator"], 27)
    check("face not visible %", pv[C.FACE_NOT_VISIBLE]["percentage"], 10.0)
    # audio: 10 windows x 0.48 s = 4.8 s of 60 s = 8%
    check("speech seconds", pv[C.SPEECH_DETECTED]["numerator"], 4.8)
    check("speech %", pv[C.SPEECH_DETECTED]["percentage"], 8.0)
    check("suspicious %", pv[C.SUSPICIOUS_SOUND]["percentage"], 2.4)
    # SC2 score by hand = sum of points: objects 13.33/10 -> full 35, head 12.5/25 = .5 * 10 = 5,
    # face 10/20 = .5 * 12 = 6, speech 8/20 = .4 * 6 = 2.4, suspicious 2.4/20 = .12 * 6 = 0.72; others 0
    check("score", r["score"], round(35 + 5 + 6 + 2.4 + 0.72, 1))
    check("objects points", pv[C.PROHIBITED_OBJECT]["score"], 35.0)
    check("head pose points", pv[C.HEAD_POSE_DEVIATION]["score"], 5.0)
    check("every type has an explanation", all(st.get("score_explanation") for st in pv.values()), True)
    check("overall explanation lists objects", "prohibited object 35.0" in r["score_explanation"], True)
    check("timeline sorted, video+audio", [e["source"] for e in r["timeline"]].count("audio"), 2)


def verdicts():
    print("(2) verdict paths")
    cfg = C.get_config("v2")
    ok_audio = audio(10.0)
    cases = [
        ("impostor majority", video(30, 3, {C.IDENTITY_MISMATCH: list(range(0, 20))}), ok_audio, "INVALID", "Unauthorized"),
        ("phone 4 s at 5%", video(240, 3, {C.PROHIBITED_OBJECT: list(range(100, 112))}), ok_audio, "INVALID", "Prohibited"),
        ("second person 1 s, 2%", video(150, 3, {C.MULTIPLE_PERSONS: [50, 51, 52]}), ok_audio, "SUSPICIOUS", "MULTIPLE_PERSONS"),
        ("gaze 40% only", video(30, 3, {C.GAZE_AWAY: list(range(3, 14))}), ok_audio, "SUSPICIOUS", "GAZE_AWAY"),
        ("clean", video(30, 3), ok_audio, "VALID", "No violation"),
        ("clean but muted mic", video(30, 3), audio(10.0, status="digital_silence"), "SUSPICIOUS", "Audio source"),
        ("voice mismatch 6 s", video(30, 3), audio(60.0, {C.VOICE_MISMATCH: list(range(10, 23))}), "INVALID", "Voice"),
    ]
    for name, v, a, exp_label, keyword in cases:
        r = aggregate(v, a, cfg)
        print(f"   {name:22s} -> {r['verdict']:10s} score {r['score']:5.1f} | {r['reasons'][0]}")
        check(f"{name} label", r["verdict"], exp_label)
        check(f"{name} reason mentions '{keyword}'", any(keyword in x for x in r["reasons"]), True)


def scoring_and_limits():
    print("(2b) SC2 points (critical floor, severity order) and LIM2 percentage limits")
    from aggregation import evidence as ev
    cfg = C.get_config("v2")
    # one phone sample out of 300 (0.4%): 0.4/10 * 35 = 1.4 -> raised to the critical floor 17.5
    r = aggregate(video(300, 3, {C.PROHIBITED_OBJECT: [100]}), audio(100.0), cfg)
    check("single phone sample -> critical floor 17.5", r["per_violation"][C.PROHIBITED_OBJECT]["score"], 17.5)
    check("single phone sample -> SUSPICIOUS", r["verdict"], "SUSPICIOUS")
    # gaze away all the time scores less than a brief head turn would at the same % (gaze lowest)
    mp = cfg["aggregation"]["max_points"]
    check("gaze max < head pose max < critical max",
          mp[C.GAZE_AWAY] < mp[C.HEAD_POSE_DEVIATION] < min(mp[t] for t, sv in cfg["aggregation"]["severity"].items()
                                                              if sv == "critical"), True)
    # head pose 20% of 100 analysis samples: under the 25% limit; persons 6%: over the 5% limit
    v = video(103, 3, {C.HEAD_POSE_DEVIATION: list(range(10, 30)), C.MULTIPLE_PERSONS: list(range(40, 46))})
    a = audio(60.0, {C.WHISPERING: list(range(10, 14)), C.VOICE_MISMATCH: list(range(12, 18))})
    r = aggregate(v, a, cfg)
    gp = r["group_percentages"]
    check("head pose group %", gp["head_pose"], 20.0)
    # voice = union of windows 10..17 = 8 x 0.48 = 3.84 s of 60 s = 6.4% (overlap counted once)
    check("voice group % (union)", gp["voice"], 6.4)
    rep_ = {"group_percentages": gp, "evidence": {g: {"items": []} for g in ev.GROUPS}}
    c = ev.counters(rep_, cfg)
    check("head pose 20% <= 25% not exceeded", c["head_pose"]["exceeded"], False)
    check("persons 5.83% > 5% exceeded", c["multiple_faces"]["exceeded"], True)
    check("voice 6.4% > 5% exceeded", c["voice"]["exceeded"], True)
    rep_["evidence"]["object"]["items"] = [{}]
    check("one object event exceeded (any detection)", ev.counters(rep_, cfg)["object"]["exceeded"], True)


def legacy_on_clip():
    print("(3) legacy profile verdict on the real clip == v1 output")
    from tests.eval_video import run
    _, res = run("legacy")
    r = aggregate(res, None, C.get_config("legacy"))["legacy"]
    check("legacy verdict", r["verdict"], "EXAM INVALID - Prohibited objects found")
    check("legacy counts (pose, gaze, persons, objects)",
          (r["counts"][C.HEAD_POSE_DEVIATION], r["counts"][C.GAZE_AWAY], r["counts"][C.MULTIPLE_PERSONS],
           r["counts"][C.PROHIBITED_OBJECT]), (3, 38, 4, 11))
    check("legacy analyzed frames", r["analyzed_frames"], 35)


def saved_report():
    print("(4) saved report from the CLI run (tests/results/G1_sample_report)")
    d = paths.RESULTS / "G1_sample_report"
    rep = json.loads((d / "report.json").read_text())
    for k in ("meta", "verdict", "per_violation", "timeline", "video", "audio", "warnings", "thresholds"):
        check(f"key {k}", k in rep, True)
    ev = [e for e in rep["timeline"] if e["source"] == "video"]
    check("every video event has an evidence file", all((d / e.get("evidence", "missing")).exists() for e in ev), True)
    check("audio silent warning present", any("AUDIO_SOURCE_SILENT" in w for w in rep["warnings"]), True)


def main():
    stats_math()
    verdicts()
    scoring_and_limits()
    legacy_on_clip()
    saved_report()
    print("G1 AGGREGATION:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
