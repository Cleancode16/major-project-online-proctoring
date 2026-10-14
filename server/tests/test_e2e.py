"""
T2 end-to-end: enrollment -> analysis (video + audio) -> aggregation -> JSON report -> report API.

Test clip (SYNTHETIC audio on REAL video): the real sample clip with a generated soundtrack:
  1.0-4.5 s  enrolled voice (TTS "David", same voice as the enrollment) asks a question
  6.0-9.5 s  a different voice (TTS "Zira") answers
  10.0-12.6 s phone ring
Enrollment: face = the real reference photo, voice = TTS David reading passage 0.
Expected: INVALID (phone in view, video); audio events VOICE_MISMATCH + MULTIPLE_SPEAKERS around 6-9.5 s,
SUSPICIOUS_SOUND (phone) around 10-12.6 s, all on the same clock as the video events.

Run (from server/):  ..\\.venv\\Scripts\\python.exe -m tests.test_e2e
"""

import json
import subprocess
import sys

import numpy as np

import config as C
import enrollment_store
from audio import enrollment
from audio import io as aio
from audio.passages import PASSAGES
from tests import paths
from tests.make_synthetic_audio import ring

SYN = paths.DATA / "synthetic"
CLIP = SYN / "e2e_clip.mp4"
CAND = "e2e_candidate"
FAILS = []


def check(name, cond, info=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name} {info}")
    if not cond:
        FAILS.append(name)


def build_clip():
    SYN.mkdir(parents=True, exist_ok=True)
    dur = 12.67
    x = np.random.default_rng(1).standard_normal(int(dur * aio.SR)).astype(np.float32) * 10 ** (-55 / 20)

    def put(sig, at, db=-24):
        sig = sig / (np.sqrt(np.mean(sig[np.abs(sig) > 1e-3] ** 2)) + 1e-9) * 10 ** (db / 20)
        i = int(at * aio.SR)
        n = min(len(sig), len(x) - i)
        x[i:i + n] += sig[:n]

    tts = paths.DATA / "audio" / "tts"
    put(aio.load_file(tts / "david_s0.wav")[:int(3.5 * aio.SR)], 1.0)
    put(aio.load_file(tts / "zira_s3.wav")[:int(3.5 * aio.SR)], 6.0)
    put(ring(2.6), 10.0, -20)
    wav = SYN / "e2e_audio.wav"
    aio.save_wav(wav, np.clip(x, -1, 1))
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(paths.SAMPLE_VIDEO), "-i", str(wav),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(CLIP)], check=True)


def main():
    print("(1) build test clip")
    build_clip()
    a = aio.load_from_video(CLIP)
    check("clip has audio", a is not None and aio.source_status(a) == "ok", f"{len(a) / aio.SR:.2f} s")

    print("(2) enrollment (face photo + voice reading) -> local store")
    cfg = C.get_config("v2")
    import cv2
    from video import face
    img = cv2.imread(str(paths.SAMPLE_REF))
    emb, _, _ = face.reference_from_image(img, cfg["face"])
    r = enrollment.evaluate_recording(aio.load_file(paths.DATA / "audio" / "enroll_david_p0.wav"), PASSAGES[0], cfg)
    check("voice enrollment accepted", r["ok"], str(r["problems"]))
    d = enrollment_store.save(CAND, img, emb, r["voiceprint"], {"consent": True, "passage": 0})
    st = enrollment_store.load(CAND)
    check("enrollment stored and reloaded", st["voiceprint"] is not None and st["face_embedding"] is not None, str(d))

    print("(3) CLI analysis using the stored enrollment")
    out = paths.RESULTS / "E2E_report"
    proc = subprocess.run([sys.executable, "app.py", "--candidate", CAND, "--video", str(CLIP), "--out", str(out)],
                          capture_output=True, text=True, cwd=paths.ROOT)
    tail = proc.stdout[proc.stdout.find("FINAL RESULT") - 62:]
    print(tail)
    check("CLI exit code 0", proc.returncode == 0, proc.stderr[-400:] if proc.returncode else "")
    rep = json.loads((out / "report.json").read_text())

    print("(4) report content")
    ev = rep["timeline"]
    types = {(e["source"], e["type"]) for e in ev}
    check("verdict INVALID", rep["verdict"]["label"] == "INVALID", rep["verdict"]["headline"])
    check("video phone event", ("video", C.PROHIBITED_OBJECT) in types)
    check("video multiple-persons event", ("video", C.MULTIPLE_PERSONS) in types)

    def near(t, a, b):
        return any(e["type"] == t and e["start_s"] < b and e["end_s"] > a for e in ev)
    check("VOICE_MISMATCH in 6-9.5 s", near(C.VOICE_MISMATCH, 6.0, 9.5))
    check("no VOICE_MISMATCH while the candidate speaks (1-4.5 s)", not near(C.VOICE_MISMATCH, 1.3, 4.2))
    check("MULTIPLE_SPEAKERS present", near(C.MULTIPLE_SPEAKERS, 1.0, 9.5))
    check("SUSPICIOUS_SOUND (phone) in 10-12.6 s", near(C.SUSPICIOUS_SOUND, 10.0, 12.6))
    phone = [e for e in ev if e["type"] == C.SUSPICIOUS_SOUND]
    check("sound category = phone", bool(phone) and phone[0].get("details", {}).get("category") == "phone")
    check("audio measured with voiceprint", rep["audio"]["status"] == "ok" and rep["audio"]["voiceprint"])
    check("timeline sorted on one clock", [e["start_s"] for e in ev] == sorted(e["start_s"] for e in ev))
    check("video+audio percentages present", all(rep["per_violation"][t]["measurable"] for t in
                                                  (C.PROHIBITED_OBJECT, C.SPEECH_DETECTED, C.VOICE_MISMATCH)))

    print("(5) the server's report API serves this report and its evidence (Flask test client)")
    import shutil
    import config as C
    import server
    dst = C.REPORTS_DIR / "e2e_test_report"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(out, dst)
    client = server.app.test_client()
    r = client.get("/api/reports/e2e_test_report")
    check("GET /api/reports/<id>", r.status_code == 200 and r.get_json()["verdict"]["label"] == "INVALID")
    ev = next((e["evidence"] for e in rep["timeline"] if e.get("evidence")), None)
    check("GET evidence file", ev is not None and
          client.get(f"/api/reports/e2e_test_report/files/{ev}").status_code == 200)
    check("evidence grouped (voice/object/multiple_faces/head_pose)",
          set(rep.get("evidence", {})) == {"voice", "object", "multiple_faces", "head_pose"})
    check("voice evidence has a WAV clip", any((it.get("file") or "").endswith(".wav")
                                               for it in rep["evidence"]["voice"]["items"]))
    check("object group exceeded -> session violated + uploaded",
          rep["session_violation"]["violated"] and rep["evidence_upload"]["object"]["status"] == "uploaded")
    check("path traversal refused", client.get("/api/reports/..%2Fenrollment").status_code == 404)
    print("T2 E2E:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
