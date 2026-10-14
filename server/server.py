"""
Online Proctoring System - backend server (v2). See flow.txt PART II-E.

  Flask REST API (port 5000, localhost only) used by the React client in client/proctoring
  + WebRTC signalling for live proctoring (aiortc, see webrtc_session.py).

Run from the repository root:   .venv\\Scripts\\python.exe server\\server.py
The previous Flask/Socket.IO server is preserved in legacy/server_v1.py.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import asyncio  # noqa: E402
import base64  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from flask import Flask, Response, jsonify, request, send_file, send_from_directory  # noqa: E402
from flask_cors import CORS  # noqa: E402

import config as C  # noqa: E402
import enrollment_store  # noqa: E402
import proctor  # noqa: E402
from aggregation import evidence as evidence_store  # noqa: E402
from aggregation.report import _jsonable  # noqa: E402
from audio import enrollment as voice_enrollment  # noqa: E402
from audio import io as aio  # noqa: E402
from audio.passages import PASSAGES  # noqa: E402
from video import face, models  # noqa: E402
from webrtc_session import LiveSession, monitors_for_sample  # noqa: E402

print(">>> Server starting...")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 ** 3      # recorded exam videos can be large

CFG = C.get_config("v2")
UPLOADS = C.DATA_DIR / "uploads"

# aiortc needs asyncio; Flask is threaded. One event loop runs in a background thread and
# routes submit coroutines to it (flow.txt §E).
LOOP = asyncio.new_event_loop()
threading.Thread(target=LOOP.run_forever, name="webrtc-loop", daemon=True).start()

SESSIONS = {}     # session_id -> LiveSession
JOBS = {}         # job_id -> dict (recorded-video analysis)
STATE_LOCK = threading.Lock()


def run_async(coro, timeout=60):
    return asyncio.run_coroutine_threadsafe(coro, LOOP).result(timeout)


def json_response(obj, status=200):
    return Response(json.dumps(obj, default=_jsonable), status=status, mimetype="application/json")


def error(message, status=400):
    return jsonify({"error": message}), status


def warm_models():
    """Load FaceNet / YOLO once at start-up so the first request is not slow."""
    try:
        models.facenet()
        models.embed_faces([np.zeros((160, 160, 3), np.uint8)])
        models.yolo(CFG["objects"]["weights"])
        print(">>> Models loaded")
    except Exception:
        traceback.print_exc()


def decode_image(data_url):
    if not data_url:
        return None
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def stored_enrollment(candidate):
    if not candidate or not enrollment_store.exists(candidate):
        return None
    return enrollment_store.load(candidate)


def report_dir(report_id):
    """Resolve a report folder name safely inside data/reports (no path traversal)."""
    base = C.REPORTS_DIR.resolve()
    d = (base / report_id).resolve()
    if d.parent != base or not (d / "report.json").exists():
        return None
    return d


# ======================================================================= health
@app.get("/api/health")
def health():
    return jsonify({"status": "healthy", "sessions": len(SESSIONS), "timestamp": time.time()})


# ======================================================================= enrollment
@app.get("/api/enrollment/passage")
def enrollment_passage():
    exclude = request.args.get("exclude", type=int)
    choices = [i for i in range(len(PASSAGES)) if i != exclude] or [0]
    i = int(np.random.default_rng().choice(choices))
    return jsonify({"index": i, "text": PASSAGES[i], "record_s": CFG["enroll"]["record_s"],
                    "max_attempts": CFG["enroll"]["max_attempts"]})


@app.get("/api/enrollment/<candidate>")
def enrollment_status(candidate):
    st = stored_enrollment(candidate)
    if st is None:
        return jsonify({"candidate": candidate, "face": False, "voice": False, "meta": {}})
    return json_response({"candidate": candidate, "face": st["face_embedding"] is not None,
                          "voice": st["voiceprint"] is not None, "meta": st["meta"]})


@app.post("/api/enrollment/face")
def enrollment_face():
    data = request.get_json(silent=True) or {}
    candidate = (data.get("candidate") or "").strip()
    if not candidate:
        return error("Candidate ID is required.")
    if not data.get("consent"):
        return error("Consent is required before enrollment.")
    img = decode_image(data.get("image"))
    if img is None:
        return error("Invalid image data.")
    fc = CFG["face"]
    faces = face.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), models.face_detector(fc["det_confidence"]),
                              min_face_px=fc["min_face_px"])
    if not faces:
        return error("No face found. Face the camera in good light and try again.")
    if len(faces) > 1:
        return error(f"{len(faces)} faces found. Only you should be in the photo.")
    if faces[0]["bbox"][2] < 0.15 * img.shape[1]:
        return error("Face too small. Move closer to the camera.")
    emb, _, _ = face.reference_from_image(img, fc)
    enrollment_store.save(candidate, img, emb, None,
                          {"consent": True, "consent_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "voice_attempts": 0})
    return jsonify({"success": True, "message": "Reference face enrolled.",
                    "bbox": list(faces[0]["bbox"]), "image_size": [img.shape[1], img.shape[0]]})


@app.post("/api/enrollment/voice")
def enrollment_voice():
    candidate = (request.form.get("candidate") or "").strip()
    passage = request.form.get("passage", type=int)
    upload = request.files.get("audio")
    st = stored_enrollment(candidate)
    if st is None or not st["meta"].get("consent"):
        return error("Enroll the reference photo (with consent) first.")
    if upload is None or passage is None or not 0 <= passage < len(PASSAGES):
        return error("Missing recording or passage.")
    ec = CFG["enroll"]
    attempts = int(st["meta"].get("voice_attempts", 0))
    if attempts >= ec["max_attempts"]:
        return error(f"Maximum of {ec['max_attempts']} voice attempts reached. Re-enroll the photo to start over.", 429)

    UPLOADS.mkdir(parents=True, exist_ok=True)
    tmp = UPLOADS / f"voice_{uuid.uuid4().hex}{Path(upload.filename or 'rec.webm').suffix or '.webm'}"
    upload.save(tmp)
    try:
        x = aio.load_from_video(tmp)          # ffmpeg decodes webm/ogg/wav/... to 16 kHz mono
    finally:
        tmp.unlink(missing_ok=True)           # the raw recording is never kept (flow.txt §B.3)
    if x is None:
        return error("The recording contains no audio.")
    r = voice_enrollment.evaluate_recording(x, PASSAGES[passage], CFG)
    attempts += 1
    meta = {"voice_attempts": attempts}
    if r["ok"]:
        meta.update(voice_ok=True, passage=passage, voice_metrics=r["metrics"], asr=r["asr"])
    enrollment_store.save(candidate, voiceprint=r["voiceprint"] if r["ok"] else None, meta=meta)
    return json_response({"ok": r["ok"], "problems": r["problems"], "warnings": r["warnings"],
                          "metrics": r["metrics"], "asr": r["asr"], "attempt": attempts,
                          "max_attempts": ec["max_attempts"]})


# ======================================================================= live (WebRTC)
@app.post("/api/webrtc/offer")
def webrtc_offer():
    data = request.get_json(silent=True) or {}
    candidate = (data.get("candidate") or "").strip()
    st = stored_enrollment(candidate)
    if st is None or st["face_embedding"] is None:
        return error("No face enrollment for this candidate. Complete enrollment first.")
    if not data.get("sdp") or not data.get("type"):
        return error("Missing SDP offer.")
    sid = uuid.uuid4().hex[:12]
    session = LiveSession(sid, candidate, CFG, st["face_embedding"], st["voiceprint"])
    try:
        sdp, sdp_type = run_async(session.accept(data["sdp"], data["type"]))
    except Exception as exc:
        traceback.print_exc()
        return error(f"WebRTC negotiation failed: {exc}", 500)
    with STATE_LOCK:
        SESSIONS[sid] = session
    return jsonify({"session_id": sid, "sdp": sdp, "type": sdp_type})


@app.post("/api/session/<sid>/stop")
def session_stop(sid):
    with STATE_LOCK:
        session = SESSIONS.pop(sid, None)
    if session is None:
        return error("Unknown or already stopped session.", 404)
    try:
        report, path = run_async(session.stop(), timeout=900)
    except Exception as exc:
        traceback.print_exc()
        return error(f"Could not build the report: {exc}", 500)
    return json_response({"report_id": path.parent.name, "report": report})


# ======================================================================= recorded video
def _run_job(job_id, video_path, st, candidate, use_audio):
    job = JOBS[job_id]

    def progress(t, dur, s):
        job.update(progress=round(min(1.0, t / max(dur, 1e-6)), 3), t=round(t, 2), duration=round(dur, 2))
        if s is not None:
            job["monitors"] = monitors_for_sample(s)

    try:
        report, path, _, _ = proctor.analyze_files(video_path, st["face_embedding"], CFG, st["voiceprint"],
                                                   "video" if use_audio else "none", candidate, progress=progress,
                                                   session_id=job_id)
        job.update(status="done", progress=1.0, report_id=path.parent.name, report=report)
    except Exception as exc:
        traceback.print_exc()
        job.update(status="error", error=str(exc))


@app.post("/api/analyze-video")
def analyze_video():
    candidate = (request.form.get("candidate") or "").strip()
    upload = request.files.get("video")
    use_audio = request.form.get("use_audio", "true").lower() in ("1", "true", "yes", "on")
    st = stored_enrollment(candidate)
    if st is None or st["face_embedding"] is None:
        return error("No face enrollment for this candidate. Complete enrollment first.")
    if upload is None:
        return error("No video file uploaded.")
    job_id = uuid.uuid4().hex[:12]
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"{job_id}{Path(upload.filename or 'video.mp4').suffix or '.mp4'}"
    upload.save(path)
    JOBS[job_id] = {"status": "running", "progress": 0.0, "t": 0.0, "duration": None, "monitors": None,
                    "report_id": None, "report": None, "error": None}
    threading.Thread(target=_run_job, args=(job_id, path, st, candidate, use_audio), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.get("/api/analyze-video/<job_id>")
def analyze_video_status(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return error("Unknown job.", 404)
    return json_response(job)


# ======================================================================= reports
@app.get("/api/reports")
def reports_list():
    out = []
    if C.REPORTS_DIR.exists():
        for p in sorted(C.REPORTS_DIR.glob("*/report.json"), reverse=True):
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            out.append({"id": p.parent.name, "created": r["meta"].get("created"),
                        "candidate": r["meta"].get("candidate"), "session_id": r["meta"].get("session_id"),
                        "verdict": r["verdict"]["label"], "score": r["verdict"]["score"],
                        "violated": (r.get("session_violation") or {}).get("violated", False)})
    return jsonify(out)


@app.get("/api/reports/<report_id>")
def report_get(report_id):
    d = report_dir(report_id)
    if d is None:
        return error("Report not found.", 404)
    return Response((d / "report.json").read_text(encoding="utf-8"), mimetype="application/json")


@app.get("/api/reports/<report_id>/pdf")
def report_pdf(report_id):
    """PDF version of a report (flow.txt §C.4 / §E.1); rebuilt on every download so it follows the current layout."""
    d = report_dir(report_id)
    if d is None:
        return error("Report not found.", 404)
    try:
        from aggregation.pdf_report import write_report_pdf
        pdf = write_report_pdf(d / "report.json")
    except Exception as exc:
        traceback.print_exc()
        return error(f"Could not build the PDF: {exc}", 500)
    return send_file(pdf, mimetype="application/pdf", as_attachment=True,
                     download_name=f"proctoring_report_{report_id}.pdf")


@app.get("/api/reports/<report_id>/docx")
def report_docx(report_id):
    """Word version of a report (flow.txt §C.4 / §E.1, DOCX1); built on every download."""
    d = report_dir(report_id)
    if d is None:
        return error("Report not found.", 404)
    try:
        from aggregation.docx_report import write_report_docx
        docx = write_report_docx(d / "report.json")
    except Exception as exc:
        traceback.print_exc()
        return error(f"Could not build the Word file: {exc}", 500)
    return send_file(docx, mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                     as_attachment=True, download_name=f"proctoring_report_{report_id}.docx")


@app.get("/api/reports/<report_id>/files/<path:relpath>")
def report_file(report_id, relpath):
    """Evidence files of a report (flow.txt §E.1): jpg / wav / json inside the report folder only."""
    d = report_dir(report_id)
    if d is None:
        return error("Report not found.", 404)
    target = (d / relpath).resolve()
    if d not in target.parents or target.suffix.lower() not in (".jpg", ".wav", ".json") or not target.is_file():
        return error("File not found.", 404)
    return send_from_directory(target.parent, target.name)


# ======================================================================= evidence upload (flow.txt §C.6)
@app.post("/api/evidence/upload")
def evidence_upload():
    session_id = (request.form.get("session_id") or "").strip()
    group = (request.form.get("group") or "").strip()
    if not session_id or group not in evidence_store.GROUPS:
        return error("session_id and a valid group are required.")
    try:
        summary = json.loads(request.form.get("summary") or "{}")
        metadata = json.loads(request.form.get("metadata") or "[]")
    except json.JSONDecodeError:
        return error("summary / metadata must be JSON.")
    files = [(f.filename or "file", f.read()) for f in request.files.getlist("files")]
    folder = evidence_store.store_upload(session_id, group, summary, metadata, files)
    return jsonify({"success": True, "session_id": session_id, "group": group, "files": len(files),
                    "stored": folder.name})


@app.get("/api/evidence/<session_id>")
def evidence_list(session_id):
    base = evidence_store.UPLOAD_DIR.resolve()
    d = (base / session_id).resolve()
    if d.parent != base or not d.is_dir():
        return error("No uploaded evidence for this session.", 404)
    out = {}
    for m in sorted(d.glob("*/manifest.json")):
        out[m.parent.name] = json.loads(m.read_text(encoding="utf-8"))
    return json_response({"session_id": session_id, "groups": out})


if __name__ == "__main__":
    threading.Thread(target=warm_models, daemon=True).start()
    print("=" * 50)
    print("Online Proctoring Server (v2) on http://localhost:5000")
    print("REST API under /api, WebRTC signalling at /api/webrtc/offer")
    print("=" * 50)
    # localhost only: face/voice data must not be reachable from the network (flow.txt PART II-D)
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
