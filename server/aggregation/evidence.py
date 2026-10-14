"""
Evidence package, per-group counters, session violation and upload (flow.txt §C.5, §C.6).

Groups (each sorted by time):
  voice           VOICE_MISMATCH, MULTIPLE_SPEAKERS, WHISPERING -> one WAV clip per merged voice violation
  object          PROHIBITED_OBJECT                                              -> annotated snapshot
  multiple_faces  MULTIPLE_PERSONS                                               -> annotated snapshot
  head_pose       HEAD_POSE_DEVIATION                                            -> annotated snapshot
"""

import json
import re
import time
from pathlib import Path

import numpy as np

import config as C

GROUPS = {
    # voice VIOLATIONS only (user decision 2026-10-14): plain SPEECH_DETECTED stays in stats/timeline, not evidence
    "voice": [C.VOICE_MISMATCH, C.MULTIPLE_SPEAKERS, C.WHISPERING],
    "object": [C.PROHIBITED_OBJECT],
    "multiple_faces": [C.MULTIPLE_PERSONS],
    "head_pose": [C.HEAD_POSE_DEVIATION],
    "spoof": [C.FACE_SPOOF],                  # FAS1
}
GROUP_LABELS = {"voice": "Voice", "object": "Prohibited objects", "multiple_faces": "Multiple faces",
                "head_pose": "Head pose", "spoof": "Face spoofing"}
GROUP_OF = {t: g for g, types in GROUPS.items() for t in types}
SR = 16000
UPLOAD_DIR = C.DATA_DIR / "evidence_uploads"


# ---------------------------------------------------------------------------- build
def _pose_peak(s):
    return max(abs(s.get("rel_yaw", 0.0)), abs(s.get("rel_pitch", 0.0)), abs(s.get("rel_roll", 0.0)))


PEAK_KEY = {
    "object": lambda s: max(s.get("objects", {}).values(), default=0.0),
    "multiple_faces": lambda s: (s.get("persons", 0), s.get("n_faces", 0)),
    "head_pose": _pose_peak,
    "spoof": lambda s: 1.0 - (s.get("spoof_score") if s.get("spoof_score") is not None else 1.0),
}


def _video_item(group, event, samples, cfg, folder, idx):
    first, n = event.get("first_index"), event.get("n", 1)
    run = samples[first:first + n] if first is not None else []
    with_img = [s for s in run if s.get("_thumb")]
    snap = max(with_img, key=PEAK_KEY[group]) if with_img else (max(run, key=PEAK_KEY[group]) if run else None)
    item = {}
    if snap is not None and snap.get("_thumb"):
        name = f"{idx:02d}_{event['type'].lower()}_{event['start_s']:.1f}s.jpg"
        (folder / name).write_bytes(snap["_thumb"])
        item["file"] = f"evidence/{group}/{name}"
        item["snapshot_t"] = snap["t"]
    else:
        item["file"] = None
        item["note"] = "no image stored for this event (snapshot limit reached)"
    if snap is None:
        return item
    if group == "object":
        boxes = snap.get("object_boxes") or [{"label": k, "conf": v} for k, v in snap.get("objects", {}).items()]
        item["objects"] = [{"label": b["label"], "confidence": round(float(b["conf"]), 3)} for b in boxes]
        best = {}
        for s in run:
            for label, conf in s.get("objects", {}).items():
                best[label] = max(best.get(label, 0.0), round(float(conf), 3))
        item["max_confidence"] = best
    elif group == "multiple_faces":
        item["face_count"] = snap.get("n_faces", 0)
        item["person_count"] = snap.get("persons", 0)
        item["max_persons"] = max(s.get("persons", 0) for s in run)
    elif group == "head_pose":
        hp = cfg["head_pose"]
        item["angles_deg"] = {k: round(snap[f"rel_{k}"], 1) for k in ("yaw", "pitch", "roll") if f"rel_{k}" in snap}
        item["max_abs_angles_deg"] = {k: round(max((abs(s[f"rel_{k}"]) for s in run if f"rel_{k}" in s), default=0.0), 1)
                                      for k in ("yaw", "pitch", "roll")}
        item["allowed_deg"] = {"yaw": hp["yaw_thr"], "pitch": hp["pitch_thr"], "roll": hp["roll_thr"]}
        item["allowed_duration_s"] = hp["persist_s"]
    elif group == "spoof":                       # FAS1
        ac = cfg.get("antispoof", {})
        item["real_score"] = snap.get("spoof_score")
        item["min_real_score"] = min((s["spoof_score"] for s in run if s.get("spoof_score") is not None), default=None)
        item["real_threshold"] = ac.get("real_thr")
        item["allowed_duration_s"] = ac.get("persist_s")
    return item


def _audio_item(event, audio, cfg, folder, idx):
    from audio import io as aio
    item = {"details": event.get("details", {})}
    x = (audio or {}).get("_x")
    ec = cfg["evidence"]
    if x is None or len(x) == 0:
        item["file"] = None
        item["note"] = "audio not available"
        return item
    total = len(x) / SR
    a = max(0.0, event["start_s"] - ec["clip_pad_s"])
    b = min(total, event["end_s"] + ec["clip_pad_s"])
    b = min(b, a + ec["clip_max_s"])
    name = f"{idx:02d}_{event['type'].lower()}_{event['start_s']:.1f}s.wav"
    aio.save_wav(folder / name, x[int(a * SR):int(b * SR)])
    item.update(file=f"evidence/voice/{name}", clip_start_s=round(a, 2), clip_end_s=round(b, 2))
    segs = audio.get("segments") or []
    item["speech_s_in_event"] = round(sum(max(0.0, min(e, event["end_s"]) - max(s, event["start_s"])) for s, e in segs), 2)
    wins = [w for w in audio.get("windows", []) if w["t_end"] > event["start_s"] and w["t_start"] < event["end_s"]]
    sims = [w["speaker_sim"] for w in wins if w.get("speaker_sim") is not None]
    if sims:
        item["speaker_similarity_min"] = round(float(min(sims)), 3)
        item["speaker_threshold"] = cfg["speaker"]["verify_thr"]
    if wins:
        item["whisper_score_max"] = round(float(max(w["cats"]["whisper"] for w in wins)), 3)
    return item


VOICE_ORDER = [C.VOICE_MISMATCH, C.MULTIPLE_SPEAKERS, C.WHISPERING]   # headline type of a merged item


def merge_voice_events(events, gap_s):
    """Overlapping / adjacent (gap <= gap_s) voice violations -> one event with all their types (flow.txt §C.5)."""
    merged = []
    for e in sorted(events, key=lambda e: e["start_s"]):
        if merged and e["start_s"] <= merged[-1]["end_s"] + gap_s:
            m = merged[-1]
            m["end_s"] = max(m["end_s"], e["end_s"])
            m["types"] = sorted(set(m["types"]) | {e["type"]}, key=VOICE_ORDER.index)
            m["details_by_type"][e["type"]] = e.get("details", {})
            m["parts"].append(e)
        else:
            merged.append({"start_s": e["start_s"], "end_s": e["end_s"], "source": e["source"], "types": [e["type"]],
                           "details_by_type": {e["type"]: e.get("details", {})}, "parts": [e]})
    for m in merged:
        m["type"] = m["types"][0]
        m["duration_s"] = round(m["end_s"] - m["start_s"], 2)
        m["details"] = m["details_by_type"]
    return merged


def build_evidence(report, video, audio, cfg, out_dir):
    """Write the evidence files and return {group: {label, items: [...]}} (items sorted by start time)."""
    samples = (video or {}).get("samples", [])
    evidence = {g: {"label": GROUP_LABELS[g], "items": []} for g in GROUPS}
    voice_events = [e for e in report["timeline"] if GROUP_OF.get(e["type"]) == "voice"]
    merged_voice = merge_voice_events(voice_events, cfg["evidence"].get("voice_merge_gap_s", 0.0))
    others = [e for e in report["timeline"] if GROUP_OF.get(e["type"]) not in (None, "voice")]
    for event in sorted(others + merged_voice, key=lambda e: (e["start_s"], e["type"])):
        group = GROUP_OF.get(event["type"])
        folder = out_dir / "evidence" / group
        folder.mkdir(parents=True, exist_ok=True)
        idx = len(evidence[group]["items"]) + 1
        item = {"id": f"{group}-{idx:03d}", "type": event["type"], "source": event["source"],
                "start_s": event["start_s"], "end_s": event["end_s"], "duration_s": event["duration_s"]}
        if "types" in event:                     # merged voice violation
            item["types"] = event["types"]
        if event["source"] == "video":
            item.update(_video_item(group, event, samples, cfg, folder, idx))
        else:
            item.update(_audio_item(event, audio, cfg, folder, idx))
        if item.get("file"):
            for part in event.get("parts", [event]):
                part["evidence"] = item["file"]  # also linked from the timeline events
        evidence[group]["items"].append(item)
    for group, data in evidence.items():
        if data["items"]:
            (out_dir / "evidence" / group / "metadata.json").write_text(
                json.dumps({"group": group, "items": data["items"]}, indent=2, default=float))
    return evidence


# ---------------------------------------------------------------------------- counters + verdict
def limit_text(limit):
    if limit["basis"] == "percent":
        return f"{limit['value']:g}% of session"
    return "any detection" if limit["value"] == 0 else f"{limit['value']} events"


def counter(group, count, pct, limit):
    """One group's counter (LIM2): percent basis compares the session share, count basis the event count."""
    value = pct if limit["basis"] == "percent" else count
    return {"count": count, "percentage": round(pct, 2), "basis": limit["basis"], "limit": limit["value"],
            "limit_text": limit_text(limit), "threshold": limit["value"], "exceeded": value > limit["value"]}


def counters(report, cfg):
    lim = cfg["evidence"]["limits"]
    pct = report.get("group_percentages") or {}
    return {g: counter(g, len(report["evidence"][g]["items"]), pct.get(g, 0.0), lim[g]) for g in GROUPS}


def _exceeded_text(c):
    if c.get("basis") == "percent":
        return f"{c['percentage']:.1f}% > {c['limit_text']}"
    return f"{c['count']} event(s), limit {c.get('limit_text', c.get('threshold'))}"


def apply_session_violation(report, counts):
    exceeded = [g for g, c in counts.items() if c["exceeded"]]
    report["evidence_counters"] = counts
    report["session_violation"] = {"violated": bool(exceeded), "groups_exceeded": exceeded}
    if not exceeded:
        return
    reason = "Evidence threshold exceeded: " + ", ".join(
        f"{GROUP_LABELS[g].lower()} ({_exceeded_text(counts[g])})" for g in exceeded)
    v = report["verdict"]
    if v["label"] == "VALID":          # a violated session is never VALID (flow.txt §C.6)
        v.update(label="SUSPICIOUS", headline=reason, reasons=[reason])
    else:
        v["reasons"].append(reason)


# ---------------------------------------------------------------------------- upload
def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))[:120] or "x"


def store_upload(session_id, group, summary, metadata, files):
    """Evidence store used by POST /api/evidence/upload and by the local (url = None) upload. files: [(name, bytes)]."""
    folder = UPLOAD_DIR / _safe(session_id) / _safe(group)
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for name, data in files:
        safe = _safe(Path(name).name)
        (folder / safe).write_bytes(data)
        names.append(safe)
    manifest = {"session_id": session_id, "group": group, "summary": summary, "metadata": metadata,
                "files": names, "received": time.strftime("%Y-%m-%d %H:%M:%S")}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    return folder


def upload_evidence(report, out_dir, cfg):
    """Upload every exceeded group's package (flow.txt §C.6). Returns {group: result}."""
    uc = cfg["evidence_upload"]
    counts = report.get("evidence_counters", {})
    session_id = report["meta"].get("session_id")
    summary = {"session_id": session_id, "candidate": report["meta"].get("candidate"),
               "verdict": report["verdict"]["label"], "score": report["verdict"]["score"],
               "headline": report["verdict"]["headline"], "counters": counts,
               "violated": report.get("session_violation", {}).get("violated", False)}
    results = {}
    for group, c in counts.items():
        if not c["exceeded"]:
            results[group] = {"status": "skipped", "reason": "limit not exceeded"}
            continue
        if not uc.get("enabled", True):
            results[group] = {"status": "skipped", "reason": "upload disabled in config"}
            continue
        items = report["evidence"][group]["items"]
        files = [(Path(it["file"]).name, (out_dir / it["file"]).read_bytes())
                 for it in items if it.get("file") and (out_dir / it["file"]).exists()]
        target = uc.get("url") or "local evidence store"
        try:
            if uc.get("url"):
                import requests
                mime = {".jpg": "image/jpeg", ".wav": "audio/wav"}
                resp = requests.post(
                    uc["url"], timeout=uc.get("timeout_s", 30),
                    data={"session_id": session_id, "group": group, "summary": json.dumps(summary, default=float),
                          "metadata": json.dumps(items, default=float)},
                    files=[("files", (name, data, mime.get(Path(name).suffix, "application/octet-stream")))
                           for name, data in files])
                resp.raise_for_status()
                results[group] = {"status": "uploaded", "target": target, "http_status": resp.status_code,
                                  "files": len(files), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            else:
                folder = store_upload(session_id, group, summary, items, files)
                results[group] = {"status": "uploaded", "target": f"{target}: {folder}", "files": len(files),
                                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        except Exception as exc:  # network / server errors are reported, never fatal for the report
            results[group] = {"status": "failed", "target": target, "error": str(exc)}
    return results


def finalize_evidence(report, video, audio, cfg, out_dir):
    """Build evidence, counters, session violation (mutates report). Upload is done after report.json is written."""
    report["evidence"] = build_evidence(report, video, audio, cfg, out_dir)
    apply_session_violation(report, counters(report, cfg))
