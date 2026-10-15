"""JSON report + console summary (flow.txt §C.4)."""

import json
import platform
import time
import uuid

import numpy as np

import config as C
from aggregation.aggregate import aggregate


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes, bytearray)):
        return None
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def _limitations(core, video, audio):
    """FUS1: what the result cannot say (missing checks, model fallbacks, uncalibrated scores)."""
    out = ["Detector scores (YOLO, YAMNet, similarity, liveness) are not calibrated probabilities; the risk score is "
           "a rule-based summary of validated events, not a probability of misconduct."]
    missing = (core.get("assessment") or {}).get("missing_checks") or []
    if missing:
        out.append("Not checked in this session: " + ", ".join(m.lower().replace("_", " ") for m in missing) + ".")
    if video and video.get("gaze_fallback"):
        out.append("Gaze used the fallback method: " + video["gaze_fallback"])
    if video and video.get("antispoof_error"):
        out.append("Liveness check unavailable: " + video["antispoof_error"])
    if audio and audio.get("status") == "ok":
        out.append("Audio quality is only measured for speaker decisions (speech fraction); whispering and sound events "
                   "have no per-event quality measure.")
    return out


def build_report(video, audio, cfg, meta=None):
    core = aggregate(video, audio, cfg)
    warnings = []
    if video and not video.get("baseline"):
        warnings.append("Head-pose/gaze baseline was not calibrated: pose and gaze were not evaluated.")
    elif video and video["baseline"].get("fallback"):
        warnings.append("Calibration used the fallback (no frontal, verified sample in the first 10 s).")
    pcal = ((video or {}).get("head_pose") or {}).get("calibration") or {}
    if pcal.get("adaptive") and pcal.get("confidence") == "reduced":   # HP3
        warnings.append("HEAD_POSE_CALIBRATION_REDUCED: fixed head-pose thresholds were used ("
                        + "; ".join(pcal.get("problems", [])) + ").")
    if video and video.get("gaze_fallback"):            # GZ3: L2CS-Net could not be loaded
        warnings.append(video["gaze_fallback"])
    if video and video.get("antispoof_error"):          # FAS1: anti-spoofing could not be loaded
        warnings.append(video["antispoof_error"])
    if video and video.get("samples"):                  # DET2: dark video -> detections unreliable
        lc = cfg.get("lighting", {})
        br = [s["brightness"] for s in video["samples"] if s.get("brightness") is not None]
        if br and lc and sum(b < lc["low_mean"] for b in br) >= lc["report_frac"] * len(br):
            warnings.append(f"LOW_LIGHT: the camera image was dark (median brightness {float(np.median(br)):.0f}/255, "
                            f"limit {lc['low_mean']:.0f}) in {100 * sum(b < lc['low_mean'] for b in br) / len(br):.0f}% "
                            "of the samples; phones, partly visible persons and gaze are detected less reliably.")
    a_status = (audio or {}).get("status", "unavailable")
    if a_status != "ok":
        warnings.append({"unavailable": "AUDIO_UNAVAILABLE: no audio track / no microphone.",
                         "digital_silence": "AUDIO_SOURCE_SILENT: the audio is silent (muted or blocked microphone)."}
                        .get(a_status, a_status))
    elif not (audio or {}).get("voiceprint"):
        warnings.append("No enrolled voiceprint: VOICE_MISMATCH was not evaluated.")

    meta = dict(meta or {})
    meta.setdefault("session_id", uuid.uuid4().hex[:12])   # used for the evidence upload (flow.txt §C.6)
    report = {
        "meta": {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "profile": cfg.get("profile"),
                 "python": platform.python_version(), **meta},
        "verdict": {"label": core["verdict"], "headline": core["reasons"][0] if core["reasons"] else "",
                    "reasons": core["reasons"], "score": core["score"],
                    "score_explanation": core["score_explanation"],
                    # BUF1: VALID / SUSPICIOUS / INVALID stay for the frontend; coverage is reported separately
                    "assessment_status": (core.get("assessment") or {}).get("status", "COMPLETE"),
                    "review_priority": (core.get("assessment") or {}).get("review_priority")},
        "assessment": core.get("assessment"),
        # FUS1: every event record (confirmed, uncertain, filtered, unavailable), chronological, with stable ids;
        # "timeline" keeps the confirmed + uncertain ones (same objects) for older readers
        "events": core.get("events"),
        "fusion": core.get("fusion"),
        "limitations": _limitations(core, video, audio),
        "per_violation": core["per_violation"],
        "group_percentages": core["group_percentages"],
        "timeline": core["timeline"],
        "video": None if not video else {
            **video.get("video", {}), "n_samples": len(video["samples"]), "sample_rate_hz": video["rate_hz"],
            "baseline": video["baseline"], "calibration": video["calibration"],
            "gaze_method": video.get("gaze_method"), "antispoof": video.get("antispoof"),
            "head_pose": video.get("head_pose")},          # HP3 adaptive thresholds audit
        "audio": None if audio is None else {k: audio.get(k) for k in
                                             ("status", "duration_s", "speech_s", "vad_engine", "voiceprint", "segments",
                                              "vad_rejected")},
        "warnings": warnings,
        "thresholds": {k: v for k, v in cfg.items() if k != "profile"},
    }
    if "legacy" in core:
        report["legacy_v1"] = core["legacy"]
    return report


def save_report(report, video, out_dir=None, name="report", audio=None):
    """
    Writes the grouped evidence package (flow.txt §C.5), applies counters / session violation (§C.6),
    writes report.json, uploads the evidence of exceeded groups and stores the upload results in the report.
    Returns the report path.
    """
    from aggregation import evidence as ev
    cfg = report["thresholds"]                      # the full config used for this run
    out = out_dir or (C.REPORTS_DIR / f"{report['meta'].get('candidate', 'candidate')}_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    ev.finalize_evidence(report, video, audio, cfg, out)
    path = out / f"{name}.json"
    path.write_text(json.dumps(report, indent=2, default=_jsonable))
    report["evidence_upload"] = ev.upload_evidence(report, out, cfg)
    path.write_text(json.dumps(report, indent=2, default=_jsonable))
    try:                                            # PDF version next to the JSON (flow.txt §C.4, PDF1)
        from aggregation.pdf_report import write_report_pdf
        write_report_pdf(path)
    except Exception as exc:                        # never break the JSON report because of the PDF
        print(f">>> PDF report not written: {exc}")
    return path


def console_summary(report):
    v = report["verdict"]
    lines = ["", "=" * 60, "FINAL RESULT", "=" * 60]
    vid = report.get("video") or {}
    if vid:
        b = vid.get("baseline") or {}
        lines.append(f"Video: {vid.get('duration_s', 0):.1f} s, {vid.get('n_samples')} samples @ "
                     f"{vid.get('sample_rate_hz', 0):.2f}/s")
        if b:
            lines.append(f"Baseline head pose: yaw={b['yaw']:.1f} pitch={b['pitch']:.1f} roll={b['roll']:.1f} "
                         f"({b['n_samples']} calibration samples)")
        hp = vid.get("head_pose") or {}
        if hp.get("calibration"):                    # HP3
            c, fin = hp["calibration"], hp["final_thresholds_deg"]
            ini = c["initial_thresholds_deg"]
            lines.append(f"Head-pose thresholds ({c['status']}, confidence {c['confidence']}): "
                         f"yaw/pitch/roll {ini['yaw']:.1f}/{ini['pitch']:.1f}/{ini['roll']:.1f} deg at calibration, "
                         f"{fin['yaw']:.1f}/{fin['pitch']:.1f}/{fin['roll']:.1f} deg at the end"
                         + (" (fixed fallback)" if c["fallback_thresholds_used"] else ""))
    a = report.get("audio") or {}
    lines.append(f"Audio: {a.get('status', 'not analysed')}"
                 + (f", {a.get('duration_s', 0):.1f} s, speech {a.get('speech_s', 0):.1f} s, VAD {a.get('vad_engine')}"
                    if a.get("status") == "ok" else ""))
    lines.append("-" * 60)
    lines.append(f"{'Violation':22s}{'%':>8s}{'time s':>9s}{'events':>8s}  basis")
    for t, s in report["per_violation"].items():
        if not s["measurable"]:
            lines.append(f"{t:22s}{'n/a':>8s}{'':>9s}{'':>8s}  not measurable")
            continue
        basis = f"{s['numerator']}/{s['denominator']} {s['unit']}"
        lines.append(f"{t:22s}{s['percentage']:8.1f}{s['duration_s']:9.1f}{s['count']:8d}  {basis}")
    if report.get("fusion"):                         # FUS1 evidence families
        lines.append("-" * 60)
        lines.append("Evidence families (confirmed events, points, de-duplicated s):  " + "  ".join(
            f"{n} {f['confirmed_events']}/{f['points']:.1f}/{f['deduplicated_s']:.1f}"
            for n, f in report["fusion"]["families"].items() if f["confirmed_events"] or f["uncertain_events"]))
        a = report["assessment"]
        lines.append(f"Review: {a['status']} (priority {a['review_priority']}), events " +
                     ", ".join(f"{k} {v}" for k, v in a["event_counts"].items()))
    if report.get("assessment"):                     # BUF1 tolerance buffer
        lines.append("-" * 60)
        lines.append(f"{'Buffer':22s}{'raw':>6s}{'filt':>6s}{'merged':>8s}{'conf':>6s}{'unconf':>8s}  strength")
        for t, s in report["per_violation"].items():
            if s["measurable"] and s.get("merged_event_count"):
                lines.append(f"{t:22s}{s['raw_detection_count']:6d}{s['filtered_detection_count']:6d}"
                             f"{s['merged_event_count']:8d}{s['confirmed_event_count']:6d}"
                             f"{s['unconfirmed_event_count']:8d}  {s['evidence_strength']}"
                             + ("  REVIEW" if s.get("review_required") else ""))
        a = report["assessment"]
        lines.append(f"Assessment: {a['status']}" + (" - " + "; ".join(a["coverage"]["problems"])
                                                     if a["coverage"]["problems"] else ""))
    lines.append("-" * 60)
    for e in report["timeline"][:20]:
        tag = "" if e.get("status", "confirmed") == "confirmed" else "  (unconfirmed)"
        lines.append(f"  {e['start_s']:7.2f}-{e['end_s']:7.2f}s  {e['source']:5s} {e['type']}{tag}")
    if len(report["timeline"]) > 20:
        lines.append(f"  ... {len(report['timeline']) - 20} more events in the JSON report")
    for w in report["warnings"]:
        lines.append(f"WARNING: {w}")
    lines += ["=" * 60, f"VIOLATION SCORE: {v['score']} / 100", f"  {v.get('score_explanation', '')}",
              f"FINAL VERDICT: {v['label']} - {v['headline']}"]
    for r in v["reasons"][1:]:
        lines.append(f"  also: {r}")
    if "legacy_v1" in report:
        lines.append(f"(v1 logic: {report['legacy_v1']['verdict']})")
    if report.get("evidence_counters"):
        lines.append("-" * 60)
        lines.append("Evidence (events, % of session / limit):  " + "  ".join(
            f"{g} {c['count']}, {c.get('percentage', 0):.1f}% / {c.get('limit_text', c['threshold'])}"
            f"{' EXCEEDED' if c['exceeded'] else ''}" for g, c in report["evidence_counters"].items()))
        for g, r in (report.get("evidence_upload") or {}).items():
            if r["status"] != "skipped":
                lines.append(f"  upload {g}: {r['status']} -> {r.get('target')}" + (f" ({r['error']})" if r.get("error") else ""))
        if report.get("session_violation", {}).get("violated"):
            lines.append("SESSION MARKED AS VIOLATED")
    lines.append("=" * 60)
    return "\n".join(lines)
