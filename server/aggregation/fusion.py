"""
FUS1 multimodal evidence-fusion engine (config "fusion"): one explainable path from detector output to the verdict.

Lifecycle of an event (existing modules do the detection and temporal work; this module unifies and explains it):
  1. raw detection         video/pipeline.py, audio/monitor.py   per-sample / per-window raw flags + measurements
  2. observation quality   pipeline (pose quality, brightness), monitor (speech fraction)       -> observation_quality
  3. filtered / accepted   detector persistence (filter_runs)     raw runs that never persist   -> status "filtered"
  4. merged                aggregation/events.py + buffer.merge_events (per-type merge gap)
  5. persistence rules     buffer: confirm_s + min_support per type
  6. confirmed / uncertain buffer                                  -> validation_status + validation_reason
  7. evidence fusion       this module: evidence families (priority de-duplication of correlated time),
                           relations between overlapping events, cross-modal corroboration
  8. score contribution    aggregate.score_violation (per type, unchanged model) split over the type's confirmed
                           events in proportion to their validated time, weighted by observation quality
  9. report                report.py (events, fusion summary, assessment), evidence.py (evidence refs)
Coverage gaps (no face, no audio, missing model) become "unavailable" records, never normal behaviour.

Detector confidence is only reported where the detector exposes a score (YOLO, YAMNet); it is uncalibrated, is not
used for scoring, and is never a probability of cheating.
"""

import config as C
from aggregation.denominators import FACE_DEPENDENT, counts_toward, observed
from aggregation.events import events_from_flags, video_slots

SCHEMA_VERSION = "fus1"
# detectors that expose a classifier score per detection (uncalibrated)
CONFIDENCE_KIND = {C.PROHIBITED_OBJECT: "yolo_class_score_uncalibrated",
                   C.SUSPICIOUS_SOUND: "yamnet_class_score_uncalibrated",
                   C.WHISPERING: "yamnet_class_score_uncalibrated"}
# what the existing per-event "peak" measures for the other detectors (kept as metadata, not as confidence)
PEAK_MEANING = {C.IDENTITY_MISMATCH: "max face dissimilarity (1 - cosine similarity)",
                C.FACE_SPOOF: "max spoof score (1 - real-face score)",
                C.MULTIPLE_PERSONS: "max persons in view",
                C.HEAD_POSE_DEVIATION: "max absolute head angle vs baseline (deg)",
                C.VOICE_MISMATCH: "max voice dissimilarity (1 - similarity)",
                C.SPEECH_DETECTED: "max speech fraction of a window"}
SPEAKER_TYPES = {C.VOICE_MISMATCH, C.MULTIPLE_SPEAKERS}


# ---------------------------------------------------------------------------- families
def family_of(vtype, cfg):
    for name, fam in cfg["fusion"]["families"].items():
        if vtype in fam["members"]:
            return name
    return None


def correlated_types(vtype, cfg):
    """Higher-priority members of vtype's evidence family: their confirmed time is not scored again for vtype."""
    fam = cfg["fusion"]["families"].get(family_of(vtype, cfg) or "", {})
    if fam.get("rule") != "priority_dedup":
        return []
    members = fam["members"]
    return members[:members.index(vtype)]


def check_required(stats, cov, cfg):
    """Required checks that could not be measured make the assessment INCONCLUSIVE (coverage problem)."""
    missing = [t for t in cfg["fusion"]["required_checks"] if t in stats and not stats[t]["measurable"]]
    if missing:
        cov["problems"].append("required checks could not be performed: " + ", ".join(m.lower() for m in missing))
        cov["status"] = "INCONCLUSIVE"
    cov["missing_checks"] = [t for t, s in stats.items() if not s["measurable"]]


# ---------------------------------------------------------------------------- normalisation
def _video_quality(samples, idx, vtype, cfg):
    """good / low from the samples behind the event (dark frames, unreliable head-pose landmarks)."""
    if not idx:
        return None
    low_mean = cfg.get("lighting", {}).get("low_mean")
    bad = 0
    for i in idx:
        s = samples[i]
        dark = low_mean is not None and s.get("brightness") is not None and s["brightness"] < low_mean
        pose_issue = vtype in FACE_DEPENDENT and bool(s.get("pose_quality_issues"))
        bad += dark or pose_issue
    return "low" if bad / len(idx) >= cfg["fusion"]["low_quality_frac"] else "good"


def _audio_quality(windows, idx, vtype, cfg):
    """Speaker decisions need clean speech: mean speech fraction of the event windows. Other audio: not measured."""
    fracs = [windows[i]["speech_frac"] for i in idx if windows[i].get("speech_frac") is not None]
    if vtype not in SPEAKER_TYPES or not fracs:
        return None
    frac = sum(fracs) / len(fracs)
    return "good" if frac >= cfg["fusion"]["min_speech_frac"] else "low"


def _record(e, *, event_id, status, reason, severity, family, raw_n, valid_n, slot_s, quality, metadata):
    """Common event schema (FUS1). Existing keys stay: type (= event_type), source (= modality), start_s / end_s
    (= start / end time), duration_s (= duration_seconds)."""
    peak = e.get("peak")
    e.update(event_id=event_id, modality=e["source"], validation_status=status, validation_reason=reason,
             severity=severity, family=family,
             confidence=peak if e["type"] in CONFIDENCE_KIND and peak is not None else None,
             confidence_kind=CONFIDENCE_KIND.get(e["type"]) if peak is not None else None,
             observation_quality=quality, raw_detection_count=raw_n, validated_detection_count=valid_n,
             persistence_s=round(valid_n * slot_s, 2), evidence_refs=e.get("evidence_refs", []),
             score_contribution=0.0, related_event_ids=[], metadata=metadata)
    return e


def normalize(timeline, stats, video, audio, cfg):
    """Gives every merged event (confirmed / unconfirmed, from buffer.apply) the common schema, in place."""
    samples = (video or {}).get("samples") or []
    rate = (video or {}).get("rate_hz") or 1.0
    windows = (audio or {}).get("windows") or []
    sev = cfg["aggregation"]["severity"]
    counters = {}
    for e in sorted(timeline, key=lambda e: (e["start_s"], e["type"])):
        vtype = e["type"]
        counters[vtype] = counters.get(vtype, 0) + 1
        if e["source"] == "video":
            idx = list(range(e["first_index"], e["first_index"] + e["n"]))
            idx = [i for i in idx if observed(samples[i], vtype)]   # PERF2: reused YOLO results are not observations
            raw_n = sum(1 for i in idx if samples[i].get("raw", {}).get(vtype, samples[i]["flags"].get(vtype)))
            valid_n = sum(1 for i in idx if samples[i]["flags"].get(vtype))
            quality, slot = _video_quality(samples, [i for i in idx if samples[i]["flags"].get(vtype)], vtype, cfg), 1.0 / rate
        else:
            mid = lambda w: (w["slot"][0] + w["slot"][1]) / 2  # noqa: E731
            idx = [i for i, w in enumerate(windows) if e["start_s"] - 1e-6 <= mid(w) <= e["end_s"] + 1e-6]
            raw_n = sum(1 for i in idx if windows[i].get("raw", {}).get(vtype, windows[i].get("flags", {}).get(vtype)))
            valid_n = sum(1 for i in idx if windows[i].get("flags", {}).get(vtype))
            from aggregation.aggregate import SLOT_S
            quality, slot = _audio_quality(windows, idx, vtype, cfg), SLOT_S
        confirmed = e.get("status", "confirmed") == "confirmed"
        pol = e.get("buffer_policy") or cfg["buffer"]["policies"].get(vtype, {})
        reason = (f"confirmed: {e['duration_s']:.2f} s with {valid_n} validated observation(s) "
                  f"(needs >= {pol.get('confirm_s')} s and >= {pol.get('min_support')})" if confirmed
                  else f"uncertain: {e.get('buffer_reason', 'below the confirmation rule')}")
        if confirmed and quality == "low":
            reason += "; observation quality low (score weighted down, no critical override)"
        meta = {k: e[k] for k in ("details",) if e.get(k)}
        if e.get("peak") is not None and vtype in PEAK_MEANING:
            meta["peak"], meta["peak_meaning"] = e["peak"], PEAK_MEANING[vtype]
        if e.get("merged_from", 1) > 1:
            meta["merged_runs"] = e["merged_from"]
        _record(e, event_id=f"{vtype.lower()}-{counters[vtype]:03d}", status="confirmed" if confirmed else "uncertain",
                reason=reason, severity=sev.get(vtype), family=family_of(vtype, cfg), raw_n=raw_n, valid_n=valid_n,
                slot_s=slot, quality=quality, metadata=meta)
    return timeline


# ---------------------------------------------------------------------------- scoring split
def allocate_points(stats, events, parts, cfg):
    """
    Per-event score contributions. The per-type points (aggregate.score_violation, unchanged model: share of time,
    saturation, critical floor) are split over the type's confirmed events in proportion to their validated time,
    then weighted by observation quality (fusion.quality_weights). Returns the adjusted per-type points.
    """
    qw = cfg["fusion"]["quality_weights"]
    out = dict(parts)
    for vtype, pts in parts.items():
        evs = [e for e in events if e["type"] == vtype and e["validation_status"] == "confirmed"]
        if pts <= 0 or not evs:
            continue
        total_t = sum(e["persistence_s"] for e in evs) or float(len(evs))
        new = 0.0
        for e in evs:
            share = (e["persistence_s"] or 1.0) / total_t
            w = qw.get(e["observation_quality"] or "unknown", 1.0)
            e["score_contribution"] = round(pts * share * w, 2)
            e["score_basis"] = (f"{100 * share:.0f}% of the {pts:.1f} {vtype.lower().replace('_', ' ')} points "
                                f"(validated time {e['persistence_s']:.1f} s)" + (f" x quality weight {w}" if w != 1 else ""))
            new += pts * share * w
        if abs(new - pts) > 1e-6:
            st = stats[vtype]
            st["score_explanation"] += f" Weighted by observation quality: {new:.1f} of {pts:.1f} points."
            st["score"] = round(new, 1)
        out[vtype] = new
    return out


# ---------------------------------------------------------------------------- filtered + unavailable records
def _runs(slots, mask, gap_s, min_s):
    """[(start, end, n)] runs of True in mask (merged across gaps <= gap_s, at least min_s long)."""
    evs = events_from_flags(slots, mask, "x", "x", gap_s=gap_s)
    return [(e["start_s"], e["end_s"], e["n"]) for e in evs if e["duration_s"] >= min_s]


def filtered_events(video, audio, stats, cfg):
    """Raw detection runs that the detectors' persistence rules rejected (audit trail; never scored)."""
    fc = cfg["fusion"]
    out = []
    samples = (video or {}).get("samples") or []
    if samples:
        slots = video_slots(samples, video["rate_hz"])
        for vtype in C.VIDEO_TYPES:
            if not stats.get(vtype, {}).get("measurable"):
                continue
            mask = [counts_toward(s, vtype) and bool(s.get("raw", {}).get(vtype)) and not s.get("flags", {}).get(vtype)
                    for s in samples]
            out += [(vtype, "video", a, b, n) for a, b, n in _runs(slots, mask, 0.05, 0.0)][:fc["max_filtered_per_type"]]
    windows = (audio or {}).get("windows") or []
    if windows and (audio or {}).get("status") == "ok":
        slots = [w["slot"] for w in windows]
        for vtype in C.AUDIO_TYPES:
            if not stats.get(vtype, {}).get("measurable"):
                continue
            mask = [bool(w.get("raw", {}).get(vtype)) and not w.get("flags", {}).get(vtype) for w in windows]
            out += [(vtype, "audio", a, b, n) for a, b, n in _runs(slots, mask, 0.05, 0.0)][:fc["max_filtered_per_type"]]
    recs, counters = [], {}
    for vtype, src, a, b, n in sorted(out, key=lambda r: (r[2], r[0])):
        counters[vtype] = counters.get(vtype, 0) + 1
        e = {"type": vtype, "source": src, "start_s": a, "end_s": b, "duration_s": round(b - a, 2), "n": n}
        recs.append(_record(e, event_id=f"f-{vtype.lower()}-{counters[vtype]:03d}", status="filtered",
                            reason=f"{n} raw detection(s) did not persist long enough for the detector to flag them",
                            severity=cfg["aggregation"]["severity"].get(vtype), family=family_of(vtype, cfg),
                            raw_n=n, valid_n=0, slot_s=0.0, quality=None, metadata={}))
    return recs


def unavailable_events(video, audio, stats, cfg):
    """Coverage gaps as explicit records: face checks impossible, audio missing, voiceprint or model missing."""
    fc = cfg["fusion"]
    recs = []
    samples = (video or {}).get("samples") or []
    v_end = samples[-1]["t"] + 1.0 / video["rate_hz"] if samples else 0.0

    def add(kind, a, b, reason, checks):
        e = {"type": kind, "source": "coverage", "start_s": round(a, 2), "end_s": round(b, 2),
             "duration_s": round(b - a, 2)}
        recs.append(_record(e, event_id=f"u-{kind.lower()}-{len(recs) + 1:03d}", status="unavailable", reason=reason,
                            severity=None, family=None, raw_n=0, valid_n=0, slot_s=0.0, quality=None,
                            metadata={"checks": checks}))

    if samples:
        slots = video_slots(samples, video["rate_hz"])
        mask = [not counts_toward(s, C.IDENTITY_MISMATCH) for s in samples]
        for a, b, _ in _runs(slots, mask, fc["gap_merge_s"], fc["min_gap_s"]):
            add("FACE_CHECKS_UNAVAILABLE", a, b, "no visible, verifiable face: identity, liveness, head pose and gaze "
                "could not be assessed (FACE_NOT_VISIBLE covers the absence itself)",
                [C.IDENTITY_MISMATCH, C.FACE_SPOOF, C.HEAD_POSE_DEVIATION, C.GAZE_AWAY])
        if video.get("antispoof_error"):
            add("LIVENESS_MODEL_UNAVAILABLE", 0.0, v_end, video["antispoof_error"], [C.FACE_SPOOF])
    a_status = (audio or {}).get("status", "unavailable")
    a_end = (audio or {}).get("duration_s") or v_end
    if a_status != "ok":
        add("AUDIO_UNAVAILABLE", 0.0, v_end, f"audio source {a_status.replace('_', ' ')}: no audio check was possible",
            list(C.AUDIO_TYPES))
    elif not stats.get(C.VOICE_MISMATCH, {}).get("measurable"):
        add("VOICE_CHECK_UNAVAILABLE", 0.0, a_end, "no enrolled voiceprint: the voice was not compared",
            [C.VOICE_MISMATCH])
    return recs


# ---------------------------------------------------------------------------- relations + summary
def _overlap(a, b, gap):
    return min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]) >= -gap


def relate(events, cfg):
    """Links overlapping confirmed / uncertain events: same family (correlated), corroborating (independent
    modalities in a configured pair) or co-occurring. Individual events are kept; only links are added."""
    fc = cfg["fusion"]
    pairs = {frozenset(p) for p in fc["corroboration_pairs"]}
    live = [e for e in events if e["validation_status"] in ("confirmed", "uncertain")]
    rels = []
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            if a["type"] == b["type"] or not _overlap(a, b, fc["relation_gap_s"]):
                continue
            if a["family"] and a["family"] == b["family"]:
                kind = "correlated_same_family"
            elif frozenset((a["type"], b["type"])) in pairs:
                kind = "corroborating"
            else:
                kind = "co_occurring"
            ov = max(0.0, min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]))
            rels.append({"a": a["event_id"], "b": b["event_id"], "kind": kind, "overlap_s": round(ov, 2)})
            a["related_event_ids"].append(b["event_id"])
            b["related_event_ids"].append(a["event_id"])
    return rels


def families_summary(events, stats, cfg):
    out = {}
    for name, fam in cfg["fusion"]["families"].items():
        mem = [t for t in fam["members"] if t in stats]
        evs = [e for e in events if e["type"] in mem]
        pts = round(sum(e["score_contribution"] for e in evs), 2)
        out[name] = {"label": fam["label"], "members": mem, "rule": fam.get("rule", "independent"),
                     "confirmed_events": sum(e["validation_status"] == "confirmed" for e in evs),
                     "uncertain_events": sum(e["validation_status"] == "uncertain" for e in evs),
                     "filtered_events": sum(e["validation_status"] == "filtered" for e in evs),
                     "deduplicated_s": round(sum(stats[t].get("correlated_duration_s", 0.0) for t in mem), 2),
                     "points": pts, "contributed": pts > 0,
                     "measurable": any(stats[t]["measurable"] for t in mem)}
    return out


def assess(stats, events, families, rels, cov, score, label, cfg):
    """assessment status (COMPLETE / REVIEW_REQUIRED / INCONCLUSIVE) and review priority, separate from the verdict."""
    measured = [s for s in stats.values() if s["measurable"]]
    review = [t for t, s in stats.items() if s.get("review_required")]
    if cov["status"] == "INCONCLUSIVE":
        status = "INCONCLUSIVE"
    elif label != "VALID" or review:
        status = "REVIEW_REQUIRED"
    else:
        status = "COMPLETE"
    sev = cfg["aggregation"]["severity"]
    heavy = cfg["fusion"]["priority_severities"]
    fams = {e["family"] for e in events if e["validation_status"] == "confirmed" and sev.get(e["type"]) in heavy}
    corroborated = [r for r in rels if r["kind"] == "corroborating"]
    if label == "INVALID" or len(fams) >= cfg["fusion"]["high_priority_families"] or corroborated:
        priority = "high"
    elif status != "COMPLETE":
        priority = "medium"
    else:
        priority = "low"
    count = lambda st: sum(1 for e in events if e["validation_status"] == st)  # noqa: E731
    strongest = max((s.get("evidence_strength", "none") for s in measured),
                    key=["none", "weak", "moderate", "strong"].index, default="none")
    return {"status": status, "review_priority": priority, "retest_recommended": status == "INCONCLUSIVE",
            "risk_score": score, "verdict": label, "evidence_strength": strongest, "review_required": review,
            "independent_families_confirmed": sorted(f for f in fams if f),
            "corroborations": len(corroborated),
            "contributing_families": [n for n, f in families.items() if f["contributed"]],
            "event_counts": {s: count(s) for s in ("confirmed", "uncertain", "filtered", "unavailable")},
            "confirmed_events": count("confirmed"), "unconfirmed_events": count("uncertain"),
            "filtered_detections": sum(s.get("filtered_detection_count", 0) for s in measured),
            "missing_checks": cov.get("missing_checks", []), "coverage": cov,
            "confidence_note": "Detector scores are not calibrated probabilities and were not used for scoring.",
            "note": "The risk score summarises validated detector events; it is not proof of misconduct. "
                    "A reviewer must read it together with the exam rules and the evidence."}


def finalize(timeline, stats, video, audio, cfg, cov, score, label):
    """All event records (chronological), relations, family summary and the assessment."""
    events = timeline + filtered_events(video, audio, stats, cfg) + unavailable_events(video, audio, stats, cfg)
    events.sort(key=lambda e: (e["start_s"], e["validation_status"] != "confirmed", e["type"]))
    rels = relate(events, cfg)
    fams = families_summary(events, stats, cfg)
    return {"events": events,
            "fusion": {"schema_version": SCHEMA_VERSION, "families": fams, "relations": rels},
            "assessment": assess(stats, events, fams, rels, cov, score, label, cfg)}
