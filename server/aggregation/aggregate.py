"""Per-type statistics, weighted score and verdict over video + audio (flow.txt PART II-C)."""

import config as C
from aggregation.denominators import counts_toward
from aggregation.events import events_from_flags, video_slots

SLOT_S = 0.48


def video_stats(video, cfg):
    samples, rate = video["samples"], video["rate_hz"]
    slots = video_slots(samples, rate) if samples else []
    stats, events = {}, []
    for vtype in C.VIDEO_TYPES:
        member = [counts_toward(s, vtype) and vtype in s.get("flags", {}) for s in samples]
        den = sum(member)
        flags = [m and s["flags"][vtype] for m, s in zip(member, samples)]
        num = sum(flags)
        evs = events_from_flags(slots, flags, vtype, "video",
                                peaks=[_video_peak(s, vtype) for s in samples],
                                details=lambda idx, vt=vtype: _video_details(samples, idx, vt))
        for e in evs:  # exact first sample of the run (rounded start_s cannot be used to look it up)
            e["sample_t"] = samples[e["first_index"]]["t"]
        events += evs
        stats[vtype] = {"source": "video", "unit": "samples", "numerator": num, "denominator": den,
                        "percentage": round(100.0 * num / den, 2) if den else 0.0,
                        "duration_s": round(num / rate, 2), "count": len(evs), "measurable": den > 0,
                        "longest_event_s": max((e["duration_s"] for e in evs), default=0.0)}
    return stats, events


def _video_peak(s, vtype):
    if vtype == C.IDENTITY_MISMATCH:
        return None if s.get("identity_score") is None else round(1 - s["identity_score"], 3)
    if vtype == C.PROHIBITED_OBJECT:  # v2 stores {name: confidence}
        return max(s.get("objects", {}).values(), default=None)
    if vtype == C.MULTIPLE_PERSONS:
        return s.get("persons")
    if vtype == C.FACE_SPOOF:
        return None if s.get("spoof_score") is None else round(1 - s["spoof_score"], 3)
    if vtype == C.HEAD_POSE_DEVIATION and "rel_yaw" in s:
        return round(max(abs(s["rel_yaw"]), abs(s["rel_pitch"]), abs(s["rel_roll"])), 1)
    return None


def _video_details(samples, idx, vtype):
    ss = [samples[i] for i in idx]
    if vtype == C.PROHIBITED_OBJECT:
        names = sorted({k for s in ss for k in s.get("objects", {})})
        return {"objects": names}
    if vtype == C.MULTIPLE_PERSONS:
        return {"max_persons": max(s.get("persons", 0) for s in ss)}
    if vtype == C.IDENTITY_MISMATCH:
        sc = [s["identity_score"] for s in ss if s.get("identity_score") is not None]
        return {"min_score": round(min(sc), 3) if sc else None}
    if vtype == C.HEAD_POSE_DEVIATION:
        keys = ("rel_yaw", "rel_pitch", "rel_roll")
        return {k: round(max((s[k] for s in ss if k in s), key=abs), 1) for k in keys if any(k in s for s in ss)}
    if vtype == C.FACE_SPOOF:
        sc = [s["spoof_score"] for s in ss if s.get("spoof_score") is not None]
        return {"min_real_score": round(min(sc), 3) if sc else None}
    if vtype == C.GAZE_AWAY:
        labels = [s["gaze"]["label"] for s in ss if s.get("gaze")]
        return {"direction": max(set(labels), key=labels.count) if labels else None}
    return {}


def audio_stats(audio, cfg):
    stats = {}
    ok = audio is not None and audio.get("status") == "ok"
    dur = audio["duration_s"] if ok else 0.0
    for vtype in C.AUDIO_TYPES:
        measurable = ok and (vtype != C.VOICE_MISMATCH or audio.get("voiceprint"))
        if not measurable:
            stats[vtype] = {"source": "audio", "unit": "seconds", "numerator": 0.0, "denominator": dur,
                            "percentage": 0.0, "duration_s": 0.0, "count": 0, "measurable": False,
                            "longest_event_s": 0.0}
            continue
        n = sum(1 for w in audio["windows"] if w.get("flags", {}).get(vtype))
        evs = [e for e in audio["events"] if e["type"] == vtype]
        stats[vtype] = {"source": "audio", "unit": "seconds", "numerator": round(n * SLOT_S, 2),
                        "denominator": dur, "percentage": round(100.0 * n * SLOT_S / dur, 2) if dur else 0.0,
                        "duration_s": round(n * SLOT_S, 2), "count": len(evs), "measurable": True,
                        "longest_event_s": max((e["duration_s"] for e in evs), default=0.0)}
    return stats, (audio or {}).get("events", []) if ok else []


def _nice(vtype):
    return vtype.lower().replace("_", " ")


def _buffer_text(st):
    """One sentence on what the tolerance buffer did (BUF1)."""
    text = (f"{st['raw_detection_count']} raw detection(s), {st['filtered_detection_count']} filtered; "
            f"{st['merged_event_count']} merged event(s): {st['confirmed_event_count']} confirmed, "
            f"{st['unconfirmed_event_count']} unconfirmed ({st['confirmed_duration_s']:.1f} s confirmed")
    if st["correlated_duration_s"]:
        text += f", {st['correlated_duration_s']:.1f} s already scored by a correlated violation"
    if st["grace_applied_s"]:
        text += f", {st['grace_applied_s']:.1f} s tolerated by the grace budget"
    return text + ")."


def score_violation(vtype, st, ac):
    """Points of one violation (SC2) + a one-line explanation. Mutates st."""
    mx, sat = ac["max_points"][vtype], ac["saturation_pct"][vtype]
    sev = ac["severity"][vtype]
    st.update(max_points=mx, saturation_pct=sat, severity=sev, critical=sev == "critical")
    if not st["measurable"]:
        st.update(score=None, score_explanation=f"Not measured in this session - 0 of {mx} points.")
        return 0.0
    buffered = "scored_percentage" in st
    pct = st["scored_percentage"] if buffered else st["percentage"]
    if st["percentage"] <= 0:
        st.update(score=0.0, score_explanation=f"Did not occur - 0 of {mx} points.")
        return 0.0
    if buffered and pct <= 0:
        st.update(score=0.0, score_explanation=f"{_buffer_text(st)} Nothing left to score - 0 of {mx} points.")
        return 0.0
    frac = min(1.0, pct / sat)
    head = (f"{_buffer_text(st)} Scored {st['scored_duration_s']:.1f} s = {pct:.1f}% of the "
            f"{st['valid_observation_duration_s']:.1f} s monitored. " if buffered else
            f"{pct:.1f}% of {st['unit']} ({st['numerator']} of {st['denominator']}), {st['count']} event(s). ")
    text = (head + f"Full {mx} points at {sat}% or more, so " +
            (f"it reaches the cap: 100% of {mx}" if frac >= 1.0 else f"{pct:.1f}/{sat} = {100 * frac:.0f}% of {mx}"))
    pts = mx * frac
    text += f" = {pts:.1f}"
    weak = _weak_evidence(st)
    if st["critical"] and pts < mx * ac["critical_floor"] and not weak:
        pts = mx * ac["critical_floor"]
        text += (f", raised to {pts:.1f} because a critical violation scores at least "
                 f"{100 * ac['critical_floor']:.0f}% of its points once it occurs")
    elif st["critical"] and pts < mx * ac["critical_floor"]:
        text += f" (no critical floor: {weak})"
    st.update(score=round(pts, 1), score_explanation=f"{text}. Score {pts:.1f} of {mx} points ({sev} severity).")
    return pts


def _weak_evidence(st):
    """FUS1: why a critical rule (floor / override) must not apply, or None. Weak evidence or low observation quality
    sends the violation to review instead of letting it decide the verdict on its own."""
    if st.get("evidence_strength") == "weak":
        return "evidence strength is weak"
    if st.get("observation_quality") == "low":
        return "observation quality is low"
    return None


def _pct(st):
    """Share used by the overrides / review rules: confirmed time when the buffer ran, else flagged time."""
    return st.get("confirmed_percentage", st["percentage"])


def _longest(st):
    return st.get("longest_confirmed_event_s", st["longest_event_s"])


def score_and_verdict(stats, audio, cfg, cov=None, events=None):
    ac = cfg["aggregation"]
    parts = {vtype: score_violation(vtype, st, ac) for vtype, st in stats.items()}
    if events is not None:                      # FUS1: per-event contributions (quality-weighted)
        from aggregation.fusion import allocate_points
        parts = allocate_points(stats, events, parts, cfg)
    parts = list(parts.items())
    total = sum(p for _, p in parts)
    score = round(min(100.0, total), 1)
    scored = sorted(((v, p) for v, p in parts if p > 0), key=lambda x: -x[1])
    if scored:
        expl = (f"Overall score {score:.1f} / 100 = sum of the violation points ("
                + " + ".join(f"{_nice(v)} {p:.1f}" for v, p in scored) + ")"
                + (f", capped at 100 (raw {total:.1f})" if total > 100 else "") + ". ")
    else:
        expl = "Overall score 0 / 100: no violation scored any points. "
    expl += (f"Score >= {ac['invalid_score']:.0f} -> INVALID, >= {ac['suspicious_score']:.0f} -> SUSPICIOUS; "
             "critical violations (multiple persons, prohibited objects, identity, face spoofing, voice) weigh most, "
             "head pose little and eye gaze least.")

    reasons = []
    for vtype, pct_thr, ev_s, text in ac["critical"]:
        st = stats.get(vtype)
        if not st or not st["measurable"]:
            continue
        # identity keeps v1's strict-majority rule (> 50%); the others use >=. BUF1: confirmed events only
        pct, longest = _pct(st), _longest(st)
        pct_hit = pct > pct_thr if vtype == C.IDENTITY_MISMATCH else pct >= pct_thr
        if (pct_hit or longest >= ev_s) and not _weak_evidence(st):
            basis = "confirmed " if "confirmed_percentage" in st else ""
            reasons.append(f"{text} ({pct:.1f}% of {st['unit']}, longest {basis}event {longest:.1f} s)")
        elif pct_hit or longest >= ev_s:              # limits reached, but not by validated, well-observed evidence
            st["review_required"] = True
            st["override_withheld"] = f"critical override not applied: {_weak_evidence(st)}"
    if reasons:
        return score, "INVALID", reasons, expl

    soft = [f"{v} at {_pct(s):.1f}% (>= {s['saturation_pct']}%)" for v, s in stats.items()
            if s["measurable"] and not s["critical"] and _pct(s) >= s["saturation_pct"]]
    audio_status = (audio or {}).get("status", "unavailable")
    if score >= ac["invalid_score"]:
        return score, "INVALID", [f"Violation score {score} >= {ac['invalid_score']}"] + soft, expl
    review = []
    if score >= ac["suspicious_score"]:
        review.append(f"Violation score {score} >= {ac['suspicious_score']}")
    for v, s in stats.items():
        if not s["measurable"]:
            continue
        if "review_required" not in s:                     # buffer off (legacy profile): previous rule
            if s["critical"] and s["count"] > 0:
                review.append(f"{v}: {s['count']} event(s), {s['duration_s']:.1f} s (critical type, below INVALID limits)")
        elif s["review_required"]:
            parts = []
            if s["confirmed_event_count"]:
                parts.append(f"{s['confirmed_event_count']} confirmed event(s), {s['confirmed_duration_s']:.1f} s, "
                             f"{s['evidence_strength']} evidence")
            if s["unconfirmed_event_count"] and s["buffer_policy"].get("review_unconfirmed"):
                parts.append(f"{s['unconfirmed_event_count']} unconfirmed event(s) to check")
            review.append(f"{v}: " + "; ".join(parts) + " (manual review, below INVALID limits)")
    review += soft
    if ac["audio_silent_is_suspicious"] and audio_status != "ok":
        review.append(f"Audio source {audio_status.replace('_', ' ')}: audio could not be monitored")
    if cov and cov["status"] == "INCONCLUSIVE":            # never VALID without enough monitoring
        review.append("Assessment inconclusive, retest recommended: " + "; ".join(cov["problems"]))
    if review:
        return score, "SUSPICIOUS", review, expl
    return score, "VALID", ["No violation above its limits"], expl


def group_percentages(stats, audio, masks=None):
    """% of the analysed session per evidence group (LIM2, flow.txt §C.6). BUF1: confirmed time only."""
    pct = lambda t: _pct(stats[t]) if t in stats and stats[t]["measurable"] else 0.0  # noqa: E731
    out = {"head_pose": pct(C.HEAD_POSE_DEVIATION), "multiple_faces": pct(C.MULTIPLE_PERSONS),
           "object": pct(C.PROHIBITED_OBJECT), "spoof": pct(C.FACE_SPOOF), "voice": 0.0}
    if audio is not None and audio.get("status") == "ok" and audio.get("duration_s"):
        vtypes = [t for t in (C.VOICE_MISMATCH, C.MULTIPLE_SPEAKERS, C.WHISPERING) if stats.get(t, {}).get("measurable")]
        if masks is not None:
            flagged = lambda i, w, t: masks[t][i]  # noqa: E731
        else:
            flagged = lambda i, w, t: w.get("flags", {}).get(t)  # noqa: E731
        n = sum(1 for i, w in enumerate(audio["windows"])
                if any(flagged(i, w, t) for t in vtypes))  # union, no double count
        out["voice"] = round(100.0 * n * SLOT_S / audio["duration_s"], 2)
    return out


def legacy_verdict(video):
    """v1 decision logic, reproduced for the legacy profile (regression only)."""
    s = video["samples"]
    cal = sum(1 for x in s if x["phase"] == "calibration")
    analyzed = len(s) - cal if video["baseline"] else len(s)
    same = sum(1 for x in s if x.get("identity_ok_raw"))
    diff = sum(1 for x in s if x.get("identity_ok_raw") is False)
    cnt = {t: sum(1 for x in s if x.get("flags", {}).get(t)) for t in C.VIDEO_TYPES}
    res = {
        "identity": "AUTHORIZED PERSON" if same > diff else "UNAUTHORIZED PERSON",
        "deviation": "DEVIATED" if analyzed and cnt[C.HEAD_POSE_DEVIATION] >= 0.25 * analyzed else "NOT DEVIATED",
        "gaze": "SUSPICIOUS EYE MOVEMENT" if analyzed and cnt[C.GAZE_AWAY] >= 0.30 * analyzed else "NORMAL EYE MOVEMENT",
        "persons": "MULTIPLE PERSONS DETECTED" if analyzed and cnt[C.MULTIPLE_PERSONS] >= 0.20 * analyzed else "SINGLE PERSON",
        "objects": "PROHIBITED OBJECTS DETECTED" if analyzed and cnt[C.PROHIBITED_OBJECT] >= 0.15 * analyzed else "NO PROHIBITED OBJECTS",
        "analyzed_frames": analyzed, "counts": cnt,
    }
    if res["identity"] == "UNAUTHORIZED PERSON":
        v = "EXAM INVALID - Unauthorized person detected"
    elif res["persons"] == "MULTIPLE PERSONS DETECTED":
        v = "EXAM INVALID - Multiple persons detected"
    elif res["objects"] == "PROHIBITED OBJECTS DETECTED":
        v = "EXAM INVALID - Prohibited objects found"
    elif res["deviation"] == "DEVIATED" or res["gaze"] == "SUSPICIOUS EYE MOVEMENT":
        v = "EXAM SUSPICIOUS - Review required"
    else:
        v = "EXAM VALID - No violations detected"
    res["verdict"] = v
    return res


def aggregate(video, audio, cfg):
    """Returns the report core: per_violation stats, score, verdict, reasons, timeline."""
    vstats, vevents = video_stats(video, cfg) if video else ({}, [])
    astats, aevents = audio_stats(audio, cfg)
    stats = {**vstats, **astats}
    cov, masks, events = None, None, None
    timeline = sorted(vevents + aevents, key=lambda e: (e["start_s"], e["type"]))
    if cfg.get("buffer", {}).get("enabled") and video:      # BUF1 tolerance buffer + FUS1 evidence fusion
        from aggregation import buffer, fusion
        timeline, masks = buffer.apply(stats, video, audio, vevents, aevents, cfg)
        cov = buffer.coverage(video, audio, cfg)
        fusion.check_required(stats, cov, cfg)
        for vtype, st in stats.items():                    # critical flag is needed for review_required
            st["critical"] = cfg["aggregation"]["severity"][vtype] == "critical"
        events = fusion.normalize(timeline, stats, video, audio, cfg)
        for vtype, st in stats.items():                    # type-level quality: all confirmed events low
            q = [e["observation_quality"] for e in events if e["type"] == vtype and e["validation_status"] == "confirmed"]
            if q and all(x == "low" for x in q):
                st["observation_quality"] = "low"
        buffer.annotate(stats, cov, cfg)
    score, label, reasons, expl = score_and_verdict(stats, audio, cfg, cov, events)
    out = {"per_violation": stats, "score": score, "verdict": label, "reasons": reasons, "score_explanation": expl,
           "group_percentages": group_percentages(stats, audio, masks), "timeline": timeline}
    if events is not None:
        fused = fusion.finalize(timeline, stats, video, audio, cfg, cov, score, label)
        out.update(events=fused["events"], fusion=fused["fusion"], assessment=fused["assessment"])
    if video and cfg["aggregation"]["verdict"] == "legacy":
        out["legacy"] = legacy_verdict(video)
    return out
