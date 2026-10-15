"""
BUF1 violation-level tolerance buffer + assessment coverage (config "buffer").

Three levels on top of the detectors' own smoothing / persistence (video/pipeline.py, audio/monitor.py):
  detection buffer  raw model output -> persistence-filtered flag (already done upstream; counted here as
                    raw_detection_count / filtered_detection_count so nothing is hidden)
  violation buffer  flagged runs closer than merge_gap_s are merged; a merged event is CONFIRMED only when it
                    lasts >= confirm_s and is backed by >= min_support flagged samples/windows, else UNCONFIRMED
  session buffer    the first grace_s seconds of confirmed time of a type earn no points (0 for critical types),
                    and time already scored by a correlated type is not scored twice

Only confirmed events add points, drive the critical overrides and the evidence counters. Unconfirmed events
stay in the timeline (status "unconfirmed") and can still require review. Nothing is dropped.
"""

import config as C

LIGHT_SENSITIVE = {C.PROHIBITED_OBJECT, C.MULTIPLE_PERSONS, C.GAZE_AWAY, C.FACE_SPOOF}
CONFIDENCE_NOTE = ("Detector scores are raw model outputs, not empirically calibrated probabilities; "
                   "a score of 0.9 does not mean 90% accuracy.")


def merge_events(events, gap_s):
    """Events of one type (sorted) closer than gap_s -> one event; keeps the parts for the audit trail."""
    merged = []
    for e in sorted(events, key=lambda e: e["start_s"]):
        if merged and e["start_s"] - merged[-1]["end_s"] <= gap_s:
            m = merged[-1]
            m["end_s"] = max(m["end_s"], e["end_s"])
            m["parts"].append(e)
        else:
            merged.append({**e, "parts": [e]})
    for m in merged:
        m["duration_s"] = round(m["end_s"] - m["start_s"], 2)
        peaks = [p["peak"] for p in m["parts"] if p.get("peak") is not None]
        if "peak" in m:
            m["peak"] = max(peaks) if peaks else None
        if m["parts"][0].get("first_index") is not None:   # video: span of sample indices for the evidence snapshot
            last = m["parts"][-1]
            m["n"] = last["first_index"] + last["n"] - m["first_index"]
        m["merged_from"] = len(m["parts"])
    return merged


def _video_support(samples, vtype):
    """Per sample: (member of the population, raw detection, final flag)."""
    from aggregation.denominators import counts_toward
    out = []
    for s in samples:
        member = counts_toward(s, vtype) and vtype in s.get("flags", {})
        flag = member and bool(s["flags"][vtype])
        out.append((member, member and bool(s.get("raw", {}).get(vtype, flag)), flag))
    return out


def _confirm(merged, support_of, pol):
    for m in merged:
        m["support"] = support_of(m)
        ok = m["duration_s"] >= pol["confirm_s"] and m["support"] >= pol["min_support"]
        m["status"] = "confirmed" if ok else "unconfirmed"
        if not ok:
            m["buffer_reason"] = (f"{m['duration_s']:.2f} s / {m['support']} supporting observation(s); "
                                  f"needs >= {pol['confirm_s']} s and >= {pol['min_support']}")
    return merged


def buffer_video(video, vtype, events, pol):
    """Returns (merged events, per-sample confirmed mask, counts)."""
    samples, rate = video["samples"], video["rate_hz"]
    sup = _video_support(samples, vtype)
    merged = merge_events(events, pol["merge_gap_s"])

    def support_of(m):
        return sum(1 for i in range(m["first_index"], m["first_index"] + m["n"]) if sup[i][2])

    _confirm(merged, support_of, pol)
    mask = [False] * len(samples)
    for m in merged:
        if m["status"] == "confirmed":
            for i in range(m["first_index"], m["first_index"] + m["n"]):
                mask[i] = sup[i][2]
    den = sum(1 for m_, _, _ in sup if m_)
    counts = {"raw_detection_count": sum(1 for _, r, _ in sup if r),
              "filtered_detection_count": sum(1 for _, r, f in sup if r and not f),
              "valid_observation_duration_s": round(den / rate, 2),
              "unavailable_duration_s": round((len(samples) - den) / rate, 2),
              "slot_s": 1.0 / rate, "den": den}
    return merged, mask, counts


def buffer_audio(audio, vtype, events, pol, measurable):
    """Returns (merged events, per-window confirmed mask, counts). Window i owns audio['windows'][i]['slot']."""
    from aggregation.aggregate import SLOT_S
    windows = audio.get("windows", []) if measurable else []
    merged = merge_events(events, pol["merge_gap_s"])
    flagged = [bool(w.get("flags", {}).get(vtype)) for w in windows]
    mids = [(w["slot"][0] + w["slot"][1]) / 2 for w in windows]

    def inside(i, m):
        return m["start_s"] - 1e-6 <= mids[i] <= m["end_s"] + 1e-6

    _confirm(merged, lambda m: sum(1 for i in range(len(windows)) if flagged[i] and inside(i, m)), pol)
    mask = [flagged[i] and any(m["status"] == "confirmed" and inside(i, m) for m in merged)
            for i in range(len(windows))]
    dur = audio["duration_s"] if measurable else 0.0
    counts = {"raw_detection_count": sum(1 for w, f in zip(windows, flagged) if w.get("raw", {}).get(vtype, f)),
              "filtered_detection_count": sum(1 for w, f in zip(windows, flagged) if w.get("raw", {}).get(vtype) and not f),
              "valid_observation_duration_s": round(dur, 2),
              "unavailable_duration_s": 0.0 if measurable else round((audio or {}).get("duration_s") or 0.0, 2),
              "slot_s": SLOT_S, "den": dur}
    return merged, mask, counts


def correlated_types_safe(vtype, cfg):
    from aggregation.fusion import correlated_types
    return correlated_types(vtype, cfg) if "fusion" in cfg else []


def apply(stats, video, audio, video_events, audio_events, cfg):
    """
    Adds the buffer fields to every stats[vtype] and returns (timeline, masks).
    masks[vtype] = per-sample (video) / per-window (audio) confirmed flags, used for group percentages.
    """
    bc = cfg["buffer"]
    pols = bc["policies"]
    timeline, masks = [], {}
    for vtype, st in stats.items():
        pol = pols[vtype]
        evs = [e for e in (video_events if st["source"] == "video" else audio_events) if e["type"] == vtype]
        if st["source"] == "video":
            merged, mask, cnt = buffer_video(video, vtype, evs, pol)
        else:
            merged, mask, cnt = buffer_audio(audio or {}, vtype, evs, pol, st["measurable"])
        masks[vtype] = mask
        conf = [m for m in merged if m["status"] == "confirmed"]
        unconf = [m for m in merged if m["status"] == "unconfirmed"]
        confirmed_s = sum(mask) * cnt["slot_s"]
        den_s = cnt["valid_observation_duration_s"]
        peaks = [m["peak"] for m in conf if m.get("peak") is not None]
        st.update({k: v for k, v in cnt.items() if k not in ("slot_s", "den")})
        st.update(merged_event_count=len(merged), confirmed_event_count=len(conf), unconfirmed_event_count=len(unconf),
                  confirmed_duration_s=round(confirmed_s, 2),
                  confirmed_percentage=round(100.0 * confirmed_s / den_s, 2) if den_s else 0.0,
                  longest_confirmed_event_s=max((m["duration_s"] for m in conf), default=0.0),
                  peak_raw_score=max(peaks) if peaks else None, calibrated_confidence=None,
                  confidence_note=CONFIDENCE_NOTE,
                  validation_metrics=cfg.get("validation_metrics", {}).get(vtype,
                                                                           {"status": "not measured on a labelled set"}),
                  buffer_policy=dict(pol), correlated_with=correlated_types_safe(vtype, cfg))
        for m in merged:
            m.pop("parts", None)
        timeline += merged

    # session buffer: correlated time is scored once (e.g. a head turn also moves the gaze), then the grace budget
    from aggregation.aggregate import SLOT_S
    for vtype, st in stats.items():
        pol = pols[vtype]
        mask = masks[vtype]
        from aggregation.fusion import correlated_types        # FUS1: evidence-family priority de-duplication
        others = [masks[o] for o in correlated_types(vtype, cfg)
                  if o in masks and len(masks[o]) == len(mask) and stats[o]["measurable"]]
        overlap = sum(1 for i, a in enumerate(mask) if a and any(om[i] for om in others))
        slot = SLOT_S if st["source"] == "audio" else 1.0 / video["rate_hz"]
        dedup_s = round(overlap * slot, 2)
        left = max(0.0, st["confirmed_duration_s"] - dedup_s)
        scored_s = max(0.0, left - pol["grace_s"])
        den_s = st["valid_observation_duration_s"]
        st.update(correlated_duration_s=dedup_s, grace_applied_s=round(min(pol["grace_s"], left), 2),
                  scored_duration_s=round(scored_s, 2),
                  scored_percentage=round(100.0 * scored_s / den_s, 2) if den_s else 0.0)
    return sorted(timeline, key=lambda e: (e["start_s"], e["type"])), masks


def coverage(video, audio, cfg):
    """Assessment coverage: how much of the session could actually be monitored, and with usable input."""
    bc, lc = cfg["buffer"], cfg.get("lighting", {})
    out = {"video_monitored_s": 0.0, "video_usable_frac": 0.0, "low_light_frac": 0.0,
           "audio_monitored_s": (audio or {}).get("duration_s", 0.0) if (audio or {}).get("status") == "ok" else 0.0,
           "min_monitored_s": bc["min_monitored_s"], "min_video_usable_frac": bc["min_video_usable_frac"]}
    samples = (video or {}).get("samples") or []
    if samples:
        dark = sum(1 for s in samples if s.get("brightness") is not None and lc and s["brightness"] < lc["low_mean"])
        out.update(video_monitored_s=round(len(samples) / video["rate_hz"], 2),
                   video_usable_frac=round(1.0 - dark / len(samples), 3), low_light_frac=round(dark / len(samples), 3))
    problems = []
    if not samples:
        problems.append("no video was analysed")
    else:
        if out["video_monitored_s"] < bc["min_monitored_s"]:
            problems.append(f"only {out['video_monitored_s']:.1f} s of video was analysed "
                            f"(minimum {bc['min_monitored_s']:.0f} s)")
        if out["video_usable_frac"] < bc["min_video_usable_frac"]:
            problems.append(f"only {100 * out['video_usable_frac']:.0f}% of the video had usable input "
                            f"(minimum {100 * bc['min_video_usable_frac']:.0f}%; the rest was too dark)")
    out["status"] = "INCONCLUSIVE" if problems else "COMPLETE"
    out["problems"] = problems
    return out


def evidence_strength(st, cov, cfg):
    """none | weak (only unconfirmed) | moderate | strong; one level lower on light-sensitive types in a dark video."""
    if not st["measurable"] or not st["merged_event_count"]:
        return "none", None
    if not st["confirmed_event_count"]:
        level = 1
    else:
        long_ = st["longest_confirmed_event_s"] >= 2 * st["buffer_policy"]["confirm_s"]
        level = 3 if long_ or st["confirmed_event_count"] >= 2 else 2
    note = None
    lc = cfg.get("lighting", {})
    if st.get("type") in LIGHT_SENSITIVE and lc and cov["low_light_frac"] >= lc.get("report_frac", 1.0):
        level = max(1, level - 1)
        note = "downgraded: low light makes this detector less reliable"
    return ("none", "weak", "moderate", "strong")[level], note


def annotate(stats, cov, cfg):
    """evidence_strength / review_required per type (after scoring, which sets st['critical'])."""
    for vtype, st in stats.items():
        st["type"] = vtype
        strength, note = evidence_strength(st, cov, cfg)
        st.pop("type")
        st["evidence_strength"] = strength
        if note:
            st["evidence_note"] = note
        pol = st["buffer_policy"]
        st["review_required"] = bool(st["measurable"] and (
            (st.get("critical") and st["confirmed_event_count"] and strength != "strong")
            or (pol.get("review_unconfirmed") and st["unconfirmed_event_count"])))
