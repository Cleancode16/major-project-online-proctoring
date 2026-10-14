"""
PDF version of a proctoring report (flow.txt §C.4 "PDF report", PDF1). Uses reportlab, standard fonts only.

  build_pdf(report, report_dir, out_path)  -> out_path
  CLI:  python -m aggregation.pdf_report <report.json> [out.pdf] [--evidence-dir DIR]
"""

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import CondPageBreak, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# reference palette slots 1-2 (same as the web results page)
VIDEO = colors.HexColor("#2a78d6")
AUDIO = colors.HexColor("#eb6834")
INK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#475569")
GRID = colors.HexColor("#cbd5e1")
LIGHT = colors.HexColor("#f1f5f9")
VERDICT_STYLE = {
    "VALID": (colors.HexColor("#dcfce7"), colors.HexColor("#166534"), "PASS"),
    "SUSPICIOUS": (colors.HexColor("#fef3c7"), colors.HexColor("#92400e"), "REVIEW"),
    "INVALID": (colors.HexColor("#fee2e2"), colors.HexColor("#991b1b"), "FAIL"),
}
GROUPS = [("head_pose", "Head pose movement"), ("multiple_faces", "Multiple faces"),
          ("object", "Prohibited objects"), ("voice", "Voice violations")]
# evidence groups shown in the downloadable PDF / Word files: voice clips are audio only, so they are left out
DOC_GROUPS = [(g, label) for g, label in GROUPS if g != "voice"]
PAGE_W, PAGE_H = A4
CONTENT_W = PAGE_W - 30 * mm

_REPL = {"≥": ">=", "≤": "<=", "Δ": "d", "→": "->", "–": "-", "—": "-", "•": "-"}


def safe(text):
    """Text that the standard PDF fonts can render, XML-escaped for Paragraph."""
    s = str(text if text is not None else "")
    for a, b in _REPL.items():
        s = s.replace(a, b)
    s = s.encode("cp1252", "replace").decode("cp1252")
    return escape(s)


def nice(t):
    return str(t).lower().replace("_", " ")


def counter_cells(c):
    """(% of session, limit) texts for an evidence counter; old reports only have count / threshold."""
    pct = f"{c['percentage']:.1f}%" if "percentage" in c else "-"
    return pct, str(c.get("limit_text", c.get("threshold", "-")))


def score_rows(report):
    """(violation, severity, points, max, explanation) sorted by points; empty for reports before SC2."""
    rows = [(t, s.get("severity", ""), s.get("score"), s.get("max_points"), s.get("score_explanation", ""))
            for t, s in report["per_violation"].items() if "max_points" in s]
    return sorted(rows, key=lambda r: (-(r[2] or 0), -(r[3] or 0)))


def points_text(s):
    if "max_points" not in s:
        return str(s.get("weight", ""))           # report from before SC2
    return f"{s['score']:.1f} / {s['max_points']}" if s.get("score") is not None else f"- / {s['max_points']}"


def limits_text(ev):
    if ev.get("limits"):
        return ", ".join(f"{nice(g)} " + (f"> {v['value']:g}% of session" if v["basis"] == "percent" else
                                          ("any detection" if v["value"] == 0 else f"> {v['value']} events"))
                         for g, v in ev["limits"].items())
    return ", ".join(f"{nice(g)} > {n} events" for g, n in (ev.get("thresholds") or {}).items())


def fmt_t(s):
    s = float(s or 0.0)
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


# ------------------------------------------------------------------------------------- styles
_ss = getSampleStyleSheet()
ST = {
    "title": ParagraphStyle("t", parent=_ss["Title"], fontSize=20, leading=24, textColor=INK, alignment=TA_LEFT,
                            spaceAfter=2),
    "sub": ParagraphStyle("s", parent=_ss["Normal"], fontSize=9, leading=12, textColor=MUTED),
    "h2": ParagraphStyle("h2", parent=_ss["Heading2"], fontSize=13, leading=16, textColor=INK, spaceBefore=10,
                         spaceAfter=6),
    "h3": ParagraphStyle("h3", parent=_ss["Heading3"], fontSize=11, leading=14, textColor=INK, spaceBefore=6,
                         spaceAfter=4),
    "body": ParagraphStyle("b", parent=_ss["Normal"], fontSize=9, leading=12, textColor=INK),
    "small": ParagraphStyle("sm", parent=_ss["Normal"], fontSize=7.5, leading=9.5, textColor=MUTED),
    "cell": ParagraphStyle("c", parent=_ss["Normal"], fontSize=8, leading=10, textColor=INK),
}


def P(text, style="body"):
    return Paragraph(text, ST[style])


def table(rows, widths, header=True, zebra=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("TEXTCOLOR", (0, 0), (-1, -1), INK),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LINEBELOW", (0, 0), (-1, -1), 0.25, GRID)]
    if header:
        style += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8), ("BACKGROUND", (0, 0), (-1, 0), LIGHT)]
    if zebra:
        style += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")])]
    t.setStyle(TableStyle(style))
    return t


# ------------------------------------------------------------------------------------- charts
def legend(d, x, y):
    for i, (label, col) in enumerate((("video", VIDEO), ("audio", AUDIO))):
        d.add(Rect(x + i * 60, y, 8, 8, fillColor=col, strokeColor=None))
        d.add(String(x + i * 60 + 12, y + 1, label, fontName="Helvetica", fontSize=7, fillColor=MUTED))


def percent_chart(per_violation):
    rows = sorted(((t, s) for t, s in per_violation.items() if s.get("measurable")), key=lambda r: -r[1]["percentage"])
    if not rows:
        return None
    row_h, label_w = 15, 120
    plot_w = CONTENT_W - label_w - 40
    h = len(rows) * row_h + 30
    d = Drawing(CONTENT_W, h)
    top = h - 18
    vmax = max(10.0, max(s["percentage"] for _, s in rows) * 1.1)
    legend(d, label_w, h - 10)
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        x = label_w + f * plot_w
        d.add(Line(x, 12, x, top, strokeColor=GRID, strokeWidth=0.4))
        d.add(String(x, 2, f"{f * vmax:.0f}%", fontName="Helvetica", fontSize=6.5, fillColor=MUTED, textAnchor="middle"))
    for i, (t, s) in enumerate(rows):
        y = top - (i + 1) * row_h + 3
        w = max(1.5, s["percentage"] / vmax * plot_w)
        d.add(String(label_w - 5, y + 2, nice(t), fontName="Helvetica", fontSize=7, fillColor=INK, textAnchor="end"))
        d.add(Rect(label_w, y, w, 9, fillColor=VIDEO if s.get("source") == "video" else AUDIO, strokeColor=None))
        d.add(String(label_w + w + 4, y + 2, f"{s['percentage']:.1f}%", fontName="Helvetica", fontSize=7, fillColor=INK))
    return d


def timeline_chart(timeline, duration):
    if not timeline:
        return None
    types = list(dict.fromkeys(e["type"] for e in timeline))
    row_h, label_w = 14, 120
    plot_w = CONTENT_W - label_w - 10
    h = len(types) * row_h + 30
    d = Drawing(CONTENT_W, h)
    top = h - 18
    end = max([duration or 0.0] + [e["end_s"] for e in timeline] + [1.0])
    legend(d, label_w, h - 10)
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        x = label_w + f * plot_w
        d.add(Line(x, 12, x, top, strokeColor=GRID, strokeWidth=0.4))
        d.add(String(x, 2, fmt_t(f * end), fontName="Helvetica", fontSize=6.5, fillColor=MUTED, textAnchor="middle"))
    for i, t in enumerate(types):
        y = top - (i + 1) * row_h + 3
        d.add(String(label_w - 5, y + 2, nice(t), fontName="Helvetica", fontSize=7, fillColor=INK, textAnchor="end"))
    for e in timeline:
        i = types.index(e["type"])
        y = top - (i + 1) * row_h + 3
        x0 = label_w + e["start_s"] / end * plot_w
        w = max(1.5, (max(e["end_s"], e["start_s"] + 0.2) - e["start_s"]) / end * plot_w)
        d.add(Rect(x0, y, w, 8, fillColor=VIDEO if e["source"] == "video" else AUDIO,
                   strokeColor=colors.white, strokeWidth=0.3))
    return d


# ------------------------------------------------------------------------------------- sections
def _header(report):
    m = report.get("meta", {})
    return [P(f"Proctoring report - candidate {safe(m.get('candidate', '-'))}", "title"),
            P(f"Session {safe(m.get('session_id', '-'))} &nbsp;|&nbsp; {safe(m.get('created', ''))} &nbsp;|&nbsp; "
              f"mode {safe(m.get('mode', 'recorded'))} &nbsp;|&nbsp; profile {safe(m.get('profile', ''))}", "sub"),
            Spacer(1, 8)]


def _verdict(report):
    v = report["verdict"]
    bg, fg, word = VERDICT_STYLE.get(v["label"], VERDICT_STYLE["SUSPICIOUS"])
    left = [P(f"<font size=16><b>{safe(v['label'])}</b></font> &nbsp; <font size=9>({word})</font>", "body"),
            Spacer(1, 3), P(f"<b>{safe(v['headline'])}</b>", "body")]
    left += [P(f"- {safe(r)}", "small") for r in v["reasons"][1:]]
    if v.get("score_explanation"):
        left += [Spacer(1, 3), P(f"<i>{safe(v['score_explanation'])}</i>", "small")]
    right = [P(f"<para alignment='right'><font size=8>Violation score</font><br/>"
               f"<font size=22><b>{v['score']:.1f}</b></font><br/><font size=7>of 100</font></para>", "body")]
    t = Table([[left, right]], colWidths=[CONTENT_W - 35 * mm, 35 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), 1.2, fg),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("TEXTCOLOR", (0, 0), (-1, -1), fg)]))
    return [t, Spacer(1, 8)]


def _summary(report):
    vid, aud = report.get("video") or {}, report.get("audio") or {}
    b = vid.get("baseline") or {}
    rows = [[P("<b>Video analysed</b>", "cell"), P("<b>Audio</b>", "cell"), P("<b>Events</b>", "cell")],
            [P(f"{vid.get('duration_s', 0):.0f} s, {vid.get('n_samples', 0)} samples @ "
               f"{(vid.get('sample_rate_hz') or 0):.2f}/s", "cell"),
             P(safe((aud.get("status") or "not analysed").replace("_", " ")) +
               (f", {aud.get('duration_s', 0):.0f} s, speech {aud.get('speech_s', 0):.1f} s" if aud.get("status") == "ok" else ""),
               "cell"),
             P(f"{len(report.get('timeline', []))}" + (f" &nbsp; (baseline yaw {b['yaw']:.1f}, pitch {b['pitch']:.1f}, "
                                                       f"roll {b['roll']:.1f} deg)" if b else ""), "cell")]]
    out = [table(rows, [CONTENT_W / 3] * 3, zebra=False), Spacer(1, 6)]
    for w in report.get("warnings", []):
        out.append(P(f"<font color='#92400e'><b>Warning:</b> {safe(w)}</font>", "body"))
    return out


def _scoring(report):
    rows = score_rows(report)
    if not rows:
        return []
    v = report["verdict"]
    out = [P("Scoring", "h2"),
           P(f"<b>Overall score {v['score']:.1f} / 100 - {safe(v['label'])}.</b> {safe(v.get('score_explanation', ''))}",
             "body"), Spacer(1, 4)]
    trs = [["Violation", "Severity", "Points", "Explanation"]]
    for t, sev, pts, mx, expl in rows:
        trs.append([P(safe(nice(t)), "cell"), sev, f"{pts:.1f} / {mx}" if pts is not None else f"- / {mx}",
                    P(safe(expl), "small")])
    out.append(table(trs, [34 * mm, 16 * mm, 18 * mm, CONTENT_W - 68 * mm]))
    return out


def _evidence_counters(report):
    counters = report.get("evidence_counters") or {}
    if not counters:
        return []
    sv = report.get("session_violation") or {}
    up = report.get("evidence_upload") or {}
    out = [P("Session violation and evidence counters", "h2")]
    if sv.get("violated"):
        out.append(P(f"<font color='#991b1b'><b>SESSION MARKED AS VIOLATED</b> - limits exceeded for: "
                     f"{safe(', '.join(nice(g) for g in sv.get('groups_exceeded', [])))}</font>", "body"))
        out.append(Spacer(1, 4))
    rows = [["Group", "Events", "% of session", "Limit", "Exceeded", "Upload", "Target / reason"]]
    for g, label in GROUPS:
        c, u = counters.get(g, {}), up.get(g, {})
        pct, lim = counter_cells(c)
        rows.append([label, str(c.get("count", 0)), pct, P(safe(lim), "small"),
                     "YES" if c.get("exceeded") else "no", u.get("status", "-"),
                     P(safe(u.get("error") or u.get("target") or u.get("reason") or ""), "small")])
    out.append(table(rows, [32 * mm, 13 * mm, 18 * mm, 26 * mm, 16 * mm, 17 * mm, CONTENT_W - 122 * mm]))
    return out


def _per_violation(report):
    rows = [["Violation", "Source", "%", "Numerator / denominator", "Seconds", "Events", "Severity", "Points"]]
    for t, s in report["per_violation"].items():
        rows.append([nice(t), s.get("source", ""),
                     f"{s['percentage']:.1f}" if s.get("measurable") else "n/a",
                     f"{s['numerator']} / {s['denominator']} {s['unit']}" if s.get("measurable") else "not measurable",
                     f"{s.get('duration_s', 0):.1f}", str(s.get("count", 0)),
                     s.get("severity") or ("critical" if s.get("critical") else ""), points_text(s)])
    out = [P("Violations by type", "h2"),
           table(rows, [36 * mm, 13 * mm, 12 * mm, 46 * mm, 16 * mm, 14 * mm, 14 * mm, CONTENT_W - 151 * mm])]
    chart = percent_chart(report["per_violation"])
    if chart:
        out += [Spacer(1, 6), P("Violation percentage by type", "h3"), chart]
    vid, aud = report.get("video") or {}, report.get("audio") or {}
    tl = timeline_chart(report.get("timeline", []), max(vid.get("duration_s") or 0, aud.get("duration_s") or 0))
    if tl:
        out += [CondPageBreak(60 * mm), P("Timeline (video and audio on one time axis)", "h3"), tl]
    return out


def _caption(group, it):
    base = f"<b>{fmt_t(it['start_s'])} - {fmt_t(it['end_s'])}</b> ({it['duration_s']:.1f} s)"
    if group == "object":
        objs = ", ".join(f"{o['label']} {100 * o['confidence']:.0f}%" for o in it.get("objects", []))
        return f"{base}<br/>{safe(objs or 'object')}"
    if group == "multiple_faces":
        return (f"{base}<br/>{it.get('face_count', '?')} face(s), {it.get('person_count', '?')} person(s), "
                f"max {it.get('max_persons', '?')}")
    if group == "head_pose":
        a, lim = it.get("angles_deg", {}), it.get("allowed_deg", {})
        return (f"{base}<br/>yaw {a.get('yaw', '-')}, pitch {a.get('pitch', '-')}, roll {a.get('roll', '-')} deg "
                f"(limits {lim.get('yaw')}/{lim.get('pitch')}/{lim.get('roll')}); allowed {it.get('allowed_duration_s')} s")
    return base


def _image(path, width):
    try:
        iw, ih = ImageReader(str(path)).getSize()
    except Exception:
        return None
    return Image(str(path), width=width, height=width * ih / iw)


def _evidence(report, evidence_dir):
    ev = report.get("evidence") or {}
    if not ev:
        return []
    out = [CondPageBreak(80 * mm), P("Evidence", "h2")]
    for g, label in DOC_GROUPS:                     # voice clips are not part of the PDF (audio only)
        items = (ev.get(g) or {}).get("items", [])
        out.append(P(f"{label} &nbsp; <font size=8 color='#475569'>({len(items)} event(s))</font>", "h3"))
        if not items:
            out.append(P("No events.", "small"))
            continue
        cell_w = (CONTENT_W - 8 * mm) / 3
        cells = []
        for it in items:
            img = _image(evidence_dir / it["file"], cell_w - 4) if it.get("file") else None
            cap = P(_caption(g, it), "small")
            cells.append([img, cap] if img is not None else [P(safe(it.get("note") or "image not available"), "small"), cap])
        rows = [cells[i:i + 3] + [""] * (3 - len(cells[i:i + 3])) for i in range(0, len(cells), 3)]
        t = Table(rows, colWidths=[cell_w + 2.6 * mm] * 3)
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        out.append(t)
    return out


def _events(report):
    tl = report.get("timeline", [])
    if not tl:
        return []
    rows = [["Start", "End", "Duration", "Source", "Violation", "Details"]]
    for e in tl:
        rows.append([fmt_t(e["start_s"]), fmt_t(e["end_s"]), f"{e['duration_s']:.1f} s", e["source"], nice(e["type"]),
                     P(safe(json.dumps(e.get("details") or {})[:160]), "small")])
    return [CondPageBreak(50 * mm), P("All events", "h2"),
            table(rows, [18 * mm, 18 * mm, 18 * mm, 14 * mm, 38 * mm, CONTENT_W - 106 * mm])]


def _thresholds(report):
    th = report.get("thresholds") or {}
    hp, gz, ob, ev = th.get("head_pose", {}), th.get("gaze", {}), th.get("objects", {}), th.get("evidence", {})
    lines = [
        f"Head pose: yaw {hp.get('yaw_thr')} / pitch {hp.get('pitch_thr')} / roll {hp.get('roll_thr')} deg, "
        f"for {hp.get('persist_s')} s (method {hp.get('method', 'pnp')})",
        f"Gaze: {gz.get('h_thr_deg')} deg horizontal / {gz.get('v_thr_deg')} deg vertical, for {gz.get('persist_s')} s",
        f"Face match threshold (cosine): {th.get('face', {}).get('cos_threshold')}; "
        f"voice match threshold: {th.get('speaker', {}).get('verify_thr')}",
        "Objects: " + ", ".join(f"{name} >= {ob.get('conf', {}).get(cls, ob.get('conf', {}).get(str(cls), '?'))}"
                                for cls, name in (ob.get("classes") or {}).items()),
        "Session violated when: " + limits_text(ev),
    ]
    return [CondPageBreak(40 * mm), P("Thresholds used", "h2")] + [P(f"- {safe(s)}", "small") for s in lines]


def _page(canvas, doc, report):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    m = report.get("meta", {})
    canvas.drawString(15 * mm, 9 * mm, f"Online Proctoring - candidate {m.get('candidate', '-')} - session "
                                       f"{m.get('session_id', '-')}")
    canvas.drawRightString(PAGE_W - 15 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


# ------------------------------------------------------------------------------------- build
def build_pdf(report, evidence_dir, out_path):
    """Write the PDF for `report` (dict). evidence_dir = folder containing evidence/<group>/... files."""
    out_path = Path(out_path)
    evidence_dir = Path(evidence_dir)
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=14 * mm, bottomMargin=16 * mm,
                            title=f"Proctoring report {report.get('meta', {}).get('candidate', '')}",
                            author="Online Proctoring System")
    story = []
    story += _header(report)
    story += _verdict(report)
    story += _summary(report)
    story += _scoring(report)
    story += _evidence_counters(report)
    story += _per_violation(report)
    story += _evidence(report, evidence_dir)
    story += _events(report)
    story += _thresholds(report)
    doc.build(story, onFirstPage=lambda c, d: _page(c, d, report), onLaterPages=lambda c, d: _page(c, d, report))
    return out_path


def write_report_pdf(report_json_path):
    """Build report.pdf next to report.json (used at the end of a session and by the API)."""
    p = Path(report_json_path)
    report = json.loads(p.read_text(encoding="utf-8"))
    return build_pdf(report, p.parent, p.with_suffix(".pdf"))


def _find_evidence_dir(json_path, report):
    """Evidence next to the JSON, else data/reports/<id> (id from the file name proctoring_report_<id>.json)."""
    import config as C
    first = next((it["file"] for g in (report.get("evidence") or {}).values() for it in g.get("items", [])
                  if it.get("file")), None)
    cands = [json_path.parent]
    stem = json_path.stem
    if stem.startswith("proctoring_report_"):
        cands.append(C.REPORTS_DIR / stem[len("proctoring_report_"):])
    cands += sorted(C.REPORTS_DIR.glob(f"{report.get('meta', {}).get('candidate', '__none__')}_*"), reverse=True)
    for c in cands:
        if first is None or (c / first).exists():
            return c
    return json_path.parent


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    src = Path(argv[0])
    out = Path(argv[1]) if len(argv) > 1 and not argv[1].startswith("--") else src.with_suffix(".pdf")
    report = json.loads(src.read_text(encoding="utf-8"))
    ev_dir = Path(argv[argv.index("--evidence-dir") + 1]) if "--evidence-dir" in argv else _find_evidence_dir(src, report)
    build_pdf(report, ev_dir, out)
    print(f"PDF written: {out} (evidence from {ev_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
