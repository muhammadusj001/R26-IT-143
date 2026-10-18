"""
PDF session report generator.

Turns a SessionTracker.get_summary() dict (optionally enriched by the
caller with a few state-only fields — frames_processed, model_status —
that live outside the tracker's own scope) into a polished A4 PDF using
reportlab's platypus layer.

Every section is defensive: missing or empty data renders a plain
"no data" sentence instead of raising, so a report can always be
generated even for a session with nothing interesting in it.
"""

from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)

NAVY = colors.HexColor("#0d1725")
NAVY_LIGHT = colors.HexColor("#16233a")
ACCENT = colors.HexColor("#0e7490")   # muted teal/aqua, printable
ROW_LIGHT = colors.HexColor("#f3f6f9")
ROW_WHITE = colors.white
TEXT_DIM = colors.HexColor("#4b5563")
DANGER = colors.HexColor("#b91c1c")
WARN = colors.HexColor("#b45309")

SYSTEM_NAME = "R26-IT-143 AI Smart Swimming Pool Monitoring System"

_styles = getSampleStyleSheet()
_TITLE = ParagraphStyle(
    "ReportTitle", parent=_styles["Title"], textColor=NAVY,
    fontSize=22, leading=26, spaceAfter=2,
)
_SUBTITLE = ParagraphStyle(
    "ReportSubtitle", parent=_styles["Normal"], textColor=TEXT_DIM,
    fontSize=10.5, leading=14, spaceAfter=1,
)
_SECTION = ParagraphStyle(
    "SectionHeading", parent=_styles["Heading2"], textColor=NAVY,
    fontSize=13.5, leading=17, spaceBefore=16, spaceAfter=8,
    borderColor=ACCENT, borderWidth=0,
)
_BODY = ParagraphStyle(
    "ReportBody", parent=_styles["Normal"], textColor=colors.HexColor("#1f2937"),
    fontSize=10, leading=14,
)
_MUTED = ParagraphStyle(
    "ReportMuted", parent=_styles["Normal"], textColor=TEXT_DIM,
    fontSize=10, leading=14, spaceAfter=4,
)
_CELL = ParagraphStyle("Cell", parent=_styles["Normal"], fontSize=9.5, leading=12)
_CELL_HEAD = ParagraphStyle(
    "CellHead", parent=_styles["Normal"], fontSize=9.5, leading=12,
    textColor=colors.white, fontName="Helvetica-Bold",
)


def generate_session_report(summary: dict, output_path: str) -> str:
    """Build a professional A4 PDF session report at output_path.

    summary is expected to be (at minimum) the dict returned by
    SessionTracker.get_summary(); the caller may add "frames_processed"
    and "model_status" (dict of module -> status string) for the fuller
    sections, but every field is optional — nothing here raises if a
    key is missing or a data set is empty.
    """
    summary = summary or {}

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=20 * mm, bottomMargin=18 * mm,
        leftMargin=18 * mm, rightMargin=18 * mm,
        title="Pool Session Report",
    )

    story = []
    _build_header(story)
    _build_overview(story, summary)
    _build_occupancy(story, summary)
    _build_water_quality(story, summary)
    _build_safety_events(story, summary)
    _build_maintenance(story, summary)
    _build_system_notes(story, summary)

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output_path


# ── Header ───────────────────────────────────────────────────
def _build_header(story):
    story.append(Paragraph("Pool Session Report", _TITLE))
    story.append(Paragraph(SYSTEM_NAME, _SUBTITLE))
    story.append(Paragraph(
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", _SUBTITLE,
    ))
    story.append(_hr())
    story.append(Spacer(1, 4))


def _hr():
    t = Table([[""]], colWidths=["100%"], rowHeights=[1.2])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1.2, ACCENT)]))
    return t


# ── Section 1 — Session Overview ────────────────────────────
def _build_overview(story, summary):
    story.append(Paragraph("1. Session Overview", _SECTION))
    rows = [
        ("Start time", summary.get("start_time") or "—"),
        ("End time", summary.get("end_time") or "—"),
        ("Duration", _duration_label(summary)),
        ("Frames processed", _fmt_num(summary.get("frames_processed"))),
    ]
    story.append(_kv_table(rows))


def _duration_label(summary):
    formatted = summary.get("duration_formatted")
    seconds = summary.get("duration_seconds")
    if formatted and seconds is not None:
        return f"{formatted} (mm:ss) — {seconds}s total"
    return formatted or "—"


# ── Section 2 — Occupancy & Bather Load ─────────────────────
def _build_occupancy(story, summary):
    story.append(Paragraph("2. Occupancy &amp; Bather Load", _SECTION))
    rows = [
        ("Peak swimmers", _fmt_num(summary.get("peak_occupancy"))),
        ("Average swimmers", _fmt_num(summary.get("average_occupancy"))),
        ("Total bather-hours", _fmt_num(summary.get("total_bather_hours"))),
        ("Peak density level reached", summary.get("peak_density_level") or "—"),
    ]
    story.append(_kv_table(rows))


# ── Section 3 — Water Quality ───────────────────────────────
WATER_LABELS = {
    "ph": "pH",
    "temperature": "Temperature (°C)",
    "chlorine": "Chlorine (ppm)",
    "turbidity": "Turbidity (NTU)",
    "tds": "TDS (ppm)",
}


def _build_water_quality(story, summary):
    story.append(Paragraph("3. Water Quality", _SECTION))
    water = summary.get("water_quality") or {}
    has_data = any(
        (water.get(field) or {}).get("avg") is not None for field in WATER_LABELS
    )

    if not has_data:
        story.append(Paragraph("No water quality readings recorded during this session.", _MUTED))
    else:
        header = ["Parameter", "Min", "Max", "Avg"]
        data = [header]
        for field, label in WATER_LABELS.items():
            stats = water.get(field) or {}
            data.append([
                label,
                _fmt_num(stats.get("min")),
                _fmt_num(stats.get("max")),
                _fmt_num(stats.get("avg")),
            ])
        story.append(_styled_table(data, col_widths=[60 * mm, 32 * mm, 32 * mm, 32 * mm]))

    story.append(Spacer(1, 8))
    status_seconds = summary.get("water_status_seconds") or {}
    if status_seconds:
        story.append(Paragraph(
            "Approximate time spent in each status (based on periodic sampling):", _BODY,
        ))
        parts = [f"{status}: {_fmt_seconds(secs)}" for status, secs in status_seconds.items()]
        story.append(Paragraph(" &nbsp;·&nbsp; ".join(parts), _MUTED))


# ── Section 4 — Safety Events ───────────────────────────────
def _build_safety_events(story, summary):
    story.append(Paragraph("4. Safety Events", _SECTION))
    alerts = summary.get("alerts") or []

    if not alerts:
        story.append(Paragraph("No safety events recorded during this session.", _MUTED))
        return

    header = ["Time", "Module", "Severity", "Message"]
    data = [header]
    for a in alerts:
        severity = str(a.get("severity", "") or "")
        data.append([
            a.get("time", "—"),
            a.get("module", "—"),
            severity.upper() or "—",
            Paragraph(a.get("message", "—"), _CELL),
        ])
    table = _styled_table(
        data, col_widths=[22 * mm, 26 * mm, 24 * mm, 84 * mm], wrap_last=True,
    )
    # Extra: color the severity column for danger/warning rows
    style_extra = []
    for i, a in enumerate(alerts, start=1):
        sev = str(a.get("severity", "")).lower()
        if sev == "danger":
            style_extra.append(("TEXTCOLOR", (2, i), (2, i), DANGER))
        elif sev == "warning":
            style_extra.append(("TEXTCOLOR", (2, i), (2, i), WARN))
    if style_extra:
        table.setStyle(TableStyle(style_extra))
    story.append(table)


# ── Section 5 — Maintenance Recommendations ─────────────────
def _build_maintenance(story, summary):
    story.append(Paragraph("5. Maintenance Recommendations", _SECTION))
    recs = summary.get("maintenance_recommendations") or []

    if not recs:
        story.append(Paragraph("No maintenance required.", _MUTED))
        return

    header = ["Action", "Priority", "Overdue by"]
    data = [header]
    for r in recs:
        if not isinstance(r, dict):
            continue
        overdue = r.get("overdue_by")
        overdue_label = f"{overdue} person-hours" if overdue is not None else "—"
        data.append([
            Paragraph(str(r.get("action", "—")), _CELL),
            str(r.get("priority", "—")),
            overdue_label,
        ])
    if len(data) == 1:
        story.append(Paragraph("No maintenance required.", _MUTED))
        return
    story.append(_styled_table(data, col_widths=[80 * mm, 32 * mm, 44 * mm], wrap_first=True))


# ── Section 6 — System Notes ────────────────────────────────
MODULE_LABELS = {
    "crowd": "Crowd Detection (Component 1)",
    "drowning": "Drowning Detection (Component 3)",
    "garbage": "Garbage Detection (Component 4)",
    "water_quality": "Water Quality Prediction (Component 2)",
}


def _build_system_notes(story, summary):
    story.append(Paragraph("6. System Notes", _SECTION))
    model_status = summary.get("model_status") or {}

    if not model_status:
        story.append(Paragraph("Model status information not available for this session.", _MUTED))
        return

    header = ["AI Module", "Status"]
    data = [header]
    for key, label in MODULE_LABELS.items():
        if key in model_status:
            data.append([label, str(model_status[key])])
    for key, val in model_status.items():
        if key not in MODULE_LABELS:
            data.append([key, str(val)])
    if len(data) == 1:
        story.append(Paragraph("Model status information not available for this session.", _MUTED))
        return
    story.append(_styled_table(data, col_widths=[90 * mm, 66 * mm]))


# ── Footer ───────────────────────────────────────────────────
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(TEXT_DIM)
    canvas.drawString(18 * mm, 12 * mm, "Generated by R26-IT-143 Smart Pool Monitoring System")
    canvas.drawRightString(
        A4[0] - 18 * mm, 12 * mm, f"Page {doc.page}",
    )
    canvas.setStrokeColor(colors.HexColor("#e5e7eb"))
    canvas.line(18 * mm, 16 * mm, A4[0] - 18 * mm, 16 * mm)
    canvas.restoreState()


# ── Small table/format helpers ──────────────────────────────
def _kv_table(rows):
    data = [[Paragraph(f"<b>{k}</b>", _CELL), Paragraph(str(v), _CELL)] for k, v in rows]
    t = Table(data, colWidths=[55 * mm, 111 * mm])
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
    ]
    for i in range(len(rows)):
        if i % 2 == 1:
            style.append(("BACKGROUND", (0, i), (-1, i), ROW_LIGHT))
    t.setStyle(TableStyle(style))
    return t


def _styled_table(data, col_widths=None, wrap_first=False, wrap_last=False):
    """data[0] is the header row (plain strings); data[1:] are body rows
    (cells may be plain strings or already-wrapped Paragraphs)."""
    header = [Paragraph(str(h), _CELL_HEAD) for h in data[0]]
    rows = [header] + data[1:]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY_LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ROW_LIGHT))
        else:
            style.append(("BACKGROUND", (0, i), (-1, i), ROW_WHITE))
    t.setStyle(TableStyle(style))
    return t


def _fmt_num(v):
    if v is None:
        return "—"
    return str(v)


def _fmt_seconds(secs):
    try:
        secs = int(secs)
    except (TypeError, ValueError):
        return "—"
    m, s = divmod(secs, 60)
    if m:
        return f"{m}m {s}s"
    return f"{s}s"
