from __future__ import annotations

from html import escape
from io import BytesIO
import os
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .policy import resolve_report_policy
from .schemas import EvidenceReportData


NAVY = colors.HexColor("#0B3B6E")
NAVY_DARK = colors.HexColor("#082B50")
GRID = colors.HexColor("#B8CCE0")
PALE = colors.HexColor("#EAF3FC")
TEXT = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#52677C")
DANGER = colors.HexColor("#A51D2D")


def _watermark_opacity(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(0.0, min(0.20, value))


# Kept deliberately subtle so evidence photographs and report text remain
# readable while the Gujarat Police and classification marks stay visible.
LOGO_WATERMARK_OPACITY = _watermark_opacity(
    "EVIDENCE_LOGO_WATERMARK_OPACITY",
    0.06,
)
CLASSIFICATION_WATERMARK_OPACITY = _watermark_opacity(
    "EVIDENCE_CLASSIFICATION_WATERMARK_OPACITY",
    0.06,
)


def _p(value: Any) -> str:
    return escape(str(value if value is not None else "-"), quote=False).replace("\n", "<br/>")


def _iso(value) -> str:
    if value is None:
        return "-"
    try:
        return value.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(value)


def _confidence(value: float | None) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _table(data, widths, *, header=True, font_size=8, valign="TOP"):
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), valign),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style.extend([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFD")]),
        ])
    table.setStyle(TableStyle(style))
    return table


def _draw_logo_watermark(
    canvas,
    logo_path: Path,
    *,
    page_width: float,
    page_height: float,
    alpha: float = LOGO_WATERMARK_OPACITY,
):
    if not logo_path.is_file():
        return
    try:
        logo = ImageReader(str(logo_path))
        canvas.saveState()
        if hasattr(canvas, "setFillAlpha"):
            canvas.setFillAlpha(alpha)
        if hasattr(canvas, "setStrokeAlpha"):
            canvas.setStrokeAlpha(alpha)
        width = 108 * mm
        height = 130 * mm
        canvas.drawImage(
            logo,
            (page_width - width) / 2,
            (page_height - height) / 2 - 8 * mm,
            width=width,
            height=height,
            preserveAspectRatio=True,
            mask="auto",
        )
        canvas.restoreState()
    except Exception:
        pass


def _page_decorator(data: EvidenceReportData, logo_path: Path):
    policy = resolve_report_policy(data.classification_code)

    def draw(canvas, doc):
        width, height = A4
        canvas.saveState()
        canvas.setStrokeColor(NAVY)
        canvas.setLineWidth(0.8)
        canvas.line(13 * mm, height - 15 * mm, width - 13 * mm, height - 15 * mm)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(NAVY_DARK)
        canvas.drawString(13 * mm, height - 11 * mm, "INTEL-I | DIGITAL EVIDENCE REPORT")
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(width - 13 * mm, height - 11 * mm, data.classification)
        canvas.restoreState()

        _draw_logo_watermark(canvas, logo_path, page_width=width, page_height=height)

        canvas.saveState()
        try:
            if hasattr(canvas, "setFillAlpha"):
                canvas.setFillAlpha(CLASSIFICATION_WATERMARK_OPACITY)
            canvas.setFillColor(colors.HexColor("#C62828"))
            canvas.setFont("Helvetica-Bold", 20)
            canvas.translate(width / 2, height / 2)
            canvas.rotate(38)
            canvas.drawCentredString(0, 0, policy.watermark_text[:48])
        finally:
            canvas.restoreState()

        canvas.saveState()
        canvas.setStrokeColor(NAVY)
        canvas.setLineWidth(0.7)
        canvas.line(13 * mm, 13 * mm, width - 13 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(13 * mm, 8 * mm, f"Report ID: {data.report_id} | Export: {data.export_id}")
        canvas.drawRightString(width - 13 * mm, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    return draw


def render_incident_evidence_pdf(data: EvidenceReportData, logo_path: Path) -> bytes:
    output = BytesIO()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("EvidenceTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=NAVY_DARK, alignment=TA_CENTER, spaceAfter=3 * mm)
    subtitle = ParagraphStyle("EvidenceSubtitle", parent=styles["BodyText"], fontSize=10, leading=13, textColor=MUTED, alignment=TA_CENTER)
    h1 = ParagraphStyle("EvidenceH1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=NAVY, spaceBefore=4 * mm, spaceAfter=2 * mm)
    h2 = ParagraphStyle("EvidenceH2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=NAVY_DARK, spaceBefore=3 * mm, spaceAfter=1.5 * mm)
    body = ParagraphStyle("EvidenceBody", parent=styles["BodyText"], fontSize=8.7, leading=12, textColor=TEXT)
    small = ParagraphStyle("EvidenceSmall", parent=styles["BodyText"], fontSize=7.4, leading=9.5, textColor=TEXT)
    tiny = ParagraphStyle("EvidenceTiny", parent=styles["BodyText"], fontSize=6.8, leading=8.3, textColor=TEXT)
    decision = ParagraphStyle("EvidenceDecision", parent=body, fontName="Helvetica-Bold", textColor=DANGER, leading=12.5)

    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title="INTEL-I Digital Evidence Report",
        author="INTEL-I Analytics Service",
        subject=f"Incident evidence report {data.incident_id}",
        keywords="INTEL-I, digital evidence, incident, Gujarat Police",
    )
    story = []

    if logo_path.is_file():
        try:
            logo = Image(str(logo_path), width=37 * mm, height=45 * mm, kind="proportional")
            logo.hAlign = "CENTER"
            story.extend([Spacer(1, 6 * mm), logo, Spacer(1, 2 * mm)])
        except Exception:
            story.append(Spacer(1, 11 * mm))
    else:
        story.append(Spacer(1, 11 * mm))

    story.extend([
        Paragraph("INTEL-I", title),
        Paragraph("DIGITAL EVIDENCE REPORT", title),
        Paragraph(_p(data.incident_type.replace("_", " ").title()), subtitle),
        Spacer(1, 5 * mm),
    ])

    cover = [
        ["Report ID", Paragraph(_p(data.report_id), small), "Classification", Paragraph(_p(data.classification), small)],
        ["Case / Incident ID", Paragraph(_p(data.case_id), small), "Report version", data.report_version],
        ["Generated (UTC)", _iso(data.generated_at), "Generated by", Paragraph(_p(data.generated_by), small)],
        ["Scope", Paragraph(_p(data.scope), small), "Approver", Paragraph(_p(data.approver), small)],
        ["Export ID", Paragraph(_p(data.export_id), small), "Source manifest SHA-256", Paragraph(_p(data.source_manifest_sha256), tiny)],
    ]
    cover_table = Table(cover, colWidths=[31 * mm, 51 * mm, 34 * mm, 58 * mm], hAlign="LEFT")
    cover_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("BACKGROUND", (0, 0), (0, -1), PALE),
        ("BACKGROUND", (2, 0), (2, -1), PALE),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([cover_table, Paragraph("Evidence purpose", h2), Paragraph(_p(resolve_report_policy(data.classification_code).evidence_purpose), body), Paragraph("Handling notice", h2), Paragraph(_p(resolve_report_policy(data.classification_code).handling_notice), body), PageBreak()])

    story.append(Paragraph("1. Executive Summary", h1))
    summary = [
        ["Relevant exhibits", "Status", "Cameras", "First relevant evidence", "Latest relevant evidence"],
        [
            str(len(data.exhibits)),
            _p(data.incident_status),
            _p(", ".join(data.camera_names or data.camera_ids) or "Not configured"),
            _iso(data.timeline[0].event_time if data.timeline else data.started_at),
            _iso(data.timeline[-1].event_time if data.timeline else data.ended_at or data.started_at),
        ],
    ]
    story.append(_table(summary, [26 * mm, 28 * mm, 46 * mm, 42 * mm, 42 * mm], font_size=7.4))
    story.extend([Paragraph("Key finding", h2), Paragraph(_p(data.description), body)])

    story.append(Paragraph("2. Alert and Subject Details", h1))
    subject_rows = [
        ["Field", "Recorded value"],
        ["Incident ID", str(data.incident_id)],
        ["Incident type", _p(data.incident_type)],
        ["Severity", _p(data.severity)],
        ["Status", _p(data.incident_status)],
        ["Evaluation status", _p(data.evaluation_status or "Not recorded")],
        ["Evaluation confidence", _confidence(data.evaluation_confidence)],
        ["Primary track / identity", _p(data.primary_track_id or "Not recorded")],
        ["Global vehicle ID", _p(data.global_vehicle_id or "Not recorded")],
        ["Plate candidate(s)", _p(", ".join(data.plate_candidates) or "Not recorded")],
        ["Watchlist result(s)", _p(", ".join(data.watchlist_results) or "Not recorded")],
    ]
    story.append(_table(subject_rows, [48 * mm, 136 * mm], font_size=8))

    story.append(Paragraph("3. Relevant Event Timeline", h1))
    timeline_rows = [["UTC event time", "Event", "Camera", "Result", "Confidence", "Time provenance"]]
    for item in data.timeline[:250]:
        provenance = item.timestamp_source or item.timestamp_quality or "Not recorded"
        if isinstance(item.source_pts_seconds, (int, float)):
            provenance += f" / PTS {float(item.source_pts_seconds):.3f}"
        timeline_rows.append([
            _iso(item.event_time),
            Paragraph(_p(item.event_type), tiny),
            Paragraph(_p(item.camera_name or item.camera_id or "-"), tiny),
            Paragraph(_p(item.result or "-"), tiny),
            _confidence(item.confidence),
            Paragraph(_p(provenance), tiny),
        ])
    if len(timeline_rows) == 1:
        timeline_rows.append(["-", "No timeline records", "-", "-", "-", "-"])
    story.append(_table(timeline_rows, [30 * mm, 31 * mm, 34 * mm, 46 * mm, 18 * mm, 25 * mm], font_size=6.9))

    story.append(Paragraph("4. Analytical Assessment", h1))
    assessment = [
        ["Assessment dimension", "INTEL-I recorded assessment"],
        ["Event classification", Paragraph(_p(data.incident_type.replace("_", " ")), small)],
        ["Observed evidence", Paragraph(_p(data.description), small)],
        ["Evidence confidence", Paragraph(_p(f"{_confidence(data.evaluation_confidence)} (incident evaluation confidence)"), small)],
        ["Operational status", Paragraph(_p(data.incident_status), small)],
        ["Review requirement", Paragraph("Authorised investigator review is required before operational or evidentiary reliance.", small)],
    ]
    story.append(_table(assessment, [50 * mm, 134 * mm], font_size=7.6))
    story.extend([
        Paragraph("Analytical Decision", h2),
        Paragraph(_p("The report records the system's stored analytical assessment and supporting evidence. It does not make an automated legal determination, accusation, or enforcement decision."), decision),
    ])

    if data.journey:
        story.append(Paragraph("5. Multi-Camera Correlation / Journey", h1))
        journey_rows = [["Time", "Camera", "Location", "Identity / Plate", "Confidence"]]
        for point in data.journey[:250]:
            identity = point.get("global_vehicle_id") or point.get("global_person_id") or point.get("plate") or "-"
            location = point.get("location")
            if not location and point.get("latitude") is not None and point.get("longitude") is not None:
                location = f"{point.get('latitude')}, {point.get('longitude')}"
            journey_rows.append([
                Paragraph(_p(point.get("timestamp") or "-"), tiny),
                Paragraph(_p(point.get("camera_name") or point.get("camera_id") or "-"), tiny),
                Paragraph(_p(location or "Not configured"), tiny),
                Paragraph(_p(identity), tiny),
                _confidence(point.get("confidence")),
            ])
        story.append(_table(journey_rows, [34 * mm, 42 * mm, 47 * mm, 42 * mm, 19 * mm], font_size=7))

    exhibit_section_no = 6 if data.journey else 5
    story.extend([PageBreak(), Paragraph(f"{exhibit_section_no}. Evidence Exhibits", h1), Paragraph("Original source evidence remains retained in the evidence store. Images below are derived report exhibits generated for authorised investigation and review.", body), Spacer(1, 2 * mm)])
    if not data.exhibits:
        story.append(Paragraph("No evidence exhibits were available for this incident at export time.", body))
    for exhibit in data.exhibits:
        details = [
            f"Evidence ID: {exhibit.evidence_id}",
            f"Alert ID: {'ALERT-' + str(exhibit.alert_id) if exhibit.alert_id else '-'}",
            f"Event time (UTC): {_iso(exhibit.event_time)}",
            f"Camera: {exhibit.camera_name or exhibit.camera_id or '-'} ({exhibit.camera_id or '-'})",
            f"Location: {exhibit.location or 'Not configured'}",
            f"Subject: {exhibit.subject or '-'}",
            f"Rule / severity: {exhibit.rule or '-'} / {exhibit.severity or '-'}",
            f"Watchlist: {exhibit.watchlist_result or '-'}",
            f"Confidence: {_confidence(exhibit.confidence)}",
            f"Source snapshot SHA-256: {exhibit.snapshot_sha256 or '-'}",
        ]
        if exhibit.description:
            details.append(f"Description: {exhibit.description}")
        details_html = "<br/>".join(_p(line) for line in details)
        exhibit_text = Paragraph(
            f"<b>{_p(exhibit.exhibit_id)} - {_p(exhibit.title)}</b><br/><br/>{details_html}",
            tiny,
        )
        if exhibit.image_bytes:
            try:
                image = Image(BytesIO(exhibit.image_bytes), width=74 * mm, height=52 * mm, kind="proportional")
                block = Table([[image, exhibit_text]], colWidths=[78 * mm, 106 * mm])
            except Exception:
                block = Table([[exhibit_text]], colWidths=[184 * mm])
        else:
            block = Table([[exhibit_text]], colWidths=[184 * mm])
        block.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.45, GRID),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F8FC")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([KeepTogether([block, Spacer(1, 4 * mm)])])

    integrity_no = exhibit_section_no + 1
    limitations_no = integrity_no + 1
    story.extend([PageBreak(), Paragraph(f"{integrity_no}. Integrity, Provenance and Audit", h1)])
    provenance = [["Control", "Recorded value / requirement"]]
    provenance.extend([[Paragraph(_p(name), small), Paragraph(_p(value), small)] for name, value in data.provenance_rows])
    provenance.append([Paragraph("Source evidence manifest", small), Paragraph(_p(data.source_manifest_sha256), tiny)])
    if data.model_context:
        provenance.append([Paragraph("Model / pipeline context", small), Paragraph(_p("; ".join(data.model_context)), tiny)])
    story.append(_table(provenance, [52 * mm, 132 * mm], font_size=7.4))

    story.append(Paragraph(f"{limitations_no}. Limitations and Investigator Notes", h1))
    for index, item in enumerate(data.limitations, start=1):
        story.append(Paragraph(f"{index}. {_p(item)}", body))
        story.append(Spacer(1, 1 * mm))

    story.extend([
        Paragraph("Authorisation", h2),
        _table([
            ["Prepared by", Paragraph(_p(data.generated_by), small), "Reviewed by", "____________________________"],
            ["Date", data.generated_at.strftime("%Y-%m-%d"), "Review date", "____________________________"],
        ], [31 * mm, 53 * mm, 31 * mm, 69 * mm], header=False, font_size=8),
    ])

    decorator = _page_decorator(data, logo_path)
    doc.build(story, onFirstPage=decorator, onLaterPages=decorator)
    return output.getvalue()