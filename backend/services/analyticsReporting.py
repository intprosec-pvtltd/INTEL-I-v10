"""Evaluator-friendly analytics report extraction and PDF/CSV rendering."""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO, StringIO
import csv
import hashlib
import uuid
from typing import Any

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from db.intelligence_model import PlateObservation, VehicleObservation
from db.model import Alert, Camera, Snapshot


REPORT_COLUMNS = (
    "record_type", "event_id", "vehicle_id", "plate", "timestamp",
    "camera_id", "camera_name", "location", "latitude", "longitude",
    "confidence", "watchlist_result", "watchlist_category", "rule",
    "severity", "timestamp_source", "timestamp_quality", "source_pts_seconds",
    "ingested_at", "evidence_path", "evidence_sha256",
)


def utc_naive(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    return aware.isoformat().replace("+00:00", "Z")


def _snapshot_sha(snapshot: Snapshot | None) -> str | None:
    if snapshot is None:
        return None
    return snapshot.sha256 or hashlib.sha256(snapshot.image_data).hexdigest()


def collect_analytics_rows(
    db: Session,
    *,
    user_id: int,
    start: datetime | None = None,
    end: datetime | None = None,
    camera_id: str | None = None,
    watchlist_only: bool = False,
    limit: int = 5000,
) -> list[dict[str, Any]]:
    start, end = utc_naive(start), utc_naive(end)
    safe_limit = max(1, min(int(limit), 10000))
    cameras = db.query(Camera).filter(Camera.user_id == int(user_id)).all()
    camera_map = {camera.cam_id: camera for camera in cameras}
    camera_ids = set(camera_map)
    if camera_id:
        camera_ids &= {str(camera_id)}
    if not camera_ids:
        return []

    alert_query = db.query(Alert).filter(
        Alert.user_id == int(user_id),
        Alert.cam_id.in_(camera_ids),
    )
    if start:
        alert_query = alert_query.filter(or_(
            Alert.source_timestamp >= start,
            and_(Alert.source_timestamp.is_(None), Alert.created_at >= start),
        ))
    if end:
        alert_query = alert_query.filter(or_(
            Alert.source_timestamp <= end,
            and_(Alert.source_timestamp.is_(None), Alert.created_at <= end),
        ))
    if watchlist_only:
        alert_query = alert_query.filter(or_(
            Alert.watchlist_status.isnot(None),
            Alert.watchlist_entry_id.isnot(None),
            Alert.rule.ilike("%WATCHLIST%"),
        ))
    alerts = alert_query.order_by(Alert.created_at.desc()).limit(safe_limit).all()

    rows: list[dict[str, Any]] = []
    for alert in alerts:
        camera = camera_map.get(alert.cam_id)
        snapshot = alert.snapshots[0] if alert.snapshots else None
        timestamp = alert.source_timestamp or alert.created_at
        rows.append({
            "record_type": "ALERT",
            "event_id": f"ALERT-{alert.id}",
            "vehicle_id": None,
            "plate": alert.plate,
            "timestamp": _iso(timestamp),
            "camera_id": alert.cam_id,
            "camera_name": camera.camera_name if camera else alert.cam_id,
            "location": camera.location_name if camera else None,
            "latitude": camera.latitude if camera else None,
            "longitude": camera.longitude if camera else None,
            "confidence": alert.watchlist_match_confidence,
            "watchlist_result": alert.watchlist_status or ("MATCH" if alert.watchlist_entry_id else "NOT_MATCHED"),
            "watchlist_category": alert.watchlist_category,
            "rule": alert.rule,
            "severity": alert.level or alert.alert_type,
            "timestamp_source": alert.timestamp_source or "LEGACY_INGEST_TIME",
            "timestamp_quality": alert.timestamp_quality or "LEGACY_INGEST_TIME",
            "source_pts_seconds": alert.source_pts_seconds,
            "ingested_at": _iso(alert.created_at),
            "evidence_path": f"/snapshot/{snapshot.id}" if snapshot else None,
            "evidence_sha256": _snapshot_sha(snapshot),
            "_snapshot_bytes": snapshot.image_data if snapshot else None,
        })

    remaining = max(0, safe_limit - len(rows))
    if remaining and not watchlist_only:
        plate_query = db.query(PlateObservation).filter(PlateObservation.camera_id.in_(camera_ids))
        if start:
            plate_query = plate_query.filter(PlateObservation.frame_timestamp >= start)
        if end:
            plate_query = plate_query.filter(PlateObservation.frame_timestamp <= end)
        plates = plate_query.order_by(PlateObservation.frame_timestamp.desc()).limit(remaining).all()
        vehicle_ids = {plate.vehicle_id for plate in plates if plate.vehicle_id is not None}
        observation_map: dict[tuple[int, str], VehicleObservation] = {}
        if vehicle_ids:
            observations = db.query(VehicleObservation).filter(
                VehicleObservation.vehicle_id.in_(vehicle_ids),
                VehicleObservation.camera_id.in_(camera_ids),
            ).order_by(VehicleObservation.frame_timestamp.desc()).all()
            for observation in observations:
                observation_map.setdefault((observation.vehicle_id, observation.camera_id), observation)
        for plate in plates:
            camera = camera_map.get(plate.camera_id)
            vehicle_id = None
            if plate.vehicle_id:
                observation = observation_map.get((plate.vehicle_id, plate.camera_id))
                vehicle_id = observation.global_vehicle_id if observation else None
            metadata = plate.metadata_json if isinstance(plate.metadata_json, dict) else {}
            rows.append({
                "record_type": "ANPR",
                "event_id": f"PLATE-{plate.id}",
                "vehicle_id": vehicle_id,
                "plate": plate.normalized_plate or plate.plate_text,
                "timestamp": _iso(plate.frame_timestamp),
                "camera_id": plate.camera_id,
                "camera_name": camera.camera_name if camera else plate.camera_id,
                "location": camera.location_name if camera else None,
                "latitude": camera.latitude if camera else None,
                "longitude": camera.longitude if camera else None,
                "confidence": plate.confidence,
                "watchlist_result": metadata.get("watchlist_status") or "NOT_EVALUATED",
                "watchlist_category": metadata.get("watchlist_category"),
                "rule": "ANPR_OBSERVATION",
                "severity": None,
                "timestamp_source": metadata.get("timestamp_source") or "SOURCE_FRAME_TIME",
                "timestamp_quality": metadata.get("timestamp_quality") or "SOURCE_FRAME_TIME",
                "source_pts_seconds": metadata.get("source_pts_seconds"),
                "ingested_at": _iso(plate.created_at),
                "evidence_path": metadata.get("snapshot_path"),
                "evidence_sha256": metadata.get("evidence_sha256"),
                "_snapshot_bytes": None,
            })

    rows.sort(key=lambda item: item.get("timestamp") or "", reverse=True)
    return rows[:safe_limit]


def report_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cameras = {row.get("camera_id") for row in rows if row.get("camera_id")}
    plates = {row.get("plate") for row in rows if row.get("plate")}
    matches = sum(1 for row in rows if str(row.get("watchlist_result") or "").upper() in {"MATCH", "MATCHED", "CONFIRMED", "ACTIVE"})
    return {
        "record_count": len(rows),
        "camera_count": len(cameras),
        "unique_plate_count": len(plates),
        "watchlist_match_count": matches,
        "first_timestamp": rows[-1].get("timestamp") if rows else None,
        "last_timestamp": rows[0].get("timestamp") if rows else None,
    }


def public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]


def new_report_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"INTEL-I-{stamp}-{uuid.uuid4().hex[:8].upper()}"


def _csv_safe(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def render_csv(rows: list[dict[str, Any]], report_id: str) -> bytes:
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=("report_id", *REPORT_COLUMNS), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        clean = {key: _csv_safe(value) for key, value in row.items() if not key.startswith("_")}
        writer.writerow({"report_id": report_id, **clean})
    return output.getvalue().encode("utf-8-sig")


def render_pdf(rows: list[dict[str, Any]], report_id: str, filters: dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output = BytesIO()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("IntelTitle", parent=styles["Title"], textColor=colors.HexColor("#0B3B6E"), fontSize=20, leading=24)
    small = ParagraphStyle("IntelSmall", parent=styles["BodyText"], fontSize=7.5, leading=9)
    label = ParagraphStyle("IntelLabel", parent=styles["BodyText"], fontSize=9, leading=11, textColor=colors.HexColor("#294A6D"))
    doc = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="INTEL-I Analytics Output Report",
        author="INTEL-I",
        subject="Vehicle, ANPR, watchlist and evidence report",
    )
    summary = report_summary(rows)
    story = [
        Paragraph("INTEL-I Analytics Output Report", title_style),
        Paragraph(f"Report ID: {report_id}", label),
        Paragraph(f"Generated: {_iso(datetime.now(timezone.utc))}", label),
        Spacer(1, 4 * mm),
    ]
    summary_data = [
        ["Records", "Cameras", "Unique plates", "Watchlist matches", "First evidence", "Latest evidence"],
        [summary["record_count"], summary["camera_count"], summary["unique_plate_count"], summary["watchlist_match_count"], summary["first_timestamp"] or "-", summary["last_timestamp"] or "-"],
    ]
    summary_table = Table(summary_data, repeatRows=1, colWidths=[22*mm, 22*mm, 28*mm, 32*mm, 48*mm, 48*mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0B3B6E")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#EAF3FC")),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#B8CCE0")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (0,0), (3,-1), "CENTER"),
    ]))
    story.extend([summary_table, Spacer(1, 5 * mm)])

    headers = ["Type", "Vehicle / Plate", "Timestamp", "Camera", "Location", "Confidence", "Watchlist", "Rule / Severity", "Evidence"]
    data = [headers]
    for row in rows:
        subject = " / ".join(filter(None, [str(row.get("vehicle_id") or ""), str(row.get("plate") or "")])) or "-"
        confidence = row.get("confidence")
        data.append([
            row.get("record_type") or "-",
            Paragraph(subject, small),
            Paragraph(
                f"{row.get('timestamp') or '-'}<br/>{row.get('timestamp_quality') or '-'}"
                + (f" · PTS {row.get('source_pts_seconds'):.3f}" if isinstance(row.get('source_pts_seconds'), (int, float)) else ""),
                small,
            ),
            Paragraph(f"{row.get('camera_name') or '-'}<br/>{row.get('camera_id') or ''}", small),
            Paragraph(str(row.get("location") or "-"), small),
            f"{float(confidence):.3f}" if confidence is not None else "-",
            Paragraph(f"{row.get('watchlist_result') or '-'}<br/>{row.get('watchlist_category') or ''}", small),
            Paragraph(f"{row.get('rule') or '-'}<br/>{row.get('severity') or ''}", small),
            Paragraph(f"{row.get('evidence_path') or '-'}<br/>{(row.get('evidence_sha256') or '')[:12]}", small),
        ])
    table = Table(data, repeatRows=1, colWidths=[14*mm, 28*mm, 35*mm, 33*mm, 39*mm, 20*mm, 30*mm, 37*mm, 34*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0B3B6E")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 7.5),
        ("FONTSIZE", (0,1), (-1,-1), 7),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#B8CCE0")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F4F8FC")]),
        ("ALIGN", (5,1), (5,-1), "RIGHT"),
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(table)

    evidence_rows = [row for row in rows if row.get("_snapshot_bytes")][:24]
    if evidence_rows:
        story.extend([PageBreak(), Paragraph("Evidence Appendix", title_style), Spacer(1, 3*mm)])
        appendix = []
        for row in evidence_rows:
            try:
                image = Image(BytesIO(row["_snapshot_bytes"]), width=52*mm, height=32*mm, kind="proportional")
            except Exception:
                continue
            description = Paragraph(
                f"<b>{row.get('event_id')}</b><br/>{row.get('timestamp') or '-'}<br/>"
                f"{row.get('camera_name') or row.get('camera_id')} · {row.get('location') or 'Location unavailable'}<br/>"
                f"Plate: {row.get('plate') or '-'} · Watchlist: {row.get('watchlist_result') or '-'}<br/>"
                f"SHA-256: {row.get('evidence_sha256') or '-'}",
                small,
            )
            appendix.append([image, description])
        if appendix:
            evidence_table = Table(appendix, colWidths=[58*mm, 115*mm], hAlign="LEFT")
            evidence_table.setStyle(TableStyle([
                ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#B8CCE0")),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F4F8FC")),
                ("LEFTPADDING", (0,0), (-1,-1), 5),
                ("RIGHTPADDING", (0,0), (-1,-1), 5),
                ("TOPPADDING", (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ]))
            story.append(evidence_table)

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#52677C"))
        canvas.drawString(12*mm, 7*mm, f"INTEL-I · {report_id}")
        canvas.drawRightString(landscape(A4)[0]-12*mm, 7*mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()
