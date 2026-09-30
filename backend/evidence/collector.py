from __future__ import annotations

from datetime import datetime, timezone
import os
import re
import uuid
from typing import Any

from sqlalchemy.orm import Session

from db.advanced_intelligence_model import IncidentEvidence, PersonWatchlistEntry
from db.model import Alert, Camera, Incident, Snapshot

from services.analyticsReporting import new_report_id
from services.personRecognition import decrypt_bytes

from .images import EvidenceImageError, prepare_snapshot_for_report
from .integrity import canonical_manifest_sha256
from .policy import resolve_report_policy
from .schemas import EvidenceReportData, Exhibit, TimelineItem


_MAX_EVIDENCE = max(1, min(int(os.getenv("EVIDENCE_REPORT_MAX_EXHIBITS", "100")), 250))
_MAX_JOURNEY = max(1, min(int(os.getenv("EVIDENCE_REPORT_MAX_JOURNEY_POINTS", "250")), 1000))
_SAFE_TEXT_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def _safe_text(value: Any, max_len: int = 2000) -> str:
    text = _SAFE_TEXT_RE.sub("", str(value or "")).strip()
    return text[:max_len]


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _confidence(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return max(0.0, min(1.0, number))


def _camera_label(camera: Camera | None, camera_id: str | None) -> tuple[str | None, str | None]:
    if camera is None:
        return camera_id, None
    return _safe_text(camera.camera_name, 200) or camera_id, _safe_text(camera.location_name, 300) or None


def _extract_plate(alert: Alert | None, metadata: dict[str, Any]) -> str | None:
    values = [
        getattr(alert, "plate", None) if alert else None,
        metadata.get("plate"),
        metadata.get("plate_text"),
        metadata.get("normalized_plate"),
        metadata.get("plate_number"),
        metadata.get("watchlist_plate"),
    ]
    for value in values:
        text = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
        if 4 <= len(text) <= 16:
            return text
    return None


def _metadata_camera_id(metadata: dict[str, Any]) -> str | None:
    journey = metadata.get("journey_observation") if isinstance(metadata.get("journey_observation"), dict) else {}
    return _safe_text(journey.get("camera_id") or metadata.get("camera_id"), 100) or None


def _person_watchlist_entry_id(
    incident: Incident,
    evidence: IncidentEvidence,
    alert: Alert | None,
) -> int | None:
    metadata = evidence.metadata_json if isinstance(evidence.metadata_json, dict) else {}
    state = incident.evaluation_evidence if isinstance(incident.evaluation_evidence, dict) else {}
    journey = state.get("watchlist_journey") if isinstance(state.get("watchlist_journey"), dict) else {}
    candidates = (
        metadata.get("person_watchlist_entry_id"),
        metadata.get("watchlist_entry_id"),
        getattr(alert, "watchlist_entry_id", None) if alert else None,
        journey.get("person_watchlist_entry_id"),
        journey.get("watchlist_entry_id"),
    )
    for value in candidates:
        try:
            entry_id = int(value)
        except (TypeError, ValueError):
            continue
        if entry_id > 0:
            return entry_id
    return None


def _extract_journey(incident: Incident) -> list[dict[str, Any]]:
    state = incident.evaluation_evidence if isinstance(incident.evaluation_evidence, dict) else {}
    candidates: list[Any] = []
    for key in ("watchlist_journey", "journey", "vehicle_journey", "person_journey"):
        value = state.get(key)
        if isinstance(value, dict):
            for route_key in ("route", "observations", "points"):
                if isinstance(value.get(route_key), list):
                    candidates = value.get(route_key)
                    break
        elif isinstance(value, list):
            candidates = value
        if candidates:
            break
    result: list[dict[str, Any]] = []
    for raw in candidates[:_MAX_JOURNEY]:
        if not isinstance(raw, dict):
            continue
        result.append({
            "timestamp": _safe_text(raw.get("timestamp") or raw.get("time"), 80) or None,
            "camera_id": _safe_text(raw.get("camera_id"), 100) or None,
            "camera_name": _safe_text(raw.get("camera_name"), 200) or None,
            "location": _safe_text(raw.get("location") or raw.get("location_name"), 300) or None,
            "latitude": raw.get("latitude"),
            "longitude": raw.get("longitude"),
            "confidence": _confidence(raw.get("correlation_confidence", raw.get("confidence"))),
            "plate": _safe_text(raw.get("plate") or raw.get("plate_number"), 32) or None,
            "global_vehicle_id": _safe_text(raw.get("global_vehicle_id"), 100) or None,
            "global_person_id": _safe_text(raw.get("global_person_id"), 100) or None,
        })
    return result


def _model_context(incident: Incident, evidence_rows: list[IncidentEvidence], snapshots: dict[int, Snapshot]) -> list[str]:
    values: list[str] = []
    state = incident.evaluation_evidence if isinstance(incident.evaluation_evidence, dict) else {}
    for source in [state] + [row.metadata_json for row in evidence_rows if isinstance(row.metadata_json, dict)] + [s.metadata_json for s in snapshots.values() if isinstance(s.metadata_json, dict)]:
        if not isinstance(source, dict):
            continue
        for key in ("model", "model_name", "model_version", "face_model", "detector_model", "recognizer_model", "anpr_model", "reid_model", "pipeline_version"):
            value = _safe_text(source.get(key), 200)
            if value and value not in values:
                values.append(f"{key}: {value}")
    return values[:40]


def collect_incident_report_data(
    db: Session,
    *,
    incident_id: int,
    user_id: int,
    generated_by: str,
    classification: str | None = None,
) -> EvidenceReportData:
    policy = resolve_report_policy(classification)
    incident = db.query(Incident).filter(Incident.id == int(incident_id), Incident.user_id == int(user_id)).first()
    if not incident:
        raise LookupError("Incident not found")

    evidence_rows = (
        db.query(IncidentEvidence)
        .filter(IncidentEvidence.incident_id == int(incident_id), IncidentEvidence.user_id == int(user_id))
        .order_by(IncidentEvidence.created_at.asc(), IncidentEvidence.id.asc())
        .limit(_MAX_EVIDENCE)
        .all()
    )
    alert_ids = sorted({int(row.alert_id) for row in evidence_rows if row.alert_id})
    snapshot_ids = sorted({int(row.snapshot_id) for row in evidence_rows if row.snapshot_id})
    alerts = (
        db.query(Alert).filter(Alert.user_id == int(user_id), Alert.id.in_(alert_ids)).all()
        if alert_ids else []
    )
    alert_map = {int(item.id): item for item in alerts}
    # Snapshot has no user_id column, so tenant ownership is verified through
    # the parent Alert instead of trusting IncidentEvidence.snapshot_id alone.
    snapshots = (
        db.query(Snapshot)
        .join(Alert, Snapshot.alert_id == Alert.id)
        .filter(Snapshot.id.in_(snapshot_ids), Alert.user_id == int(user_id))
        .all()
        if snapshot_ids else []
    )
    snapshot_map = {int(item.id): item for item in snapshots}

    # A watchlist reference is sometimes exposed through the protected person
    # image endpoint without a Snapshot row (for example, imported/legacy
    # incidents and SQL-seeded tests). Resolve only tenant-owned entries so the
    # same enrolled image can still be embedded in the evidence PDF.
    reference_entry_ids: set[int] = set()
    for row in evidence_rows:
        metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
        evidence_role = str(metadata.get("evidence_role") or "").strip().upper()
        if (
            (not row.snapshot_id or int(row.snapshot_id) not in snapshot_map)
            and (
                str(row.evidence_type or "").strip().upper() == "WATCHLIST_REFERENCE"
                or evidence_role == "WATCHLIST_REFERENCE"
            )
        ):
            alert = alert_map.get(int(row.alert_id)) if row.alert_id else None
            entry_id = _person_watchlist_entry_id(incident, row, alert)
            if entry_id is not None:
                reference_entry_ids.add(entry_id)
    reference_entries = (
        db.query(PersonWatchlistEntry)
        .filter(
            PersonWatchlistEntry.user_id == int(user_id),
            PersonWatchlistEntry.id.in_(sorted(reference_entry_ids)),
        )
        .all()
        if reference_entry_ids else []
    )
    reference_entry_map = {int(item.id): item for item in reference_entries}

    camera_ids = {str(incident.cam_id)} if incident.cam_id else set()
    for alert in alerts:
        if alert.cam_id:
            camera_ids.add(str(alert.cam_id))
    for row in evidence_rows:
        metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
        camera_id = _metadata_camera_id(metadata)
        if camera_id:
            camera_ids.add(camera_id)
    journey = _extract_journey(incident)
    for point in journey:
        if point.get("camera_id"):
            camera_ids.add(str(point["camera_id"]))

    cameras = db.query(Camera).filter(Camera.user_id == int(user_id), Camera.cam_id.in_(sorted(camera_ids))).all() if camera_ids else []
    camera_map = {str(camera.cam_id): camera for camera in cameras}

    timeline: list[TimelineItem] = []
    exhibits: list[Exhibit] = []
    plates: list[str] = []
    watchlist_results: list[str] = []
    limitations: list[str] = []

    for row in evidence_rows:
        metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
        alert = alert_map.get(int(row.alert_id)) if row.alert_id else None
        camera_id = (str(alert.cam_id) if alert and alert.cam_id else _metadata_camera_id(metadata) or incident.cam_id)
        camera = camera_map.get(str(camera_id)) if camera_id else None
        camera_name, location = _camera_label(camera, camera_id)
        event_time = _utc(
            (alert.source_timestamp or alert.created_at) if alert else row.created_at
        )
        confidence = _confidence(
            getattr(alert, "watchlist_match_confidence", None) if alert else metadata.get("confidence", metadata.get("match_score"))
        )
        plate = _extract_plate(alert, metadata)
        if plate and plate not in plates:
            plates.append(plate)
        watchlist = _safe_text(
            (getattr(alert, "watchlist_status", None) or getattr(alert, "watchlist_category", None)) if alert else metadata.get("watchlist_status"),
            120,
        ) or None
        if watchlist and watchlist not in watchlist_results:
            watchlist_results.append(watchlist)

        result_bits = [plate, watchlist]
        result = " / ".join(x for x in result_bits if x) or _safe_text(row.description, 220) or _safe_text(row.evidence_type, 80)
        timeline.append(TimelineItem(
            event_time=event_time,
            event_type=_safe_text(getattr(alert, "rule", None) if alert else row.evidence_type, 100) or "EVIDENCE",
            camera_id=_safe_text(camera_id, 100) or None,
            camera_name=camera_name,
            result=result,
            confidence=confidence,
            timestamp_source=_safe_text(getattr(alert, "timestamp_source", None) if alert else metadata.get("timestamp_source"), 80) or None,
            timestamp_quality=_safe_text(getattr(alert, "timestamp_quality", None) if alert else metadata.get("timestamp_quality"), 80) or None,
            source_pts_seconds=getattr(alert, "source_pts_seconds", None) if alert else metadata.get("source_pts_seconds"),
            alert_id=int(alert.id) if alert else None,
            evidence_id=int(row.id),
        ))

        snapshot = snapshot_map.get(int(row.snapshot_id)) if row.snapshot_id else None
        prepared_bytes = None
        image_type = None
        snapshot_sha = None
        if snapshot is not None:
            try:
                prepared_bytes, image_type, computed_sha = prepare_snapshot_for_report(snapshot.image_data, snapshot.image_type)
                stored_sha = str(snapshot.sha256 or "").strip().lower()
                if stored_sha and stored_sha != computed_sha:
                    limitations.append(
                        f"Evidence {row.id}: stored snapshot SHA-256 did not match the current evidence bytes; the computed source hash is reported and integrity review is required."
                    )
                snapshot_sha = computed_sha
            except EvidenceImageError:
                limitations.append(f"Evidence {row.id}: snapshot could not be embedded because it failed report image validation.")
                snapshot_sha = snapshot.sha256 or None
        else:
            evidence_role = str(metadata.get("evidence_role") or "").strip().upper()
            is_watchlist_reference = (
                str(row.evidence_type or "").strip().upper() == "WATCHLIST_REFERENCE"
                or evidence_role == "WATCHLIST_REFERENCE"
            )
            if is_watchlist_reference:
                entry_id = _person_watchlist_entry_id(incident, row, alert)
                entry = reference_entry_map.get(entry_id) if entry_id is not None else None
                if entry is not None and entry.reference_image_data:
                    try:
                        reference_bytes = decrypt_bytes(entry.reference_image_data)
                        prepared_bytes, image_type, snapshot_sha = prepare_snapshot_for_report(
                            reference_bytes,
                            entry.reference_image_content_type,
                        )
                    except (EvidenceImageError, RuntimeError, ValueError):
                        limitations.append(
                            f"Evidence {row.id}: the protected watchlist reference image could not be embedded in the report."
                        )
                else:
                    limitations.append(
                        f"Evidence {row.id}: no tenant-owned watchlist reference image was available at export time."
                    )

        object_type = _safe_text(row.object_type, 50) or "Evidence"
        title = _safe_text(row.evidence_type, 80).replace("_", " ").title() or f"{object_type} evidence"
        if plate:
            subject = f"{object_type.title()} - plate {plate}"
        else:
            subject = _safe_text(row.object_reference, 150) or object_type.title()
        exhibits.append(Exhibit(
            exhibit_id=f"E-{len(exhibits)+1:03d}",
            title=title,
            evidence_id=int(row.id),
            alert_id=int(row.alert_id) if row.alert_id else None,
            snapshot_id=int(row.snapshot_id) if row.snapshot_id else None,
            event_time=event_time,
            camera_id=_safe_text(camera_id, 100) or None,
            camera_name=camera_name,
            location=location,
            subject=subject,
            rule=_safe_text(getattr(alert, "rule", None), 100) if alert else _safe_text(metadata.get("rule"), 100) or None,
            severity=_safe_text(getattr(alert, "level", None), 40) if alert else _safe_text(metadata.get("severity"), 40) or None,
            confidence=confidence,
            watchlist_result=watchlist,
            description=_safe_text(row.description, 1000) or None,
            snapshot_sha256=snapshot_sha,
            image_bytes=prepared_bytes,
            image_type=image_type,
            metadata={},
        ))

    timeline.sort(key=lambda item: item.event_time or datetime.min.replace(tzinfo=timezone.utc))
    camera_names = []
    locations = []
    for camera_id in sorted(camera_ids):
        camera_name, location = _camera_label(camera_map.get(camera_id), camera_id)
        if camera_name and camera_name not in camera_names:
            camera_names.append(camera_name)
        if location and location not in locations:
            locations.append(location)
    if camera_ids and not locations:
        limitations.append("Camera geolocation/location name was not configured for the evidence included in this report.")
    if not exhibits:
        limitations.append("No incident-evidence rows were available at export time; the report contains incident metadata only.")
    if any((item.timestamp_source or "").upper().startswith("LEGACY") or (item.timestamp_quality or "").upper().startswith("LEGACY") for item in timeline):
        limitations.append("One or more event times are legacy ingestion times and must not be represented as source PTS/frame time.")
    limitations.extend([
        "This document records system-generated analytical results and does not by itself establish identity, ownership, intent, or guilt.",
        "Original source media and the platform evidence store must be preserved and reviewed before evidentiary reliance.",
    ])

    severity = _safe_text(incident.evaluation_severity, 40) or "INFO"
    description = _safe_text(incident.evaluation_reason or incident.evidence, 3000) or "No incident description was recorded."
    scope_parts = []
    domain = _safe_text(incident.incident_type, 100).upper()
    if "VEHICLE" in domain or "ANPR" in domain:
        scope_parts.extend(["Vehicle", "ANPR"])
    if "PERSON" in domain or "FACE" in domain:
        scope_parts.extend(["Person", "Face recognition"])
    if "BEHAV" in domain or "SUSPIC" in domain or "CHAIN" in domain or "VIOLENCE" in domain:
        scope_parts.append("Behavioural analytics")
    if watchlist_results or "WATCHLIST" in domain:
        scope_parts.append("Watchlist")
    if journey:
        scope_parts.append("Multi-camera correlation / journey")
    if not scope_parts:
        scope_parts.append("Incident evidence")

    model_context = _model_context(incident, evidence_rows, snapshot_map)
    provenance_rows = [
        ("Original evidence", "Retained separately in the INTEL-I evidence store; this PDF contains derived report exhibits only."),
        ("Timestamp fields", "Source frame time/source PTS, timestamp quality/source, and ingestion time are kept as distinct fields when available."),
        ("Derived exhibit integrity", "SHA-256 is recorded for each embedded source snapshot before report-safe re-encoding."),
        ("Processing context", "INTEL-I analytics, correlation, watchlist and incident evaluation data are included only when present in authoritative stored records."),
        ("Export audit", "The export is recorded in the INTEL-I audit log with report ID, export ID, classification, evidence manifest hash and final PDF SHA-256."),
        ("Retention and disclosure", "Apply the organisation's approved evidence-retention and authorised-disclosure policy."),
    ]

    manifest_payload = {
        "incident": {
            "id": int(incident.id),
            "user_id": int(user_id),
            "incident_type": incident.incident_type,
            "status": incident.status,
            "primary_track_id": incident.primary_track_id,
            "global_vehicle_id": getattr(incident, "global_vehicle_id", None),
            "started_at": _utc(incident.started_at),
            "ended_at": _utc(incident.ended_at),
            "evaluation_status": incident.evaluation_status,
            "evaluation_confidence": incident.evaluation_confidence,
            "evaluation_severity": incident.evaluation_severity,
            "evaluation_reason": incident.evaluation_reason,
            "evaluation_evidence": incident.evaluation_evidence,
        },
        "alerts": [
            {
                "id": int(alert.id),
                "cam_id": alert.cam_id,
                "rule": alert.rule,
                "level": alert.level,
                "plate": alert.plate,
                "watchlist_entry_id": alert.watchlist_entry_id,
                "watchlist_category": alert.watchlist_category,
                "watchlist_status": alert.watchlist_status,
                "watchlist_match_type": alert.watchlist_match_type,
                "watchlist_match_confidence": alert.watchlist_match_confidence,
                "watchlist_version": alert.watchlist_version,
                "source_timestamp": _utc(alert.source_timestamp),
                "created_at": _utc(alert.created_at),
                "timestamp_source": alert.timestamp_source,
                "timestamp_quality": alert.timestamp_quality,
                "source_pts_seconds": alert.source_pts_seconds,
            }
            for alert in sorted(alerts, key=lambda item: int(item.id))
        ],
        "evidence": [
            {
                "id": int(row.id),
                "alert_id": int(row.alert_id) if row.alert_id else None,
                "snapshot_id": int(row.snapshot_id) if row.snapshot_id else None,
                "evidence_type": row.evidence_type,
                "object_type": row.object_type,
                "object_reference": row.object_reference,
                "description": row.description,
                "created_at": _utc(row.created_at),
                "snapshot_sha256": next((x.snapshot_sha256 for x in exhibits if x.evidence_id == int(row.id)), None),
            }
            for row in evidence_rows
        ],
    }

    generated_at = datetime.now(timezone.utc)
    report_id = new_report_id()
    export_id = f"EXP-{uuid.uuid4().hex[:12].upper()}"
    return EvidenceReportData(
        report_id=report_id,
        export_id=export_id,
        incident_id=int(incident.id),
        generated_at=generated_at,
        generated_by=_safe_text(generated_by, 200) or "INTEL-I authorised user",
        classification=policy.classification_label,
        classification_code=policy.classification_code,
        case_id=f"INC-{incident.id}",
        report_version="2.0",
        scope=" / ".join(dict.fromkeys(scope_parts)),
        incident_type=_safe_text(incident.incident_type, 100),
        incident_status=_safe_text(incident.status, 50),
        severity=severity,
        description=description,
        evaluation_status=_safe_text(incident.evaluation_status, 50) or None,
        evaluation_confidence=_confidence(incident.evaluation_confidence),
        evaluation_reason=_safe_text(incident.evaluation_reason, 2500) or None,
        camera_ids=sorted(camera_ids),
        camera_names=camera_names,
        locations=locations,
        started_at=_utc(incident.started_at),
        ended_at=_utc(incident.ended_at),
        primary_track_id=_safe_text(incident.primary_track_id, 100) or None,
        global_vehicle_id=_safe_text(getattr(incident, "global_vehicle_id", None), 100) or None,
        plate_candidates=plates,
        watchlist_results=watchlist_results,
        timeline=timeline,
        journey=journey,
        exhibits=exhibits,
        limitations=list(dict.fromkeys(limitations)),
        provenance_rows=provenance_rows,
        source_manifest_sha256=canonical_manifest_sha256(manifest_payload),
        model_context=model_context,
    )
