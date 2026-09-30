"""Analytics report preview and export endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy import or_
from sqlalchemy.orm import Session

from auth.auth import get_current_user
from db.database import getDB
from db.intelligence_model import PlateObservation, VehicleObservation
from db.model import Alert, AuditLog, Camera, CameraIntegration, CameraTimeSync, Snapshot, User
from security.rbac import require_permissions
from evidence.collector import collect_incident_report_data
from evidence.policy import allowed_classifications, resolve_report_policy
from evidence.service import build_incident_evidence_report

from services.analyticsReporting import (
    collect_analytics_rows,
    new_report_id,
    public_rows,
    render_csv,
    render_pdf,
    report_summary,
)


router = APIRouter(prefix="/api/reports", tags=["analytics-reports"])


@router.get("/government-e2e/readiness")
def government_e2e_readiness(
    hours: int = Query(default=24, ge=1, le=168),
    db: Session = Depends(getDB),
    current_user: User = Depends(require_permissions("intelligence.view")),
):
    """Evidence-based readiness gate; never marks a live demonstration complete without data."""
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
    cameras = db.query(Camera).filter(
        Camera.user_id == current_user.id,
        Camera.external_provider == "sentinel",
    ).all()
    camera_ids = [camera.cam_id for camera in cameras]
    integration_count = db.query(CameraIntegration).filter(
        CameraIntegration.user_id == current_user.id,
        CameraIntegration.provider_type == "sentinel",
        CameraIntegration.last_sync_status == "SUCCEEDED",
    ).count()
    online_count = sum(
        1 for camera in cameras
        if str(camera.connection_state or camera.external_live_status or "").upper() in {"ONLINE", "LIVE", "DEGRADED"}
    )
    pts_count = anpr_count = detection_count = alert_count = watchlist_count = evidence_count = 0
    if camera_ids:
        pts_count = db.query(CameraTimeSync).filter(
            CameraTimeSync.camera_id.in_(camera_ids),
            or_(
                CameraTimeSync.timestamp_quality == "SOURCE_PTS",
                CameraTimeSync.sync_method == "PTS",
            ),
        ).count()
        detection_count = db.query(VehicleObservation).filter(
            VehicleObservation.camera_id.in_(camera_ids),
            VehicleObservation.frame_timestamp >= cutoff,
        ).count()
        anpr_count = db.query(PlateObservation).filter(
            PlateObservation.camera_id.in_(camera_ids),
            PlateObservation.frame_timestamp >= cutoff,
        ).count()
        alerts = db.query(Alert).filter(
            Alert.user_id == current_user.id,
            Alert.cam_id.in_(camera_ids),
            Alert.created_at >= cutoff,
        )
        alert_count = alerts.count()
        watchlist_count = alerts.filter(or_(
            Alert.watchlist_entry_id.isnot(None),
            Alert.watchlist_status.isnot(None),
            Alert.rule.ilike("%WATCHLIST%"),
        )).count()
        evidence_count = db.query(Snapshot).join(Alert, Snapshot.alert_id == Alert.id).filter(
            Alert.user_id == current_user.id,
            Alert.cam_id.in_(camera_ids),
            Alert.created_at >= cutoff,
        ).count()
    counts = {
        "successful_sentinel_integrations": integration_count,
        "sentinel_cameras": len(cameras),
        "online_cameras": online_count,
        "pts_aligned_cameras": pts_count,
        "vehicle_detections": detection_count,
        "anpr_observations": anpr_count,
        "alerts": alert_count,
        "watchlist_results": watchlist_count,
        "evidence_items": evidence_count,
    }
    gates = {
        "catalogue_synchronized": integration_count > 0 and len(cameras) > 0,
        "camera_online": online_count > 0,
        "pts_timing_verified": pts_count > 0,
        "vehicle_detection_verified": detection_count > 0,
        "anpr_verified": anpr_count > 0,
        "watchlist_evaluation_verified": watchlist_count > 0,
        "alert_verified": alert_count > 0,
        "evidence_verified": evidence_count > 0,
        "report_export_available": True,
    }
    return {
        "complete": all(gates.values()),
        "window_hours": hours,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "gates": gates,
        "counts": counts,
    }


def _range(start: datetime | None, end: datetime | None) -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    end_value = end or now
    start_value = start or (end_value - timedelta(days=7))
    if start_value.tzinfo is None:
        start_value = start_value.replace(tzinfo=timezone.utc)
    if end_value.tzinfo is None:
        end_value = end_value.replace(tzinfo=timezone.utc)
    if start_value > end_value:
        raise HTTPException(status_code=422, detail="start must be before end")
    if (end_value - start_value) > timedelta(days=366):
        raise HTTPException(status_code=422, detail="Report range cannot exceed 366 days")
    return start_value, end_value


def _rows(db, user_id, start, end, camera_id, watchlist_only, limit):
    start_value, end_value = _range(start, end)
    rows = collect_analytics_rows(
        db,
        user_id=user_id,
        start=start_value,
        end=end_value,
        camera_id=camera_id,
        watchlist_only=watchlist_only,
        limit=limit,
    )
    return rows, start_value, end_value


@router.get("/analytics/preview")
def preview_report(
    start: datetime | None = Query(default=None),
    end: datetime | None = Query(default=None),
    camera_id: str | None = Query(default=None, max_length=100),
    watchlist_only: bool = Query(default=False),
    limit: int = Query(default=200, ge=1, le=1000),
    db: Session = Depends(getDB),
    current_user: User = Depends(require_permissions("intelligence.view")),
):
    rows, start_value, end_value = _rows(db, current_user.id, start, end, camera_id, watchlist_only, limit)
    return {
        "summary": report_summary(rows),
        "filters": {
            "start": start_value.isoformat(),
            "end": end_value.isoformat(),
            "camera_id": camera_id,
            "watchlist_only": watchlist_only,
        },
        "rows": public_rows(rows),
    }


@router.get("/analytics/export")
def export_report(
    format: Literal["pdf", "csv"] = Query(default="pdf"),
    start: datetime | None = Query(default=None),
    end: datetime | None = Query(default=None),
    camera_id: str | None = Query(default=None, max_length=100),
    watchlist_only: bool = Query(default=False),
    limit: int = Query(default=5000, ge=1, le=10000),
    db: Session = Depends(getDB),
    current_user: User = Depends(require_permissions("intelligence.view")),
):
    rows, start_value, end_value = _rows(db, current_user.id, start, end, camera_id, watchlist_only, limit)
    report_id = new_report_id()
    filters = {
        "start": start_value.isoformat(),
        "end": end_value.isoformat(),
        "camera_id": camera_id,
        "watchlist_only": watchlist_only,
    }
    if format == "csv":
        payload = render_csv(rows, report_id)
        media_type = "text/csv; charset=utf-8"
    else:
        try:
            payload = render_pdf(rows, report_id, filters)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Unable to generate PDF report") from exc
        media_type = "application/pdf"

    db.add(AuditLog(
        user_id=current_user.id,
        action="ANALYTICS_REPORT_EXPORTED",
        resource_type="analytics_report",
        resource_id=report_id,
        details={"format": format, "record_count": len(rows), "filters": filters},
    ))
    db.commit()
    filename = f"intel-i-analytics-{report_id}.{format}"
    return Response(
        content=payload,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store, private",
            "X-Content-Type-Options": "nosniff",
            "X-Report-ID": report_id,
        },
    )


@router.get("/evidence/classifications")
def evidence_report_classifications(
    current_user: User = Depends(require_permissions("intelligence.view")),
):
    """Return the server-controlled classification codes available for evidence exports."""
    return {"classifications": allowed_classifications()}


@router.get("/incidents/{incident_id}/evidence/preview")
def preview_incident_evidence_report(
    incident_id: int,
    classification: str = Query(default="RESTRICTED", min_length=3, max_length=32),
    db: Session = Depends(getDB),
    current_user: User = Depends(require_permissions("intelligence.view")),
):
    """Return a tenant-scoped, image-free preview of the dynamic evidence-report data."""
    try:
        resolve_report_policy(classification)
        data = collect_incident_report_data(
            db,
            incident_id=incident_id,
            user_id=current_user.id,
            generated_by=current_user.full_name or current_user.email,
            classification=classification,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Incident not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return data.public_dict()


@router.get("/incidents/{incident_id}/evidence/export")
def export_incident_evidence_report(
    incident_id: int,
    request: Request,
    classification: str = Query(default="RESTRICTED", min_length=3, max_length=32),
    db: Session = Depends(getDB),
    current_user: User = Depends(require_permissions("intelligence.view")),
):
    """Generate a watermarked, tenant-scoped, auditable incident evidence PDF."""
    try:
        resolve_report_policy(classification)
        data, payload, pdf_sha256 = build_incident_evidence_report(
            db,
            incident_id=incident_id,
            user_id=current_user.id,
            generated_by=current_user.full_name or current_user.email,
            classification=classification,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Incident not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to generate incident evidence PDF") from exc

    # The final delivered bytes are hashed after all branding/watermarking is applied.
    # The report itself carries the source-evidence manifest hash; the final PDF hash
    # is persisted in the audit ledger because a PDF cannot contain its own final hash.
    details = {
        "report_id": data.report_id,
        "export_id": data.export_id,
        "incident_id": int(incident_id),
        "classification": data.classification,
        "report_version": data.report_version,
        "source_manifest_sha256": data.source_manifest_sha256,
        "pdf_sha256": pdf_sha256,
        "pdf_size_bytes": len(payload),
        "exhibit_count": len(data.exhibits),
        "camera_count": len(data.camera_ids),
        "snapshot_hashes": [
            {"exhibit_id": item.exhibit_id, "snapshot_id": item.snapshot_id, "sha256": item.snapshot_sha256}
            for item in data.exhibits
            if item.snapshot_sha256
        ],
    }
    client_host = request.client.host if request.client else None
    db.add(AuditLog(
        user_id=current_user.id,
        action="INCIDENT_EVIDENCE_REPORT_EXPORTED",
        resource_type="incident_evidence_report",
        resource_id=data.report_id,
        details=details,
        source_ip=client_host,
    ))
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        # Fail closed: do not release an evidence export that could not be audited.
        raise HTTPException(status_code=503, detail="Evidence export audit could not be persisted") from exc

    filename = f"intel-i-evidence-incident-{int(incident_id)}-{data.report_id}.pdf"
    return Response(
        content=payload,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store, private, max-age=0",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
            "X-Report-ID": data.report_id,
            "X-Export-ID": data.export_id,
            "X-Evidence-Manifest-SHA256": data.source_manifest_sha256,
            "X-PDF-SHA256": pdf_sha256,
        },
    )
