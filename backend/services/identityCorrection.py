"""Operator-controlled global vehicle identity correction.

Corrections are durable in PostgreSQL and mirrored into the in-memory
correlator. Every mutation is auditable. This module never silently merges
identities based on plate alone.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

from sqlalchemy.orm import Session

from db.model import AuditLog, Camera
from db.intelligence_model import Vehicle, VehicleObservation
from vehicleCorrelation import get_global_vehicle, reassign_global_identity


def _clean_id(value: str) -> str:
    value = str(value or "").strip().upper()
    if not value or len(value) > 100:
        raise ValueError("Invalid global vehicle ID")
    return value


def correct_identity(
    db: Session,
    *,
    user_id: int,
    action: str,
    source_global_vehicle_id: str,
    target_global_vehicle_id: str | None = None,
    observation_ids: list[int] | None = None,
    reason: str,
    source_ip: str | None = None,
) -> dict[str, Any]:
    action = str(action or "").strip().upper()
    if action not in {"SPLIT", "MERGE", "CORRECT", "MARK_UNCERTAIN"}:
        raise ValueError("Unsupported identity correction action")
    reason = str(reason or "").strip()
    if len(reason) < 5 or len(reason) > 1000:
        raise ValueError("A correction reason of 5-1000 characters is required")

    source_id = _clean_id(source_global_vehicle_id)
    target_id = _clean_id(target_global_vehicle_id) if target_global_vehicle_id else None
    if action == "MERGE" and not target_id:
        raise ValueError("MERGE requires target_global_vehicle_id")
    if action == "MERGE" and target_id == source_id:
        raise ValueError("Cannot merge an identity into itself")

    source = db.query(Vehicle).filter(Vehicle.global_vehicle_id == source_id).first()
    if not source:
        raise ValueError("Source global vehicle not found")
    if target_id:
        target = db.query(Vehicle).filter(Vehicle.global_vehicle_id == target_id).first()
        if not target:
            raise ValueError("Target global vehicle not found")

    ids = [int(x) for x in (observation_ids or [])]
    if len(ids) > 5000:
        raise ValueError("Too many observations in one correction")

    query = db.query(VehicleObservation).filter(VehicleObservation.global_vehicle_id == source_id)
    if ids:
        query = query.filter(VehicleObservation.id.in_(ids))
    observations = query.all()
    owned_camera_ids = {str(c.cam_id) for c in db.query(Camera).filter(Camera.user_id == user_id).all()}
    if any(str(o.camera_id) not in owned_camera_ids for o in observations):
        raise ValueError("Identity contains observations outside the authenticated user's cameras")
    if ids and len(observations) != len(set(ids)):
        raise ValueError("One or more observations do not belong to source identity")
    if not observations:
        raise ValueError("No observations selected for correction")

    if action == "MERGE":
        new_id = target_id
        for obs in observations:
            obs.global_vehicle_id = new_id
            obs.vehicle_id = target.id
        source.status = "MERGED"
        source.metadata_json = {**(source.metadata_json or {}), "merged_into": new_id}
    elif action in {"SPLIT", "CORRECT"}:
        new_id = target_id or f"GV-CORR-{uuid.uuid4().hex[:20].upper()}"
        target = db.query(Vehicle).filter(Vehicle.global_vehicle_id == new_id).first()
        if not target:
            target = Vehicle(global_vehicle_id=new_id, status="ACTIVE", metadata_json={"created_by_correction": True})
            db.add(target)
            db.flush()
        for obs in observations:
            obs.global_vehicle_id = new_id
            obs.vehicle_id = target.id
    else:
        new_id = source_id
        source.status = "UNCERTAIN"

    audit = AuditLog(
        user_id=user_id,
        action=f"VEHICLE_IDENTITY_{action}",
        resource_type="GLOBAL_VEHICLE",
        resource_id=source_id,
        details={
            "target_global_vehicle_id": new_id,
            "observation_ids": ids,
            "reason": reason,
            "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        source_ip=source_ip,
    )
    db.add(audit)
    db.commit()

    # Keep the live correlator consistent with durable state. This is best-effort
    # after commit; the database remains the source of truth across restarts.
    try:
        reassign_global_identity(source_id, new_id, [f"{o.camera_id}:{o.local_track_id}" for o in observations if o.local_track_id])
    except Exception:
        pass

    return {
        "action": action,
        "source_global_vehicle_id": source_id,
        "target_global_vehicle_id": new_id,
        "observation_count": len(observations),
        "audit_log_id": int(audit.id),
        "status": "APPLIED",
    }
