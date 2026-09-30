"""Authenticated APIs for Sentinel and VMS/NVR catalogue management."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth.auth import get_current_user
from connectors.catalogue_manager import normalize_catalogue_provider
from db.database import getDB
from db.model import AuditLog, CameraIntegration, CameraIntegrationSyncRun, User
from security.rbac import ADMIN_ROLES, normalize_role
from services.cameraIntegration import (
    discover_integration,
    encrypt_integration_config,
    integration_view,
    sync_integration,
)


router = APIRouter(prefix="/api/integrations", tags=["camera-integrations"])
ingest_router = APIRouter(prefix="/api/ingest", tags=["camera-ingest"])


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class IntegrationCreate(StrictModel):
    name: str = Field(min_length=2, max_length=150)
    provider_type: Literal["sentinel", "generic_vms", "generic_nvr"]
    config: dict[str, Any]
    enabled: bool = True
    auto_sync: bool = True
    sync_interval_seconds: int = Field(default=300, ge=60, le=86400)

    @field_validator("config")
    @classmethod
    def validate_config(cls, value):
        if not isinstance(value, dict) or not value:
            raise ValueError("Integration configuration is required")
        if not value.get("base_url"):
            raise ValueError("Integration base_url is required")
        return value


class IntegrationUpdate(StrictModel):
    name: str | None = Field(default=None, min_length=2, max_length=150)
    config: dict[str, Any] | None = None
    enabled: bool | None = None
    auto_sync: bool | None = None
    sync_interval_seconds: int | None = Field(default=None, ge=60, le=86400)


def _integration_admin(current_user: User = Depends(get_current_user)) -> User:
    if normalize_role(current_user.role) not in ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="Administrator access is required for camera integrations")
    return current_user


def _owned(db: Session, integration_id: int, user_id: int) -> CameraIntegration:
    row = db.query(CameraIntegration).filter(
        CameraIntegration.id == integration_id,
        CameraIntegration.user_id == user_id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Camera integration not found")
    return row


@router.get("")
def list_integrations(
    db: Session = Depends(getDB),
    current_user: User = Depends(_integration_admin),
):
    rows = db.query(CameraIntegration).filter(
        CameraIntegration.user_id == current_user.id,
    ).order_by(CameraIntegration.created_at.desc()).all()
    return {"count": len(rows), "integrations": [integration_view(row) for row in rows]}


@router.post("", status_code=status.HTTP_201_CREATED)
def create_integration(
    payload: IntegrationCreate,
    db: Session = Depends(getDB),
    current_user: User = Depends(_integration_admin),
):
    provider = normalize_catalogue_provider(payload.provider_type)
    encrypted, fingerprint = encrypt_integration_config(payload.config)
    row = CameraIntegration(
        user_id=current_user.id,
        name=payload.name,
        provider_type=provider,
        config_encrypted=encrypted,
        config_fingerprint=fingerprint,
        enabled=payload.enabled,
        auto_sync=payload.auto_sync,
        sync_interval_seconds=payload.sync_interval_seconds,
        last_sync_status="NOT_SYNCED",
    )
    try:
        db.add(row)
        db.commit()
        db.refresh(row)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="An integration with this name already exists") from exc
    db.add(AuditLog(
        user_id=current_user.id,
        action="CAMERA_INTEGRATION_CREATED",
        resource_type="camera_integration",
        resource_id=str(row.id),
        details={"provider_type": provider, "auto_sync": bool(row.auto_sync)},
    ))
    db.commit()
    return integration_view(row)


@router.patch("/{integration_id}")
def update_integration(
    integration_id: int,
    payload: IntegrationUpdate,
    db: Session = Depends(getDB),
    current_user: User = Depends(_integration_admin),
):
    if not payload.model_fields_set:
        raise HTTPException(status_code=422, detail="At least one field is required")
    row = _owned(db, integration_id, current_user.id)
    if payload.name is not None:
        row.name = payload.name
    if payload.config is not None:
        encrypted, fingerprint = encrypt_integration_config(payload.config)
        row.config_encrypted = encrypted
        row.config_fingerprint = fingerprint
        row.last_sync_status = "CONFIGURATION_CHANGED"
    if payload.enabled is not None:
        row.enabled = payload.enabled
    if payload.auto_sync is not None:
        row.auto_sync = payload.auto_sync
    if payload.sync_interval_seconds is not None:
        row.sync_interval_seconds = payload.sync_interval_seconds
    try:
        db.commit()
        db.refresh(row)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="An integration with this name already exists") from exc
    db.add(AuditLog(
        user_id=current_user.id,
        action="CAMERA_INTEGRATION_UPDATED",
        resource_type="camera_integration",
        resource_id=str(row.id),
        details={"changed_fields": sorted(payload.model_fields_set)},
    ))
    db.commit()
    return integration_view(row)


@router.delete("/{integration_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_integration(
    integration_id: int,
    db: Session = Depends(getDB),
    current_user: User = Depends(_integration_admin),
):
    row = _owned(db, integration_id, current_user.id)
    for camera in list(row.cameras):
        camera.integration_id = None
        camera.externally_managed = False
    db.add(AuditLog(
        user_id=current_user.id,
        action="CAMERA_INTEGRATION_DELETED",
        resource_type="camera_integration",
        resource_id=str(row.id),
        details={"provider_type": row.provider_type, "retained_camera_count": len(row.cameras)},
    ))
    db.delete(row)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{integration_id}/discover")
def discover_cameras(
    integration_id: int,
    db: Session = Depends(getDB),
    current_user: User = Depends(_integration_admin),
):
    row = _owned(db, integration_id, current_user.id)
    if not row.enabled:
        raise HTTPException(status_code=409, detail="Camera integration is disabled")
    try:
        result = discover_integration(row)
        return result.safe_dict()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Camera catalogue discovery failed ({type(exc).__name__})") from exc


def _run_sync(integration_id: int, db: Session, current_user: User, trigger: str):
    row = _owned(db, integration_id, current_user.id)
    if not row.enabled:
        raise HTTPException(status_code=409, detail="Camera integration is disabled")
    try:
        result = sync_integration(db, row, trigger=trigger)
        db.add(AuditLog(
            user_id=current_user.id,
            action="CAMERA_INTEGRATION_SYNCHRONIZED",
            resource_type="camera_integration",
            resource_id=str(row.id),
            details={key: result.get(key) for key in (
                "provider_type", "status", "discovered_count", "created_count",
                "updated_count", "unchanged_count", "skipped_count", "offline_count",
            )},
        ))
        db.commit()
        return result
    except RuntimeError as exc:
        if "already running" in str(exc):
            raise HTTPException(status_code=409, detail="Camera catalogue synchronization is already running") from exc
        raise HTTPException(status_code=502, detail=f"Camera catalogue synchronization failed ({type(exc).__name__})") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Camera catalogue synchronization failed ({type(exc).__name__})") from exc


@router.post("/{integration_id}/sync")
def sync_cameras(
    integration_id: int,
    db: Session = Depends(getDB),
    current_user: User = Depends(_integration_admin),
):
    return _run_sync(integration_id, db, current_user, "manual")


@ingest_router.post("/sync")
def sync_ingest_catalogue(
    integration_id: int = Query(..., ge=1),
    db: Session = Depends(getDB),
    current_user: User = Depends(_integration_admin),
):
    """Compatibility endpoint for Sentinel /api/ingest catalogue synchronization."""
    return _run_sync(integration_id, db, current_user, "api_ingest")


@router.get("/{integration_id}/runs")
def list_sync_runs(
    integration_id: int,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(getDB),
    current_user: User = Depends(_integration_admin),
):
    _owned(db, integration_id, current_user.id)
    rows = db.query(CameraIntegrationSyncRun).filter(
        CameraIntegrationSyncRun.integration_id == integration_id,
    ).order_by(CameraIntegrationSyncRun.started_at.desc()).limit(limit).all()
    return {
        "count": len(rows),
        "runs": [
            {
                "id": row.id,
                "trigger": row.trigger,
                "status": row.status,
                "discovered_count": row.discovered_count,
                "created_count": row.created_count,
                "updated_count": row.updated_count,
                "skipped_count": row.skipped_count,
                "offline_count": row.offline_count,
                "source_revision": row.source_revision,
                "error_code": row.error_code,
                "error_message": row.error_message,
                "started_at": row.started_at.isoformat() if row.started_at else None,
                "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            }
            for row in rows
        ],
    }
