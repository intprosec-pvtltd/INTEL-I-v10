"""Production camera-onboarding and paginated fleet-management APIs."""
from __future__ import annotations

import asyncio
import io
import os
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth.auth import get_current_user
from db.database import getDB
from db.model import AuditLog, Camera, CameraAIProfile, CameraCredential, CameraGroup, CameraOnboardingItem, CameraOnboardingJob, User
from security.cameraSource import encrypt_camera_source
from security.connectorConfig import encrypt_connector_config
from security.rbac import ADMIN_ROLES, normalize_role
from services.cameraOnboarding import create_job, discover_private_network, job_view, run_job, sanitize_url, validate_authorized_network

router = APIRouter(prefix="/api", tags=["camera-onboarding"])
_progress_sender = None


def set_onboarding_progress_sender(sender):
    global _progress_sender
    _progress_sender = sender


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RTSPImport(StrictModel):
    urls: list[str] = Field(min_length=1, max_length=5000)
    credential_profile_id: int | None = None
    group_id: int | None = None
    ai_profile_id: int | None = None


class DiscoveryRequest(StrictModel):
    network: str
    ports: list[int] = Field(default_factory=lambda: [554, 8554, 80, 8080], max_length=16)
    credential_profile_id: int | None = None

    @field_validator("network")
    @classmethod
    def private_network(cls, value):
        validate_authorized_network(value)
        return value


class CredentialCreate(StrictModel):
    name: str = Field(min_length=2, max_length=150)
    username: str | None = Field(default=None, max_length=255)
    password: str = Field(min_length=1, max_length=1024)
    auth_type: Literal["basic", "digest", "rtsp", "onvif"] = "rtsp"


class GroupCreate(StrictModel):
    name: str = Field(min_length=1, max_length=150)
    parent_id: int | None = None
    group_type: str = Field(default="logical", max_length=50)
    metadata: dict[str, Any] | None = None


class AIProfileCreate(StrictModel):
    name: str = Field(min_length=2, max_length=150)
    configuration: dict[str, bool] = Field(default_factory=dict)
    processing_fps: float = Field(default=5, gt=0, le=30)
    preferred_stream: Literal["main", "substream"] = "substream"
    evidence_enabled: bool = True


class BulkAction(StrictModel):
    camera_ids: list[str] = Field(min_length=1, max_length=5000)
    action: Literal["enable", "disable", "change_credential", "change_group", "assign_ai_profile", "enable_analytics", "disable_analytics", "health_check", "refresh_metadata", "reconnect", "delete"]
    value: int | str | None = None
    confirmed: bool = False


def admin(user: User = Depends(get_current_user)) -> User:
    if os.getenv("CAMERA_ONBOARDING_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        raise HTTPException(status_code=503, detail="Camera onboarding is disabled")
    if normalize_role(user.role) not in ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="Administrator access is required")
    return user


def _start(job_id: str):
    if os.getenv("CAMERA_ONBOARDING_INLINE_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"}:
        asyncio.create_task(run_job(job_id, sender=_progress_sender))


def _items_from_rtsp(payload: RTSPImport):
    rows = []
    seen = set()
    for index, raw in enumerate(payload.urls, start=1):
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        rows.append({"camera_name": f"Imported Camera {index}", "source": value, "source_type": "rtsp", "credential_profile_id": payload.credential_profile_id, "group_id": payload.group_id, "ai_profile_id": payload.ai_profile_id})
    return rows


@router.post("/camera-onboarding/import/rtsp", status_code=202)
async def import_rtsp(payload: RTSPImport, db: Session = Depends(getDB), user: User = Depends(admin)):
    rows = _items_from_rtsp(payload)
    if not rows:
        raise HTTPException(status_code=400, detail="No unique RTSP URLs were provided")
    job = create_job(db, user.id, "rtsp", rows)
    _start(job.id)
    return job_view(job)


@router.post("/camera-onboarding/discover", status_code=202)
async def discover(payload: DiscoveryRequest, db: Session = Depends(getDB), user: User = Depends(admin)):
    rows = await discover_private_network(payload.network, payload.ports)
    for row in rows:
        row["credential_profile_id"] = payload.credential_profile_id
    job = create_job(db, user.id, "network_discovery", rows, {"network": payload.network, "reachable_endpoints": len(rows)})
    _start(job.id)
    return job_view(job)


@router.post("/camera-onboarding/import/csv", status_code=202)
async def import_inventory(file: UploadFile = File(...), db: Session = Depends(getDB), user: User = Depends(admin)):
    filename = (file.filename or "").lower()
    if not filename.endswith((".csv", ".xlsx")):
        raise HTTPException(status_code=400, detail="Only CSV and XLSX files are supported")
    content = await file.read()
    if len(content) > int(os.getenv("CAMERA_IMPORT_MAX_BYTES", "10485760")):
        raise HTTPException(status_code=413, detail="Camera inventory file is too large")
    try:
        frame = pd.read_csv(io.BytesIO(content), dtype=str) if filename.endswith(".csv") else pd.read_excel(io.BytesIO(content), dtype=str)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to read camera inventory") from exc
    frame = frame.fillna("")
    rows, invalid = [], []
    seen_ids, seen_urls = set(), set()
    for number, raw in enumerate(frame.to_dict(orient="records"), start=2):
        row = {str(k).strip().lower(): str(v).strip() for k, v in raw.items()}
        camera_id = row.get("camera_id") or row.get("cam_id")
        camera_name = row.get("camera_name") or camera_id
        source = row.get("rtsp_url") or row.get("source")
        problems = []
        if not camera_name or not source:
            problems.append("camera_id/name and source are required")
        if camera_id in seen_ids: problems.append("duplicate camera ID in file")
        if source in seen_urls: problems.append("duplicate source in file")
        try:
            lat = float(row["latitude"]) if row.get("latitude") else None
            lon = float(row["longitude"]) if row.get("longitude") else None
            if (lat is None) != (lon is None) or (lat is not None and not (-90 <= lat <= 90 and -180 <= lon <= 180)): problems.append("invalid coordinates")
        except ValueError:
            problems.append("invalid coordinates"); lat = lon = None
        if problems:
            invalid.append({"row": number, "camera_id": camera_id, "errors": problems})
            continue
        camera_id = camera_id or f"IMPORT_{number:06d}"
        seen_ids.add(camera_id); seen_urls.add(source)
        rows.append({"camera_id": camera_id, "camera_name": camera_name, "source": source, "source_type": row.get("stream_type") or "rtsp", "latitude": lat, "longitude": lon, "location": row.get("location"), "district": row.get("district"), "zone": row.get("zone")})
    if not rows:
        raise HTTPException(status_code=422, detail={"message": "No valid camera rows", "invalid": invalid[:100]})
    job = create_job(db, user.id, "inventory", rows, {"filename": file.filename, "valid": len(rows), "invalid": len(invalid), "invalid_preview": invalid[:100]})
    _start(job.id)
    return {**job_view(job), "valid": len(rows), "invalid": len(invalid), "invalid_preview": invalid[:100]}


@router.get("/camera-onboarding/jobs")
def jobs(limit: int = Query(20, ge=1, le=100), db: Session = Depends(getDB), user: User = Depends(get_current_user)):
    rows = db.query(CameraOnboardingJob).filter(CameraOnboardingJob.user_id == user.id).order_by(CameraOnboardingJob.created_at.desc()).limit(limit).all()
    return {"jobs": [job_view(row) for row in rows]}


@router.get("/camera-onboarding/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(getDB), user: User = Depends(get_current_user)):
    row = db.query(CameraOnboardingJob).filter(CameraOnboardingJob.id == job_id, CameraOnboardingJob.user_id == user.id).first()
    if not row: raise HTTPException(status_code=404, detail="Onboarding job not found")
    return job_view(row)


@router.get("/camera-onboarding/jobs/{job_id}/items")
def job_items(job_id: str, page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200), db: Session = Depends(getDB), user: User = Depends(get_current_user)):
    owned = db.query(CameraOnboardingJob.id).filter(CameraOnboardingJob.id == job_id, CameraOnboardingJob.user_id == user.id).first()
    if not owned: raise HTTPException(status_code=404, detail="Onboarding job not found")
    query = db.query(CameraOnboardingItem).filter(CameraOnboardingItem.job_id == job_id)
    total = query.count(); rows = query.order_by(CameraOnboardingItem.ordinal).offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "page": page, "page_size": page_size, "items": [{"id": row.id, "camera_id": row.camera_id, "camera_name": row.camera_name, "status": row.status, "error_code": row.error_code, "error_message": row.error_message, "metadata": row.result_metadata} for row in rows]}


@router.get("/camera-fleet")
def fleet(search: str | None = None, camera_status: str | None = Query(None, alias="status"), district: str | None = None, zone: str | None = None, source: str | None = None, group_id: int | None = None, page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200), db: Session = Depends(getDB), user: User = Depends(get_current_user)):
    fleet_base = db.query(Camera).filter(
        Camera.user_id == user.id,
        Camera.source_type != "upload",
    )
    query = fleet_base
    if search:
        term = f"%{search.strip()}%"; query = query.filter(or_(Camera.cam_id.ilike(term), Camera.camera_name.ilike(term), Camera.location_name.ilike(term)))
    if camera_status: query = query.filter(Camera.connection_state == camera_status.upper())
    if district: query = query.filter(Camera.district == district)
    if zone: query = query.filter(Camera.zone == zone)
    if source: query = query.filter(or_(Camera.external_provider == source, Camera.source_type == source))
    if group_id: query = query.filter(Camera.group_id == group_id)
    total = query.count(); rows = query.order_by(Camera.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    counts = dict(fleet_base.with_entities(Camera.connection_state, func.count(Camera.id)).group_by(Camera.connection_state).all())
    return {"total": total, "page": page, "page_size": page_size, "summary": {"total": fleet_base.count(), **{str(k or 'OFFLINE').lower(): v for k, v in counts.items()}}, "cameras": [{"id": row.id, "camera_id": row.cam_id, "camera_name": row.camera_name, "location": row.location_name, "district": row.district, "zone": row.zone, "source": row.external_provider or row.source_type, "resolution": f"{row.stream_width}x{row.stream_height}" if row.stream_width and row.stream_height else None, "codec": row.codec, "health": row.connection_state or "OFFLINE", "last_seen": row.last_seen_at.isoformat() if row.last_seen_at else None, "processing_enabled": row.processing_enabled, "analytics_enabled": row.analytics_enabled, "ai_profile_id": row.ai_profile_id, "group_id": row.group_id} for row in rows]}


@router.get("/credential-profiles")
def credentials(db: Session = Depends(getDB), user: User = Depends(admin)):
    rows = db.query(CameraCredential).filter(CameraCredential.user_id == user.id).order_by(CameraCredential.name).all()
    return {"profiles": [{"id": row.id, "name": row.name, "username": row.username, "auth_type": row.auth_type, "secret_configured": True} for row in rows]}


@router.post("/credential-profiles", status_code=201)
def create_credential(payload: CredentialCreate, db: Session = Depends(getDB), user: User = Depends(admin)):
    row = CameraCredential(user_id=user.id, name=payload.name, username=payload.username, auth_type=payload.auth_type, secret_encrypted=encrypt_connector_config({"password": payload.password}))
    db.add(row)
    try: db.commit(); db.refresh(row)
    except IntegrityError as exc: db.rollback(); raise HTTPException(status_code=409, detail="Credential profile name already exists") from exc
    return {"id": row.id, "name": row.name, "username": row.username, "auth_type": row.auth_type, "secret_configured": True}


@router.get("/camera-groups")
def groups(db: Session = Depends(getDB), user: User = Depends(get_current_user)):
    rows = db.query(CameraGroup).filter(CameraGroup.user_id == user.id).order_by(CameraGroup.parent_id, CameraGroup.name).all()
    return {"groups": [{"id": row.id, "name": row.name, "parent_id": row.parent_id, "group_type": row.group_type, "metadata": row.metadata_json} for row in rows]}


@router.post("/camera-groups", status_code=201)
def create_group(payload: GroupCreate, db: Session = Depends(getDB), user: User = Depends(admin)):
    if payload.parent_id and not db.query(CameraGroup.id).filter(CameraGroup.id == payload.parent_id, CameraGroup.user_id == user.id).first(): raise HTTPException(status_code=400, detail="Parent group not found")
    row = CameraGroup(user_id=user.id, name=payload.name, parent_id=payload.parent_id, group_type=payload.group_type, metadata_json=payload.metadata); db.add(row); db.commit(); db.refresh(row)
    return {"id": row.id, "name": row.name, "parent_id": row.parent_id, "group_type": row.group_type, "metadata": row.metadata_json}


@router.get("/camera-ai-profiles")
def ai_profiles(db: Session = Depends(getDB), user: User = Depends(get_current_user)):
    rows = db.query(CameraAIProfile).filter(CameraAIProfile.user_id == user.id).order_by(CameraAIProfile.name).all()
    return {"profiles": [{"id": row.id, "name": row.name, "configuration": row.configuration, "processing_fps": row.processing_fps, "preferred_stream": row.preferred_stream, "evidence_enabled": row.evidence_enabled} for row in rows]}


@router.post("/camera-ai-profiles", status_code=201)
def create_ai_profile(payload: AIProfileCreate, db: Session = Depends(getDB), user: User = Depends(admin)):
    row = CameraAIProfile(user_id=user.id, **payload.model_dump()); db.add(row); db.commit(); db.refresh(row)
    return {"id": row.id, **payload.model_dump()}


@router.post("/cameras/bulk-actions", status_code=202)
def bulk_action(payload: BulkAction, db: Session = Depends(getDB), user: User = Depends(admin)):
    if payload.action == "delete" and not payload.confirmed: raise HTTPException(status_code=400, detail="Bulk deletion requires explicit confirmation")
    cameras = db.query(Camera).filter(Camera.user_id == user.id, Camera.cam_id.in_(set(payload.camera_ids))).all()
    if not cameras: raise HTTPException(status_code=404, detail="No matching cameras found")
    for camera in cameras:
        if payload.action == "enable": camera.is_active = True
        elif payload.action == "disable": camera.is_active = False; camera.processing_enabled = False; camera.desired_state = "STOPPED"; camera.connection_state = "DISABLED"
        elif payload.action == "change_credential": camera.credential_profile_id = int(payload.value) if payload.value else None
        elif payload.action == "change_group": camera.group_id = int(payload.value) if payload.value else None
        elif payload.action == "assign_ai_profile": camera.ai_profile_id = int(payload.value) if payload.value else None
        elif payload.action == "enable_analytics": camera.analytics_enabled = True
        elif payload.action == "disable_analytics": camera.analytics_enabled = False
        elif payload.action == "health_check": camera.connection_state = "CONNECTING"; camera.last_health_check = None
        elif payload.action == "refresh_metadata": camera.external_last_synced_at = None
        elif payload.action == "reconnect": camera.connection_state = "CONNECTING"; camera.retry_count = 0
        elif payload.action == "delete": db.delete(camera)
    db.add(AuditLog(user_id=user.id, action=f"CAMERA_BULK_{payload.action.upper()}", resource_type="camera", details={"camera_ids": [c.cam_id for c in cameras], "count": len(cameras)}))
    db.commit()
    return {"status": "completed", "action": payload.action, "affected": len(cameras)}
