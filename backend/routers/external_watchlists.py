from __future__ import annotations
import logging
import re
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy import func
from db.database import getDB
from db.external_watchlist_model import ExternalWatchlistSource, ExternalWatchlistRecord, ExternalWatchlistSyncHistory
from db.model import User, AuditLog
from auth.auth import get_current_user
from security.rbac import require_super_admin
from integrations.government_watchlist.providers.rest import validate_remote_url
from integrations.government_watchlist.security import encrypt_credential
from integrations.government_watchlist.service import safe_source, test_source, sync_source, deactivate_source_records

logger = logging.getLogger("external-watchlist-api")
router = APIRouter(prefix="/api/watchlist-integrations", tags=["External Watchlists"])
_ALLOWED_CANONICAL = {"external_id", "entity_type", "category", "active", "name", "plate", "priority", "case_reference", "image_data", "image_content_type", "make", "model", "color", "description"}

class SourceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    description: str | None = Field(default=None, max_length=2000)
    provider_type: Literal["REST_JSON"] = "REST_JSON"
    base_url: str = Field(min_length=10, max_length=2048)
    auth_type: Literal["NONE", "BEARER", "API_KEY_HEADER"] = "NONE"
    credential: str | None = Field(default=None, max_length=4096)
    api_key_header: str | None = Field(default=None, max_length=100)
    entity_types: list[Literal["PERSON", "VEHICLE"]] = Field(default_factory=lambda: ["PERSON", "VEHICLE"], min_length=1)
    field_mapping: dict[str, str] = Field(default_factory=dict)
    records_path: str = Field(default="records", max_length=250)
    sync_mode: Literal["MANUAL", "INTERVAL"] = "INTERVAL"
    sync_interval_seconds: int = Field(default=300, ge=30, le=86400)
    pagination: dict[str, Any] = Field(default_factory=dict)
    tls_verify: bool = True

    @field_validator("field_mapping")
    @classmethod
    def validate_mapping(cls, value):
        if len(value) > 30 or any(k not in _ALLOWED_CANONICAL or not isinstance(v, str) or not v or len(v) > 250 for k, v in value.items()):
            raise ValueError("Field mapping contains unsupported keys or invalid paths")
        return value

class SourcePatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=150)
    description: str | None = Field(default=None, max_length=2000)
    base_url: str | None = Field(default=None, min_length=10, max_length=2048)
    enabled: bool | None = None
    auth_type: Literal["NONE", "BEARER", "API_KEY_HEADER"] | None = None
    credential: str | None = Field(default=None, max_length=4096)
    api_key_header: str | None = Field(default=None, max_length=100)
    entity_types: list[Literal["PERSON", "VEHICLE"]] | None = Field(default=None, min_length=1)
    field_mapping: dict[str, str] | None = None
    records_path: str | None = Field(default=None, max_length=250)
    sync_mode: Literal["MANUAL", "INTERVAL"] | None = None
    sync_interval_seconds: int | None = Field(default=None, ge=30, le=86400)
    pagination: dict[str, Any] | None = None
    tls_verify: bool | None = None

    @field_validator("field_mapping")
    @classmethod
    def validate_patch_mapping(cls, value):
        if value is not None and (len(value) > 30 or any(k not in _ALLOWED_CANONICAL or not isinstance(v, str) or not v or len(v) > 250 for k, v in value.items())):
            raise ValueError("Field mapping contains unsupported keys or invalid paths")
        return value


def _valid_api_key_header(value: str | None) -> bool:
    if not value or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", value):
        return False
    return value.lower() not in {"host", "authorization", "cookie", "set-cookie", "content-length", "transfer-encoding", "connection"}


def _audit(db, actor, request, action, source_id, details=None):
    db.add(AuditLog(user_id=actor.id, action=action, resource_type="external_watchlist_source", resource_id=str(source_id) if source_id else None,
                    details=details or {}, source_ip=request.client.host if request.client else None))
    db.commit()


def _owned(db, source_id, actor):
    obj = db.query(ExternalWatchlistSource).filter(ExternalWatchlistSource.id == source_id, ExternalWatchlistSource.owner_user_id == actor.id).first()
    if obj is None:
        raise HTTPException(status_code=404, detail="External watchlist source not found")
    return obj


def _validated_url(url):
    try:
        return validate_remote_url(url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@router.get("")
def list_sources(actor: User = Depends(require_super_admin()), db: Session = Depends(getDB)):
    rows = db.query(ExternalWatchlistSource).filter(ExternalWatchlistSource.owner_user_id == actor.id).order_by(ExternalWatchlistSource.created_at.desc()).all()
    output = []
    for source in rows:
        counts = db.query(ExternalWatchlistRecord.entity_type, func.count(ExternalWatchlistRecord.id)).filter(ExternalWatchlistRecord.source_id == source.id, ExternalWatchlistRecord.active.is_(True)).group_by(ExternalWatchlistRecord.entity_type).all()
        item = safe_source(source); item["person_records"] = next((int(n) for t,n in counts if t == "PERSON"), 0); item["vehicle_records"] = next((int(n) for t,n in counts if t == "VEHICLE"), 0)
        output.append(item)
    return {"sources": output}

@router.post("", status_code=201)
def create_source(payload: SourceCreate, request: Request, actor: User = Depends(require_super_admin()), db: Session = Depends(getDB)):
    if payload.auth_type != "NONE" and not payload.credential:
        raise HTTPException(status_code=422, detail="A credential is required for the selected authentication type")
    if payload.auth_type == "API_KEY_HEADER" and not _valid_api_key_header(payload.api_key_header):
        raise HTTPException(status_code=422, detail="A valid API key header name is required")
    if payload.auth_type == "NONE" and payload.credential:
        raise HTTPException(status_code=422, detail="Credential must be empty when authentication is NONE")
    source = ExternalWatchlistSource(owner_user_id=actor.id, name=payload.name.strip(), description=payload.description,
        base_url=_validated_url(payload.base_url), auth_type=payload.auth_type, credential_ciphertext=encrypt_credential(payload.credential),
        api_key_header=payload.api_key_header, entity_types=list(dict.fromkeys(payload.entity_types)), field_mapping=payload.field_mapping,
        records_path=payload.records_path, sync_mode=payload.sync_mode, sync_interval_seconds=payload.sync_interval_seconds, pagination=payload.pagination,
        tls_verify=payload.tls_verify, created_by=actor.id, enabled=True, read_only=True)
    db.add(source); db.commit(); db.refresh(source); _audit(db, actor, request, "EXTERNAL_WATCHLIST_SOURCE_CREATED", source.id, {"name": source.name, "auth_type": source.auth_type})
    return {"source": safe_source(source)}

@router.get("/{source_id}")
def get_source(source_id: int, actor: User = Depends(require_super_admin()), db: Session = Depends(getDB)):
    return {"source": safe_source(_owned(db, source_id, actor))}

@router.patch("/{source_id}")
def patch_source(source_id: int, payload: SourcePatch, request: Request, actor: User = Depends(require_super_admin()), db: Session = Depends(getDB)):
    source = _owned(db, source_id, actor); data = payload.model_dump(exclude_unset=True)
    credential = data.pop("credential", None)
    for key, value in data.items():
        if key == "base_url" and value is not None: value = _validated_url(value)
        if key == "name" and value is not None: value = value.strip()
        if value is not None: setattr(source, key, value)
    if credential:
        source.credential_ciphertext = encrypt_credential(credential)
    elif source.auth_type == "NONE":
        source.credential_ciphertext = None
    if source.auth_type == "API_KEY_HEADER" and not _valid_api_key_header(source.api_key_header):
        raise HTTPException(status_code=422, detail="A valid API key header name is required")
    if source.auth_type != "NONE" and not source.credential_ciphertext:
        raise HTTPException(status_code=422, detail="Set a credential for the selected authentication type")
    if source.auth_type == "NONE": source.credential_ciphertext = None
    source.updated_at = datetime.utcnow()
    if data.get("enabled") is False:
        deactivate_source_records(db, source)
    db.commit(); db.refresh(source)
    action = "EXTERNAL_WATCHLIST_SOURCE_DISABLED" if data.get("enabled") is False else "EXTERNAL_WATCHLIST_SOURCE_ENABLED" if data.get("enabled") is True else "EXTERNAL_WATCHLIST_SOURCE_UPDATED"
    _audit(db, actor, request, action, source.id, {"fields": sorted(k for k in data if k != "credential"), "credential_updated": bool(credential)})
    return {"source": safe_source(source)}

@router.post("/{source_id}/test")
def test_connection(source_id: int, request: Request, actor: User = Depends(require_super_admin()), db: Session = Depends(getDB)):
    source = _owned(db, source_id, actor)
    try:
        result = test_source(source)
    except Exception as exc:
        logger.info("External watchlist test failed | source_id=%s error_type=%s", source.id, type(exc).__name__)
        result = {"connected": False, "error": "Connection failed; verify endpoint, credentials, TLS, and response format"}
    _audit(db, actor, request, "EXTERNAL_WATCHLIST_TESTED", source.id, {"connected": bool(result.get("connected")), "records_available": result.get("records_available")})
    return result

@router.post("/{source_id}/sync")
def sync(source_id: int, request: Request, actor: User = Depends(require_super_admin()), db: Session = Depends(getDB)):
    source = _owned(db, source_id, actor)
    if not source.enabled: raise HTTPException(status_code=409, detail="Source is disabled")
    result = sync_source(db, source, actor.id)
    _audit(db, actor, request, "EXTERNAL_WATCHLIST_SYNCED", source.id, {k: result.get(k) for k in ("received", "created", "updated", "unchanged", "rejected", "status")})
    return result

@router.get("/{source_id}/status")
def status(source_id: int, actor: User = Depends(require_super_admin()), db: Session = Depends(getDB)):
    source = _owned(db, source_id, actor)
    return {"source": safe_source(source), "record_counts": {kind.lower(): db.query(func.count(ExternalWatchlistRecord.id)).filter(ExternalWatchlistRecord.source_id == source.id, ExternalWatchlistRecord.entity_type == kind, ExternalWatchlistRecord.active.is_(True)).scalar() for kind in ("PERSON", "VEHICLE")}}

@router.get("/{source_id}/sync-history")
def sync_history(source_id: int, limit: int = 50, actor: User = Depends(require_super_admin()), db: Session = Depends(getDB)):
    source = _owned(db, source_id, actor); limit = max(1, min(limit, 200))
    rows = db.query(ExternalWatchlistSyncHistory).filter(ExternalWatchlistSyncHistory.source_id == source.id).order_by(ExternalWatchlistSyncHistory.created_at.desc()).limit(limit).all()
    return {"history": [{"id": int(x.id), "action": x.action, "result": x.result, "records_received": x.records_received, "records_created": x.records_created, "records_updated": x.records_updated, "records_unchanged": x.records_unchanged, "records_rejected": x.records_rejected, "duration_ms": x.duration_ms, "message": x.message, "created_at": x.created_at.isoformat()} for x in rows]}

@router.get("/{source_id}/field-mapping")
def get_mapping(source_id: int, actor: User = Depends(require_super_admin()), db: Session = Depends(getDB)):
    source = _owned(db, source_id, actor); return {"field_mapping": source.field_mapping or {}}

@router.put("/{source_id}/field-mapping")
def put_mapping(source_id: int, mapping: dict[str, str], request: Request, actor: User = Depends(require_super_admin()), db: Session = Depends(getDB)):
    SourceCreate.model_validate({"name": "validation", "base_url": "https://example.com", "field_mapping": mapping})
    source = _owned(db, source_id, actor); source.field_mapping = mapping; source.updated_at = datetime.utcnow(); db.commit()
    _audit(db, actor, request, "EXTERNAL_WATCHLIST_MAPPING_UPDATED", source.id, {"field_count": len(mapping)})
    return {"field_mapping": mapping}

@router.delete("/{source_id}")
def delete_source(source_id: int, request: Request, actor: User = Depends(require_super_admin()), db: Session = Depends(getDB)):
    source = _owned(db, source_id, actor)
    # Remove only rows explicitly linked to this connector, preserving native watchlist entries.
    for row in db.query(ExternalWatchlistRecord).filter(ExternalWatchlistRecord.source_id == source.id).all():
        if row.entity_type == "PERSON":
            db.query(__import__("db.advanced_intelligence_model", fromlist=["PersonWatchlistEntry"]).PersonWatchlistEntry).filter_by(user_id=actor.id, external_reference=f"extwl:{source.id}:{row.external_id}").delete(synchronize_session=False)
        elif row.local_watchlist_entry_id:
            db.query(__import__("db.watchlist_model", fromlist=["WatchlistEntry"]).WatchlistEntry).filter_by(id=row.local_watchlist_entry_id, user_id=actor.id).delete(synchronize_session=False)
    _audit(db, actor, request, "EXTERNAL_WATCHLIST_SOURCE_DELETED", source.id, {"name": source.name})
    db.delete(source); db.commit(); return {"success": True}
