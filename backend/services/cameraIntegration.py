"""Transactional Sentinel/VMS camera discovery and metadata synchronization."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import re
import threading
from typing import Any
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from connectors.catalogue import CatalogueCamera, CatalogueResult
from connectors.catalogue_manager import create_catalogue_connector, normalize_catalogue_provider
from db.crud import create_camera
from db.database import SessionLocal
from db.model import Camera, CameraIntegration, CameraIntegrationSyncRun
from security.connectorConfig import decrypt_connector_config, encrypt_connector_config


logger = logging.getLogger(__name__)
_sync_locks: dict[int, threading.Lock] = {}
_sync_locks_guard = threading.RLock()


def utc_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def encrypt_integration_config(config: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(config, dict):
        raise ValueError("Integration configuration must be an object")
    provider_config = dict(config)
    provider_config.setdefault("verify_tls", True)
    encrypted = encrypt_connector_config(provider_config)
    if not encrypted:
        raise ValueError("Integration configuration cannot be empty")
    # Fingerprint the randomized ciphertext, not the plaintext configuration.
    # A deterministic hash of a low-entropy password or token would otherwise
    # give a database-only attacker an unnecessary offline guessing oracle.
    return encrypted, hashlib.sha256(encrypted.encode("utf-8")).hexdigest()


def integration_view(row: CameraIntegration) -> dict[str, Any]:
    origin = None
    try:
        config = decrypt_connector_config(row.config_encrypted)
        base_url = str(config.get("base_url") or "")
        parsed = urlparse(base_url)
        if parsed.scheme and parsed.netloc:
            origin = f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        origin = None
    return {
        "id": row.id,
        "name": row.name,
        "provider_type": row.provider_type,
        "enabled": bool(row.enabled),
        "auto_sync": bool(row.auto_sync),
        "sync_interval_seconds": int(row.sync_interval_seconds or 300),
        "endpoint_origin": origin,
        "credentials_configured": bool(row.config_encrypted),
        "last_sync_started_at": row.last_sync_started_at.isoformat() if row.last_sync_started_at else None,
        "last_sync_at": row.last_sync_at.isoformat() if row.last_sync_at else None,
        "last_sync_status": row.last_sync_status,
        "last_error_code": row.last_error_code,
        "last_error_message": row.last_error_message,
        "last_camera_count": int(row.last_camera_count or 0),
        "last_created_count": int(row.last_created_count or 0),
        "last_updated_count": int(row.last_updated_count or 0),
        "last_offline_count": int(row.last_offline_count or 0),
        "source_revision": row.source_revision,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _lock_for(integration_id: int) -> threading.Lock:
    with _sync_locks_guard:
        return _sync_locks.setdefault(int(integration_id), threading.Lock())


def _stable_camera_id(provider: str, external_id: str) -> str:
    prefix = "SENTINEL" if provider == "sentinel" else "VMS"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(external_id)).strip("_.-") or "CAMERA"
    digest = hashlib.sha256(f"{provider}:{external_id}".encode("utf-8")).hexdigest()[:10].upper()
    available = 100 - len(prefix) - len(digest) - 2
    return f"{prefix}_{slug[:max(1, available)]}_{digest}"


def _direction(value: str | None) -> str | None:
    if not value:
        return None
    normalized = str(value).strip().upper().replace("-", "").replace("_", "")
    aliases = {
        "N": "NORTH", "NE": "NORTHEAST", "E": "EAST", "SE": "SOUTHEAST",
        "S": "SOUTH", "SW": "SOUTHWEST", "W": "WEST", "NW": "NORTHWEST",
    }
    value = aliases.get(normalized, normalized)
    allowed = {"NORTH", "NORTHEAST", "EAST", "SOUTHEAST", "SOUTH", "SOUTHWEST", "WEST", "NORTHWEST", "UNKNOWN"}
    return value if value in allowed else "UNKNOWN"


def _safe_provider_metadata(value: Any, depth: int = 0) -> Any:
    """Keep bounded descriptive metadata while excluding secret/URL material."""
    if depth > 4:
        return None
    if isinstance(value, dict):
        result = {}
        blocked = (
            "url", "uri", "secret", "token", "password", "credential",
            "api_key", "apikey", "authorization", "cookie", "passphrase",
        )
        for raw_key, item in list(value.items())[:100]:
            key = str(raw_key)[:80]
            if any(part in key.lower() for part in blocked):
                continue
            result[key] = _safe_provider_metadata(item, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_provider_metadata(item, depth + 1) for item in list(value)[:100]]
    if isinstance(value, str):
        cleaned = value.strip()
        if re.search(r"(?:rtsp|rtsps|https?|wss?)://", cleaned, re.IGNORECASE):
            return "[REDACTED_URL]"
        return cleaned[:1000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:1000]


def _metadata_document(camera: CatalogueCamera) -> tuple[dict[str, Any], str]:
    streams = []
    for stream in camera.streams[:16]:
        item = stream.safe_dict()
        item["url_sha256"] = hashlib.sha256(stream.url.encode("utf-8")).hexdigest()
        streams.append(item)
    document = {
        "external_id": camera.external_id,
        "name": camera.name,
        "live_status": camera.live_status,
        "location_name": camera.location_name,
        "latitude": camera.latitude,
        "longitude": camera.longitude,
        "altitude": camera.altitude,
        "heading": camera.heading,
        "fov": camera.fov,
        "direction": camera.direction,
        "road_name": camera.road_name,
        "city": camera.city,
        "state": camera.state,
        "country": camera.country,
        "vendor": camera.vendor,
        "streams": streams,
        "provider_metadata": _safe_provider_metadata(camera.metadata),
    }
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return document, hashlib.sha256(encoded).hexdigest()


def discover_integration(row: CameraIntegration, *, transport=None) -> CatalogueResult:
    config = decrypt_connector_config(row.config_encrypted)
    connector = create_catalogue_connector(row.provider_type, config, transport=transport)
    return connector.fetch_catalogue()


def _upsert_discovered_camera(
    db: Session,
    *,
    integration: CameraIntegration,
    descriptor: CatalogueCamera,
    now: datetime,
) -> str:
    primary = descriptor.preferred_stream()
    if primary is None:
        return "skipped"
    provider = normalize_catalogue_provider(integration.provider_type)
    existing = (
        db.query(Camera)
        .filter(
            Camera.user_id == integration.user_id,
            Camera.external_provider == provider,
            Camera.external_camera_id == descriptor.external_id,
        )
        .first()
    )
    cam_id = existing.cam_id if existing else _stable_camera_id(provider, descriptor.external_id)
    if existing is None:
        collision = (
            db.query(Camera.id)
            .filter(Camera.user_id == integration.user_id, Camera.cam_id == cam_id)
            .first()
        )
        if collision:
            raise RuntimeError("Generated external camera ID collided with an existing camera")

    document, metadata_hash = _metadata_document(descriptor)
    changed = existing is None or existing.external_metadata_hash != metadata_hash
    row = create_camera(
        db=db,
        user_id=integration.user_id,
        cam_id=cam_id,
        camera_name=descriptor.name,
        zone=None,
        source=primary.url,
        source_type=primary.source_type,
        latitude=descriptor.latitude,
        longitude=descriptor.longitude,
        location_name=descriptor.location_name,
        connector_config=None,
        vendor=descriptor.vendor or ("Sentinel" if provider == "sentinel" else "VMS/NVR"),
        vendor_device_id=descriptor.external_id,
        stream_profile=primary.profile,
        direction=_direction(descriptor.direction),
        stream_fps=primary.fps,
        stream_width=primary.width,
        stream_height=primary.height,
        transport=primary.transport,
        codec=primary.codec,
        altitude=descriptor.altitude,
        heading=descriptor.heading,
        fov=descriptor.fov,
        road_name=descriptor.road_name,
        city=descriptor.city,
        state=descriptor.state,
        country=descriptor.country,
        commit=False,
    )
    alternate_streams = []
    for stream in descriptor.streams[:8]:
        candidate = {"url": stream.url, **stream.safe_dict()}
        prospective = [*alternate_streams, candidate]
        size = len(json.dumps(prospective, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        if size > 12000:
            break
        alternate_streams.append(candidate)
    encrypted_alternates = encrypt_connector_config({
        "provider_type": provider,
        "external_camera_id": descriptor.external_id,
        "alternate_streams": alternate_streams,
    })
    row.integration_id = integration.id
    row.external_provider = provider
    row.external_camera_id = descriptor.external_id
    row.external_live_status = descriptor.live_status
    row.external_metadata = document
    row.external_metadata_hash = metadata_hash
    row.external_last_seen_at = now
    row.external_last_synced_at = now
    row.externally_managed = True
    row.connector_config_encrypted = encrypted_alternates
    row.connector_type = provider
    row.resolved_source_type = primary.source_type
    return "created" if existing is None else "updated" if changed else "unchanged"


def sync_integration(
    db: Session,
    integration: CameraIntegration,
    *,
    trigger: str = "manual",
    transport=None,
) -> dict[str, Any]:
    lock = _lock_for(integration.id)
    if not lock.acquire(blocking=False):
        raise RuntimeError("Integration synchronization is already running")
    run = CameraIntegrationSyncRun(
        integration_id=integration.id,
        trigger=str(trigger or "manual")[:30],
        status="RUNNING",
        started_at=utc_naive(),
    )
    db.add(run)
    integration.last_sync_started_at = run.started_at
    integration.last_sync_status = "RUNNING"
    db.commit()
    try:
        result = discover_integration(integration, transport=transport)
        now = utc_naive()
        counts = {"created": 0, "updated": 0, "unchanged": 0, "skipped": 0}
        seen: set[str] = set()
        for descriptor in result.cameras:
            seen.add(descriptor.external_id)
            outcome = _upsert_discovered_camera(
                db,
                integration=integration,
                descriptor=descriptor,
                now=now,
            )
            counts[outcome] += 1

        missing_query = db.query(Camera).filter(
            Camera.integration_id == integration.id,
            Camera.externally_managed.is_(True),
        )
        offline_count = 0
        deactivate_missing = False
        try:
            deactivate_missing = bool(decrypt_connector_config(integration.config_encrypted).get("deactivate_missing", False))
        except Exception:
            pass
        for camera in missing_query.all():
            if camera.external_camera_id not in seen:
                camera.external_live_status = "MISSING"
                camera.external_last_synced_at = now
                if deactivate_missing:
                    camera.is_active = False
                offline_count += 1
            elif camera.external_live_status in {"OFFLINE", "DEGRADED", "MISSING"}:
                offline_count += 1

        integration.last_sync_at = now
        integration.last_sync_status = "SUCCEEDED"
        integration.last_error_code = None
        integration.last_error_message = None
        integration.last_camera_count = len(result.cameras)
        integration.last_created_count = counts["created"]
        integration.last_updated_count = counts["updated"]
        integration.last_offline_count = offline_count
        integration.source_revision = result.source_revision
        integration.etag = result.etag

        run.status = "SUCCEEDED"
        run.discovered_count = len(result.cameras)
        run.created_count = counts["created"]
        run.updated_count = counts["updated"]
        run.skipped_count = counts["skipped"]
        run.offline_count = offline_count
        run.source_revision = result.source_revision
        run.completed_at = now
        db.commit()
        return {
            "integration_id": integration.id,
            "provider_type": integration.provider_type,
            "status": "SUCCEEDED",
            "discovered_count": len(result.cameras),
            "created_count": counts["created"],
            "updated_count": counts["updated"],
            "unchanged_count": counts["unchanged"],
            "skipped_count": counts["skipped"],
            "offline_count": offline_count,
            "source_revision": result.source_revision,
            "completed_at": now.isoformat(),
        }
    except Exception as exc:
        db.rollback()
        integration = db.query(CameraIntegration).filter(CameraIntegration.id == integration.id).first()
        run = db.query(CameraIntegrationSyncRun).filter(CameraIntegrationSyncRun.id == run.id).first()
        now = utc_naive()
        error_code = type(exc).__name__[:80]
        if integration:
            integration.last_sync_at = now
            integration.last_sync_status = "FAILED"
            integration.last_error_code = error_code
            integration.last_error_message = "Camera catalogue synchronization failed"
        if run:
            run.status = "FAILED"
            run.error_code = error_code
            run.error_message = "Camera catalogue synchronization failed"
            run.completed_at = now
        db.commit()
        logger.warning(
            "Camera catalogue synchronization failed | integration_id=%s provider=%s error=%s",
            getattr(integration, "id", None),
            getattr(integration, "provider_type", "unknown"),
            error_code,
        )
        raise
    finally:
        lock.release()


def sync_due_integrations() -> list[dict[str, Any]]:
    """Synchronize due integrations from the background scheduler."""
    now = utc_naive()
    with SessionLocal() as db:
        rows = db.query(CameraIntegration).filter(
            CameraIntegration.enabled.is_(True),
            CameraIntegration.auto_sync.is_(True),
        ).all()
        due_ids = []
        for row in rows:
            elapsed = (now - row.last_sync_at).total_seconds() if row.last_sync_at else None
            if elapsed is None or elapsed >= max(60, int(row.sync_interval_seconds or 300)):
                due_ids.append(row.id)

    results = []
    for integration_id in due_ids:
        with SessionLocal() as db:
            row = db.query(CameraIntegration).filter(CameraIntegration.id == integration_id).first()
            if not row or not row.enabled or not row.auto_sync:
                continue
            try:
                results.append(sync_integration(db, row, trigger="scheduled"))
            except Exception:
                results.append({"integration_id": integration_id, "status": "FAILED"})
    return results
