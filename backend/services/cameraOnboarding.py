"""Job-based camera discovery, validation, registration, and fleet operations."""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import socket
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from urllib.parse import quote, urlsplit, urlunsplit

from sqlalchemy.orm import Session

from db.crud import create_camera
from db.database import SessionLocal
from db.model import Camera, CameraCredential, CameraOnboardingItem, CameraOnboardingJob, AuditLog
from security.cameraSource import decrypt_camera_source, encrypt_camera_source
from security.connectorConfig import decrypt_connector_config, encrypt_connector_config
from services.timestampedCapture import (
    CameraSourceError,
    _build_sentinel_rtsp_url,
    _is_sentinel_rtsp_source,
)

logger = logging.getLogger(__name__)
ProgressSender = Callable[[int, dict[str, Any]], Awaitable[None]]

HEALTH_STATES = {
    "ONLINE", "OFFLINE", "CONNECTING", "DISCOVERED", "VALIDATING",
    "AUTHENTICATION_FAILED", "NETWORK_UNREACHABLE", "STREAM_UNAVAILABLE",
    "NO_VIDEO_TRACK", "PTS_INVALID", "DEGRADED", "AI_WORKER_UNAVAILABLE", "DISABLED",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def sanitize_url(value: str | None) -> str:
    if not value:
        return ""
    try:
        parts = urlsplit(str(value))
        if not parts.hostname:
            return "[REDACTED_SOURCE]"
        host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
        port = f":{parts.port}" if parts.port else ""
        auth = "***:***@" if parts.username is not None else ""
        return urlunsplit((parts.scheme, f"{auth}{host}{port}", parts.path, parts.query, ""))
    except Exception:
        return "[REDACTED_SOURCE]"


def normalize_rtsp(value: str) -> str:
    parts = urlsplit(str(value).strip())
    if parts.scheme.lower() not in {"rtsp", "rtsps"} or not parts.hostname:
        raise ValueError("A valid RTSP URL is required")
    if parts.port and not 1 <= parts.port <= 65535:
        raise ValueError("RTSP port is invalid")
    host = parts.hostname.lower()
    port = parts.port or 554
    path = re.sub(r"/+", "/", parts.path or "/").rstrip("/") or "/"
    return f"rtsp://{host}:{port}{path}"


def apply_credentials(source: str, profile: CameraCredential | None) -> str:
    if profile is None:
        return source
    secret = decrypt_connector_config(profile.secret_encrypted)
    password = str(secret.get("password") or "")
    username = str(profile.username or secret.get("username") or "")
    if not username:
        return source
    parts = urlsplit(source)
    host = f"[{parts.hostname}]" if parts.hostname and ":" in parts.hostname else parts.hostname
    port = f":{parts.port}" if parts.port else ""
    netloc = f"{quote(username, safe='')}:{quote(password, safe='')}@{host}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))


def validate_authorized_network(network: str) -> ipaddress._BaseNetwork:
    try:
        parsed = ipaddress.ip_network(network, strict=False)
    except ValueError as exc:
        raise ValueError("Invalid network range") from exc
    if not parsed.is_private or parsed.is_loopback or parsed.is_link_local or parsed.is_multicast:
        raise ValueError("Discovery is restricted to authorized private network ranges")
    max_hosts = int(os.getenv("CAMERA_DISCOVERY_MAX_HOSTS", "1024"))
    if parsed.num_addresses > max(1, min(max_hosts, 4096)) + 2:
        raise ValueError("Network range exceeds CAMERA_DISCOVERY_MAX_HOSTS")
    return parsed


async def discover_private_network(network: str, ports: list[int] | None = None) -> list[dict[str, Any]]:
    parsed = validate_authorized_network(network)
    ports = ports or [554, 8554, 80, 8080]
    timeout = max(0.2, min(float(os.getenv("CAMERA_DISCOVERY_TIMEOUT_SECONDS", "3")), 15.0))
    limit = asyncio.Semaphore(max(1, min(int(os.getenv("CAMERA_ONBOARDING_MAX_CONCURRENCY", "10")), 50)))

    async def probe(host, port):
        async with limit:
            try:
                _reader, writer = await asyncio.wait_for(asyncio.open_connection(str(host), int(port)), timeout=timeout)
                writer.close()
                await writer.wait_closed()
                scheme = "rtsp" if int(port) in {554, 8554} else "http"
                return {"camera_id": f"DISCOVERED_{str(host).replace('.', '_')}_{port}", "camera_name": f"Camera {host}", "ip_address": str(host), "source": f"{scheme}://{host}:{port}/", "source_type": scheme, "status": "DISCOVERED"}
            except (OSError, asyncio.TimeoutError):
                return None

    rows = await asyncio.gather(*(probe(host, port) for host in parsed.hosts() for port in ports))
    return [row for row in rows if row]


def _validate_stream(source: str) -> dict[str, Any]:
    """Open a bounded decoder, confirm video and frames, then always close it."""
    import av

    timeout = max(1.0, min(float(os.getenv("CAMERA_DISCOVERY_TIMEOUT_SECONDS", "10")), 30.0))
    frame_target = max(1, min(int(os.getenv("CAMERA_VALIDATION_FRAME_COUNT", "5")), 30))
    container = None
    started = time.monotonic()

    # Sentinel URLs stored by Intel-I deliberately contain no credentials.
    # Resolve them from backend-only environment variables immediately before
    # opening the decoder. Never persist, log, or return validation_source.
    validation_source = str(source or "").strip()
    if not validation_source:
        return {
            "status": "STREAM_UNAVAILABLE",
            "error_code": "CAMERA_SOURCE_MISSING",
            "error_message": "Camera stream source is missing",
        }

    try:
        if _is_sentinel_rtsp_source(validation_source):
            validation_source = _build_sentinel_rtsp_url(validation_source)
    except CameraSourceError as exc:
        status = (
            "AUTHENTICATION_FAILED"
            if exc.code == "SENTINEL_CREDENTIALS_MISSING"
            else "STREAM_UNAVAILABLE"
        )
        return {
            "status": status,
            "error_code": exc.code,
            "error_message": str(exc),
        }

    try:
        container = av.open(validation_source, mode="r", options={"rtsp_transport": "tcp", "stimeout": str(int(timeout * 1_000_000)), "rw_timeout": str(int(timeout * 1_000_000))})
        stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if stream is None:
            return {"status": "NO_VIDEO_TRACK", "error_code": "NO_VIDEO_TRACK", "error_message": "No video track was found"}
        frames = 0
        pts_seen = False
        for frame in container.decode(stream):
            frames += 1
            pts_seen = pts_seen or (frame.pts is not None and frame.time_base is not None)
            if frames >= frame_target or time.monotonic() - started >= timeout:
                break
        if frames == 0:
            return {"status": "STREAM_UNAVAILABLE", "error_code": "FRAME_DECODE_FAILED", "error_message": "Connected, but no video frame could be decoded"}
        rate = float(stream.average_rate) if stream.average_rate else None
        return {"status": "ONLINE", "width": stream.codec_context.width, "height": stream.codec_context.height, "resolution": f"{stream.codec_context.width}x{stream.codec_context.height}", "fps": rate, "codec": stream.codec_context.name, "timestamp_status": "VALID" if pts_seen else "UNAVAILABLE", "validation_stages": ["NETWORK_REACHABLE", "RTSP_CONNECTED", "VIDEO_STREAM_FOUND", "FRAME_DECODED", "STREAM_HEALTHY"]}
    except Exception as exc:
        message = str(exc).lower()
        if "401" in message or "unauthorized" in message or "authentication" in message:
            return {"status": "AUTHENTICATION_FAILED", "error_code": "AUTHENTICATION_FAILED", "error_message": "Authentication failed"}
        if "timed out" in message or "unreachable" in message or "refused" in message:
            return {"status": "NETWORK_UNREACHABLE", "error_code": "NETWORK_UNREACHABLE", "error_message": "Unable to connect to camera"}
        return {"status": "STREAM_UNAVAILABLE", "error_code": "STREAM_UNAVAILABLE", "error_message": "Camera stream is unavailable"}
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                logger.warning("Camera validation decoder close failed")


def create_job(db: Session, user_id: int, job_type: str, items: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> CameraOnboardingJob:
    job = CameraOnboardingJob(id=str(uuid.uuid4()), user_id=user_id, job_type=job_type[:30], status="QUEUED", total=len(items), request_metadata=metadata or {})
    db.add(job)
    db.flush()
    for ordinal, payload in enumerate(items, start=1):
        source = str(payload.get("source") or payload.get("rtsp_url") or "").strip()
        safe_payload = dict(payload)
        safe_payload.pop("password", None)
        safe_payload.pop("username", None)
        safe_payload.pop("source", None)
        safe_payload.pop("rtsp_url", None)
        db.add(CameraOnboardingItem(job_id=job.id, ordinal=ordinal, camera_id=str(payload.get("camera_id") or "")[:100] or None, camera_name=str(payload.get("camera_name") or "")[:150] or None, source_encrypted=encrypt_camera_source(source) if source else None, input_data_encrypted=encrypt_connector_config(safe_payload), status="QUEUED"))
    db.add(AuditLog(user_id=user_id, action="CAMERA_ONBOARDING_JOB_CREATED", resource_type="camera_onboarding_job", resource_id=job.id, details={"job_type": job_type, "total": len(items)}))
    db.commit()
    db.refresh(job)
    return job


def job_view(job: CameraOnboardingJob) -> dict[str, Any]:
    return {"job_id": job.id, "job_type": job.job_type, "status": job.status.lower(), "total": job.total, "completed": job.completed, "successful": job.successful, "failed": job.failed, "duplicate": job.duplicate, "processing": max(0, job.total - job.completed), "created_at": job.created_at.isoformat() if job.created_at else None, "completed_at": job.completed_at.isoformat() if job.completed_at else None}


async def run_job(job_id: str, sender: ProgressSender | None = None) -> None:
    db = SessionLocal()
    try:
        job = db.query(CameraOnboardingJob).filter(CameraOnboardingJob.id == job_id).first()
        if not job:
            return
        job.status, job.started_at = "RUNNING", utcnow()
        db.commit()
        items = db.query(CameraOnboardingItem).filter(CameraOnboardingItem.job_id == job_id).order_by(CameraOnboardingItem.ordinal).all()
        semaphore = asyncio.Semaphore(max(1, min(int(os.getenv("CAMERA_ONBOARDING_MAX_CONCURRENCY", "10")), 50)))

        async def process(item_id: int):
            async with semaphore:
                local = SessionLocal()
                try:
                    item = local.query(CameraOnboardingItem).filter(CameraOnboardingItem.id == item_id).first()
                    parent = local.query(CameraOnboardingJob).filter(CameraOnboardingJob.id == job_id).first()
                    item.status = "VALIDATING"
                    local.commit()
                    payload = decrypt_connector_config(item.input_data_encrypted)
                    source = decrypt_camera_source(item.source_encrypted) if item.source_encrypted else ""
                    profile = local.query(CameraCredential).filter(CameraCredential.id == payload.get("credential_profile_id"), CameraCredential.user_id == parent.user_id).first() if payload.get("credential_profile_id") else None
                    effective_source = apply_credentials(source, profile)
                    duplicate = None
                    for camera in local.query(Camera).filter(Camera.user_id == parent.user_id).all():
                        if item.camera_id and camera.cam_id == item.camera_id:
                            duplicate = camera
                            break
                        try:
                            existing = decrypt_camera_source(camera.encrypted_source)
                            if normalize_rtsp(existing) == normalize_rtsp(effective_source):
                                duplicate = camera
                                break
                        except Exception:
                            continue
                    if duplicate:
                        item.status, item.error_code, item.error_message = "DUPLICATE", "ALREADY_REGISTERED", "Already registered"
                        item.registered_camera_id = duplicate.id
                    else:
                        result = await asyncio.to_thread(_validate_stream, effective_source)
                        item.result_metadata = result
                        if result["status"] == "ONLINE":
                            cam_id = item.camera_id or f"CAM_{parent.user_id}_{uuid.uuid4().hex[:12].upper()}"
                            camera = create_camera(local, parent.user_id, cam_id, source, payload.get("source_type") or "rtsp", camera_name=item.camera_name or cam_id, zone=payload.get("zone"), latitude=payload.get("latitude"), longitude=payload.get("longitude"), location_name=payload.get("location"), stream_fps=result.get("fps"), stream_width=result.get("width"), stream_height=result.get("height"), codec=result.get("codec"), commit=False)
                            camera.connection_state = "ONLINE"
                            camera.last_seen_at = utcnow()
                            camera.last_health_check = utcnow()
                            camera.timestamp_status = result.get("timestamp_status")
                            camera.district = payload.get("district")
                            camera.credential_profile_id = profile.id if profile else None
                            camera.group_id = payload.get("group_id")
                            camera.ai_profile_id = payload.get("ai_profile_id")
                            camera.processing_enabled = False
                            camera.analytics_enabled = bool(payload.get("analytics_enabled", False))
                            item.status, item.registered_camera_id = "ONLINE", camera.id
                        else:
                            item.status = result["status"]
                            item.error_code, item.error_message = result.get("error_code"), result.get("error_message")
                    item.updated_at = utcnow()
                    local.commit()
                except Exception:
                    local.rollback()
                    item = local.query(CameraOnboardingItem).filter(CameraOnboardingItem.id == item_id).first()
                    if item:
                        item.status, item.error_code, item.error_message = "STREAM_UNAVAILABLE", "VALIDATION_ERROR", "Camera validation failed"
                        local.commit()
                    logger.exception("Camera onboarding item failed | job_id=%s item_id=%s", job_id, item_id)
                finally:
                    local.close()

        for offset in range(0, len(items), max(1, int(os.getenv("CAMERA_ONBOARDING_MAX_CONCURRENCY", "10")))):
            await asyncio.gather(*(process(item.id) for item in items[offset:offset + max(1, int(os.getenv("CAMERA_ONBOARDING_MAX_CONCURRENCY", "10")))]))
            db.expire_all()
            current = db.query(CameraOnboardingJob).filter(CameraOnboardingJob.id == job_id).first()
            rows = db.query(CameraOnboardingItem).filter(CameraOnboardingItem.job_id == job_id).all()
            current.completed = sum(row.status not in {"QUEUED", "VALIDATING"} for row in rows)
            current.successful = sum(row.status == "ONLINE" for row in rows)
            current.duplicate = sum(row.status == "DUPLICATE" for row in rows)
            current.failed = current.completed - current.successful - current.duplicate
            db.commit()
            if sender:
                await sender(current.user_id, {"type": "camera_onboarding_progress", **job_view(current)})
        job = db.query(CameraOnboardingJob).filter(CameraOnboardingJob.id == job_id).first()
        job.status = "COMPLETED" if job.failed == 0 else "REQUIRES_ATTENTION"
        job.completed_at = utcnow()
        db.commit()
        if sender:
            await sender(job.user_id, {"type": "camera_onboarding_progress", **job_view(job)})
    finally:
        db.close()
