import os
import math
import logging
import secrets
import asyncio
import json
import time
import threading
import re
import httpx
from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, quote
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dotenv import load_dotenv
from jose import jwt, JWTError
load_dotenv()

try:
    _rtsp_timeout_seconds = float(
        os.getenv("RTSP_READ_TIMEOUT_SECONDS", "5.0")
    )
except (TypeError, ValueError):
    _rtsp_timeout_seconds = 5.0
_rtsp_timeout_seconds = max(1.0, min(60.0, _rtsp_timeout_seconds))
_rtsp_timeout_us = int(_rtsp_timeout_seconds * 1_000_000)
os.environ[
    "OPENCV_FFMPEG_CAPTURE_OPTIONS"
] = f"rtsp_transport;tcp|rw_timeout;{_rtsp_timeout_us}"

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Body,
    WebSocket,
    Depends,
    Query,
    HTTPException,
    Request,
    WebSocketDisconnect,
    WebSocketException,
)
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from cache.redisState import (
    set_camera_state,
    update_camera_status,
    increment_camera_frame,
    delete_camera_state,
    set_upload_state,
    delete_upload_state,
    is_camera_running,
    set_stream_session,
    get_stream_session,
    get_active_live_camera_count_from_redis,
    get_camera_state,
)
from fastapi.responses import StreamingResponse, Response,JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, EmailStr,Field
from typing import Any, Dict
from db.model import User,RefreshToken,Camera,Alert,Snapshot,Incident,CameraHealth,AuditLog,SystemEvent,CameraTimeSync,StreamMetric,CameraConnection,CameraIntegration,OutboxEvent,indian_time
from auth.auth import (
    verify_password, create_access_token, ACCESS_TOKEN_EXPIRE_HOURS,
    REFRESH_TOKEN_EXPIRE_DAYS, create_refresh_token, decode_token,
    get_current_user, get_current_user_ws, PRIVATE_KEY, PUBLIC_KEY, ALGORITHM,
    JWT_ISSUER, JWT_AUDIENCE,
)
from systemCapacity import get_system_capacity, can_start_camera
from security.redisClient import redis_ping
from ultralytics import YOLO
import cv2
import numpy as np
import uuid
import supervision as sv
from config import *
from tracker import (trackDetections,clear_tracker,trackVehicleDetections,clear_vehicle_tracker,extractVehicleTracks)
from services.cameraQuality import assess_frame_quality
from services.frameEnhancement import adaptive_enhance, enhance_vehicle_crop
from vehicleState import update_vehicle_state
from vehicleCorrelation import correlate_vehicle,cleanup_global_vehicles,clear_camera_correlation_state
from core.camera_state import CameraState
from core.camera_contract import normalize_camera_config, normalize_source_type
from services.cameraState import normalize_camera_state

try:
    from vehicleCorrelation import (
        get_global_vehicle,
        get_all_global_vehicles,
        get_vehicle_journey,
        get_vehicle_camera_visits,
        get_vehicle_plate_evidence,
        get_global_vehicle_id,
    )
except ImportError:
    get_global_vehicle = None
    get_all_global_vehicles = None
    get_vehicle_journey = None
    get_vehicle_camera_visits = None
    get_vehicle_plate_evidence = None
    get_global_vehicle_id = None
from security.cameraSource import decrypt_camera_source
from security.connectorConfig import decrypt_connector_config
from connectors.manager import validate_connector_config
from vehicleCrop import crop_vehicle
from vehicleReid import get_vehicle_embedding_result
from vehicleANPR import detect_and_read_plate, clear_camera_plate_states
from zonemanager import ZoneManager
from poseutils import iou, human_pose_valid
from state_manager import RuntimeState
from eventfusion import EventFusion, crime_near_person
from snapshot import should_snapshot, cleanAlertMemory, crop_and_encode_snapshot,forget_snapshot_track
from alertDecision import *
from cameraConfig import camera_config
from security.rateLimit import limiter, setup_rate_limiter
from db.database import Base, SessionLocal, engine, get_db, getDB
from routers.gis import router as gis_router
from routers.watchlist import router as watchlist_router
from db.crud import (save_alert,save_snapshot_to_db,get_alerts,get_alerts_after_id,create_camera,get_cameras,delete_camera,get_camera,get_camera_zones,update_camera_zones,get_snapshot_for_user,create_uploaded_video,get_uploaded_video_by_stream,get_uploaded_video_by_cam,delete_uploaded_video_by_cam,set_camera_desired_state)
from services.intelligence_persistence import (
    create_global_vehicle,
    save_vehicle_observation,
    save_correlation_decision,
    save_activity_event,
)
from telegramAlert import sendTelegramAlert
from services.watchlist_service import (
    create_exact_watchlist_alert,
    watchlist_journey_lookback_seconds,
)
from services.identityCorrection import correct_identity
from routers.advanced_intelligence import router as advanced_intelligence_router
from routers.person_intelligence import router as person_intelligence_router
from routers.intelligence_assistant import router as intelligence_assistant_router
from routers.opensearch_search import router as opensearch_search_router
from routers.model_deployment import router as model_deployment_router
from routers.government_data import router as government_data_router
from routers.rbac import router as rbac_router
from routers.integrations import router as integrations_router, ingest_router
from routers.reports import router as reports_router
from routers.auth_security import router as auth_security_router
from db.auth_security_model import UserMFA
from schemas.auth_security import MFAChallengeResponse, MFALoginVerifyRequest
from services.authSecurityService import verify_mfa
from security.redisClient import redis_client, make_key
from cache.cameraLease import get_camera_runtime, get_worker_heartbeat, distributed_capacity_summary, list_camera_runtimes
from services.realtimeBus import start_realtime_subscriber, stop_realtime_subscriber
from security.rbac import permissions_for
from db.advanced_intelligence_model import IncidentEvidence, PersonWatchlistEntry
from db.intelligence_model import VehicleObservation
from services.personRecognition import (
    decrypt_bytes,
    decrypt_embedding,
    detect_face_evidence_candidates,
    extract_embeddings,
    match_embeddings,
)
from services.vehicleAttributes import infer_vehicle_attributes
from services.advancedIntelligenceEngine import AdvancedIntelligenceEngine
from services.personAppearance import appearance_features
from services.personCorrelation import person_correlation_engine
from services.ollamaVision import ollama_plate_verifier
from services.investigationSummary import investigation_summary_worker
from services.evidenceIntegrity import sha256_bytes
from services.objectStorage import (
    ObjectNotFoundError,
    ObjectStorageError,
    get_object_storage,
    storage_readiness,
)
from services.gis_engine import project_pixel_to_geo, haversine_distance_meters, initial_bearing_degrees
from services.alertExplainability import build_alert_explanation
from services.behavioralAnalytics import evaluate_behavior, COOLDOWN as BEHAVIOR_ALERT_COOLDOWN
from services.systemHealth import build_system_health
from services.cameraHealth import camera_health_registry
from services.timeSync import time_synchronizer, system_clock_health
from services.streamManager import FramePacket, stream_manager
from services.mediaTimeline import SourcePTSRequiredError, timeline_from_environment
from services.timestampedCapture import open_timestamped_capture
from services.integrationScheduler import start_integration_scheduler, stop_integration_scheduler
from services.cameraIntegration import integration_view
from services.aiScheduler import AIJob, AIPriority, scheduler as ai_scheduler
from services.vehicleAnalytics import VehicleAnalyticsEngine
from services.observationValidation import validate_vehicle_observation
from ai.tensorrt.runtime import runtime as tensorrt_runtime
from events.outbox import init_outbox
from events.kafkaProducer import init_kafka, close_kafka, kafka_status
from events.outboxWorker import start_outbox_worker, stop_outbox_worker
from monitoring.metrics import (
    setup_prometheus,
    alerts_detected_total,
    redis_health,
    kafka_health,
    active_ws_clients,
    alert_processing_seconds,
    camera_read_failures_total,
    camera_reconnect_total,
    camera_decoder_failures_total,
    vehicle_inference_seconds,
    vehicle_detections_total,
    vehicle_tracking_active,
    vehicle_correlation_seconds,
    vehicle_correlation_score,
    reid_inference_seconds,
    anpr_inference_seconds,
    gpu_utilization_percent,
    gpu_memory_percent,
    ai_queue_depth, ai_workers_active, ai_jobs_submitted_total,
    ai_jobs_completed_total, ai_jobs_failed_total, ai_jobs_dropped_total,
    ai_gpu_utilization_percent, ai_gpu_memory_percent,
    vehicle_observations_accepted_total, vehicle_observations_rejected_total,
    vehicle_reid_attempts_total, vehicle_anpr_attempts_total, vehicle_pipeline_errors_total,
)
import magic


class CameraStartRequest(BaseModel):
    source: str
    source_type: str = "webcam"
    camera_name: str = ""
    zone: str = ""


class CameraCreate(BaseModel):
    camera_name: str = ""
    zone: str = ""
    source: str
    source_type: str = "rtsp"

    latitude: float | None = Field(
        default=None,
        ge=-90,
        le=90,
    )

    longitude: float | None = Field(
        default=None,
        ge=-180,
        le=180,
    )

    location_name: str | None = Field(
        default=None,
        max_length=255,
    )

    connector_config: dict | None = None

    vendor: str | None = Field(
        default=None,
        max_length=100,
    )

    vendor_device_id: str | None = Field(
        default=None,
        max_length=150,
    )

    stream_profile: str | None = Field(
        default=None,
        max_length=256,
    )

    direction: str | None = Field(default=None, max_length=20)
    stream_fps: float | None = Field(default=None, gt=0, le=120)
    stream_width: int | None = Field(default=None, ge=160, le=16384)
    stream_height: int | None = Field(default=None, ge=120, le=16384)
    transport: str | None = Field(default=None, max_length=20)
    codec: str | None = Field(default=None, max_length=32)
    altitude: float | None = Field(default=None, ge=-1000, le=10000)
    heading: float | None = Field(default=None, ge=0, le=360)
    fov: float | None = Field(default=None, gt=0, le=360)
    road_name: str | None = Field(default=None, max_length=255)
    city: str | None = Field(default=None, max_length=100)
    state: str | None = Field(default=None, max_length=100)
    country: str | None = Field(default=None, max_length=100)


class CameraConnectorTestRequest(BaseModel):
    source: str
    source_type: str
    connector_config: dict | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ZonesUpdateRequest(BaseModel):
    zones: list

class CameraZoneUpdateRequest(BaseModel):
    zone: str

class RefreshRequest(BaseModel):
    refresh_token: str

class LogoutRequest(BaseModel):
    refresh_token: str

app = FastAPI(
    title="Production CCTV Pre-Event Alert Backend",
    docs_url="/docs" if os.getenv("ENV", "dev") != "prod" else None,
    redoc_url=None,
)
setup_rate_limiter(app)
setup_prometheus(app)

app.include_router(gis_router)
app.include_router(watchlist_router)
app.include_router(advanced_intelligence_router)
app.include_router(person_intelligence_router)
app.include_router(intelligence_assistant_router)
app.include_router(opensearch_search_router)
app.include_router(model_deployment_router)
app.include_router(government_data_router)
app.include_router(rbac_router)
app.include_router(integrations_router)
app.include_router(ingest_router)
app.include_router(reports_router)
app.include_router(auth_security_router)

ENV = os.getenv("ENV", "dev").strip().lower()
IS_PROD = ENV == "prod"

DISTRIBUTED_CAMERA_WORKERS_ENABLED = os.getenv(
    "INTEL_I_DISTRIBUTED_CAMERA_WORKERS", "false"
).strip().lower() in {"1", "true", "yes", "on"}
WORKER_PREVIEW_TOKEN = os.getenv("INTEL_I_WORKER_PREVIEW_TOKEN", "").strip()
if DISTRIBUTED_CAMERA_WORKERS_ENABLED and IS_PROD and not WORKER_PREVIEW_TOKEN:
    raise RuntimeError(
        "INTEL_I_WORKER_PREVIEW_TOKEN is required when distributed camera workers are enabled in production"
    )
if DISTRIBUTED_CAMERA_WORKERS_ENABLED and not WORKER_PREVIEW_TOKEN:
    WORKER_PREVIEW_TOKEN = "intel-i-local-preview"

AUTO_CREATE_SCHEMA = (
    os.getenv("AUTO_CREATE_SCHEMA", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)


if not IS_PROD and AUTO_CREATE_SCHEMA:
    Base.metadata.create_all(bind=engine)

FRONTEND_URL = os.getenv("FRONTEND_URL")

if not FRONTEND_URL:
    raise RuntimeError("FRONTEND_URL is missing")

FRONTEND_URL = FRONTEND_URL.strip().rstrip("/")
_frontend_origin = urlparse(FRONTEND_URL)

if _frontend_origin.scheme not in {"http", "https"}:
    raise RuntimeError("FRONTEND_URL must use http or https")

if not _frontend_origin.netloc:
    raise RuntimeError("FRONTEND_URL must contain a valid origin")

if _frontend_origin.path not in {"", "/"} or _frontend_origin.query or _frontend_origin.fragment:
    raise RuntimeError("FRONTEND_URL must be an origin without a path, query, or fragment")

if IS_PROD and _frontend_origin.scheme != "https":
    raise RuntimeError("Production FRONTEND_URL must use HTTPS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(
        [FRONTEND_URL]
        + [
            origin.strip().rstrip("/")
            for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
            if origin.strip()
        ]
    )),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "X-CSRF-Token",
        "Authorization"],
    expose_headers=[
        "Content-Disposition",
        "X-Report-ID",
        "X-Export-ID",
        "X-Evidence-Manifest-SHA256",
        "X-PDF-SHA256",
    ],
)

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

CSRF_EXEMPT_PATHS = {
    "/",
    "/health",
    "/auth/csrf",
    "/auth/login",
    "/auth/mfa/login/verify",
    "/auth/password/forgot",
    "/auth/password/reset",
}

@app.middleware("http")
async def csrf_protect(request: Request, call_next):
    if request.method in SAFE_METHODS:
        return await call_next(request)

    if request.url.path in CSRF_EXEMPT_PATHS:
        return await call_next(request)

    csrf_cookie = request.cookies.get(CSRF_COOKIE_NAME)
    csrf_header = request.headers.get("X-CSRF-Token")

    if not csrf_cookie or not csrf_header or csrf_cookie != csrf_header:
        return JSONResponse(
            status_code=403,
            content={"detail": "Invalid CSRF token"},
        )

    return await call_next(request)

logger = logging.getLogger(__name__)


def _record_audit_event(db, *, user_id=None, action: str, resource_type: str, resource_id=None, details=None, source_ip=None):
    """Best-effort persistent audit record; never logs secrets."""
    try:
        db.add(AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id is not None else None,
            details=details or {},
            source_ip=source_ip,
        ))
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Audit log write failed | action=%s resource=%s", action, resource_type)


def _record_system_event(*, event_type: str, component: str, message: str, severity: str = "INFO", metadata=None):
    """Persist a non-secret system lifecycle event."""
    try:
        with SessionLocal() as db:
            db.add(SystemEvent(
                event_type=event_type,
                severity=severity,
                component=component,
                message=message,
                metadata_json=metadata or {},
            ))
            db.commit()
    except Exception:
        logger.debug("System event persistence failed", exc_info=True)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)

    # These headers are safe defaults for the API and do not expose
    # application secrets or implementation details.
    response.headers.setdefault(
        "X-Content-Type-Options",
        "nosniff",
    )
    response.headers.setdefault(
        "X-Frame-Options",
        "DENY",
    )
    response.headers.setdefault(
        "Referrer-Policy",
        "no-referrer",
    )
    response.headers.setdefault(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=()",
    )

    if IS_PROD:
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains",
        )

    return response


@app.get("/health", include_in_schema=False)
def liveness_health():
    """Process liveness for Docker/RunPod health checks."""
    return {"status": "ok", "service": "intel-i-backend"}


@app.get("/ready", include_in_schema=False)
def readiness_health():
    """Dependency readiness; Kafka is required only when explicitly enabled."""
    database_ok = False
    try:
        with SessionLocal() as readiness_db:
            readiness_db.execute(text("SELECT 1"))
            database_ok = True
    except Exception:
        logger.exception("Readiness database check failed")

    try:
        redis_ok = bool(redis_ping())
    except Exception:
        redis_ok = False
        logger.exception("Readiness Redis check failed")

    kafka_required = os.getenv("KAFKA_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    kafka_ok = bool(kafka_status()) if kafka_required else True

    from services.runtimeModelReadiness import cached_model_readiness
    model_state = cached_model_readiness()
    model_required = os.getenv(
        "MODEL_READINESS_REQUIRED", "true" if IS_PROD else "false"
    ).strip().lower() in {"1", "true", "yes", "on"}
    models_ok = model_state.get("status") == "READY"

    ollama_required = os.getenv(
        "OLLAMA_VISION_REQUIRED", "true" if (IS_PROD and ollama_plate_verifier.enabled) else "false"
    ).strip().lower() in {"1", "true", "yes", "on"}
    ollama_state = ollama_plate_verifier.probe_model() if ollama_plate_verifier.enabled else {"enabled": False, "available": False, "reason": "disabled"}
    ollama_ok = bool(ollama_state.get("available")) if ollama_required else True

    ready = bool(
        database_ok and redis_ok and kafka_ok
        and (models_ok or not model_required)
        and ollama_ok
    )
    payload = {
        "status": "ready" if ready else "not_ready",
        "database": database_ok,
        "redis": redis_ok,
        "kafka": kafka_ok if kafka_required else "disabled",
        "models": model_state,
        "models_required": model_required,
        "ollama_vision": ollama_state,
        "ollama_vision_required": ollama_required,
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)


COOKIE_SECURE = IS_PROD
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax").strip().lower()
COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN") or None

if COOKIE_SAMESITE not in {"lax", "strict", "none"}:
    raise RuntimeError("COOKIE_SAMESITE must be lax, strict, or none")

if IS_PROD and COOKIE_SAMESITE == "none" and not COOKIE_SECURE:
    raise RuntimeError("SameSite=None requires Secure cookies in production")

if COOKIE_DOMAIN and (
    COOKIE_DOMAIN.strip() != COOKIE_DOMAIN
    or "/" in COOKIE_DOMAIN
    or " " in COOKIE_DOMAIN
):
    raise RuntimeError("COOKIE_DOMAIN is invalid")
def cookie_options(path: str, max_age: int | None = None):
    options = {
        "httponly": True,
        "secure": COOKIE_SECURE,
        "samesite": COOKIE_SAMESITE,
        "path": path,
    }

    if max_age is not None:
        options["max_age"] = max_age

    if COOKIE_DOMAIN:
        options["domain"] = COOKIE_DOMAIN

    return options

ACCESS_COOKIE_NAME = "access_token"
REFRESH_COOKIE_NAME = "refresh_token"
CSRF_COOKIE_NAME = "csrf_token"

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "temp_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def _bounded_int_env(
    name: str,
    default: int,
    minimum: int = 1,
    maximum: int = 10_000_000,
) -> int:
    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = int(raw_value.strip())
    except (TypeError, ValueError):
        return default

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )

def _bounded_float_env(
    name: str,
    default: float,
    minimum: float = 0.0,
    maximum: float = 1_000_000.0,
) -> float:
    """
    Safely read a bounded floating-point value from environment variables.
    """
    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = float(raw_value.strip())
    except (TypeError, ValueError):
        return default

    if not math.isfinite(value):
        return default

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )

MAX_UPLOAD_SIZE_MB = _bounded_int_env(
    "MAX_UPLOAD_SIZE_MB",
    500,
    1,
    4096,
)
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

STREAM_SESSION_TTL = _bounded_int_env(
    "STREAM_SESSION_TTL",
    60,
    15,
    3600,
)

STREAM_IDLE_SLEEP_SECONDS = _bounded_float_env(
    "STREAM_IDLE_SLEEP_SECONDS",
    0.03,
    0.01,
    1.0,
)



OUTPUT_STREAM_FPS = _bounded_int_env(
    "OUTPUT_STREAM_FPS",
    int(DEFAULT_FPS),
    1,
    60,
)

RTSP_RECONNECT_INITIAL_SECONDS = _bounded_float_env(
    "RTSP_RECONNECT_INITIAL_SECONDS",
    60.0,
    1.0,
    60.0,
)

RTSP_RECONNECT_MAX_SECONDS = _bounded_float_env(
    "RTSP_RECONNECT_MAX_SECONDS",
    60.0,
    1.0,
    60.0,
)

RTSP_READ_TIMEOUT_SECONDS = _bounded_float_env(
    "RTSP_READ_TIMEOUT_SECONDS",
    60.0,
    1.0,
    60.0,
)

STREAM_GAP_THRESHOLD_SECONDS = _bounded_float_env(
    "STREAM_GAP_THRESHOLD_SECONDS",
    3.0,
    0.5,
    60.0,
)

STREAM_BUFFER_SIZE = _bounded_int_env("STREAM_BUFFER_SIZE", 3, 1, 30)
CAMERA_HEALTH_PERSIST_SECONDS = _bounded_float_env("CAMERA_HEALTH_PERSIST_SECONDS", 5.0, 1.0, 60.0)
CAMERA_HEALTH_STALE_SECONDS = _bounded_float_env("CAMERA_HEALTH_STALE_SECONDS", 8.0, 2.0, 120.0)
CAMERA_HEALTH_FPS_MIN = _bounded_float_env("CAMERA_HEALTH_FPS_MIN", 0.5, 0.1, 120.0)
CAMERA_MAX_CLOCK_OFFSET_MS = _bounded_float_env("CAMERA_MAX_CLOCK_OFFSET_MS", 5000.0, 100.0, 120000.0)
STREAM_METRIC_PERSIST_SECONDS = _bounded_float_env("STREAM_METRIC_PERSIST_SECONDS", 5.0, 1.0, 60.0)
SCENE_DISCONTINUITY_THRESHOLD_SECONDS = _bounded_float_env(
    "SCENE_DISCONTINUITY_THRESHOLD_SECONDS",
    3.0,
    0.5,
    60.0,
)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_MAGIC_VIDEO_TYPES = {
    "video/mp4",
    "video/x-msvideo",
    "video/quicktime",
    "video/x-matroska"
}

def stream_cookie_name(stream_type: str, target_id: str) -> str:
    if stream_type not in {"camera", "upload"}:
        raise ValueError("Invalid stream type")

    safe_target = re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        str(target_id or ""),
    )[:128]

    if not safe_target:
        raise ValueError("Invalid stream target")

    return f"stream_sid_{stream_type}_{safe_target}"

def set_auth_cookies(response: Response, access_token: str, refresh_token: str):
    response.set_cookie(
        key=ACCESS_COOKIE_NAME,
        value=access_token,
        **cookie_options(
            path="/",
            max_age=60 * 60 * ACCESS_TOKEN_EXPIRE_HOURS,
        ),
    )

    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=refresh_token,
        **cookie_options(
            path="/auth",
            max_age=60 * 60 * 24 * REFRESH_TOKEN_EXPIRE_DAYS,
        ),
    )

def set_stream_cookie(
    response: Response,
    stream_type: str,
    target_id: str,
    sid: str,
):
    path = "/stream" if stream_type == "camera" else "/stream-upload"

    options = {
        "key": stream_cookie_name(stream_type, target_id),
        "value": sid,
        "httponly": True,
        "secure": COOKIE_SECURE,
        "samesite": COOKIE_SAMESITE,
        "path": path,
        "max_age": STREAM_SESSION_TTL,
    }

    if COOKIE_DOMAIN:
        options["domain"] = COOKIE_DOMAIN

    response.set_cookie(**options)

def set_csrf_cookie(response: Response) -> str:
    csrf_token = secrets.token_urlsafe(32)

    options = cookie_options(
        path="/",
        max_age=60 * 60 * 24 * REFRESH_TOKEN_EXPIRE_DAYS,
    )

    options["httponly"] = False

    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=csrf_token,
        **options,
    )

    return csrf_token


def delete_cookie_exact(response: Response, key: str, path: str):
    response.delete_cookie(
        key=key,
        path=path,
        domain=COOKIE_DOMAIN,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
    )


def clear_auth_cookies(response: Response):
    delete_cookie_exact(response, ACCESS_COOKIE_NAME, "/")
    delete_cookie_exact(response, REFRESH_COOKIE_NAME, "/auth")
    delete_cookie_exact(response, CSRF_COOKIE_NAME, "/")
    
def get_cookie_user(request: Request, db: Session) -> User:
    token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    payload = decode_token(token)

    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid access token")

    subject = payload.get("sub")

    if not subject:
        raise HTTPException(status_code=401, detail="Invalid token subject")

    try:
        user_id = int(subject)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token subject")

    user = db.query(User).filter(User.id == user_id).first()

    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not active")

    return user

def validate_video_upload(videoFile: UploadFile):
    filename = Path(videoFile.filename or "").name

    if not filename or filename in {".", ".."}:
        raise HTTPException(
            status_code=400,
            detail="Invalid video filename",
        )

    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video extension",
        )

    content_type = (
        videoFile.content_type or ""
    ).split(";", 1)[0].strip().lower()

    # Do not trust this header for security. It is only an early rejection;
    # validate_saved_video_file() performs the authoritative libmagic check.
    if content_type and content_type not in ALLOWED_MAGIC_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video MIME type",
        )

def validate_saved_video_file(file_path: Path):
    if not file_path.exists():
        raise HTTPException(status_code=400, detail="Uploaded file missing")

    real_mime = magic.from_file(str(file_path), mime=True)

    if real_mime not in ALLOWED_MAGIC_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video content type: {real_mime}",
        )

    cap = cv2.VideoCapture(str(file_path))

    try:
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Invalid or corrupted video")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_fps = cap.get(cv2.CAP_PROP_FPS)

        if frame_count <= 0:
            raise HTTPException(status_code=400, detail="Video has no frames")

        if duration_fps <= 0 or duration_fps > 240:
            raise HTTPException(status_code=400, detail="Invalid video FPS")

        ok, frame = cap.read()

        if not ok or frame is None:
            raise HTTPException(status_code=400, detail="Video has no readable frames")

    finally:
        cap.release()

def safe_storage_key(cam_id: str, ext: str) -> str:
    safe_cam_id = (
        str(cam_id)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("..", "_")
    )

    return f"{safe_cam_id}{ext}"


def resolve_upload_ingest_path(storage_key: str) -> Path:
    """
    Local, short-lived ingest path used only while validating a new upload.

    Durable uploaded-video bytes live in object storage. Keeping this path
    separate prevents MinIO/S3 object keys from ever being interpreted as
    arbitrary local filesystem paths.
    """
    safe_key = Path(storage_key).name
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR / f".ingest-{uuid.uuid4().hex}-{safe_key}"


def resolve_uploaded_video_path(storage_key: str) -> Path:
    """
    Materialize a private uploaded-video object to the verified processing cache.

    OpenCV/FFmpeg continue receiving a normal local path, so the existing AI,
    ANPR, tracking and correlation pipeline does not need to know about MinIO.
    """
    try:
        return get_object_storage().materialize_to_cache(str(storage_key))
    except ObjectNotFoundError as exc:
        raise FileNotFoundError(str(storage_key)) from exc


person_model = YOLO(MODEL_PERSON)
pose_model = YOLO(MODEL_POSE)
crime_model = YOLO(MODEL_CRIME)
ppe_model = YOLO(MODEL_PPE)

# ------------------------------------------------------------
# VEHICLE DETECTION MODEL
# ------------------------------------------------------------
#
# This is the existing YOLOv8m vehicle detector.
# It is separate from the license-plate model and Re-ID model.
#
VEHICLE_MODEL_PATH = os.getenv(
    "VEHICLE_MODEL",
    "models/yolov8m.pt",
)

vehicle_model = YOLO(
    VEHICLE_MODEL_PATH
)

VEHICLE_MODEL_REQUIRED = (
    os.getenv("VEHICLE_MODEL_REQUIRED", "true" if os.getenv("ENV", "dev").strip().lower() == "prod" else "false")
    .strip().lower()
    in {"1", "true", "yes", "on"}
)
if VEHICLE_MODEL_REQUIRED and not Path(VEHICLE_MODEL_PATH).is_file():
    raise RuntimeError(
        f"Production vehicle model is missing: {VEHICLE_MODEL_PATH}"
    )

VEHICLE_DETECTOR_MODE = os.getenv(
    "VEHICLE_DETECTOR_MODE",
    "dedicated",
).strip().lower()

VEHICLE_CONF = _bounded_float_env(
    "VEHICLE_CONF",
    0.35,
    0.05,
    0.99,
)

VEHICLE_IOU = _bounded_float_env(
    "VEHICLE_IOU",
    0.45,
    0.05,
    0.95,
)

VEHICLE_REID_MIN_CROP_QUALITY = _bounded_float_env(
    "VEHICLE_REID_MIN_CROP_QUALITY",
    0.45,
    0.10,
    0.95,
)

VEHICLE_REID_RUN_EVERY_N_FRAMES = _bounded_int_env(
    "VEHICLE_REID_RUN_EVERY_N_FRAMES",
    5,
    1,
    30,
)


def _resolve_vehicle_class_ids(model):
    """Resolve vehicle classes from model.names, with COCO fallback."""
    names = getattr(model, "names", {}) or {}
    if isinstance(names, list):
        names = {i: value for i, value in enumerate(names)}

    wanted = {
        "car", "truck", "bus", "motorcycle", "motorbike",
        "van", "vehicle", "auto", "auto-rickshaw", "rickshaw",
        "bicycle",
    }
    ids = []

    for raw_id, name in names.items():
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError):
            continue

        normalized = (
            str(name or "")
            .strip()
            .lower()
            .replace("_", "-")
        )

        if (
            normalized in wanted
            or any(
                token in normalized
                for token in (
                    "car",
                    "truck",
                    "bus",
                    "motor",
                    "vehicle",
                    "rickshaw",
                )
            )
        ):
            ids.append(class_id)

    return sorted(set(ids)) or [2, 3, 5, 7]


def _resolve_configured_vehicle_class_ids(model):
    raw = os.getenv("VEHICLE_CLASS_IDS", "").strip()
    if raw:
        ids = []
        for token in raw.split(","):
            try:
                value = int(token.strip())
                if value >= 0:
                    ids.append(value)
            except (TypeError, ValueError):
                continue
        if ids:
            return sorted(set(ids))
    return _resolve_vehicle_class_ids(model)


VEHICLE_CLASS_IDS = _resolve_configured_vehicle_class_ids(vehicle_model)
vehicle_analytics_engine = VehicleAnalyticsEngine(
    fallback_model=vehicle_model,
    class_ids=VEHICLE_CLASS_IDS,
    device=DEVICE,
)

logger.info(
    "Vehicle model loaded | path=%s",
    VEHICLE_MODEL_PATH,
)

# ------------------------------------------------------------
# Persistent AI model metadata
# ------------------------------------------------------------
PERSON_MODEL_VERSION = os.getenv("PERSON_MODEL_VERSION", "unknown")
POSE_MODEL_VERSION = os.getenv("POSE_MODEL_VERSION", "unknown")
VEHICLE_MODEL_VERSION = os.getenv("VEHICLE_MODEL_VERSION", "unknown")
VEHICLE_REID_MODEL_VERSION = os.getenv(
    "VEHICLE_REID_MODEL_VERSION",
    "unknown",
)
VEHICLE_ANPR_MODEL_VERSION = os.getenv(
    "VEHICLE_ANPR_MODEL_VERSION",
    "unknown",
)
EVENTFUSION_VERSION = os.getenv(
    "EVENTFUSION_VERSION",
    "unknown",
)

# Vehicle-attribute logging cadence.
# The PP-LCNet model is already invoked for each tracked vehicle crop.
# We throttle terminal logging per camera/track so a video does not flood
# the backend console while still exposing real model output.
try:
    VEHICLE_ATTRIBUTE_LOG_EVERY_N_FRAMES = max(1, int(os.getenv("VEHICLE_ATTRIBUTE_LOG_EVERY_N_FRAMES", "1")))
except (TypeError, ValueError):
    VEHICLE_ATTRIBUTE_LOG_EVERY_N_FRAMES = 1

VEHICLE_ATTRIBUTE_OVERLAY_ENABLED = (
    os.getenv("VEHICLE_ATTRIBUTE_OVERLAY_ENABLED", "false")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

# Detection overlays intentionally show only person/vehicle tracking and ANPR.
# Vehicle make/model/color/subtype attributes remain available to the backend
# intelligence pipeline and logs, but are not rendered on the video.
VEHICLE_DETECTION_OVERLAY_ENABLED = (
    os.getenv("VEHICLE_DETECTION_OVERLAY_ENABLED", "true")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)
ANPR_OVERLAY_ENABLED = (
    os.getenv("ANPR_OVERLAY_ENABLED", "true")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

# Routine plate detections are displayed on the video instead of flooding the
# backend terminal. Set this to true only when ANPR console diagnostics are
# explicitly required.
ANPR_TERMINAL_LOGGING_ENABLED = (
    os.getenv("ANPR_TERMINAL_LOGGING_ENABLED", "false")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

FRAME_ENHANCEMENT_ENABLED = (
    os.getenv("FRAME_ENHANCEMENT_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
FRAME_QUALITY_CHECK_EVERY_N_FRAMES = _bounded_int_env(
    "FRAME_QUALITY_CHECK_EVERY_N_FRAMES", 5, 1, 60
)
FRAME_ENHANCEMENT_QUALITY_THRESHOLD = _bounded_float_env(
    "FRAME_ENHANCEMENT_QUALITY_THRESHOLD", 0.72, 0.30, 0.95
)
FRAME_QUALITY_OVERLAY_ENABLED = (
    os.getenv("FRAME_QUALITY_OVERLAY_ENABLED", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)

# Alerts are still evaluated, persisted and delivered, but their rule/severity
# boxes must not cover the clean person/vehicle/ANPR detection view.
ALERT_OVERLAY_ENABLED = (
    os.getenv("ALERT_OVERLAY_ENABLED", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)

ADVANCED_INTELLIGENCE_ENABLED = (
    os.getenv("ADVANCED_INTELLIGENCE_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
CAMERA_TAMPER_ENABLED = (
    os.getenv("CAMERA_TAMPER_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
FROZEN_FRAME_ENABLED = (
    os.getenv("FROZEN_FRAME_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
RISK_SCORING_ENABLED = (
    os.getenv("RISK_SCORING_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
INCIDENT_CORRELATION_ENABLED = (
    os.getenv("INCIDENT_CORRELATION_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
EVIDENCE_HASHING_ENABLED = (
    os.getenv("EVIDENCE_HASHING_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
BEHAVIOR_BEST_EVIDENCE_ENABLED = (
    os.getenv("BEHAVIOR_BEST_EVIDENCE_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
BEHAVIOR_EVIDENCE_RULES = {
    value.strip().casefold()
    for value in os.getenv(
        "BEHAVIOR_EVIDENCE_RULES",
        "Chain Snatching Risk,Fighting,Assault,Weapon Threat",
    ).split(",")
    if value.strip()
}
BEHAVIOR_EVIDENCE_WINDOW_SECONDS = _bounded_float_env(
    "BEHAVIOR_EVIDENCE_WINDOW_SECONDS", 5.0, 1.0, 15.0
)
BEHAVIOR_EVIDENCE_MIN_COLLECTION_SECONDS = _bounded_float_env(
    "BEHAVIOR_EVIDENCE_MIN_COLLECTION_SECONDS", 2.0, 0.5, 10.0
)
BEHAVIOR_EVIDENCE_SAMPLE_FPS = _bounded_float_env(
    "BEHAVIOR_EVIDENCE_SAMPLE_FPS", 1.0, 0.25, 4.0
)
BEHAVIOR_EVIDENCE_MIN_FACE_SIZE = _bounded_int_env(
    "BEHAVIOR_EVIDENCE_MIN_FACE_SIZE", 45, 20, 300
)
BEHAVIOR_EVIDENCE_MIN_FACE_CONFIDENCE = _bounded_float_env(
    "BEHAVIOR_EVIDENCE_MIN_FACE_CONFIDENCE", 0.75, 0.10, 0.99
)
BEHAVIOR_EVIDENCE_MIN_BLUR_SCORE = _bounded_float_env(
    "BEHAVIOR_EVIDENCE_MIN_BLUR_SCORE", 30.0, 0.0, 1000.0
)
BEHAVIOR_EVIDENCE_EARLY_ACCEPT_SCORE = _bounded_float_env(
    "BEHAVIOR_EVIDENCE_EARLY_ACCEPT_SCORE", 0.88, 0.50, 0.99
)
BEHAVIOR_EVIDENCE_ROI_PADDING_RATIO = _bounded_float_env(
    "BEHAVIOR_EVIDENCE_ROI_PADDING_RATIO", 0.35, 0.0, 1.0
)
BEHAVIOR_EVIDENCE_MAX_CONCURRENT_INCIDENTS = _bounded_int_env(
    "BEHAVIOR_EVIDENCE_MAX_CONCURRENT_INCIDENTS", 2, 1, 100
)
BEHAVIOR_EVIDENCE_ALLOW_CONTEXT_FALLBACK = (
    os.getenv("BEHAVIOR_EVIDENCE_ALLOW_CONTEXT_FALLBACK", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
try:
    _behavior_camera_groups_value = json.loads(
        os.getenv("BEHAVIOR_EVIDENCE_CAMERA_GROUPS_JSON", "{}") or "{}"
    )
    BEHAVIOR_EVIDENCE_CAMERA_GROUPS = {
        str(key): str(value)
        for key, value in _behavior_camera_groups_value.items()
    } if isinstance(_behavior_camera_groups_value, dict) else {}
except (TypeError, ValueError, json.JSONDecodeError):
    BEHAVIOR_EVIDENCE_CAMERA_GROUPS = {}
ADAPTIVE_PIPELINE_ENABLED = (
    os.getenv("ADAPTIVE_PIPELINE_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
PERSON_APPEARANCE_ENABLED = (
    os.getenv("PERSON_APPEARANCE_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
CROWD_INTELLIGENCE_ENABLED = (
    os.getenv("CROWD_INTELLIGENCE_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
ANPR_CHARACTER_VOTING_ENABLED = (
    os.getenv("ANPR_CHARACTER_VOTING_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
CONFIDENCE_CALIBRATION_ENABLED = (
    os.getenv("CONFIDENCE_CALIBRATION_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
EDSR_ENABLED = (
    os.getenv("EDSR_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
EDSR_MODEL_PATH = os.getenv("EDSR_MODEL_PATH", "./models/EDSR_x4.pb")
EDSR_DEVICE = os.getenv("EDSR_DEVICE", "cpu").strip().lower()
EDSR_MIN_QUALITY = _bounded_float_env("EDSR_MIN_QUALITY", 0.60, 0.20, 0.90)

# DarkIR is used only for low-light/very-dark frames and always runs
# before YOLO. EDSR remains a post-detection ROI enhancement stage.
DARKIR_ENABLED = (
    os.getenv("DARKIR_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
DARKIR_MODEL_PATH = os.getenv("DARKIR_MODEL_PATH", "./models/DarkIR_384.pt")
DARKIR_DEVICE = os.getenv("DARKIR_DEVICE", "auto").strip().lower()
DARKIR_MAX_WIDTH = _bounded_int_env("DARKIR_MAX_WIDTH", 1280, 320, 1920)
DARKIR_MAX_HEIGHT = _bounded_int_env("DARKIR_MAX_HEIGHT", 720, 240, 1080)
DARKIR_STRICT = (
    os.getenv("DARKIR_STRICT", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)
DARKIR_SHA256 = os.getenv("DARKIR_SHA256", "").strip().lower()
DARKIR_MIN_QUALITY = _bounded_float_env("DARKIR_MIN_QUALITY", 0.72, 0.20, 0.95)
ADAPTIVE_OUTPUT_ENABLED = (
    os.getenv("ADAPTIVE_OUTPUT_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)

zone_manager = ZoneManager()
runtime_state = RuntimeState()
event_fusion = EventFusion(runtime_state)


cameraIDS = {}
cameraFrameCount = {}
activeAlertBoxes = {}
cameraQualityStates = {}
cameraQualityLock = threading.Lock()
advanced_intelligence = AdvancedIntelligenceEngine(
    edsr_model_path=EDSR_MODEL_PATH,
    edsr_device=EDSR_DEVICE,
    darkir_model_path=DARKIR_MODEL_PATH,
    darkir_device=DARKIR_DEVICE,
    darkir_max_width=DARKIR_MAX_WIDTH,
    darkir_max_height=DARKIR_MAX_HEIGHT,
    darkir_strict=DARKIR_STRICT,
    darkir_sha256=DARKIR_SHA256,
)
cameraAdaptivePolicies = {}
cameraQualityTelemetry = {}
cameraTamperTelemetry = {}
cameraFrozenTelemetry = {}
cameraTimelines = {}
personAppearanceStates = {}
behaviorEvidenceLock = threading.RLock()
behaviorEvidenceGroups = {}
behaviorEvidenceCameraGroups = {}
personAppearanceLastFrame = {}
uploadStreamToCam = {}
clients = set()
uploadvideos = {}

cameraWorkers = {}
cameraStopEvents = {}
cameraLocks = {}
latestJpegs = {}
uploadCamToStream = {}
cameraConnectionStates = {}

# Last frame on which vehicle-attribute output was printed for a
# camera/track pair. Bounded by active tracks and cleaned with camera state.
vehicleAttributeLastLogFrame = {}

# Last frame/value logged for each camera/track ANPR result.
vehiclePlateLastLogFrame = {}
vehiclePlateLastValue = {}

# -----------------------------------------------------------------
# Cross-camera correlation + GIS presentation state
# -----------------------------------------------------------------
# The correlation engine owns identity matching. This bounded registry
# stores only presentation-safe metadata needed by the GIS/dashboard:
# global vehicle ID, local camera/track, timestamps, similarity, and
# camera coordinates. Embeddings, crops, plaintext sources, and plate
# text are never stored here.
correlationGisState = {}
cameraCorrelationMeta = {}
liveVehiclePositionLastEmit = {}
personGisState = {}
livePersonPositionLastEmit = {}
personTopologyCache = {}
MAX_CORRELATION_GIS_VEHICLES = _bounded_int_env(
    "MAX_CORRELATION_GIS_VEHICLES",
    10000,
    100,
    100000,
)
MAX_CORRELATION_GIS_OBSERVATIONS = _bounded_int_env(
    "MAX_CORRELATION_GIS_OBSERVATIONS",
    200,
    10,
    1000,
)
CORRELATION_GIS_DEDUP_SECONDS = _bounded_float_env(
    "CORRELATION_GIS_DEDUP_SECONDS",
    2.0,
    0.25,
    60.0,
)
LIVE_VEHICLE_POSITION_INTERVAL_SECONDS = _bounded_float_env(
    "LIVE_VEHICLE_POSITION_INTERVAL_SECONDS",
    0.50,
    0.10,
    5.0,
)
LIVE_VEHICLE_STALE_SECONDS = _bounded_float_env(
    "LIVE_VEHICLE_STALE_SECONDS",
    10.0,
    2.0,
    120.0,
)

MAX_CAMERA_WORKERS = _bounded_int_env("MAX_CAMERA_WORKERS", 64, 1, 128)
executor = ThreadPoolExecutor(max_workers=MAX_CAMERA_WORKERS)

main_loop = None

# FastAPI startup/readiness state. This is separate from process liveness.
SYSTEM_INITIALIZATION_COMPLETE = False
SYSTEM_INITIALIZATION_STARTED_AT = time.time()
SYSTEM_INITIALIZATION_COMPLETED_AT = None

state_lock = threading.RLock()
model_lock = threading.Lock()
client_lock = asyncio.Lock()


# -----------------------------------------------------------------------------
# ALERT PERSISTENCE CAMERA/OWNER RESOLUTION
# -----------------------------------------------------------------------------
# A normal alert is always tied to a persisted Camera row.  The runtime fallback
# below exists only for a narrow race/integrity case: a worker was started from
# an authenticated, backend-generated camera/upload ID and the Camera row cannot
# be read at alert-persistence time.  It never trusts an arbitrary client ID.
#
# This prevents detections from disappearing silently while still preserving
# tenant isolation.  Missing Camera rows are logged loudly so the inventory can
# be repaired instead of being hidden.
ALERT_RUNTIME_OWNER_FALLBACK_ENABLED = (
    os.getenv("ALERT_RUNTIME_OWNER_FALLBACK_ENABLED", "true")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)


def _runtime_owner_from_backend_camera_id(cam_id: str) -> int | None:
    """Extract owner only from INTEL-I generated CAM_/UPLOAD_ identifiers."""
    candidate = str(cam_id or "").strip()
    match = re.fullmatch(
        r"(?:CAM|UPLOAD)_(\d+)_([A-Za-z0-9][A-Za-z0-9_.-]{0,127})",
        candidate,
    )
    if not match:
        return None
    try:
        owner_id = int(match.group(1))
    except (TypeError, ValueError):
        return None
    return owner_id if owner_id > 0 else None


def _resolve_alert_persistence_context(
    db: Session,
    cam_id: str,
    source_type: str | None = None,
):
    """Return (camera, user_id, resolution_source) for durable alert writes.

    Resolution order:
      1. Exact Camera.cam_id row (authoritative path).
      2. Safe runtime-owner fallback only when all of the following are true:
         - fallback is enabled;
         - the ID is currently known by the internal runtime;
         - the ID matches INTEL-I's generated CAM_/UPLOAD_ format;
         - the embedded owner is an active database user;
         - for UPLOAD_ IDs, the uploaded-video ownership row also exists.

    The fallback deliberately does NOT auto-create a Camera row because source
    credentials/connector metadata must never be reconstructed from an alert.
    """
    normalized_cam_id = str(cam_id or "").strip()
    normalized_source_type = str(source_type or "").strip().lower()

    if not normalized_cam_id:
        logger.error("[ALERT-PERSIST] empty camera id; persistence denied")
        return None, None, "invalid_camera_id"

    camera = (
        db.query(Camera)
        .filter(Camera.cam_id == normalized_cam_id)
        .first()
    )

    if camera is not None and getattr(camera, "user_id", None):
        return camera, int(camera.user_id), "database"

    if not ALERT_RUNTIME_OWNER_FALLBACK_ENABLED:
        logger.error(
            "[ALERT-PERSIST] camera row missing and runtime fallback disabled | "
            "cam_id=%s source_type=%s",
            normalized_cam_id,
            normalized_source_type or "unknown",
        )
        return None, None, "camera_missing"

    with state_lock:
        runtime_known = bool(
            normalized_cam_id in cameraIDS
            or normalized_cam_id in cameraWorkers
            or normalized_cam_id in cameraConnectionStates
            or normalized_cam_id in uploadCamToStream
        )

    if not runtime_known:
        logger.error(
            "[ALERT-PERSIST] camera row missing and ID is not a trusted runtime source | "
            "cam_id=%s source_type=%s",
            normalized_cam_id,
            normalized_source_type or "unknown",
        )
        return None, None, "untrusted_runtime_id"

    owner_id = _runtime_owner_from_backend_camera_id(normalized_cam_id)
    if owner_id is None:
        logger.error(
            "[ALERT-PERSIST] trusted runtime source has non-canonical ID; "
            "owner cannot be proven | cam_id=%s",
            normalized_cam_id,
        )
        return None, None, "owner_unresolved"

    owner = (
        db.query(User)
        .filter(
            User.id == owner_id,
            User.is_active.is_(True),
        )
        .first()
    )
    if owner is None:
        logger.error(
            "[ALERT-PERSIST] runtime camera owner is missing/inactive | "
            "cam_id=%s owner_id=%s",
            normalized_cam_id,
            owner_id,
        )
        return None, None, "owner_inactive"

    # UPLOAD_ ownership can be independently verified against the upload record.
    if normalized_cam_id.startswith("UPLOAD_") or normalized_source_type == "upload":
        try:
            uploaded = get_uploaded_video_by_cam(
                db=db,
                cam_id=normalized_cam_id,
                user_id=owner_id,
            )
        except Exception:
            logger.exception(
                "[ALERT-PERSIST] upload ownership verification failed | cam_id=%s",
                normalized_cam_id,
            )
            return None, None, "upload_owner_verification_failed"

        if uploaded is None:
            logger.error(
                "[ALERT-PERSIST] upload row missing; owner fallback denied | "
                "cam_id=%s owner_id=%s",
                normalized_cam_id,
                owner_id,
            )
            return None, None, "upload_row_missing"

    logger.error(
        "[ALERT-PERSIST] Camera row missing; using verified runtime-owner fallback | "
        "cam_id=%s owner_id=%s source_type=%s",
        normalized_cam_id,
        owner_id,
        normalized_source_type or "unknown",
    )
    return None, owner_id, "verified_runtime_fallback"


def _valid_global_vehicle_id(value: str) -> str:
    """Validate the public correlation identifier format."""
    candidate = str(value or "").strip().upper()
    if not candidate or len(candidate) > 32:
        raise HTTPException(status_code=400, detail="Invalid vehicle ID")
    import re
    if not re.fullmatch(r"GV-[A-F0-9]{8,32}", candidate):
        raise HTTPException(status_code=400, detail="Invalid vehicle ID")
    return candidate


def _safe_coordinate(value, minimum, maximum):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number < minimum or number > maximum:
        return None
    return number


def _cache_camera_correlation_meta(camera):
    """Cache only non-secret camera metadata used by GIS."""
    if camera is None:
        return
    cam_id = str(getattr(camera, "cam_id", "") or "").strip()
    if not cam_id:
        return
    cameraCorrelationMeta[cam_id] = {
        "camera_name": str(getattr(camera, "camera_name", "") or cam_id)[:200],
        "latitude": _safe_coordinate(getattr(camera, "latitude", None), -90, 90),
        "longitude": _safe_coordinate(getattr(camera, "longitude", None), -180, 180),
        "heading": _safe_coordinate(getattr(camera, "heading", None), 0, 360),
        "fov": _safe_coordinate(getattr(camera, "fov", None), 0, 360),
        "gis_calibration": getattr(camera, "gis_calibration", None),
        "user_id": getattr(camera, "user_id", None),
    }


def _record_correlation_gis_observation(vehicle_data, correlation):
    """
    Store sanitized GIS telemetry and emit a throttled live vehicle-position
    event. A calibrated fixed camera produces an observed geographic position;
    without calibration we retain the camera coordinate as a clearly labelled
    CAMERA_ANCHOR rather than pretending it is the vehicle's exact position.
    """
    if not isinstance(vehicle_data, dict) or not isinstance(correlation, dict):
        return

    global_id = str(
        correlation.get("global_vehicle_id") or ""
    ).strip().upper()
    camera_id = str(
        vehicle_data.get("camera_id") or ""
    ).strip()
    track_id = vehicle_data.get("track_id")

    if not global_id or not camera_id or track_id is None:
        return

    if not re.fullmatch(r"GV-[A-F0-9]{8,32}", global_id):
        return

    now = time.time()
    meta = cameraCorrelationMeta.get(camera_id, {})
    camera_lat = _safe_coordinate(meta.get("latitude"), -90, 90)
    camera_lon = _safe_coordinate(meta.get("longitude"), -180, 180)

    # Vehicle contact point: bottom-centre of the bounding box.
    bbox = vehicle_data.get("bbox")
    pixel_x = pixel_y = None
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            if all(np.isfinite(v) for v in (x1, y1, x2, y2)):
                pixel_x = (x1 + x2) / 2.0
                pixel_y = y2
        except (TypeError, ValueError):
            pass

    latitude = camera_lat
    longitude = camera_lon
    position_type = "CAMERA_ANCHOR"
    calibration = meta.get("gis_calibration")

    if (
        pixel_x is not None
        and pixel_y is not None
        and isinstance(calibration, dict)
    ):
        projected = project_pixel_to_geo(
            pixel_x,
            pixel_y,
            calibration=calibration,
        )
        if projected is not None:
            latitude, longitude = projected
            position_type = "OBSERVED_CALIBRATED"

    similarity = _safe_float(
        correlation.get("similarity", 0.0), 0.0, 0.0, 1.0
    )
    correlation_score = _safe_float(
        correlation.get("correlation_score", similarity),
        similarity,
        0.0,
        1.0,
    )

    observation = {
        "timestamp": now,
        "camera_id": camera_id,
        "camera_name": str(meta.get("camera_name") or camera_id)[:200],
        "latitude": latitude,
        "longitude": longitude,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "position_type": position_type,
        "track_id": str(track_id)[:64],
        "vehicle_type": str(
            vehicle_data.get("vehicle_type") or "vehicle"
        )[:64],
        "similarity": similarity,
        "correlation_score": correlation_score,
        "supporting_frames": max(
            0,
            min(
                1000,
                int(correlation.get("supporting_frames", 0) or 0),
            ),
        ),
        "consistency": _safe_float(
            correlation.get("consistency", 0.0),
            0.0,
            0.0,
            1.0,
        ),
        "strong_match": bool(correlation.get("strong_match", False)),
        "match_type": str(
            correlation.get("match_type") or "unknown"
        )[:64],
        "correlation_evidence": correlation.get("correlation_evidence")
        if isinstance(correlation.get("correlation_evidence"), dict)
        else {},
        "anpr_score": _safe_float(correlation.get("anpr_score"), 0.0, 0.0, 1.0),
        "reid_score": _safe_float(correlation.get("reid_score"), 0.0, 0.0, 1.0),
        "metadata_score": _safe_float(correlation.get("metadata_score"), 0.0, 0.0, 1.0),
    }

    user_id = meta.get("user_id")

    with state_lock:
        state = correlationGisState.get(global_id)
        if state is None:
            if len(correlationGisState) >= MAX_CORRELATION_GIS_VEHICLES:
                oldest_id = min(
                    correlationGisState,
                    key=lambda key: correlationGisState[key].get(
                        "last_seen", 0.0
                    ),
                )
                correlationGisState.pop(oldest_id, None)

            state = {
                "global_vehicle_id": global_id,
                "first_seen": now,
                "last_seen": now,
                "last_camera_id": camera_id,
                "last_track_id": str(track_id)[:64],
                "vehicle_type": observation["vehicle_type"],
                "last_similarity": similarity,
                "last_correlation_score": correlation_score,
                "strong_match": bool(observation["strong_match"]),
                "observations": [],
            }
            correlationGisState[global_id] = state

        observations = state["observations"]

        # Persisted journey history is sampled; live position is emitted
        # separately at a higher rate for smooth map movement.
        append_history = True
        if observations:
            last = observations[-1]
            append_history = not (
                last.get("camera_id") == camera_id
                and last.get("track_id") == observation["track_id"]
                and now - float(last.get("timestamp", 0.0))
                < CORRELATION_GIS_DEDUP_SECONDS
            )

        if append_history:
            observations.append(observation)
            if len(observations) > MAX_CORRELATION_GIS_OBSERVATIONS:
                del observations[:-MAX_CORRELATION_GIS_OBSERVATIONS]

        state["last_seen"] = now
        state["last_camera_id"] = camera_id
        state["last_track_id"] = observation["track_id"]
        state["vehicle_type"] = observation["vehicle_type"]
        state["last_similarity"] = similarity
        state["last_correlation_score"] = correlation_score
        state["strong_match"] = bool(
            state.get("strong_match") or observation["strong_match"]
        )

        # Derive speed/heading only from calibrated geographic observations.
        # Camera-anchor points are not vehicle positions and must not be used
        # for physical speed calculations.
        if position_type == "OBSERVED_CALIBRATED" and len(observations) >= 2:
            previous = observations[-2]
            try:
                if previous.get("position_type") == "OBSERVED_CALIBRATED":
                    distance_m = haversine_distance_meters(
                        previous.get("latitude"), previous.get("longitude"),
                        latitude, longitude,
                    )
                    elapsed_s = float(now) - float(previous.get("timestamp", now))
                    if distance_m is not None and elapsed_s > 0.05:
                        derived_speed = (distance_m / elapsed_s) * 3.6
                        if 0.0 <= derived_speed <= 300.0:
                            vehicle_data["speed_kmh"] = derived_speed
                    bearing = initial_bearing_degrees(
                        previous.get("latitude"), previous.get("longitude"),
                        latitude, longitude,
                    )
                    if bearing is not None:
                        vehicle_data["heading"] = bearing
            except Exception:
                pass

    # High-rate live delivery is deliberately separated from route history.
    # This avoids flooding PostgreSQL/history APIs while keeping the map live.
    last_emit = liveVehiclePositionLastEmit.get(global_id, 0.0)
    if (
        user_id
        and now - last_emit >= LIVE_VEHICLE_POSITION_INTERVAL_SECONDS
        and latitude is not None
        and longitude is not None
    ):
        liveVehiclePositionLastEmit[global_id] = now
        payload = {
            "type": "vehicle_position",
            "event": "VEHICLE_POSITION_UPDATED",
            "user_id": int(user_id),
            "vehicle_id": global_id,
            "global_vehicle_id": global_id,
            "camera_id": camera_id,
            "track_id": str(track_id)[:64],
            "latitude": latitude,
            "longitude": longitude,
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "position_type": position_type,
            "vehicle_type": observation["vehicle_type"],
            "speed_kmh": _safe_float(
                vehicle_data.get("speed_kmh"),
                0.0,
                0.0,
                300.0,
            ),
            "heading": _safe_float(
                vehicle_data.get("heading"),
                meta.get("heading") or 0.0,
                0.0,
                360.0,
            ),
            "confidence": _safe_float(
                correlation.get("correlation_score", 0.0),
                0.0,
                0.0,
                1.0,
            ),
            "timestamp": now,
        }
        if main_loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    broadcast_vehicle_position(payload),
                    main_loop,
                )
            except Exception:
                logger.debug(
                    "Vehicle GIS WebSocket scheduling failed | vehicle=%s",
                    global_id,
                    exc_info=True,
                )

    return observation


def _sanitize_gis_vehicle(state):
    """Return only dashboard-safe vehicle journey metadata."""
    if not isinstance(state, dict):
        return None
    return {
        "global_vehicle_id": state.get("global_vehicle_id"),
        "first_seen": state.get("first_seen"),
        "last_seen": state.get("last_seen"),
        "last_camera_id": state.get("last_camera_id"),
        "last_track_id": state.get("last_track_id"),
        "vehicle_type": state.get("vehicle_type"),
        "last_similarity": state.get("last_similarity", 0.0),
        "last_correlation_score": state.get("last_correlation_score", 0.0),
        "strong_match": bool(state.get("strong_match", False)),
        "observations": [
            dict(item)
            for item in state.get("observations", [])
            if isinstance(item, dict)
        ],
    }


def _user_camera_ids(db: Session, user_id: int) -> set[str]:
    return {
        str(camera.cam_id)
        for camera in get_cameras(db, user_id)
        if getattr(camera, "cam_id", None)
    }


def _vehicle_belongs_to_user(
    state: dict,
    owned_camera_ids: set[str],
) -> bool:
    if not isinstance(state, dict):
        return False
    return any(
        str(obs.get("camera_id")) in owned_camera_ids
        for obs in state.get("observations", [])
        if isinstance(obs, dict)
    )

def _safe_float(value, default=0.0, minimum=None, maximum=None):
    """Convert untrusted numeric data into a bounded finite float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def _safe_int(value, default=0, minimum=None, maximum=None):
    """Convert untrusted numeric data into a bounded integer."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def _valid_camera_id(value: str) -> str:
    """Validate camera IDs before using them in GIS/correlation lookups."""
    candidate = str(value or "").strip()
    if not candidate or len(candidate) > 128:
        raise HTTPException(status_code=400, detail="Invalid camera ID")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", candidate):
        raise HTTPException(status_code=400, detail="Invalid camera ID")
    return candidate


def _build_gis_route(observations, owned_camera_ids):
    """
    Build a sanitized GIS route.

    No embeddings, source URLs, crops, or raw ANPR values are included.
    Duplicate consecutive camera observations are retained only when the
    timestamp changes sufficiently to represent a meaningful transition.
    """
    route = []

    for obs in observations:
        if not isinstance(obs, dict):
            continue

        camera_id = str(obs.get("camera_id") or "").strip()

        if camera_id not in owned_camera_ids:
            continue

        latitude = _safe_coordinate(
            obs.get("latitude"),
            -90,
            90,
        )
        longitude = _safe_coordinate(
            obs.get("longitude"),
            -180,
            180,
        )

        # Cameras without coordinates remain valid correlation observations,
        # but cannot be rendered as GIS route points.
        if latitude is None or longitude is None:
            continue

        point = {
            "camera_id": camera_id,
            "camera_name": str(
                obs.get("camera_name") or camera_id
            )[:200],
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": _safe_float(
                obs.get("timestamp"),
                0.0,
                0.0,
            ),
            "track_id": str(
                obs.get("track_id") or ""
            )[:64],
            "vehicle_type": str(
                obs.get("vehicle_type") or "vehicle"
            )[:64],
            "similarity": _safe_float(
                obs.get("similarity"),
                0.0,
                0.0,
                1.0,
            ),
            "correlation_score": _safe_float(
                obs.get("correlation_score"),
                0.0,
                0.0,
                1.0,
            ),
            "supporting_frames": _safe_int(
                obs.get("supporting_frames"),
                0,
                0,
                1000,
            ),
            "consistency": _safe_float(
                obs.get("consistency"),
                0.0,
                0.0,
                1.0,
            ),
            "strong_match": bool(
                obs.get("strong_match", False)
            ),
            "match_type": str(
                obs.get("match_type") or "unknown"
            )[:64],
        }

        # Avoid emitting repeated points from the same camera in a single
        # frame window. This makes frontend GIS rendering stable.
        if route:
            previous = route[-1]
            if (
                previous["camera_id"] == point["camera_id"]
                and abs(
                    previous["timestamp"]
                    - point["timestamp"]
                ) < CORRELATION_GIS_DEDUP_SECONDS
            ):
                route[-1] = point
                continue

        route.append(point)

    route.sort(
        key=lambda item: item["timestamp"]
    )
    return route


def _haversine_km(lat1, lon1, lat2, lon2):
    """Return great-circle distance in kilometres."""
    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except (TypeError, ValueError):
        return None

    values = (lat1, lon1, lat2, lon2)
    if not all(np.isfinite(v) for v in values):
        return None

    if not (
        -90 <= lat1 <= 90
        and -90 <= lat2 <= 90
        and -180 <= lon1 <= 180
        and -180 <= lon2 <= 180
    ):
        return None

    radius_km = 6371.0088

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return radius_km * 2 * math.atan2(
        math.sqrt(max(0.0, a)),
        math.sqrt(max(0.0, 1.0 - a)),
    )


def _build_gis_segments(route):
    """
    Convert consecutive GIS points into frontend-friendly transitions.
    Segments are descriptive only; they are not road-network routes.
    """
    segments = []

    for previous, current in zip(
        route,
        route[1:],
    ):
        if previous["camera_id"] == current["camera_id"]:
            continue

        distance_km = _haversine_km(
            previous["latitude"],
            previous["longitude"],
            current["latitude"],
            current["longitude"],
        )

        elapsed_seconds = max(
            0.0,
            current["timestamp"] - previous["timestamp"],
        )

        segments.append(
            {
                "from_camera_id": previous["camera_id"],
                "from_camera_name": previous["camera_name"],
                "to_camera_id": current["camera_id"],
                "to_camera_name": current["camera_name"],
                "from_latitude": previous["latitude"],
                "from_longitude": previous["longitude"],
                "to_latitude": current["latitude"],
                "to_longitude": current["longitude"],
                "from_timestamp": previous["timestamp"],
                "to_timestamp": current["timestamp"],
                "elapsed_seconds": elapsed_seconds,
                "distance_km": (
                    round(distance_km, 3)
                    if distance_km is not None
                    else None
                ),
                "similarity": current["similarity"],
                "correlation_score": current["correlation_score"],
                "strong_match": current["strong_match"],
                "match_type": current["match_type"],
            }
        )

    return segments


def _dispatch_worker_realtime_event(message: dict) -> None:
    """Bridge worker Redis Pub/Sub events onto this API process's sockets."""
    if not DISTRIBUTED_CAMERA_WORKERS_ENABLED or main_loop is None:
        return
    if not isinstance(message, dict):
        return
    kind = str(message.get("kind") or "").strip().lower()
    payload = message.get("payload")
    if not isinstance(payload, dict):
        return
    try:
        if kind == "alert":
            future = asyncio.run_coroutine_threadsafe(sendAlert(payload), main_loop)
        elif kind == "position":
            future = asyncio.run_coroutine_threadsafe(broadcast_vehicle_position(payload), main_loop)
        else:
            return

        def _done(done_future):
            try:
                done_future.result()
            except Exception:
                logger.exception("Worker realtime event delivery failed | kind=%s", kind)

        future.add_done_callback(_done)
    except Exception:
        logger.exception("Unable to schedule worker realtime event | kind=%s", kind)


async def _camera_health_watchdog():
    """Mark stale active streams degraded without killing other cameras."""
    while True:
        try:
            await asyncio.sleep(2.0)
            with state_lock:
                active_ids = {cam_id for cam_id, worker in cameraWorkers.items() if worker and not worker.done()}
            for cam_id in camera_health_registry.stale_cameras(active_ids):
                try:
                    _set_camera_connection_state(cam_id, CameraState.DEGRADED)
                    camera_health_registry.set_state(cam_id, CameraState.DEGRADED, error_code="STALE_HEARTBEAT", error_message="No decoded frame within health timeout")
                except Exception:
                    logger.debug("Camera health watchdog update failed | cam_id=%s", cam_id, exc_info=True)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Camera health watchdog iteration failed")


@app.on_event("startup")
async def startup_event():
    # OpenSearch is an optional production search/RAG dependency. When marked
    # required, startup fails closed instead of advertising a false READY state.
    try:
        from services.openSearchIntelligence import opensearch_intelligence
        if opensearch_intelligence.enabled:
            opensearch_intelligence.ensure_indices()
            os_health = opensearch_intelligence.health()
            if opensearch_intelligence.required and os_health.get("status") != "READY":
                raise RuntimeError("required OpenSearch is not ready")
            logger.info("OpenSearch intelligence status | %s", os_health)
    except Exception:
        logger.exception("OpenSearch startup readiness check failed")
        try:
            from services.openSearchIntelligence import opensearch_intelligence
            if opensearch_intelligence.required:
                raise
        except ImportError:
            pass
    global main_loop, SYSTEM_INITIALIZATION_COMPLETE, SYSTEM_INITIALIZATION_STARTED_AT, SYSTEM_INITIALIZATION_COMPLETED_AT

    SYSTEM_INITIALIZATION_COMPLETE = False
    SYSTEM_INITIALIZATION_STARTED_AT = time.time()
    main_loop = asyncio.get_running_loop()
    if DISTRIBUTED_CAMERA_WORKERS_ENABLED:
        start_realtime_subscriber(_dispatch_worker_realtime_event)

    # Hard AI model readiness: configured -> present -> checksum -> loaded -> inference.
    from services.runtimeModelReadiness import verify_models
    model_state = verify_models(force=True)
    model_required = os.getenv(
        "MODEL_READINESS_REQUIRED", "true" if IS_PROD else "false"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if model_required and model_state.get("status") != "READY":
        raise RuntimeError(f"Required AI models are not ready: {model_state}")
    logger.info("AI model runtime readiness | %s", model_state)

    # Gemma/Ollama is advisory unless explicitly required, but when enabled the
    # exact configured model must exist in /api/tags before READY is advertised.
    ollama_state = ollama_plate_verifier.probe_model()
    ollama_required = os.getenv(
        "OLLAMA_VISION_REQUIRED", "true" if (IS_PROD and ollama_plate_verifier.enabled) else "false"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if ollama_required and not ollama_state.get("available"):
        raise RuntimeError(f"Required Ollama vision model is unavailable: {ollama_state}")
    logger.info("Ollama/Gemma deployment verification | %s", ollama_state)

    init_outbox()

    redis_ok = False
    kafka_ok = False
    object_storage_ok = False

    try:
        redis_ok = bool(redis_ping())
    except Exception:
        logger.exception("Redis startup health check failed")

    redis_health.set(1 if redis_ok else 0)

    try:
        kafka_ok = bool(init_kafka())
    except Exception:
        logger.exception("Kafka startup initialization failed")

    kafka_health.set(1 if kafka_ok else 0)

    try:
        storage = get_object_storage()
        # The infrastructure init container normally creates the bucket.
        # ensure_ready(create_bucket=True) also makes startup idempotent when
        # INTEL-I is deployed without that helper container.
        storage.ensure_ready(create_bucket=True)
        object_storage_ok = True
    except Exception:
        logger.exception("Object storage startup readiness check failed")

    if kafka_ok:
        try:
            start_outbox_worker()
        except Exception:
            logger.exception("Failed to start outbox worker")
            kafka_health.set(0)
        try:
            from events.savedAlertConsumer import start_saved_alert_consumer
            start_saved_alert_consumer(loop=main_loop, websocket_sender=sendAlert)
        except Exception:
            logger.exception("Failed to start saved-alert Kafka realtime consumer")
            if os.getenv("KAFKA_SAVED_ALERT_WS_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}:
                kafka_health.set(0)
    else:
        logger.warning(
            "Kafka unavailable. Outbox worker not started."
        )

    # Readiness is stricter than process liveness. The platform is READY only
    # when the database, Redis and event bus are all available.
    db_ok = False
    try:
        with SessionLocal() as readiness_db:
            readiness_db.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        logger.exception("Database startup readiness check failed")

    logger.info(
        "Infrastructure health | database=%s redis=%s kafka=%s object_storage=%s",
        db_ok, redis_ok, kafka_ok, object_storage_ok,
    )

    logger.info(
        "Persistent AI intelligence models registered | PostgreSQL=%s",
        bool(engine),
    )

    if ADVANCED_INTELLIGENCE_ENABLED:
        logger.info(
            "Advanced intelligence enabled | edsr=%s model=%s device=%s tamper=%s frozen=%s risk=%s incident=%s",
            EDSR_ENABLED, EDSR_MODEL_PATH, EDSR_DEVICE, CAMERA_TAMPER_ENABLED,
            FROZEN_FRAME_ENABLED, RISK_SCORING_ENABLED, INCIDENT_CORRELATION_ENABLED,
        )
        if EDSR_ENABLED:
            logger.info(
                "EDSR status | %s",
                advanced_intelligence.edsr.load(),
            )
        if DARKIR_ENABLED:
            logger.info(
                "DarkIR status | %s",
                advanced_intelligence.darkir_status(),
            )

    if ANPR_TERMINAL_LOGGING_ENABLED:
        try:
            logger.info("Ollama/Gemma ANPR status | %s", ollama_plate_verifier.probe_model())
        except Exception:
            logger.exception("Ollama/Gemma ANPR startup health check failed")

    ai_scheduler.start(_scheduled_ai_handler)
    _sync_ai_scheduler_metrics()
    start_integration_scheduler()

    SYSTEM_INITIALIZATION_COMPLETED_AT = time.time()
    kafka_required = os.getenv("KAFKA_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    object_storage_required = os.getenv(
        "OBJECT_STORAGE_REQUIRED",
        "true" if IS_PROD else "false",
    ).strip().lower() in {"1", "true", "yes", "on"}

    SYSTEM_INITIALIZATION_COMPLETE = bool(
        db_ok
        and redis_ok
        and (kafka_ok or not kafka_required)
        and (object_storage_ok or not object_storage_required)
        and (model_state.get("status") == "READY" or not model_required)
        and (bool(ollama_state.get("available")) or not ollama_required)
    )
    asyncio.create_task(_camera_health_watchdog())
    _record_system_event(
        event_type="SYSTEM_READY" if SYSTEM_INITIALIZATION_COMPLETE else "SYSTEM_DEGRADED",
        component="startup",
        severity="INFO" if SYSTEM_INITIALIZATION_COMPLETE else "WARNING",
        message="INTEL-I initialization completed",
        metadata={
            "database": db_ok,
            "redis": redis_ok,
            "kafka": kafka_ok,
            "object_storage": object_storage_ok,
        },
    )
    logger.info(
        "INTEL-I initialization complete | duration_seconds=%.2f",
        SYSTEM_INITIALIZATION_COMPLETED_AT - SYSTEM_INITIALIZATION_STARTED_AT,
    )


@app.on_event("shutdown")
async def shutdown_event():
    # Distributed live-camera workers are independent processes and must not be
    # terminated when an API pod restarts. Local upload/legacy workers still
    # belong to this process and are drained normally.
    for camID in list(cameraStopEvents.keys()):
        stop_camera_worker(camID)

    stop_realtime_subscriber()
    stop_outbox_worker()
    try:
        from events.savedAlertConsumer import stop_saved_alert_consumer
        stop_saved_alert_consumer()
    except Exception:
        logger.exception("Failed to stop saved-alert consumer cleanly")
    close_kafka()
    ai_scheduler.stop(wait=True)
    ollama_plate_verifier.shutdown()
    investigation_summary_worker.shutdown()
    await stop_integration_scheduler()

    executor.shutdown(
        wait=True,
        cancel_futures=True,
    )
    
clients_by_user: dict[int, set[WebSocket]] = {}
MAX_WEBSOCKETS_PER_USER = max(
    1,
    min(20, int(os.getenv("MAX_WEBSOCKETS_PER_USER", "4"))),
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    user_id = None
    connection_id = secrets.token_urlsafe(12)

    try:
        user_id = await get_current_user_ws(websocket)

        async with client_lock:
            connection_limit_reached = (
                len(clients_by_user.get(user_id, set()))
                >= MAX_WEBSOCKETS_PER_USER
            )
        if connection_limit_reached:
            await websocket.close(code=1013, reason="Too many active sessions")
            return

        await websocket.accept()

        async with client_lock:
            clients_by_user.setdefault(user_id, set()).add(websocket)
            active_ws_clients.set(
                sum(len(items) for items in clients_by_user.values())
            )

        logger.info(
            "WebSocket connected | user_id=%s active_clients=%s",
            user_id,
            sum(len(items) for items in clients_by_user.values()),
        )

        await websocket.send_json(
            {
                "type": "connected",
                "connection_id": connection_id,
                "server_time": datetime.now(timezone.utc).isoformat(),
                "heartbeat_seconds": 25,
            }
        )

        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=25.0,
                )
            except asyncio.TimeoutError:
                await websocket.send_json(
                    {
                        "type": "heartbeat",
                        "server_time": datetime.now(timezone.utc).isoformat(),
                    }
                )
                continue

            normalized = str(message or "").strip().lower()
            if normalized in {"pong", '{"type":"pong"}'}:
                continue
            if normalized in {"ping", '{"type":"ping"}'}:
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected | user_id=%s", user_id)

    except WebSocketException:
        logger.warning("Unauthorized WebSocket rejected")

    except Exception:
        logger.exception("WebSocket unexpected error | user_id=%s", user_id)

        try:
            await websocket.close(code=1011)
        except RuntimeError:
            pass


    finally:
        async with client_lock:
            if user_id is not None and user_id in clients_by_user:
                clients_by_user[user_id].discard(websocket)

                if not clients_by_user[user_id]:
                    del clients_by_user[user_id]

            active_ws_clients.set(
                sum(len(items) for items in clients_by_user.values())
            )


async def broadcast_vehicle_position(payload: dict):
    """Broadcast live vehicle-position telemetry to authenticated dashboard clients."""
    if not isinstance(payload, dict):
        return

    user_id = payload.get("user_id")
    if not user_id:
        return

    message = json.dumps(payload, default=str)
    disconnected = []

    async with client_lock:
        current_clients = list(
            clients_by_user.get(int(user_id), set())
        )

    for client in current_clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.append(client)

    if disconnected:
        async with client_lock:
            user_clients = clients_by_user.get(int(user_id), set())
            for client in disconnected:
                user_clients.discard(client)
            if not user_clients:
                clients_by_user.pop(int(user_id), None)

        active_ws_clients.set(
            sum(len(items) for items in clients_by_user.values())
        )


def _correlate_and_persist_person(*, cam_id, track_id, bbox, confidence, appearance, frame_timestamp):
    """Attach GlobalPersonID, persist evidence, and emit calibrated live GIS."""
    meta = cameraCorrelationMeta.get(str(cam_id), {})
    user_id = meta.get("user_id")
    if not user_id or not isinstance(appearance, dict) or not appearance.get("valid"):
        return None
    cache = personTopologyCache.get(int(user_id))
    now = time.time()
    if not cache or now - cache[0] > 30.0:
        db = SessionLocal()
        try:
            owned = {row.cam_id for row in db.query(Camera).filter(Camera.user_id == int(user_id)).all()}
            transitions = {
                (row.source_camera_id, row.destination_camera_id)
                for row in db.query(CameraConnection).filter(
                    CameraConnection.active.is_(True),
                    CameraConnection.source_camera_id.in_(owned),
                    CameraConnection.destination_camera_id.in_(owned),
                ).all()
            } if owned else set()
            personTopologyCache[int(user_id)] = (now, transitions)
        finally:
            db.close()
    else:
        transitions = cache[1]
    result = person_correlation_engine.correlate(
        user_id=int(user_id), camera_id=str(cam_id), local_track_id=str(track_id),
        appearance=appearance, timestamp=float(frame_timestamp), allowed_transitions=transitions,
    )
    observed_at = datetime.fromtimestamp(float(frame_timestamp), tz=timezone.utc).replace(tzinfo=None)
    person_correlation_engine.persist(
        result=result, user_id=int(user_id), camera_id=str(cam_id), local_track_id=str(track_id),
        confidence=_safe_float(confidence, 0.0, 0.0, 1.0), timestamp=observed_at,
        bbox=bbox, pose=None, appearance=appearance,
    )
    x1, y1, x2, y2 = [float(v) for v in bbox]
    pixel_x, pixel_y = (x1 + x2) / 2.0, y2
    latitude = _safe_coordinate(meta.get("latitude"), -90, 90)
    longitude = _safe_coordinate(meta.get("longitude"), -180, 180)
    position_type = "CAMERA_ANCHOR"
    calibration = meta.get("gis_calibration")
    if isinstance(calibration, dict):
        projected = project_pixel_to_geo(pixel_x, pixel_y, calibration=calibration)
        if projected is not None:
            latitude, longitude = projected
            position_type = "OBSERVED_CALIBRATED"
    gid = result["global_person_id"]
    state = personGisState.setdefault(gid, {"global_person_id": gid, "observations": []})
    observation = {"timestamp": float(frame_timestamp), "camera_id": str(cam_id), "track_id": str(track_id), "latitude": latitude, "longitude": longitude, "pixel_x": pixel_x, "pixel_y": pixel_y, "position_type": position_type, "confidence": result.get("score", 0.0)}
    state["observations"].append(observation)
    del state["observations"][:-1000]
    state.update({"last_seen": float(frame_timestamp), "last_camera_id": str(cam_id), "last_track_id": str(track_id), "confidence_status": result.get("confidence_status")})
    now = time.time()
    if latitude is not None and longitude is not None and now - livePersonPositionLastEmit.get(gid, 0.0) >= LIVE_VEHICLE_POSITION_INTERVAL_SECONDS:
        livePersonPositionLastEmit[gid] = now
        payload = {"type": "person_position", "event": "PERSON_POSITION_UPDATED", "user_id": int(user_id), "global_person_id": gid, "person_id": gid, "camera_id": str(cam_id), "track_id": str(track_id), "latitude": latitude, "longitude": longitude, "pixel_x": pixel_x, "pixel_y": pixel_y, "position_type": position_type, "confidence": result.get("score", 0.0), "confidence_status": result.get("confidence_status"), "timestamp": now}
        if main_loop is not None:
            asyncio.run_coroutine_threadsafe(broadcast_vehicle_position(payload), main_loop)
    return result


def normalize_alert_severity(value):
    """Normalize legacy alert levels into the UI severity ladder.

    UI contract:
      HIGH   -> red
      MEDIUM -> orange
      LOW    -> yellow
      INFO   -> blue

    Existing database/rule values such as CRITICAL and WARNING are kept
    intact in their original fields; ``severity`` is the stable presentation
    field consumed by the frontend.
    """
    raw = str(value or "LOW").strip().upper()
    if raw in {"CRITICAL", "HIGH"}:
        return "HIGH"
    if raw in {"WARNING", "MEDIUM"}:
        return "MEDIUM"
    if raw == "INFO":
        return "INFO"
    return "LOW"


def _iso_utc(value):
    """Serialize database/PTS datetimes as an unambiguous UTC ISO-8601 value."""
    if not isinstance(value, datetime):
        return None
    aware = (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )
    return aware.isoformat(timespec="milliseconds").replace("+00:00", "Z")


async def sendAlert(alert: dict):
    """Broadcast one normalized alert event over the authenticated WebSocket.

    This function is intentionally the single delivery point for live alerts.
    Camera workers may call it from worker threads through
    ``asyncio.run_coroutine_threadsafe``; the actual send always executes on
    FastAPI's main event loop. The database remains the durable source of
    truth, while WebSocket is the real-time delivery channel.
    """
    if not isinstance(alert, dict):
        logger.warning("WebSocket alert skipped because payload is not a dict")
        return

    user_id = alert.get("user_id")

    if not user_id:
        logger.warning("WebSocket alert skipped because user_id missing")
        return

    payload = dict(alert)

    # One stable protocol for every live alert producer.
    payload["type"] = "alert"
    payload.setdefault("event", "ALERT_CREATED")
    payload.setdefault("id", payload.get("alert_id"))
    payload.setdefault("alert_id", payload.get("id"))
    payload.setdefault("cam_id", payload.get("camera_id"))
    payload.setdefault("camera_id", payload.get("cam_id"))
    payload.setdefault("alert_rule", payload.get("rule"))
    payload.setdefault("rule", payload.get("alert_rule"))
    payload.setdefault("level", payload.get("alert_level") or payload.get("priority"))
    payload.setdefault("alert_level", payload.get("level"))
    payload["severity"] = normalize_alert_severity(
        payload.get("severity")
        or payload.get("level")
        or payload.get("alert_level")
        or payload.get("priority")
    )
    payload.setdefault("source_type", payload.get("source") or "unknown")
    payload["source_type"] = str(payload.get("source_type") or "unknown").strip().lower()
    payload.setdefault("has_snapshot", bool(payload.get("snapshot_id")))
    payload.setdefault("snapshot_saved", bool(payload.get("snapshot_id")))

    if payload.get("snapshot_id") and not payload.get("snapshot_path"):
        payload["snapshot_path"] = f'/snapshot/{payload["snapshot_id"]}'

    disconnected = []

    async with client_lock:
        current_clients = list(clients_by_user.get(int(user_id), set()))

    if not current_clients:
        logger.warning(
            "WebSocket alert had no connected client | user_id=%s alert_id=%s event=%s",
            user_id,
            payload.get("alert_id"),
            payload.get("event"),
        )
        return

    logger.info(
        "WebSocket broadcasting alert | user_id=%s alert_id=%s clients=%s source=%s severity=%s",
        user_id,
        payload.get("alert_id"),
        len(current_clients),
        payload.get("source_type"),
        payload.get("severity"),
    )

    for client in current_clients:
        try:
            await client.send_text(json.dumps(payload, default=str))
        except Exception:
            logger.exception("WebSocket send failed | user_id=%s", user_id)
            disconnected.append(client)

    if disconnected:
        async with client_lock:
            user_clients = clients_by_user.get(int(user_id), set())

            for client in disconnected:
                user_clients.discard(client)

            if not user_clients and int(user_id) in clients_by_user:
                del clients_by_user[int(user_id)]


def create_stream_session(user_id: int, target_id: str, stream_type: str):
    sid = str(uuid.uuid4())

    set_stream_session(
        sid=sid,
        user_id=user_id,
        target_id=target_id,
        stream_type=stream_type,
        ttl=STREAM_SESSION_TTL,
    )

    return sid


def validate_stream_session(sid: str, target_id: str, stream_type: str):
    session = get_stream_session(sid)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired stream session")
    if str(session.get("target_id")) != str(target_id):
        raise HTTPException(status_code=401, detail="Invalid stream target")
    if str(session.get("stream_type")) != str(stream_type):
        raise HTTPException(status_code=401, detail="Invalid stream type")
    return session

def center_of_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def valid_person_box(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    area = width * height
    aspect = height / width
    if area < MIN_PERSON_BOX_AREA:
        return False
    if aspect > MAX_BOX_ASPECT_RATIO:
        return False
    if x2 < 5 or y2 < 5:
        return False
    if x1 > frame_w - 5 or y1 > frame_h - 5:
        return False
    return True

def remember_alert_box(camID, track_id, rule, box, level, confidence, ts):
    if track_id is None or box is None or rule is None:
        return
    activeAlertBoxes.setdefault(camID, {})
    real_track_id = str(track_id)
    key = real_track_id
    old = activeAlertBoxes[camID].get(key)
    if old:
        old_priority = LEVEL_PRIORITY.get(old.get("level", "LOW"), 0)
        new_priority = LEVEL_PRIORITY.get(level, 0)
        if new_priority < old_priority:
            return
    activeAlertBoxes[camID][key] = {
        "track_id": real_track_id,
        "rule": rule,
        "box": tuple(map(int, box)),
        "level": level,
        "confidence": float(confidence),
        "last_seen": ts,
    }
    
def refresh_active_alert_box(camID, track_id, box, ts):
    if camID not in activeAlertBoxes:
        return
    real_track_id = str(track_id)
    for key, data in activeAlertBoxes[camID].items():
        if data.get("track_id") == real_track_id:
            data["box"] = tuple(map(int, box))
            data["last_seen"] = ts

def cleanup_alert_boxes(camID, current_track_ids):
    if camID not in activeAlertBoxes:
        return
    current_track_ids = {str(x) for x in current_track_ids}
    for key, data in list(activeAlertBoxes[camID].items()):
        track_id = str(data.get("track_id", ""))
        parts = track_id.split("_")
        still_active = all(part in current_track_ids for part in parts)
        if not still_active:
            del activeAlertBoxes[camID][key]
            forget_snapshot_track(camID, track_id)

def draw_active_alert_boxes(frame, camID):
    if camID not in activeAlertBoxes:
        return frame

    for data in activeAlertBoxes[camID].values():
        x1, y1, x2, y2 = data["box"]

        color = (
            (0, 0, 255)
            if data["level"] == "CRITICAL"
            else (0, 165, 255)
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        cv2.putText(
            frame,
            f"{data['level']}: {data['rule']} {data['confidence']:.2f}",
            (x1, max(30, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

    return frame


def match_pose_to_box(pose_result, person_box):
    if pose_result.boxes is None or pose_result.keypoints is None:
        return None

    pose_boxes = pose_result.boxes.xyxy.cpu().numpy()
    pose_kpts = pose_result.keypoints.data.cpu().numpy()

    best_score = 0.0
    best_kpts = None

    for pbox, kpts in zip(pose_boxes, pose_kpts):
        candidate_box = tuple(map(int, pbox[:4]))
        score = iou(candidate_box, person_box)

        if score > best_score:
            best_score = score
            best_kpts = kpts.tolist()

    if best_score >= 0.35:
        return best_kpts

    return None


def normalize_class_name(name: str) -> str:
    return (
        str(name)
        .lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def get_crime_detections(frame):
    detections = []

    try:
        with model_lock:
            result = crime_model(
                frame,
                conf=CRIME_CONF,
                device=DEVICE,
                verbose=False,
            )[0]

        if result.boxes is None:
            return detections

        names = result.names

        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            raw_name = names[int(cls)]
            name = normalize_class_name(raw_name)

            if name in IGNORED_CRIME_CLASSES:
                continue

            confidence = float(conf)

            if confidence < CRIME_CONF:
                continue

            x1, y1, x2, y2 = map(int, box[:4])

            detections.append(
                {
                    "name": name,
                    "raw_name": raw_name,
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                }
            )

    except Exception as e:
        logger.exception("Suspicious model error")

    return detections


def get_ppe_detections(frame,cam_id=None):
    detections = []

    try:
        with model_lock:
            result = ppe_model(
                frame,
                conf=PPE_CONF,
                device=DEVICE,
                verbose=False,
            )[0]

        if result.boxes is None:
            return detections

        names = result.names

        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            raw_name = names[int(cls)]
            name = normalize_class_name(raw_name)

            if name not in PPE_MASK_CLASSES and name not in PPE_HELMET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box[:4])

            detections.append(
                {
                    "name": name,
                    "raw_name": raw_name,
                    "box": (x1, y1, x2, y2),
                    "confidence": float(conf),
                    "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                }
            )

    except Exception as e:
        logger.exception("PPE model error | cam_id=%s", cam_id)

    return detections


def get_alert_type(rule: str) -> str:
    return str(rule).replace("_", " ").title()


def build_alert_message(rule, camID, source_type, zone_name=None, confidence=None):
    zone_part = f" | Zone: {zone_name}" if source_type == "rtsp" and zone_name else ""
    conf_part = f" | Confidence: {confidence:.2f}" if confidence is not None else ""

    return (
        f"{get_alert_type(rule)} detected before incident escalation | "
        f"Camera: {camID}"
        f"{zone_part}"
        f"{conf_part}"
    )


def _persist_global_vehicle_safe(global_vehicle_id, vehicle_type=None, color=None, make=None, model=None):
    """Persist a global vehicle identity without breaking the live worker."""
    if not global_vehicle_id:
        return None

    try:
        return create_global_vehicle(
            global_vehicle_id=str(global_vehicle_id),
            vehicle_type=(str(vehicle_type) if vehicle_type is not None else None),
            color=str(color)[:50] if color else None,
            make=str(make)[:100] if make else None,
            model=str(model)[:100] if model else None,
        )
    except Exception:
        # The ID may already exist. The persistence service rolls back its
        # transaction; the in-memory correlation engine remains authoritative.
        logger.debug(
            "Global vehicle already persisted or persistence failed | global_id=%s",
            global_vehicle_id,
            exc_info=True,
        )
        return None


def _persist_vehicle_observation_safe(vehicle_data):
    """
    Persist identity observations only at the existing identity-analysis
    cadence. Never write every video frame to PostgreSQL.
    """
    if not isinstance(vehicle_data, dict):
        return None

    try:
        bbox = vehicle_data.get("bbox")
        if bbox is not None:
            bbox = [int(v) for v in bbox]

        from datetime import datetime, timezone

        return save_vehicle_observation(
            camera_id=str(vehicle_data.get("camera_id") or ""),
            local_track_id=(
                str(vehicle_data.get("track_id"))
                if vehicle_data.get("track_id") is not None
                else None
            ),
            global_vehicle_id=(
                str(vehicle_data.get("global_vehicle_id"))
                if vehicle_data.get("global_vehicle_id")
                else None
            ),
            vehicle_type=str(
                vehicle_data.get("vehicle_type") or "vehicle"
            ),
            confidence=_safe_float(
                vehicle_data.get("confidence"),
                0.0,
                0.0,
                1.0,
            ),
            frame_timestamp=(
                vehicle_data.get("frame_timestamp")
                if isinstance(vehicle_data.get("frame_timestamp"), datetime)
                else datetime.fromtimestamp(
                    _safe_float(vehicle_data.get("timestamp"), time.time()),
                    tz=timezone.utc,
                ).replace(tzinfo=None)
            ),
            bbox=bbox,
            model_name="vehicle_detection",
            model_version=VEHICLE_MODEL_VERSION,
            metadata={
                "analytics_timestamp": _safe_float(
                    vehicle_data.get("analytics_timestamp"),
                    0.0,
                    0.0,
                ),
                "source_pts_seconds": vehicle_data.get("source_pts_seconds"),
                "timestamp_source": vehicle_data.get("timestamp_source"),
                "timestamp_quality": vehicle_data.get("timestamp_quality"),
                "correlation_type": vehicle_data.get(
                    "correlation_type"
                ),
                "correlation_similarity": _safe_float(
                    vehicle_data.get("correlation_similarity"),
                    0.0,
                    0.0,
                    1.0,
                ),
                "correlation_score": _safe_float(
                    vehicle_data.get("correlation_score"),
                    0.0,
                    0.0,
                    1.0,
                ),
                "plate_detected": bool(vehicle_data.get("plate")),
                "color": vehicle_data.get("color"),
                "make": vehicle_data.get("make"),
                "model": vehicle_data.get("model"),
                "attribute_confidence": vehicle_data.get("attribute_confidence"),
                "attribute_model": vehicle_data.get("attribute_model"),
                "observation_validation_status": vehicle_data.get("observation_validation_status"),
                "observation_validation_confidence": vehicle_data.get("observation_validation_confidence"),
                "observation_validation_quality": vehicle_data.get("observation_validation_quality"),
                "observation_validation_reasons": vehicle_data.get("observation_validation_reasons"),
                "observation_validation_warnings": vehicle_data.get("observation_validation_warnings"),
                "observation_monotonic": vehicle_data.get("observation_monotonic"),
                "gis": {
                    "latitude": vehicle_data.get("gis_latitude"),
                    "longitude": vehicle_data.get("gis_longitude"),
                    "position_type": vehicle_data.get("gis_position_type"),
                    "pixel_x": vehicle_data.get("gis_pixel_x"),
                    "pixel_y": vehicle_data.get("gis_pixel_y"),
                },
            },
        )
    except Exception:
        logger.exception(
            "Vehicle observation persistence failed | cam_id=%s track=%s",
            vehicle_data.get("camera_id"),
            vehicle_data.get("track_id"),
        )
        return None


def _persist_correlation_decision_safe(vehicle_data, correlation):
    """Persist an auditable MATCH/UNCERTAIN/REJECT decision."""
    if not isinstance(vehicle_data, dict):
        return None
    if not isinstance(correlation, dict):
        return None

    decision = str(
        correlation.get("decision")
        or correlation.get("match_type")
        or (
            "MATCH"
            if correlation.get("strong_match")
            else "UNCERTAIN"
        )
    ).upper()

    if decision not in {"MATCH", "UNCERTAIN", "REJECT"}:
        decision = "UNCERTAIN"

    try:
        return save_correlation_decision(
            source_camera_id=str(
                vehicle_data.get("camera_id") or ""
            ),
            source_track_id=(
                str(vehicle_data.get("track_id"))
                if vehicle_data.get("track_id") is not None
                else None
            ),
            candidate_global_vehicle_id=(
                str(correlation.get("global_vehicle_id"))
                if correlation.get("global_vehicle_id")
                else None
            ),
            decision=decision,
            final_score=_safe_float(
                correlation.get(
                    "correlation_score",
                    correlation.get("similarity", 0.0),
                ),
                0.0,
                0.0,
                1.0,
            ),
            reid_score=_safe_float(
                correlation.get("similarity"),
                0.0,
                0.0,
                1.0,
            ),
            plate_score=_safe_float(
                correlation.get("plate_score"),
                0.0,
                0.0,
                1.0,
            ),
            vehicle_type_score=_safe_float(
                correlation.get("vehicle_type_score"),
                0.0,
                0.0,
                1.0,
            ),
            track_quality_score=_safe_float(
                correlation.get("track_quality"),
                0.0,
                0.0,
                1.0,
            ),
            temporal_score=_safe_float(
                correlation.get("temporal_score"),
                0.0,
                0.0,
                1.0,
            ),
            gis_score=_safe_float(
                correlation.get("gis_score"),
                0.0,
                0.0,
                1.0,
            ),
            direction_score=_safe_float(
                correlation.get("direction_score"),
                0.0,
                0.0,
                1.0,
            ),
            reason=str(
                correlation.get("reason")
                or correlation.get("match_type")
                or "Vehicle correlation decision"
            )[:2000],
            model_name="vehicle_correlation_engine",
            model_version=VEHICLE_REID_MODEL_VERSION,
            evidence_metadata=correlation.get("correlation_evidence")
            or {
                "anpr_lpr": {"score": correlation.get("anpr_score", 0.0)},
                "vehicle_reid": {"score": correlation.get("reid_score", 0.0)},
                "metadata": {"score": correlation.get("metadata_score", 0.0)},
            },
        )
    except Exception:
        logger.exception(
            "Correlation decision persistence failed | cam_id=%s track=%s",
            vehicle_data.get("camera_id"),
            vehicle_data.get("track_id"),
        )
        return None


def _persist_activity_event_safe(alert_obj, alert, camID, source_type):
    """Persist an EventFusion activity after the alert creation gate."""
    if alert_obj is None or not isinstance(alert, dict):
        return None

    try:
        from datetime import datetime, timezone

        return save_activity_event(
            event_id=f"ACT-{int(alert_obj.id)}",
            camera_id=str(camID),
            track_id=(
                str(alert.get("track_id"))
                if alert.get("track_id") is not None
                else None
            ),
            rule_id=str(alert.get("rule") or "UNKNOWN"),
            event_type=str(alert.get("rule") or "UNKNOWN"),
            severity=str(
                alert.get("level")
                or alert.get("severity")
                or "LOW"
            ).upper(),
            started_at=datetime.now(timezone.utc).replace(
                tzinfo=None
            ),
            ended_at=None,
            confidence=_safe_float(
                alert.get("confidence"),
                0.0,
                0.0,
                1.0,
            ),
            status="CONFIRMED",
            evidence={
                "source_type": str(
                    source_type or "unknown"
                ),
                "box": (
                    list(map(int, alert["box"]))
                    if alert.get("box") is not None
                    else None
                ),
                "zone": alert.get("zone"),
            },
            model_name="eventfusion",
            model_version=EVENTFUSION_VERSION,
            metadata={
                "alert_id": int(alert_obj.id),
                "source_type": str(
                    source_type or "unknown"
                ),
            },
        )
    except Exception:
        logger.exception(
            "Activity event persistence failed | cam_id=%s rule=%s alert_id=%s",
            camID,
            alert.get("rule"),
            getattr(alert_obj, "id", None),
        )
        return None


def _uses_deferred_behavior_evidence(rule) -> bool:
    return bool(
        BEHAVIOR_BEST_EVIDENCE_ENABLED
        and str(rule or "").strip().casefold() in BEHAVIOR_EVIDENCE_RULES
    )


def _behavior_evidence_area(camera, cam_id, zone_name=None) -> str:
    configured = BEHAVIOR_EVIDENCE_CAMERA_GROUPS.get(str(cam_id))
    if configured:
        return configured.strip().casefold()

    for value in (
        getattr(camera, "location_name", None) if camera is not None else None,
        zone_name,
        getattr(camera, "zone", None) if camera is not None else None,
    ):
        normalized = str(value or "").strip()
        if normalized:
            return normalized.casefold()

    # Without an explicit camera group or shared location, do not merge two
    # unrelated cameras merely because they produced the same rule at once.
    return f"camera:{cam_id}".casefold()


def _expanded_behavior_roi(frame, box):
    if frame is None or getattr(frame, "size", 0) == 0:
        return None, None
    try:
        frame_h, frame_w = frame.shape[:2]
        if box is None or len(box) < 4:
            return frame.copy(), (0, 0, frame_w, frame_h)
        x1, y1, x2, y2 = [int(round(float(value))) for value in box[:4]]
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w, x2))
        y2 = max(0, min(frame_h, y2))
        if x2 <= x1 or y2 <= y1:
            return frame.copy(), (0, 0, frame_w, frame_h)

        pad_x = int((x2 - x1) * BEHAVIOR_EVIDENCE_ROI_PADDING_RATIO)
        pad_y = int((y2 - y1) * BEHAVIOR_EVIDENCE_ROI_PADDING_RATIO)
        crop_box = (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(frame_w, x2 + pad_x),
            min(frame_h, y2 + pad_y),
        )
        cx1, cy1, cx2, cy2 = crop_box
        roi = frame[cy1:cy2, cx1:cx2].copy()
        return (roi if roi.size else None), crop_box
    except (TypeError, ValueError, IndexError):
        frame_h, frame_w = frame.shape[:2]
        return frame.copy(), (0, 0, frame_w, frame_h)


def _behavior_context_quality(roi, alert_confidence=0.0) -> float:
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        blur_score = min(1.0, max(0.0, blur / 150.0))
        brightness_score = max(0.0, 1.0 - abs(brightness - 125.0) / 125.0)
        contrast_score = min(1.0, max(0.0, contrast / 64.0))
        return max(
            0.0,
            min(
                1.0,
                blur_score * 0.45
                + brightness_score * 0.20
                + contrast_score * 0.15
                + _safe_float(alert_confidence, 0.0, 0.0, 1.0) * 0.20,
            ),
        )
    except Exception:
        return _safe_float(alert_confidence, 0.0, 0.0, 1.0) * 0.20


def _behavior_face_quality(face, roi_shape) -> float:
    roi_h, roi_w = roi_shape[:2]
    face_area = max(
        1,
        int(face.get("face_width", 0)) * int(face.get("face_height", 0)),
    )
    size_score = min(1.0, face_area / max(1.0, float(roi_h * roi_w) * 0.08))
    detection_score = _safe_float(face.get("detection_score"), 0.0, 0.0, 1.0)
    blur_score = min(1.0, _safe_float(face.get("blur_score"), 0.0) / 150.0)
    frontal_score = _safe_float(face.get("frontal_score"), 0.5, 0.0, 1.0)
    brightness = _safe_float(face.get("brightness"), 0.0, 0.0, 255.0)
    brightness_score = max(0.0, 1.0 - abs(brightness - 125.0) / 125.0)
    return max(
        0.0,
        min(
            1.0,
            detection_score * 0.35
            + blur_score * 0.25
            + size_score * 0.20
            + frontal_score * 0.15
            + brightness_score * 0.05,
        ),
    )


def _register_behavior_evidence_alert(
    *,
    user_id,
    camera,
    cam_id,
    rule,
    zone_name,
    alert_id,
    box,
    confidence,
    level,
    source_type,
):
    now = time.monotonic()
    area = _behavior_evidence_area(camera, cam_id, zone_name)
    scope = f"{int(user_id)}|{str(rule).strip().casefold()}|{area}"

    with behaviorEvidenceLock:
        expired = [
            key for key, group in behaviorEvidenceGroups.items()
            if group.get("finalized")
            or now > float(group.get("deadline", 0.0)) + 30.0
        ]
        for key in expired:
            _remove_behavior_evidence_group_locked(key)

        group_key = None
        for key, group in behaviorEvidenceGroups.items():
            if (
                group.get("scope") == scope
                and not group.get("finalizing")
                and now <= float(group.get("deadline", 0.0))
            ):
                group_key = key
                break

        is_new = group_key is None
        if is_new:
            active_count = sum(
                1 for group in behaviorEvidenceGroups.values()
                if not group.get("finalizing") and not group.get("finalized")
            )
            if active_count >= BEHAVIOR_EVIDENCE_MAX_CONCURRENT_INCIDENTS:
                return None, False
            group_key = f"BEHAVIOR-{uuid.uuid4().hex[:20]}"
            behaviorEvidenceGroups[group_key] = {
                "key": group_key,
                "scope": scope,
                "user_id": int(user_id),
                "rule": str(rule),
                "area": area,
                "created_at": time.time(),
                "created_monotonic": now,
                "deadline": now + BEHAVIOR_EVIDENCE_WINDOW_SECONDS,
                "finalizing": False,
                "finalized": False,
                "scheduled": False,
                "early_ready": False,
                "candidate_count": 0,
                "best_face": None,
                "best_context": None,
                "incident_id": None,
                "cameras": {},
                "payloads": {},
            }

        group = behaviorEvidenceGroups[group_key]
        camera_id = str(cam_id)
        group["cameras"][camera_id] = {
            "camera_id": camera_id,
            "alert_id": int(alert_id),
            "box": [int(float(value)) for value in box[:4]],
            "confidence": _safe_float(confidence, 0.0, 0.0, 1.0),
            "level": str(level),
            "source_type": str(source_type or "unknown"),
            "last_sample_at": 0.0,
        }
        behaviorEvidenceCameraGroups.setdefault(camera_id, set()).add(group_key)
        return group_key, is_new


def _remove_behavior_evidence_group_locked(group_key):
    group = behaviorEvidenceGroups.pop(group_key, None)
    if not group:
        return
    for camera_id in list((group.get("cameras") or {}).keys()):
        keys = behaviorEvidenceCameraGroups.get(str(camera_id))
        if keys is not None:
            keys.discard(group_key)
            if not keys:
                behaviorEvidenceCameraGroups.pop(str(camera_id), None)


def _consider_behavior_evidence_frame(group_key, cam_id, clean_frame, *, force=False):
    camera_id = str(cam_id)
    now = time.monotonic()
    interval = 1.0 / max(0.25, BEHAVIOR_EVIDENCE_SAMPLE_FPS)

    with behaviorEvidenceLock:
        group = behaviorEvidenceGroups.get(group_key)
        if not group or group.get("finalizing") or group.get("finalized"):
            return False
        observation = (group.get("cameras") or {}).get(camera_id)
        if not observation:
            return False
        if (
            not force
            and now - float(observation.get("last_sample_at", 0.0)) < interval
        ):
            return False
        observation["last_sample_at"] = now
        box = list(observation.get("box") or [])
        alert_id = int(observation.get("alert_id"))
        confidence = float(observation.get("confidence", 0.0))

    roi, crop_box = _expanded_behavior_roi(clean_frame, box)
    if roi is None:
        return False

    success, encoded = cv2.imencode(
        ".jpg",
        roi,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    if not success:
        return False
    image_data = encoded.tobytes()
    context_score = _behavior_context_quality(roi, confidence)

    try:
        faces = detect_face_evidence_candidates(
            roi,
            minimum_face_size=BEHAVIOR_EVIDENCE_MIN_FACE_SIZE,
            minimum_detection_score=BEHAVIOR_EVIDENCE_MIN_FACE_CONFIDENCE,
            minimum_blur_score=BEHAVIOR_EVIDENCE_MIN_BLUR_SCORE,
            max_faces=4,
        )
    except Exception:
        logger.exception(
            "Behavior evidence face-quality detection failed | group=%s camera=%s",
            group_key,
            camera_id,
        )
        faces = []

    best_face = None
    best_face_score = 0.0
    for face in faces:
        score = _behavior_face_quality(face, roi.shape)
        if score > best_face_score:
            best_face_score = score
            best_face = dict(face)

    candidate = {
        "camera_id": camera_id,
        "alert_id": alert_id,
        "image_data": image_data,
        "crop_box": list(crop_box),
        "context_score": context_score,
        "face_score": best_face_score,
        "face": best_face,
        "captured_at": time.time(),
    }

    with behaviorEvidenceLock:
        group = behaviorEvidenceGroups.get(group_key)
        if not group or group.get("finalizing") or group.get("finalized"):
            return False
        group["candidate_count"] = int(group.get("candidate_count", 0)) + 1
        current_context = group.get("best_context")
        if (
            current_context is None
            or context_score > float(current_context.get("context_score", 0.0))
        ):
            group["best_context"] = candidate
        if best_face is not None:
            current_face = group.get("best_face")
            if (
                current_face is None
                or best_face_score > float(current_face.get("face_score", 0.0))
            ):
                group["best_face"] = candidate
            if best_face_score >= BEHAVIOR_EVIDENCE_EARLY_ACCEPT_SCORE:
                group["early_ready"] = True
        return True


def _sample_active_behavior_evidence(clean_frame, cam_id):
    with behaviorEvidenceLock:
        keys = list(behaviorEvidenceCameraGroups.get(str(cam_id), set()))
    for group_key in keys:
        _consider_behavior_evidence_frame(group_key, cam_id, clean_frame)


def _persist_behavior_incident_observation(
    db,
    *,
    group_key,
    alert_obj,
    user_id,
    cam_id,
    rule,
    level,
    confidence,
):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    incident = (
        db.query(Incident)
        .filter(
            Incident.user_id == int(user_id),
            Incident.primary_track_id == str(group_key),
            Incident.incident_type == str(rule)[:100],
            Incident.status.in_(["OPEN", "UNDER_REVIEW", "ESCALATED"]),
            Incident.started_at >= now - timedelta(seconds=60),
        )
        .order_by(Incident.started_at.desc())
        .first()
    )
    created = incident is None
    observation = {
        "camera_id": str(cam_id),
        "alert_id": int(alert_obj.id),
        "confidence": _safe_float(confidence, 0.0, 0.0, 1.0),
        "timestamp": now.isoformat(),
    }

    if incident is None:
        selection = {
            "status": "COLLECTING",
            "selection_window_seconds": BEHAVIOR_EVIDENCE_WINDOW_SECONDS,
            "observed_cameras": [str(cam_id)],
            "observations": [observation],
        }
        incident = Incident(
            user_id=int(user_id),
            cam_id=str(cam_id),
            primary_track_id=str(group_key),
            incident_type=str(rule)[:100],
            status="OPEN",
            evidence=json.dumps({"source": "EVENTFUSION", "group_key": group_key})[:10000],
            evaluation_status="PENDING",
            evaluation_confidence=_safe_float(confidence, 0.0, 0.0, 1.0),
            evaluation_severity=str(level or "INFO")[:50],
            evaluation_reason="Selecting the clearest face-visible behavioral evidence frame.",
            evaluation_evidence={"behavior_evidence_selection": selection},
        )
        db.add(incident)
        db.flush()
    else:
        state = dict(incident.evaluation_evidence or {})
        selection = dict(state.get("behavior_evidence_selection") or {})
        observations = list(selection.get("observations") or [])
        if not any(int(item.get("alert_id", -1)) == int(alert_obj.id) for item in observations):
            observations.append(observation)
        cameras = {str(value) for value in selection.get("observed_cameras") or []}
        cameras.add(str(cam_id))
        selection.update(
            {
                "status": "COLLECTING",
                "selection_window_seconds": BEHAVIOR_EVIDENCE_WINDOW_SECONDS,
                "observed_cameras": sorted(cameras),
                "observations": observations[-100:],
            }
        )
        state["behavior_evidence_selection"] = selection
        incident.evaluation_evidence = state
        incident.evaluation_confidence = max(
            float(incident.evaluation_confidence or 0.0),
            _safe_float(confidence, 0.0, 0.0, 1.0),
        )
        incident.evaluation_severity = str(level or incident.evaluation_severity or "INFO")[:50]

    with behaviorEvidenceLock:
        group = behaviorEvidenceGroups.get(group_key)
        if group is not None:
            group["incident_id"] = int(incident.id)
    return incident, created


def _set_behavior_evidence_payload(group_key, cam_id, payload):
    with behaviorEvidenceLock:
        group = behaviorEvidenceGroups.get(group_key)
        if group is not None:
            group.setdefault("payloads", {})[str(cam_id)] = dict(payload)


def _finalize_behavior_evidence_db(group_key):
    with behaviorEvidenceLock:
        group = behaviorEvidenceGroups.get(group_key)
        if not group:
            return None
        candidate = group.get("best_face")
        used_face = candidate is not None
        if candidate is None and BEHAVIOR_EVIDENCE_ALLOW_CONTEXT_FALLBACK:
            candidate = group.get("best_context")
        snapshot = {
            "key": group_key,
            "user_id": int(group["user_id"]),
            "rule": str(group["rule"]),
            "area": str(group["area"]),
            "incident_id": group.get("incident_id"),
            "candidate_count": int(group.get("candidate_count", 0)),
            "camera_ids": sorted((group.get("cameras") or {}).keys()),
            "candidate": candidate,
            "used_face": used_face,
            "payloads": dict(group.get("payloads") or {}),
        }

    candidate = snapshot["candidate"]
    winning_camera_id = str(candidate.get("camera_id")) if candidate else None
    winning_alert_id = int(candidate.get("alert_id")) if candidate else None
    base_payload = dict(
        snapshot["payloads"].get(winning_camera_id)
        or next(iter(snapshot["payloads"].values()), {})
    )
    snapshot_id = None
    evidence_sha256 = None

    db = SessionLocal()
    try:
        alert_obj = db.query(Alert).filter(Alert.id == winning_alert_id).first() if winning_alert_id else None
        if candidate is not None and alert_obj is not None:
            evidence_role = (
                "BEST_FACE_BEHAVIOR_EVIDENCE"
                if snapshot["used_face"]
                else "BEST_CONTEXT_BEHAVIOR_EVIDENCE"
            )
            snapshot_obj = save_snapshot_to_db(
                db=db,
                alert_id=int(alert_obj.id),
                snapshot_data=candidate["image_data"],
                is_original=False,
                metadata={
                    "evidence_role": evidence_role,
                    "behavior_group_key": group_key,
                    "winning_camera_id": winning_camera_id,
                    "face_score": float(candidate.get("face_score", 0.0)),
                    "context_score": float(candidate.get("context_score", 0.0)),
                },
            )
            if snapshot_obj is not None:
                snapshot_id = int(snapshot_obj.id)
                if EVIDENCE_HASHING_ENABLED:
                    evidence_sha256 = sha256_bytes(candidate["image_data"])

        incident = None
        if snapshot.get("incident_id"):
            incident = db.query(Incident).filter(
                Incident.id == int(snapshot["incident_id"]),
                Incident.user_id == int(snapshot["user_id"]),
            ).first()
        if incident is not None:
            state = dict(incident.evaluation_evidence or {})
            selection = dict(state.get("behavior_evidence_selection") or {})
            selection.update(
                {
                    "status": "SELECTED" if snapshot_id else "NO_CANDIDATE",
                    "candidate_count": snapshot["candidate_count"],
                    "observed_cameras": snapshot["camera_ids"],
                    "winning_camera_id": winning_camera_id,
                    "winning_alert_id": winning_alert_id,
                    "snapshot_id": snapshot_id,
                    "face_visible": bool(snapshot["used_face"]),
                    "face_score": (
                        float(candidate.get("face_score", 0.0)) if candidate else 0.0
                    ),
                    "context_fallback": bool(candidate is not None and not snapshot["used_face"]),
                    "finalized_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            state["behavior_evidence_selection"] = selection
            incident.evaluation_evidence = state
            incident.evaluation_reason = (
                "Best clear face-visible behavioral evidence selected."
                if snapshot["used_face"]
                else "No clear face passed quality gates; best clean event-context snapshot selected."
            )

            if snapshot_id is not None:
                existing = db.query(IncidentEvidence).filter(
                    IncidentEvidence.incident_id == int(incident.id),
                    IncidentEvidence.snapshot_id == int(snapshot_id),
                ).first()
                if existing is None:
                    db.add(
                        IncidentEvidence(
                            incident_id=int(incident.id),
                            user_id=int(snapshot["user_id"]),
                            alert_id=winning_alert_id,
                            snapshot_id=int(snapshot_id),
                            evidence_type=(
                                "BEST_FACE_BEHAVIOR_EVIDENCE"
                                if snapshot["used_face"]
                                else "BEST_CONTEXT_BEHAVIOR_EVIDENCE"
                            ),
                            object_type="PERSON_PAIR",
                            object_reference=group_key[:150],
                            description=(
                                f"Selected primary evidence for {snapshot['rule']} from {winning_camera_id}"
                            )[:500],
                            metadata_json=selection,
                        )
                    )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Behavior evidence finalization failed | group=%s", group_key)
        return None
    finally:
        db.close()

    base_payload.update(
        {
            "type": "alert",
            "event": "BEHAVIOR_EVIDENCE_FINALIZED",
            "behavior_group_key": group_key,
            "incident_id": snapshot.get("incident_id"),
            "camera_ids": snapshot["camera_ids"],
            "winning_camera_id": winning_camera_id,
            "candidate_count": snapshot["candidate_count"],
            "evidence_selection_status": "SELECTED" if snapshot_id else "NO_CANDIDATE",
            "face_visible": bool(snapshot["used_face"]),
            "face_quality_score": (
                float(candidate.get("face_score", 0.0)) if candidate else 0.0
            ),
            "context_fallback": bool(candidate is not None and not snapshot["used_face"]),
            "snapshot_id": snapshot_id,
            "snapshot_path": f"/snapshot/{snapshot_id}" if snapshot_id else None,
            "snapshot_saved": snapshot_id is not None,
            "has_snapshot": snapshot_id is not None,
            "evidence_sha256": evidence_sha256,
            "created_at": time.time(),
        }
    )
    if snapshot["used_face"]:
        base_payload["message"] = (
            f"{snapshot['rule']} | best clear face evidence selected from {winning_camera_id}"
        )
    else:
        base_payload["message"] = (
            f"{snapshot['rule']} | no clear face found; best event-context snapshot selected from {winning_camera_id}"
        )
    return base_payload


async def _finalize_behavior_evidence_after_window(group_key):
    while True:
        await asyncio.sleep(0.25)
        with behaviorEvidenceLock:
            group = behaviorEvidenceGroups.get(group_key)
            if not group:
                return
            now = time.monotonic()
            elapsed = now - float(group.get("created_monotonic", now))
            camera_count = len(group.get("cameras") or {})
            ready = (
                now >= float(group["deadline"])
                or (
                    bool(group.get("early_ready"))
                    and elapsed >= BEHAVIOR_EVIDENCE_MIN_COLLECTION_SECONDS
                    and camera_count >= 2
                )
            )
            if not ready:
                continue
            if group.get("finalizing"):
                return
            group["finalizing"] = True
            break

    payload = await asyncio.to_thread(_finalize_behavior_evidence_db, group_key)
    try:
        if payload:
            await sendAlert(payload)
            if payload.get("level") in NOTIFICATION_LEVELS:
                await sendTelegramAlert(payload)
    except Exception:
        logger.exception("Final behavior evidence delivery failed | group=%s", group_key)
    finally:
        with behaviorEvidenceLock:
            group = behaviorEvidenceGroups.get(group_key)
            if group is not None:
                group["finalized"] = True
            _remove_behavior_evidence_group_locked(group_key)


async def handle_alerts(frame, camID, alerts, source_type="rtsp"):
    if not alerts:
        return

    with alert_processing_seconds.time():
        alerts.sort(
            key=lambda x: LEVEL_PRIORITY.get(x.get("level", "LOW"), 0),
            reverse=True,
        )

        for alert in alerts:
            rule = alert.get("rule")
            if not rule:
                continue

            confidence = float(alert.get("confidence", 0.0))
            level = alert.get("level") or get_alert_level(rule, confidence)
            box = alert.get("box")
            track_id = alert.get("track_id") or alert.get("track")
            track_id_str = str(track_id) if track_id is not None else "global"
            defer_behavior_evidence = _uses_deferred_behavior_evidence(rule)
            behavior_group_key = None
            behavior_group_is_new = False

            # Vehicle alerts can be generated before the identity enrichment
            # stage. Recover the local-track -> global identity mapping here
            # so incident/watchlist context remains connected to the vehicle.
            if (
                not alert.get("global_vehicle_id")
                and get_global_vehicle_id is not None
                and track_id is not None
            ):
                try:
                    mapped_global_id = get_global_vehicle_id(
                        str(camID),
                        track_id,
                    )
                    if mapped_global_id:
                        alert["global_vehicle_id"] = mapped_global_id
                except Exception:
                    logger.debug(
                        "Local/global vehicle mapping lookup failed | cam_id=%s track=%s",
                        camID,
                        track_id,
                        exc_info=True,
                    )

            # Advanced risk/incident fusion stays additive to the existing alert rules.
            risk_result = None
            incident_result = None
            explanation = None
            if ADVANCED_INTELLIGENCE_ENABLED:
                try:
                    risk_payload = {
                        "watchlist_match": bool(alert.get("watchlist_entry_id") or alert.get("watchlist_match")),
                        "restricted_zone": bool(alert.get("camera_mode") == "restricted" or alert.get("zone")),
                        "behavior_anomaly": True,
                        "cross_camera_match": bool(alert.get("global_vehicle_id")),
                        "camera_tamper": bool(alert.get("camera_tampered")),
                        "plate_stable": bool(alert.get("stable_plate") or alert.get("plate")),
                    }
                    if RISK_SCORING_ENABLED:
                        risk_result = advanced_intelligence.create_risk(risk_payload)
                        if risk_result.get("severity") == "CRITICAL":
                            level = "CRITICAL"
                        elif risk_result.get("severity") == "HIGH" and level not in {"CRITICAL"}:
                            level = "HIGH"
                    if INCIDENT_CORRELATION_ENABLED:
                        entity_key = str(alert.get("global_vehicle_id") or f"{camID}:{track_id_str}")
                        incident_result = advanced_intelligence.correlate(
                            event_id=f"ALERT-{int(time.time()*1000)}",
                            camera_id=str(camID),
                            entity_key=entity_key,
                            event_type=str(rule),
                            confidence=confidence,
                            evidence={"rule": rule, "risk": risk_result},
                        )
                    explanation = build_alert_explanation(
                        rule=str(rule), camera_id=str(camID), confidence=confidence,
                        evidence={"risk": risk_result, "incident": incident_result},
                    )
                except Exception:
                    logger.exception("Advanced alert intelligence failed | cam_id=%s rule=%s", camID, rule)

            zone_name = None
            if source_type == "rtsp":
                zone_data = alert.get("zone")

                if isinstance(zone_data, dict):
                    zone_name = zone_data.get("zone_name")
                elif isinstance(zone_data, str):
                    zone_name = zone_data

            snapshot_data = None

            if (
                not defer_behavior_evidence
                and level in SNAPSHOT_LEVELS
                and box is not None
            ):
                if should_snapshot(camID, track_id_str, rule):
                    snapshot_data = crop_and_encode_snapshot(frame, box)

            alert_id = None
            snapshot_id = None
            user_id = None

            db = SessionLocal()

            try:
                camera, user_id, owner_resolution = _resolve_alert_persistence_context(
                    db=db,
                    cam_id=str(camID),
                    source_type=source_type,
                )

                if user_id is None:
                    logger.error(
                        "[ALERT-PERSIST] alert dropped because owner could not be resolved | "
                        "cam_id=%s source_type=%s rule=%s track=%s",
                        camID,
                        source_type,
                        rule,
                        track_id_str,
                    )
                    continue

                logger.debug(
                    "[ALERT-PERSIST] persistence attempt | cam_id=%s owner_id=%s "
                    "owner_source=%s rule=%s level=%s track=%s confidence=%.4f",
                    camID,
                    user_id,
                    owner_resolution,
                    rule,
                    level,
                    track_id_str,
                    confidence,
                )

                alert_obj, created = save_alert(
                    db=db,
                    user_id=int(user_id),
                    alert_type=level,
                    alert_rule=rule,
                    track_id=track_id_str,
                    cam_id=str(camID),
                    source_type=source_type,
                    zone=zone_name,
                    level=level,
                    cooldown_seconds=RULE_COOLDOWN_SECONDS.get(rule, 180),
                    source_timestamp=alert.get("source_timestamp"),
                    source_pts_seconds=alert.get("source_pts_seconds"),
                    timestamp_source=alert.get("timestamp_source"),
                    timestamp_quality=alert.get("timestamp_quality"),
                    confidence_score=confidence,
                )

                if not created or alert_obj is None:
                    logger.debug(
                        "[ALERT-PERSIST] duplicate/cooldown suppression | "
                        "cam_id=%s owner_id=%s rule=%s track=%s",
                        camID,
                        user_id,
                        rule,
                        track_id_str,
                    )
                    continue

                alert_id = int(alert_obj.id)

                if defer_behavior_evidence and box is not None:
                    behavior_group_key, behavior_group_is_new = (
                        _register_behavior_evidence_alert(
                            user_id=int(user_id),
                            camera=camera,
                            cam_id=str(camID),
                            rule=str(rule),
                            zone_name=zone_name,
                            alert_id=alert_id,
                            box=box,
                            confidence=confidence,
                            level=level,
                            source_type=source_type,
                        )
                    )

                    # If the bounded selector is at capacity, preserve the
                    # original behavior so this alert still receives a clean
                    # snapshot instead of silently losing evidence.
                    if behavior_group_key is None:
                        defer_behavior_evidence = False
                        if level in SNAPSHOT_LEVELS:
                            snapshot_data = crop_and_encode_snapshot(frame, box)

                logger.info(
                    "[ALERT-PERSIST] alert saved | alert_id=%s cam_id=%s owner_id=%s "
                    "rule=%s level=%s snapshot_candidate=%s",
                    alert_id,
                    camID,
                    user_id,
                    rule,
                    level,
                    bool(snapshot_data),
                )

                # Incident evidence is independent of the alert snapshot
                # cooldown. A newly created incident must have an evidence
                # frame whenever a vehicle/person bounding box is available.
                incident_snapshot_data = (
                    None if defer_behavior_evidence else snapshot_data
                )
                if (
                    not defer_behavior_evidence
                    and
                    incident_result
                    and incident_snapshot_data is None
                    and box is not None
                ):
                    try:
                        incident_snapshot_data = crop_and_encode_snapshot(
                            frame,
                            box,
                        )
                    except Exception:
                        logger.exception(
                            "Incident evidence snapshot capture failed | cam_id=%s track=%s",
                            camID,
                            track_id_str,
                        )

                _persist_activity_event_safe(
                    alert_obj=alert_obj,
                    alert=alert,
                    camID=camID,
                    source_type=source_type,
                )

                if snapshot_data:
                    snapshot_obj = save_snapshot_to_db(
                        db=db,
                        alert_id=alert_id,
                        snapshot_data=snapshot_data,
                    )

                    if snapshot_obj:
                        snapshot_id = snapshot_obj.id

                if defer_behavior_evidence and behavior_group_key:
                    incident_obj, incident_created = (
                        _persist_behavior_incident_observation(
                            db=db,
                            group_key=behavior_group_key,
                            alert_obj=alert_obj,
                            user_id=int(user_id),
                            cam_id=str(camID),
                            rule=str(rule),
                            level=str(level),
                            confidence=float(confidence),
                        )
                    )
                    db.commit()
                    db.refresh(incident_obj)
                else:
                    incident_obj, incident_created = _persist_incident_for_alert(
                        db=db,
                        alert_obj=alert_obj,
                        alert_payload=alert,
                        snapshot_id=snapshot_id,
                        snapshot_data=incident_snapshot_data,
                        original_frame=frame if incident_result else None,
                        cam_id=str(camID),
                        track_id=track_id_str,
                        user_id=int(user_id),
                        rule=str(rule),
                        level=str(level),
                        confidence=float(confidence),
                        incident_result=incident_result,
                    )

                if incident_obj is not None:
                    incident_result = {
                        **(incident_result or {}),
                        "incident_id": int(incident_obj.id),
                        "incident_event_count": (
                            incident_result or {}
                        ).get(
                            "event_count",
                            1,
                        ),
                        "incident_created": bool(incident_created),
                        "snapshot_id": snapshot_id,
                    }

            except Exception:
                db.rollback()
                logger.exception(
                    "Alert DB/snapshot save failed | cam_id=%s rule=%s track_id=%s",
                    camID,
                    rule,
                    track_id_str,
                )
                continue

            finally:
                db.close()

            evidence_sha256 = None
            if EVIDENCE_HASHING_ENABLED and snapshot_data:
                try:
                    evidence_sha256 = sha256_bytes(snapshot_data)
                except Exception:
                    logger.exception("Evidence hash calculation failed | cam_id=%s alert_id=%s", camID, alert_id)

            alertData = {
                "id": alert_id,
                "alert_id": alert_id,
                "user_id": user_id,
                "cam_id": camID,
                "source_type": source_type,
                "alert_type": level,
                "alert_level": level,
                "level": level,
                "rule": rule,
                "alert_rule": rule,
                "track_id": track_id_str,
                "zone": zone_name,
                "confidence": confidence,
                "box": list(map(int, box)) if box is not None else None,
                "message": build_alert_message(
                    rule=rule,
                    camID=camID,
                    source_type=source_type,
                    zone_name=zone_name,
                    confidence=confidence,
                ),
                "snapshot_id": snapshot_id,
                "snapshot_saved": snapshot_id is not None,
                "snapshot_path": f"/snapshot/{snapshot_id}" if snapshot_id else None,
                "has_snapshot": snapshot_id is not None,
                "evidence_sha256": evidence_sha256,
                "created_at": time.time(),
                "source_timestamp": (
                    alert.get("source_timestamp").isoformat()
                    if isinstance(alert.get("source_timestamp"), datetime)
                    else alert.get("source_timestamp")
                ),
                "source_pts_seconds": alert.get("source_pts_seconds"),
                "timestamp_source": alert.get("timestamp_source"),
                "timestamp_quality": alert.get("timestamp_quality"),
                "risk_score": (risk_result or {}).get("score"),
                "risk_severity": (risk_result or {}).get("severity"),
                "risk_explanation": (risk_result or {}).get("explanation"),
                "incident_id": (incident_result or {}).get("incident_id"),
                "global_vehicle_id": alert.get("global_vehicle_id"),
                "incident_snapshot_id": snapshot_id if incident_result else None,
                "incident_event_count": (incident_result or {}).get("event_count"),
                "alert_explanation": explanation,
                "behavior_group_key": behavior_group_key,
                "evidence_selection_status": (
                    "COLLECTING"
                    if defer_behavior_evidence and behavior_group_key
                    else "IMMEDIATE"
                ),
                "face_visible": None,
            }

            remember_alert_box(
                camID=camID,
                track_id=track_id_str,
                rule=rule,
                box=box,
                level=level,
                confidence=confidence,
                ts=time.time(),
            )

            if defer_behavior_evidence and behavior_group_key:
                _set_behavior_evidence_payload(
                    behavior_group_key,
                    camID,
                    alertData,
                )
                await asyncio.to_thread(
                    _consider_behavior_evidence_frame,
                    behavior_group_key,
                    camID,
                    frame,
                    force=True,
                )
                schedule_finalizer = False
                with behaviorEvidenceLock:
                    behavior_group = behaviorEvidenceGroups.get(behavior_group_key)
                    if behavior_group is not None and not behavior_group.get("scheduled"):
                        behavior_group["scheduled"] = True
                        schedule_finalizer = True
                if schedule_finalizer:
                    asyncio.create_task(
                        _finalize_behavior_evidence_after_window(
                            behavior_group_key
                        )
                    )

            # Send one immediate text-only frontend alert for the correlated
            # incident. Additional cameras join the same evidence-selection
            # group without producing duplicate popup notifications.
            if not defer_behavior_evidence or behavior_group_is_new:
                try:
                    await sendAlert(
                        {
                            **alertData,
                            "type": "alert",
                            "event": (
                                "BEHAVIOR_EVIDENCE_COLLECTING"
                                if defer_behavior_evidence
                                else alertData.get("event", "ALERT_CREATED")
                            ),
                        }
                    )
                except Exception:
                    logger.exception(
                        "WebSocket alert send failed | alert_id=%s cam_id=%s",
                        alert_id,
                        camID,
                    )

            telegram_ok = True

            try:
                if level in NOTIFICATION_LEVELS and not defer_behavior_evidence:
                    telegram_ok = await sendTelegramAlert(alertData)

                if not telegram_ok:
                    logger.warning(
                        "Telegram alert not sent | alert_id=%s cam_id=%s rule=%s level=%s snapshot_id=%s",
                        alert_id,
                        camID,
                        rule,
                        level,
                        snapshot_id,
                    )

            except Exception:
                logger.exception(
                    "Telegram alert send failed | alert_id=%s cam_id=%s rule=%s",
                    alert_id,
                    camID,
                    rule,
                )

            alerts_detected_total.labels(
                level=level,
                rule=rule,
                source_type=source_type or "unknown",
            ).inc()
            

def _persist_incident_for_alert(
    db,
    *,
    alert_obj,
    alert_payload: dict,
    snapshot_id: int | None,
    snapshot_data: bytes | None,
    original_frame: np.ndarray | None,
    cam_id: str,
    track_id: str,
    user_id: int,
    rule: str,
    level: str,
    confidence: float,
    incident_result: dict | None,
):
    """Persist an operational incident and link its evidence snapshot.

    The incident is deduplicated against the same correlated vehicle (or
    local track when no global identity exists) within the incident window.
    The alert remains the durable event; IncidentEvidence links the exact
    evidence frame to the incident.
    """
    if not incident_result:
        return None, False

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    global_vehicle_id = str(
        alert_payload.get("global_vehicle_id") or ""
    ).strip() or None
    entity_key = global_vehicle_id or f"{cam_id}:{track_id}"

    query = db.query(Incident).filter(
        Incident.user_id == user_id,
        Incident.status.in_(["OPEN", "UNDER_REVIEW", "ESCALATED"]),
        Incident.started_at >= now - timedelta(seconds=180),
    )

    if global_vehicle_id:
        query = query.filter(Incident.global_vehicle_id == global_vehicle_id)
    else:
        query = query.filter(
            Incident.primary_track_id == entity_key,
            Incident.cam_id == cam_id,
        )

    incident = query.order_by(Incident.started_at.desc()).first()
    created = incident is None

    if incident is None:
        incident = Incident(
            user_id=user_id,
            cam_id=cam_id,
            primary_track_id=entity_key,
            global_vehicle_id=global_vehicle_id,
            incident_type=str(rule)[:100],
            status="OPEN",
            evidence=json.dumps(
                {
                    "source": "AI_ALERT",
                    "event_count": incident_result.get("event_count"),
                    "entity_key": entity_key,
                },
                default=str,
            )[:10000],
            evaluation_status="PENDING",
            evaluation_confidence=max(0.0, min(1.0, float(confidence or 0.0))),
            evaluation_severity=str(level or "INFO")[:50],
            evaluation_evidence={
                "incident_correlation": incident_result,
                "alert_id": alert_obj.id if alert_obj else None,
            },
        )
        db.add(incident)
        db.flush()
    else:
        incident.global_vehicle_id = (
            incident.global_vehicle_id or global_vehicle_id
        )
        incident.evaluation_confidence = max(
            float(incident.evaluation_confidence or 0.0),
            float(confidence or 0.0),
        )
        incident.evaluation_severity = str(level or incident.evaluation_severity or "INFO")[:50]
        incident.evaluation_evidence = {
            "incident_correlation": incident_result,
            "last_alert_id": alert_obj.id if alert_obj else None,
        }

    # Guarantee one incident snapshot for a newly created incident. Reuse the
    # alert snapshot if it already exists; otherwise persist the supplied frame.
    if snapshot_id is None and created and snapshot_data and alert_obj is not None:
        snapshot_obj = save_snapshot_to_db(
            db=db,
            alert_id=int(alert_obj.id),
            snapshot_data=snapshot_data,
            is_original=False,
            metadata={
                "evidence_role": "VEHICLE_CROP",
                "incident_id_pending": True,
                "camera_id": cam_id,
                "track_id": track_id,
            },
        )
        snapshot_id = int(snapshot_obj.id) if snapshot_obj else None

    if created and original_frame is not None and alert_obj is not None:
        original_snapshot_data = None
        try:
            success, encoded = cv2.imencode(".jpg", original_frame)
            if success:
                original_snapshot_data = encoded.tobytes()
        except Exception:
            logger.exception(
                "Original incident evidence encoding failed | cam_id=%s track=%s",
                cam_id,
                track_id,
            )

        original_obj = save_snapshot_to_db(
            db=db,
            alert_id=int(alert_obj.id),
            snapshot_data=original_snapshot_data,
            is_original=True,
            metadata={
                "evidence_role": "ORIGINAL_FRAME",
                "camera_id": cam_id,
                "track_id": track_id,
            },
        )
        if original_obj is not None:
            db.add(
                IncidentEvidence(
                    incident_id=int(incident.id),
                    user_id=user_id,
                    alert_id=int(alert_obj.id),
                    snapshot_id=int(original_obj.id),
                    evidence_type="ORIGINAL_FRAME",
                    object_type="VEHICLE",
                    object_reference=(
                        global_vehicle_id or entity_key
                    )[:150],
                    description="Original unannotated CCTV evidence frame.",
                    metadata_json={
                        "sha256": getattr(original_obj, "sha256", None),
                        "camera_id": cam_id,
                        "track_id": track_id,
                    },
                )
            )

    if snapshot_id is not None:
        existing = (
            db.query(IncidentEvidence)
            .filter(
                IncidentEvidence.incident_id == incident.id,
                IncidentEvidence.snapshot_id == snapshot_id,
            )
            .first()
        )
        if existing is None:
            db.add(
                IncidentEvidence(
                    incident_id=int(incident.id),
                    user_id=user_id,
                    alert_id=int(alert_obj.id) if alert_obj is not None else None,
                    snapshot_id=int(snapshot_id),
                    evidence_type="SNAPSHOT",
                    object_type="VEHICLE",
                    object_reference=(
                        global_vehicle_id or entity_key
                    )[:150],
                    description=(
                        f"Incident evidence snapshot for {rule}"
                    )[:500],
                    metadata_json={
                        "global_vehicle_id": global_vehicle_id,
                        "camera_id": cam_id,
                        "track_id": track_id,
                        "incident_event_count": incident_result.get("event_count"),
                        "confidence": confidence,
                    },
                )
            )

    db.commit()
    db.refresh(incident)
    return incident, created


def submit_alert_task(frame, camID, alerts_to_send, source_type):
    if not alerts_to_send:
        return

    if main_loop is None:
        logger.warning("Main event loop not ready. Alert skipped.")
        return

    future = asyncio.run_coroutine_threadsafe(
        handle_alerts(
            frame=frame.copy(),
            camID=camID,
            alerts=alerts_to_send,
            source_type=source_type,
        ),
        main_loop,
    )

    def _done_callback(f):
        try:
            f.result()
        except Exception as e:
            logger.exception("Alert task error")

    future.add_done_callback(_done_callback)



def _validate_capture_source(src, source_type: str) -> None:
    """Validate a capture source without logging or exposing credentials."""
    normalized_type = str(source_type or "").strip().lower()

    if normalized_type in {"onvif", "vendor_api", "vendor_sdk"}:
        # Connector-specific configuration is validated before the source
        # is opened. The connector resolves to a normalized stream/frame
        # source in _open_video_capture().
        return

    if normalized_type == "rtsp":
        if not isinstance(src, str):
            raise ValueError("RTSP source must be a string")

        source = src.strip()

        if not source.lower().startswith("rtsp://"):
            raise ValueError("RTSP source must use rtsp://")

        if "\r" in source or "\n" in source:
            raise ValueError("Invalid RTSP source")

        parsed = urlparse(source)

        if parsed.scheme.lower() != "rtsp" or not parsed.hostname:
            raise ValueError("Invalid RTSP source")

    elif normalized_type in {"http", "https", "live", "hls"}:
        if not isinstance(src, str):
            raise ValueError("Network source must be a string")

        source = src.strip()

        if "\r" in source or "\n" in source:
            raise ValueError("Invalid network source")

        parsed = urlparse(source)

        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Invalid network source")

        if normalized_type == "hls":
            path_lower = (parsed.path or "").lower()
            if not path_lower.endswith(".m3u8"):
                raise ValueError("HLS source must point to a .m3u8 playlist")


def _open_video_capture(src, source_type: str):
    """Open any supported INTEL-I source through a capture-like interface."""
    normalized_type = str(source_type or "").strip().lower()

    if normalized_type in {"onvif", "vendor_api", "vendor_sdk"}:
        if not isinstance(src, dict):
            raise ValueError("Connector configuration is missing")

        validate_connector_config(
            normalized_type,
            src,
        )

        from connectors.manager import open_camera_source

        capture, _resolved_type, _metadata = open_camera_source(
            normalized_type,
            src,
        )
        return capture

    _validate_capture_source(src, normalized_type)

    if normalized_type == "webcam":
        capture = cv2.VideoCapture(src)
    else:
        if not hasattr(cv2, "CAP_FFMPEG"):
            raise RuntimeError("OpenCV FFmpeg backend is unavailable")

        # OPENCV_FFMPEG_CAPTURE_OPTIONS is set before cv2 import.
        # rw_timeout bounds FFmpeg socket I/O so a dead RTSP endpoint does
        # not leave the analytics worker blocked indefinitely.
        capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)

        # OpenCV/FFmpeg can consume an HLS .m3u8 playlist directly.
        # HLS is HTTPS/HTTP, so the RTSP transport option is simply ignored.

        # OpenCV exposes these timeout properties for FFmpeg-backed network
        # capture on supported builds. They complement FFmpeg rw_timeout.
        timeout_ms = int(_rtsp_timeout_seconds * 1000)
        for property_name in (
            "CAP_PROP_OPEN_TIMEOUT_MSEC",
            "CAP_PROP_READ_TIMEOUT_MSEC",
        ):
            property_id = getattr(cv2, property_name, None)
            if property_id is not None:
                try:
                    capture.set(property_id, timeout_ms)
                except Exception:
                    # The FFmpeg backend may reject a property; rw_timeout
                    # remains the primary bounded-I/O safeguard.
                    pass

    if not capture.isOpened():
        try:
            capture.release()
        except Exception:
            pass
        return None

    return capture

def _persist_camera_time_sync(cam_id, bundle):
    """Upsert latest clock state at a bounded cadence; never write per frame."""
    now = time.monotonic()
    key = f"time_sync:{cam_id}"
    with state_lock:
        last = float(cameraQualityStates.get(key, 0.0) or 0.0)
        if now - last < CAMERA_HEALTH_PERSIST_SECONDS:
            return
        cameraQualityStates[key] = now
    try:
        with SessionLocal() as db:
            row = db.query(CameraTimeSync).filter(CameraTimeSync.camera_id == str(cam_id)).first()
            status = time_synchronizer.status(cam_id)
            if row is None:
                row = CameraTimeSync(camera_id=str(cam_id))
                db.add(row)
            row.camera_timestamp = bundle.camera_timestamp.replace(tzinfo=None) if bundle.camera_timestamp else None
            row.server_timestamp = bundle.server_timestamp.replace(tzinfo=None)
            row.ingestion_timestamp = bundle.ingestion_timestamp.replace(tzinfo=None)
            row.normalized_timestamp = bundle.normalized_timestamp.replace(tzinfo=None)
            row.clock_offset_ms = bundle.clock_offset_ms
            row.jitter_ms = status.get("jitter_ms")
            row.sync_status = bundle.sync_status
            row.sync_method = bundle.sync_method
            row.sample_count = int(status.get("samples") or 0)
            row.source_timestamp_available = bool(status.get("source_timestamp_available"))
            row.pts_seconds = bundle.pts_seconds
            row.timestamp_quality = bundle.timestamp_quality
            row.discontinuity_count = int(status.get("discontinuity_count") or 0)
            row.last_discontinuity_reason = status.get("last_discontinuity_reason")
            row.last_sync_at = bundle.server_timestamp.replace(tzinfo=None)
            row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            db.commit()
    except Exception:
        logger.debug("Camera time sync persistence failed | cam_id=%s", cam_id, exc_info=True)


def _persist_stream_metric_if_due(cam_id):
    now = time.monotonic()
    with state_lock:
        last = cameraQualityStates.get(f"stream_metric:{cam_id}", 0.0)
        if now - float(last) < STREAM_METRIC_PERSIST_SECONDS:
            return
        cameraQualityStates[f"stream_metric:{cam_id}"] = now
    try:
        telemetry = camera_health_registry.get(cam_id) or {}
        stream_stats = stream_manager.stats(cam_id)
        decoded = int(telemetry.get("decoded_frames") or 0)
        dropped = int(telemetry.get("dropped_frames") or 0)
        total = decoded + dropped
        with SessionLocal() as db:
            db.add(StreamMetric(
                camera_id=str(cam_id),
                observed_fps=telemetry.get("fps"),
                target_fps=telemetry.get("fps"),
                decoded_frames=decoded,
                processed_frames=int(telemetry.get("processed_frames") or 0),
                dropped_frames=dropped,
                dropped_ratio=(dropped / total) if total else 0.0,
                read_latency_ms=telemetry.get("latency_ms"),
                queue_depth=int(stream_stats.get("queue_depth") or 0),
                queue_capacity=int(stream_stats.get("queue_capacity") or STREAM_BUFFER_SIZE),
                reconnect_count=int(telemetry.get("reconnects") or 0),
                discontinuity_count=int((cameraTimelines.get(str(cam_id)).status() if cameraTimelines.get(str(cam_id)) else {}).get("discontinuity_count") or 0),
                last_frame_at=(datetime.fromisoformat(telemetry["last_frame_at"]).replace(tzinfo=None) if telemetry.get("last_frame_at") else None),
            ))
            db.commit()
    except Exception:
        logger.debug("Stream metric persistence failed | cam_id=%s", cam_id, exc_info=True)


def _camera_health_details(db, user_id):
    result = []
    for camera in get_cameras(db, user_id):
        try:
            state = normalize_camera_state(cameraConnectionStates.get(camera.cam_id, camera.connection_state or "OFFLINE"))
        except ValueError:
            state = CameraState.OFFLINE
        telemetry = camera_health_registry.get(camera.cam_id) or {}
        stream = stream_manager.stats(camera.cam_id) if camera.cam_id in cameraWorkers else {"queue_depth": 0, "queue_capacity": STREAM_BUFFER_SIZE, "dropped_frames": 0}
        clock = time_synchronizer.status(camera.cam_id)
        result.append({
            "cam_id": camera.cam_id,
            "camera_name": camera.camera_name,
            "status": state.value,
            "source_type": camera.source_type,
            "is_active": bool(camera.is_active),
            "last_health_check": camera.last_health_check.isoformat() if camera.last_health_check else None,
            "stream": {
                "fps": telemetry.get("fps"),
                "target_fps": telemetry.get("fps"),
                "latency_ms": telemetry.get("latency_ms"),
                "decoded_frames": telemetry.get("decoded_frames", 0),
                "processed_frames": telemetry.get("processed_frames", 0),
                "dropped_frames": telemetry.get("dropped_frames", 0),
                "queue_depth": stream.get("queue_depth", 0),
                "queue_capacity": stream.get("queue_capacity", STREAM_BUFFER_SIZE),
                "reconnects": telemetry.get("reconnects", 0),
                "last_frame_at": telemetry.get("last_frame_at"),
            },
            "time_sync": clock,
            "media_timeline": (
                cameraTimelines[camera.cam_id].status()
                if camera.cam_id in cameraTimelines
                else {
                    "camera_id": camera.cam_id,
                    "strict_pts": bool(STRICT_PTS_REQUIRED),
                    "discontinuity_count": int(clock.get("discontinuity_count") or 0),
                }
            ),
        })
    return result


def _set_camera_connection_state(
    cam_id: str,
    state: str | CameraState,
) -> None:
    """Set the single canonical camera state across memory, Redis and DB.

    The worker can call this without owning a request-scoped SQLAlchemy session.
    A short-lived session is used only for the state transition so the database
    remains the authoritative persistent state while Redis serves fast runtime
    reads.
    """
    try:
        normalized = normalize_camera_state(state)
    except ValueError:
        logger.warning("Invalid camera state | cam_id=%s state=%r", cam_id, state)
        normalized = CameraState.OFFLINE

    with state_lock:
        cameraConnectionStates[cam_id] = normalized.value
    try:
        camera_health_registry.set_state(cam_id, normalized)
    except Exception:
        logger.debug("Camera health registry state update failed | cam_id=%s", cam_id, exc_info=True)

    try:
        runtime = get_camera_state(cam_id) or {}
        set_camera_state(
            cam_id=cam_id,
            source=cam_id,
            source_type=runtime.get("source_type", "unknown"),
            state=normalized,
            frame_count=int(runtime.get("frame_count", 0) or 0),
        )
    except Exception:
        logger.debug("Redis camera state update failed | cam_id=%s", cam_id, exc_info=True)

    try:
        db = SessionLocal()
        try:
            camera = db.query(Camera).filter(Camera.cam_id == str(cam_id)).first()
            if camera:
                camera.connection_state = normalized.value
                camera.last_health_check = datetime.now(timezone.utc).replace(tzinfo=None)
                db.add(CameraHealth(
                    camera_id=str(cam_id),
                    state=normalized.value,
                    frame_count=int(cameraFrameCount.get(cam_id, 0) or 0),
                ))
                db.commit()
        finally:
            db.close()
    except Exception:
        logger.debug("Persistent camera state update failed | cam_id=%s", cam_id, exc_info=True)


def _reset_camera_scene_state(camID: str, reason: str = "stream_discontinuity") -> None:
    try:
        clear_tracker(camID)
    except Exception:
        logger.exception(
            "Person tracker reset failed | cam_id=%s reason=%s",
            camID,
            reason,
        )

    try:
        clear_vehicle_tracker(camID)
    except Exception:
        logger.exception(
            "Vehicle tracker reset failed | cam_id=%s reason=%s",
            camID,
            reason,
        )

    try:
        clear_camera_correlation_state(camID)
    except Exception:
        logger.exception("Vehicle correlation reset failed | cam_id=%s reason=%s", camID, reason)

    try:
        clear_camera_plate_states(camID)
    except Exception:
        logger.exception("ANPR temporal state reset failed | cam_id=%s reason=%s", camID, reason)

    try:
        person_correlation_engine.reset_camera(camID)
    except Exception:
        logger.exception("Person correlation reset failed | cam_id=%s reason=%s", camID, reason)

    try:
        advanced_intelligence.reset_camera(camID)
    except Exception:
        logger.exception("Camera visual baseline reset failed | cam_id=%s reason=%s", camID, reason)

    with state_lock:
        runtime_state.tracks.pop(camID, None)
        runtime_state.pairs.pop(camID, None)
        activeAlertBoxes.pop(camID, None)
        cameraFrameCount[camID] = 0
        cameraAdaptivePolicies.pop(camID, None)
        stream_manager.clear(camID)

        prefix = f"{camID}:"
        for key in list(personAppearanceStates):
            if str(key).startswith(prefix):
                personAppearanceStates.pop(key, None)
                personAppearanceLastFrame.pop(key, None)

        # Clear bounded per-track logging state for this camera.
        for key in list(vehicleAttributeLastLogFrame):
            if key.startswith(prefix):
                vehicleAttributeLastLogFrame.pop(key, None)
        for key in list(vehiclePlateLastLogFrame):
            if key.startswith(prefix):
                vehiclePlateLastLogFrame.pop(key, None)
        for key in list(vehiclePlateLastValue):
            if key.startswith(prefix):
                vehiclePlateLastValue.pop(key, None)

    reset_method = getattr(event_fusion, "reset_camera", None)

    if callable(reset_method):
        try:
            reset_method(camID)
        except Exception:
            logger.exception(
                "EventFusion reset failed | cam_id=%s reason=%s",
                camID,
                reason,
            )


def _run_watchlist_match_for_vehicle(
    vehicle_data,
    camID,
    source_type,
    snapshot_data=None,
):
    plate = (
        vehicle_data.get("stable_plate")
        or vehicle_data.get("plate")
    )

    if not plate:
        return

    db = SessionLocal()

    try:
        camera, user_id, owner_resolution = _resolve_alert_persistence_context(
            db=db,
            cam_id=str(camID),
            source_type=source_type,
        )

        if user_id is None:
            logger.error(
                "[ALERT-PERSIST] vehicle watchlist alert skipped: owner unresolved | "
                "cam_id=%s source_type=%s",
                camID,
                source_type,
            )
            return

        logger.debug(
            "[ALERT-PERSIST] vehicle watchlist owner resolved | "
            "cam_id=%s owner_id=%s source=%s",
            camID,
            user_id,
            owner_resolution,
        )

        confidence = (
            vehicle_data.get(
                "stable_plate_confidence"
            )
        )

        if confidence is None:
            confidence = (
                vehicle_data.get(
                    "plate_confidence",
                    0.0,
                )
            )

        try:
            confidence = max(
                0.0,
                min(
                    1.0,
                    float(confidence),
                ),
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        payload = create_exact_watchlist_alert(
            db,
            user_id=int(user_id),
            camera_id=str(camID),
            plate=str(plate),
            track_id=(
                str(vehicle_data.get("track_id"))
                if vehicle_data.get("track_id") is not None
                else None
            ),
            source_type=source_type,
            global_vehicle_id=vehicle_data.get(
                "global_vehicle_id"
            ),
            confidence=confidence,
            correlation_confidence=vehicle_data.get("correlation_score"),
            snapshot_data=snapshot_data,
            vehicle_attributes={
                "vehicle_type": vehicle_data.get("vehicle_type"),
                "vehicle_subtype": vehicle_data.get("vehicle_subtype"),
                "vehicle_detection_type": vehicle_data.get("vehicle_detection_type"),
                "color": vehicle_data.get("color"),
                "make": vehicle_data.get("make"),
                "model": vehicle_data.get("model"),
                "confidence": vehicle_data.get("attribute_confidence"),
                "attribute_model": vehicle_data.get("attribute_model"),
                "attribute_model_version": vehicle_data.get("attribute_model_version"),
            },
        )

        if not payload:
            logger.info(
                "[INTEL-I][WATCHLIST] NO_MATCH "
                "camera=%s track=%s plate_present=true confidence=%.3f",
                camID,
                vehicle_data.get("track_id"),
                confidence,
            )
            return

        # ================================================================
        # NORMALIZE WATCHLIST ALERT PAYLOAD
        # ================================================================

        payload = dict(payload)

        payload.setdefault(
            "type",
            "alert",
        )

        payload.setdefault(
            "event",
            "WATCHLIST_MATCH",
        )

        payload.setdefault(
            "alert_id",
            payload.get("id"),
        )

        payload.setdefault(
            "id",
            payload.get("alert_id"),
        )

        payload.setdefault(
            "user_id",
            int(user_id),
        )

        payload.setdefault(
            "cam_id",
            str(camID),
        )

        payload.setdefault(
            "camera_id",
            str(camID),
        )

        payload.setdefault(
            "source_type",
            source_type or "rtsp",
        )

        payload.setdefault(
            "plate",
            str(plate),
        )

        payload.setdefault(
            "track_id",
            (
                str(vehicle_data.get("track_id"))
                if vehicle_data.get("track_id") is not None
                else None
            ),
        )

        payload.setdefault(
            "confidence",
            confidence,
        )

        raw_level = (
            payload.get("severity")
            or payload.get("level")
            or payload.get("alert_level")
            or payload.get("priority")
        )

        if not raw_level:
            raw_level = "HIGH"

        normalized_level = normalize_alert_severity(
            raw_level
        )

        payload["severity"] = normalized_level

        payload.setdefault(
            "level",
            raw_level,
        )

        payload.setdefault(
            "alert_level",
            raw_level,
        )

        # ================================================================
        # WATCHLIST RULE
        # ================================================================

        payload.setdefault(
            "rule",
            "WATCHLIST_MATCH",
        )

        payload.setdefault(
            "alert_rule",
            payload.get("rule"),
        )

        # ================================================================
        # SNAPSHOT NORMALIZATION
        # ================================================================

        snapshot_id = (
            payload.get("snapshot_id")
        )

        if snapshot_id:
            try:
                snapshot_id = int(
                    snapshot_id
                )
                payload["snapshot_id"] = snapshot_id
            except (
                TypeError,
                ValueError,
            ):
                snapshot_id = None
                payload["snapshot_id"] = None

        payload["snapshot_saved"] = (
            snapshot_id is not None
        )

        payload["has_snapshot"] = (
            snapshot_id is not None
        )

        if snapshot_id:
            payload.setdefault(
                "snapshot_path",
                f"/snapshot/{snapshot_id}",
            )

        # ================================================================
        # WATCHLIST METADATA
        # ================================================================

        payload.setdefault(
            "watchlist_match_type",
            "EXACT",
        )

        payload.setdefault(
            "watchlist_status",
            "ACTIVE",
        )

        payload.setdefault(
            "message",
            (
                f"Exact watchlist match detected | "
                f"Camera: {camID} | "
                f"Plate: {plate}"
            ),
        )

        payload.setdefault(
            "created_at",
            time.time(),
        )

        # ================================================================
        # LOG
        # ================================================================

        logger.warning(
            "[INTEL-I][WATCHLIST] ALERT_CREATED "
            "alert_id=%s camera=%s track=%s plate_present=true "
            "severity=%s category=%s snapshot_id=%s",
            payload.get("alert_id"),
            camID,
            payload.get("track_id"),
            payload.get("severity"),
            payload.get("watchlist_category") or payload.get("category"),
            payload.get("snapshot_id"),
        )

        # ================================================================
        # LIVE WEBSOCKET
        # ================================================================

        durable_ws = os.getenv("KAFKA_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"} and os.getenv("KAFKA_SAVED_ALERT_WS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        if durable_ws:
            logger.info("Watchlist realtime delivery delegated to saved-alert Kafka consumer | alert_id=%s", payload.get("alert_id"))
        elif main_loop is None:
            logger.warning(
                "Watchlist WebSocket skipped: main loop unavailable | "
                "alert_id=%s",
                payload.get("alert_id"),
            )
        else:

            websocket_future = (
                asyncio.run_coroutine_threadsafe(
                    sendAlert(payload),
                    main_loop,
                )
            )

            def _watchlist_ws_done(future):
                try:
                    future.result()

                    logger.info(
                        "WATCHLIST_WEBSOCKET_SENT | "
                        "alert_id=%s | cam_id=%s",
                        payload.get("alert_id"),
                        camID,
                    )

                except Exception:
                    logger.exception(
                        "WATCHLIST_WEBSOCKET_FAILED | "
                        "alert_id=%s | cam_id=%s",
                        payload.get("alert_id"),
                        camID,
                    )

            websocket_future.add_done_callback(
                _watchlist_ws_done
            )

        # ================================================================
        # TELEGRAM
        # ================================================================
        #
        # IMPORTANT:
        #
        # This was missing from your current vehicle path.
        #
        # Do NOT call Telegram directly from the worker thread.
        # Schedule it on FastAPI's main asyncio event loop.
        # ================================================================

        if main_loop is None:
            logger.warning(
                "Watchlist Telegram skipped: main loop unavailable | "
                "alert_id=%s",
                payload.get("alert_id"),
            )

            return

        telegram_future = (
            asyncio.run_coroutine_threadsafe(
                sendTelegramAlert(payload),
                main_loop,
            )
        )

        def _watchlist_telegram_done(future):
            try:
                telegram_result = future.result()

                if telegram_result:

                    logger.warning(
                        "WATCHLIST_TELEGRAM_SENT | "
                        "alert_id=%s | "
                        "cam_id=%s | "
                        "plate_present=true | "
                        "snapshot_id=%s",
                        payload.get("alert_id"),
                        camID,
                        payload.get("snapshot_id"),
                    )

                else:

                    logger.error(
                        "WATCHLIST_TELEGRAM_NOT_SENT | "
                        "alert_id=%s | "
                        "cam_id=%s | "
                        "plate_present=true | "
                        "snapshot_id=%s",
                        payload.get("alert_id"),
                        camID,
                        payload.get("snapshot_id"),
                    )

            except Exception:
                logger.exception(
                    "WATCHLIST_TELEGRAM_FAILED | "
                    "alert_id=%s | cam_id=%s",
                    payload.get("alert_id"),
                    camID,
                )

        telegram_future.add_done_callback(
            _watchlist_telegram_done
        )

    except Exception:

        db.rollback()

        logger.exception(
            "Watchlist match processing failed | cam_id=%s",
            camID,
        )

    finally:
        db.close()


def _encode_behavior_snapshot(frame):
    try:
        ok, enc=cv2.imencode('.jpg',frame,[int(cv2.IMWRITE_JPEG_QUALITY),88]); return enc.tobytes() if ok else None
    except Exception:return None

def _dispatch_behavior_events(
    frame, camID, source_type, person_count, vehicle_count, ts,
    frame_timestamp=None, timestamp_source=None, timestamp_quality=None,
):
    events = evaluate_behavior(
        str(camID),
        int(person_count),
        int(vehicle_count),
        float(ts),
    )

    for event in events:
        db = SessionLocal()
        try:
            camera, user_id, owner_resolution = _resolve_alert_persistence_context(
                db=db,
                cam_id=str(camID),
                source_type=source_type,
            )
            if user_id is None:
                logger.error(
                    "[ALERT-PERSIST] behavioral alert dropped: owner unresolved | "
                    "cam_id=%s source_type=%s rule=%s",
                    camID,
                    source_type,
                    event.get("rule"),
                )
                continue

            level = normalize_alert_severity(event["level"])
            rule = event["rule"]

            obj, created = save_alert(
                db=db,
                user_id=int(user_id),
                alert_type=level,
                alert_rule=rule,
                track_id="behavioral",
                cam_id=str(camID),
                source_type=source_type,
                zone=None,
                level=level,
                cooldown_seconds=BEHAVIOR_ALERT_COOLDOWN,
                source_timestamp=frame_timestamp,
                timestamp_source=timestamp_source,
                timestamp_quality=timestamp_quality,
                confidence_score=float(event["confidence"]),
            )

            if not created or obj is None:
                logger.debug(
                    "[ALERT-PERSIST] behavioral alert suppressed by cooldown | "
                    "cam_id=%s owner_id=%s rule=%s",
                    camID,
                    user_id,
                    rule,
                )
                continue

            sid = None
            data = _encode_behavior_snapshot(frame)
            if data:
                snap = save_snapshot_to_db(
                    db=db,
                    alert_id=int(obj.id),
                    snapshot_data=data,
                )
                sid = int(snap.id) if snap else None

            payload = {
                "type": "alert",
                "event": "BEHAVIORAL_ALERT",
                "id": int(obj.id),
                "alert_id": int(obj.id),
                "user_id": int(user_id),
                "cam_id": str(camID),
                "camera_id": str(camID),
                "source_type": source_type,
                "alert_type": level,
                "alert_level": level,
                "level": level,
                "severity": level,
                "rule": rule,
                "alert_rule": rule,
                "event_type": event["event_type"],
                "person_count": int(person_count),
                "vehicle_count": int(vehicle_count),
                "confidence": float(event["confidence"]),
                "message": event["message"],
                "snapshot_id": sid,
                "snapshot_path": f"/snapshot/{sid}" if sid else None,
                "snapshot_saved": bool(sid),
                "has_snapshot": bool(sid),
                "created_at": time.time(),
            }

            logger.info(
                "[ALERT-PERSIST] behavioral alert saved | alert_id=%s cam_id=%s "
                "owner_id=%s owner_source=%s rule=%s",
                obj.id,
                camID,
                user_id,
                owner_resolution,
                rule,
            )

            if main_loop is not None:
                asyncio.run_coroutine_threadsafe(sendAlert(payload), main_loop)
                if level in {"HIGH", "MEDIUM"}:
                    asyncio.run_coroutine_threadsafe(
                        sendTelegramAlert(payload),
                        main_loop,
                    )
        except Exception:
            db.rollback()
            logger.exception(
                "Behavioral alert failed | cam_id=%s",
                camID,
            )
        finally:
            db.close()


def _face_box_iou(first_box, second_box):
    """Return IoU for two face boxes without retaining biometric data."""
    ax1, ay1, ax2, ay2 = map(float, first_box[:4])
    bx1, by1, bx2, by2 = map(float, second_box[:4])
    ix1=max(ax1,bx1); iy1=max(ay1,by1)
    ix2=min(ax2,bx2); iy2=min(ay2,by2)
    intersection=max(0.0,ix2-ix1)*max(0.0,iy2-iy1)
    first_area=max(0.0,ax2-ax1)*max(0.0,ay2-ay1)
    second_area=max(0.0,bx2-bx1)*max(0.0,by2-by1)
    union=first_area+second_area-intersection
    return intersection/union if union>0.0 else 0.0


def _extract_live_faces_from_person_regions(frame, tracked_persons, camID):
    """Detect faces in enlarged upper-body ROIs and retain true face boxes.

    The tracker supplies full-body boxes. Only the upper portion is sent to
    YuNet and modestly upscaled when needed. Enrollment-only extraction is
    deliberately not used in the live pipeline.
    """
    if frame is None or not isinstance(frame,np.ndarray) or frame.size==0:
        return []
    tracker_ids=getattr(tracked_persons,'tracker_id',None) if tracked_persons is not None else None
    boxes=getattr(tracked_persons,'xyxy',None) if tracked_persons is not None else None
    if tracker_ids is None or boxes is None:
        return []

    frame_height,frame_width=frame.shape[:2]
    collected=[]
    for person_index in range(len(tracked_persons)):
        try:
            raw_box=[float(value) for value in boxes[person_index][:4]]
            px1,py1,px2,py2=raw_box
            person_width=max(0.0,px2-px1); person_height=max(0.0,py2-py1)
            if person_width<16.0 or person_height<32.0:
                continue

            # Include the head and shoulders with horizontal/context padding.
            roi_x1=max(0,int(math.floor(px1-person_width*0.18)))
            roi_y1=max(0,int(math.floor(py1-person_height*0.10)))
            roi_x2=min(frame_width,int(math.ceil(px2+person_width*0.18)))
            roi_y2=min(frame_height,int(math.ceil(py1+person_height*0.62)))
            if roi_x2-roi_x1<24 or roi_y2-roi_y1<24:
                continue

            roi=frame[roi_y1:roi_y2,roi_x1:roi_x2]
            if roi.size==0:
                continue

            roi_height,roi_width=roi.shape[:2]
            longest=max(roi_width,roi_height)
            scale=min(2.0,max(1.0,640.0/max(1.0,float(longest))))
            inference_roi=roi
            if scale>1.05:
                inference_roi=cv2.resize(
                    roi,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )

            roi_faces=extract_embeddings(
                inference_roi,
                profile='live',
                allow_fallback=True,
            )
            for face in roi_faces:
                local_box=face.get('box')
                if not isinstance(local_box,(list,tuple)) or len(local_box)<4:
                    continue
                full_box=[
                    max(0,int(round(float(local_box[0])/scale))+roi_x1),
                    max(0,int(round(float(local_box[1])/scale))+roi_y1),
                    min(frame_width,int(round(float(local_box[2])/scale))+roi_x1),
                    min(frame_height,int(round(float(local_box[3])/scale))+roi_y1),
                ]
                if full_box[2]<=full_box[0] or full_box[3]<=full_box[1]:
                    continue
                translated=dict(face)
                translated['box']=full_box
                translated['source']='tracked-upper-body-roi'
                translated['tracker_id']=str(tracker_ids[person_index])

                duplicate_index=None
                for index,existing in enumerate(collected):
                    if _face_box_iou(existing['box'],full_box)>=0.50:
                        duplicate_index=index
                        break
                if duplicate_index is None:
                    collected.append(translated)
                elif float(translated.get('detection_score') or 0.0)>float(collected[duplicate_index].get('detection_score') or 0.0):
                    collected[duplicate_index]=translated
        except RuntimeError:
            # Model/configuration failures must reach the camera-level log.
            raise
        except Exception:
            logger.warning(
                'Person watchlist ROI extraction failed | cam_id=%s person_index=%s',
                camID,
                person_index,
                exc_info=True,
            )
    return collected


def _run_person_watchlist_match(
    frame, camID, source_type, tracked_persons=None, frame_timestamp=None,
    timestamp_source=None, timestamp_quality=None, snapshot_frame=None,
):
    db=SessionLocal()
    try:
        camera, user_id, owner_resolution = _resolve_alert_persistence_context(
            db=db,
            cam_id=str(camID),
            source_type=source_type,
        )
        if user_id is None:
            logger.error(
                '[ALERT-PERSIST] person watchlist skipped: owner unresolved | cam_id=%s source_type=%s',
                camID,
                source_type,
            )
            return

        camera_name = str(getattr(camera, 'camera_name', None) or camID)
        camera_latitude = getattr(camera, 'latitude', None) if camera is not None else None
        camera_longitude = getattr(camera, 'longitude', None) if camera is not None else None
        camera_location_name = getattr(camera, 'location_name', None) if camera is not None else None

        entries=db.query(PersonWatchlistEntry).filter(
            PersonWatchlistEntry.user_id==int(user_id),
            PersonWatchlistEntry.status=='ACTIVE',
        ).all()
        if not entries:return
        try:
            faces=_extract_live_faces_from_person_regions(
                frame,
                tracked_persons,
                camID,
            )
            if not faces:
                # Retain full-frame detection for untracked people and for
                # tracker boxes that do not contain a usable face.
                faces=extract_embeddings(
                    frame,
                    profile='live',
                    allow_fallback=True,
                )
        except RuntimeError:
            logger.exception(
                'Face recognition unavailable | cam_id=%s source_type=%s',
                camID,
                source_type,
            )
            return
        refs=[]
        for e in entries:
            try:refs.append((e,decrypt_embedding(e.embedding_encrypted,e.embedding_dimension)))
            except Exception:logger.exception('Person embedding unavailable | entry_id=%s',e.id)
        for face in faces:
            entry,score=match_embeddings(face['embedding'],refs)
            if entry is None:continue
            # Connect the face watchlist hit to the person correlation track
            # whose body box contains the face centre.
            global_person_id=None; correlated_track_id='face-watchlist'; correlation_score=0.0
            try:
                face_box=[float(value) for value in face['box'][:4]]
                face_cx=(face_box[0]+face_box[2])/2.0; face_cy=(face_box[1]+face_box[3])/2.0
                if tracked_persons is not None and tracked_persons.tracker_id is not None:
                    best_area=None
                    for person_index in range(len(tracked_persons)):
                        person_box=[float(value) for value in tracked_persons.xyxy[person_index][:4]]
                        if person_box[0] <= face_cx <= person_box[2] and person_box[1] <= face_cy <= person_box[3]:
                            area=max(1.0,(person_box[2]-person_box[0])*(person_box[3]-person_box[1]))
                            if best_area is None or area < best_area:
                                best_area=area
                                correlated_track_id=str(tracked_persons.tracker_id[person_index])
                    state=personAppearanceStates.get(f"{camID}:{correlated_track_id}",{})
                    global_person_id=state.get('global_person_id')
                    correlation_score=float(state.get('correlation_score') or 0.0)
            except Exception:
                logger.debug('Person watchlist correlation lookup failed | cam_id=%s',camID,exc_info=True)
            level='HIGH' if entry.category in {'MISSING','WANTED'} else 'LOW'
            rule='PERSON_WATCHLIST_MISSING' if entry.category=='MISSING' else 'PERSON_WATCHLIST_WANTED' if entry.category=='WANTED' else 'PERSON_WATCHLIST_MATCH'
            alert_track_id=f'person-watchlist-{int(entry.id)}'
            obj,created=save_alert(
                db,int(user_id),level,rule,alert_track_id,str(camID),source_type,None,level,120,
                source_timestamp=frame_timestamp, timestamp_source=timestamp_source,
                timestamp_quality=timestamp_quality,
                confidence_score=float(score),
            )
            if not created or obj is None:continue
            clean_snapshot_frame = snapshot_frame if snapshot_frame is not None else frame
            snap=save_snapshot_to_db(db,int(obj.id),crop_and_encode_snapshot(clean_snapshot_frame,face['box'])); sid=int(snap.id) if snap else None
            # The enrolled entry is the durable identity anchor. A transient
            # appearance GlobalPersonID can change after a restart or a long
            # gap, while the watchlist entry remains stable for months/years.
            identity_key=f'PERSON-WATCHLIST-{int(entry.id)}'
            journey_cutoff=(
                datetime.now(timezone.utc).replace(tzinfo=None)
                - timedelta(seconds=watchlist_journey_lookback_seconds())
            )
            inc=(db.query(Incident).filter(
                Incident.user_id==int(user_id),
                Incident.primary_track_id==identity_key,
                Incident.incident_type.like('PERSON_WATCHLIST%'),
                Incident.status.in_(['OPEN','UNDER_REVIEW','ESCALATED']),
                Incident.started_at>=journey_cutoff,
            ).order_by(Incident.started_at.desc()).first())
            previous_incident=None
            if inc is None:
                previous_incident=(db.query(Incident).filter(
                    Incident.user_id==int(user_id),
                    Incident.primary_track_id==identity_key,
                    Incident.incident_type.like('PERSON_WATCHLIST%'),
                    Incident.status.in_(['RESOLVED','CLOSED']),
                    Incident.started_at>=journey_cutoff,
                ).order_by(Incident.started_at.desc()).first())
            observed_at=(frame_timestamp.replace(tzinfo=None) if isinstance(frame_timestamp, datetime) else datetime.now(timezone.utc).replace(tzinfo=None))
            observation={'sequence':1,'camera_id':str(camID),'camera_name':camera_name,'latitude':camera_latitude,'longitude':camera_longitude,'location_name':camera_location_name,'source_type':source_type,'timestamp':observed_at.isoformat(),'track_id':correlated_track_id,'snapshot_id':sid,'confidence':float(score),'correlation_confidence':correlation_score}
            created_incident=inc is None
            if created_incident:
                previous_evidence=(previous_incident.evaluation_evidence if previous_incident is not None and isinstance(previous_incident.evaluation_evidence,dict) else {})
                previous_journey=dict(previous_evidence.get('watchlist_journey') or {})
                previous_route=list(previous_journey.get('route') or [])
                observation['sequence']=int(previous_journey.get('observation_count') or len(previous_route))+1
                combined_route=(previous_route+[observation])[-200:]
                related_incident_ids=list(previous_journey.get('related_incident_ids') or [])
                if previous_incident is not None:related_incident_ids.append(int(previous_incident.id))
                related_incident_ids=list(dict.fromkeys(related_incident_ids))[-100:]
                journey={'identity_type':'PERSON','identity_id':identity_key,'global_person_id':global_person_id,'person_watchlist_entry_id':int(entry.id),'watchlist_person_name':entry.full_name,'reference':entry.full_name,'reference_image_available':bool(entry.reference_image_data),'reference_image_url':f'/api/intelligence/person-watchlist/{int(entry.id)}/image' if entry.reference_image_data else None,'category':entry.category,'tracking_status':'ACTIVE_TRACKING' if len(combined_route)>1 else 'WAITING_FOR_NEXT_CAMERA','first_camera_id':previous_journey.get('first_camera_id') or str(camID),'last_camera_id':str(camID),'first_seen':previous_journey.get('first_seen') or observation['timestamp'],'last_seen':observation['timestamp'],'camera_count':len({str(item.get('camera_id')) for item in combined_route}),'observation_count':int(previous_journey.get('observation_count') or 0)+1,'previous_incident_id':int(previous_incident.id) if previous_incident is not None else None,'related_incident_ids':related_incident_ids,'route':combined_route}
                detection_location=camera_location_name or camera_name or str(camID)
                inc=Incident(user_id=int(user_id),cam_id=str(camID),primary_track_id=identity_key,incident_type=rule,status='OPEN',evaluation_status='PENDING',evaluation_severity=level,evaluation_confidence=float(score),evaluation_reason=f'Watchlist person {entry.full_name} detected in {detection_location}',evaluation_evidence={'watchlist_journey':journey})
                db.add(inc);db.flush()
            else:
                evidence_state=dict(inc.evaluation_evidence) if isinstance(inc.evaluation_evidence,dict) else {}
                journey=dict(evidence_state.get('watchlist_journey') or {});route=list(journey.get('route') or [])
                observation['sequence']=int(journey.get('observation_count') or len(route))+1;route.append(observation)
                journey.update({'identity_type':'PERSON','identity_id':identity_key,'global_person_id':global_person_id or journey.get('global_person_id'),'person_watchlist_entry_id':int(entry.id),'watchlist_person_name':entry.full_name,'reference':entry.full_name,'reference_image_available':bool(entry.reference_image_data),'reference_image_url':f'/api/intelligence/person-watchlist/{int(entry.id)}/image' if entry.reference_image_data else None,'category':entry.category,'tracking_status':'ACTIVE_TRACKING' if len(route)>1 else 'WAITING_FOR_NEXT_CAMERA','first_camera_id':journey.get('first_camera_id') or str(camID),'last_camera_id':str(camID),'first_seen':journey.get('first_seen') or observation['timestamp'],'last_seen':observation['timestamp'],'camera_count':len({str(item.get('camera_id')) for item in route}),'observation_count':int(journey.get('observation_count') or 0)+1,'route':route[-200:]})
                evidence_state['watchlist_journey']=journey;inc.evaluation_evidence=evidence_state;inc.cam_id=str(camID);inc.evaluation_confidence=max(float(inc.evaluation_confidence or 0.0),float(score))
            reference_snapshot_id=journey.get('reference_snapshot_id')
            try:reference_snapshot_id=int(reference_snapshot_id) if reference_snapshot_id is not None else None
            except (TypeError,ValueError):reference_snapshot_id=None
            if not reference_snapshot_id and entry.reference_image_data:
                try:
                    reference_bytes=decrypt_bytes(entry.reference_image_data)
                    reference_snapshot=save_snapshot_to_db(db,int(obj.id),{'image_data':reference_bytes,'image_type':entry.reference_image_content_type or 'image/jpeg'},metadata={'evidence_role':'WATCHLIST_REFERENCE','person_watchlist_entry_id':int(entry.id),'watchlist_person_name':entry.full_name,'watchlist_category':entry.category})
                    reference_snapshot_id=int(reference_snapshot.id) if reference_snapshot else None
                except Exception:
                    logger.exception('Unable to preserve watchlist reference image as incident evidence | entry_id=%s alert_id=%s',entry.id,obj.id)
            if reference_snapshot_id:
                journey['reference_snapshot_id']=int(reference_snapshot_id);journey['reference_snapshot_path']=f'/snapshot/{int(reference_snapshot_id)}'
                evidence_state=dict(inc.evaluation_evidence) if isinstance(inc.evaluation_evidence,dict) else {};evidence_state['watchlist_journey']=journey;inc.evaluation_evidence=evidence_state
            detection_location=camera_location_name or camera_name or str(camID)
            detection_message=f'Watchlist person {entry.full_name} detected in {detection_location}'
            db.add(IncidentEvidence(incident_id=int(inc.id),user_id=int(user_id),alert_id=int(obj.id),snapshot_id=sid,evidence_type='SNAPSHOT' if sid else 'ALERT',object_type='PERSON',object_reference=entry.full_name,description=detection_message,metadata_json={'match_score':float(score),'global_person_id':global_person_id,'person_watchlist_entry_id':int(entry.id),'watchlist_person_name':entry.full_name,'watchlist_category':entry.category,'evidence_role':'CAMERA_DETECTION','journey_observation':observation}))
            existing_reference_evidence=db.query(IncidentEvidence).filter(IncidentEvidence.incident_id==int(inc.id),IncidentEvidence.user_id==int(user_id),IncidentEvidence.evidence_type=='WATCHLIST_REFERENCE').first()
            if reference_snapshot_id and existing_reference_evidence is None:
                db.add(IncidentEvidence(incident_id=int(inc.id),user_id=int(user_id),alert_id=int(obj.id),snapshot_id=int(reference_snapshot_id),evidence_type='WATCHLIST_REFERENCE',object_type='PERSON',object_reference=entry.full_name,description=f'User-uploaded watchlist reference image for {entry.full_name}',metadata_json={'person_watchlist_entry_id':int(entry.id),'watchlist_person_name':entry.full_name,'watchlist_category':entry.category,'evidence_role':'WATCHLIST_REFERENCE'}))
            payload={'type':'alert','event':'PERSON_WATCHLIST_MATCH','id':int(obj.id),'alert_id':int(obj.id),'incident_id':int(inc.id),'incident_created':created_incident,'watchlist_journey':journey,'global_person_id':global_person_id,'user_id':int(user_id),'cam_id':str(camID),'camera_id':str(camID),'camera_name':camera_name,'source_type':source_type,'zone':camera_location_name,'location_name':camera_location_name,'track_id':alert_track_id,'alert_type':level,'alert_level':level,'level':level,'severity':level,'rule':rule,'alert_rule':rule,'person_watchlist_entry_id':int(entry.id),'watchlist_entry_id':int(entry.id),'person_name':entry.full_name,'watchlist_person_name':entry.full_name,'person_category':entry.category,'watchlist_category':entry.category,'match_type':'FACE_THRESHOLD','watchlist_match_type':'FACE_THRESHOLD','match_confidence':float(score),'watchlist_match_confidence':float(score),'reference_snapshot_id':int(reference_snapshot_id) if reference_snapshot_id else None,'reference_snapshot_path':f'/snapshot/{int(reference_snapshot_id)}' if reference_snapshot_id else None,'reference_image_url':f'/api/intelligence/person-watchlist/{int(entry.id)}/image' if entry.reference_image_data else None,'snapshot_id':sid,'snapshot_path':f'/snapshot/{sid}' if sid else None,'snapshot_saved':bool(sid),'has_snapshot':bool(sid),'message':detection_message,'created_at':time.time()}
            db.add(OutboxEvent(topic=os.getenv('KAFKA_SAVED_ALERT_TOPIC','cctv.alerts.saved'),event_key=str(obj.id),payload=json.dumps(payload,default=str),status='PENDING',attempts=0,next_attempt_at=indian_time()))
            db.commit()
            durable_ws = os.getenv('KAFKA_ENABLED','false').strip().lower() in {'1','true','yes','on'} and os.getenv('KAFKA_SAVED_ALERT_WS_ENABLED','true').strip().lower() in {'1','true','yes','on'}
            if main_loop is not None and not durable_ws:
                asyncio.run_coroutine_threadsafe(sendAlert(payload),main_loop)
            if main_loop is not None and level in {'HIGH','MEDIUM'}:
                asyncio.run_coroutine_threadsafe(sendTelegramAlert(payload),main_loop)
    except Exception:db.rollback();logger.exception('Person watchlist processing failed | cam_id=%s',camID)
    finally:db.close()

def _log_vehicle_attribute_result(
    camID,
    track_id,
    frame_count,
    result,
    source_type=None,
):
    """Log actual PP-LCNet output at a controlled cadence."""
    if track_id is None or not isinstance(result, dict):
        return

    try:
        current_frame = int(frame_count)
    except (TypeError, ValueError):
        current_frame = 0

    key = f"{camID}:{track_id}"

    with state_lock:
        last_frame = vehicleAttributeLastLogFrame.get(key)
        if (
            last_frame is not None
            and current_frame - last_frame < VEHICLE_ATTRIBUTE_LOG_EVERY_N_FRAMES
        ):
            return
        vehicleAttributeLastLogFrame[key] = current_frame

    vehicle_type = str(
        result.get("vehicle_detection_type")
        or result.get("vehicle_type")
        or "vehicle"
    ).strip()[:64]

    vehicle_subtype = str(
        result.get("vehicle_subtype")
        or "-"
    ).strip()[:64]

    color = str(
        result.get("color")
        or "unknown"
    ).strip()[:64]

    make = result.get("make")
    model = result.get("model")

    make_text = str(make).strip()[:100] if make not in (None, "") else "-"
    model_text = str(model).strip()[:100] if model not in (None, "") else "-"

    try:
        confidence = float(result.get("attribute_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    logger.info(
        "[INTEL-I][VEHICLE] camera=%s source=%s track=%s frame=%s "
        "type=%s subtype=%s color=%s make=%s model=%s "
        "attribute_confidence=%.3f model=%s version=%s",
        camID,
        str(source_type or "unknown"),
        track_id,
        current_frame,
        vehicle_type,
        vehicle_subtype,
        color,
        make_text,
        model_text,
        confidence,
        str(result.get("attribute_model") or "unknown"),
        str(result.get("attribute_model_version") or "unknown"),
    )

    # Immediate, human-readable console line for vehicle attributes.
    # This intentionally uses print(..., flush=True) so the information
    # is visible even when the Python logging configuration is not sending
    # INFO messages to the terminal.
    try:
        print(
            "[INTEL-I][VEHICLE] "
            f"TRACK={track_id} "
            f"TYPE={vehicle_type} "
            f"SUBTYPE={vehicle_subtype} "
            f"COLOR={color} "
            f"CONF={confidence:.3f} "
            f"CAMERA={camID}",
            flush=True,
        )
    except Exception:
        pass


def _log_vehicle_plate_result(
    camID,
    track_id,
    frame_count,
    plate_result,
):
    """Log a recognized plate when first seen/changed, then throttle repeats."""
    if not ANPR_TERMINAL_LOGGING_ENABLED:
        return

    if track_id is None or not isinstance(plate_result, dict):
        return

    plate = str(
        plate_result.get("plate")
        or plate_result.get("stable_plate")
        or ""
    ).strip().upper()

    if not plate:
        return

    try:
        current_frame = int(frame_count)
    except (TypeError, ValueError):
        current_frame = 0

    try:
        confidence = float(
            plate_result.get(
                "plate_confidence",
                plate_result.get("raw_plate_confidence", 0.0),
            ) or 0.0
        )
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    try:
        detection_confidence = float(
            plate_result.get("plate_detection_confidence", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        detection_confidence = 0.0
    detection_confidence = max(0.0, min(1.0, detection_confidence))

    status = str(
        plate_result.get("plate_status") or "UNKNOWN"
    ).upper()

    key = f"{camID}:{track_id}"
    with state_lock:
        previous_plate = vehiclePlateLastValue.get(key)
        last_frame = vehiclePlateLastLogFrame.get(key)

        should_log = (
            previous_plate != plate
            or last_frame is None
            or (
                last_frame is not None
                and current_frame - last_frame >= VEHICLE_ATTRIBUTE_LOG_EVERY_N_FRAMES
                and status == "STABLE"
            )
        )

        if not should_log:
            return

        vehiclePlateLastValue[key] = plate
        vehiclePlateLastLogFrame[key] = current_frame

    logger.info(
        "[INTEL-I][ANPR] camera=%s track=%s plate_present=true "
        "plate_confidence=%.3f detection_confidence=%.3f "
        "status=%s support=%s observations=%s",
        camID,
        track_id,
        confidence,
        detection_confidence,
        status,
        plate_result.get("plate_support_count", 0),
        plate_result.get("plate_observation_count", 0),
    )


def processFrame(
    frame,
    camID,
    frame_count=0,
    fps=None,
    source_type="rtsp",
    analytics_ts=None,
    frame_timestamp=None,
    timestamp_source="UNKNOWN",
    timestamp_quality="UNKNOWN",
    source_pts_seconds=None,
):
    """
    Main per-frame analytics pipeline.

    Pipeline:
        quality -> person tracking -> pose/crime/PPE -> EventFusion
        -> vehicle tracking -> Re-ID -> ANPR -> cross-camera correlation
        -> alert processing -> rendering

    Security/robustness:
        - Never logs decrypted camera URLs.
        - Per-model failures are isolated so one analytics stage cannot
          terminate the camera worker.
        - External/model data is validated before use.
        - Video overlays expose local person/vehicle tracker IDs.
        - Recognized license plates are rendered on their vehicle boxes.
        - Vehicle attribute text is intentionally not rendered on the video.
        - Runtime/CPU/GPU diagnostics are not printed from this function.
    """

    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return frame

    try:
        # Behavioural time is sourced from MediaTimeline/decoder PTS. Frame
        # counters are identifiers only and never become an event clock.
        try:
            current_frame_count = int(frame_count)
        except (TypeError, ValueError):
            current_frame_count = 0

        if current_frame_count < 0:
            current_frame_count = 0

        try:
            ts = float(analytics_ts)
        except (TypeError, ValueError):
            ts = float("nan")
        if not np.isfinite(ts) or ts < 0:
            logger.error("Frame rejected because PTS analytics time is unavailable | cam_id=%s", camID)
            return frame

        # Cross-camera identity uses the normalized evidence timestamp derived
        # from decoder PTS. It remains comparable across cameras and restarts.
        if isinstance(frame_timestamp, datetime):
            frame_timestamp_utc = (
                frame_timestamp.astimezone(timezone.utc)
                if frame_timestamp.tzinfo
                else frame_timestamp.replace(tzinfo=timezone.utc)
            )
        else:
            logger.error("Frame rejected because PTS evidence timestamp is unavailable | cam_id=%s", camID)
            return frame
        correlation_ts = frame_timestamp_utc.timestamp()

        # ------------------------------------------------------------
        # FRAME NORMALIZATION
        # ------------------------------------------------------------
        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

        frame_h, frame_w = frame.shape[:2]

        if frame_h <= 0 or frame_w <= 0:
            return frame

        # Preserve a clean, resized CCTV frame before any zones, quality text,
        # person boxes, vehicle boxes, ANPR text or alert overlays are drawn.
        # Every persisted/downloaded/Telegram evidence image is derived from
        # this copy; ``frame`` remains the annotated live-view frame.
        clean_evidence_frame = frame.copy()

        # ByteTrack receives PTS-observed/source FPS where possible. Behaviour
        # durations themselves use ``ts`` from PTS, never a frame counter.
        # Tracker FPS affects buffer sizing only. Event durations always use
        # ``ts`` above; the value here is decoder-declared or PTS-observed.
        tracker_fps = _safe_float(fps, 1.0, 0.1, 240.0)

        cleanAlertMemory()

        # ------------------------------------------------------------
        # CAMERA MODE / ZONES
        # ------------------------------------------------------------
        camera_mode = camera_config.get_mode(camID)
        is_camera_restricted = (
            str(camera_mode).strip().lower() == "restricted"
        )

        try:
            zone_manager.draw(frame, camID)
        except Exception:
            logger.exception(
                "Zone rendering failed | cam_id=%s",
                camID,
            )

        # ------------------------------------------------------------
        # CAMERA QUALITY + ADAPTIVE ENHANCEMENT
        # ------------------------------------------------------------
        try:
            should_check_quality = (
                current_frame_count % max(1, FRAME_QUALITY_CHECK_EVERY_N_FRAMES) == 0
            )
            if should_check_quality:
                quality = assess_frame_quality(frame)
                with cameraQualityLock:
                    cameraQualityStates[camID] = dict(quality)
            else:
                with cameraQualityLock:
                    quality = dict(cameraQualityStates.get(
                        camID,
                        {
                            "ok": True, "quality_score": 1.0,
                            "mode": "GOOD", "needs_enhancement": False,
                            "flags": [], "reason": "not_sampled", "metrics": {},
                        },
                    ))
        except Exception:
            logger.exception("Camera quality assessment failed | cam_id=%s", camID)
            quality = {
                "ok": True, "quality_score": 1.0, "mode": "UNKNOWN",
                "needs_enhancement": False, "flags": [],
                "reason": "quality_assessment_failed", "metrics": {},
            }

        enhancement_info = {"enhanced": False, "operations": []}

        # ------------------------------------------------------------
        # VIDEO INTELLIGENCE ORDER
        #
        # LOW LIGHT / VERY DARK:
        #     DarkIR -> YOLO
        #
        # NORMAL / OTHER QUALITY:
        #     existing classical quality enhancement -> YOLO
        #
        # IMPORTANT:
        #     EDSR is NOT applied to the full frame here.
        #     EDSR remains a post-detection ROI enhancement stage below.
        # ------------------------------------------------------------
        low_light_frame = (
            str(quality.get("mode") or "").strip().upper() == "LOW_LIGHT"
            or bool(
                {"low_light", "very_dark"}
                & {str(v).strip().lower() for v in quality.get("flags", [])}
            )
        )

        # Camera tamper/frozen health must inspect the ORIGINAL decoded frame,
        # before any classical enhancement or DarkIR transformation.
        if ADVANCED_INTELLIGENCE_ENABLED and (CAMERA_TAMPER_ENABLED or FROZEN_FRAME_ENABLED):
            try:
                health = advanced_intelligence.camera_health(camID, frame)
                cameraTamperTelemetry[camID] = health.get("tamper", {})
                cameraFrozenTelemetry[camID] = health.get("frozen", {})

                tamper = health.get("tamper", {})
                frozen = health.get("frozen", {})

                if tamper.get("tampered"):
                    _set_camera_connection_state(camID, "DEGRADED")
                    logger.warning(
                        "[INTEL-I][CAMERA_TAMPER] cam_id=%s reason=%s score=%.3f",
                        camID, tamper.get("reason"), float(tamper.get("score", 0.0)),
                    )
                if frozen.get("frozen"):
                    _set_camera_connection_state(camID, "DEGRADED")
                    logger.warning(
                        "[INTEL-I][CAMERA_FROZEN] cam_id=%s identical_frames=%s score=%.3f",
                        camID, frozen.get("consecutive_identical"), float(frozen.get("score", 0.0)),
                    )

            except Exception:
                logger.exception("Camera health intelligence failed | cam_id=%s", camID)

        if (
            FRAME_ENHANCEMENT_ENABLED
            and not (DARKIR_ENABLED and low_light_frame)
            and float(quality.get("quality_score", 1.0)) < FRAME_ENHANCEMENT_QUALITY_THRESHOLD
            and bool(quality.get("needs_enhancement", False))
        ):
            try:
                enhancement_info = adaptive_enhance(frame, quality)
                frame = enhancement_info.get("frame", frame)
            except Exception:
                logger.exception("Adaptive frame enhancement failed | cam_id=%s", camID)

        if ADVANCED_INTELLIGENCE_ENABLED:
            try:
                if DARKIR_ENABLED and low_light_frame:
                    if (
                        float(quality.get("quality_score", 1.0))
                        <= DARKIR_MIN_QUALITY
                    ):
                        result = advanced_intelligence.restore_low_light(
                            frame,
                            quality=quality,
                        )
                        if result.get("model_used"):
                            frame = result["image"]
                            enhancement_info.setdefault("operations", []).append("DarkIR")
                            enhancement_info["enhanced"] = True
                            enhancement_info["darkir"] = {
                                "device": result.get("device"),
                                "inference_ms": result.get("inference_ms"),
                                "resized_for_model": result.get("resized_for_model"),
                            }
                        elif DARKIR_STRICT:
                            raise RuntimeError(
                                "DarkIR was required for a low-light frame but could not be executed."
                            )
                    else:
                        logger.debug(
                            "[INTEL-I][DARKIR] quality score %.3f above threshold %.3f",
                            float(quality.get("quality_score", 1.0)),
                            DARKIR_MIN_QUALITY,
                        )
            except Exception:
                logger.exception("Advanced frame intelligence failed | cam_id=%s", camID)
                if DARKIR_STRICT and low_light_frame and DARKIR_ENABLED:
                    raise

        cameraQualityTelemetry[camID] = {
            **quality,
            "enhancement": enhancement_info,
            "tamper": cameraTamperTelemetry.get(camID),
            "frozen": cameraFrozenTelemetry.get(camID),
        }

        if FRAME_QUALITY_OVERLAY_ENABLED:
            try:
                text = f"Q:{float(quality.get('quality_score',1.0))*100:.0f}% | {str(quality.get('mode','UNKNOWN'))[:20]}"
                if enhancement_info.get("enhanced"):
                    text += " | ENHANCED"
                cv2.putText(frame, text, (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2, cv2.LINE_AA)
            except Exception:
                pass

        # ------------------------------------------------------------
        # PERSON + VEHICLE DETECTION
        # Use one YOLOv8m model for COCO person/vehicle classes.
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # PERSON + VEHICLE DETECTION
        #
        # Production mode uses a dedicated vehicle detector so vehicle
        # recall is not coupled to the person model's class filtering.
        # ``shared`` remains available for constrained GPUs.
        # ------------------------------------------------------------
        person_detections = sv.Detections.empty()
        vehicle_detections = sv.Detections.empty()

        # Keep the two detector paths isolated.  A CUDA/model error in the
        # vehicle path must never turn a valid person result into an empty
        # overlay (the former combined try block did exactly that).
        try:
            with model_lock:
                person_result = person_model(
                    frame, classes=[0], conf=PERSON_CONF,
                    device=DEVICE, verbose=False,
                )[0]
            person_detections = sv.Detections.from_ultralytics(person_result)
        except Exception:
            logger.exception("Person detection failed | cam_id=%s device=%s", camID, DEVICE)
            person_detections = sv.Detections.empty()

        try:
            if VEHICLE_DETECTOR_MODE == "shared":
                with model_lock:
                    # Compatibility/performance mode using the existing
                    # multi-class model.
                    shared_result = person_model(
                        frame,
                        classes=[2, 3, 5, 7],
                        conf=VEHICLE_CONF,
                        iou=VEHICLE_IOU,
                        device=DEVICE,
                        verbose=False,
                    )[0]
                    vehicle_detections = sv.Detections.from_ultralytics(
                        shared_result
                    )
            else:
                _vehicle_infer_started = time.perf_counter()
                vehicle_detections = vehicle_analytics_engine.detect(frame)
                vehicle_inference_seconds.observe(time.perf_counter() - _vehicle_infer_started)
                try:
                    class_names = getattr(vehicle_analytics_engine.model(), "names", {}) or {}
                    for _class_id in (vehicle_detections.class_id if vehicle_detections.class_id is not None else []):
                        vehicle_detections_total.labels(str(camID), str(class_names.get(int(_class_id), _class_id))[:64]).inc()
                except Exception:
                    pass
        except Exception:
            logger.exception("Vehicle detection failed | cam_id=%s mode=%s", camID, VEHICLE_DETECTOR_MODE)
            vehicle_detections = sv.Detections.empty()

        # ------------------------------------------------------------
        # PERSON BYTE TRACK
        # ------------------------------------------------------------
        try:
            tracked_persons = trackDetections(
                camID=camID,
                detections=person_detections,
                fps=tracker_fps,
            )
        except Exception:
            logger.exception(
                "Person tracking failed | cam_id=%s",
                camID,
            )
            tracked_persons = sv.Detections.empty()

        # ------------------------------------------------------------
        # VEHICLE BYTE TRACK
        # ------------------------------------------------------------
        try:
            tracked_vehicles = trackVehicleDetections(
                camID=camID,
                detections=vehicle_detections,
                fps=tracker_fps,
            )
            try:
                vehicle_tracking_active.labels(str(camID)).set(float(len(tracked_vehicles or [])))
            except Exception:
                pass
        except Exception:
            logger.exception(
                "Vehicle tracking failed | cam_id=%s",
                camID,
            )
            tracked_vehicles = sv.Detections.empty()

        # ------------------------------------------------------------
        # ADVANCED CROWD / APPEARANCE / ADAPTIVE OUTPUT
        # ------------------------------------------------------------
        if ADVANCED_INTELLIGENCE_ENABLED and ADAPTIVE_PIPELINE_ENABLED:
            try:
                activity_score = min(1.0, (len(tracked_persons) + len(tracked_vehicles)) / 12.0)
                policy = advanced_intelligence.adaptive_policy(
                    source_fps=float(tracker_fps),
                    activity_score=activity_score,
                    quality_score=float(quality.get("quality_score", 1.0)),
                    network_limited=False,
                )
                cameraAdaptivePolicies[camID] = policy
            except Exception:
                logger.exception("Adaptive pipeline policy failed | cam_id=%s", camID)

        if ADVANCED_INTELLIGENCE_ENABLED and CROWD_INTELLIGENCE_ENABLED:
            try:
                crowd = advanced_intelligence.crowd_update(
                    camera_id=str(camID),
                    count=len(tracked_persons),
                    frame_area=frame_w * frame_h,
                    timestamp_seconds=float(ts),
                )
                cameraQualityTelemetry[camID]["crowd"] = crowd
                if crowd.get("anomaly"):
                    logger.warning(
                        "[INTEL-I][CROWD] cam_id=%s count=%s growth=%.3f severity=%s",
                        camID, crowd.get("count"), float(crowd.get("growth_rate", 0.0)), crowd.get("severity"),
                    )
            except Exception:
                logger.exception("Crowd intelligence failed | cam_id=%s", camID)

        # ------------------------------------------------------------
        # CRIME / SUSPICIOUS DETECTION
        #
        # Run at the configured processing interval. Empty results are
        # valid and do not disable normal person tracking.
        # ------------------------------------------------------------
        crime_detections = []
        run_behavior_models = (
            current_frame_count % max(1, PROCESS_EVERY_N_FRAMES) == 0
        )

        if run_behavior_models:
            try:
                crime_detections = get_crime_detections(frame)
            except Exception:
                logger.exception(
                    "Suspicious activity detection failed | cam_id=%s",
                    camID,
                )
                crime_detections = []

        # ------------------------------------------------------------
        # PPE DETECTION
        # ------------------------------------------------------------
        ppe_detections = []

        if run_behavior_models:
            try:
                ppe_detections = get_ppe_detections(frame)
            except Exception:
                logger.exception(
                    "PPE detection failed | cam_id=%s",
                    camID,
                )
                ppe_detections = []

        # ------------------------------------------------------------
        # PERSON POSE
        # ------------------------------------------------------------
        pose_result = None

        if len(tracked_persons) > 0:
            try:
                with model_lock:
                    pose_result = pose_model(
                        frame,
                        conf=POSE_CONF,
                        device=DEVICE,
                        verbose=False,
                    )[0]
            except Exception:
                logger.exception(
                    "Pose detection failed | cam_id=%s",
                    camID,
                )
                pose_result = None

        # ------------------------------------------------------------
        # PERSON EVENT FUSION
        # ------------------------------------------------------------
        alerts_to_send = []
        current_track_ids = set()
        person_map = {}

        if (
            tracked_persons is not None
            and len(tracked_persons) > 0
            and tracked_persons.tracker_id is not None
        ):
            for index in range(len(tracked_persons)):
                raw_track_id = tracked_persons.tracker_id[index]

                if raw_track_id is None:
                    continue

                try:
                    current_track_id = int(raw_track_id)
                except (TypeError, ValueError):
                    continue

                current_track_ids.add(
                    str(current_track_id)
                )

                try:
                    person_box = tuple(
                        int(value)
                        for value in tracked_persons.xyxy[index]
                    )
                except (TypeError, ValueError):
                    continue

                x1, y1, x2, y2 = person_box

                if ADVANCED_INTELLIGENCE_ENABLED and PERSON_APPEARANCE_ENABLED:
                    try:
                        key = f"{camID}:{current_track_id}"
                        last = personAppearanceLastFrame.get(key, -999999)
                        if current_frame_count - last >= 15:
                            person_crop = frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
                            if person_crop.size:
                                personAppearanceStates[key] = appearance_features(person_crop)
                                personAppearanceLastFrame[key] = current_frame_count
                                try:
                                    person_confidence = (
                                        tracked_persons.confidence[index]
                                        if tracked_persons.confidence is not None
                                        else 0.0
                                    )
                                    person_identity = _correlate_and_persist_person(
                                        cam_id=camID,
                                        track_id=current_track_id,
                                        bbox=person_box,
                                        confidence=person_confidence,
                                        appearance=personAppearanceStates[key],
                                        frame_timestamp=correlation_ts,
                                    )
                                    if person_identity:
                                        personAppearanceStates[key]["global_person_id"] = person_identity["global_person_id"]
                                        personAppearanceStates[key]["correlation_score"] = person_identity["score"]
                                except Exception:
                                    logger.exception("Person correlation failed | cam_id=%s track=%s", camID, current_track_id)
                    except Exception:
                        logger.debug("Person appearance extraction failed | cam_id=%s track=%s", camID, current_track_id, exc_info=True)

                if not valid_person_box(
                    person_box,
                    frame_w,
                    frame_h,
                ):
                    continue

                center = center_of_box(person_box)

                refresh_active_alert_box(
                    camID,
                    current_track_id,
                    person_box,
                    ts,
                )

                # ----------------------------------------------------
                # EXISTING STATE / KEYPOINT FALLBACK
                # ----------------------------------------------------
                with state_lock:
                    existing_state = (
                        runtime_state.tracks
                        .get(camID, {})
                        .get(current_track_id)
                    )

                kpts = None

                if pose_result is not None:
                    kpts = match_pose_to_box(
                        pose_result,
                        person_box,
                    )

                if (
                    kpts is None
                    and existing_state is not None
                    and existing_state.keypoints
                ):
                    kpts = existing_state.keypoints[-1][1]

                has_pose = human_pose_valid(kpts)

                # ----------------------------------------------------
                # PPE / CRIME ASSOCIATION
                # ----------------------------------------------------
                associated_ppe = []

                for detection in ppe_detections:
                    detection_box = detection.get("box")

                    if detection_box is None:
                        continue

                    if crime_near_person(
                        detection_box,
                        person_box,
                    ):
                        associated_ppe.append(detection)

                associated_crime = []

                for detection in crime_detections:
                    detection_box = detection.get("box")

                    if detection_box is None:
                        continue

                    if crime_near_person(
                        detection_box,
                        person_box,
                    ):
                        associated_crime.append(detection)

                # ----------------------------------------------------
                # IMPORTANT:
                # Do NOT skip the person merely because pose is
                # unavailable. Crime/PPE rules must still be allowed
                # to reach EventFusion.
                # ----------------------------------------------------
                if (
                    not has_pose
                    and not associated_ppe
                    and not associated_crime
                ):
                    # Still draw and register the person for pair
                    # tracking only when there is enough state.
                    with state_lock:
                        track_state = runtime_state.update_track(
                            camID,
                            current_track_id,
                            person_box,
                            center,
                            kpts,
                            frame_w,
                            frame_h,
                            ts,
                        )

                    person_map[current_track_id] = {
                        "state": track_state,
                        "box": person_box,
                        "center": center,
                        "kpts": kpts,
                    }

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 165, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"person P{current_track_id}",
                        (
                            x1,
                            max(25, y1 - 8),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 165, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    continue

                # ----------------------------------------------------
                # RESTRICTED ZONE
                # ----------------------------------------------------
                try:
                    restricted, restricted_name = (
                        zone_manager.is_restricted_zone(
                            camID,
                            center,
                        )
                    )
                except Exception:
                    restricted = False
                    restricted_name = None

                # ----------------------------------------------------
                # RUNTIME STATE
                # ----------------------------------------------------
                with state_lock:
                    track_state = runtime_state.update_track(
                        camID,
                        current_track_id,
                        person_box,
                        center,
                        kpts,
                        frame_w,
                        frame_h,
                        ts,
                    )

                    person_alerts = event_fusion.evaluate_track(
                        camID,
                        current_track_id,
                        track_state,
                        person_box,
                        center,
                        kpts,
                        frame_w,
                        frame_h,
                        ts,
                        crime_detections,
                        restricted_zone=restricted,
                        restricted_zone_name=restricted_name,
                    )

                # ----------------------------------------------------
                # PERSON ALERT DECISION
                # ----------------------------------------------------
                for alert in person_alerts:
                    if not isinstance(alert, dict):
                        continue

                    rule = alert.get("rule")

                    if not rule:
                        continue

                    alert["track_id"] = current_track_id

                    alert_center = center_of_box(
                        alert.get("box", person_box)
                    )

                    try:
                        zone_result = zone_manager.validate(
                            camID,
                            rule,
                            alert_center,
                        )
                    except Exception:
                        logger.exception(
                            "Zone validation failed | cam_id=%s rule=%s",
                            camID,
                            rule,
                        )
                        continue

                    if not zone_result.get("allowed", False):
                        continue

                    level = RULE_LEVEL.get(
                        rule,
                        ALERT_LEVEL_WARNING,
                    )

                    cooldown = RULE_COOLDOWN_SECONDS.get(
                        rule,
                        30,
                    )

                    with state_lock:
                        can_send = runtime_state.can_alert(
                            track_state,
                            rule,
                            cooldown,
                            ts,
                        )

                    if not can_send:
                        continue

                    alert["level"] = level
                    alert["zone"] = zone_result

                    if is_camera_restricted:
                        alert["camera_mode"] = "restricted"

                        if alert["level"] == ALERT_LEVEL_LOW:
                            alert["level"] = ALERT_LEVEL_WARNING
                        elif alert["level"] == ALERT_LEVEL_WARNING:
                            alert["level"] = ALERT_LEVEL_CRITICAL
                    else:
                        alert["camera_mode"] = "not_restricted"

                    allowed, _ = final_alert_allowed(
                        alert,
                        zone_result,
                    )

                    if not allowed:
                        continue

                    alerts_to_send.append(alert)

                # ----------------------------------------------------
                # PERSON OVERLAY
                # ----------------------------------------------------
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 165, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"person P{current_track_id}",
                    (
                        x1,
                        max(25, y1 - 8),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 165, 0),
                    2,
                    cv2.LINE_AA,
                )

                person_map[current_track_id] = {
                    "state": track_state,
                    "box": person_box,
                    "center": center,
                    "kpts": kpts,
                }

        # ------------------------------------------------------------
        # PERSON-TO-PERSON EVENT FUSION
        # ------------------------------------------------------------
        ids = list(person_map.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1 = ids[i]
                id2 = ids[j]

                try:
                    with state_lock:
                        pair_alerts = event_fusion.evaluate_pair(
                            camID,
                            id1,
                            person_map[id1],
                            id2,
                            person_map[id2],
                            frame_w,
                            frame_h,
                            ts,
                        )

                        _, pair_state = runtime_state.get_pair(
                            camID,
                            id1,
                            id2,
                            ts,
                        )
                except Exception:
                    logger.exception(
                        "Pair EventFusion failed | cam_id=%s pair=%s_%s",
                        camID,
                        id1,
                        id2,
                    )
                    continue

                for alert in pair_alerts:
                    if not isinstance(alert, dict):
                        continue

                    rule = alert.get("rule")

                    if not rule:
                        continue

                    alert["track_id"] = f"{id1}_{id2}"

                    alert_box = alert.get("box")

                    if alert_box is None:
                        continue

                    alert_center = center_of_box(alert_box)

                    try:
                        zone_result = zone_manager.validate(
                            camID,
                            rule,
                            alert_center,
                        )
                    except Exception:
                        logger.exception(
                            "Pair zone validation failed | cam_id=%s rule=%s",
                            camID,
                            rule,
                        )
                        continue

                    if not zone_result.get("allowed", False):
                        continue

                    level = RULE_LEVEL.get(
                        rule,
                        ALERT_LEVEL_WARNING,
                    )

                    cooldown = RULE_COOLDOWN_SECONDS.get(
                        rule,
                        30,
                    )

                    with state_lock:
                        can_send = runtime_state.can_alert(
                            pair_state,
                            rule,
                            cooldown,
                            ts,
                        )

                    if not can_send:
                        continue

                    alert["level"] = level
                    alert["zone"] = zone_result

                    if is_camera_restricted:
                        alert["camera_mode"] = "restricted"

                        if alert["level"] == ALERT_LEVEL_LOW:
                            alert["level"] = ALERT_LEVEL_WARNING
                        elif alert["level"] == ALERT_LEVEL_WARNING:
                            alert["level"] = ALERT_LEVEL_CRITICAL
                    else:
                        alert["camera_mode"] = "not_restricted"

                    allowed, _ = final_alert_allowed(
                        alert,
                        zone_result,
                    )

                    if not allowed:
                        continue

                    alerts_to_send.append(alert)

        # ------------------------------------------------------------
        # VEHICLE TRACK EXTRACTION
        # ------------------------------------------------------------
        vehicle_tracks = []

        try:
            vehicle_tracks = extractVehicleTracks(
                camID=camID,
                detections=tracked_vehicles,
                timestamp=ts,
            )
        except Exception:
            logger.exception(
                "Vehicle track extraction failed | cam_id=%s",
                camID,
            )
            vehicle_tracks = []

        _dispatch_behavior_events(
            clean_evidence_frame, camID, source_type,
            len(tracked_persons) if tracked_persons is not None else 0,
            len(tracked_vehicles) if tracked_vehicles is not None else 0,
            ts,
            frame_timestamp=frame_timestamp_utc.replace(tzinfo=None),
            timestamp_source=timestamp_source,
            timestamp_quality=timestamp_quality,
        )
        if current_frame_count % 15 == 0:
            _run_person_watchlist_match(
                frame, camID, source_type, tracked_persons,
                frame_timestamp=frame_timestamp_utc.replace(tzinfo=None),
                timestamp_source=timestamp_source,
                timestamp_quality=timestamp_quality,
                snapshot_frame=clean_evidence_frame,
            )

        # ------------------------------------------------------------
        # VEHICLE RE-ID / ANPR / CORRELATION
        # ------------------------------------------------------------
        should_run_vehicle_identity = (
            current_frame_count % VEHICLE_REID_RUN_EVERY_N_FRAMES == 0
        )

        for vehicle in vehicle_tracks:
            track_id = vehicle.get("track_id")
            bbox = vehicle.get("bbox")

            if track_id is None or bbox is None:
                continue

            crop = None

            try:
                crop = crop_vehicle(
                    frame,
                    bbox,
                    padding=10,
                )
            except Exception:
                logger.exception(
                    "Vehicle crop failed | cam_id=%s track=%s",
                    camID,
                    track_id,
                )

            if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
                continue

            # Keep original vehicle pixels as canonical evidence and as the
            # default Re-ID input. EDSR is a task-specific ROI enhancement,
            # never a blanket replacement for all downstream models.
            original_vehicle_crop = crop
            reid_crop = original_vehicle_crop
            anpr_crop = original_vehicle_crop
            try:
                roi_quality = assess_frame_quality(original_vehicle_crop)
            except Exception:
                roi_quality = {"quality_score": 1.0, "mode": "UNKNOWN"}

            roi_quality_score = _safe_float(roi_quality.get("quality_score"), 1.0, 0.0, 1.0)
            roi_h, roi_w = original_vehicle_crop.shape[:2]
            # Super-resolution is useful mainly for small/poor target ROIs.
            # Do not spend GPU on already-good, large vehicle crops.
            edsr_candidate = (
                ADVANCED_INTELLIGENCE_ENABLED
                and EDSR_ENABLED
                and (roi_quality_score < EDSR_MIN_QUALITY or roi_w < 420 or roi_h < 220)
            )
            if edsr_candidate:
                try:
                    sr = advanced_intelligence.upscale_when_needed(original_vehicle_crop, roi_quality_score)
                    if sr.get("model_used") and isinstance(sr.get("image"), np.ndarray) and sr["image"].size:
                        anpr_crop = sr["image"]
                except Exception:
                    logger.debug("Vehicle ANPR EDSR enhancement failed | cam_id=%s track=%s", camID, track_id, exc_info=True)

            embedding = None

            # --------------------------------------------------------
            # RE-ID
            # --------------------------------------------------------
            reid_quality = 0.0
            if should_run_vehicle_identity:
                try:
                    _reid_started = time.perf_counter()
                    vehicle_reid_attempts_total.inc()
                    reid_quality_probe = get_vehicle_embedding_result(reid_crop)
                    reid_inference_seconds.observe(time.perf_counter() - _reid_started)
                    reid_quality = _safe_float(
                        reid_quality_probe.get("crop_quality"),
                        0.0, 0.0, 1.0
                    ) if isinstance(reid_quality_probe, dict) else 0.0

                    if reid_quality < VEHICLE_REID_MIN_CROP_QUALITY:
                        embedding = None
                    else:
                        embedding = (
                            reid_quality_probe.get("embedding")
                            if isinstance(reid_quality_probe, dict)
                            else None
                        )
                except Exception:
                    embedding = None
                    reid_quality = 0.0

            # --------------------------------------------------------
            # ANPR
            #
            # Plate is stored internally and logged only when detected.
            # It is NOT rendered on the video.
            # --------------------------------------------------------
            plate_result = {
                "plate": None,
                "plate_confidence": 0.0,
                "plate_detection_confidence": 0.0,
            }
            anpr_frame_timestamp = None

            if should_run_vehicle_identity:
                try:
                    # ANPR receives the real camera + local ByteTrack
                    # identity so persistence and audit records remain
                    # associated with the correct vehicle track.
                    #
                    # The global vehicle ID is intentionally omitted here:
                    # it is not known until the correlation engine evaluates
                    # the Re-ID/plate/temporal evidence.
                    anpr_frame_timestamp = frame_timestamp_utc.replace(tzinfo=None)

                    _anpr_started = time.perf_counter()
                    vehicle_anpr_attempts_total.inc()
                    result = detect_and_read_plate(
                        anpr_crop,
                        camera_id=str(camID),
                        local_track_id=str(track_id),
                        global_vehicle_id=None,
                        frame_timestamp=anpr_frame_timestamp,
                        metadata={
                            "analytics_timestamp": float(ts),
                            "correlation_timestamp": float(correlation_ts),
                            "source_type": str(source_type or "unknown"),
                            "vehicle_type": str(
                                vehicle.get("vehicle_type") or "vehicle"
                            )[:64],
                            "vehicle_detection_confidence": _safe_float(
                                vehicle.get("confidence"),
                                0.0,
                                0.0,
                                1.0,
                            ),
                            "track_context": "bytetrack_local_vehicle_track",
                        },
                    )

                    anpr_inference_seconds.observe(time.perf_counter() - _anpr_started)

                    if isinstance(result, dict):
                        plate_result.update(result)

                    _log_vehicle_plate_result(
                        camID=camID,
                        track_id=track_id,
                        frame_count=current_frame_count,
                        plate_result=plate_result,
                    )

                    plate_text = plate_result.get("plate")

                    if plate_text:
                        try:
                            plate_conf = float(
                                plate_result.get(
                                    "plate_confidence",
                                    0.0,
                                )
                            )
                        except (TypeError, ValueError):
                            plate_conf = 0.0

                        try:
                            det_conf = float(
                                plate_result.get(
                                    "plate_detection_confidence",
                                    0.0,
                                )
                            )
                        except (TypeError, ValueError):
                            det_conf = 0.0

                        # Never place plaintext plate values in application
                        # logs. The plate remains inside the controlled
                        # intelligence pipeline/database.
                        if ANPR_TERMINAL_LOGGING_ENABLED:
                            logger.info(
                                "[ANPR] camera=%s track=%s detected "
                                "ocr_conf=%.3f det_conf=%.3f",
                                camID,
                                track_id,
                                max(0.0, min(1.0, plate_conf)),
                                max(0.0, min(1.0, det_conf)),
                            )
                except Exception:
                    vehicle_pipeline_errors_total.labels(stage="anpr").inc()
                    # ANPR failure must never terminate the camera worker.
                    if ANPR_TERMINAL_LOGGING_ENABLED:
                        logger.exception(
                            "[ANPR] processing failed | camera=%s track=%s",
                            camID,
                            track_id,
                        )

            vehicle_data = {
                **vehicle,
                "camera_id": camID,
                "track_id": track_id,
                "timestamp": correlation_ts,
                "analytics_timestamp": ts,
                "frame_timestamp": frame_timestamp_utc.replace(tzinfo=None),
                "source_pts_seconds": source_pts_seconds,
                "timestamp_source": timestamp_source,
                "timestamp_quality": timestamp_quality,
                "embedding": embedding,
                "reid_quality": reid_quality,
                "embedding_quality": (1.0 if embedding is not None else 0.0),
                "crop": crop,
                "plate": plate_result.get("plate"),
                "plate_confidence": plate_result.get(
                    "plate_confidence",
                    0.0,
                ),
                "plate_detection_confidence": plate_result.get(
                    "plate_detection_confidence",
                    0.0,
                ),
                # Explicit evidence-level fields. These remain separate
                # from the eventual cross-camera identity confidence.
                "plate_evidence_available": bool(
                    plate_result.get("plate")
                ),
                "plate_persistence": plate_result.get(
                    "persistence"
                ),
                "anpr_observation_timestamp": (
                    anpr_frame_timestamp
                    if should_run_vehicle_identity
                    else None
                ),
                "frame_quality_score": float(quality.get("quality_score", 1.0)),
                "frame_quality_mode": str(quality.get("mode") or "UNKNOWN"),
                "roi_quality_score": roi_quality_score,
                "roi_quality_mode": str(roi_quality.get("mode") or "UNKNOWN"),
                "edsr_used_for_anpr": anpr_crop is not original_vehicle_crop,
                "original_roi_size": [int(roi_w), int(roi_h)],
                "anpr_roi_size": [int(anpr_crop.shape[1]), int(anpr_crop.shape[0])],
                "camera_tampered": bool(cameraTamperTelemetry.get(camID, {}).get("tampered", False)),
                "stream_frozen": bool(cameraFrozenTelemetry.get(camID, {}).get("frozen", False)),
                "observation_monotonic": time.monotonic(),
                "observation_validation_status": "PENDING",
            }

            # --------------------------------------------------------
            # VEHICLE ATTRIBUTES — IDENTITY EVIDENCE BEFORE VALIDATION
            # --------------------------------------------------------
            try:
                attribute_crop = crop
                if FRAME_ENHANCEMENT_ENABLED:
                    try:
                        attribute_crop = enhance_vehicle_crop(crop)
                    except Exception:
                        logger.exception(
                            "Vehicle crop enhancement failed | cam_id=%s track=%s",
                            camID,
                            track_id,
                        )
                        attribute_crop = crop

                attribute_result = infer_vehicle_attributes(
                    attribute_crop,
                    vehicle.get("vehicle_type"),
                )
                if isinstance(attribute_result, dict):
                    detector_type = str(
                        vehicle.get("vehicle_type")
                        or "vehicle"
                    ).strip()

                    attribute_result["vehicle_detection_type"] = detector_type

                    predicted_subtype = (
                        attribute_result.get("vehicle_subtype")
                        or (
                            attribute_result.get("vehicle_type")
                            if attribute_result.get("vehicle_type") != detector_type
                            else None
                        )
                    )

                    if predicted_subtype:
                        attribute_result["vehicle_subtype"] = str(
                            predicted_subtype
                        ).strip()

                    # Keep the detector's coarse type as vehicle_type.
                    attribute_result["vehicle_type"] = detector_type

                    vehicle_data.update(attribute_result)

                    _log_vehicle_attribute_result(
                        camID=camID,
                        track_id=track_id,
                        frame_count=current_frame_count,
                        result=attribute_result,
                        source_type=source_type,
                    )
                else:
                    logger.warning(
                        "[VEHICLE-ATTR] unexpected result type | cam_id=%s track=%s type=%s",
                        camID,
                        track_id,
                        type(attribute_result).__name__,
                    )
            except Exception:
                vehicle_pipeline_errors_total.labels(stage="attributes").inc()
                logger.exception(
                    "Vehicle attribute inference failed | cam_id=%s track=%s source_type=%s",
                    camID,
                    track_id,
                    source_type,
                )


            # --------------------------------------------------------
            # PHASE 9 — OBSERVATION VALIDATION GATE (COMPLETE OBSERVATION)
            # Detection/tracking output is an observation, not identity truth.
            # Correlation and watchlist decisions are allowed only after the
            # observation passes structural, quality and freshness checks.
            # --------------------------------------------------------
            try:
                validation = validate_vehicle_observation(
                    vehicle_data,
                    frame_width=frame_w,
                    frame_height=frame_h,
                )
                vehicle_data["observation_validation_status"] = (
                    "ACCEPTED" if validation.accepted else "REJECTED"
                )
                vehicle_data["observation_validation_confidence"] = validation.confidence
                vehicle_data["observation_validation_quality"] = validation.quality
                vehicle_data["observation_validation_reasons"] = list(validation.reasons)
                vehicle_data["observation_validation_warnings"] = list(validation.warnings)
                if validation.accepted:
                    vehicle_observations_accepted_total.inc()
                else:
                    for reason in (validation.reasons or ("UNKNOWN",)):
                        vehicle_observations_rejected_total.labels(reason=str(reason)[:64]).inc()
            except Exception:
                logger.exception(
                    "Vehicle observation validation failed | cam_id=%s track=%s",
                    camID, track_id,
                )
                vehicle_data["observation_validation_status"] = "REJECTED"
                vehicle_data["observation_validation_reasons"] = ["VALIDATION_EXCEPTION"]
                vehicle_data["observation_validation_warnings"] = []

            observation_accepted = (
                vehicle_data.get("observation_validation_status") == "ACCEPTED"
            )


            # --------------------------------------------------------
            # CROSS-CAMERA CORRELATION
            #
            # Correlation is evaluated before local state persistence so
            # update_vehicle_state() receives the complete observation,
            # including any global identity and correlation evidence.
            # --------------------------------------------------------
            correlation = None

            if observation_accepted and embedding is not None:
                try:
                    _correlation_started = time.perf_counter()
                    correlation = correlate_vehicle(
                        vehicle_data
                    )
                    vehicle_correlation_seconds.observe(time.perf_counter() - _correlation_started)
                except Exception:
                    vehicle_pipeline_errors_total.labels(stage="correlation").inc()
                    logger.exception(
                        "Vehicle correlation failed | cam_id=%s track=%s",
                        camID,
                        track_id,
                    )

            if isinstance(correlation, dict):
                vehicle_data["global_vehicle_id"] = (
                    correlation.get("global_vehicle_id")
                )
                vehicle_data["correlation_similarity"] = (
                    correlation.get("similarity", 0.0)
                )
                vehicle_data["correlation_type"] = (
                    correlation.get("match_type")
                )
                vehicle_data["correlation_score"] = (
                    correlation.get("correlation_score", 0.0)
                )
                try:
                    vehicle_correlation_score.set(
                        _safe_float(correlation.get("correlation_score"), 0.0, 0.0, 1.0)
                    )
                except Exception:
                    pass
                vehicle_data["correlation_decision"] = (
                    correlation.get(
                        "decision",
                        correlation.get("match_type"),
                    )
                )
                vehicle_data["correlation_strong_match"] = bool(
                    correlation.get("strong_match", False)
                )
                vehicle_data["correlation_supporting_frames"] = (
                    correlation.get("supporting_frames", 0)
                )
                vehicle_data["correlation_consistency"] = (
                    correlation.get("consistency", 0.0)
                )

                # Surface established plate evidence separately from the
                # raw OCR observation whenever the correlation engine
                # exposes it. This preserves the confidence ladder:
                # OCR != vehicle identity confidence.
                established_plate = None
                global_id = correlation.get("global_vehicle_id")

                if global_id and get_vehicle_plate_evidence is not None:
                    try:
                        established_plate = get_vehicle_plate_evidence(
                            str(global_id)
                        )
                    except Exception:
                        logger.debug(
                            "Established plate evidence lookup failed | "
                            "global_id=%s",
                            global_id,
                            exc_info=True,
                        )

                if isinstance(established_plate, dict):
                    vehicle_data["stable_plate"] = (
                        established_plate.get("plate")
                    )
                    vehicle_data["stable_plate_confidence"] = (
                        established_plate.get(
                            "plate_confidence",
                            0.0,
                        )
                    )
                    vehicle_data["stable_plate_support_count"] = (
                        established_plate.get(
                            "plate_support_count",
                            0,
                        )
                    )
                    vehicle_data["plate_match_strength"] = (
                        established_plate.get(
                            "plate_match_strength"
                        )
                    )

                gis_observation = _record_correlation_gis_observation(
                    vehicle_data,
                    correlation,
                )
                if isinstance(gis_observation, dict):
                    vehicle_data["gis_latitude"] = gis_observation.get("latitude")
                    vehicle_data["gis_longitude"] = gis_observation.get("longitude")
                    vehicle_data["gis_position_type"] = gis_observation.get("position_type")
                    vehicle_data["gis_pixel_x"] = gis_observation.get("pixel_x")
                    vehicle_data["gis_pixel_y"] = gis_observation.get("pixel_y")

                # --------------------------------------------------------
                # PERSIST GLOBAL ID + CORRELATION DECISION
                # --------------------------------------------------------
                global_vehicle_id = vehicle_data.get(
                    "global_vehicle_id"
                )

                if global_vehicle_id:
                    _persist_global_vehicle_safe(global_vehicle_id=global_vehicle_id,vehicle_type=vehicle_data.get("vehicle_type"),color=vehicle_data.get("color"),make=vehicle_data.get("make"),model=vehicle_data.get("model"))

                _persist_correlation_decision_safe(
                    vehicle_data=vehicle_data,
                    correlation=correlation,
                )

            # --------------------------------------------------------
            # CONTINUOUS LIVE GIS UPDATE
            #
            # Re-ID/ANPR are intentionally throttled. Once a local track has
            # already been assigned a Global Vehicle ID, every valid tracked
            # frame can update its calibrated map position without rerunning
            # the expensive identity models. This is what makes the GIS
            # marker move in real time rather than every ANPR interval.
            # --------------------------------------------------------
            if correlation is None and get_global_vehicle_id is not None:
                try:
                    existing_global_id = get_global_vehicle_id(
                        str(camID), str(track_id)
                    )
                    if existing_global_id:
                        existing_global = (
                            get_global_vehicle(str(existing_global_id))
                            if get_global_vehicle is not None
                            else None
                        )
                        live_correlation = {
                            "global_vehicle_id": str(existing_global_id),
                            "similarity": _safe_float(
                                (existing_global or {}).get("last_similarity"),
                                0.0, 0.0, 1.0,
                            ),
                            "correlation_score": _safe_float(
                                (existing_global or {}).get("last_correlation_score"),
                                0.0, 0.0, 1.0,
                            ),
                            "match_type": "existing_local_track_live",
                            "strong_match": bool(
                                (existing_global or {}).get("confidence_status")
                                in {"CONFIRMED", "PROBABLE"}
                            ),
                            "supporting_frames": max(
                                0, int((existing_global or {}).get("correlation_count", 0) or 0)
                            ),
                            "consistency": 1.0,
                            "correlation_evidence": {
                                "available_branches": ["vehicle_reid", "metadata"],
                                "evidence_strength": "LIVE_TRACK_CONTINUITY",
                            },
                        }
                        vehicle_data["global_vehicle_id"] = str(existing_global_id)
                        live_gis_observation = _record_correlation_gis_observation(
                            vehicle_data,
                            live_correlation,
                        )
                        if isinstance(live_gis_observation, dict):
                            vehicle_data["gis_latitude"] = live_gis_observation.get("latitude")
                            vehicle_data["gis_longitude"] = live_gis_observation.get("longitude")
                            vehicle_data["gis_position_type"] = live_gis_observation.get("position_type")
                            vehicle_data["gis_pixel_x"] = live_gis_observation.get("pixel_x")
                            vehicle_data["gis_pixel_y"] = live_gis_observation.get("pixel_y")
                except Exception:
                    logger.debug(
                        "Continuous vehicle GIS update failed | cam_id=%s track=%s",
                        camID, track_id, exc_info=True
                    )

            # --------------------------------------------------------
            # ANPR FUSION METADATA
            # --------------------------------------------------------
            # Character voting + source-aware PP-OCR/temporal/Gemma fusion
            # now execute inside vehicleANPR.detect_and_read_plate(), before
            # validation/correlation. Surface those canonical results here.
            fused = plate_result.get("plate_fusion") if isinstance(plate_result, dict) else None
            if isinstance(fused, dict):
                vehicle_data["plate_fusion"] = fused
                if fused.get("plate") and fused.get("decision") in {"CONFIRMED", "PROBABLE"}:
                    vehicle_data["stable_plate"] = fused.get("plate")
                    vehicle_data["stable_plate_confidence"] = fused.get("confidence", 0.0)
                    vehicle_data["plate_confidence_status"] = fused.get("decision")
            if plate_result.get("character_voted_plate"):
                vehicle_data["character_voted_plate"] = plate_result.get("character_voted_plate")
                vehicle_data["character_vote_confidence"] = plate_result.get("character_vote_confidence", 0.0)
            if plate_result.get("ollama_plate_verification"):
                vehicle_data["ollama_plate_verification"] = plate_result.get("ollama_plate_verification")
            if CONFIDENCE_CALIBRATION_ENABLED:
                try:
                    support_count = _safe_float(plate_result.get("plate_support_count", 0), 0.0, 0.0, 100.0)
                    obs_count = _safe_float(plate_result.get("plate_observation_count", 0), 0.0, 0.0, 100.0)
                    temporal_support = min(1.0, support_count / max(1.0, obs_count if obs_count > 0 else 5.0))
                    vehicle_data["calibrated_plate_confidence"] = advanced_intelligence.calibrator.calibrate(
                        _safe_float(vehicle_data.get("stable_plate_confidence", vehicle_data.get("plate_confidence", 0.0)), 0.0, 0.0, 1.0),
                        quality_score=float(quality.get("quality_score", 1.0)),
                        temporal_support=temporal_support,
                    )
                except Exception:
                    logger.exception("ANPR confidence calibration failed | cam_id=%s track=%s", camID, track_id)

            # Exact watchlist matching happens after stable plate evidence
            # and correlation are available. Possible/fuzzy matches are not
            # promoted to automatic alerts.
            # Build the evidence frame only for a vehicle that has stable
            # plate evidence. The watchlist service will persist it only
            # after an exact watchlist match is confirmed.
            watchlist_snapshot_data = None
            if observation_accepted and (
                vehicle_data.get("stable_plate")
                or vehicle_data.get("plate")
            ) and bbox is not None:
                try:
                    watchlist_snapshot_data = crop_and_encode_snapshot(
                        clean_evidence_frame,
                        bbox,
                    )
                except Exception:
                    logger.exception(
                        "Watchlist evidence snapshot failed | cam_id=%s track=%s",
                        camID,
                        track_id,
                    )

            if observation_accepted:
                _run_watchlist_match_for_vehicle(
                    vehicle_data,
                    camID,
                    source_type,
                    snapshot_data=watchlist_snapshot_data,
                )
            try:
                update_vehicle_state(vehicle_data,correlation)
            except Exception:
                logger.exception(
                    "Vehicle state update failed | cam_id=%s track=%s",
                    camID,
                    track_id,
                )

            if should_run_vehicle_identity and observation_accepted:
                _persist_vehicle_observation_safe(
                    vehicle_data=vehicle_data,
                )

            # --------------------------------------------------------
            # VEHICLE + ANPR OVERLAY
            #
            # Draw the tracked vehicle box and local track ID. When ANPR has a
            # result, render the plate and confidence on the same box. Vehicle
            # attributes (subtype/color/make/model) remain backend evidence and
            # are intentionally excluded from the detection screen.
            # --------------------------------------------------------
            if VEHICLE_DETECTION_OVERLAY_ENABLED:
                try:
                    x1, y1, x2, y2 = [
                        int(value)
                        for value in bbox
                    ]

                    x1 = max(0, min(x1, frame_w - 1))
                    y1 = max(0, min(y1, frame_h - 1))
                    x2 = max(0, min(x2, frame_w - 1))
                    y2 = max(0, min(y2, frame_h - 1))

                    if x2 > x1 and y2 > y1:
                        vehicle_type = str(
                            vehicle_data.get("vehicle_detection_type")
                            or vehicle_data.get("vehicle_type")
                            or vehicle.get("vehicle_type")
                            or "vehicle"
                        ).strip()[:24]

                        overlay_lines = [
                            f"vehicle V{int(track_id)} | {vehicle_type}",
                        ]

                        plate_text = str(
                            vehicle_data.get("plate") or ""
                        ).strip().upper()[:24]

                        if ANPR_OVERLAY_ENABLED and plate_text:
                            plate_confidence = _safe_float(
                                vehicle_data.get("plate_confidence"),
                                0.0,
                                0.0,
                                1.0,
                            )
                            overlay_lines.append(
                                f"ANPR: {plate_text} ({plate_confidence:.0%})"
                            )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2,
                        )

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.52
                        thickness = 2
                        line_height = 22
                        padding = 6

                        text_sizes = [
                            cv2.getTextSize(
                                line,
                                font,
                                font_scale,
                                thickness,
                            )[0]
                            for line in overlay_lines
                        ]

                        panel_width = (
                            max(width for width, _ in text_sizes)
                            + padding * 2
                        )
                        panel_height = (
                            len(overlay_lines) * line_height
                            + padding * 2
                        )

                        panel_x1 = x1
                        panel_y2 = y1 - 4
                        panel_y1 = panel_y2 - panel_height

                        if panel_y1 < 0:
                            panel_y1 = y2 + 4
                            panel_y2 = panel_y1 + panel_height

                        if panel_x1 + panel_width > frame_w:
                            panel_x1 = max(0, frame_w - panel_width)

                        if panel_y2 > frame_h:
                            panel_y2 = frame_h
                            panel_y1 = max(0, panel_y2 - panel_height)

                        panel_x2 = min(
                            frame_w - 1,
                            panel_x1 + panel_width,
                        )

                        cv2.rectangle(
                            frame,
                            (int(panel_x1), int(panel_y1)),
                            (int(panel_x2), int(panel_y2)),
                            (20, 20, 20),
                            -1,
                        )

                        cv2.rectangle(
                            frame,
                            (int(panel_x1), int(panel_y1)),
                            (int(panel_x2), int(panel_y2)),
                            (0, 255, 0),
                            1,
                        )

                        text_x = int(panel_x1 + padding)
                        text_y = int(panel_y1 + padding + 16)

                        for index, line in enumerate(overlay_lines):
                            cv2.putText(
                                frame,
                                line,
                                (
                                    text_x,
                                    text_y + index * line_height,
                                ),
                                font,
                                font_scale,
                                (255, 255, 255),
                                thickness,
                                cv2.LINE_AA,
                            )

                except Exception:
                    logger.exception(
                        "Vehicle overlay failed | cam_id=%s track=%s",
                        camID,
                        track_id,
                    )

        # ------------------------------------------------------------
        # GLOBAL VEHICLE CLEANUP
        # ------------------------------------------------------------
        try:
            cleanup_global_vehicles(correlation_ts)
        except TypeError:
            try:
                cleanup_global_vehicles()
            except Exception:
                pass
        except Exception:
            logger.exception(
                "Vehicle correlation cleanup failed | cam_id=%s",
                camID,
            )

        # ------------------------------------------------------------
        # ALERT BOXES
        # ------------------------------------------------------------
        for alert in alerts_to_send:
            alert.setdefault("source_timestamp", frame_timestamp_utc.replace(tzinfo=None))
            alert.setdefault("source_pts_seconds", source_pts_seconds)
            alert.setdefault("timestamp_source", timestamp_source)
            alert.setdefault("timestamp_quality", timestamp_quality)
            remember_alert_box(
                camID=camID,
                track_id=(
                    alert.get("track_id")
                    or alert.get("track")
                ),
                rule=alert.get("rule"),
                box=alert.get("box"),
                level=alert.get("level"),
                confidence=alert.get("confidence", 0.0),
                ts=ts,
            )

        cleanup_alert_boxes(
            camID,
            current_track_ids,
        )

        if ALERT_OVERLAY_ENABLED:
            frame = draw_active_alert_boxes(
                frame,
                camID,
            )

        # Active high-risk behavioral incidents sample only the clean ROI at
        # the configured low rate. The selector retains one best face frame
        # and one best context fallback, never an eight-second raw-frame queue.
        if BEHAVIOR_BEST_EVIDENCE_ENABLED:
            try:
                _sample_active_behavior_evidence(
                    clean_evidence_frame,
                    camID,
                )
            except Exception:
                logger.exception(
                    "Behavior evidence sampling failed | cam_id=%s",
                    camID,
                )

        # ------------------------------------------------------------
        # ASYNC ALERT DELIVERY
        # ------------------------------------------------------------
        submit_alert_task(
            clean_evidence_frame,
            camID,
            alerts_to_send,
            source_type,
        )

        # ------------------------------------------------------------
        # RUNTIME STATE CLEANUP
        # ------------------------------------------------------------
        with state_lock:
            runtime_state.cleanup(
                camID,
                ts,
            )

        return frame

    except Exception:
        # Never allow one malformed frame to kill the camera worker.
        logger.exception(
            "Frame processing failed | cam_id=%s",
            camID,
        )
        return frame

def _scheduled_ai_handler(job: AIJob):
    """Run one complete AI analytics job inside the shared Phase 8 pool."""
    started = time.monotonic()
    result = processFrame(
        frame=job.frame,
        camID=job.camera_id,
        frame_count=job.frame_number,
        fps=job.source_fps,
        source_type=job.source_type,
        analytics_ts=job.analytics_ts,
        frame_timestamp=job.frame_timestamp,
        timestamp_source=job.timestamp_source,
        timestamp_quality=job.timestamp_quality,
        source_pts_seconds=job.source_pts_seconds,
    )
    try:
        vehicle_inference_seconds.observe(time.monotonic() - started)
    except Exception:
        pass
    return result


def _sync_ai_scheduler_metrics() -> dict:
    state = ai_scheduler.status()
    ai_queue_depth.set(state["queue_depth"])
    ai_workers_active.set(state["active_jobs"])
    gpu = state.get("gpu") or {}
    if gpu.get("utilization_percent") is not None:
        ai_gpu_utilization_percent.set(gpu["utilization_percent"])
    if gpu.get("memory_percent") is not None:
        ai_gpu_memory_percent.set(gpu["memory_percent"])
    return state


def camera_worker(
    src,
    camID,
    source_type="rtsp",
):
    cap = None
    reconnect_delay = RTSP_RECONNECT_INITIAL_SECONDS
    first_open = True
    next_output_deadline = time.monotonic()
    strict_pts = bool(STRICT_PTS_REQUIRED)
    timeline = timeline_from_environment(str(camID), strict_pts=strict_pts)
    with state_lock:
        cameraTimelines[str(camID)] = timeline

    try:
        with state_lock:
            stop_event = cameraStopEvents.get(camID)

        if stop_event is None:
            logger.warning(
                "No stop event found for camera worker | cam_id=%s",
                camID,
            )
            update_camera_status(camID, False, state=CameraState.OFFLINE)
            return

        while not stop_event.is_set():
            if not is_camera_running(camID):
                break

            # --------------------------------------------------------
            # OPEN / REOPEN SOURCE
            # --------------------------------------------------------
            if cap is None:
                try:
                    cap = open_timestamped_capture(
                        src,
                        source_type,
                        opencv_factory=_open_video_capture,
                        timeout_seconds=RTSP_READ_TIMEOUT_SECONDS,
                    )
                except Exception:
                    camera_decoder_failures_total.labels(
                        source_type=source_type,
                    ).inc()
                    camera_health_registry.decoder_failure(camID, message="decoder initialization failed")
                    _set_camera_connection_state(camID, "RECONNECTING")
                    logger.exception(
                        "Camera decoder initialization failed | cam_id=%s source_type=%s",
                        camID,
                        source_type,
                    )
                    cap = None

                if cap is None:
                    camera_decoder_failures_total.labels(
                        source_type=source_type,
                    ).inc()
                    if source_type == "upload":
                        logger.warning(
                            "Upload video could not be opened | cam_id=%s",
                            camID,
                        )
                        break

                    logger.warning(
                        "Camera source unavailable; reconnect scheduled | cam_id=%s source_type=%s retry_seconds=%.2f",
                        camID,
                        source_type,
                        reconnect_delay,
                    )

                    stop_event.wait(reconnect_delay)
                    reconnect_delay = min(
                        RTSP_RECONNECT_MAX_SECONDS,
                        reconnect_delay * 2.0,
                    )
                    continue
                if not first_open:
                    timeline.force_discontinuity("SUCCESSFUL_RECONNECT")

                first_open = False
                reconnect_delay = RTSP_RECONNECT_INITIAL_SECONDS
                next_output_deadline = time.monotonic()

                update_camera_status(
                    camID,
                    True,
                )
                _set_camera_connection_state(camID, "ONLINE")

                logger.info(
                    "Camera decoder connected | cam_id=%s source_type=%s transport=%s decoder=%s",
                    camID,
                    source_type,
                    "tcp" if source_type == "rtsp" else "n/a",
                    (getattr(cap, "metadata", {}) or {}).get("decoder", "unknown"),
                )
            read_started = time.monotonic()
            decoded = cap.read_timestamped()
            ok, frame = decoded.ok, decoded.frame
            read_elapsed = time.monotonic() - read_started

            if not ok or frame is None:
                if source_type == "upload":
                    logger.info(
                        "Upload video finished | cam_id=%s",
                        camID,
                    )
                    break

                camera_read_failures_total.labels(
                    source_type=source_type,
                ).inc()
                camera_health_registry.read_failure(camID, error_code="READ_FAILED", message="camera frame read failed")
                _set_camera_connection_state(camID, "DEGRADED")

                logger.warning(
                    "Camera frame read failed; decoder will reconnect | cam_id=%s source_type=%s read_seconds=%.2f",
                    camID,
                    source_type,
                    read_elapsed,
                )

                try:
                    cap.release()
                except Exception:
                    pass

                cap = None
                _set_camera_connection_state(camID, "RECONNECTING")
                camera_reconnect_total.labels(
                    source_type=source_type,
                ).inc()
                camera_health_registry.reconnect(camID)

                stop_event.wait(reconnect_delay)

                reconnect_delay = min(
                    RTSP_RECONNECT_MAX_SECONDS,
                    max(
                        RTSP_RECONNECT_INITIAL_SECONDS,
                        reconnect_delay * 2.0,
                    ),
                )
                continue
            if read_elapsed >= STREAM_GAP_THRESHOLD_SECONDS:
                _set_camera_connection_state(camID, "DEGRADED")
                logger.warning(
                    "Camera inter-frame gap exceeded threshold | cam_id=%s source_type=%s gap_seconds=%.2f threshold_seconds=%.2f",
                    camID,
                    source_type,
                    read_elapsed,
                    STREAM_GAP_THRESHOLD_SECONDS,
                )
                timeline.force_discontinuity("INTER_FRAME_GAP")

            reconnect_delay = RTSP_RECONNECT_INITIAL_SECONDS
            _set_camera_connection_state(camID, "ONLINE")
            received_at = datetime.now(timezone.utc)
            received_monotonic = time.monotonic()
            try:
                timing = timeline.observe(
                    pts_seconds=decoded.pts_seconds,
                    received_at=received_at,
                    received_monotonic=received_monotonic,
                    frame=frame,
                )
            except SourcePTSRequiredError:
                camera_health_registry.decoder_failure(
                    camID,
                    message="decoder did not provide presentation timestamps",
                )
                _set_camera_connection_state(camID, "DEGRADED")
                logger.error(
                    "Strict PTS policy rejected a frame | cam_id=%s source_type=%s",
                    camID,
                    source_type,
                )
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                if source_type == "upload":
                    break
                stop_event.wait(reconnect_delay)
                continue

            if timing.discontinuity:
                logger.warning(
                    "Media discontinuity detected; camera-local state reset | cam_id=%s reason=%s pts=%s scene_score=%.3f",
                    camID,
                    timing.discontinuity_reason,
                    timing.pts_seconds,
                    timing.scene_change_score,
                )
                _reset_camera_scene_state(camID, reason=timing.discontinuity_reason or "media_discontinuity")

            packet = FramePacket(
                camera_id=str(camID),
                frame=frame,
                frame_number=int(cameraFrameCount.get(camID, 0)),
                received_at=received_at,
                received_monotonic=received_monotonic,
                source_timestamp=timing.normalized_timestamp,
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                pts_seconds=timing.pts_seconds,
                time_base=decoded.time_base,
                timestamp_source=timing.timestamp_source,
                timestamp_quality=timing.timestamp_quality,
                analytics_seconds=timing.analytics_seconds,
                discontinuity=timing.discontinuity,
                discontinuity_reason=timing.discontinuity_reason,
                key_frame=decoded.key_frame,
            )
            dropped_before = stream_manager.stats(camID).get("dropped_frames", 0)
            stream_manager.ingest(packet)
            buffered = stream_manager.next_frame(camID, timeout=0.05)
            if buffered is None:
                camera_health_registry.drop(camID)
                continue
            frame = buffered.frame
            camera_health_registry.frame(
                camID,
                read_latency_ms=read_elapsed * 1000.0,
                queue_depth=stream_manager.stats(camID).get("queue_depth", 0),
            )
            dropped_after = stream_manager.stats(camID).get("dropped_frames", 0)
            if dropped_after > dropped_before:
                camera_health_registry.drop(camID, dropped_after - dropped_before)

            ts_bundle = time_synchronizer.observe(
                camID,
                camera_timestamp=buffered.source_timestamp,
                server_timestamp=buffered.received_at,
                pts_seconds=buffered.pts_seconds,
                timestamp_source=buffered.timestamp_source,
                timestamp_quality=buffered.timestamp_quality,
                discontinuity=buffered.discontinuity,
                discontinuity_reason=buffered.discontinuity_reason,
            )
            _persist_camera_time_sync(camID, ts_bundle)
            _persist_stream_metric_if_due(camID)
            with state_lock:
                frame_count = cameraFrameCount.get(
                    camID,
                    0,
                )

            analytics_ts = float(buffered.analytics_seconds)
            decoder_metadata = getattr(cap, "metadata", {}) or {}
            source_fps = timing.observed_fps or decoder_metadata.get("fps")

            # --------------------------------------------------------
            # PHASE 8: SHARED AI SCHEDULER
            # --------------------------------------------------------
            future = ai_scheduler.submit(
                camera_id=str(camID),
                frame=frame,
                frame_number=frame_count,
                analytics_ts=analytics_ts,
                source_type=source_type,
                frame_timestamp=buffered.source_timestamp,
                source_fps=source_fps,
                source_pts_seconds=buffered.pts_seconds,
                timestamp_source=buffered.timestamp_source,
                timestamp_quality=buffered.timestamp_quality,
                priority=int(AIPriority.NORMAL),
            )
            ai_jobs_submitted_total.inc()
            _sync_ai_scheduler_metrics()
            try:
                processed = future.result(timeout=AI_JOB_TIMEOUT_SECONDS + 1.0)
            except TimeoutError:
                # ``TimeoutError`` stringifies to an empty string.  Report it
                # accurately and keep the camera worker alive; an older JPEG
                # remains available until the next completed annotated frame.
                ai_jobs_failed_total.inc()
                _sync_ai_scheduler_metrics()
                logger.warning(
                    "AI job timed out; frame skipped | cam_id=%s timeout_seconds=%.1f",
                    camID,
                    AI_JOB_TIMEOUT_SECONDS + 1.0,
                )
                continue
            except Exception as exc:
                message = str(exc).strip()[:180] or type(exc).__name__
                if "AI job dropped:" in message:
                    reason = message.split("AI job dropped:", 1)[1].strip() or "unknown"
                    ai_jobs_dropped_total.labels(reason=reason).inc()
                    camera_health_registry.drop(camID)
                else:
                    ai_jobs_failed_total.inc()
                _sync_ai_scheduler_metrics()
                logger.warning("AI job unavailable; frame skipped | cam_id=%s reason=%s", camID, message)
                continue
            ai_jobs_completed_total.inc()
            _sync_ai_scheduler_metrics()
            adaptive_policy = cameraAdaptivePolicies.get(camID, {})
            output_every = max(1, int(adaptive_policy.get("process_every_n_frames", 1))) if ADAPTIVE_OUTPUT_ENABLED else 1

            # A live camera must always publish its latest annotated frame.
            # Output thinning here caused the frontend to retain a stale,
            # unannotated JPEG and made working detections look invisible.
            jpeg_quality = int(adaptive_policy.get("jpeg_quality", 85))
            success, buffer = cv2.imencode(
                ".jpg", processed,
                [int(cv2.IMWRITE_JPEG_QUALITY), max(40, min(95, jpeg_quality))],
            )

            if success:
                with state_lock:
                    lock = cameraLocks.get(camID)

                if lock:
                    with lock:
                        latestJpegs[camID] = buffer.tobytes()
            increment_camera_frame(camID)

            with state_lock:
                cameraFrameCount[camID] = (
                    cameraFrameCount.get(camID, 0) + 1
                )
            if source_type == "upload":
                next_output_deadline += (
                    1.0 / float(OUTPUT_STREAM_FPS)
                )

                wait_seconds = (
                    next_output_deadline
                    - time.monotonic()
                )

                if wait_seconds > 0:
                    stop_event.wait(wait_seconds)
                else:
                    next_output_deadline = time.monotonic()

    except Exception:
        logger.exception(
            "Camera worker crashed | cam_id=%s source_type=%s",
            camID,
            source_type,
        )

    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                logger.exception(
                    "Camera decoder release failed | cam_id=%s",
                    camID,
                )
        _set_camera_connection_state(
            camID,
            CameraState.OFFLINE,
        )

        update_camera_status(
            camID,
            False,
            state=CameraState.OFFLINE,
            worker_running=False,
        )

        stream_id = None

        with state_lock:
            if source_type == "upload":
                stream_id = uploadCamToStream.pop(
                    camID,
                    None,
                )

                if stream_id:
                    uploadvideos.pop(
                        stream_id,
                        None,
                    )

                    uploadStreamToCam.pop(
                        stream_id,
                        None,
                    )

            cameraIDS.pop(
                camID,
                None,
            )

            cameraWorkers.pop(
                camID,
                None,
            )

            cameraStopEvents.pop(
                camID,
                None,
            )

            cameraFrameCount.pop(
                camID,
                None,
            )

            cameraConnectionStates.pop(
                camID,
                None,
            )

            cameraTimelines.pop(str(camID), None)

            activeAlertBoxes.pop(
                camID,
                None,
            )

            lock = cameraLocks.pop(
                camID,
                None,
            )

            if lock:
                with lock:
                    latestJpegs.pop(
                        camID,
                        None,
                    )
            else:
                latestJpegs.pop(
                    camID,
                    None,
                )

        if source_type == "upload":
            delete_upload_state(
                stream_id=stream_id,
                cam_id=camID,
            )

        camera_config.remove(
            camID
        )

        logger.info(
            "Camera worker stopped and cleaned up | cam_id=%s source_type=%s",
            camID,
            source_type,
        )

def start_camera_worker(
    camID,
    src,
    source_type,
):
    """
    Start one worker for a camera or uploaded video.

    Worker lifecycle:
        worker_running=True

    Connection health:
        ONLINE / DEGRADED / RECONNECTING / OFFLINE

    These states are intentionally independent.
    """

    camID = str(camID or "").strip()
    source_type = str(
        source_type or ""
    ).strip().lower()

    if not camID:
        raise ValueError(
            "Camera ID is required"
        )

    if src is None:
        raise ValueError(
            "Camera source is required"
        )

    # --------------------------------------------------------
    # 1. Prevent duplicate workers
    # --------------------------------------------------------

    with state_lock:
        existing_worker = cameraWorkers.get(
            camID
        )

        if (
            existing_worker
            and not existing_worker.done()
        ):
            logger.info(
                "Camera worker already running | "
                "cam_id=%s source_type=%s",
                camID,
                source_type,
            )
            return

        # ----------------------------------------------------
        # 2. Create fresh synchronization state
        # ----------------------------------------------------

        cameraLocks.setdefault(
            camID,
            threading.Lock(),
        )

        cameraStopEvents[camID] = (
            threading.Event()
        )

        cameraFrameCount[camID] = 0

    # --------------------------------------------------------
    # 3. Prepare stream buffer
    # --------------------------------------------------------

    stream_manager.ensure(
        camID,
        STREAM_BUFFER_SIZE,
    )

    # --------------------------------------------------------
    # 4. Prepare camera health registry
    # --------------------------------------------------------

    camera_health_registry.ensure(
        camID,
        queue_capacity=STREAM_BUFFER_SIZE,
    )

    # --------------------------------------------------------
    # 5. IMPORTANT:
    #    Explicitly mark NEW worker as running.
    # --------------------------------------------------------

    set_camera_state(
        cam_id=camID,
        source=camID,
        source_type=source_type,
        status=True,
        state=CameraState.ONLINE,
        frame_count=0,
        worker_running=True,
    )

    # --------------------------------------------------------
    # 6. Submit worker
    # --------------------------------------------------------

    with state_lock:
        # Protect against a race where another request started
        # a worker between our initial check and submit.
        existing_worker = cameraWorkers.get(
            camID
        )

        if (
            existing_worker
            and not existing_worker.done()
        ):
            logger.info(
                "Camera worker became active during startup | "
                "cam_id=%s",
                camID,
            )
            return

        worker_future = executor.submit(
            camera_worker,
            src,
            camID,
            source_type,
        )

        cameraWorkers[camID] = (
            worker_future
        )

    logger.info(
        "Camera worker started | "
        "cam_id=%s source_type=%s",
        camID,
        source_type,
    )
    
def stop_camera_worker(camID):
    with state_lock:
        stop_event = cameraStopEvents.get(camID)

        if stop_event:
            stop_event.set()

        # Camera stop is also a scene boundary.
        try:
            clear_tracker(camID)
        except Exception:
            logger.exception(
                "Person tracker cleanup failed | cam_id=%s",
                camID,
            )

        try:
            clear_vehicle_tracker(camID)
        except Exception:
            logger.exception(
                "Vehicle tracker cleanup failed | cam_id=%s",
                camID,
            )

        runtime_state.tracks.pop(
            camID,
            None,
        )

        runtime_state.pairs.pop(
            camID,
            None,
        )

        activeAlertBoxes.pop(
            camID,
            None,
        )

        reset_method = getattr(
            event_fusion,
            "reset_camera",
            None,
        )

        if callable(reset_method):
            try:
                reset_method(camID)
            except Exception:
                logger.exception(
                    "EventFusion cleanup failed | cam_id=%s",
                    camID,
                )

    # --------------------------------------------------------
    # Explicit user stop = worker_running=False
    # --------------------------------------------------------

    update_camera_status(
        camID,
        False,
        state=CameraState.OFFLINE,
        worker_running=False,
    )

    lock = cameraLocks.get(camID)

    if lock:
        with lock:
            latestJpegs.pop(
                camID,
                None,
            )

    stream_manager.clear(camID)
    stream_manager.remove(camID)

    camera_health_registry.remove(camID)

    camera_config.remove(camID)


async def stream_latest_frame(
    camID: str,
    request: Request,
):
    while True:
        if await request.is_disconnected():
            logger.info(
                "Stream client disconnected | cam_id=%s",
                camID,
            )
            break

        # ----------------------------------------------------
        # Camera worker intentionally stopped
        # ----------------------------------------------------

        if not is_camera_running(camID):
            logger.info(
                "Stream stopped because camera worker inactive | cam_id=%s",
                camID,
            )
            break

        # ----------------------------------------------------
        # Read latest JPEG
        # ----------------------------------------------------

        jpeg = None

        with state_lock:
            lock = cameraLocks.get(camID)

        if lock:
            with lock:
                jpeg = latestJpegs.get(camID)

        # ----------------------------------------------------
        # Send frame if available
        # ----------------------------------------------------

        if jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
                b"Pragma: no-cache\r\n"
                b"Expires: 0\r\n"
                b"Content-Length: "
                + str(len(jpeg)).encode("ascii")
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )


        await asyncio.sleep(
            STREAM_IDLE_SLEEP_SECONDS
        )

async def stream_distributed_camera(
    camera_pk: int,
    camID: str,
    request: Request,
):
    """Proxy authenticated MJPEG from the worker that currently owns a camera.

    Ownership is resolved for every reconnect so a browser stream can recover
    after worker failover without exposing the worker endpoint publicly.
    """
    timeout = httpx.Timeout(connect=3.0, read=None, write=5.0, pool=5.0)
    last_intent_check = 0.0
    while True:
        if await request.is_disconnected():
            return

        try:
            runtime = get_camera_runtime(int(camera_pk))
        except Exception:
            logger.exception("Distributed camera runtime lookup failed | camera_pk=%s cam_id=%s", camera_pk, camID)
            runtime = None

        if not runtime or str(runtime.get("state") or "").upper() not in {"CLAIMED", "RUNNING", "RECONNECTING"}:
            # Close a stopped/deleted camera stream instead of leaving the
            # browser connected forever. During a worker crash/failover the
            # desired state remains RUNNING, so this loop keeps waiting for
            # the replacement worker and preserves stream recovery.
            now = time.monotonic()
            if now - last_intent_check >= 1.0:
                last_intent_check = now
                try:
                    with SessionLocal() as intent_db:
                        row = intent_db.query(Camera).filter(Camera.id == int(camera_pk)).first()
                        if row is None or str(row.desired_state or "STOPPED").upper() != "RUNNING":
                            return
                except Exception:
                    logger.debug(
                        "Distributed stream intent check failed | camera_pk=%s cam_id=%s",
                        camera_pk, camID, exc_info=True,
                    )
            await asyncio.sleep(0.5)
            continue

        worker_id = str(runtime.get("worker_id") or "").strip()
        preview_base_url = str(runtime.get("preview_base_url") or "").strip().rstrip("/")
        if worker_id:
            try:
                heartbeat = get_worker_heartbeat(worker_id) or {}
                preview_base_url = str(heartbeat.get("preview_base_url") or preview_base_url).strip().rstrip("/")
            except Exception:
                logger.debug("Worker heartbeat lookup failed | worker_id=%s", worker_id, exc_info=True)

        if not preview_base_url:
            await asyncio.sleep(0.5)
            continue

        target = f"{preview_base_url}/internal/stream/{quote(str(camID), safe='')}"
        headers = {"X-Intel-I-Worker-Token": WORKER_PREVIEW_TOKEN}

        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                async with client.stream("GET", target, headers=headers) as response:
                    if response.status_code != 200:
                        logger.warning(
                            "Worker preview unavailable | camera_pk=%s cam_id=%s worker_id=%s status=%s",
                            camera_pk, camID, worker_id, response.status_code,
                        )
                        await asyncio.sleep(0.75)
                        continue
                    async for chunk in response.aiter_bytes():
                        if await request.is_disconnected():
                            return
                        if chunk:
                            yield chunk
        except asyncio.CancelledError:
            return
        except Exception:
            logger.warning(
                "Worker preview disconnected; resolving current owner | camera_pk=%s cam_id=%s worker_id=%s",
                camera_pk, camID, worker_id, exc_info=True,
            )
            await asyncio.sleep(0.75)


def get_active_live_camera_count():
    return get_active_live_camera_count_from_redis()

@app.get("/")
def home():
    return {"message": "Production CCTV pre-event alert backend running"}

@app.get("/api/system/workers")
def distributed_worker_inventory(
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    if not DISTRIBUTED_CAMERA_WORKERS_ENABLED:
        return {
            "enabled": False,
            "worker_count": 0,
            "total_capacity": 0,
            "owned_cameras": 0,
            "available_capacity": 0,
            "workers": [],
            "camera_runtimes": [],
        }
    summary = distributed_capacity_summary()
    # Camera runtime rows contain tenant identifiers and internal worker URLs.
    # Filter to the authenticated user's cameras and redact routing-only fields
    # before exposing operational status to the UI.
    owned_camera_pks = {int(camera.id) for camera in get_cameras(db, current_user.id)}
    camera_runtimes = []
    for item in list_camera_runtimes():
        if int(item.get("camera_pk") or -1) not in owned_camera_pks:
            continue
        camera_runtimes.append({
            key: value
            for key, value in item.items()
            if key not in {"user_id", "preview_base_url"}
        })
    safe_workers = [
        {
            key: value
            for key, value in item.items()
            if key != "preview_base_url"
        }
        for item in (summary.get("workers") or [])
    ]
    return {
        "enabled": True,
        "worker_count": summary.get("worker_count", 0),
        "total_capacity": summary.get("total_capacity", 0),
        "owned_cameras": summary.get("owned_cameras", 0),
        "available_capacity": summary.get("available_capacity", 0),
        "workers": safe_workers,
        "camera_runtimes": camera_runtimes,
    }


@app.get("/api/system/health")
def system_health(
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    # Build the health inventory from persistent cameras first, then overlay
    # fast runtime worker state. This prevents a dashboard from showing zero
    # cameras merely because no worker has emitted a runtime heartbeat yet.
    authoritative_camera_states = {}
    authoritative_camera_details = []
    distributed_runtimes_by_pk: dict[int, dict] = {}
    if DISTRIBUTED_CAMERA_WORKERS_ENABLED:
        try:
            distributed_runtimes_by_pk = {
                int(item.get("camera_pk")): item
                for item in list_camera_runtimes()
                if item.get("camera_pk") is not None
            }
        except Exception:
            logger.exception("Failed to read distributed camera runtimes for system health")

    active_runtime_count = 0
    try:
        for camera in get_cameras(db, current_user.id):
            runtime = distributed_runtimes_by_pk.get(int(camera.id)) if DISTRIBUTED_CAMERA_WORKERS_ENABLED else None
            if runtime:
                runtime_state = str(runtime.get("state") or "").upper()
                if runtime_state in {"CLAIMED", "RUNNING", "RECONNECTING"}:
                    active_runtime_count += 1
                raw_connection_state = str(runtime.get("connection_state") or "").upper()
                if not raw_connection_state:
                    raw_connection_state = "RECONNECTING" if runtime_state == "RECONNECTING" else (camera.connection_state or "OFFLINE")
            else:
                raw_connection_state = cameraConnectionStates.get(
                    camera.cam_id, camera.connection_state or "OFFLINE"
                )
            try:
                state = normalize_camera_state(raw_connection_state)
            except ValueError:
                state = CameraState.OFFLINE
            authoritative_camera_states[camera.cam_id] = state.value
            telemetry = camera_health_registry.get(camera.cam_id) or {}
            stream = stream_manager.stats(camera.cam_id) if camera.cam_id in cameraWorkers else {"queue_depth": 0, "queue_capacity": STREAM_BUFFER_SIZE}
            if runtime:
                telemetry = {
                    **telemetry,
                    "fps": runtime.get("processing_fps", telemetry.get("fps")),
                    "last_frame_at": runtime.get("last_frame_at", telemetry.get("last_frame_at")),
                }
            authoritative_camera_details.append({
                "cam_id": camera.cam_id,
                "camera_pk": int(camera.id),
                "camera_name": camera.camera_name,
                "status": state.value,
                "source_type": camera.source_type,
                "is_active": bool(camera.is_active),
                "desired_state": str(getattr(camera, "desired_state", "STOPPED") or "STOPPED"),
                "worker_id": runtime.get("worker_id") if runtime else None,
                "worker_state": runtime.get("state") if runtime else None,
                "last_health_check": camera.last_health_check.isoformat() if camera.last_health_check else None,
                "stream": {
                    "fps": telemetry.get("fps"),
                    "target_fps": telemetry.get("fps"),
                    "latency_ms": telemetry.get("latency_ms"),
                    "decoded_frames": telemetry.get("decoded_frames", 0),
                    "processed_frames": telemetry.get("processed_frames", 0),
                    "dropped_frames": telemetry.get("dropped_frames", 0),
                    "queue_depth": stream.get("queue_depth", 0),
                    "queue_capacity": stream.get("queue_capacity", STREAM_BUFFER_SIZE),
                    "reconnects": telemetry.get("reconnects", 0),
                    "last_frame_at": telemetry.get("last_frame_at"),
                },
                "time_sync": time_synchronizer.status(camera.cam_id),
            })
    except Exception:
        logger.exception("Failed to build authoritative camera health inventory")
        authoritative_camera_states = dict(cameraConnectionStates)

    effective_active_camera_workers = (
        active_runtime_count
        if DISTRIBUTED_CAMERA_WORKERS_ENABLED
        else len(cameraWorkers)
    )
    payload = build_system_health(
        db=db,
        redis_ping=redis_ping,
        kafka_status=kafka_status,
        camera_states=authoritative_camera_states,
        active_camera_workers=effective_active_camera_workers,
        websocket_clients=len(clients),
        camera_details=authoritative_camera_details,
    )
    payload["time_sync"] = {
        "system_clock": system_clock_health(),
        "max_camera_clock_offset_ms": CAMERA_MAX_CLOCK_OFFSET_MS,
        "cameras": [time_synchronizer.status(c.cam_id) for c in get_cameras(db, current_user.id)],
    }
    payload["stream_manager"] = {
        "buffer_capacity": STREAM_BUFFER_SIZE,
        "active_streams": effective_active_camera_workers,
        "buffers": stream_manager.all_stats() if not DISTRIBUTED_CAMERA_WORKERS_ENABLED else {},
    }
    if DISTRIBUTED_CAMERA_WORKERS_ENABLED:
        try:
            distributed_summary = distributed_capacity_summary()
            payload["distributed_workers"] = {
                "worker_count": distributed_summary.get("worker_count", 0),
                "total_capacity": distributed_summary.get("total_capacity", 0),
                "owned_cameras": distributed_summary.get("owned_cameras", 0),
                "available_capacity": distributed_summary.get("available_capacity", 0),
                "workers": [
                    {key: value for key, value in item.items() if key != "preview_base_url"}
                    for item in (distributed_summary.get("workers") or [])
                ],
            }
        except Exception:
            logger.debug("Distributed worker capacity unavailable for system health", exc_info=True)
            payload["distributed_workers"] = {
                "worker_count": 0,
                "total_capacity": 0,
                "owned_cameras": active_runtime_count,
                "available_capacity": 0,
                "workers": [],
            }

    try:
        gpu_payload = payload.get("ai", {}).get("gpu", {})
        if gpu_payload.get("available"):
            gpu_utilization_percent.set(float(gpu_payload.get("utilization_percent", 0.0)))
            gpu_memory_percent.set(float(gpu_payload.get("memory_percent", 0.0)))
    except Exception:
        pass
    if not SYSTEM_INITIALIZATION_COMPLETE:
        payload["overall"] = "INITIALIZING"

    payload["initializing"] = not SYSTEM_INITIALIZATION_COMPLETE
    payload["initialization"] = {
        "status": (
            "READY"
            if SYSTEM_INITIALIZATION_COMPLETE
            else "INITIALIZING"
        ),
        "started_at": SYSTEM_INITIALIZATION_STARTED_AT,
        "completed_at": SYSTEM_INITIALIZATION_COMPLETED_AT,
    }

    return payload


@app.get("/api/system/startup-health")
def startup_health(
    response: Response,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    """Return the live readiness state consumed by SystemStartupOverlay.

    This endpoint never reports readiness from elapsed time. Every service row
    is derived from an authenticated request and a real runtime/configuration
    probe. Optional accelerators and idle camera streams may be degraded without
    blocking login; failed required services keep ``ready`` false.
    """

    from services.personRecognition import model_status as person_model_status
    from services.runtimeModelReadiness import cached_model_readiness
    from vehicleANPR import get_ocr_runtime_status

    checked_at = time.time()
    started_at = time.perf_counter()
    response.headers["Cache-Control"] = "no-store, max-age=0"

    def env_enabled(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def check_row(
        *,
        status_value: str,
        healthy: bool,
        required: bool,
        detail: str,
        latency_ms: float | None = None,
    ) -> dict[str, Any]:
        row = {
            "status": status_value,
            "healthy": bool(healthy),
            "required": bool(required),
            "detail": detail,
        }
        if latency_ms is not None:
            row["latency_ms"] = round(float(latency_ms), 2)
        return row

    # Reuse the production health inventory so Redis, Kafka, GPU, cameras,
    # WebSockets and PostgreSQL are checked by the same code as System Health.
    health = system_health(db=db, current_user=current_user)
    database = health.get("database", {})
    redis_state = health.get("redis", {})
    kafka_state = health.get("kafka", {})
    gpu_state = health.get("ai", {}).get("gpu", {})
    camera_state = health.get("cameras", {})

    # Object storage is a backend requirement in production unless explicitly
    # overridden. Do not expose endpoints, credentials, paths or raw exceptions.
    storage_required = env_enabled("OBJECT_STORAGE_REQUIRED", IS_PROD)
    try:
        storage_state = storage_readiness()
    except Exception:
        logger.exception("Startup object-storage readiness check failed")
        storage_state = {"healthy": False, "active_backend": "unavailable"}
    storage_ok = bool(storage_state.get("healthy"))

    # SELECT PostGIS_Version() proves the extension is available in the same
    # authenticated database session; SELECT 1 alone cannot prove GIS support.
    postgis_ok = False
    postgis_latency_ms = None
    if database.get("healthy"):
        postgis_started_at = time.perf_counter()
        try:
            postgis_version = db.execute(
                text("SELECT PostGIS_Version()")
            ).scalar()
            postgis_ok = bool(postgis_version)
            postgis_latency_ms = (
                time.perf_counter() - postgis_started_at
            ) * 1000
        except Exception:
            logger.exception("Startup PostGIS readiness check failed")

    kafka_required = env_enabled("KAFKA_ENABLED", False)
    event_ok = bool(redis_state.get("healthy")) and (
        bool(kafka_state.get("healthy")) or not kafka_required
    )

    gpu_required = env_enabled(
        "GPU_READINESS_REQUIRED",
        str(os.getenv("DEVICE", "")).lower().startswith("cuda"),
    )
    gpu_ok = bool(gpu_state.get("available") and gpu_state.get("healthy"))

    model_state = cached_model_readiness()
    model_required = env_enabled("MODEL_READINESS_REQUIRED", IS_PROD)
    models_ready = model_state.get("status") == "READY"
    scheduler_state = ai_scheduler.status()
    ai_ready = bool(scheduler_state.get("running")) and models_ready

    # Recognition readiness combines the verified YuNet/SFace model rows with
    # the configured ANPR OCR runtime. Lazy-loaded OCR is ready when its signed
    # model artifacts exist; an attempted initialization failure is not ready.
    face_state = person_model_status()
    model_rows = list(model_state.get("models") or [])
    face_rows = [
        row for row in model_rows
        if row.get("id") in {"yunet_2023mar", "sface_2021dec"}
        and row.get("enabled")
    ]
    face_ready = (
        all(bool(row.get("ready")) for row in face_rows)
        if face_rows
        else bool(
            face_state.get("detector_present")
            and face_state.get("recognizer_present")
        )
    )

    try:
        ocr_state = get_ocr_runtime_status()
    except Exception:
        logger.exception("Startup ANPR OCR readiness check failed")
        ocr_state = {"ocr_initialization_failed": True}

    ocr_artifacts_ready = bool(
        ocr_state.get("weights_present")
        and ocr_state.get("dictionary_present")
        and ocr_state.get("paddleocr_present")
    )
    if not ocr_artifacts_ready:
        ocr_model_path = str(ocr_state.get("ocr_onnx_model") or "").strip()
        ocr_dictionary_path = str(ocr_state.get("ocr_dictionary") or "").strip()

        def backend_file_exists(raw_path: str) -> bool:
            if not raw_path:
                return False
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = Path(__file__).resolve().parent / candidate
            return candidate.is_file()

        ocr_artifacts_ready = bool(
            ocr_model_path
            and ocr_dictionary_path
            and backend_file_exists(ocr_model_path)
            and backend_file_exists(ocr_dictionary_path)
        )
    ocr_ready = bool(
        ocr_state.get("ocr_loaded") or ocr_artifacts_ready
    ) and not bool(ocr_state.get("ocr_initialization_failed"))
    recognition_ready = bool(face_ready and ocr_ready)
    recognition_required = env_enabled("RECOGNITION_READINESS_REQUIRED", IS_PROD)

    configured_cameras = int(camera_state.get("total") or 0)
    online_cameras = int(camera_state.get("online") or 0)
    active_workers = int(camera_state.get("active_workers") or 0)
    streaming_required = env_enabled("CAMERA_STREAMING_REQUIRED", False)
    streaming_ready = configured_cameras == 0 or (
        online_cameras > 0 and active_workers > 0
    )

    backend_ok = bool(health.get("backend", {}).get("healthy")) and (
        storage_ok or not storage_required
    )

    services = {
        "security": check_row(
            status_value="READY",
            healthy=True,
            required=True,
            detail="Authenticated session verified",
        ),
        "backend": check_row(
            status_value=(
                "READY" if backend_ok and storage_ok
                else "DEGRADED" if backend_ok
                else "FAILED"
            ),
            healthy=backend_ok,
            required=True,
            detail=(
                f"API online; {storage_state.get('active_backend', 'object')} storage ready"
                if storage_ok
                else "API online; optional object storage unavailable"
                if not storage_required
                else "Required object storage unavailable"
            ),
        ),
        "database": check_row(
            status_value="READY" if database.get("healthy") else "FAILED",
            healthy=bool(database.get("healthy")),
            required=True,
            detail=(
                "PostgreSQL connection verified"
                if database.get("healthy")
                else "PostgreSQL connection unavailable"
            ),
            latency_ms=database.get("latency_ms"),
        ),
        "event-services": check_row(
            status_value="READY" if event_ok else "FAILED",
            healthy=event_ok,
            required=True,
            detail=(
                "Redis ready; Kafka ready"
                if kafka_required and event_ok
                else "Redis ready; Kafka disabled by configuration"
                if event_ok
                else "Redis or required Kafka is unavailable"
            ),
        ),
        "gpu": check_row(
            status_value=(
                "READY" if gpu_ok
                else "FAILED" if gpu_required
                else "DEGRADED"
            ),
            healthy=gpu_ok,
            required=gpu_required,
            detail=(
                "NVIDIA GPU telemetry verified"
                if gpu_ok
                else "GPU unavailable; CPU fallback remains available"
                if not gpu_required
                else "Required NVIDIA GPU unavailable"
            ),
        ),
        "ai": check_row(
            status_value=(
                "READY" if ai_ready
                else "FAILED" if model_required
                else "DEGRADED"
            ),
            healthy=ai_ready,
            required=model_required,
            detail=(
                "AI scheduler running; enabled models verified"
                if ai_ready
                else "One or more enabled AI models are not verified"
            ),
        ),
        "recognition": check_row(
            status_value=(
                "READY" if recognition_ready
                else "FAILED" if recognition_required
                else "DEGRADED"
            ),
            healthy=recognition_ready,
            required=recognition_required,
            detail=(
                "ANPR/OCR and face-recognition artifacts verified"
                if recognition_ready
                else "ANPR/OCR or face-recognition artifacts are unavailable"
            ),
        ),
        "gis": check_row(
            status_value="READY" if postgis_ok else "FAILED",
            healthy=postgis_ok,
            required=True,
            detail=(
                "PostGIS extension and correlation database verified"
                if postgis_ok
                else "PostGIS extension unavailable"
            ),
            latency_ms=postgis_latency_ms,
        ),
        "streaming": check_row(
            status_value=(
                "READY" if streaming_ready
                else "FAILED" if streaming_required
                else "DEGRADED"
            ),
            healthy=streaming_ready,
            required=streaming_required,
            detail=(
                "Camera service ready; no cameras configured"
                if configured_cameras == 0
                else f"{online_cameras}/{configured_cameras} cameras online; {active_workers} workers active"
            ),
        ),
    }

    blocking_services = [
        service_id
        for service_id, row in services.items()
        if row["required"] and not row["healthy"]
    ]
    degraded_services = [
        service_id
        for service_id, row in services.items()
        if not row["healthy"] and not row["required"]
    ]
    ready = not blocking_services

    return {
        "status": (
            "READY" if ready and not degraded_services
            else "DEGRADED" if ready
            else "NOT_READY"
        ),
        "ready": ready,
        "checked_at": checked_at,
        "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "services": services,
        "blocking_services": blocking_services,
        "degraded_services": degraded_services,
    }

def require_admin(current_user: User = Depends(get_current_user)):
    from security.rbac import ADMIN_ROLES, normalize_role
    if normalize_role(current_user.role) not in ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@app.get("/api/cameras/health")
def camera_health_inventory(
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    """Authoritative camera health + stream + time-sync inventory."""
    return {
        "count": len(get_cameras(db, current_user.id)),
        "cameras": _camera_health_details(db, current_user.id),
        "stale_after_seconds": CAMERA_HEALTH_STALE_SECONDS,
    }


@app.get("/api/cameras/{camID}/health")
def camera_health_detail(
    camID: str,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    camera = get_camera(db, camID, current_user.id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return next(item for item in _camera_health_details(db, current_user.id) if item["cam_id"] == camID)


@app.get("/api/cameras/{camID}/time-sync")
def camera_time_sync_detail(
    camID: str,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    camera = get_camera(db, camID, current_user.id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return time_synchronizer.status(camID)


@app.get("/admin/health")
def admin_health(current_user: User = Depends(require_admin)):
    redis_ok = redis_ping()
    kafka_ok = kafka_status()
    kafka_required = os.getenv("KAFKA_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    object_storage_required = os.getenv(
        "OBJECT_STORAGE_REQUIRED",
        "true" if IS_PROD else "false",
    ).strip().lower() in {"1", "true", "yes", "on"}

    try:
        object_storage_status = storage_readiness()
    except Exception:
        logger.exception("Object storage admin health check failed")
        object_storage_status = {
            "healthy": False,
            "detail": "ObjectStorageError",
        }

    storage_ok = bool(object_storage_status.get("healthy"))
    overall_ok = bool(
        redis_ok
        and (kafka_ok or not kafka_required)
        and (storage_ok or not object_storage_required)
    )

    return {
        "status": "ok" if overall_ok else "degraded",
        "redis": redis_ok,
        "kafka": kafka_ok if kafka_required else "disabled",
        "object_storage": object_storage_status,
        "active_camera_workers": len(cameraWorkers),
        "active_websocket_clients": len(clients),
    }

@app.post("/video-upload")
async def uploadvideo(
    request: Request,
    videoFile: UploadFile = File(...),
    camera_name: str | None = Form(default=None),
    latitude: float | None = Form(default=None),
    longitude: float | None = Form(default=None),
    location_name: str | None = Form(default=None),
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    del request 
    validate_video_upload(videoFile)

    # Uploaded videos participate in the same cross-camera correlation and
    # GIS pipeline as live cameras. A coordinate pair is optional so ordinary
    # uploads remain supported, but one coordinate without the other is
    # rejected because it cannot define a valid camera location.
    if (latitude is None) != (longitude is None):
        raise HTTPException(
            status_code=422,
            detail="latitude and longitude must be provided together",
        )
    if latitude is not None and not -90.0 <= latitude <= 90.0:
        raise HTTPException(status_code=422, detail="Invalid latitude")
    if longitude is not None and not -180.0 <= longitude <= 180.0:
        raise HTTPException(status_code=422, detail="Invalid longitude")

    streamID = str(uuid.uuid4())
    camID = f"UPLOAD_{current_user.id}_{streamID[:8]}"
    ext = Path(videoFile.filename or ".mp4").suffix.lower()
    storage_key = safe_storage_key(camID, ext)
    temp_path = resolve_upload_ingest_path(storage_key)
    total_size = 0
    uploaded_record_created = False
    camera_record_created = False
    object_stored = False

    def cleanup_failed_upload():
        try:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        except Exception:
            logger.exception(
                "Upload temp file cleanup failed | stream_id=%s cam_id=%s",
                streamID,
                camID,
            )

        if object_stored:
            try:
                get_object_storage().delete_object(
                    storage_key,
                    delete_cache=True,
                    ignore_missing=True,
                )
            except Exception:
                logger.exception(
                    "Object storage rollback failed | stream_id=%s cam_id=%s",
                    streamID,
                    camID,
                )

        try:
            delete_upload_state(
                stream_id=streamID,
                cam_id=camID,
            )
        except Exception:
            logger.exception(
                "Redis upload state cleanup failed | stream_id=%s cam_id=%s",
                streamID,
                camID,
            )

        try:
            delete_camera_state(camID)
        except Exception:
            logger.exception(
                "Redis camera state cleanup failed | cam_id=%s",
                camID,
            )
        if camera_record_created:
            try:
                delete_camera(
                    db=db,
                    cam_id=camID,
                    user_id=current_user.id,
                )
            except Exception:
                db.rollback()
                logger.exception(
                    "Failed to rollback uploaded camera record | cam_id=%s",
                    camID,
                )

        if uploaded_record_created:
            try:
                delete_uploaded_video_by_cam(
                    db=db,
                    cam_id=camID,
                    user_id=current_user.id,
                )
            except Exception:
                db.rollback()
                logger.exception(
                    "Failed to rollback uploaded video record | cam_id=%s",
                    camID,
                )

        with state_lock:
            uploadvideos.pop(streamID, None)
            uploadStreamToCam.pop(streamID, None)
            uploadCamToStream.pop(camID, None)
            cameraIDS.pop(camID, None)
            cameraFrameCount.pop(camID, None)
            cameraConnectionStates.pop(camID, None)
            cameraWorkers.pop(camID, None)
            cameraStopEvents.pop(camID, None)
            cameraLocks.pop(camID, None)
            latestJpegs.pop(camID, None)

    try:
        with temp_path.open("wb") as buffer:
            while True:
                chunk = await videoFile.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video exceeds {MAX_UPLOAD_SIZE_MB}MB limit",
                    )

                buffer.write(chunk)

        await videoFile.close()

        validate_saved_video_file(temp_path)

        stored_object = await asyncio.to_thread(
            get_object_storage().put_file,
            temp_path,
            storage_key,
            content_type=videoFile.content_type,
            metadata={
                "stream-id": streamID,
                "camera-id": camID,
                "owner-user-id": str(current_user.id),
                "original-filename": Path(videoFile.filename or "").name,
            },
            verify_remote=True,
        )
        object_stored = True

        # The durable copy is now private object storage. Remove the ingest
        # file before continuing; OpenCV/FFmpeg will later receive a verified
        # cache materialization of the object.
        temp_path.unlink(missing_ok=True)

        uploaded = create_uploaded_video(
            db=db,
            user_id=current_user.id,
            cam_id=camID,
            stream_id=streamID,
            storage_key=storage_key,
            original_filename=Path(videoFile.filename or "").name,
            mime_type=videoFile.content_type,
            size_bytes=stored_object.size_bytes,
        )
        uploaded_record_created = True

        upload_camera = create_camera(
            db=db,
            user_id=current_user.id,
            cam_id=camID,
            camera_name=(camera_name or f"Uploaded Video {streamID[:8]}")[:200],
            zone="Upload",
            source=camID,
            source_type="upload",
            latitude=latitude,
            longitude=longitude,
            location_name=(location_name[:255] if location_name else None),
        )
        camera_record_created = True
        _cache_camera_correlation_meta(upload_camera)

        with state_lock:
            uploadvideos[streamID] = storage_key
            uploadStreamToCam[streamID] = camID
            uploadCamToStream[camID] = streamID
            cameraIDS[camID] = camID
            cameraFrameCount[camID] = 0
            
        set_upload_state(
            stream_id=streamID,
            cam_id=camID,
            storage_key=storage_key,
        )

        set_camera_state(
            cam_id=camID,
            source=camID,
            source_type="upload",
            status=False,
            frame_count=0,
        )

        _record_audit_event(
            db,
            user_id=current_user.id,
            action="VIDEO_UPLOADED",
            resource_type="uploaded_video",
            resource_id=streamID,
            details={
                "cam_id": camID,
                "size_bytes": stored_object.size_bytes,
                "sha256": stored_object.sha256,
                "storage_backend": stored_object.backend,
                "source_type": "upload",
                "has_gis_location": latitude is not None and longitude is not None,
            },
        )

        return {
            "streamID": uploaded.stream_id,
            "camID": uploaded.cam_id,
            "camera_name": upload_camera.camera_name,
            "latitude": latitude,
            "longitude": longitude,
            "location_name": location_name,
        }

    except HTTPException:
        cleanup_failed_upload()
        raise

    except Exception:
        logger.exception(
            "Video upload failed | stream_id=%s cam_id=%s",
            streamID,
            camID,
        )
        cleanup_failed_upload()
        raise HTTPException(
            status_code=500,
            detail="Video upload failed",
        )


@app.post("/stream-session/upload/{streamID}")
def create_upload_stream_session(
    streamID: str,
    response: Response,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    uploaded = get_uploaded_video_by_stream(
        db=db,
        stream_id=streamID,
        user_id=current_user.id,
    )

    if not uploaded:
        raise HTTPException(status_code=404, detail="Upload stream not found")

    camera = get_camera(db, uploaded.cam_id, current_user.id)

    if not camera:
        raise HTTPException(status_code=404, detail="Upload camera not found")

    sid = create_stream_session(
        user_id=current_user.id,
        target_id=streamID,
        stream_type="upload",
    )

    set_stream_cookie(response, "upload", streamID, sid)

    return {"message": "Upload stream session created"}


@app.get("/stream-upload/{streamID}")
async def streamUpload(
    streamID: str,
    request: Request,
):
    sid = request.cookies.get(stream_cookie_name("upload", streamID))

    if not sid:
        raise HTTPException(status_code=401, detail="Missing stream session")

    session = validate_stream_session(
        sid=sid,
        target_id=streamID,
        stream_type="upload",
    )

    user_id = session["user_id"]

    with SessionLocal() as db:
        uploaded = get_uploaded_video_by_stream(
            db=db,
            stream_id=streamID,
            user_id=user_id,
        )

        if not uploaded:
            raise HTTPException(status_code=404, detail="Upload stream not found")

        uploaded_cam_id = str(uploaded.cam_id)
        uploaded_storage_key = str(uploaded.storage_key)

        camera = get_camera(db, uploaded_cam_id, user_id)

        if not camera:
            raise HTTPException(status_code=404, detail="Upload camera not found")

        _cache_camera_correlation_meta(camera)

    try:
        video_path = await asyncio.to_thread(
            resolve_uploaded_video_path,
            uploaded_storage_key,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Uploaded video object missing")
    except ObjectStorageError:
        logger.exception(
            "Uploaded video object materialization failed | stream_id=%s cam_id=%s",
            streamID,
            uploaded_cam_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Uploaded video storage is temporarily unavailable",
        )

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded video object missing")

    with state_lock:
        cameraIDS[uploaded_cam_id] = uploaded_cam_id
        cameraFrameCount[uploaded_cam_id] = 0
        uploadvideos[streamID] = uploaded_storage_key
        uploadStreamToCam[streamID] = uploaded_cam_id
        uploadCamToStream[uploaded_cam_id] = streamID

    start_camera_worker(uploaded_cam_id, str(video_path), "upload")

    return StreamingResponse(
        stream_latest_frame(uploaded_cam_id, request),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
    
@app.post("/stream-session/camera/{camID}")
def create_camera_stream_session(
    camID: str,
    response: Response,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    camera = get_camera(db, camID, current_user.id)

    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    sid = create_stream_session(
        user_id=current_user.id,
        target_id=camID,
        stream_type="camera",
    )

    set_stream_cookie(response, "camera", camID, sid)

    return {"message": "Stream session created"}


@app.get("/stream/{camID}")
async def streamCam(
    camID: str,
    request: Request,
):
    sid = request.cookies.get(stream_cookie_name("camera", camID))

    if not sid:
        raise HTTPException(status_code=401, detail="Missing stream session")

    session = validate_stream_session(
        sid=sid,
        target_id=camID,
        stream_type="camera",
    )

    # Validate ownership with a short-lived DB session, then close it
    # before returning the long-lived MJPEG stream.
    with SessionLocal() as db:
        camera = get_camera(
            db,
            camID,
            session["user_id"],
        )

        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        camera_pk = int(camera.id)

    stream_generator = (
        stream_distributed_camera(camera_pk, camID, request)
        if DISTRIBUTED_CAMERA_WORKERS_ENABLED
        else stream_latest_frame(camID, request)
    )
    return StreamingResponse(
        stream_generator,
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
    
@app.post("/camera/{camID}/start")
def startCamera(
    request: Request,
    camID: str,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    camID = str(camID or "").strip()

    if not camID:
        raise HTTPException(
            status_code=400,
            detail="Camera ID is required",
        )

    camera = get_camera(
        db,
        camID,
        current_user.id,
    )

    if not camera:
        raise HTTPException(
            status_code=404,
            detail="Camera not found",
        )

    source_type = str(
        camera.source_type or ""
    ).strip().lower()

    allowed_source_types = {
        "rtsp",
        "http",
        "https",
        "live",
        "hls",
        "webcam",
        "upload",
        "recorded_video",
        "onvif",
        "nvr",
        "vms",
        "vendor_api",
        "vendor_sdk",
    }

    if source_type not in allowed_source_types:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported camera source type: "
                f"{source_type}"
            ),
        )

    # In distributed mode the API persists intent only.  Camera credentials,
    # connector resolution, zones, model execution and reconnect lifecycle are
    # owned by the analytics worker that wins the Redis lease.
    if DISTRIBUTED_CAMERA_WORKERS_ENABLED and source_type not in {"upload", "recorded_video"}:
        set_camera_desired_state(db, camera, "RUNNING", commit=True)
        _record_audit_event(
            db,
            user_id=current_user.id,
            action="CAMERA_START_REQUESTED",
            resource_type="camera",
            resource_id=camID,
            details={"source_type": source_type, "execution": "distributed"},
        )
        return {
            "message": f"{camID} start requested",
            "status": "starting",
            "desired_state": "RUNNING",
            "source_type": source_type,
            "cam_id": camID,
            "camera_pk": int(camera.id),
            "execution": "distributed",
        }

    _cache_camera_correlation_meta(camera)
    already_running = is_camera_running(camID)

    if source_type in {
        "rtsp",
        "http",
        "https",
        "live",
        "hls",
    }:
        encrypted_source = getattr(camera, "encrypted_source", None)

        if not encrypted_source:
            logger.error(
                "Encrypted camera source missing | cam_id=%s",
                camID,
            )
            raise HTTPException(
                status_code=500,
                detail="Camera source is not configured",
            )

        try:
            source = decrypt_camera_source(encrypted_source)
        except Exception:
            logger.exception(
                "Failed to decrypt camera source | cam_id=%s",
                camID,
            )
            raise HTTPException(
                status_code=500,
                detail="Unable to access camera source",
            )

        try:
            _validate_capture_source(source, source_type)
        except ValueError as exc:
            logger.error(
                "Invalid camera source | cam_id=%s source_type=%s",
                camID,
                source_type,
            )
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    elif source_type == "webcam":
        encrypted_source = getattr(camera, "encrypted_source", None)
        if not encrypted_source:
            raise HTTPException(
                status_code=500,
                detail="Webcam source is not configured",
            )
        try:
            source = decrypt_camera_source(encrypted_source)
            source = int(str(source).strip())
        except (TypeError, ValueError):
            logger.error(
                "Invalid webcam source | cam_id=%s",
                camID,
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid webcam device ID",
            )

        if source < 0:
            raise HTTPException(
                status_code=400,
                detail="Webcam device ID cannot be negative",
            )

    else:
        # ONVIF / vendor API / vendor SDK:
        # source contains only a non-secret reference. Credentials and
        # vendor connection material are decrypted only in memory.
        encrypted_config = getattr(
            camera,
            "connector_config_encrypted",
            None,
        )
        if not encrypted_config:
            raise HTTPException(
                status_code=500,
                detail="Camera connector configuration is not configured",
            )

        try:
            connector_config = decrypt_connector_config(
                encrypted_config
            )
            validate_connector_config(
                source_type,
                connector_config,
            )
        except ValueError as exc:
            logger.error(
                "Invalid connector configuration | cam_id=%s source_type=%s",
                camID,
                source_type,
            )
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        except Exception:
            logger.exception(
                "Failed to decrypt connector configuration | cam_id=%s source_type=%s",
                camID,
                source_type,
            )
            raise HTTPException(
                status_code=500,
                detail="Unable to access camera connector configuration",
            )
        source = connector_config
    if source_type in {"onvif", "nvr", "vms", "vendor_api", "vendor_sdk"}:
        try:
            from connectors.manager import resolve_connector
            result = resolve_connector(source_type, source)
            camera.resolved_source_type = result.source_type
            camera.connection_state = "OFFLINE"
            camera.last_health_check = datetime.now(timezone.utc).replace(tzinfo=None)
            db.commit()
        except Exception:
            logger.exception(
                "Camera connector preflight failed | cam_id=%s source_type=%s",
                camID,
                source_type,
            )
            raise HTTPException(
                status_code=502,
                detail="Camera connector could not resolve a usable stream",
            )
    else:
        camera.resolved_source_type = source_type
        camera.connection_state = "OFFLINE"

    if camera.zones:
        try:
            zone_manager.set_zones(
                camID,
                camera.zones,
            )

        except Exception:
            logger.exception(
                "Failed to configure camera zones | cam_id=%s",
                camID,
            )

            raise HTTPException(
                status_code=500,
                detail="Failed to configure camera zones",
            )
    camera_config.set_mode(
        camID,
        camera.zone,
    )
    if not already_running:
        active_count = (
            get_active_live_camera_count()
        )

        capacity_check = can_start_camera(
            active_count
        )

        if not capacity_check["allowed"]:
            raise HTTPException(
                status_code=429,
                detail=capacity_check,
            )
        with state_lock:
            # cameraIDS is an internal identity map only; never store
            # plaintext connector credentials or resolved URLs here.
            cameraIDS[camID] = camID
            cameraFrameCount[camID] = 0
        try:
            start_camera_worker(
                camID,
                source,
                source_type,
            )

        except Exception:
            with state_lock:
                cameraIDS.pop(
                    camID,
                    None,
                )

                cameraFrameCount.pop(
                    camID,
                    None,
                )

            logger.exception(
                "Failed to start camera worker | cam_id=%s",
                camID,
            )

            raise HTTPException(
                status_code=500,
                detail="Failed to start camera",
            )
        set_camera_state(
            cam_id=camID,
            source=camID,
            source_type=source_type,
            status=True,
            frame_count=0,
        )

    set_camera_desired_state(db, camera, "RUNNING", commit=False)
    db.commit()
    _record_audit_event(
        db,
        user_id=current_user.id,
        action="CAMERA_STARTED",
        resource_type="camera",
        resource_id=camID,
        details={"source_type": source_type},
    )
    return {
        "message": f"{camID} started",
        "status": "running",
        "source_type": source_type,
        "cam_id": camID,
    }

@app.post("/camera/{camID}/stop")
def stopCamera(
    camID: str,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    camera = get_camera(db, camID, current_user.id)

    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    set_camera_desired_state(db, camera, "STOPPED", commit=True)

    if DISTRIBUTED_CAMERA_WORKERS_ENABLED and str(camera.source_type or "").strip().lower() not in {"upload", "recorded_video"}:
        _record_audit_event(
            db,
            user_id=current_user.id,
            action="CAMERA_STOP_REQUESTED",
            resource_type="camera",
            resource_id=camID,
            details={"execution": "distributed"},
        )
        return {
            "message": f"{camID} stop requested",
            "status": "stopping",
            "desired_state": "STOPPED",
            "cam_id": camID,
            "camera_pk": int(camera.id),
            "execution": "distributed",
        }

    stop_camera_worker(camID)
    update_camera_status(camID, False, state=CameraState.OFFLINE)
    _record_audit_event(
        db,
        user_id=current_user.id,
        action="CAMERA_STOPPED",
        resource_type="camera",
        resource_id=camID,
    )

    return {"message": f"{camID} stopped", "desired_state": "STOPPED"}

@app.post("/cameras/connection-test")
def test_unsaved_camera_connection(
    request: Request,
    payload: CameraConnectorTestRequest,
    current_user: User = Depends(get_current_user),
):
    """Test an unsaved camera connector without persisting credentials."""
    source_type = str(payload.source_type or "").strip().lower()
    if source_type not in {"onvif", "nvr", "vms", "vendor_api", "vendor_sdk"}:
        raise HTTPException(
            status_code=400,
            detail="This endpoint is for ONVIF, Vendor API, and Vendor SDK connectors",
        )

    try:
        config = validate_connector_config(
            source_type,
            payload.connector_config or {},
        )
        from connectors.manager import resolve_connector
        result = resolve_connector(source_type, config)
        return {
            "ok": True,
            "source_type": source_type,
            "resolved_source_type": result.source_type,
            "metadata": {
                k: v for k, v in result.metadata.items()
                if k not in {"username", "password", "token", "api_key"}
            },
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception(
            "Unsaved camera connector test failed | source_type=%s",
            source_type,
        )
        raise HTTPException(
            status_code=502,
            detail="Camera connector test failed",
        )


@app.post("/cameras/onvif/profiles")
def discover_unsaved_onvif_profiles(
    request: Request,
    payload: CameraConnectorTestRequest,
    current_user: User = Depends(get_current_user),
):
    """Discover ONVIF profiles without creating a camera record."""
    if str(payload.source_type or "").strip().lower() != "onvif":
        raise HTTPException(status_code=400, detail="source_type must be onvif")

    try:
        config = validate_connector_config(
            "onvif",
            payload.connector_config or {},
        )
        from connectors.onvif.connector import ONVIFConnector
        profiles = ONVIFConnector(config).discover_profiles()
        return {"count": len(profiles), "profiles": profiles}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unsaved ONVIF profile discovery failed")
        raise HTTPException(
            status_code=502,
            detail="ONVIF profile discovery failed",
        )


@app.post("/camera/{camID}/connection-test")
def test_camera_connection(
    request: Request,
    camID: str,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    """Authenticated, non-secret camera connector health test."""
    camera = get_camera(db, camID, current_user.id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    source_type = str(camera.source_type or "").strip().lower()

    try:
        if source_type in {"onvif", "vendor_api", "vendor_sdk"}:
            encrypted = getattr(camera, "connector_config_encrypted", None)
            if not encrypted:
                raise ValueError("Connector configuration is not configured")
            config = decrypt_connector_config(encrypted)
            validate_connector_config(source_type, config)

            from connectors.manager import resolve_connector
            result = resolve_connector(source_type, config)

            camera.resolved_source_type = result.source_type
            camera.connection_state = "ONLINE"
            camera.last_health_check = datetime.now(timezone.utc).replace(tzinfo=None)
            db.commit()

            return {
                "ok": True,
                "source_type": source_type,
                "resolved_source_type": result.source_type,
                "metadata": {
                    k: v for k, v in result.metadata.items()
                    if k not in {"username", "password", "token", "api_key"}
                },
            }

        encrypted_source = getattr(camera, "encrypted_source", None)
        if source_type == "webcam":
            source = int(str(camera.source).strip())
        elif encrypted_source:
            source = decrypt_camera_source(encrypted_source)
        else:
            raise ValueError("Camera source is not configured")

        cap = _open_video_capture(source, source_type)
        if cap is None:
            raise RuntimeError("Camera stream could not be opened")
        try:
            ok, _frame = cap.read()
        finally:
            try:
                cap.release()
            except Exception:
                pass

        if not ok:
            raise RuntimeError("Camera stream did not provide a frame")

        camera.resolved_source_type = source_type
        camera.connection_state = "ONLINE"
        camera.last_health_check = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()

        return {
            "ok": True,
            "source_type": source_type,
            "resolved_source_type": source_type,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "Camera connection test failed | cam_id=%s source_type=%s",
            camID,
            source_type,
        )
        camera.connection_state = "OFFLINE"
        camera.last_health_check = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()
        raise HTTPException(
            status_code=502,
            detail="Camera connection test failed",
        )


@app.get("/camera/onvif/discover")
def discover_onvif_cameras(
    current_user: User = Depends(get_current_user),
):
    """Discover ONVIF transmitters on the local network without credentials."""
    try:
        from connectors.onvif.connector import discover_onvif_devices
        devices = discover_onvif_devices()
        return {"count": len(devices), "devices": devices}
    except Exception:
        logger.exception("ONVIF discovery failed")
        raise HTTPException(
            status_code=502,
            detail="ONVIF discovery failed",
        )


@app.get("/camera/{camID}/onvif/profiles")
def get_onvif_profiles(
    camID: str,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    """Return non-secret ONVIF media profiles for an owned camera."""
    camera = get_camera(db, camID, current_user.id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    if str(camera.source_type or "").strip().lower() != "onvif":
        raise HTTPException(status_code=400, detail="Camera is not an ONVIF camera")

    encrypted = getattr(camera, "connector_config_encrypted", None)
    if not encrypted:
        raise HTTPException(status_code=500, detail="ONVIF configuration is not configured")

    try:
        config = decrypt_connector_config(encrypted)
        validate_connector_config("onvif", config)
        from connectors.onvif.connector import ONVIFConnector
        profiles = ONVIFConnector(config).discover_profiles()
        return {"count": len(profiles), "profiles": profiles}
    except Exception:
        logger.exception("ONVIF profile discovery failed | cam_id=%s", camID)
        raise HTTPException(
            status_code=502,
            detail="ONVIF profile discovery failed",
        )


@app.post("/cameras")
def addCamera(
    camera: CameraCreate,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    cam_id = f"CAM_{current_user.id}_{int(time.time())}"

    try:
        normalized_source_type = normalize_source_type(camera.source_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        newCam = create_camera(
            db=db,
            user_id=current_user.id,
            cam_id=cam_id,
            camera_name=camera.camera_name or cam_id,
            zone=camera.zone,
            source=camera.source,
            source_type=normalized_source_type,
            latitude=camera.latitude,
            longitude=camera.longitude,
            location_name=camera.location_name,
            connector_config=camera.connector_config,
            vendor=camera.vendor,
            vendor_device_id=camera.vendor_device_id,
            stream_profile=camera.stream_profile,
            direction=camera.direction,
            stream_fps=camera.stream_fps,
            stream_width=camera.stream_width,
            stream_height=camera.stream_height,
            transport=camera.transport,
            codec=camera.codec,
            altitude=camera.altitude,
            heading=camera.heading,
            fov=camera.fov,
            road_name=camera.road_name,
            city=camera.city,
            state=camera.state,
            country=camera.country,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _cache_camera_correlation_meta(newCam)

    source_type = normalized_source_type

    with state_lock:
        cameraIDS[cam_id] = cam_id
        cameraFrameCount[cam_id] = 0

    set_camera_state(
        cam_id=cam_id,
        source=cam_id,
        source_type=source_type,
        status=False,
        frame_count=0,
    )

    _record_audit_event(
        db,
        user_id=current_user.id,
        action="CAMERA_CREATED",
        resource_type="camera",
        resource_id=cam_id,
        details={"source_type": source_type, "connector_type": newCam.connector_type},
    )

    return {
        "message": "Camera saved.",
        "cam_id": cam_id,
        "source_type": source_type,
        "connector_type": newCam.connector_type,
        "vendor": newCam.vendor,
        "stream_profile": newCam.stream_profile,
    }

@app.get("/cameras")
def list_cameras(db: Session = Depends(getDB), current_user: User = Depends(get_current_user)):
    """Return the canonical, non-secret camera inventory contract."""
    cams = get_cameras(db, current_user.id)
    supported = {
        "rtsp", "http", "https", "live", "webcam", "hls",
        "onvif", "nvr", "vms", "recorded_video", "vendor_api", "vendor_sdk",
    }
    result = []
    for cam in cams:
        if str(cam.source_type or "").lower() not in supported:
            continue
        try:
            redis_state = get_camera_state(cam.cam_id) if DISTRIBUTED_CAMERA_WORKERS_ENABLED else None
            state = normalize_camera_state(
                (redis_state or {}).get("connection_state")
                or cameraConnectionStates.get(cam.cam_id, cam.connection_state or "OFFLINE")
            )
        except ValueError:
            state = CameraState.OFFLINE
        runtime = None
        if DISTRIBUTED_CAMERA_WORKERS_ENABLED:
            try:
                runtime = get_camera_runtime(int(cam.id))
            except Exception:
                runtime = None
        contract = normalize_camera_config(
            camera_id=cam.cam_id,
            name=cam.camera_name,
            source_type=cam.source_type,
            resolved_source_type=cam.resolved_source_type,
            connection_state=state,
            zone=cam.zone,
            direction=cam.direction,
            latitude=cam.latitude,
            longitude=cam.longitude,
            altitude=cam.altitude,
            heading=cam.heading,
            fov=cam.fov,
            location_name=cam.location_name,
            road_name=cam.road_name,
            city=cam.city,
            state=cam.state,
            country=cam.country,
            stream_fps=cam.stream_fps,
            stream_width=cam.stream_width,
            stream_height=cam.stream_height,
            transport=cam.transport,
            codec=cam.codec,
            connector_type=cam.connector_type,
            vendor=cam.vendor,
            vendor_device_id=cam.vendor_device_id,
            stream_profile=cam.stream_profile,
        ).public_dict()
        contract.update({
            "id": cam.id,
            "source": "Hidden",
            "camera_type": cam.source_type,
            "connection_state": state.value,
            "status": state.value,
            "is_active": bool(cam.is_active),
            "desired_state": str(getattr(cam, "desired_state", "STOPPED") or "STOPPED"),
            "desired_state_updated_at": (cam.desired_state_updated_at.isoformat() if getattr(cam, "desired_state_updated_at", None) else None),
            "worker_id": (runtime or {}).get("worker_id"),
            "worker_state": (runtime or {}).get("state"),
            "last_health_check": cam.last_health_check.isoformat() if cam.last_health_check else None,
            "integration_id": cam.integration_id,
            "external_provider": cam.external_provider,
            "external_camera_id": cam.external_camera_id,
            "external_live_status": cam.external_live_status,
            "externally_managed": bool(cam.externally_managed),
            "external_last_seen_at": cam.external_last_seen_at.isoformat() if cam.external_last_seen_at else None,
            "external_last_synced_at": cam.external_last_synced_at.isoformat() if cam.external_last_synced_at else None,
            "stream_options": ((cam.external_metadata or {}).get("streams") if isinstance(cam.external_metadata, dict) else []),
        })
        result.append(contract)
    return {"count": len(result), "cameras": result}

@app.get("/camera/{camID}")
def get_camera_details(
    camID: str,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    """Return one canonical non-secret camera contract."""
    camera = get_camera(db, camID, current_user.id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    try:
        redis_state = get_camera_state(camera.cam_id) if DISTRIBUTED_CAMERA_WORKERS_ENABLED else None
        state = normalize_camera_state(
            (redis_state or {}).get("connection_state")
            or cameraConnectionStates.get(camera.cam_id, camera.connection_state or "OFFLINE")
        )
    except ValueError:
        state = CameraState.OFFLINE
    runtime = None
    if DISTRIBUTED_CAMERA_WORKERS_ENABLED:
        try:
            runtime = get_camera_runtime(int(camera.id))
        except Exception:
            runtime = None
    payload = normalize_camera_config(
        camera_id=camera.cam_id,
        name=camera.camera_name,
        source_type=camera.source_type,
        resolved_source_type=camera.resolved_source_type,
        connection_state=state,
        zone=camera.zone,
        direction=camera.direction,
        latitude=camera.latitude,
        longitude=camera.longitude,
        altitude=camera.altitude,
        heading=camera.heading,
        fov=camera.fov,
        location_name=camera.location_name,
        road_name=camera.road_name,
        city=camera.city,
        state=camera.state,
        country=camera.country,
        stream_fps=camera.stream_fps,
        stream_width=camera.stream_width,
        stream_height=camera.stream_height,
        transport=camera.transport,
        codec=camera.codec,
        connector_type=camera.connector_type,
        vendor=camera.vendor,
        vendor_device_id=camera.vendor_device_id,
        stream_profile=camera.stream_profile,
    ).public_dict()
    payload.update({
        "id": camera.id,
        "source": "Hidden",
        "camera_type": camera.source_type,
        "status": state.value,
        "connection_state": state.value,
        "is_active": bool(camera.is_active),
        "desired_state": str(getattr(camera, "desired_state", "STOPPED") or "STOPPED"),
        "desired_state_updated_at": (camera.desired_state_updated_at.isoformat() if getattr(camera, "desired_state_updated_at", None) else None),
        "worker_id": (runtime or {}).get("worker_id"),
        "worker_state": (runtime or {}).get("state"),
        "last_health_check": camera.last_health_check.isoformat() if camera.last_health_check else None,
        "integration_id": camera.integration_id,
        "external_provider": camera.external_provider,
        "external_camera_id": camera.external_camera_id,
        "external_live_status": camera.external_live_status,
        "externally_managed": bool(camera.externally_managed),
        "external_last_seen_at": camera.external_last_seen_at.isoformat() if camera.external_last_seen_at else None,
        "external_last_synced_at": camera.external_last_synced_at.isoformat() if camera.external_last_synced_at else None,
        "stream_options": ((camera.external_metadata or {}).get("streams") if isinstance(camera.external_metadata, dict) else []),
    })
    return payload

@app.get("/api/ingest")
def ingest_camera_inventory(
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):

    cameras = get_cameras(
        db,
        current_user.id,
    )

    supported_source_types = {
        "rtsp",
        "http",
        "https",
        "webcam",
        "live",
        "hls",
        "onvif",
        "nvr",
        "vms",
        "recorded_video",
        "vendor_api",
        "vendor_sdk",
    }

    result = []

    for cam in cameras:
        _cache_camera_correlation_meta(cam)
        source_type = str(
            cam.source_type or ""
        ).strip().lower()

        if source_type not in supported_source_types:
            continue

        result.append(
            {
                "cam_id": cam.cam_id,
                "camera_name": cam.camera_name,
                "source_type": source_type,
                "connector_type": cam.connector_type,
                "vendor": cam.vendor,
                "vendor_device_id": cam.vendor_device_id,
                "stream_profile": cam.stream_profile,
                "resolved_source_type": cam.resolved_source_type,
                "connection_state": cameraConnectionStates.get(
                    cam.cam_id,
                    cam.connection_state or (
                        "ONLINE" if bool(cam.is_active) else "OFFLINE"
                    ),
                ),
                "zone": cam.zone,
                "zones": cam.zones or [],
                "is_active": bool(cam.is_active),
                "latitude": cam.latitude,
                "longitude": cam.longitude,
                "location_name": cam.location_name,
                "external_provider": cam.external_provider,
                "external_camera_id": cam.external_camera_id,
                "external_live_status": cam.external_live_status,
                "externally_managed": bool(cam.externally_managed),
                "external_last_synced_at": cam.external_last_synced_at.isoformat() if cam.external_last_synced_at else None,
                "ingestion": {
                    "backend": "ffmpeg",
                    "rtsp_transport": (
                        "tcp"
                        if source_type == "rtsp"
                        else None
                    ),
                    "hls": source_type == "hls",
                    "timestamp_mode": "SOURCE_PTS" if STRICT_PTS_REQUIRED else "PTS_OR_ESTIMATED",
                    "declared_fps": cam.stream_fps,
                    "process_every_n_frames": PROCESS_EVERY_N_FRAMES,
                    "output_stream_fps": OUTPUT_STREAM_FPS,
                    "mixed_resolution_normalization": True,
                    "h264_h265_decode_path": "ffmpeg",
                    "reconnect_enabled": (
                        source_type != "webcam"
                    ),
                    "connection_state": cameraConnectionStates.get(
                        cam.cam_id,
                        "ONLINE" if bool(cam.is_active) else "OFFLINE",
                    ),
                    "rtsp_read_timeout_seconds": (
                        RTSP_READ_TIMEOUT_SECONDS
                        if source_type == "rtsp"
                        else None
                    ),
                    "stream_gap_threshold_seconds": (
                        STREAM_GAP_THRESHOLD_SECONDS
                        if source_type != "upload"
                        else None
                    ),
                    "scene_discontinuity_threshold_seconds": SCENE_DISCONTINUITY_THRESHOLD_SECONDS,
                },
            }
        )

    return {
        "count": len(result),
        "cameras": result,
        "integration_count": db.query(CameraIntegration).filter(
            CameraIntegration.user_id == current_user.id
        ).count(),
        "integrations": [
            integration_view(row)
            for row in db.query(CameraIntegration).filter(
                CameraIntegration.user_id == current_user.id
            ).order_by(CameraIntegration.created_at.desc()).all()
        ],
    }


@app.delete("/cameras/{cam_id}")
def removeCamera(
    cam_id: str,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    camera = get_camera(db, cam_id, current_user.id)

    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    uploaded_video = None

    if camera.source_type == "upload":
        uploaded_video = get_uploaded_video_by_cam(
            db=db,
            cam_id=cam_id,
            user_id=current_user.id,
        )

    stop_camera_worker(cam_id)

    # stop_camera_worker() signals the worker, but the worker may still be
    # processing a frame and holding the uploaded video open. Wait for it to
    # finish before unlinking the file (required in particular on Windows).
    with state_lock:
        worker_future = cameraWorkers.get(cam_id)

    if worker_future and not worker_future.done():
        try:
            worker_future.result(timeout=15)
        except FuturesTimeoutError:
            logger.warning(
                "Camera worker is still stopping | cam_id=%s",
                cam_id,
            )
            raise HTTPException(
                status_code=409,
                detail="Camera processing is still stopping. Retry deletion shortly.",
            )
        except Exception:
            # A completed worker may surface its processing exception here.
            # It no longer owns the source file, so cleanup can safely continue.
            logger.exception(
                "Camera worker stopped with an error during deletion | cam_id=%s",
                cam_id,
            )

    if uploaded_video:
        try:
            get_object_storage().delete_object(
                str(uploaded_video.storage_key),
                delete_cache=True,
                ignore_missing=True,
            )
        except ObjectStorageError:
            logger.exception(
                "Uploaded video object delete failed | cam_id=%s storage_key=%s",
                cam_id,
                uploaded_video.storage_key,
            )
            raise HTTPException(
                status_code=503,
                detail="Uploaded video storage is temporarily unavailable",
            )

        delete_uploaded_video_by_cam(
            db=db,
            cam_id=cam_id,
            user_id=current_user.id,
        )
    _record_audit_event(
        db,
        user_id=current_user.id,
        action="CAMERA_DELETED",
        resource_type="camera",
        resource_id=cam_id,
        details={"source_type": camera.source_type},
    )
    deleted_camera = delete_camera(db, cam_id, current_user.id)
    if not deleted_camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    delete_camera_state(cam_id)
    with state_lock:
        stream_id = uploadCamToStream.pop(cam_id, None)
        if stream_id:
            uploadvideos.pop(stream_id, None)
            uploadStreamToCam.pop(stream_id, None)
            delete_upload_state(stream_id=stream_id, cam_id=cam_id)
        else:
            delete_upload_state(cam_id=cam_id)
        cameraIDS.pop(cam_id, None)
        cameraFrameCount.pop(cam_id, None)
        cameraWorkers.pop(cam_id, None)
        cameraStopEvents.pop(cam_id, None)
        cameraLocks.pop(cam_id, None)
        latestJpegs.pop(cam_id, None)
        cameraConnectionStates.pop(cam_id, None)
        cameraTimelines.pop(cam_id, None)
        for key in list(vehicleAttributeLastLogFrame):
            if key.startswith(f"{cam_id}:"):
                vehicleAttributeLastLogFrame.pop(key, None)
        cameraCorrelationMeta.pop(cam_id, None)

        # Remove the deleted camera from active GIS journeys immediately.
        # PostgreSQL recovery is already ownership-filtered, so deleting the
        # camera record also prevents this source from being restored later.
        for global_id in list(correlationGisState):
            state = correlationGisState.get(global_id)
            if not isinstance(state, dict):
                correlationGisState.pop(global_id, None)
                liveVehiclePositionLastEmit.pop(global_id, None)
                continue

            observations = state.get("observations")
            if not isinstance(observations, list):
                observations = []

            remaining = [
                observation
                for observation in observations
                if isinstance(observation, dict)
                and str(observation.get("camera_id") or "") != cam_id
            ]

            if not remaining:
                correlationGisState.pop(global_id, None)
                liveVehiclePositionLastEmit.pop(global_id, None)
                continue

            if len(remaining) != len(observations):
                state["observations"] = remaining
                last = remaining[-1]
                state["last_seen"] = last.get("timestamp", state.get("last_seen"))
                state["last_camera_id"] = last.get("camera_id")
                state["last_track_id"] = last.get("track_id")
                state["last_similarity"] = last.get("similarity", 0.0)
                state["last_correlation_score"] = last.get("correlation_score", 0.0)

        cameraQualityStates.pop(cam_id, None)
        cameraQualityTelemetry.pop(cam_id, None)
        cameraTamperTelemetry.pop(cam_id, None)
        cameraFrozenTelemetry.pop(cam_id, None)
        cameraAdaptivePolicies.pop(cam_id, None)
        for key in list(personAppearanceStates):
            if key.startswith(f"{cam_id}:"):
                personAppearanceStates.pop(key, None)
                personAppearanceLastFrame.pop(key, None)
    try:
        camera_health_registry.remove(cam_id)
    except Exception:
        logger.debug(
            "Camera health cleanup failed | cam_id=%s",
            cam_id,
            exc_info=True,
        )
    camera_config.remove(cam_id)
    return {"message": "Camera deleted successfully"}


def _serialize_watchlist_journey_incident(incident):
    evidence_state = (
        incident.evaluation_evidence
        if isinstance(incident.evaluation_evidence, dict)
        else {}
    )
    journey = evidence_state.get("watchlist_journey")
    if not isinstance(journey, dict):
        return None
    journey = dict(journey)
    return {
        "incident_id": int(incident.id),
        "incident_type": str(incident.incident_type or ""),
        "status": str(incident.status or "OPEN"),
        "severity": str(incident.evaluation_severity or "INFO"),
        "confidence": float(incident.evaluation_confidence or 0.0),
        "started_at": incident.started_at.isoformat() if incident.started_at else None,
        "journey": journey,
    }


@app.get("/api/watchlist/journeys")
def list_watchlist_journeys(
    identity_type: str | None = Query(default=None),
    active_only: bool = Query(default=True),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    normalized_type = str(identity_type or "").strip().upper()
    if normalized_type and normalized_type not in {"VEHICLE", "PERSON"}:
        raise HTTPException(status_code=422, detail="identity_type must be VEHICLE or PERSON")
    query = db.query(Incident).filter(Incident.user_id == int(current_user.id))
    if active_only:
        query = query.filter(Incident.status.in_(["OPEN", "UNDER_REVIEW", "ESCALATED"]))
    if normalized_type == "VEHICLE":
        query = query.filter(Incident.incident_type == "VEHICLE_WATCHLIST_MATCH")
    elif normalized_type == "PERSON":
        query = query.filter(Incident.incident_type.like("PERSON_WATCHLIST%"))
    rows = query.order_by(Incident.started_at.desc()).limit(limit).all()
    journeys = []
    for row in rows:
        serialized = _serialize_watchlist_journey_incident(row)
        if serialized:
            journeys.append(serialized)
    return {"count": len(journeys), "journeys": journeys}


@app.get("/api/watchlist/incidents/{incident_id}/journey")
def get_watchlist_incident_journey(
    incident_id: int,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    incident = db.query(Incident).filter(
        Incident.id == int(incident_id),
        Incident.user_id == int(current_user.id),
    ).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    serialized = _serialize_watchlist_journey_incident(incident)
    if not serialized:
        raise HTTPException(status_code=404, detail="Watchlist journey not available")
    evidence_rows = db.query(IncidentEvidence).filter(
        IncidentEvidence.incident_id == int(incident.id),
        IncidentEvidence.user_id == int(current_user.id),
    ).order_by(IncidentEvidence.id.asc()).all()
    serialized["evidence"] = [
        {
            "id": int(row.id),
            "camera_id": (
                row.metadata_json.get("journey_observation", {}).get("camera_id")
                if isinstance(row.metadata_json, dict)
                else None
            ),
            "snapshot_id": int(row.snapshot_id) if row.snapshot_id else None,
            "evidence_type": row.evidence_type,
            "object_type": row.object_type,
            "object_reference": row.object_reference,
            "description": row.description,
            "metadata": row.metadata_json if isinstance(row.metadata_json, dict) else {},
        }
        for row in evidence_rows
    ]
    journey = serialized["journey"]
    if str(journey.get("identity_type") or "").upper() == "PERSON":
        entry_id = journey.get("person_watchlist_entry_id")
        try:
            entry_id = int(entry_id) if entry_id is not None else None
        except (TypeError, ValueError):
            entry_id = None

        entry = None
        if entry_id is not None:
            entry = db.query(PersonWatchlistEntry).filter(
                PersonWatchlistEntry.id == entry_id,
                PersonWatchlistEntry.user_id == int(current_user.id),
            ).first()

        reference_evidence = next(
            (
                item for item in serialized["evidence"]
                if str(item.get("evidence_type") or "").upper()
                == "WATCHLIST_REFERENCE"
                or str((item.get("metadata") or {}).get("evidence_role") or "").upper()
                == "WATCHLIST_REFERENCE"
            ),
            None,
        )
        reference_snapshot_id = (
            journey.get("reference_snapshot_id")
            or ((reference_evidence or {}).get("snapshot_id"))
        )
        try:
            reference_snapshot_id = (
                int(reference_snapshot_id)
                if reference_snapshot_id is not None else None
            )
        except (TypeError, ValueError):
            reference_snapshot_id = None
        person_name = (
            journey.get("watchlist_person_name")
            or journey.get("reference")
            or (entry.full_name if entry is not None else None)
        )
        reference_image_available = bool(
            reference_snapshot_id
            or (entry is not None and entry.reference_image_data)
            or journey.get("reference_image_available")
        )
        reference_image_url = (
            f"/snapshot/{int(reference_snapshot_id)}"
            if reference_snapshot_id
            else (
                f"/api/intelligence/person-watchlist/{entry_id}/image"
                if entry_id is not None and reference_image_available
                else None
            )
        )
        journey.update(
            {
                "watchlist_person_name": person_name,
                "reference": person_name,
                "reference_image_available": reference_image_available,
                "reference_image_url": reference_image_url,
                "reference_snapshot_id": (
                    int(reference_snapshot_id) if reference_snapshot_id else None
                ),
                "reference_snapshot_path": (
                    f"/snapshot/{int(reference_snapshot_id)}"
                    if reference_snapshot_id else None
                ),
            }
        )
        serialized["watchlist_subject"] = {
            "entry_id": entry_id,
            "person_name": person_name,
            "category": journey.get("category") or (
                entry.category if entry is not None else None
            ),
            "reference_image_available": reference_image_available,
            "reference_image_url": reference_image_url,
            "reference_snapshot_id": (
                int(reference_snapshot_id) if reference_snapshot_id else None
            ),
        }
    return serialized

def serialize_alert(alert):
    first_snapshot = alert.snapshots[0] if alert.snapshots else None
    level = getattr(alert, "level", None) or alert.alert_type or "LOW"
    confidence_score = getattr(alert, "confidence_score", None)

    # Legacy vehicle-watchlist rows already persisted this dedicated score.
    # Use it as a safe fallback until the database backfill is applied.
    if confidence_score is None:
        confidence_score = getattr(alert, "watchlist_match_confidence", None)

    if confidence_score is not None:
        try:
            confidence_score = max(0.0, min(1.0, float(confidence_score)))
        except (TypeError, ValueError):
            confidence_score = None

    return {
        "alert_id": alert.id,
        "id": alert.id,
        "cam_id": alert.cam_id,
        "alert_type": level,
        "alert_level": level,
        "level": level,
        "severity": normalize_alert_severity(level),
        "confidence_score": confidence_score,
        "confidence": confidence_score,
        "alert_rule": alert.rule,
        "rule": alert.rule,
        "track_id": alert.track_id,
        "source_type": alert.source_type,
        "zone": alert.zone,
        "timestamp": _iso_utc(alert.source_timestamp or alert.created_at),
        "source_timestamp": _iso_utc(alert.source_timestamp),
        "ingested_at": _iso_utc(alert.created_at),
        "source_pts_seconds": alert.source_pts_seconds,
        "timestamp_source": alert.timestamp_source,
        "timestamp_quality": alert.timestamp_quality,
        "snapshot_saved": first_snapshot is not None,
        "snapshot_id": first_snapshot.id if first_snapshot else None,
        "snapshot_path": f"/snapshot/{first_snapshot.id}" if first_snapshot else None,
        "has_snapshot": first_snapshot is not None,
        "watchlist_entry_id": getattr(alert, "watchlist_entry_id", None),
        "plate": getattr(alert, "plate", None),
        "watchlist_category": getattr(alert, "watchlist_category", None),
        "watchlist_status": getattr(alert, "watchlist_status", None),
        "watchlist_match_type": getattr(alert, "watchlist_match_type", None),
        "watchlist_match_confidence": getattr(alert, "watchlist_match_confidence", None),
        "watchlist_version": getattr(alert, "watchlist_version", None),
    }



def _load_persisted_gis_observations(
    db: Session,
    user_id: int,
    global_vehicle_id: str | None = None,
    limit: int = 5000,
):
    """Recover GIS observations from durable vehicle-observation storage."""

    owned_camera_ids = _user_camera_ids(db, user_id)

    if not owned_camera_ids:
        return []

    # ------------------------------------------------------------
    # Build query
    # IMPORTANT:
    # Apply ALL filters before order_by/limit.
    # SQLAlchemy does not allow filter() after limit()/offset().
    # ------------------------------------------------------------
    query = (
        db.query(VehicleObservation)
        .filter(
            VehicleObservation.camera_id.in_(owned_camera_ids),
            VehicleObservation.global_vehicle_id.isnot(None),
        )
    )

    # Optional global vehicle filter
    if global_vehicle_id:
        normalized_global_vehicle_id = (
            str(global_vehicle_id).strip().upper()
        )

        query = query.filter(
            VehicleObservation.global_vehicle_id
            == normalized_global_vehicle_id
        )

    # ------------------------------------------------------------
    # Ordering must happen after filters
    # ------------------------------------------------------------
    query = query.order_by(
        VehicleObservation.frame_timestamp.desc()
    )

    # ------------------------------------------------------------
    # Apply LIMIT LAST
    # ------------------------------------------------------------
    safe_limit = max(
        100,
        min(int(limit), 20000),
    )

    query = query.limit(safe_limit)

    # Execute query
    rows = query.all()

    result = []

    # ------------------------------------------------------------
    # Convert newest-first DB results back to chronological order
    # ------------------------------------------------------------
    for row in reversed(rows):

        metadata = (
            row.metadata_json
            if isinstance(row.metadata_json, dict)
            else {}
        )

        gis = (
            metadata.get("gis")
            if isinstance(metadata.get("gis"), dict)
            else {}
        )

        # --------------------------------------------------------
        # Validate coordinates
        # --------------------------------------------------------
        lat = _safe_coordinate(
            gis.get("latitude"),
            -90,
            90,
        )

        lon = _safe_coordinate(
            gis.get("longitude"),
            -180,
            180,
        )

        if lat is None or lon is None:
            continue

        # --------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------
        if row.frame_timestamp:
            try:
                timestamp = (
                    row.frame_timestamp
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )
            except (AttributeError, ValueError, TypeError):
                timestamp = 0.0
        else:
            timestamp = 0.0

        # --------------------------------------------------------
        # Camera information
        # --------------------------------------------------------
        camera_id = str(
            row.camera_id or ""
        )

        camera_meta = cameraCorrelationMeta.get(
            camera_id,
            {},
        )

        camera_name = str(
            camera_meta.get("camera_name")
            or row.camera_id
            or camera_id
        )[:200]

        # --------------------------------------------------------
        # Correlation similarity
        # --------------------------------------------------------
        similarity = _safe_float(
            metadata.get("correlation_similarity"),
            0.0,
            0.0,
            1.0,
        )

        # --------------------------------------------------------
        # Correlation score
        # --------------------------------------------------------
        correlation_score = _safe_float(
            metadata.get("correlation_score"),
            metadata.get(
                "correlation_similarity",
                0.0,
            ),
            0.0,
            1.0,
        )

        # --------------------------------------------------------
        # Build GIS observation
        # --------------------------------------------------------
        result.append(
            {
                "timestamp": timestamp,

                "camera_id": camera_id,

                "camera_name": camera_name,

                "latitude": lat,

                "longitude": lon,

                "pixel_x": gis.get("pixel_x"),

                "pixel_y": gis.get("pixel_y"),

                "position_type": str(
                    gis.get("position_type")
                    or "CAMERA_ANCHOR"
                ),

                "track_id": str(
                    row.local_track_id
                    or ""
                )[:64],

                "vehicle_type": str(
                    row.vehicle_type
                    or "vehicle"
                )[:64],

                "similarity": similarity,

                "correlation_score": correlation_score,

                "supporting_frames": 0,

                "consistency": 0.0,

                "strong_match": False,

                "match_type": str(
                    metadata.get("correlation_type")
                    or "persisted"
                )[:64],
            }
        )

    return result

def _state_from_persisted_gis(
    db: Session,
    user_id: int,
    global_vehicle_id: str,
):
    observations = _load_persisted_gis_observations(
        db,
        user_id,
        global_vehicle_id,
    )
    if not observations:
        return None

    return {
        "global_vehicle_id": global_vehicle_id,
        "first_seen": observations[0]["timestamp"],
        "last_seen": observations[-1]["timestamp"],
        "last_camera_id": observations[-1]["camera_id"],
        "last_track_id": observations[-1]["track_id"],
        "vehicle_type": observations[-1]["vehicle_type"],
        "last_similarity": observations[-1]["similarity"],
        "last_correlation_score": observations[-1]["correlation_score"],
        "strong_match": any(
            bool(item.get("strong_match")) for item in observations
        ),
        "observations": observations[-MAX_CORRELATION_GIS_OBSERVATIONS:],
    }


@app.get("/api/vehicles/correlated")
def list_correlated_vehicles(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    owned_camera_ids = _user_camera_ids(db, current_user.id)

    with state_lock:
        states = [
            _sanitize_gis_vehicle(state)
            for state in correlationGisState.values()
            if _vehicle_belongs_to_user(state, owned_camera_ids)
        ]

    states.sort(
        key=lambda item: float(item.get("last_seen") or 0.0),
        reverse=True,
    )

    return {
        "count": len(states[:limit]),
        "vehicles": states[:limit],
    }


@app.get("/api/vehicles/{global_vehicle_id}/journey")
def get_correlated_vehicle_journey(
    global_vehicle_id: str,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    global_vehicle_id = _valid_global_vehicle_id(
        global_vehicle_id
    )

    owned_camera_ids = _user_camera_ids(
        db,
        current_user.id,
    )

    with state_lock:
        state = correlationGisState.get(
            global_vehicle_id
        )

        if state is None:
            state = _state_from_persisted_gis(
                db,
                current_user.id,
                global_vehicle_id,
            )
            if state is None:
                raise HTTPException(
                    status_code=404,
                    detail="Correlated vehicle not found",
                )

        if not _vehicle_belongs_to_user(
            state,
            owned_camera_ids,
        ):
            raise HTTPException(
                status_code=404,
                detail="Correlated vehicle not found",
            )

        sanitized = _sanitize_gis_vehicle(state)

    observations = [
        obs
        for obs in sanitized.get("observations", [])
        if isinstance(obs, dict)
        and str(obs.get("camera_id"))
        in owned_camera_ids
    ]

    observations.sort(
        key=lambda item: _safe_float(
            item.get("timestamp"),
            0.0,
            0.0,
        )
    )

    route = _build_gis_route(
        observations,
        owned_camera_ids,
    )

    segments = _build_gis_segments(route)

    camera_visits = {}

    for obs in observations:
        camera_id = _valid_camera_id(
            obs.get("camera_id")
        )

        timestamp = _safe_float(
            obs.get("timestamp"),
            0.0,
            0.0,
        )

        visit = camera_visits.setdefault(
            camera_id,
            {
                "camera_id": camera_id,
                "camera_name": str(
                    obs.get("camera_name")
                    or camera_id
                )[:200],
                "latitude": _safe_coordinate(
                    obs.get("latitude"),
                    -90,
                    90,
                ),
                "longitude": _safe_coordinate(
                    obs.get("longitude"),
                    -180,
                    180,
                ),
                "first_seen": timestamp,
                "last_seen": timestamp,
                "observation_count": 0,
                "track_ids": [],
            },
        )

        visit["first_seen"] = min(
            visit["first_seen"],
            timestamp,
        )

        visit["last_seen"] = max(
            visit["last_seen"],
            timestamp,
        )

        visit["observation_count"] += 1

        track_id = str(
            obs.get("track_id") or ""
        )[:64]

        if (
            track_id
            and track_id not in visit["track_ids"]
            and len(visit["track_ids"]) < 100
        ):
            visit["track_ids"].append(
                track_id
            )

    total_distance_km = sum(
        segment["distance_km"]
        for segment in segments
        if segment["distance_km"] is not None
    )

    return {
        "global_vehicle_id": global_vehicle_id,
        "vehicle_type": str(
            sanitized.get("vehicle_type")
            or "vehicle"
        )[:64],
        "first_seen": sanitized.get("first_seen"),
        "last_seen": sanitized.get("last_seen"),
        "last_similarity": _safe_float(
            sanitized.get("last_similarity"),
            0.0,
            0.0,
            1.0,
        ),
        "last_correlation_score": _safe_float(
            sanitized.get(
                "last_correlation_score"
            ),
            0.0,
            0.0,
            1.0,
        ),
        "strong_match": bool(
            sanitized.get("strong_match", False)
        ),
        "camera_visits": list(
            camera_visits.values()
        ),
        "route": route,
        "segments": segments,
        "transition_count": len(segments),
        "total_distance_km": round(
            total_distance_km,
            3,
        ),
    }



def _gis_timestamp(value) -> float:
    """Convert ISO/datetime/epoch values to a safe epoch timestamp."""
    if value is None:
        return 0.0
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        try:
            return max(0.0, float(dt.timestamp()))
        except (OverflowError, OSError, ValueError):
            return 0.0
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return number if math.isfinite(number) and number >= 0.0 else 0.0

    raw = str(value or "").strip()
    if not raw:
        return 0.0
    try:
        number = float(raw)
        if math.isfinite(number) and number >= 0.0:
            return number
    except (TypeError, ValueError):
        pass
    try:
        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, float(dt.timestamp()))
    except (TypeError, ValueError, OverflowError, OSError):
        return 0.0


def _watchlist_vehicle_gis_paths(
    db: Session,
    user_id: int,
    owned_camera_ids: set[str],
    limit: int,
) -> list[dict]:
    """
    Convert durable VEHICLE_WATCHLIST_MATCH incident journeys into the same
    presentation-safe shape used by /api/gis/vehicle-paths.

    A watchlist hit is GIS-visible from Camera A even before a GV-* identity
    exists. Camera B then creates the first cross-camera transition.
    """
    if not owned_camera_ids:
        return []

    safe_limit = max(1, min(int(limit), 500))
    incidents = (
        db.query(Incident)
        .filter(
            Incident.user_id == int(user_id),
            Incident.incident_type == "VEHICLE_WATCHLIST_MATCH",
            Incident.status.in_(["OPEN", "UNDER_REVIEW", "ESCALATED"]),
        )
        .order_by(Incident.started_at.desc())
        .limit(safe_limit)
        .all()
    )

    paths = []
    for incident in incidents:
        evidence_state = (
            incident.evaluation_evidence
            if isinstance(incident.evaluation_evidence, dict)
            else {}
        )
        journey = evidence_state.get("watchlist_journey")
        if not isinstance(journey, dict):
            continue

        raw_route = journey.get("route")
        if not isinstance(raw_route, list):
            continue

        observations = []
        for raw_obs in raw_route[-MAX_CORRELATION_GIS_OBSERVATIONS:]:
            if not isinstance(raw_obs, dict):
                continue

            camera_id = str(raw_obs.get("camera_id") or "").strip()
            if not camera_id or camera_id not in owned_camera_ids:
                continue

            latitude = _safe_coordinate(raw_obs.get("latitude"), -90, 90)
            longitude = _safe_coordinate(raw_obs.get("longitude"), -180, 180)
            camera_meta = cameraCorrelationMeta.get(camera_id, {})

            if latitude is None:
                latitude = _safe_coordinate(camera_meta.get("latitude"), -90, 90)
            if longitude is None:
                longitude = _safe_coordinate(camera_meta.get("longitude"), -180, 180)
            if latitude is None or longitude is None:
                continue

            confidence = _safe_float(
                raw_obs.get(
                    "correlation_confidence",
                    raw_obs.get("confidence", incident.evaluation_confidence),
                ),
                0.0,
                0.0,
                1.0,
            )

            observations.append({
                "timestamp": _gis_timestamp(raw_obs.get("timestamp")),
                "camera_id": camera_id,
                "camera_name": str(
                    raw_obs.get("camera_name")
                    or camera_meta.get("camera_name")
                    or camera_id
                )[:200],
                "latitude": latitude,
                "longitude": longitude,
                "pixel_x": None,
                "pixel_y": None,
                "position_type": "CAMERA_ANCHOR",
                "track_id": str(raw_obs.get("track_id") or "")[:64],
                "vehicle_type": "watchlist_vehicle",
                "similarity": confidence,
                "correlation_score": confidence,
                "supporting_frames": 0,
                "consistency": confidence,
                "strong_match": True,
                "match_type": "WATCHLIST_EXACT",
            })

        route = _build_gis_route(observations, owned_camera_ids)
        if not route:
            continue

        segments = _build_gis_segments(route)
        global_vehicle_id = str(
            journey.get("global_vehicle_id")
            or getattr(incident, "global_vehicle_id", None)
            or ""
        ).strip().upper()
        identity_id = str(
            journey.get("identity_id")
            or getattr(incident, "primary_track_id", None)
            or f"WATCHLIST-{incident.id}"
        ).strip()
        public_identity = global_vehicle_id or identity_id or f"WATCHLIST-{incident.id}"

        paths.append({
            "global_vehicle_id": public_identity,
            "correlation_global_vehicle_id": global_vehicle_id or None,
            "identity_id": identity_id,
            "identity_source": "WATCHLIST",
            "incident_id": int(incident.id),
            "watchlist_entry_id": journey.get("watchlist_entry_id"),
            "reference": str(journey.get("reference") or "")[:64],
            "category": str(journey.get("category") or "")[:64],
            "tracking_status": str(
                journey.get("tracking_status") or "WAITING_FOR_NEXT_CAMERA"
            )[:64],
            "vehicle_type": "watchlist_vehicle",
            "first_seen": route[0]["timestamp"],
            "last_seen": route[-1]["timestamp"],
            "last_similarity": route[-1]["similarity"],
            "last_correlation_score": route[-1]["correlation_score"],
            "strong_match": True,
            "camera_count": len({point["camera_id"] for point in route}),
            "observation_count": len(route),
            "route": route,
            "segments": segments,
            "transition_count": len(segments),
        })

    return paths


@app.get("/api/gis/vehicle-paths")
def get_gis_vehicle_paths(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    """Return user-scoped, presentation-safe cross-camera vehicle journeys.

    Live in-memory observations are preferred. Recent durable observations are
    recovered in ONE bounded query after a restart, avoiding the previous N+1
    query pattern that could make this endpoint time out.
    """
    owned_camera_ids = _user_camera_ids(db, current_user.id)
    if not owned_camera_ids:
        return {"count": 0, "paths": []}

    with state_lock:
        memory_states = [
            item
            for item in (
                _sanitize_gis_vehicle(state)
                for state in correlationGisState.values()
                if _vehicle_belongs_to_user(state, owned_camera_ids)
            )
            if isinstance(item, dict)
        ]

    # Keep the live state authoritative, but recover additional identities from
    # PostgreSQL with a single bounded query. This is intentionally not one DB
    # query per Global Vehicle ID.
    state_by_id = {
        str(item.get("global_vehicle_id") or "").strip().upper(): item
        for item in memory_states
        if item.get("global_vehicle_id")
    }

    recovery_row_limit = min(10000, max(500, limit * 40))
    rows = (
        db.query(VehicleObservation)
        .filter(
            VehicleObservation.camera_id.in_(owned_camera_ids),
            VehicleObservation.global_vehicle_id.isnot(None),
        )
        .order_by(VehicleObservation.frame_timestamp.desc())
        .limit(recovery_row_limit)
        .all()
    )

    persisted_by_id = {}
    # DB rows are newest-first; process oldest-first so each route is naturally
    # chronological and can be bounded without losing its latest observation.
    for row in reversed(rows):
        gid = str(row.global_vehicle_id or "").strip().upper()
        if not gid or gid in state_by_id:
            continue

        metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
        gis = metadata.get("gis") if isinstance(metadata.get("gis"), dict) else {}
        latitude = _safe_coordinate(gis.get("latitude"), -90, 90)
        longitude = _safe_coordinate(gis.get("longitude"), -180, 180)
        if latitude is None or longitude is None:
            continue

        try:
            timestamp = (
                row.frame_timestamp.replace(tzinfo=timezone.utc).timestamp()
                if row.frame_timestamp else 0.0
            )
        except (AttributeError, TypeError, ValueError):
            timestamp = 0.0

        camera_id = str(row.camera_id or "").strip()
        if camera_id not in owned_camera_ids:
            continue
        camera_meta = cameraCorrelationMeta.get(camera_id, {})
        similarity = _safe_float(metadata.get("correlation_similarity"), 0.0, 0.0, 1.0)
        score = _safe_float(metadata.get("correlation_score"), similarity, 0.0, 1.0)
        observation = {
            "timestamp": timestamp,
            "camera_id": camera_id,
            "camera_name": str(camera_meta.get("camera_name") or camera_id)[:200],
            "latitude": latitude,
            "longitude": longitude,
            "pixel_x": gis.get("pixel_x"),
            "pixel_y": gis.get("pixel_y"),
            "position_type": str(gis.get("position_type") or "CAMERA_ANCHOR")[:32],
            "track_id": str(row.local_track_id or "")[:64],
            "vehicle_type": str(row.vehicle_type or "vehicle")[:64],
            "similarity": similarity,
            "correlation_score": score,
            "supporting_frames": 0,
            "consistency": 0.0,
            "strong_match": str(metadata.get("correlation_type") or "").lower() == "cross_camera",
            "match_type": str(metadata.get("correlation_type") or "persisted")[:64],
        }
        state = persisted_by_id.setdefault(gid, {
            "global_vehicle_id": gid,
            "first_seen": timestamp,
            "last_seen": timestamp,
            "last_camera_id": camera_id,
            "last_track_id": observation["track_id"],
            "vehicle_type": observation["vehicle_type"],
            "last_similarity": similarity,
            "last_correlation_score": score,
            "strong_match": False,
            "observations": [],
        })
        state["last_seen"] = timestamp
        state["last_camera_id"] = camera_id
        state["last_track_id"] = observation["track_id"]
        state["vehicle_type"] = observation["vehicle_type"]
        state["last_similarity"] = similarity
        state["last_correlation_score"] = score
        state["strong_match"] = bool(state["strong_match"] or observation["strong_match"])
        state["observations"].append(observation)
        if len(state["observations"]) > MAX_CORRELATION_GIS_OBSERVATIONS:
            del state["observations"][:-MAX_CORRELATION_GIS_OBSERVATIONS]

    state_by_id.update(persisted_by_id)
    states = list(state_by_id.values())
    states.sort(key=lambda item: _safe_float(item.get("last_seen"), 0.0, 0.0), reverse=True)

    paths = []
    for state in states:
        if len(paths) >= limit:
            break
        observations = [
            obs for obs in state.get("observations", [])
            if isinstance(obs, dict) and str(obs.get("camera_id")) in owned_camera_ids
        ]
        route = _build_gis_route(observations, owned_camera_ids)
        if not route:
            continue
        segments = _build_gis_segments(route)
        camera_count = len({point["camera_id"] for point in route})
        paths.append({
            "global_vehicle_id": state.get("global_vehicle_id"),
            "vehicle_type": str(state.get("vehicle_type") or "vehicle")[:64],
            "first_seen": state.get("first_seen"),
            "last_seen": state.get("last_seen"),
            "last_similarity": _safe_float(state.get("last_similarity"), 0.0, 0.0, 1.0),
            "last_correlation_score": _safe_float(state.get("last_correlation_score"), 0.0, 0.0, 1.0),
            "strong_match": bool(state.get("strong_match", False) or len(segments) > 0),
            "camera_count": camera_count,
            "observation_count": len(route),
            "route": route,
            "segments": segments,
            "transition_count": len(segments),
        })

    watchlist_paths = _watchlist_vehicle_gis_paths(
        db,
        current_user.id,
        owned_camera_ids,
        limit,
    )

    merged = {}
    for item in paths + watchlist_paths:
        key = str(item.get("global_vehicle_id") or "").strip()
        if not key:
            continue

        existing = merged.get(key)
        if existing is None:
            merged[key] = item
            continue

        combined_route = _build_gis_route(
            list(existing.get("route") or []) + list(item.get("route") or []),
            owned_camera_ids,
        )
        combined_segments = _build_gis_segments(combined_route)
        preferred = item if item.get("identity_source") == "WATCHLIST" else existing

        combined = dict(existing)
        for field in (
            "identity_id",
            "identity_source",
            "incident_id",
            "watchlist_entry_id",
            "reference",
            "category",
            "tracking_status",
            "correlation_global_vehicle_id",
        ):
            if preferred.get(field) is not None:
                combined[field] = preferred.get(field)

        combined["route"] = combined_route
        combined["segments"] = combined_segments
        combined["camera_count"] = len({p["camera_id"] for p in combined_route})
        combined["observation_count"] = len(combined_route)
        combined["transition_count"] = len(combined_segments)
        combined["first_seen"] = combined_route[0]["timestamp"] if combined_route else 0.0
        combined["last_seen"] = combined_route[-1]["timestamp"] if combined_route else 0.0
        combined["strong_match"] = bool(
            combined.get("strong_match")
            or item.get("strong_match")
            or combined_segments
        )
        merged[key] = combined

    final_paths = list(merged.values())
    final_paths.sort(
        key=lambda item: _safe_float(item.get("last_seen"), 0.0, 0.0),
        reverse=True,
    )
    final_paths = final_paths[:limit]

    return {"count": len(final_paths), "paths": final_paths}


@app.get("/api/gis/cameras")
def get_gis_cameras(
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    cameras = get_cameras(
        db,
        current_user.id,
    )

    result = []

    for camera in cameras:
        latitude = _safe_coordinate(
            getattr(camera, "latitude", None),
            -90,
            90,
        )
        longitude = _safe_coordinate(
            getattr(camera, "longitude", None),
            -180,
            180,
        )

        if latitude is None or longitude is None:
            continue

        camera_id = _valid_camera_id(
            camera.cam_id
        )

        result.append(
            {
                "cam_id": camera_id,
                "camera_id": camera_id,

                "camera_name": str(
                    camera.camera_name or camera_id
                )[:200],

                # ---------------------------------------------------------
                # GIS LOCATION
                # ---------------------------------------------------------
                "latitude": latitude,
                "longitude": longitude,

                "location_name": (
                    str(camera.location_name).strip()
                    if getattr(camera, "location_name", None)
                    else None
                ),

                "road_name": (
                    str(camera.road_name).strip()
                    if getattr(camera, "road_name", None)
                    else None
                ),

                "city": (
                    str(camera.city).strip()
                    if getattr(camera, "city", None)
                    else None
                ),

                "state": (
                    str(camera.state).strip()
                    if getattr(camera, "state", None)
                    else None
                ),

                "country": (
                    str(camera.country).strip()
                    if getattr(camera, "country", None)
                    else None
                ),

                # ---------------------------------------------------------
                # MONITORING
                # ---------------------------------------------------------
                "zone": str(
                    camera.zone or "not_restricted"
                )[:200],

                "source_type": str(
                    camera.source_type or ""
                )[:32],

                "is_active": bool(
                    camera.is_active
                ),

                "connection_state": cameraConnectionStates.get(
                    camera_id,
                    "ONLINE"
                    if bool(camera.is_active)
                    else "OFFLINE",
                ),

                "gis_calibrated": bool(
                    isinstance(
                        getattr(camera, "gis_calibration", None),
                        dict,
                    )
                    and getattr(
                        camera,
                        "gis_calibration",
                        {},
                    ).get("homography")
                ),
            }
        )

    return {
        "count": len(result),
        "cameras": result,
    }



@app.get("/api/gis/cameras/{camera_id}/calibration")
def get_camera_gis_calibration(
    camera_id: str,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    camera = (
        db.query(Camera)
        .filter(
            Camera.cam_id == str(camera_id).strip(),
            Camera.user_id == current_user.id,
        )
        .first()
    )
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    calibration = getattr(camera, "gis_calibration", None)
    return {
        "camera_id": str(camera.cam_id),
        "calibrated": bool(
            isinstance(calibration, dict)
            and calibration.get("homography")
        ),
        "calibration": calibration,
    }



@app.post("/api/gis/cameras/{camera_id}/calibration")
def set_camera_gis_calibration(
    camera_id: str,
    payload: dict = Body(...),
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):

    camera = (
        db.query(Camera)
        .filter(
            Camera.cam_id == str(camera_id).strip(),
            Camera.user_id == current_user.id,
        )
        .first()
    )
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    image_points = payload.get("image_points")
    geo_points = payload.get("geo_points")

    if (
        not isinstance(image_points, list)
        or not isinstance(geo_points, list)
        or len(image_points) != len(geo_points)
        or len(image_points) < 4
        or len(image_points) > 16
    ):
        raise HTTPException(
            status_code=422,
            detail="Provide 4-16 corresponding image_points and geo_points.",
        )

    image_array = []
    geo_array = []

    try:
        for image_point, geo_point in zip(image_points, geo_points):
            if (
                not isinstance(image_point, (list, tuple))
                or len(image_point) != 2
                or not isinstance(geo_point, (list, tuple))
                or len(geo_point) != 2
            ):
                raise ValueError("Each point must contain exactly two numbers.")

            x, y = float(image_point[0]), float(image_point[1])
            lat, lon = float(geo_point[0]), float(geo_point[1])

            if not all(np.isfinite(v) for v in (x, y, lat, lon)):
                raise ValueError("Points must be finite numbers.")
            if x < 0 or y < 0 or x > FRAME_WIDTH or y > FRAME_HEIGHT:
                raise ValueError(
                    f"Image point must be inside processed frame {FRAME_WIDTH}x{FRAME_HEIGHT}."
                )
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Invalid latitude/longitude.")


            image_array.append([x, y])
            geo_array.append([lon, lat])
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    src = np.asarray(image_array, dtype=np.float32)
    # Fit in metres around a local origin. Raw degree-valued lat/lon
    # homographies are poorly conditioned and make small geographic errors
    # harder to interpret. The local tangent/equirectangular plane is valid
    # for the road-sized footprints used by a fixed CCTV camera.
    latlon = np.asarray(geo_array, dtype=np.float64)
    origin_lat = float(np.mean(latlon[:, 1]))
    origin_lon = float(np.mean(latlon[:, 0]))
    meters_per_lat = 111_320.0
    meters_per_lon = meters_per_lat * max(0.1, math.cos(math.radians(origin_lat)))
    dst_metric = np.column_stack((
        (latlon[:, 0] - origin_lon) * meters_per_lon,
        (latlon[:, 1] - origin_lat) * meters_per_lat,
    )).astype(np.float32)

    matrix, mask = cv2.findHomography(
        src,
        dst_metric,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(
            payload.get("ransac_threshold_m", 2.5)
        ),
    )

    if matrix is None or matrix.shape != (3, 3):
        raise HTTPException(
            status_code=422,
            detail="Unable to calculate a stable homography. Choose well-spread road points.",
        )

    matrix = matrix / matrix[2, 2] if abs(float(matrix[2, 2])) > 1e-12 else matrix
    if not np.isfinite(matrix).all():
        raise HTTPException(status_code=422, detail="Calibration produced non-finite coefficients.")

    inlier_count = int(mask.sum()) if mask is not None else 0
    if inlier_count < 4:
        raise HTTPException(
            status_code=422,
            detail="Calibration requires at least four RANSAC inliers.",
        )

    try:
        condition_number = float(np.linalg.cond(matrix))
    except Exception:
        condition_number = float("inf")
    if not np.isfinite(condition_number) or condition_number > 1e10:
        raise HTTPException(
            status_code=422,
            detail="Calibration matrix is ill-conditioned; choose wider-spread ground points.",
        )

    projected = cv2.perspectiveTransform(
        src.reshape(-1, 1, 2),
        matrix,
    ).reshape(-1, 2)

    residuals = []
    for predicted, actual in zip(projected, dst_metric):
        residuals.append(
            math.hypot(
                float(predicted[0]) - float(actual[0]),
                float(predicted[1]) - float(actual[1]),
            )
        )

    rms_error_m = float(
        math.sqrt(
            sum(value * value for value in residuals)
            / max(1, len(residuals))
        )
    )

    max_error_m = float(max(residuals) if residuals else 0.0)
    max_allowed_error_m = _bounded_float_env(
        "GIS_MAX_CALIBRATION_RMS_ERROR_M",
        25.0,
        0.5,
        500.0,
    )
    if rms_error_m > max_allowed_error_m:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Calibration RMS error {rms_error_m:.2f}m exceeds "
                f"configured limit {max_allowed_error_m:.2f}m."
            ),
        )

    calibration = {
        "version": 2,
        "method": "pixel_to_local_metric_ground_plane_homography",
        "image_size": [int(FRAME_WIDTH), int(FRAME_HEIGHT)],
        "homography_xy": matrix.astype(float).tolist(),
        "origin": {
            "latitude": origin_lat,
            "longitude": origin_lon,
        },
        "rms_error_m": round(rms_error_m, 3),
        "max_error_m": round(max_error_m, 3),
        "inlier_count": inlier_count,
        "matrix_condition_number": round(condition_number, 3),
        "point_count": len(image_points),
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }

    camera.gis_calibration = calibration
    db.add(camera)
    db.commit()
    db.refresh(camera)
    _cache_camera_correlation_meta(camera)

    return {
        "success": True,
        "camera_id": str(camera.cam_id),
        "calibrated": True,
        "calibration": calibration,
    }


@app.get("/api/intelligence/vehicle-production-status")
def vehicle_production_status(
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    """Expose production vehicle-pipeline readiness without leaking secrets."""
    owned_cameras = get_cameras(db, current_user.id)
    calibrated = sum(
        1
        for camera in owned_cameras
        if isinstance(getattr(camera, "gis_calibration", None), dict)
        and (
            getattr(camera, "gis_calibration", {}).get("homography_xy")
            or getattr(camera, "gis_calibration", {}).get("homography")
        )
    )
    return {
        "vehicle_detection": {
            "enabled": True,
            "model_path": str(VEHICLE_MODEL_PATH),
            "model_exists": Path(VEHICLE_MODEL_PATH).is_file(),
            "confidence_threshold": VEHICLE_CONF,
            "iou_threshold": VEHICLE_IOU,
            "detector_mode": VEHICLE_DETECTOR_MODE,
            "class_ids": VEHICLE_CLASS_IDS,
        },
        "tracking": {
            "tracker": "ByteTrack",
            "reid_frame_interval": VEHICLE_REID_RUN_EVERY_N_FRAMES,
            "reid_min_crop_quality": VEHICLE_REID_MIN_CROP_QUALITY,
        },
        "correlation": {
            "architecture": ["ANPR/LPR", "Vehicle Re-ID", "Metadata", "Correlation Engine"],
            "anpr_weight": 0.35,
            "reid_weight": 0.40,
            "metadata_weight": 0.25,
            "statuses": ["CONFIRMED", "PROBABLE", "POSSIBLE", "UNKNOWN", "REJECTED"],
        },
        "gis": {
            "owned_cameras": len(owned_cameras),
            "calibrated_cameras": calibrated,
            "calibration_required_for_exact_vehicle_position": True,
        },
        "deepstream": {
            "enabled": os.getenv("DEEPSTREAM_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
            "mode": os.getenv("VIDEO_PIPELINE_BACKEND", "opencv").strip().lower(),
        },
    }




@app.get("/api/ai/tensorrt/status")
def tensorrt_status(current_user: User = Depends(get_current_user)):
    """Return non-secret TensorRT readiness and artifact validation state."""
    return tensorrt_runtime.status(validate_files=True)

@app.get("/api/ai/scheduler/status")
def ai_scheduler_status(current_user: User = Depends(get_current_user)):
    """Return non-secret AI scheduler/resource telemetry."""
    state = _sync_ai_scheduler_metrics()
    state.pop("camera_pending", None)
    return state

@app.get("/api/ai/vehicle-pipeline/status")
def vehicle_pipeline_status(current_user: User = Depends(get_current_user)):
    """Return non-secret Phase 9 vehicle pipeline readiness and model status."""
    return {
        "phase": 9,
        "vehicle_detector": vehicle_analytics_engine.status(),
        "tensorrt": tensorrt_runtime.status(validate_files=True),
        "reid": {"enabled": True, "provider": os.getenv("VEHICLE_REID_PROVIDER", "auto")},
        "anpr": {
            "enabled": True,
            "plate_model": os.getenv(
                "ANPR_PLATE_MODEL", "models/license_plate_yolov8m.pt"
            ),
            "ocr_engine": os.getenv("ANPR_OCR_ENGINE", "awiros_paddle"),
            "recognizer": os.getenv("ANPR_RECOGNIZER_NAME", "Awiros-ANPR-OCR"),
        },
        "observation_validation": {"enabled": True},
        "correlation": {"enabled": True},
    }

@app.get("/api/intelligence/status")
def advanced_intelligence_status(current_user: User = Depends(get_current_user)):
    return {
        "enabled": ADVANCED_INTELLIGENCE_ENABLED,
        "camera_tamper": CAMERA_TAMPER_ENABLED,
        "frozen_frame": FROZEN_FRAME_ENABLED,
        "risk_scoring": RISK_SCORING_ENABLED,
        "incident_correlation": INCIDENT_CORRELATION_ENABLED,
        "evidence_hashing": EVIDENCE_HASHING_ENABLED,
        "adaptive_pipeline": ADAPTIVE_PIPELINE_ENABLED,
        "person_appearance": PERSON_APPEARANCE_ENABLED,
        "crowd_intelligence": CROWD_INTELLIGENCE_ENABLED,
        "anpr_character_voting": ANPR_CHARACTER_VOTING_ENABLED,
        "confidence_calibration": CONFIDENCE_CALIBRATION_ENABLED,
        "ollama_plate_verification": ollama_plate_verifier.health(),
        "ollama_investigation_summaries": {
            "enabled": investigation_summary_worker.enabled,
            "model": investigation_summary_worker.model,
        },
        "person_correlation": {
            "enabled": PERSON_APPEARANCE_ENABLED,
            "active_global_person_ids": len(personGisState),
        },
        "edsr": advanced_intelligence.edsr.status(),
        "cameras": {
            cam_id: {
                "quality": cameraQualityTelemetry.get(cam_id),
                "adaptive_policy": cameraAdaptivePolicies.get(cam_id),
                "tamper": cameraTamperTelemetry.get(cam_id),
                "frozen": cameraFrozenTelemetry.get(cam_id),
            }
            for cam_id in list(cameraQualityTelemetry.keys())
        },
    }

class AlertFeedbackRequest(BaseModel):
    decision: str
    note: str = ""

class VehicleIdentityCorrectionRequest(BaseModel):
    action: str
    source_global_vehicle_id: str
    target_global_vehicle_id: str | None = None
    observation_ids: list[int] = Field(default_factory=list, max_length=5000)
    reason: str = Field(min_length=5, max_length=1000)


@app.post("/api/alerts/{alert_id}/feedback")
def add_alert_feedback(
    alert_id: int,
    req: AlertFeedbackRequest,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    alert_obj = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert_obj:
        raise HTTPException(status_code=404, detail="Alert not found")

    if (
        not _can_view_all_alerts(current_user)
        and int(getattr(alert_obj, "user_id", 0) or 0) != int(current_user.id)
    ):
        raise HTTPException(status_code=404, detail="Alert not found")

    if not ADVANCED_INTELLIGENCE_ENABLED:
        raise HTTPException(status_code=503, detail="Advanced intelligence disabled")

    try:
        record = advanced_intelligence.operator_feedback(
            alert_id=str(alert_id),
            decision=req.decision,
            operator_id=str(current_user.id),
            note=req.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"message": "Feedback saved", "feedback": record}


@app.get("/system/capacity")
def system_capacity(
    current_user: User = Depends(get_current_user),
):
    active_count = int(get_active_live_camera_count() or 0)
    capacity = get_system_capacity() or {}

    max_live_cameras = int(
        capacity.get("max_live_cameras", 0) or 0
    )

    return {
        "active_live_cameras": active_count,
        "max_live_cameras": max_live_cameras,
        "available_camera_slots": max(
            0,
            max_live_cameras - active_count,
        ),
        "system": capacity,
    }

def _can_view_all_alerts(current_user: User) -> bool:
    """Return True only for the platform-wide Super Administrator role."""
    normalized_role = (
        str(getattr(current_user, "role", "") or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    return normalized_role == "super_admin"


@app.get("/alerts/recover")
def recoverAlerts(
    after_id: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    """Durable missed-alert replay for WebSocket reconnects."""
    alerts = get_alerts_after_id(
        db=db,
        user_id=current_user.id,
        after_id=after_id,
        limit=limit,
        include_all_users=_can_view_all_alerts(current_user),
        visible_camera_ids=(
            None
            if _can_view_all_alerts(current_user)
            else _user_camera_ids(db, int(current_user.id))
        ),
    )

    return {
        "after_id": after_id,
        "count": len(alerts),
        "alerts": [
            serialize_alert(alert)
            for alert in alerts
        ],
    }


@app.get("/alerts")
def getAllAlerts(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    alerts = get_alerts(
        db=db,
        limit=limit,
        user_id=current_user.id,
        include_all_users=_can_view_all_alerts(current_user),
        visible_camera_ids=(
            None
            if _can_view_all_alerts(current_user)
            else _user_camera_ids(db, int(current_user.id))
        ),
    )

    return {
        "count": len(alerts),
        "alerts": [
            serialize_alert(alert)
            for alert in alerts
        ],
    }


# Keep the dynamic route after all fixed /alerts routes.
@app.get("/alerts/{camID}")
def getAlerts(
    camID: str,
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    if _can_view_all_alerts(current_user):
        # Historical alerts remain investigable even if the camera inventory row
        # was removed.  Super-admin access is already enforced by role.
        camera = (
            db.query(Camera)
            .filter(Camera.cam_id == camID)
            .first()
        )
    else:
        camera = get_camera(
            db,
            camID,
            current_user.id,
        )
        if not camera:
            # Allow an owned historical alert source whose Camera row has been
            # removed, without exposing another user's camera identifier.
            owned_alert_exists = (
                db.query(Alert.id)
                .filter(
                    Alert.cam_id == camID,
                    Alert.user_id == current_user.id,
                )
                .first()
                is not None
            )
            if not owned_alert_exists:
                raise HTTPException(
                    status_code=404,
                    detail="Camera not found",
                )

    alerts = get_alerts(
        db=db,
        cam_id=camID,
        limit=limit,
        user_id=current_user.id,
        include_all_users=_can_view_all_alerts(current_user),
    )

    return {
        "camera_id": camID,
        "alert_count": len(alerts),
        "alerts": [
            serialize_alert(alert)
            for alert in alerts
        ],
    }
@app.post("/api/vehicles/identity/correction")
def vehicle_identity_correction(
    request: VehicleIdentityCorrectionRequest,
    req: Request,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    try:
        result = correct_identity(
            db,
            user_id=int(current_user.id),
            action=request.action,
            source_global_vehicle_id=request.source_global_vehicle_id,
            target_global_vehicle_id=request.target_global_vehicle_id,
            observation_ids=request.observation_ids,
            reason=request.reason,
            source_ip=req.client.host if req.client else None,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc



@app.get("/snapshot/{snapshot_id}")
def getSnapshot(
    snapshot_id: int,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    if _can_view_all_alerts(current_user):
        snapshot = (
            db.query(Snapshot)
            .filter(Snapshot.id == snapshot_id)
            .first()
        )
    else:
        snapshot = get_snapshot_for_user(
            db=db,
            snapshot_id=snapshot_id,
            user_id=current_user.id,
        )
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    headers = {}
    if getattr(snapshot, "sha256", None):
        headers["X-Evidence-SHA256"] = str(snapshot.sha256)
    headers["X-Evidence-Original"] = (
        "true" if bool(getattr(snapshot, "is_original", False)) else "false"
    )

    return Response(
        content=snapshot.image_data,
        media_type=snapshot.image_type or "image/jpeg",
        headers=headers,
    )

@app.get("/zones/{camID}")
def getZones(
    camID: str,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    zones = get_camera_zones(
        db=db,
        cam_id=camID,
        user_id=current_user.id,
    )

    if zones is None:
        raise HTTPException(
            status_code=404,
            detail="Camera not found"
        )

    return {
        "cam_id": camID,
        "zones": zones,
    }


@app.post("/zones/{camID}")
def updateZones(
    camID: str,
    req: ZonesUpdateRequest,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    camera = update_camera_zones(
        db=db,
        cam_id=camID,
        user_id=current_user.id,
        zones=req.zones,
    )

    if camera is None:
        raise HTTPException(
            status_code=404,
            detail="Camera not found"
        )
    zone_manager.set_zones(camID, req.zones)

    return {
        "message": "Zones updated",
        "cam_id": camID,
        "zones": camera.zones,
    }

@app.put("/camera/{camID}/zone")
def update_camera_zone(
    camID: str,
    req: CameraZoneUpdateRequest,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    camera = get_camera(
        db,
        camID,
        current_user.id,
    )

    if not camera:
        raise HTTPException(
            status_code=404,
            detail="Camera not found",
        )

    camera.zone = req.zone

    db.commit()
    db.refresh(camera)
    camera_config.set_mode(
        camID,
        camera.zone
    )

    return {
        "cam_id": camID,
        "zone": camera.zone,
    }



@app.get("/auth/csrf")
def csrf(response: Response):
    csrf_token = set_csrf_cookie(response)
    return {"csrf_token": csrf_token}

@app.post("/auth/login")
@limiter.limit(os.getenv("RATE_LIMIT_LOGIN", "10/minute"))
def login(
    request: Request,
    response: Response,
    req: LoginRequest,
    db: Session = Depends(getDB),
):
    """Authenticate password first, then require MFA before issuing a session."""
    email = req.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()

    # Keep the same public error for unknown account and bad password.
    if not user or not verify_password(req.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User account disabled")

    mfa = (
        db.query(UserMFA)
        .filter(UserMFA.user_id == user.id, UserMFA.enabled.is_(True))
        .first()
    )

    if mfa is not None:
        challenge_ttl = max(60, min(600, int(os.getenv("MFA_CHALLENGE_TTL_SECONDS", "300"))))
        now = datetime.now(timezone.utc)
        challenge_jti = str(uuid.uuid4())
        payload = {
            "sub": str(user.id),
            "user_id": user.id,
            "role": user.role,
            "ver": int(user.token_version or 0),
            "jti": challenge_jti,
            "type": "mfa_challenge",
            "iss": JWT_ISSUER,
            "aud": JWT_AUDIENCE,
            "iat": now,
            "exp": now + timedelta(seconds=challenge_ttl),
        }
        mfa_token = jwt.encode(payload, PRIVATE_KEY, algorithm=ALGORITHM)

        # Production requires Redis so MFA challenges are one-time credentials.
        if redis_client is None:
            if IS_PROD:
                raise HTTPException(status_code=503, detail="Authentication service unavailable")
        else:
            try:
                redis_client.set(
                    make_key(f"auth:mfa-challenge:{challenge_jti}"),
                    str(user.id),
                    ex=challenge_ttl,
                    nx=True,
                )
            except Exception:
                logger.exception("Unable to persist MFA challenge | user_id=%s", user.id)
                raise HTTPException(status_code=503, detail="Authentication service unavailable")

        # No access/refresh cookies are issued before MFA succeeds.
        clear_auth_cookies(response)
        return MFAChallengeResponse(
            mfa_required=True,
            mfa_token=mfa_token,
            expires_in=challenge_ttl,
        )

    try:
        token_payload = {
            "sub": str(user.id),
            "user_id": user.id,
            "role": user.role,
            "ver": int(user.token_version or 0),
        }
        access_token = create_access_token(token_payload)
        refresh_token, token_jti, expires_at = create_refresh_token(token_payload)
        db.add(RefreshToken(user_id=user.id, token_jti=token_jti, expires_at=expires_at, revoked=False))
        db.commit()
        set_auth_cookies(response, access_token, refresh_token)
        csrf_token = set_csrf_cookie(response)
        return {
            "message": "Login successful",
            "mfa_required": False,
            "csrf_token": csrf_token,
            "access_expires_in_seconds": 60 * 60 * ACCESS_TOKEN_EXPIRE_HOURS,
            "user": {
                "id": user.id,
                "full_name": user.full_name,
                "email": user.email,
                "role": user.role,
                "department": user.department,
                "manager_id": user.manager_id,
                "permissions": permissions_for(user.role),
            },
        }
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        logger.exception("Login failed | user_id=%s", user.id)
        raise HTTPException(status_code=500, detail="Login failed")


@app.post("/auth/mfa/login/verify")
@limiter.limit(os.getenv("RATE_LIMIT_MFA_LOGIN_VERIFY", "10/minute"))
def verify_login_mfa(
    request: Request,
    response: Response,
    req: MFALoginVerifyRequest,
    db: Session = Depends(getDB),
):
    """Consume a one-time MFA challenge, verify TOTP/recovery code, then issue the real session."""
    try:
        payload = jwt.decode(
            req.mfa_token,
            PUBLIC_KEY,
            algorithms=[ALGORITHM],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_iss": True,
                "verify_aud": True,
            },
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired MFA challenge")

    if payload.get("type") != "mfa_challenge":
        raise HTTPException(status_code=401, detail="Invalid or expired MFA challenge")

    try:
        user_id = int(payload.get("sub"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid or expired MFA challenge")

    challenge_jti = str(payload.get("jti") or "").strip()
    if not challenge_jti:
        raise HTTPException(status_code=401, detail="Invalid or expired MFA challenge")

    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid or expired MFA challenge")

    if int(payload.get("ver", -1)) != int(user.token_version or 0):
        raise HTTPException(status_code=401, detail="Invalid or expired MFA challenge")

    # Atomically consume the challenge BEFORE factor verification. A failed
    # factor attempt therefore requires a fresh password login and cannot
    # brute-force one signed challenge repeatedly.
    if redis_client is None:
        if IS_PROD:
            raise HTTPException(status_code=503, detail="Authentication service unavailable")
    else:
        key = make_key(f"auth:mfa-challenge:{challenge_jti}")
        try:
            stored_user = redis_client.getdel(key)
        except Exception:
            logger.exception("Unable to consume MFA challenge | user_id=%s", user_id)
            raise HTTPException(status_code=503, detail="Authentication service unavailable")
        if stored_user is None or str(stored_user) != str(user_id):
            raise HTTPException(status_code=401, detail="Invalid or expired MFA challenge")

    try:
        verify_mfa(
            db,
            user_id=user_id,
            code=req.code,
            allow_recovery_code=True,
            source_ip=request.client.host if request.client else None,
        )
        db.refresh(user)

        token_payload = {
            "sub": str(user.id),
            "user_id": user.id,
            "role": user.role,
            "ver": int(user.token_version or 0),
        }
        access_token = create_access_token(token_payload)
        refresh_token, token_jti, expires_at = create_refresh_token(token_payload)
        db.add(RefreshToken(user_id=user.id, token_jti=token_jti, expires_at=expires_at, revoked=False))
        db.commit()
        set_auth_cookies(response, access_token, refresh_token)
        csrf_token = set_csrf_cookie(response)
        return {
            "message": "Login successful",
            "mfa_required": False,
            "csrf_token": csrf_token,
            "access_expires_in_seconds": 60 * 60 * ACCESS_TOKEN_EXPIRE_HOURS,
            "user": {
                "id": user.id,
                "full_name": user.full_name,
                "email": user.email,
                "role": user.role,
                "department": user.department,
                "manager_id": user.manager_id,
                "permissions": permissions_for(user.role),
            },
        }
    except PermissionError:
        raise HTTPException(status_code=401, detail="Invalid MFA code")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid MFA code")
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        logger.exception("MFA login completion failed | user_id=%s", user_id)
        raise HTTPException(status_code=500, detail="Login failed")


@app.post("/auth/refresh")
@limiter.limit(os.getenv("RATE_LIMIT_REFRESH", "20/minute"))
def refresh_access_token(
    request: Request,
    response: Response,
    db: Session = Depends(getDB),
):
    refresh_token = request.cookies.get(REFRESH_COOKIE_NAME)

    if not refresh_token:
        raise HTTPException(
            status_code=401,
            detail="Missing refresh token",
        )

    payload = decode_token(refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=401,
            detail="Invalid refresh token",
        )

    token_jti = str(payload.get("jti") or "").strip()
    subject = payload.get("sub")

    if not token_jti or not subject:
        raise HTTPException(
            status_code=401,
            detail="Invalid refresh token",
        )

    try:
        user_id = int(subject)
    except (TypeError, ValueError):
        clear_auth_cookies(response)
        raise HTTPException(
            status_code=401,
            detail="Invalid refresh token subject",
        )

    # Lock the refresh-token row so two concurrent refresh requests cannot
    # both rotate the same token successfully.
    stored_token = (
        db.query(RefreshToken)
        .filter(
            RefreshToken.token_jti == token_jti,
            RefreshToken.user_id == user_id,
        )
        .with_for_update()
        .first()
    )

    if not stored_token or stored_token.revoked:
        clear_auth_cookies(response)
        raise HTTPException(
            status_code=401,
            detail="Invalid refresh token",
        )

    now = indian_time()

    if stored_token.expires_at <= now:
        stored_token.revoked = True
        db.commit()
        clear_auth_cookies(response)
        raise HTTPException(
            status_code=401,
            detail="Refresh token expired",
        )

    user = (
        db.query(User)
        .filter(
            User.id == user_id,
            User.is_active.is_(True),
        )
        .first()
    )

    if not user:
        stored_token.revoked = True
        db.commit()
        clear_auth_cookies(response)
        raise HTTPException(
            status_code=401,
            detail="User not active",
        )

    # Keep access and refresh token subject semantics identical to login:
    # `sub` is always the immutable numeric user ID.
    token_payload = {
        "sub": str(user.id),
        "user_id": user.id,
        "role": user.role,
        "ver": int(user.token_version or 0),
    }

    try:
        stored_token.revoked = True

        access_token = create_access_token(token_payload)

        new_refresh_token, new_jti, expires_at = (
            create_refresh_token(token_payload)
        )

        db.add(
            RefreshToken(
                user_id=user.id,
                token_jti=new_jti,
                expires_at=expires_at,
                revoked=False,
            )
        )

        db.commit()

    except Exception:
        db.rollback()
        logger.exception(
            "Refresh-token rotation failed | user_id=%s",
            user.id,
        )
        raise HTTPException(
            status_code=500,
            detail="Token refresh failed",
        )

    set_auth_cookies(
        response,
        access_token,
        new_refresh_token,
    )
    csrf_token = set_csrf_cookie(response)

    return {
        "message": "Token refreshed",
        "csrf_token": csrf_token,
    }


@app.post("/auth/logout")
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(getDB),
):
    refresh_token = request.cookies.get(REFRESH_COOKIE_NAME)

    if refresh_token:
        try:
            payload = decode_token(refresh_token)

            if payload.get("type") == "refresh":
                token_jti = payload.get("jti")

                stored_token = db.query(RefreshToken).filter(
                    RefreshToken.token_jti == token_jti
                ).first()

                if stored_token:
                    stored_token.revoked = True
                    db.commit()

        except Exception:
            db.rollback()

    clear_auth_cookies(response)

    return {"message": "Logout successful"}

@app.get("/auth/me")
def get_me(
    current_user: User = Depends(get_current_user),
):
    return {
        "id": current_user.id,
        "full_name": current_user.full_name,
        "email": current_user.email,
        "role": current_user.role,
        "department": current_user.department,
        "manager_id": current_user.manager_id,
        "permissions": permissions_for(current_user.role),
        "is_active": current_user.is_active
    }