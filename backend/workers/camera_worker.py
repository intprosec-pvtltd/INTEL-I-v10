from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import os
import secrets
import signal
import socket
import threading
import time
import uuid
from typing import Any

import cv2
import psutil
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from cache.cameraLease import (
    CAMERA_LEASE_TTL_SECONDS,
    CAMERA_RUNTIME_TTL_SECONDS,
    WORKER_HEARTBEAT_TTL_SECONDS,
    CameraLease,
    acquire_camera_lease,
    delete_camera_runtime,
    delete_worker_heartbeat,
    release_camera_lease,
    renew_camera_lease,
    set_camera_runtime,
    set_worker_heartbeat,
    list_worker_heartbeats,
)
from db.crud import get_camera_for_worker, get_requested_running_cameras
from cache.redisState import get_camera_state, increment_camera_frame
from db.database import SessionLocal
from db.model import CameraAIProfile
from monitoring.metrics import (
    PROMETHEUS_TOKEN,
    camera_lease_events_total,
    distributed_worker_owned_cameras,
)
from services.cameraSourceResolver import resolve_camera_row
from services.realtimeBus import publish_realtime_event
from runtime.analytics_runtime import AnalyticsRuntime
from inference.client import CentralInferenceClient
from inference.settings import SETTINGS as INFERENCE_SETTINGS
from services.cameraPipeline import CameraPipeline
from services.workerBalancer import preferred_worker_id, worker_load_score
from inference.policy import CAMERA_POLICIES

logger = logging.getLogger("intel_i.distributed_camera_worker")

# This module is the dedicated camera-worker entry point.  The shared backend
# .env correctly uses INTEL_I_PROCESS_ROLE=api for Uvicorn, but load_dotenv()
# may populate that value before this module reaches AnalyticsRuntime.load().
# Always override it here so the worker cannot accidentally identify itself as
# the API process and fail the GPU-runtime ownership guard.
os.environ["INTEL_I_PROCESS_ROLE"] = "camera-worker"


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _worker_id() -> str:
    configured = str(os.getenv("INTEL_I_WORKER_ID", "")).strip()
    if configured:
        return configured[:160]
    host = str(os.getenv("HOSTNAME") or socket.gethostname() or "worker")
    return f"{host}-{os.getpid()}-{uuid.uuid4().hex[:8]}"[:160]


WORKER_ID = _worker_id()
MAX_CAMERAS = _env_int("INTEL_I_MAX_CAMERAS_PER_WORKER", 5, 1, 100)
_requested_discovery_seconds = _env_float("INTEL_I_CAMERA_DISCOVERY_SECONDS", 3.0, 0.5, 60.0)
_requested_renew_seconds = _env_float("INTEL_I_CAMERA_LEASE_RENEW_SECONDS", 10.0, 1.0, 60.0)
# Prevent misconfiguration from allowing a lease/heartbeat/runtime TTL to expire
# before this worker has a chance to refresh ownership telemetry.
LEASE_RENEW_SECONDS = min(
    _requested_renew_seconds,
    max(1.0, CAMERA_LEASE_TTL_SECONDS / 3.0),
    max(1.0, CAMERA_RUNTIME_TTL_SECONDS / 3.0),
)
DISCOVERY_SECONDS = min(
    _requested_discovery_seconds,
    max(0.5, WORKER_HEARTBEAT_TTL_SECONDS / 3.0),
    LEASE_RENEW_SECONDS,
)
SHUTDOWN_WAIT_SECONDS = _env_float("INTEL_I_WORKER_SHUTDOWN_WAIT_SECONDS", 8.0, 1.0, 60.0)
PREVIEW_PORT = _env_int("INTEL_I_WORKER_PREVIEW_PORT", 9101, 1024, 65535)
PREVIEW_HOST = str(os.getenv("INTEL_I_WORKER_PREVIEW_HOST", "0.0.0.0")).strip() or "0.0.0.0"
PREVIEW_TOKEN = str(os.getenv("INTEL_I_WORKER_PREVIEW_TOKEN", "")).strip()
ENV = str(os.getenv("ENV", "dev")).strip().lower()

if ENV == "prod" and not PREVIEW_TOKEN:
    raise RuntimeError("INTEL_I_WORKER_PREVIEW_TOKEN is required in production")
if not PREVIEW_TOKEN:
    PREVIEW_TOKEN = "intel-i-local-preview"


def _preview_base_url() -> str:
    explicit = str(os.getenv("INTEL_I_WORKER_PREVIEW_ADVERTISE_URL", "")).strip().rstrip("/")
    if explicit:
        return explicit
    pod_ip = str(os.getenv("POD_IP", "")).strip()
    if pod_ip:
        return f"http://{pod_ip}:{PREVIEW_PORT}"
    # Safe local-process default. Docker/Kubernetes deployments should set an
    # explicit URL or POD_IP so the API process can reach the owning worker.
    return f"http://127.0.0.1:{PREVIEW_PORT}"


PREVIEW_BASE_URL = _preview_base_url()


@dataclass
class OwnedCamera:
    camera_pk: int
    user_id: int
    cam_id: str
    source_type: str
    lease: CameraLease
    pipeline: CameraPipeline
    last_lease_success: float
    claimed_at: float
    last_frame_count: int = 0
    last_runtime_sample: float = 0.0


class DistributedWorker:
    def __init__(self):
        self.started_at = time.time()
        self.stop_event = threading.Event()
        self.owned: dict[int, OwnedCamera] = {}
        self._last_renew = 0.0
        self._preview_server: uvicorn.Server | None = None
        self._preview_thread: threading.Thread | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None
        self.runtime: AnalyticsRuntime | None = None
        self.readiness = None
        self.readiness_state: dict[str, Any] = {"status": "STARTING"}
        self._last_readiness_check = 0.0
        self.legacy = None

    # ------------------------------------------------------------------
    # Existing analytics bridge
    # ------------------------------------------------------------------
    def _initialize_analytics_runtime(self) -> None:
        # In central mode stream workers never load heavy GPU models. They import
        # the legacy control helpers only and submit sampled frames to the central service.
        from inference.triton_client import triton_enabled
        if triton_enabled() and not (INFERENCE_SETTINGS.central_enabled and INFERENCE_SETTINGS.split_enabled):
            raise RuntimeError("Triton mode requires centralized split inference")
        if INFERENCE_SETTINGS.central_enabled and INFERENCE_SETTINGS.split_enabled:
            import importlib
            legacy = importlib.import_module("main")
            runtime = CentralInferenceClient()
            runtime.health()
            self.runtime = runtime
            self.legacy = legacy
            logger.info("Central inference enabled | url=%s", INFERENCE_SETTINGS.service_url)
        else:
            runtime = AnalyticsRuntime()
            runtime.load()
            self.runtime = runtime
            legacy = runtime.legacy
            self.legacy = legacy
        from services.workerReadiness import default_worker_readiness
        self.readiness = None if isinstance(runtime, CentralInferenceClient) else default_worker_readiness(runtime)

        # Worker-side async loop supports the existing async alert pipeline.
        loop = asyncio.new_event_loop()
        self._async_loop = loop

        def _run_loop():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._async_thread = threading.Thread(
            target=_run_loop,
            name=f"{WORKER_ID}-async",
            daemon=True,
        )
        self._async_thread.start()
        legacy.main_loop = loop

        async def _publish_alert(payload: dict):
            publish_realtime_event({
                "kind": "alert",
                "payload": payload,
                "worker_id": WORKER_ID,
            })

        async def _publish_position(payload: dict):
            publish_realtime_event({
                "kind": "position",
                "payload": payload,
                "worker_id": WORKER_ID,
            })

        # Existing processFrame/handle_alerts resolve these globals at runtime,
        # so this preserves the processing logic while routing UI events to the
        # API process rather than a worker-local WebSocket registry.
        legacy.sendAlert = _publish_alert
        legacy.broadcast_vehicle_position = _publish_position

        if isinstance(runtime, CentralInferenceClient):
            state = runtime.health()
            self.readiness_state = {"status": "READY", "inference": state}
        else:
            from services.runtimeModelReadiness import verify_models
            required = str(os.getenv("MODEL_READINESS_REQUIRED", "true" if ENV == "prod" else "false")).lower() in {"1", "true", "yes", "on"}
            state = verify_models(force=True)
            if required and state.get("status") != "READY":
                raise RuntimeError(f"Required AI models are not ready: {state}")
        logger.info("Worker inference readiness | worker_id=%s state=%s", WORKER_ID, state)

        # CameraPipeline performs explicit analytics sampling; it does not use
        # the legacy combined capture/scheduler loop.

    # ------------------------------------------------------------------
    # Internal preview server
    # ------------------------------------------------------------------
    def _create_preview_app(self) -> FastAPI:
        app = FastAPI(title="INTEL-I Worker Preview", docs_url=None, redoc_url=None)

        def _authorize(token: str | None):
            if not token or not secrets.compare_digest(token, PREVIEW_TOKEN):
                raise HTTPException(status_code=404, detail="Not found")

        @app.get("/internal/health")
        def health(x_intel_i_worker_token: str | None = Header(default=None)):
            _authorize(x_intel_i_worker_token)
            return {
                **self.readiness_state,
                "worker_id": WORKER_ID,
                "owned_cameras": len(self.owned),
                "capacity": MAX_CAMERAS,
            }

        @app.get("/metrics", include_in_schema=False)
        def metrics(request: Request):
            # Keep the worker metrics endpoint consistent with the public API:
            # production Prometheus scrapes require a Bearer token, while
            # development remains convenient for local validation.
            if ENV == "prod":
                expected = f"Bearer {PROMETHEUS_TOKEN or ''}"
                supplied = request.headers.get("Authorization", "")
                if not PROMETHEUS_TOKEN or not secrets.compare_digest(supplied, expected):
                    raise HTTPException(status_code=404, detail="Not found")
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

        @app.get("/internal/stream/{cam_id}")
        async def stream(cam_id: str, x_intel_i_worker_token: str | None = Header(default=None)):
            _authorize(x_intel_i_worker_token)
            legacy = self.legacy
            if legacy is None:
                raise HTTPException(status_code=503, detail="Worker not ready")
            if not any(item.cam_id == str(cam_id) for item in self.owned.values()):
                raise HTTPException(status_code=404, detail="Not found")

            async def _frames():
                while not self.stop_event.is_set():
                    with legacy.state_lock:
                        lock = legacy.cameraLocks.get(cam_id)
                    jpeg = None
                    if lock:
                        with lock:
                            jpeg = legacy.latestJpegs.get(cam_id)
                    if jpeg:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
                            b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n"
                        )
                    await asyncio.sleep(max(0.02, float(getattr(legacy, "STREAM_IDLE_SLEEP_SECONDS", 0.04))))

            return StreamingResponse(
                _frames(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        return app

    def _start_preview_server(self) -> None:
        config = uvicorn.Config(
            self._create_preview_app(),
            host=PREVIEW_HOST,
            port=PREVIEW_PORT,
            log_level=str(os.getenv("INTEL_I_WORKER_PREVIEW_LOG_LEVEL", "warning")),
            access_log=False,
        )
        self._preview_server = uvicorn.Server(config)
        self._preview_thread = threading.Thread(
            target=self._preview_server.run,
            name=f"{WORKER_ID}-preview",
            daemon=True,
        )
        self._preview_thread.start()

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------
    def _claim(self, camera) -> None:
        if len(self.owned) >= MAX_CAMERAS:
            return
        if int(camera.id) in self.owned:
            return

        lease = acquire_camera_lease(camera.id, camera.cam_id, WORKER_ID)
        if lease is None:
            return
        camera_lease_events_total.labels(event="acquired").inc()

        set_camera_runtime(
            camera.id,
            cam_id=camera.cam_id,
            user_id=camera.user_id,
            worker_id=WORKER_ID,
            state="CLAIMED",
            source_type=camera.source_type,
            preview_base_url=PREVIEW_BASE_URL,
        )

        try:
            resolved = resolve_camera_row(camera)
            legacy = self.legacy
            if legacy is None:
                raise RuntimeError("Analytics runtime is not initialized")

            # Preserve existing camera-local configuration and correlation/GIS
            # metadata exactly as the legacy start endpoint did.
            legacy._cache_camera_correlation_meta(camera)
            if camera.zones:
                legacy.zone_manager.set_zones(camera.cam_id, camera.zones)
            legacy.camera_config.set_mode(camera.cam_id, camera.zone)
            profile=None
            if getattr(camera,"ai_profile_id",None):
                with SessionLocal() as policy_db:
                    profile=policy_db.query(CameraAIProfile).filter(CameraAIProfile.id==int(camera.ai_profile_id),CameraAIProfile.user_id==int(camera.user_id)).first()
            CAMERA_POLICIES.configure(
                str(camera.cam_id),
                getattr(profile,"configuration",None),
                processing_fps=getattr(profile,"processing_fps",None),
                evidence_enabled=getattr(profile,"evidence_enabled",None),
                zone=getattr(camera,"zone",None),
                profile_name=getattr(profile,"name",None),
            )

            camera.resolved_source_type = resolved.source_type
            camera.connection_state = "OFFLINE"
            camera.last_health_check = legacy.datetime.now(legacy.timezone.utc).replace(tzinfo=None)
            with SessionLocal() as db:
                row = get_camera_for_worker(db, camera.id)
                if row is not None:
                    row.resolved_source_type = resolved.source_type
                    row.connection_state = "OFFLINE"
                    row.last_health_check = camera.last_health_check
                    db.commit()

            with legacy.state_lock:
                legacy.cameraLocks.setdefault(str(camera.cam_id), threading.Lock())
                legacy.cameraFrameCount[str(camera.cam_id)] = 0

            def _capture_factory(_camera):
                return legacy.open_timestamped_capture(
                    resolved.source,
                    resolved.source_type,
                    opencv_factory=legacy._open_video_capture,
                    timeout_seconds=legacy.RTSP_READ_TIMEOUT_SECONDS,
                )

            preview_state = {"processed_at": None}

            def _publish_preview(frame, _item, *, processed=False):
                # Keep boxes on the exact frame that produced them. Raw fallback
                # resumes after a stalled AI service, rather than freezing forever.
                last_processed = preview_state["processed_at"]
                if not processed and last_processed is not None and time.monotonic() - last_processed < 3.0:
                    return
                if frame is None: return
                ok,encoded=cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),int(getattr(legacy,"STREAM_JPEG_QUALITY",85))])
                if not ok: return
                cam_id=str(camera.cam_id)
                with legacy.state_lock: lock=legacy.cameraLocks.get(cam_id)
                if lock is not None:
                    with lock: legacy.latestJpegs[cam_id]=encoded.tobytes()
                increment_camera_frame(cam_id)

            def _publish_processed(_frame, _item):
                _publish_preview(_frame, _item, processed=True)
                preview_state["processed_at"] = time.monotonic()
                cam_id=str(camera.cam_id)
                with legacy.state_lock:
                    legacy.cameraFrameCount[cam_id]=legacy.cameraFrameCount.get(cam_id,0)+1

            analytics_fps=float(os.getenv("INTEL_I_ANALYTICS_FPS",str(INFERENCE_SETTINGS.active_fps)))
            pipeline=CameraPipeline(
                camera=camera,runtime=self.runtime,capture_factory=_capture_factory,
                analytics_fps=analytics_fps,preview_fps=INFERENCE_SETTINGS.preview_fps,
                on_preview=_publish_preview,on_processed=_publish_processed,
                reconnect_initial_seconds=legacy.RTSP_RECONNECT_INITIAL_SECONDS,
                reconnect_max_seconds=legacy.RTSP_RECONNECT_MAX_SECONDS,
            )
            legacy.set_camera_state(
                cam_id=str(camera.cam_id), source=str(camera.cam_id),
                source_type=resolved.source_type, status=True,
                state=legacy.CameraState.ONLINE, frame_count=0, worker_running=True,
            )
            pipeline.start()

            now = time.monotonic()
            self.owned[int(camera.id)] = OwnedCamera(
                camera_pk=int(camera.id),
                user_id=int(camera.user_id),
                cam_id=str(camera.cam_id),
                source_type=str(resolved.source_type),
                lease=lease,
                pipeline=pipeline,
                last_lease_success=now,
                claimed_at=time.time(),
                last_frame_count=0,
                last_runtime_sample=now,
            )
            set_camera_runtime(
                camera.id,
                cam_id=camera.cam_id,
                user_id=camera.user_id,
                worker_id=WORKER_ID,
                state="RUNNING",
                source_type=resolved.source_type,
                preview_base_url=PREVIEW_BASE_URL,
            )
            logger.info(
                "Camera claimed | worker_id=%s camera_pk=%s cam_id=%s source_type=%s",
                WORKER_ID, camera.id, camera.cam_id, resolved.source_type,
            )
        except Exception as exc:
            logger.exception(
                "Camera claim startup failed | worker_id=%s camera_pk=%s cam_id=%s",
                WORKER_ID, camera.id, camera.cam_id,
            )
            set_camera_runtime(
                camera.id,
                cam_id=camera.cam_id,
                user_id=camera.user_id,
                worker_id=WORKER_ID,
                state="ERROR",
                source_type=camera.source_type,
                preview_base_url=PREVIEW_BASE_URL,
                error_code=type(exc).__name__,
                error_message=str(exc),
            )
            try:
                release_camera_lease(lease)
            finally:
                camera_lease_events_total.labels(event="startup_failed").inc()

    def _stop_owned(self, camera_pk: int, reason: str) -> None:
        owned = self.owned.pop(int(camera_pk), None)
        if owned is None:
            return
        try:
            owned.pipeline.stop(timeout=SHUTDOWN_WAIT_SECONDS)
            legacy = self.legacy
            if legacy is not None:
                legacy.update_camera_status(
                    owned.cam_id, False, state=legacy.CameraState.OFFLINE, worker_running=False
                )
                with legacy.state_lock:
                    legacy.latestJpegs.pop(owned.cam_id, None)
                    legacy.cameraLocks.pop(owned.cam_id, None)
                    legacy.cameraFrameCount.pop(owned.cam_id, None)
        except Exception:
            logger.exception(
                "Camera stop failed | worker_id=%s camera_pk=%s cam_id=%s reason=%s",
                WORKER_ID, owned.camera_pk, owned.cam_id, reason,
            )
        finally:
            try:
                release_camera_lease(owned.lease)
                camera_lease_events_total.labels(event="released").inc()
            except Exception:
                logger.exception("Camera lease release failed | camera_pk=%s", owned.camera_pk)
            delete_camera_runtime(owned.camera_pk)
            logger.info(
                "Camera released | worker_id=%s camera_pk=%s cam_id=%s reason=%s",
                WORKER_ID, owned.camera_pk, owned.cam_id, reason,
            )

    def _reconcile(self) -> None:
        with SessionLocal() as db:
            requested = get_requested_running_cameras(db, limit=max(200, MAX_CAMERAS * 20))
            requested_by_pk = {int(row.id): row for row in requested}

            # Stop cameras whose persistent intent changed or whose legacy
            # worker exited unexpectedly.
            for camera_pk, owned in list(self.owned.items()):
                row = requested_by_pk.get(camera_pk)
                if row is None:
                    self._stop_owned(camera_pk, "DESIRED_STATE_STOPPED")
                    continue
                if not owned.pipeline.status().get("running"):
                    self._stop_owned(camera_pk, "PIPELINE_EXITED")

            if len(self.owned) >= MAX_CAMERAS:
                return

            for camera in requested:
                if len(self.owned) >= MAX_CAMERAS:
                    break
                if int(camera.id) in self.owned:
                    continue
                if INFERENCE_SETTINGS.worker_balancing_enabled:
                    try:
                        preferred=preferred_worker_id(int(camera.id),list_worker_heartbeats())
                        if preferred and preferred != WORKER_ID:
                            continue
                    except Exception:
                        logger.debug("Load-aware assignment unavailable; lease arbitration remains active",exc_info=True)
                self._claim(camera)

    def _renew(self) -> None:
        now = time.monotonic()
        if now - self._last_renew < LEASE_RENEW_SECONDS:
            return
        self._last_renew = now
        for camera_pk, owned in list(self.owned.items()):
            try:
                ok = renew_camera_lease(owned.lease)
            except Exception:
                logger.exception(
                    "Camera lease renewal error | worker_id=%s camera_pk=%s",
                    WORKER_ID, camera_pk,
                )
                # Fail-safe only after the full lease period has elapsed since
                # the last proven ownership, avoiding needless stop/start on a
                # very short Redis hiccup.
                if now - owned.last_lease_success >= CAMERA_LEASE_TTL_SECONDS:
                    camera_lease_events_total.labels(event="renew_error_expired").inc()
                    self._stop_owned(camera_pk, "LEASE_RENEWAL_UNPROVEN")
                continue

            if not ok:
                camera_lease_events_total.labels(event="lost").inc()
                self._stop_owned(camera_pk, "LEASE_LOST")
                continue

            owned.last_lease_success = now
            camera_lease_events_total.labels(event="renewed").inc()
            legacy_state = get_camera_state(owned.cam_id) or {}
            frame_count = max(0, int(legacy_state.get("frame_count") or 0))
            elapsed = max(1e-6, now - float(owned.last_runtime_sample or now))
            pipeline_status=owned.pipeline.status()
            processing_fps=float(pipeline_status.get("effective_detection_fps") or 0.0)
            owned.last_frame_count = frame_count
            owned.last_runtime_sample = now
            set_camera_runtime(
                owned.camera_pk,
                cam_id=owned.cam_id,
                user_id=owned.user_id,
                worker_id=WORKER_ID,
                state=(
                    "RECONNECTING"
                    if str(legacy_state.get("connection_state") or "").upper() == "RECONNECTING"
                    else "RUNNING"
                ),
                source_type=owned.source_type,
                preview_base_url=PREVIEW_BASE_URL,
                frame_count=frame_count,
                processing_fps=processing_fps,
                connection_state=legacy_state.get("connection_state"),
                last_frame_at=legacy_state.get("updated_at"),
            )

    def _heartbeat(self) -> None:
        now = time.monotonic()
        if isinstance(self.runtime, CentralInferenceClient) and now - self._last_readiness_check >= 10.0:
            self._last_readiness_check = now
            try:
                self.readiness_state = {"status": "READY", "inference": self.runtime.health()}
            except Exception as exc:
                self.readiness_state = {"status": "DEGRADED", "inference_error": type(exc).__name__}
        if self.readiness is not None and now - self._last_readiness_check >= 10.0:
            self._last_readiness_check = now
            self.readiness_state = self.readiness.evaluate()
        cpu_percent=psutil.cpu_percent(interval=None)
        memory_percent=psutil.virtual_memory().percent
        pipeline_states=[item.pipeline.status() for item in self.owned.values()]
        queue_depth=sum(int(state.get("queue_depth") or 0) for state in pipeline_states)
        queue_capacity=max(1,sum(int(state.get("queue_capacity") or 1) for state in pipeline_states) if pipeline_states else MAX_CAMERAS)
        heartbeat={"capacity":MAX_CAMERAS,"owned_cameras":len(self.owned),"cpu_percent":cpu_percent,"memory_percent":memory_percent,"queue_depth":queue_depth,"queue_capacity":queue_capacity}
        set_worker_heartbeat(
            WORKER_ID,capacity=MAX_CAMERAS,owned_cameras=len(self.owned),
            status=("DRAINING" if self.stop_event.is_set() else str(self.readiness_state.get("status","NOT_READY"))),
            preview_base_url=PREVIEW_BASE_URL,started_at=self.started_at,
            cpu_percent=cpu_percent,memory_percent=memory_percent,queue_depth=queue_depth,
            queue_capacity=heartbeat["queue_capacity"],load_score=worker_load_score(heartbeat),
        )
        distributed_worker_owned_cameras.labels(worker_id=WORKER_ID).set(len(self.owned))

    def run(self) -> int:
        logger.info(
            "INTEL-I distributed worker starting | worker_id=%s capacity=%s preview=%s",
            WORKER_ID, MAX_CAMERAS, PREVIEW_BASE_URL,
        )
        self._initialize_analytics_runtime()
        self._start_preview_server()
        self._heartbeat()

        try:
            while not self.stop_event.is_set():
                self._renew()
                self._reconcile()
                self._heartbeat()
                self.stop_event.wait(DISCOVERY_SECONDS)
        finally:
            self.shutdown()
        return 0

    def shutdown(self) -> None:
        if self.stop_event.is_set() is False:
            self.stop_event.set()
        logger.info("INTEL-I distributed worker draining | worker_id=%s", WORKER_ID)
        try:
            self._heartbeat()
        except Exception:
            pass
        for camera_pk in list(self.owned):
            self._stop_owned(camera_pk, "WORKER_SHUTDOWN")
        try:
            delete_worker_heartbeat(WORKER_ID)
        except Exception:
            pass

        if self._preview_server is not None:
            self._preview_server.should_exit = True
        if self._preview_thread and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=3.0)

        legacy = self.legacy
        if legacy is not None:
            try:
                legacy.ai_scheduler.stop(wait=True)
            except Exception:
                logger.exception("AI scheduler shutdown failed")
            try:
                legacy.executor.shutdown(wait=True, cancel_futures=True)
            except Exception:
                logger.exception("Camera executor shutdown failed")
            try:
                legacy.ollama_plate_verifier.shutdown()
            except Exception:
                pass
            try:
                legacy.investigation_summary_worker.shutdown()
            except Exception:
                pass

        if self._async_loop is not None:
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        if self._async_thread and self._async_thread.is_alive():
            self._async_thread.join(timeout=3.0)
        logger.info("INTEL-I distributed worker stopped | worker_id=%s", WORKER_ID)


def main() -> int:
    logging.basicConfig(
        level=getattr(logging, str(os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    worker = DistributedWorker()

    def _signal_handler(signum, _frame):
        logger.info("Worker signal received | worker_id=%s signal=%s", WORKER_ID, signum)
        worker.stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)
    return worker.run()


if __name__ == "__main__":
    raise SystemExit(main())
