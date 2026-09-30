"""INTEL-I centralized GPU inference service.

Owns the heavy GPU runtime in centralized mode. Stream workers only decode,
preview, schedule and submit the newest eligible frame. Primary detection is
batched across cameras; ByteTrack and conditional secondary AI retain the
existing per-camera state in the legacy analytics engine.
"""
from __future__ import annotations

import base64
import asyncio
import hmac
import logging
import math
import os
import re
import signal
import subprocess
import threading
import time
from typing import Any

os.environ["INTEL_I_PROCESS_ROLE"] = "inference-worker"

import cv2
import numpy as np
import psutil
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest, Counter, Gauge, Histogram
from fastapi.responses import Response
import uvicorn

from inference.batcher import DynamicBatcher, InferenceQueueFull, StaleFrameDropped
from inference.frame_scheduler import FrameScheduler
from inference.gpu_preprocess import status as gpu_preprocess_status
from inference.hardware_decode import detect_capability
from inference.model_metrics import MODEL_METRICS
from inference.model_registry import REGISTRY
from inference.policy import CAMERA_POLICIES
from inference.settings import SETTINGS
from inference.triton_client import TritonInferenceClient, triton_enabled, InferenceUnavailable
from inference.track_cache import TRACK_CACHE
from runtime.analytics_runtime import AnalyticsRuntime

log = logging.getLogger("intel_i.inference_worker")
app = FastAPI(title="INTEL-I Central Inference", docs_url=None, redoc_url=None, openapi_url=None)
runtime = AnalyticsRuntime()
triton_client: TritonInferenceClient | None = None
scheduler = FrameScheduler()
started_at = time.time()
batcher: DynamicBatcher | None = None
_event_loop: asyncio.AbstractEventLoop | None = None
_event_thread: threading.Thread | None = None
_context_lock = threading.RLock()
_context_loaded_at: dict[tuple[str, int], float] = {}
CAMERA_CONTEXT_REFRESH_SECONDS = max(5.0, float(os.getenv("CAMERA_CONTEXT_REFRESH_SECONDS", "30")))
CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,100}$")
REQUIRE_TOKEN = os.getenv(
    "INFERENCE_REQUIRE_INTERNAL_TOKEN",
    "true" if os.getenv("ENV", "").strip().lower() in {"prod", "production"} else "false",
).strip().lower() in {"1", "true", "yes", "on"}

gateway_rpc_latency = Histogram("intel_i_triton_gateway_seconds", "Triton batch RPC wall time", buckets=(.01,.025,.05,.1,.25,.5,1,2,5,10))
gateway_rpc_errors = Counter("intel_i_triton_gateway_errors_total", "Failed Triton RPC batches")
gateway_queue = Gauge("intel_i_inference_queue_depth", "Pending sampled frames")
gateway_expired = Gauge("intel_i_inference_expired_frames", "Frames expired since process start")

metrics_lock = threading.RLock()
metrics: dict[str, float] = {
    "requests": 0, "errors": 0, "frames": 0, "latency_ms_sum": 0.0,
    "last_latency_ms": 0.0, "decoded_frames": 0, "decode_errors": 0,
}


def _auth(token: str | None) -> None:
    expected = SETTINGS.internal_token
    if not REQUIRE_TOKEN and not expected:
        return
    if not expected:
        raise HTTPException(status_code=503, detail="inference internal authentication is not configured")
    if not token or not hmac.compare_digest(str(token), str(expected)):
        raise HTTPException(status_code=401, detail="unauthorized inference client")


def gpu_info() -> dict[str, Any]:
    out: dict[str, Any] = {
        "cuda_available": False, "name": None, "memory_used_mb": 0.0,
        "memory_reserved_mb": 0.0, "memory_total_mb": 0.0,
        "utilization_percent": None, "decoder_utilization_percent": None,
        "temperature_c": None,
    }
    try:
        import torch
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            out["name"] = torch.cuda.get_device_name(0)
            out["memory_used_mb"] = round(torch.cuda.memory_allocated(0) / 1048576, 1)
            out["memory_reserved_mb"] = round(torch.cuda.memory_reserved(0) / 1048576, 1)
            out["memory_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 1048576, 1)
    except Exception:
        log.debug("Torch GPU telemetry unavailable", exc_info=True)
    try:
        query = "utilization.gpu,utilization.decoder,temperature.gpu,memory.used,memory.total,name"
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            values = [value.strip() for value in proc.stdout.strip().splitlines()[0].split(",")]
            if len(values) >= 6:
                def _optional_float(value: str):
                    value = str(value or "").strip()
                    if not value or value.upper() in {"N/A", "NA", "[N/A]", "NOT SUPPORTED"}:
                        return None
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
                parsed = {
                    "utilization_percent": _optional_float(values[0]),
                    "decoder_utilization_percent": _optional_float(values[1]),
                    "temperature_c": _optional_float(values[2]),
                    "memory_used_mb": _optional_float(values[3]),
                    "memory_total_mb": _optional_float(values[4]),
                    "name": values[5] or out.get("name"),
                }
                # Keep Torch memory/name data whenever nvidia-smi exposes only
                # a partial metric set (common with some vGPU profiles).
                for key, value in parsed.items():
                    if value is not None:
                        out[key] = value
    except Exception:
        log.debug("nvidia-smi telemetry unavailable", exc_info=True)
    return out


def _validate_camera_id(value: str) -> str:
    value = str(value or "").strip()
    if not CAMERA_ID_RE.fullmatch(value):
        raise HTTPException(status_code=422, detail="invalid camera_id")
    return value


def _ensure_camera_context(item: dict[str, Any]) -> None:
    camera_id = str(item.get("camera_id") or "")
    user_id = int(item.get("user_id") or 0)
    key = (camera_id, user_id)
    now = time.monotonic()
    with _context_lock:
        if now - _context_loaded_at.get(key, 0.0) < CAMERA_CONTEXT_REFRESH_SECONDS:
            return
    try:
        from db.database import SessionLocal
        from db.crud import get_camera
        from db.model import CameraAIProfile
        with SessionLocal() as db:
            camera = get_camera(db, camera_id, user_id) if user_id else None
            if camera is None:
                # Uploaded-video/testing sources can exist without a Camera row.
                CAMERA_POLICIES.configure(camera_id, {}, zone=None)
            else:
                legacy = runtime.legacy
                legacy._cache_camera_correlation_meta(camera)
                if camera.zones:
                    legacy.zone_manager.set_zones(camera.cam_id, camera.zones)
                legacy.camera_config.set_mode(camera.cam_id, camera.zone)
                profile = None
                if getattr(camera, "ai_profile_id", None):
                    profile = db.query(CameraAIProfile).filter(
                        CameraAIProfile.id == int(camera.ai_profile_id),
                        CameraAIProfile.user_id == int(camera.user_id),
                    ).first()
                CAMERA_POLICIES.configure(
                    camera_id, getattr(profile, "configuration", None),
                    processing_fps=getattr(profile, "processing_fps", None),
                    evidence_enabled=getattr(profile, "evidence_enabled", None),
                    zone=getattr(camera, "zone", None), profile_name=getattr(profile, "name", None),
                )
        with _context_lock:
            _context_loaded_at[key] = now
    except Exception:
        log.exception("Failed to initialize camera context | camera_id=%s user_id=%s", camera_id, user_id)
        raise RuntimeError("camera policy context unavailable")


def _decode_item(item: dict[str, Any]) -> np.ndarray:
    data = item.get("jpeg") or b""
    if not data or len(data) > SETTINGS.max_frame_bytes:
        raise ValueError("invalid encoded frame size")
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("invalid jpeg frame")
    return frame


def handle_batch(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One primary detector invocation for the valid multi-camera batch."""
    if triton_client is not None:
        with gateway_rpc_latency.time():
            try:
                return triton_client.infer_batch(items)
            except Exception:
                gateway_rpc_errors.inc()
                raise
    started = time.perf_counter()
    results: list[dict[str, Any] | None] = [None] * len(items)
    valid_indices: list[int] = []
    frames: list[np.ndarray] = []
    for index, item in enumerate(items):
        try:
            _ensure_camera_context(item)
            frames.append(_decode_item(item)); valid_indices.append(index)
            with metrics_lock: metrics["decoded_frames"] += 1
        except Exception as exc:
            with metrics_lock:
                metrics["decode_errors"] += 1; metrics["errors"] += 1
            results[index] = {"ok": False, "camera_id": item.get("camera_id"), "error": f"{type(exc).__name__}: {exc}"}

    detections: list[dict[str, Any]] = []
    if frames:
        try:
            detections = runtime.detect_batch(frames)
            if len(detections) != len(frames):
                raise RuntimeError("primary detector batch result count mismatch")
        except Exception as exc:
            with metrics_lock: metrics["errors"] += len(frames)
            log.exception("Batched primary inference failed | batch_size=%s", len(frames))
            for index in valid_indices:
                results[index] = {"ok": False, "camera_id": items[index].get("camera_id"), "error": f"{type(exc).__name__}: {exc}"}
            detections = []

    for batch_index, item_index in enumerate(valid_indices):
        if not detections:
            break
        item = items[item_index]; frame = frames[batch_index]; detection = detections[batch_index]
        frame_started = time.perf_counter()
        try:
            person_detection = detection.get("person")
            vehicle_detection = detection.get("vehicle")
            annotated = runtime.process_frame(
                item["camera_id"], frame, float(item["timestamp"]),
                frame_count=int(item.get("frame_count") or 0),
                source_type=str(item.get("source_type") or "rtsp"),
                frame_timestamp=item.get("frame_timestamp"),
                source_pts_seconds=item.get("source_pts_seconds"),
                timestamp_source=str(item.get("timestamp_source") or "UNKNOWN"),
                timestamp_quality=str(item.get("timestamp_quality") or "UNKNOWN"),
                precomputed_person_detections=person_detection,
                precomputed_vehicle_detections=vehicle_detection,
            )
            encoded_ok, encoded_preview = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), SETTINGS.jpeg_quality])
            if not encoded_ok:
                raise RuntimeError("annotated preview encoding failed")
            hint = runtime.schedule_hint(
                str(item["camera_id"]),
                person_count=len(person_detection) if person_detection is not None else 0,
                vehicle_count=len(vehicle_detection) if vehicle_detection is not None else 0,
            )
            load_factor = 0.0 if batcher is None else min(1.0, batcher.depth / max(1, batcher.maxsize))
            scheduler.update_hint(
                str(item["camera_id"]), mode=hint.get("mode"), priority=hint.get("priority"),
                reason=hint.get("reason"), load_factor=load_factor,
            )
            schedule = scheduler.status(str(item["camera_id"]))
            schedule["reason"] = hint.get("reason")
            frame_ms = (time.perf_counter() - frame_started) * 1000.0
            results[item_index] = {
                "ok": True, "camera_id": item["camera_id"], "schedule": schedule,
                "person_count": len(person_detection) if person_detection is not None else 0,
                "vehicle_count": len(vehicle_detection) if vehicle_detection is not None else 0,
                "processing_latency_ms": round(frame_ms, 3),
                "annotated_jpeg": base64.b64encode(encoded_preview.tobytes()).decode("ascii"),
            }
            with metrics_lock:
                metrics["frames"] += 1; metrics["latency_ms_sum"] += frame_ms; metrics["last_latency_ms"] = frame_ms
        except Exception as exc:
            with metrics_lock: metrics["errors"] += 1
            log.exception("Central downstream analytics failed | camera_id=%s", item.get("camera_id"))
            results[item_index] = {"ok": False, "camera_id": item.get("camera_id"), "error": f"{type(exc).__name__}: {exc}"}

    total_ms = (time.perf_counter() - started) * 1000.0
    log.debug("Inference batch complete | batch=%s valid=%s latency_ms=%.2f queue=%s", len(items), len(frames), total_ms, batcher.depth if batcher else 0)
    return [result or {"ok": False, "error": "unresolved batch item"} for result in results]


@app.on_event("startup")
def startup() -> None:
    global batcher, _event_loop, _event_thread, triton_client
    if REQUIRE_TOKEN and not SETTINGS.internal_token:
        raise RuntimeError("INFERENCE_REQUIRE_INTERNAL_TOKEN=true but INFERENCE_INTERNAL_TOKEN is empty")
    if triton_enabled():
        triton_client = TritonInferenceClient()
        triton_client.health()
        batcher = DynamicBatcher(handle_batch, maxsize=SETTINGS.gpu_queue_max_size)
        batcher.start()
        return
    runtime.load()
    runtime.register_models(REGISTRY)

    from services.realtimeBus import publish_realtime_event
    _event_loop = asyncio.new_event_loop()
    def _run_loop() -> None:
        assert _event_loop is not None
        asyncio.set_event_loop(_event_loop); _event_loop.run_forever()
    _event_thread = threading.Thread(target=_run_loop, name="inference-events", daemon=True); _event_thread.start()
    legacy = runtime.legacy; legacy.main_loop = _event_loop
    async def _publish_alert(payload: dict):
        publish_realtime_event({"kind": "alert", "payload": payload, "worker_id": "central-inference"})
    async def _publish_position(payload: dict):
        publish_realtime_event({"kind": "position", "payload": payload, "worker_id": "central-inference"})
    legacy.sendAlert = _publish_alert; legacy.broadcast_vehicle_position = _publish_position

    gpu = gpu_info()
    if SETTINGS.gpu_required and not gpu["cuda_available"]:
        raise RuntimeError("INFERENCE_GPU_REQUIRED=true but CUDA is unavailable")
    if os.getenv("INTEL_I_TRITON_EMBEDDED") != "1":
        batcher = DynamicBatcher(handle_batch, maxsize=SETTINGS.gpu_queue_max_size); batcher.start()
    log.info("Central inference READY | batch_size=%s wait_ms=%s queue=%s gpu=%s", SETTINGS.batch_size, SETTINGS.batch_wait_ms, SETTINGS.gpu_queue_max_size, gpu.get("name"))


@app.on_event("shutdown")
def shutdown() -> None:
    if batcher: batcher.stop()
    if triton_client: triton_client.close()
    if _event_loop: _event_loop.call_soon_threadsafe(_event_loop.stop)
    if _event_thread and _event_thread.is_alive(): _event_thread.join(timeout=3)


@app.get("/health/live")
def live() -> dict[str, Any]:
    return {"status": "UP", "uptime_seconds": round(time.time() - started_at, 1)}


@app.get("/health/ready")
def ready() -> dict[str, Any]:
    if triton_client is not None:
        try:
            health = triton_client.health()
        except InferenceUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        return {**health, **batcher.status(), "runtime": triton_client.status(), "gpu": triton_client.gpu_telemetry(), "requests": triton_client.requests, "errors": triton_client.errors, "average_inference_latency_ms": batcher.last_batch_latency_ms}
    if not runtime.loaded or batcher is None:
        raise HTTPException(503, "inference runtime is not ready")
    gpu = gpu_info(); batch = batcher.status() if batcher else {"queue_depth": 0, "queue_capacity": SETTINGS.gpu_queue_max_size}
    scheduler.set_load(int(batch.get("queue_depth") or 0), int(batch.get("queue_capacity") or 1))
    with metrics_lock: local = dict(metrics)
    average = local["latency_ms_sum"] / max(1.0, local["frames"])
    return {
        "status": "READY" if runtime.loaded and batcher is not None else "NOT_READY",
        "runtime": runtime.status(), "gpu": gpu, **batch,
        "cpu_percent": psutil.cpu_percent(interval=None), "ram_percent": psutil.virtual_memory().percent,
        "average_inference_latency_ms": round(average, 3), "last_inference_latency_ms": round(local["last_latency_ms"], 3),
        "requests": int(local["requests"]), "errors": int(local["errors"]), "decoded_frames": int(local["decoded_frames"]),
        "active_cameras": len(CAMERA_POLICIES.snapshot()), "scheduler": scheduler.status(),
        "models": REGISTRY.status(), "model_metrics": MODEL_METRICS.snapshot(), "track_cache": TRACK_CACHE.snapshot(),
        "decoder": detect_capability(), "gpu_preprocessing": gpu_preprocess_status(),
    }


@app.get("/metrics")
def prometheus_metrics(x_intel_i_inference_token: str | None = Header(default=None)) -> Response:
    _auth(x_intel_i_inference_token)
    if batcher is not None:
        gateway_queue.set(batcher.depth)
        gateway_expired.set(batcher.expired)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/infer")
async def infer(
    frame: UploadFile = File(...), camera_id: str = Form(...), user_id: int = Form(0),
    timestamp: float = Form(...), source_type: str = Form("rtsp"), frame_count: int = Form(0),
    priority: int = Form(5), source_pts_seconds: str = Form(""), timestamp_source: str = Form("UNKNOWN"),
    timestamp_quality: str = Form("UNKNOWN"), x_intel_i_inference_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _auth(x_intel_i_inference_token); camera_id = _validate_camera_id(camera_id)
    if not math.isfinite(float(timestamp)) or float(timestamp) < 0: raise HTTPException(422, "invalid timestamp")
    if int(user_id) < 0: raise HTTPException(422, "invalid user_id")
    if str(source_type).strip().lower() not in {"rtsp", "http", "https", "hls", "upload", "uploaded", "file", "sentinel", "mjpeg", "onvif", "usb", "local"}:
        raise HTTPException(422, "unsupported source_type")
    if not batcher: raise HTTPException(503, "inference worker not ready")
    data = await frame.read(SETTINGS.max_frame_bytes + 1)
    if not data: raise HTTPException(400, "empty frame")
    if len(data) > SETTINGS.max_frame_bytes: raise HTTPException(413, "frame too large")
    with metrics_lock: metrics["requests"] += 1
    scheduler.set_load(batcher.depth, batcher.maxsize)
    resolved_priority = max(0, min(5, int(priority))) if SETTINGS.priority_enabled else 5
    payload = {
        "jpeg": data, "camera_id": camera_id, "user_id": int(user_id), "timestamp": float(timestamp),
        "source_type": str(source_type).strip().lower(), "frame_count": max(0, int(frame_count)),
        "source_pts_seconds": float(source_pts_seconds) if str(source_pts_seconds).strip() else None,
        "timestamp_source": str(timestamp_source)[:64], "timestamp_quality": str(timestamp_quality)[:64],
    }
    future = batcher.submit(payload, priority=resolved_priority)
    try:
        result = await asyncio.wrap_future(future)
    except (InferenceQueueFull, StaleFrameDropped) as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not result.get("ok"): raise HTTPException(status_code=500, detail=result.get("error", "inference failed"))
    return result


@app.post("/v1/cameras/{camera_id}/reset")
def reset(camera_id: str, x_intel_i_inference_token: str | None = Header(default=None)) -> dict[str, bool]:
    _auth(x_intel_i_inference_token); camera_id = _validate_camera_id(camera_id)
    scheduler.clear(camera_id)
    if triton_client is not None:
        triton_client.infer_batch([{"operation": "reset", "camera_id": camera_id}])
    else:
        runtime.reset_camera(camera_id)
    with _context_lock:
        for key in [key for key in _context_loaded_at if key[0] == camera_id]: _context_loaded_at.pop(key, None)
    return {"ok": True}


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(app, host=os.getenv("INFERENCE_HOST", "0.0.0.0"), port=int(os.getenv("INFERENCE_PORT", "9300")), log_level=os.getenv("INFERENCE_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
