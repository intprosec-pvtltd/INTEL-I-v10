from __future__ import annotations

import os
from dataclasses import dataclass


def _b(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _i(name: str, default: int, lo: int = 1, hi: int = 100000) -> int:
    try:
        value = int(os.getenv(name, default))
    except Exception:
        value = default
    return max(lo, min(hi, value))


def _f(name: str, default: float, lo: float = 0.0, hi: float = 1000.0) -> float:
    try:
        value = float(os.getenv(name, default))
    except Exception:
        value = default
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class InferenceSettings:
    central_enabled: bool = _b("CENTRAL_INFERENCE_ENABLED", True)
    split_enabled: bool = _b("INFERENCE_PROCESS_SPLIT_ENABLED", True)
    batching_enabled: bool = _b("AI_BATCHING_ENABLED", True)
    source_fps: float = _f("CAMERA_SOURCE_FPS", 25, 1, 120)
    preview_fps: float = _f("PREVIEW_FPS", 12, 1, 60)
    idle_fps: float = _f("IDLE_DETECTION_FPS", 2, .1, 30)
    active_fps: float = _f("ACTIVE_DETECTION_FPS", 5, .1, 30)
    incident_fps: float = _f("INCIDENT_DETECTION_FPS", 8, .1, 30)
    critical_fps: float = _f("CRITICAL_DETECTION_FPS", 10, .1, 30)
    behaviour_fps: float = _f("BEHAVIOUR_FPS", 3, .1, 30)
    frs_max_fps: float = _f("FRS_MAX_FPS", 2, .1, 30)
    anpr_max_fps: float = _f("ANPR_MAX_FPS", 3, .1, 30)
    reid_interval_seconds: float = _f("REID_INTERVAL_SECONDS", 2, .1, 120)
    batch_size: int = _i("INFERENCE_BATCH_SIZE", 16, 1, 32)
    batch_wait_ms: int = _i("INFERENCE_BATCH_WAIT_MS", 20, 1, 1000)
    frame_ttl_ms: int = _i("INFERENCE_FRAME_TTL_MS", 500, 10, 60000)
    camera_queue_size: int = _i("CAMERA_FRAME_QUEUE_SIZE", 2, 1, 8)
    gpu_queue_max_size: int = _i("GPU_QUEUE_MAX_SIZE", 100, 1, 10000)
    priority_aging_seconds: float = _f("INFERENCE_PRIORITY_AGING_SECONDS", 2.0, .1, 60)
    hw_decode_enabled: bool = _b("VIDEO_HW_DECODE_ENABLED", True)
    decode_device: str = os.getenv("VIDEO_DECODE_DEVICE", "auto").strip().lower()
    tensorrt_enabled: bool = _b("TENSORRT_ENABLED", True)
    tensorrt_precision: str = os.getenv("TENSORRT_PRECISION", "fp16").strip().lower()
    adaptive_fps: bool = _b("ADAPTIVE_FPS_ENABLED", True)
    priority_enabled: bool = _b("PRIORITY_INFERENCE_ENABLED", True)
    backpressure_enabled: bool = _b("INFERENCE_BACKPRESSURE_ENABLED", True)
    gpu_required: bool = _b("INFERENCE_GPU_REQUIRED", False)
    gpu_preprocessing_enabled: bool = _b("GPU_PREPROCESSING_ENABLED", True)
    service_url: str = os.getenv("INFERENCE_SERVICE_URL", "http://127.0.0.1:9300").rstrip("/")
    request_timeout: float = _f("INFERENCE_REQUEST_TIMEOUT_SECONDS", 10, .2, 120)
    jpeg_quality: int = _i("INFERENCE_JPEG_QUALITY", 90, 50, 100)
    max_frame_bytes: int = _i("INFERENCE_MAX_FRAME_BYTES", 8 * 1024 * 1024, 64 * 1024, 64 * 1024 * 1024)
    internal_token: str = os.getenv("INFERENCE_INTERNAL_TOKEN", "").strip()
    primary_detector_mode: str = os.getenv("CENTRAL_PRIMARY_DETECTOR_MODE", "split").strip().lower()
    primary_tensorrt_engine: str = os.getenv("PRIMARY_DETECTOR_TENSORRT_ENGINE", "").strip()
    worker_balancing_enabled: bool = _b("CAMERA_LOAD_BALANCING_ENABLED", True)


SETTINGS = InferenceSettings()
