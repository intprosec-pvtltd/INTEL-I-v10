"""Production camera health state machine and telemetry.

Phase 4 goals:
- independent per-camera health
- bounded/stale heartbeat detection
- FPS, latency, drops and reconnect telemetry
- rate-limited persistence
- no secrets in telemetry
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import time
from typing import Any

from core.camera_state import CameraState


def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class CameraHealthRuntime:
    camera_id: str
    state: CameraState = CameraState.OFFLINE
    last_frame_monotonic: float | None = None
    last_frame_at: datetime | None = None
    last_state_change: datetime = field(default_factory=utc_now_naive)
    decoded_frames: int = 0
    processed_frames: int = 0
    dropped_frames: int = 0
    reconnects: int = 0
    read_failures: int = 0
    decoder_failures: int = 0
    bytes_out: int = 0
    last_read_latency_ms: float | None = None
    fps_ewma: float = 0.0
    latency_ewma_ms: float = 0.0
    queue_depth: int = 0
    queue_capacity: int = 1
    last_persist_monotonic: float = 0.0
    last_error_code: str | None = None
    last_error_message: str | None = None

    def mark_frame(self, *, read_latency_ms: float, now_monotonic: float | None = None) -> None:
        now = time.monotonic() if now_monotonic is None else now_monotonic
        self.decoded_frames += 1
        self.processed_frames += 1
        self.last_frame_monotonic = now
        self.last_frame_at = utc_now_naive()
        self.last_read_latency_ms = max(0.0, float(read_latency_ms))
        alpha = 0.15
        if self.fps_ewma <= 0.0:
            self.fps_ewma = 1.0 / max(read_latency_ms / 1000.0, 0.001)
        else:
            instant_fps = 1.0 / max(read_latency_ms / 1000.0, 0.001)
            self.fps_ewma = alpha * instant_fps + (1 - alpha) * self.fps_ewma
        if self.latency_ewma_ms <= 0.0:
            self.latency_ewma_ms = self.last_read_latency_ms
        else:
            self.latency_ewma_ms = alpha * self.last_read_latency_ms + (1 - alpha) * self.latency_ewma_ms

    def snapshot(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "state": self.state.value,
            "decoded_frames": self.decoded_frames,
            "processed_frames": self.processed_frames,
            "dropped_frames": self.dropped_frames,
            "reconnects": self.reconnects,
            "read_failures": self.read_failures,
            "decoder_failures": self.decoder_failures,
            "fps": round(self.fps_ewma, 3),
            "latency_ms": round(self.latency_ewma_ms, 3),
            "queue_depth": self.queue_depth,
            "queue_capacity": self.queue_capacity,
            "last_frame_at": self.last_frame_at.isoformat() if self.last_frame_at else None,
            "last_error_code": self.last_error_code,
        }


class CameraHealthRegistry:
    def __init__(self, *, stale_after_seconds: float = 8.0, persist_interval_seconds: float = 5.0):
        self.stale_after_seconds = max(2.0, float(stale_after_seconds))
        self.persist_interval_seconds = max(1.0, float(persist_interval_seconds))
        self._items: dict[str, CameraHealthRuntime] = {}
        self._lock = threading.RLock()

    def ensure(self, camera_id: str, *, queue_capacity: int = 1) -> CameraHealthRuntime:
        camera_id = str(camera_id)
        with self._lock:
            item = self._items.get(camera_id)
            if item is None:
                item = CameraHealthRuntime(camera_id=camera_id, queue_capacity=max(1, int(queue_capacity)))
                self._items[camera_id] = item
            else:
                item.queue_capacity = max(1, int(queue_capacity))
            return item

    def set_state(self, camera_id: str, state: CameraState, *, error_code: str | None = None, error_message: str | None = None) -> CameraHealthRuntime:
        with self._lock:
            item = self.ensure(camera_id)
            if item.state != state:
                item.last_state_change = utc_now_naive()
            item.state = state
            item.last_error_code = error_code
            item.last_error_message = error_message[:240] if error_message else None
            return item

    def frame(self, camera_id: str, *, read_latency_ms: float, queue_depth: int = 0) -> CameraHealthRuntime:
        with self._lock:
            item = self.ensure(camera_id)
            item.queue_depth = max(0, int(queue_depth))
            item.mark_frame(read_latency_ms=read_latency_ms)
            return item

    def drop(self, camera_id: str, count: int = 1) -> None:
        with self._lock:
            self.ensure(camera_id).dropped_frames += max(0, int(count))

    def reconnect(self, camera_id: str) -> None:
        with self._lock:
            self.ensure(camera_id).reconnects += 1

    def read_failure(self, camera_id: str, *, error_code: str = "READ_FAILED", message: str | None = None) -> None:
        with self._lock:
            item = self.ensure(camera_id)
            item.read_failures += 1
            item.last_error_code = error_code
            item.last_error_message = message[:240] if message else None

    def decoder_failure(self, camera_id: str, *, message: str | None = None) -> None:
        with self._lock:
            item = self.ensure(camera_id)
            item.decoder_failures += 1
            item.last_error_code = "DECODER_FAILED"
            item.last_error_message = message[:240] if message else None

    def stale_cameras(self, active_camera_ids: set[str]) -> list[str]:
        now = time.monotonic()
        stale: list[str] = []
        with self._lock:
            for camera_id in active_camera_ids:
                item = self._items.get(camera_id)
                if not item or item.last_frame_monotonic is None:
                    continue
                if now - item.last_frame_monotonic > self.stale_after_seconds:
                    stale.append(camera_id)
        return stale

    def get(self, camera_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(str(camera_id))
            return item.snapshot() if item else None

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [item.snapshot() for item in self._items.values()]

    def remove(self, camera_id: str) -> None:
        with self._lock:
            self._items.pop(str(camera_id), None)


camera_health_registry = CameraHealthRegistry(
    stale_after_seconds=float(__import__('os').getenv('CAMERA_HEALTH_STALE_SECONDS', '8')),
    persist_interval_seconds=float(__import__('os').getenv('CAMERA_HEALTH_PERSIST_SECONDS', '5')),
)
