from __future__ import annotations

import base64
import numpy as np
import threading
import time
from typing import Any

import cv2
import httpx

from .frame_scheduler import FrameScheduler
from .settings import SETTINGS


class InferenceBackpressureError(RuntimeError):
    """Expected load-shedding signal; callers should drop this frame quietly."""


class CentralInferenceClient:
    """Synchronous stream-worker client for the internal central AI service.

    The client is shared by all camera pipelines in a stream-worker process.
    Preview never waits on this client; only the analytics thread does.
    """

    loaded = True

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or SETTINGS.service_url).rstrip("/")
        headers = {"User-Agent": "intel-i-camera-worker/central-inference"}
        if SETTINGS.internal_token:
            headers["X-Intel-I-Inference-Token"] = SETTINGS.internal_token
        self.client = httpx.Client(
            timeout=httpx.Timeout(SETTINGS.request_timeout, connect=min(3.0, SETTINGS.request_timeout)),
            headers=headers,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )
        self.scheduler = FrameScheduler()
        self._last_health: dict[str, Any] = {}
        self._hints: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._failures = 0
        self._circuit_open_until = 0.0

    @staticmethod
    def _camera_id(camera: Any) -> str:
        return str(getattr(camera, "cam_id", None) or getattr(camera, "id", None) or camera)

    def should_submit(self, camera: Any) -> bool:
        camera_id = self._camera_id(camera)
        return self.scheduler.should_submit(camera_id, self.priority_for(camera_id))

    def priority_for(self, camera: Any) -> int:
        camera_id = self._camera_id(camera)
        return self.scheduler.priority(camera_id)

    def target_fps(self, camera: Any) -> float:
        camera_id = self._camera_id(camera)
        return self.scheduler.target_fps(camera_id, self.priority_for(camera_id))

    def schedule_hint(self, camera: Any) -> dict[str, Any]:
        camera_id = self._camera_id(camera)
        with self._lock:
            return dict(self._hints.get(camera_id, {}))

    def _update_hint(self, camera_id: str, payload: dict[str, Any]) -> None:
        hint = payload.get("schedule") if isinstance(payload, dict) else None
        if not isinstance(hint, dict):
            return
        self.scheduler.update_hint(
            camera_id,
            mode=hint.get("mode"),
            priority=hint.get("priority"),
            reason=hint.get("reason"),
            load_factor=hint.get("load_factor"),
        )
        with self._lock:
            self._hints[camera_id] = dict(hint)

    def process_frame(self, camera, frame, timestamp, **kwargs):
        now = time.monotonic()
        if now < self._circuit_open_until:
            raise RuntimeError("central inference circuit is temporarily open")
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), SETTINGS.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("failed to encode frame for centralized inference")
        payload_bytes = encoded.tobytes()
        if len(payload_bytes) > SETTINGS.max_frame_bytes:
            raise RuntimeError("encoded inference frame exceeds configured size limit")
        camera_id = self._camera_id(camera)
        priority = int(kwargs.get("priority", self.priority_for(camera_id)))
        data = {
            "camera_id": camera_id,
            "user_id": str(int(getattr(camera, "user_id", 0) or 0)),
            "timestamp": str(float(timestamp)),
            "source_type": str(kwargs.get("source_type") or getattr(camera, "source_type", "rtsp")),
            "frame_count": str(int(kwargs.get("frame_count", 0))),
            "priority": str(priority),
            "source_pts_seconds": "" if kwargs.get("source_pts_seconds") is None else str(float(kwargs["source_pts_seconds"])),
            "timestamp_source": str(kwargs.get("timestamp_source") or "UNKNOWN"),
            "timestamp_quality": str(kwargs.get("timestamp_quality") or "UNKNOWN"),
        }
        try:
            response = self.client.post(
                self.base_url + "/v1/infer",
                data=data,
                files={"frame": ("frame.jpg", payload_bytes, "image/jpeg")},
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("ok") or str(payload.get("camera_id")) != camera_id:
                raise RuntimeError("Inference result failed or belongs to a different camera")
            self._failures = 0
            self._circuit_open_until = 0.0
            self._update_hint(camera_id, payload)
        except httpx.HTTPStatusError as exc:
            self._failures += 1
            if exc.response.status_code == 429:
                # Server-side load shedding is a real pressure signal. Push the
                # local adaptive scheduler to its maximum load factor so idle
                # and ordinary cameras immediately reduce submission cadence.
                self.scheduler.set_load(SETTINGS.gpu_queue_max_size, SETTINGS.gpu_queue_max_size)
                raise InferenceBackpressureError("central inference queue is saturated") from exc
            if self._failures >= 3:
                self._circuit_open_until = time.monotonic() + min(10.0, 2.0 ** min(self._failures - 2, 3))
            raise
        except Exception:
            self._failures += 1
            if self._failures >= 3:
                self._circuit_open_until = time.monotonic() + min(10.0, 2.0 ** min(self._failures - 2, 3))
            raise
        # Return the exact annotated image. Reusing normalized coordinates on
        # the current full-resolution capture frame would misplace the boxes.
        encoded = payload.get("annotated_jpeg")
        if not encoded:
            raise RuntimeError("Inference service did not return an annotated preview; update both workers")
        annotated = cv2.imdecode(np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8), cv2.IMREAD_COLOR)
        if annotated is None:
            raise RuntimeError("Invalid annotated inference preview")
        return annotated

    def health(self):
        response = self.client.get(self.base_url + "/health/ready", timeout=2.0)
        response.raise_for_status()
        self._last_health = response.json()
        if self._last_health.get("status") != "READY":
            raise RuntimeError("Central inference runtime is not ready")
        queue_depth = int(self._last_health.get("queue_depth") or 0)
        queue_capacity = int(self._last_health.get("queue_capacity") or SETTINGS.gpu_queue_max_size)
        self.scheduler.set_load(queue_depth, queue_capacity)
        return self._last_health

    def reset_camera(self, camera_id: str):
        self.scheduler.clear(str(camera_id))
        try:
            self.client.post(self.base_url + f"/v1/cameras/{camera_id}/reset", timeout=2.0)
        except Exception:
            pass
        with self._lock:
            self._hints.pop(str(camera_id), None)

    def close(self):
        self.client.close()
