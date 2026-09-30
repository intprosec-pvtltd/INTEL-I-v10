"""HTTP client for the isolated GPU face-recognition worker."""
from __future__ import annotations

import hmac
import io
import logging
import os
import threading
import time
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


class RemoteFRSClient:
    def __init__(self, *, base_url: str, token: str, timeout_seconds: float = 3.0, jpeg_quality: int = 92, failure_threshold: int = 3, recovery_seconds: float = 10.0) -> None:
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.token = str(token or "").strip()
        self.timeout_seconds = max(0.2, min(30.0, float(timeout_seconds)))
        self.jpeg_quality = max(70, min(100, int(jpeg_quality)))
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_seconds = max(1.0, float(recovery_seconds))
        self._triton = None
        if os.getenv("INTEL_I_INFERENCE_MODE", "local") == "triton":
            from inference.triton_client import TritonInferenceClient
            self._triton = TritonInferenceClient()
            self._triton.model = "intel_i_frs"
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._failures = 0
        self._open_until = 0.0

    def _validate(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError("FRS_REMOTE_URL must be a valid http(s) URL")
        if len(self.token) < 32:
            raise RuntimeError("FRS_INTERNAL_TOKEN must be at least 32 characters")

    def _headers(self) -> dict[str, str]:
        return {"X-Intel-I-FRS-Token": self.token}

    def health(self) -> dict[str, Any]:
        if self._triton is not None:
            return self._triton.health()
        self._validate()
        response = self._session.get(f"{self.base_url}/internal/health", headers=self._headers(), timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _before_request(self) -> None:
        with self._lock:
            if time.monotonic() < self._open_until:
                raise RuntimeError("FRS worker circuit is open")

    def _record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    def _record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._open_until = time.monotonic() + self.recovery_seconds

    def observe(self, *, user_id: int, camera_id: str, track_id: str | int, timestamp: float, face_image: np.ndarray, detector_confidence: float = 1.0) -> dict[str, Any]:
        if self._triton is not None:
            ok, encoded = cv2.imencode(".png", face_image)
            if not ok:
                raise ValueError("invalid face crop")
            result = self._triton.infer_batch([{
                "camera_id": str(camera_id), "jpeg": encoded.tobytes(),
                "user_id": int(user_id), "track_id": str(track_id),
                "timestamp": float(timestamp), "detector_confidence": float(detector_confidence),
            }])[0]
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "Triton FRS failed"))
            return result["result"]
        self._validate()
        self._before_request()
        if not isinstance(face_image, np.ndarray) or face_image.size == 0:
            raise ValueError("face_image is empty")
        ok, encoded = cv2.imencode(".jpg", np.ascontiguousarray(face_image), [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("unable to encode face image")
        try:
            response = self._session.post(
                f"{self.base_url}/internal/frs/observe",
                headers=self._headers(),
                data={"user_id": str(int(user_id)), "camera_id": str(camera_id), "track_id": str(track_id), "timestamp": str(float(timestamp)), "detector_confidence": str(float(detector_confidence))},
                files={"face_image": ("face.jpg", encoded.tobytes(), "image/jpeg")},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            result = response.json()
            self._record_success()
            return result
        except (requests.RequestException, ValueError):
            self._record_failure()
            raise

    def reload_watchlist(self) -> dict[str, Any]:
        if self._triton is not None:
            return self._triton.infer_batch([{"operation": "reload", "camera_id": "frs-reload"}])[0]
        self._validate()
        response = self._session.post(
            f"{self.base_url}/internal/watchlist/reload",
            headers=self._headers(), timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
